import argparse
import copy
import os
import os.path as osp
import time
import math
import logging

import mmcv
import torch
import numpy as np
from mmcv.runner import init_dist, IterBasedRunner, build_optimizer, IterLoader, get_host_info
from mmcv.utils import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger
from mmseg.core import DistEvalHook, EvalHook, Evaluator

from mmseg.models.losses import AutoSegLoss, map_to_one_hot

from truncated_normal import TruncatedNormal



class AutoLoss:
    def __init__(self, cfg, logger, meta):
        self.cfg_backup = cfg
        self._reset_cfg()
        
        self.logger = logger


        # build train_dataset & dataloader
        train_dataset = build_dataset(cfg.data.train)
        self.train_dataset = train_dataset
        self.train_dataloader = IterLoader(build_dataloader(
                train_dataset,
                cfg.data.samples_per_gpu,
                cfg.data.workers_per_gpu,
                # cfg.gpus will be ignored if distributed
                len(cfg.gpu_ids),
                dist=False,
                seed=cfg.seed,
                drop_last=True,
                persistent_workers=True
        ))
        
        # build val_dataset & dataloader (for fast evaluate)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        self.val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False,
            persistent_workers=True)
        
        self.num_class = len(train_dataset.CLASSES)
        
        self.num_models_per_gpu = self.cfg.num_models_per_gpu
        
        self.evaluators = [Evaluator(self.num_class, drop_bg=True) for _ in range(self.num_models_per_gpu)]
        
        
        
        self.num_samples = int(math.ceil(float(cfg.num_samples)/float(torch.distributed.get_world_size())) * torch.distributed.get_world_size())
            
        # build Gaussian Distribution
        self.num_pieces = self.cfg.num_pieces
        self.tol = self.cfg.get('tol', 5)
        self.drop_bg = self.cfg.get('drop_bg', False)
        
        
        if 'target_metric' not in self.cfg:
            self.cfg.target_metric = 'mIoU'
            self.logger.info(f'Missing target metric. Using {self.cfg.target_metric} as default.')
        self.target_metric = self.cfg.target_metric
        
        
        if self.target_metric in ['mIoU', 'FWIoU', 'BIoU']:
            # each piece has 2 control points, except for the last piece which has 1 control point.
            self.num_theta = (self.num_pieces * 2 - 1) * 2 * 2
        elif self.target_metric in ['mAcc', 'gAcc']:
            self.num_theta = (self.num_pieces * 2 - 1) * 2
        elif self.target_metric == 'BF1':
            self.num_theta = (self.num_pieces * 2 - 1) * 2 * 4
        else:
            raise NotImplementedError
                
        
        self.asl = AutoSegLoss(
            num_class=self.num_class,
            theta=torch.tensor(0),
            parameterization='bezier', 
            target_metric=self.target_metric,
            drop_bg=self.drop_bg,
            tol=self.tol
        )
        
        self.logger.info(f'Searching for metric {self.target_metric}.')
        
        if torch.distributed.get_rank() == 0:
            # initialize the parameters
            if isinstance(cfg.mu, list):
                assert len(cfg.mu) == self.num_theta
                assert len(cfg.sigma) == self.num_theta
                self.mu_x = (-torch.log(
                    1.0 / torch.tensor(cfg.mu, dtype=torch.float, device=torch.cuda.current_device()) - 1
                )).requires_grad_()
                self.sigma_x = torch.tensor(cfg.sigma, dtype=torch.float, device=torch.cuda.current_device()).log()
            else:
                self.mu_x = (-np.log(1.0/cfg.mu - 1) * torch.ones(self.num_theta, device=torch.cuda.current_device())).requires_grad_()
                self.sigma_x = ((np.log(cfg.sigma)) * torch.ones(self.num_theta, device=torch.cuda.current_device()))
                
            # We use sigmoid and exp to restrict the values of mu (0~1) and sigma (0~Inf)
            # The optimization variables are called mu_x and sigma_x, with range (-Inf ~ Inf)
            self.mu = self.mu_x.sigmoid()
            self.sigma = self.sigma_x.exp()
            self.PolicyOptimizer = torch.optim.Adam([self.mu_x], lr=self.cfg.mu_lr)
            
            
            # define LR scheduler
            if not self.cfg.get('lr_lambda', None):
                self.LR_Scheduler = torch.optim.lr_scheduler.LambdaLR(self.PolicyOptimizer, lr_lambda=(lambda epoch : 1.0))
                print('Using Constant LR Scheduler!')
            elif self.cfg.lr_lambda == 'linear':
                self.LR_Scheduler = torch.optim.lr_scheduler.LambdaLR(self.PolicyOptimizer, lr_lambda=(lambda epoch : (1 - float(epoch) / float(self.cfg.sample_times + 1))))
                print('Using Linear LR Scheduler!')
            else:
                raise NotImplementedError
            
            
    def _reset_cfg(self):
        self.cfg = copy.deepcopy(self.cfg_backup)
            
            
    def search(self):
        '''
        Main pipeline
        '''
        work_dir = self.cfg.work_dir if self.cfg.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        
        for gen in range(self.cfg.sample_times):
            theta_groups, reward_groups = self.sample()
            
            if self.cfg.baseline == 'mean':
                mean_reward = reward_groups.mean()
                std_reward = reward_groups.std()
            elif self.cfg.baseline == 'mu':
                mean_reward = reward_groups[0]
                std_reward = reward_groups.std()

            reward_groups = reward_groups - mean_reward
                
            reward_groups = reward_groups.view(-1, 1).detach()
                
            if torch.distributed.get_rank() == 0:
                init_sampler = TruncatedNormal(self.mu.detach().clone(), self.sigma.detach().clone(), 0.0, 1.0)
                init_log_prob_groups = self.sampler.log_prob(theta_groups).clone().detach()
                
                self.logger.info("... Updating")
                
                for it in range(self.cfg.update_per_sample):
                    self.mu = self.mu_x.sigmoid()
                    self.sigma = self.sigma_x.exp()
                    self.sampler = TruncatedNormal(self.mu, self.sigma, 0.0, 1.0)
                    log_prob_groups = self.sampler.log_prob(theta_groups)
                    
                    discount = (log_prob_groups - init_log_prob_groups.detach()).exp()
                    discount_clip = discount.clamp(1.0 - self.cfg.clip_eps, 1.0 + self.cfg.clip_eps)
                    
                    policy_loss = torch.min(discount * reward_groups, discount_clip * reward_groups)
                    policy_loss = -1.0 * policy_loss.mean(dim=0).sum()
                    
                    self.PolicyOptimizer.zero_grad()
                    policy_loss.backward()
                    self.PolicyOptimizer.step()
                    
                    self.logger.info(f"- {it},\tmu: {self.mu_x.sigmoid().detach().cpu().numpy()}\tsigma: {self.sigma_x.exp().detach().cpu().numpy()}")
                    
                self.logger.info(f"Current LR: {self.PolicyOptimizer.param_groups[0]['lr']}")
                self.LR_Scheduler.step()
                    
                self.mu = self.mu_x.sigmoid()
                self.sigma = self.sigma_x.exp()
                self.logger.info(f"Updated mu: {self.mu.view(-1).detach().cpu().numpy()}")
                self.logger.info(f"Updated sigma: {self.sigma.view(-1).detach().cpu().numpy()}")
            
            
            
    def sample(self):
        if torch.distributed.get_rank() == 0:
            self.mu = self.mu_x.sigmoid()
            self.sigma = self.sigma_x.exp()
            self.logger.info(f"Sampling ......")
            self.logger.info(f"mu: {self.mu.view(-1).detach().cpu().numpy()}\tsigma: {self.sigma.view(-1).detach().cpu().numpy()}")
            self.sampler = TruncatedNormal(self.mu, self.sigma, 0.0, 1.0)
        
        theta_groups = []
        reward_groups = []
        
        while len(theta_groups) < self.num_samples // (torch.distributed.get_world_size() * self.num_models_per_gpu):
            if len(theta_groups) == 0:
                theta, reward = self.sample_once(with_baseline=(self.cfg.baseline == 'mu'))
            else:
                theta, reward = self.sample_once(with_baseline=False)
            theta_groups.append(theta.detach())
            reward_groups.append(reward.detach())
            
        theta_groups = torch.cat(theta_groups, dim=0)
        reward_groups = torch.cat(reward_groups, dim=0)
        
        return theta_groups, reward_groups
        
                
    def sample_once(self, with_baseline=False):
        self._reset_cfg()
        # build model
        
        my_models = []
        optimizers = []
        my_runners = []
        for _ in range(self.num_models_per_gpu):
            # build model
            self._reset_cfg()
            my_model = build_segmentor(
                self.cfg.model).cuda()
            my_model.CLASSES = self.train_dataset.CLASSES
            my_models.append(my_model)
            # build optimizer
            optimizer = build_optimizer(my_model, self.cfg.optimizer)
            optimizers.append(optimizer)
            # build runner
            my_runner = IterBasedRunner(
                model=my_model,
                batch_processor=None,
                optimizer=optimizer,
                work_dir=self.cfg.work_dir,
                logger=logger,
                meta=meta)
            # register hooks
            my_runner.register_training_hooks(
                lr_config=self.cfg.lr_config, 
                optimizer_config=self.cfg.optimizer_config, 
                checkpoint_config=None,
                log_config=self.cfg.log_config,
                momentum_config=self.cfg.get('momentum_config', None)) 

            # an ugly walkaround to make the .log and .log.json filenames the same
            my_runner.timestamp = self.cfg.timestamp
            my_runners.append(my_runner)
        
        
        
        max_iters = self.cfg.train_iters
        workflow = self.cfg.workflow
        
        self.logger.info('workflow: %s, max: %d iters', workflow, max_iters)
        for my_runner in my_runners:
            my_runner.call_hook('before_run')
            my_runner.call_hook('before_epoch')
            my_runner._max_iters = max_iters
        dataloaders = [self.train_dataloader, self.val_dataloader]
        
        theta_group = torch.zeros((torch.distributed.get_world_size() * self.num_models_per_gpu, self.num_theta), device=torch.cuda.current_device())
        if torch.distributed.get_rank() == 0:
            if with_baseline:
                theta_group = self.sampler.sample(torch.distributed.get_world_size() * self.num_models_per_gpu - 1)
                theta_group = torch.cat([self.sampler.mu.clone().detach().view(1, -1), theta_group], dim=0)
            else:
                theta_group = self.sampler.sample(torch.distributed.get_world_size() * self.num_models_per_gpu)
        torch.distributed.broadcast(theta_group, src=0)
        
        # Convert theta to the coordinates of control points        
        theta_group_ctrl = theta_group.clone().detach()
        points_per_curve = self.num_pieces * 2 - 1
        for t in range(0, self.num_theta, 2 * points_per_curve):
            for p in range(1, points_per_curve):
                theta_group_ctrl[:, t + p*2 : t + (p+1)*2] = theta_group_ctrl[:, t + p*2 : t + (p+1)*2] * (1.0 - theta_group_ctrl[:, t + (p-1)*2 : t + p*2]) + theta_group_ctrl[:, t + (p-1)*2 : t + p*2]
        
        self.theta = theta_group[torch.distributed.get_rank()*self.num_models_per_gpu : (torch.distributed.get_rank()+1)* self.num_models_per_gpu].detach()
        self.theta_ctrl = theta_group_ctrl[torch.distributed.get_rank()*self.num_models_per_gpu : (torch.distributed.get_rank()+1)* self.num_models_per_gpu].detach()
                
        if torch.distributed.get_rank() == 0:
            self.logger.info(f"theta: {theta_group_ctrl.cpu().numpy()}")
        
        self.logger.info('... Trainig')
        while my_runners[0].iter < max_iters:
            # sample parameters
            for i, flow in enumerate(workflow):
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == 'train' and my_runners[0].iter >= max_iters:
                        break
                    iter_runner(dataloaders[i], my_runners, my_models)
                    
        time.sleep(1)  # wait for some hooks like loggers to finish

        
        reward_group = torch.zeros(torch.distributed.get_world_size() * self.num_models_per_gpu, device=torch.cuda.current_device())
        
        
        rewards = self.evaluate(self.val_dataloader, my_models)
        for i in range(self.num_models_per_gpu):
            reward_group[torch.distributed.get_rank()*self.num_models_per_gpu + i] = rewards[i].detach()
    
            print(f"RANK {torch.distributed.get_rank()} Model {i}, Eval finished.")
            
        torch.distributed.barrier()
        torch.distributed.all_reduce(reward_group)
        if torch.distributed.get_rank() == 0:
                self.logger.info(f"*** {self.target_metric}: {reward_group.view(1, -1).cpu().numpy()}")

        return theta_group, reward_group
        
        


    def train(self, data_loader, my_runners, my_models):
        '''
        Train task model
        '''
        
        for i in range(len(my_models)):
            my_models[i].train()
            my_runners[i].mode = 'train'
            my_runners[i].data_loader = data_loader
        
        data_batch = next(data_loader)
        data_batch['img'] = data_batch['img'].data[0].cuda()
        data_batch['gt_semantic_seg'] = data_batch['gt_semantic_seg'].data[0].cuda()
        
        output_list = []
        for i in range(len(my_models)):
            my_runners[i].call_hook('before_train_iter')

            outputs = my_models[i].train_step(data_batch, my_runners[i].optimizer, straight=True)
            output_list.append(outputs)
        
        for i in range(len(my_models)):
            loss = self.get_loss(output_list[i], idx=i)
            my_runners[i].outputs = {'loss': loss}
        
        for i in range(len(my_models)):
        
            log_vars = {}
            for key, value in output_list[i].items():
                if 'acc' in key:
                    log_vars[key] = value.item()
            log_vars['loss'] = my_runners[i].outputs['loss'].item()

            my_runners[i].log_buffer.update(log_vars)
            my_runners[i].call_hook('after_train_iter')
            my_runners[i]._iter += 1
            
    
    def get_loss(self, outputs, idx):
        loss = 0
        
        self.asl.theta = self.theta_ctrl[idx]
        
        for key, value in outputs.items():
            if 'loss' in key:
                pred, target, weight = value
                
                N, C, H, W = pred.shape
                
                pred = self.asl.input_softmax(pred)
                mask = (target >= 0) & (target < self.num_class)
                target_onehot = map_to_one_hot(target, self.num_class).float()
                
                pred_metric = self.asl._forward_metric(pred, target_onehot, mask)
                
                loss += weight * (1 - pred_metric)
        
        return loss
    
    
        
    def evaluate(self, data_loader, my_models):
        '''
        Evaluate the model on the val set.
        I suppose we use 1 process for each GPU, and train/evaluate 1 model on each GPU.
        To support train/evaluate 1 model on multiple GPUs, we should use the torch.distributed.new_group() function.
        I don't think we would need to train multiple models on each GPU.
        
        Notice: If we use 1 process for each GPU and train/evaluate 1 model on each GPU, the AutoLoss object can hold my_model 
        '''        
        for evaluator in self.evaluators:
            evaluator.reset()
        self._single_gpu_test(data_loader, my_models)
        
        
        if self.target_metric == 'mIoU':
            metric_class = [e.class_iou() for e in self.evaluators]
        elif self.target_metric == 'FWIoU':
            freq_class = [e.class_freq() for e in self.evaluators]
            iou_class = [e.class_iou() for e in self.evaluators]
            metric_class = [freq_class[k] * iou_class[k] * self.num_class for k in range(len(freq_class))]
        elif self.target_metric == 'BIoU':
            metric_class = [e.class_biou() for e in self.evaluators]
        elif self.target_metric == 'mAcc':
            metric_class = [e.class_acc() for e in self.evaluators]
        elif self.target_metric == 'gAcc':
            metric_class = [e.global_acc() for e in self.evaluators]
        elif self.target_metric == 'BF1':
            metric_class = [e.class_bf1() for e in self.evaluators]
            
        if self.target_metric != 'gAcc':
            metric = [m_c[~torch.isnan(m_c)].to(torch.float).mean() for m_c in metric_class]
        else:
            metric = metric_class
        
        return metric
        
    def _single_gpu_test(self, data_loader, my_models):
        '''
        Test with single GPU and add the results into evaluator.
        '''
        
        for m in my_models:
            m.eval()
        dataset = data_loader.dataset
        
        prog_bar = mmcv.ProgressBar(len(dataset))
        
        for i, data in enumerate(data_loader):
            with torch.no_grad():
                data['img'] = [data['img'][0].cuda()]
                
                for j in range(self.num_models_per_gpu):
                    result = my_models[j](data['img'], data['img_metas'], return_loss=False, val=True)
                
                    self.evaluators[j].add_batch(data['gt_semantic_seg'][0].to(torch.cuda.current_device()), result)
            
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size):
                prog_bar.update()


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')

    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args





if __name__ == '__main__':
    args = parse_args()

    cfg = Config.fromfile(args.config)
    
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        cfg.distributed = distributed
        if args.launcher == 'pytorch':
            cfg.dist_params.pop('port', None)
        init_dist(args.launcher, **cfg.dist_params)
    
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    cfg.timestamp = timestamp
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    
    logger.root.handlers.clear()
    
    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    torch.cuda.manual_seed(torch.distributed.get_rank())
    
    autoloss = AutoLoss(cfg, logger, meta)
    autoloss.search()