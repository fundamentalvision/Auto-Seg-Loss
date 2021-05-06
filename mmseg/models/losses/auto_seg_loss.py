import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss

import pdb

from ...core.evaluation.utils import map_to_one_hot, compute_edge, dilate


def map_to_one_hot(x, num_class):
    assert x.dim() == 3
    B, H, W = x.shape
    x = x.where((x >= 0) & (x < num_class), torch.tensor([num_class], dtype=x.dtype, device=x.device))
    x_onehot = x.new_zeros((B, num_class+1, H, W), dtype=torch.float)
    x_onehot = x_onehot.scatter_(dim=1, index=x.long().view(B, 1, H, W), value=1)[:, :-1, :, :]
    return x_onehot.contiguous()


@LOSSES.register_module()
class AutoSegLoss(nn.Module):
    def __init__(self,
                 num_class,
                 theta, 
                 parameterization=None, 
                 target_metric='mIoU',
                 drop_bg=False,
                 tol=5,
                 inplace=True, 
                 eps=1e-8,
                 loss_weight=1.0):
        super(AutoSegLoss, self).__init__()
        
        self.num_class = int(num_class)
        self.inplace = inplace
        
        if isinstance(theta, torch.Tensor):
            self.theta = theta.detach().clone()
        else:
            self.theta = torch.tensor(theta, dtype=torch.float, device=torch.cuda.current_device())
        
        allowed_parameterization = ['bezier', 'linear']
        if parameterization not in allowed_parameterization:
            raise KeyError('parameterization {} is not supported'.format(parameterization))
        self.parameterization = parameterization
        
        self.target_metric = target_metric
        
        self.tol = tol
        self.drop_bg = drop_bg
        
        self.input_softmax = nn.Softmax(dim=1)
        self.eps = eps
        self.loss_weight = loss_weight
        
        
    def _quadratic_solver(self, x0, x1, x2, x):
        x1_new = (x1 - x0) / (x2 - x0)
        if self.inplace:
            x_new = x.add_(-x0).div_(x2 - x0)
        else:
            x_new = (x - x0) / (x2 - x0)

        if x1_new == 0.5:
            t = x_new
        else:
            if self.inplace:
                t = x_new.mul_(1-2*x1_new).add_(x1_new**2).sqrt_().mul(-1).add_(x1_new).div_(2*x1_new - 1)
            else:
                t = (x1_new - torch.sqrt(x1_new**2 + (1-2*x1_new)*x_new)) / (2*x1_new - 1)

        return t

    def _bezier(self, x, control_points):
        
        cp = torch.cat([control_points, control_points.new_ones(2)])
        y_list = []
        for n in range(0, len(cp), 4):
            if n == 0:
                x_part = x.where(x <= cp[n+2], cp[n+2])
                if self.inplace:
                    t = self._quadratic_solver(x_part.new_zeros(1), cp[n], cp[n+2], x_part)
                    y = t.mul(2*cp[n+1]).add_(t.square().mul_(cp[n+3] - 2*cp[n+1]))
                else:
                    t = self._quadratic_solver(x_part.new_zeros(1), cp[n], cp[n+2], x_part)
                    y = -2*t*(t-1) * cp[n+1] + t**2 * cp[n+3]
            else:
                x_part = x.where((x > cp[n-2]) & (x <= cp[n+2]), cp[n+2])
                if self.inplace:
                    t = self._quadratic_solver(cp[n-2], cp[n], cp[n+2], x_part)
                    y = t.mul(2*cp[n+1] - 2*cp[n-1]).add_(t.square().mul_(cp[n-1] - 2*cp[n+1] + cp[n+3])).add_(cp[n-1])
                else:
                    t = self._quadratic_solver(cp[n-2], cp[n], cp[n+2], x_part)
                    y = (t-1)**2 * cp[n-1] - 2*t*(t-1) * cp[n+1] + t**2 * cp[n+3]

            y_list.append(y)

        y = y_list[0]
        for n, y_part in enumerate(y_list[1:]):
            y = y_part.where((x > cp[4*n+2]) & (x <= cp[4*n+6]), y)

        return y
    
    def parameterize(self, x, control_points):
        if self.parameterization == 'bezier':
            return self._bezier(x, control_points)
        else:
            raise KeyError('parameterization {} is not implemented'.format(self.parameterization))
            
    def _forward_metric(self, pred, target_onehot, mask):
        
        
        if self.target_metric == 'mIoU':
            pred_metric = forward_mIoU(pred, target_onehot, mask, self.parameterize, self.theta, self.eps)
            
        elif self.target_metric == 'FWIoU':
            pred_metric = forward_FWIoU(pred, target_onehot, mask, self.parameterize, self.theta, self.eps)
            
        elif self.target_metric == 'BIoU':
            pred_metric = forward_BIoU(pred, target_onehot, mask, self.parameterize, self.theta, self.eps, self.drop_bg, self.tol)
            
        elif self.target_metric == 'mAcc':
            pred_metric = forward_mAcc(pred, target_onehot, mask, self.parameterize, self.theta, self.eps)
            
        elif self.target_metric == 'gAcc':
            pred_metric = forward_gAcc(pred, target_onehot, mask, self.parameterize, self.theta, self.eps)
            
        elif self.target_metric == 'BF1':
            pred_metric = forward_BF1(pred, target_onehot, mask, self.parameterize, self.theta, self.eps, self.drop_bg, self.tol)
            
        else:
            raise KeyError('target_metric {} is not implemented'.format(self.target_metric))
            
        return pred_metric
        
        
    def forward(self,
                cls_score,
                label,
                **kwargs):
        
        target = label.where((label >= 0) & (label < self.num_class), torch.tensor([self.num_class], dtype=label.dtype, device=label.device))
        mask = (target >= 0) & (target < self.num_class)
        
        
        target_onehot = map_to_one_hot(target, self.num_class).float()
        pred = self.input_softmax(cls_score)
                
        pred_metric = self._forward_metric(pred, target_onehot, mask)
        
        loss = self.loss_weight * (1 - pred_metric)
        
        return loss
        
        
    
    
def forward_mIoU(input, target, mask, parameterize, theta, eps):
    
    N, C, H, W = input.shape
    
    mask = mask.float()

    input_inter = parameterize(input, theta[:len(theta)//2])
    input_union = parameterize(input, theta[len(theta)//2:])

    intersection = input_inter * target
    union = input_union + target - input_union * target

    intersection = intersection * mask.view(N, 1, H, W)
    union = union * mask.view(N, 1, H, W)

    intersection = intersection.sum(dim=3).sum(dim=2).sum(dim=0)
    union = union.sum(dim=3).sum(dim=2).sum(dim=0) + eps

    iou = intersection / union

    return iou.mean()
    

def forward_FWIoU(input, target, mask, parameterize, theta, eps):
    
    N, C, H, W = input.shape
    
    mask = mask.float()

    input_inter = parameterize(input, theta[:len(theta)//2])
    input_union = parameterize(input, theta[len(theta)//2:])

    intersection = input_inter * target
    union = input_union + target - input_union * target

    intersection = intersection * mask.view(N, 1, H, W)
    union = union * mask.view(N, 1, H, W)

    intersection = intersection.sum(dim=3).sum(dim=2).sum(dim=0)
    union = union.sum(dim=3).sum(dim=2).sum(dim=0) + eps

    iou = intersection / union
    
    freq = target.sum(dim=3).sum(dim=2).sum(dim=0)
    
    fw_iou = (iou[freq>0] * freq[freq>0]).sum() / freq[freq>0].sum()

    return fw_iou


def forward_BIoU(input, target, mask, parameterize, theta, eps, drop_bg, tol):
    target_edge = compute_edge(target)
    if tol > 0:
        target_dilate = dilate(target_edge, tol)
    else:
        target_dilate = target_edge
    if drop_bg:
        trimap = target_dilate[:, 1:, :, :].max(dim=1).values > 0.5
    else:
        trimap = target_dilate.max(dim=1).values > 0.5
        
        
    mask = (mask & trimap).float()

    N, C, H, W = input.shape

    input_inter = parameterize(input, theta[:len(theta)//2])
    input_union = parameterize(input, theta[len(theta)//2:])

    intersection = input_inter * target
    union = input_union + target - input_union * target

    intersection = intersection * mask.view(N, 1, H, W)
    union = union * mask.view(N, 1, H, W)
    target = target * mask.view(N, 1, H, W)

    intersection = intersection.sum(dim=3).sum(dim=2).sum(dim=0)
    union = union.sum(dim=3).sum(dim=2).sum(dim=0) + eps

    iou = intersection / union
    
    return iou.mean()


def forward_mAcc(input, target, mask, parameterize, theta, eps):
    N, C, H, W = input.shape
    
    mask = mask.float()

    input_inter = parameterize(input, theta)

    intersection = input_inter * target

    intersection = intersection.sum(dim=3).sum(dim=2).sum(dim=0)

    freq = target.sum(dim=3).sum(dim=2).sum(dim=0)

    acc = intersection / (freq + eps)
    
    return acc.mean()


def forward_gAcc(input, target, mask, parameterize, theta, eps):
    N, C, H, W = input.shape
    
    mask = mask.float()
    
    input_inter = parameterize(input, theta)

    intersection = input_inter * target

    intersection = intersection.sum()

    freq = target.sum()

    acc = intersection / (freq + eps)
    
    return acc


def forward_BF1(input, target, mask, parameterize, theta, eps, drop_bg, tol):
    N, C, H, W = input.shape

    # edge
    target_pad = torch.nn.ConstantPad2d(1, 1.0)(target)
    target_erode = -1 * torch.nn.MaxPool2d(3, stride=1, padding=0)(-target_pad)
    target_edge = (target_erode != target).float()

    input_pad = torch.nn.ConstantPad2d(1, 1.0)(input)
    input_erode = -1 * torch.nn.MaxPool2d(3, stride=1, padding=0)(-input_pad)
    
    theta_1 = theta[:len(theta)//4]
    theta_2 = theta[len(theta)//4:2*len(theta)//4]
    theta_3 = theta[2*len(theta)//4:3*len(theta)//4]
    theta_4 = theta[3*len(theta)//4:]

    intersection = parameterize(input, theta_1) * parameterize(input_erode, theta_2)
    union = parameterize(input, theta_3) + parameterize(input_erode, theta_4) - parameterize(input, theta_3) * parameterize(input_erode, theta_4) 

    input_edge = union - intersection


    if tol > 0:
        target_dilate = dilate(target_edge, tol)
        input_dilate = dilate(input_edge, tol)
    else:
        target_dilate = target_edge
        input_dilate = input_edge

    # input_dilate: (B, C, H, W)
    # target_dialte: (B, C, H, W)

    freq = target_edge.sum(dim=3).sum(dim=2).sum(dim=0)

    n_gt = target_edge.sum(dim=3).sum(dim=2).sum(dim=0) + eps
    n_fg = input_edge.sum(dim=3).sum(dim=2).sum(dim=0) + eps

    match_fg = torch.sum(input_edge * target_dilate, dim=3).sum(dim=2).sum(dim=0)
    match_gt = torch.sum(input_dilate * target_edge, dim=3).sum(dim=2).sum(dim=0)

    p = match_fg / n_fg
    r = match_gt / n_gt

    pred_f1 = (2 * p * r / (p + r + eps)).view(-1)

    if drop_bg:
        pred_f1 = pred_f1[1:][freq[1:] != 0].mean()
    else:
        pred_f1 = pred_f1[freq != 0].mean()

    return pred_f1
