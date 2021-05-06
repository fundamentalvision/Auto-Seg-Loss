import torch
from .utils import bfscore_val, boundary_confusion_matrix


class Evaluator(object):
    def __init__(self, num_class, drop_bg):
        self.num_class = num_class
        self.drop_bg = drop_bg
        self.confusion_matrix = torch.zeros((self.num_class,)*2).to(torch.long).cuda()
        self.confusion_matrix_boundary = torch.zeros((self.num_class,)*2).to(torch.long).cuda()
        self.n_fg = None
        self.n_gt = None
        self.match_fg = None
        self.match_gt = None

    def class_iou(self):
        iou = torch.diag(self.confusion_matrix).to(torch.float) / (
                    torch.sum(self.confusion_matrix, dim=1) + torch.sum(self.confusion_matrix, dim=0) -
                    torch.diag(self.confusion_matrix)).to(torch.float)
        return iou

    def class_freq(self):
        freq = torch.sum(self.confusion_matrix, dim=1).to(torch.float) / torch.sum(self.confusion_matrix).to(torch.float)
        return freq

    def class_biou(self):
        biou = torch.diag(self.confusion_matrix_boundary).to(torch.float) / (
                    torch.sum(self.confusion_matrix_boundary, dim=1) + torch.sum(self.confusion_matrix_boundary, dim=0) -
                    torch.diag(self.confusion_matrix_boundary)).to(torch.float)
        return biou
    
    def class_bf1(self):
        p = self.match_fg / (self.n_fg + 1e-8)
        r = self.match_gt / self.n_gt
    
        bf1 = 2 * p * r / (p + r + 1e-8)
        bf1 = bf1.view(-1)
        if self.drop_bg:
            bf1 = bf1[1:]
        
        return bf1
    
    def class_acc(self):
        acc = torch.diag(self.confusion_matrix).to(torch.float) / self.confusion_matrix.sum(dim=1).to(torch.float)
        return acc

    def global_acc(self):
        global_acc = torch.diag(self.confusion_matrix).to(torch.float).sum() / self.confusion_matrix.to(torch.float).sum()
        return global_acc

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].to(torch.long) + pre_image[mask]
        count = torch.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix
    
    def _generate_matrix_boundary(self, gt_image, output, radius):
        return boundary_confusion_matrix(output, gt_image, self.num_class, radius, self.drop_bg)
    
    def add_batch(self, gt_image, output, tol=5, radius=5):
        assert gt_image.shape == output.shape
        self.confusion_matrix += self._generate_matrix(gt_image, output)
        self.confusion_matrix_boundary += self._generate_matrix_boundary(gt_image, output, radius)

        _, n_fg, n_gt, match_fg, match_gt = bfscore_val(output, gt_image, self.num_class, tol)
        if self.n_fg is None:
            self.n_fg = n_fg
            self.n_gt = n_gt
            self.match_fg = match_fg
            self.match_gt = match_gt
        else:
            self.n_fg += n_fg
            self.n_gt += n_gt
            self.match_fg += match_fg
            self.match_gt += match_gt
    
    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_class,) * 2).to(torch.long).cuda()
        self.confusion_matrix_boundary = torch.zeros((self.num_class,) * 2).to(torch.long).cuda()
        self.n_fg = None
        self.n_gt = None
        self.match_fg = None
        self.match_gt = None


def metrics_fast(results, gt_seg_maps, num_classes, label_map=dict(), reduce_zero_label=False, drop_bg=False):
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    
    evaluator = Evaluator(num_classes, drop_bg)
    
    for i in range(num_imgs):
        label = torch.from_numpy(gt_seg_maps[i])
        if label_map is not None:
            for old_id, new_id in label_map.items():
                label[label == old_id] = new_id
        if reduce_zero_label:
            label[label == 0] = 255
            label = label - 1
            label[label == 254] = 255
        evaluator.add_batch(label[None, ...].cuda(), torch.from_numpy(results[i])[None, ...].cuda())
    
    class_iou = evaluator.class_iou()
    class_freq = evaluator.class_freq()
    class_biou = evaluator.class_biou()
    class_bf1 = evaluator.class_bf1()
    class_acc = evaluator.class_acc()
    global_acc = evaluator.global_acc()
    
    return class_iou.cpu().numpy(), class_freq.cpu().numpy(), class_biou.cpu().numpy(), class_bf1.cpu().numpy(), class_acc.cpu().numpy(), global_acc.cpu().numpy()