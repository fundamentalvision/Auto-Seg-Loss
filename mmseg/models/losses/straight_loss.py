import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weight_reduce_loss


@LOSSES.register_module()
class StraightLoss(nn.Module):
    def __init__(self, loss_weight=1.0, **kwargs):
        super(StraightLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self,
                cls_score,
                label,
                **kwargs):
        return [cls_score, label, self.loss_weight]
