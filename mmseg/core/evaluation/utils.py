import torch


def map_to_one_hot(x, num_class):
    assert x.dim() == 3
    x = x.where((x >= 0) & (x < num_class), torch.tensor([num_class], dtype=x.dtype, device=x.device))
    
    B, H, W = x.shape
    x_onehot = x.new_zeros((B, num_class+1, H, W), dtype=torch.float).cuda()
    x_onehot = x_onehot.scatter_(dim=1, index=x.long().view(B, 1, H, W), value=1)[:, :-1, :, :]
    
    return x_onehot.contiguous()


def compute_edge(seg):
    # seg: (B, C, H, W)
    
    seg = seg.float()
    
    avgpool = torch.nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
    
    seg_avg = avgpool(seg.float())
    
    
    seg = (seg != seg_avg).float() * seg
        
    return seg


def dilate(seg, tol):
    # seg: (B, C, H, W)
    
    maxpool = torch.nn.MaxPool2d(2 * int(tol) + 1, stride=1, padding=int(tol))
    
    seg = maxpool(seg.float())
        
    return seg



def bfscore_val(output, target, num_class, tol):
    # output: (B, H, W)    
    # target_not_onehot: (B, H, W)
    
    N, H, W = target.shape
    
    target = map_to_one_hot(target, num_class)
    output = map_to_one_hot(output, num_class)
    
    output_edge = compute_edge(output)
    target_edge = compute_edge(target)
        
    if tol > 0:
        target_dilate = dilate(target_edge, tol)
        output_dilate = dilate(output_edge, tol)
    else:
        target_dilate = target_edge
        output_dilate = output_edge

    # output_dilate: (B, C, H, W)
    # target_dialte: (B, C, H, W)
    output_edge = output_edge
    target_edge = target_edge

    n_gt = target_edge.sum(dim=3).sum(dim=2).sum(dim=0)
    n_fg = output_edge.sum(dim=3).sum(dim=2).sum(dim=0)
    
    match_fg = torch.sum(output_edge * target_dilate, dim=3).sum(dim=2).sum(dim=0)
    match_gt = torch.sum(output_dilate * target_edge, dim=3).sum(dim=2).sum(dim=0)
    
    p = match_fg / (n_fg + 1e-8)
    r = match_gt / n_gt
    
    f1 = 2 * p * r / (p + r + 1e-8)
    
    return f1, n_fg, n_gt, match_fg, match_gt


def boundary_confusion_matrix(output, target, num_class, radius, drop_bg=False):
    # output: (B, H, W)
    # target: (B, H, W)
    
    target_onehot = map_to_one_hot(target, num_class)
    
    target_edge = compute_edge(target_onehot)
    
    if radius > 0:
        target_dilate = dilate(target_edge, radius)
    else:
        target_dilate = target_edge
        
    # trimap: (B, H, W), binary
    # pred: (B, H, W)
    # target: (B, H, W)
    
    if drop_bg:
        trimap = target_dilate[:, 1:, :, :].max(dim=1).values.bool()
    else:
        trimap = target_dilate.max(dim=1).values.bool()
    
    valid_mask = (target >= 0) & (target < num_class)
    mask = valid_mask & trimap
    
    label = num_class * target[mask].long() + output[mask]
    count = torch.bincount(label, minlength=num_class**2)
    confusion_matrix = count.view(num_class, num_class)
    
    return confusion_matrix