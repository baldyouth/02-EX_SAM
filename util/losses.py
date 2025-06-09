import torch
import torch.nn.functional as F
from monai.losses import HausdorffDTLoss, LogHausdorffDTLoss
from util.focal_loss import FocalLoss

from util.cross_entropy_loss import CrossEntropyLoss
import torch.nn as nn

# HardNegativeMiningLoss
# LogHausdorffDTLoss

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.4, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        targets = targets.float()

        TP = (preds * targets).sum(dim=(1,2,3))
        FP = ((1 - targets) * preds).sum(dim=(1,2,3))
        FN = (targets * (1 - preds)).sum(dim=(1,2,3))

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        loss = 1 - tversky
        return loss.mean()
    
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.4, gamma=1.33, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        targets = targets.float()

        TP = (preds * targets).sum(dim=(1, 2, 3))
        FP = ((1 - targets) * preds).sum(dim=(1, 2, 3))
        FN = (targets * (1 - preds)).sum(dim=(1, 2, 3))

        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        loss = torch.pow((1 - tversky_index), self.gamma)
        return loss.mean()

class EdgeWeightedLoss(nn.Module):
    def __init__(self, mode='laplacian'):
        super().__init__()
        assert mode in ['sobel', 'laplacian'], "mode 只能是 'sobel' 或 'laplacian'"
        self.mode = mode

        # Sobel 核
        kernel_x = torch.tensor([[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]]).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.],
                                 [ 0.,  0.,  0.],
                                 [ 1.,  2.,  1.]]).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

        # Laplacian 核
        laplacian_kernel = torch.tensor([[0.,  1., 0.],
                                         [1., -4., 1.],
                                         [0.,  1., 0.]]).view(1, 1, 3, 3)
        self.register_buffer('laplacian_kernel', laplacian_kernel)

    def edge_detect(self, x):
        x = x.float() if x.dtype in [torch.half, torch.bfloat16] else x
        if self.mode == 'sobel':
            kx = self.kernel_x.to(dtype=x.dtype, device=x.device)
            ky = self.kernel_y.to(dtype=x.dtype, device=x.device)
            grad_x = F.conv2d(x, kx, padding=1)
            grad_y = F.conv2d(x, ky, padding=1)
            edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        elif self.mode == 'laplacian':
            klap = self.laplacian_kernel.to(dtype=x.dtype, device=x.device)
            edge = F.conv2d(x, klap, padding=1).abs()
        return edge

    def forward(self, logits, gt_mask):
        B, C, H, W = logits.shape
        assert C == 1, "EdgeWeightedLoss 只支持单通道输出"

        pred_edge = self.edge_detect(logits)
        gt_edge = self.edge_detect(gt_mask)

        # 归一化 gt_edge 为权重图
        max_per_sample = gt_edge.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1) + 1e-6
        weight_map = (gt_edge / max_per_sample).detach()

        # 加权 L1 Loss
        loss = F.l1_loss(pred_edge, gt_edge, reduction='none')
        return (loss * weight_map).mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = CrossEntropyLoss(use_sigmoid=True, loss_weight=1, pos_weight=[2.0])
        self.tversky_loss = FocalTverskyLoss()
        self.edge_loss = EdgeWeightedLoss(mode='laplacian')

    def forward(self, pred, target):
        loss1 = self.ce_loss(pred, target)
        loss2 = self.edge_loss(pred, target)
        loss3 = self.tversky_loss(pred, target)
        return 0.4*loss1 + 0.3*loss2 + 0.3*loss3
