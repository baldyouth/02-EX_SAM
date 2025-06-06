import torch
import torch.nn.functional as F
from monai.losses import HausdorffDTLoss, LogHausdorffDTLoss
from util.focal_loss import FocalLoss

from util.cross_entropy_loss import CrossEntropyLoss
import torch.nn as nn

# HardNegativeMiningLoss
# LogHausdorffDTLoss

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
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

class EdgeWeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1., 0., 1.],
                                 [-2., 0., 2.],
                                 [-1., 0., 1.]]).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.],
                                 [ 0.,  0.,  0.],
                                 [ 1.,  2.,  1.]]).view(1, 1, 3, 3)

        # 使用 register_buffer，确保不会被优化器更新，并随模型一起转移 device
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self, logits, gt_mask):
        B, C, H, W = logits.shape
        assert C == 1, "EdgeWeightedLoss 只支持单通道输出"

        def sobel_edge(x):
            # 确保 kernel 和输入在相同 dtype 和 device 下
            kernel_x = self.kernel_x.to(dtype=x.dtype, device=x.device)
            kernel_y = self.kernel_y.to(dtype=x.dtype, device=x.device)

            x = x.float() if x.dtype in [torch.half, torch.bfloat16] else x  # 避免 sqrt 出现 nan
            grad_x = F.conv2d(x, kernel_x, padding=1)
            grad_y = F.conv2d(x, kernel_y, padding=1)
            edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
            return edge

        pred_edge = sobel_edge(logits)
        gt_edge = sobel_edge(gt_mask)

        # 归一化 gt_edge 作为权重图
        max_per_sample = gt_edge.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1) + 1e-6
        weight_map = (gt_edge / max_per_sample).detach()

        # 计算加权 L1 loss
        loss = F.l1_loss(pred_edge, gt_edge, reduction='none')
        loss = (loss * weight_map).mean()

        return loss

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = CrossEntropyLoss(use_sigmoid=True, loss_weight=1, pos_weight=[2.0])
        self.tversky_loss = TverskyLoss()
        self.edge_loss = EdgeWeightedLoss()

    def forward(self, pred, target):
        loss1 = self.ce_loss(pred, target)
        loss2 = self.edge_loss(pred, target)
        loss3 = self.tversky_loss(pred, target)
        return 0.5*loss1 + 0.25*loss2 + 0.25*loss3
