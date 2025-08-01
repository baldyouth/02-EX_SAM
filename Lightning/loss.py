import torch
import torch.nn.functional as F

from util.cross_entropy_loss import CrossEntropyLoss
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        """
        alpha: 控制 False Positive (FP) 惩罚
        beta: 控制 False Negative (FN) 惩罚
        alpha + beta = 1 时退化为 Dice
        alpha < beta 时更关注减少 FN（适合小目标）
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, y_pred, y_true):

        y_pred = torch.sigmoid(y_pred)
        y_true = y_true.float()
        
        # Flatten
        y_pred = y_pred.view(y_pred.size(0), -1)
        y_true = y_true.view(y_true.size(0), -1)

        TP = (y_pred * y_true).sum(dim=1)
        FP = ((1 - y_true) * y_pred).sum(dim=1)
        FN = (y_true * (1 - y_pred)).sum(dim=1)

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        loss = 1 - tversky

        return loss.mean()
    
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2, gamma=2, smooth=1e-6):
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

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, x, y):
        x = torch.sigmoid(x)
        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()

        return 1 - dc

class bce_dice(nn.Module):
    def __init__(self):
        super(bce_dice, self).__init__()
        self.bce_fn = nn.BCEWithLogitsLoss()
        self.dice_fn = DiceLoss()
        # self.tversky_fn = TverskyLoss(alpha=0.7, beta=0.3)
        
        # self.edge_fn = EdgeWeightedLoss(mode='sobel')

    def forward(self, y_pred, y_true):
        bce = self.bce_fn(y_pred, y_true)
        # focal = sigmoid_focal_loss(y_pred, y_true, alpha=0.25, gamma=2, reduction='mean')

        dice = self.dice_fn(y_pred.sigmoid(), y_true)
        # tversky = self.tversky_fn(y_pred, y_true)

        # edge = self.edge_fn(y_pred, y_true)
        
        return 1 * bce + 5 * dice

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = CrossEntropyLoss(use_sigmoid=True, loss_weight=1, pos_weight=[5.0])
        self.tversky_loss = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.5)
        self.edge_loss = EdgeWeightedLoss(mode='sobel')

    def forward(self, pred, target):
        ce_loss = self.ce_loss(pred, target)
        tversky_loss = self.tversky_loss(pred, target)
        edge_loss = self.edge_loss(pred, target)
        return 0.4*ce_loss + 0.4*tversky_loss + 0.2*edge_loss
