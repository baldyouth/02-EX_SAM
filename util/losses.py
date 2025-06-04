import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import HausdorffDTLoss, LogHausdorffDTLoss
from util.cross_entropy_loss import CrossEntropyLoss
from util.focal_loss import FocalLoss

def dice_score(pred_probs, targets, threshold=0.5, eps=1e-6):
    preds = (pred_probs > threshold).float()
    targets = targets.float()

    # shape: (B, H, W) or (B, 1, H, W)
    dims = tuple(range(1, preds.ndim))  # (1, 2, 3) if (B, 1, H, W)

    intersection = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.sum()

def iou_score(pred_probs, targets, threshold=0.5, eps=1e-6):
    preds = (pred_probs > threshold).float()
    targets = targets.float()

    dims = tuple(range(1, preds.ndim))

    intersection = (preds * targets).sum(dim=dims)
    union = (preds + targets).clamp(0, 1).sum(dim=dims)
    iou = (intersection + eps) / (union + eps)
    return iou.sum()

# class FocalLoss(nn.Module):
#     def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction

#     def forward(self, logits: torch.Tensor, targets: torch.Tensor):
#         bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
#         prob = torch.sigmoid(logits)
#         p_t = targets * prob + (1 - targets) * (1 - prob)
#         modulator = (1 - p_t).pow(self.gamma)
#         loss = self.alpha * modulator * bce_loss

#         if self.reduction == "mean":
#             return loss.mean()
#         elif self.reduction == "sum":
#             return loss.sum()
#         return loss

class TverskyLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.3,
                 beta:  float = 0.7,
                 smooth: float = 1e-6,
                 reduction: str = 'mean'):
        """
        Args:
            alpha: 假阴性惩罚因子
            beta:  假阳性惩罚因子
            smooth: 防止分母为0的小常数
        """
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):

        prob = torch.sigmoid(logits)

        prob = prob.view(-1)
        targets = targets.view(-1).float()

        TP = (prob * targets).sum()
        FP = (prob * (1 - targets)).sum()
        FN = ((1 - prob) * targets).sum()

        tversky_coef = (TP + self.smooth) / \
                       (TP + self.alpha * FN + self.beta * FP + self.smooth)

        loss = 1.0 - tversky_coef

        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss * prob.numel()
        else:
            return loss

class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.hd = LogHausdorffDTLoss(
            sigmoid = True,
            include_background = False
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        return self.hd(logits, targets)

class HardNegativeMiningLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, threshold: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.gamma = gamma
        self.threshold = threshold
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        fp_mask = ((probs > self.threshold) & (targets == 0)).float()
        loss_map = fp_mask * (probs ** self.gamma)
        loss = loss_map.sum() / (fp_mask.sum() + self.eps)
        return loss

# class CombinedLoss(nn.Module):
#     def __init__(self,
#                 alpha_focal: float = 0.3,
#                 alpha_tversky: float = 0.4,
#                 alpha_boundary: float = 0.2,
#                 alpha_hnm: float = 0.1):
#         super().__init__()
#         self.focal       = FocalLoss()
#         self.tversky     = TverskyLoss(alpha=0.3, beta=0.7)
#         self.boundary    = BoundaryLoss()
#         self.hard_negative_mining = HardNegativeMiningLoss()

#         self.alpha_f     = alpha_focal
#         self.alpha_t     = alpha_tversky
#         self.alpha_b     = alpha_boundary
#         self.alpha_hnm   = alpha_hnm
    

#     def forward(self, logits, targets):
#         loss_f = self.focal(logits, targets)
#         loss_t = self.tversky(logits, targets)
#         loss_b   = self.boundary(logits, targets)
#         loss_hnm = self.hard_negative_mining(logits, targets)

#         total = (
#             self.alpha_f   * loss_f +
#             self.alpha_t   * loss_t +
#             self.alpha_b   * loss_b +
#             self.alpha_hnm * loss_hnm
#         )
#         return total
    
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = CrossEntropyLoss(use_sigmoid=True, loss_weight=0.3, pos_weight=[3.0])
        self.focal_loss = FocalLoss(loss_weight=0.7)

    def forward(self, pred, target):
        loss1 = self.ce_loss(pred, target)
        loss2 = self.focal_loss(pred, target)
        return loss1 + loss2
