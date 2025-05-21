import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = targets * prob + (1 - targets) * (1 - prob)
        modulator = (1 - p_t).pow(self.gamma)
        loss = self.alpha * modulator * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

class TverskyLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.5,
                 beta:  float = 0.5,
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

class CombinedLoss(nn.Module):
    """
    组合 Focal Loss 与 Tversky Loss: alpha * Focal + (1-alpha) * Tversky
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.focal = FocalLoss()
        self.tversky = TverskyLoss()
        self.alpha = alpha

    def forward(self, outputs, targets):
        focal_loss = self.focal(outputs, targets)
        tversky_loss = self.tversky(outputs, targets)
        return self.alpha * focal_loss + (1 - self.alpha) * tversky_loss