import torch.nn as nn
import segmentation_models_pytorch as smp

class CombinedLoss(nn.Module):
    """
    组合 Focal Loss 与 Tversky Loss: alpha * Focal + (1-alpha) * Tversky
    """
    def __init__(self, alpha=0.5):
        super().__init__()
        self.focal = smp.losses.FocalLoss(mode='binary')
        self.tversky = smp.losses.TverskyLoss(mode='binary')
        self.alpha = alpha

    def forward(self, outputs, targets):
        focal_loss = self.focal(outputs, targets)
        tversky_loss = self.tversky(outputs, targets)
        return self.alpha * focal_loss + (1 - self.alpha) * tversky_loss