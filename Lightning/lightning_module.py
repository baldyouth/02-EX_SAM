import pytorch_lightning as pl
import torch
from torchmetrics.classification import BinaryJaccardIndex
# from transformers import get_cosine_schedule_with_warmup

from model.img_model import ImgModel
from .loss import CrossEntropyLoss, FocalTverskyLoss, EdgeWeightedLoss, bce_dice

import math
from torch.optim.lr_scheduler import _LRScheduler

# Warmup CosineAnnealingLR
class CosineAnnealingWarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_steps, eta_min=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps and self.warmup_steps != 0:
            # 线性warmup阶段
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        elif step <= self.max_steps:
            # 余弦衰减阶段，衰减到eta_min
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return [self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress))
                    for base_lr in self.base_lrs]
        else:
            # 超出总步数，保持最低学习率
            return [self.eta_min for _ in self.base_lrs]

#!!! module
class LitModule(pl.LightningModule):
    def __init__(self, optimizer_config, scheduler_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        # config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        # model
        self.model = ImgModel()

        # train loss
        # self.ce_loss = CrossEntropyLoss(use_sigmoid=True, loss_weight=1, pos_weight=[5.0]) #TODO
        # self.tversky_loss = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1) #TODO
        # self.edge_loss = EdgeWeightedLoss(mode='sobel') #TODO
        self.bce_dice_loss = bce_dice()

        # val loss
        self.iou_metric = BinaryJaccardIndex(threshold=0.5)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx, *args, **kwargs):
        inputs, targets = batch
        logits = self(inputs)

        # ce_loss = self.ce_loss(logits, targets)
        # tversky_loss = self.tversky_loss(logits, targets)
        # edge_loss = self.edge_loss(logits, targets)
        # combine_loss = 0.4*ce_loss + 0.4*tversky_loss + 0.2*edge_loss
        # self.log("ce_loss", ce_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("tversky_loss", tversky_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("edge_loss", edge_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("combine_loss", combine_loss, on_step=True, on_epoch=True, prog_bar=True)

        train_bce_dice_loss = self.bce_dice_loss(logits, targets)
        self.log("train_bce_dice_loss", train_bce_dice_loss, on_step=True, on_epoch=True, prog_bar=True)

        opt = self.trainer.optimizers[0]
        lr_sam = opt.param_groups[0]['lr']
        lr_not_sam = opt.param_groups[1]['lr']

        self.log("lr_sam", lr_sam, on_epoch=True, prog_bar=False, logger=True)
        self.log("lr_not_sam", lr_not_sam, on_epoch=True, prog_bar=False, logger=True)

        return train_bce_dice_loss
    
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        inputs, targets = batch
        logits = self(inputs)

        # ce_loss = self.ce_loss(logits, targets)
        # tversky_loss = self.tversky_loss(logits, targets)
        # edge_loss = self.edge_loss(logits, targets)
        # combine_loss = 0.4*ce_loss + 0.4*tversky_loss + 0.2*edge_loss
        # self.log("ce_loss", ce_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("tversky_loss", tversky_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("edge_loss", edge_loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log("combine_loss", combine_loss, on_step=True, on_epoch=True, prog_bar=True)

        val_bce_dice_loss = self.bce_dice_loss(logits, targets)
        self.log("val_bce_dice_loss", val_bce_dice_loss, on_step=False, on_epoch=True, prog_bar=True)

        val_iou = self.iou_metric(logits.sigmoid(), targets.int())
        self.log("val_iou", val_iou, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        sam_lr = self.optimizer_config['sam_lr']
        not_sam_lr = self.optimizer_config['not_sam_lr']
        weight_decay = self.optimizer_config['weight_decay']

        sam_params = [p for n, p in self.named_parameters() if 'sam' in n and p.requires_grad]
        not_sam_params = [p for n, p in self.named_parameters() if 'sam' not in n and p.requires_grad]

        optimizer = torch.optim.Adam([
                {'params': sam_params, 'lr': sam_lr},
                {'params': not_sam_params, 'lr': not_sam_lr}
            ], 
            weight_decay=weight_decay)

        if self.scheduler_config['name'].lower() == 'cosine_restart':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, 
                    T_0=self.scheduler_config['T_0'], 
                    T_mult=self.scheduler_config['T_mult'], 
                    eta_min=self.scheduler_config['eta_min'])
        
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        elif self.scheduler_config['name'].lower() == 'cosine':
            warmup_steps = self.scheduler_config['warmup_steps']
            if warmup_steps > 0:
                scheduler = CosineAnnealingWarmupLR(
                    optimizer,
                    warmup_steps=warmup_steps,
                    max_steps=self.scheduler_config['total_steps'],
                    eta_min=self.scheduler_config['eta_min']
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.scheduler_config['total_steps'],
                    eta_min=self.scheduler_config['eta_min']
                )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            }
        else:
            raise ValueError(f"Unsupported scheduler name: {self.scheduler_config['name']}")
