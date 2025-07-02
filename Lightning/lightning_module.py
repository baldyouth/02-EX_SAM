import pytorch_lightning as pl
import torch

from model.img_model import ImgModel
from model.model_lightning import Model_Lightning
from .loss import bce_dice
from .valid import miou_from_binary_preds

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
    def __init__(self, model_config, optimizer_config, scheduler_config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        # config
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config

        # model
        self.model = ImgModel(self.model_config)
        # self.model = Model_Lightning(self.model_config)

        # train loss
        self.bce_dice_loss = bce_dice()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx, *args, **kwargs):
        inputs, targets = batch
        logits = self(inputs)

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

        val_bce_dice_loss = self.bce_dice_loss(logits, targets)
        self.log("val_bce_dice_loss", val_bce_dice_loss, on_step=False, on_epoch=True, prog_bar=True)

        val_miou = miou_from_binary_preds(logits.sigmoid()>0.5, targets)
        self.log("val_miou", val_miou, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        sam_lr = self.optimizer_config['sam_lr']
        not_sam_lr = self.optimizer_config['not_sam_lr']
        weight_decay = self.optimizer_config['weight_decay']

        sam_params = [p for n, p in self.named_parameters() if 'sam' in n and p.requires_grad]
        not_sam_params = [p for n, p in self.named_parameters() if 'sam' not in n and p.requires_grad]

        if self.optimizer_config['name'].lower() == 'adamw':
            optimizer = torch.optim.AdamW([
                    {'params': sam_params, 'lr': sam_lr},
                    {'params': not_sam_params, 'lr': not_sam_lr}
                ], 
                weight_decay=weight_decay)
        elif self.optimizer_config['name'].lower() == 'adam':
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
        elif self.scheduler_config['name'].lower() == 'poly':
            warmup_steps = self.scheduler_config['warmup_steps']
            total_steps = self.scheduler_config['total_steps']
            power = self.scheduler_config.get['power']
            eta_min = self.scheduler_config.get['eta_min']

            def poly_lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                else:
                    decay_step = current_step - warmup_steps
                    decay_total = max(1, total_steps - warmup_steps)
                    poly_decay = (1 - decay_step / decay_total) ** power
                    min_lr_factor = eta_min / self.scheduler_config['base_lr']
                    return poly_decay * (1 - min_lr_factor) + min_lr_factor

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=poly_lr_lambda
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
