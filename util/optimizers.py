import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, ReduceLROnPlateau

def get_optimizer(model, training_cfg):
    opt = training_cfg.get('optimizer', 'adamw').lower()
    lr = training_cfg.get('base_lr', 1e-3)
    wd = training_cfg.get('weight_decay', 1e-4)
    if opt == 'adamw':
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                 lr=lr, 
                                 weight_decay=wd)
    elif opt == 'sgd':
        return torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr=lr, 
                               momentum=training_cfg.get('momentum', 0.9), 
                               weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt}")


def get_scheduler(optimizer, training_cfg, total_steps):
    sched = training_cfg.get('scheduler', 'cosine_restart').lower()
    if sched == 'cosine_restart':
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=training_cfg.get('T_0', 10),
            T_mult=training_cfg.get('T_mult', 1),
            eta_min=training_cfg.get('eta_min', 1e-6)
        )
    elif sched == 'onecycle':
        cycle_momentum = True
        if training_cfg.get('optimizer').lower() == 'adamw':
            cycle_momentum = False
        return OneCycleLR(
            optimizer,
            max_lr=training_cfg.get('max_lr', training_cfg.get('base_lr', 1e-4)),
            total_steps=total_steps,
            pct_start=training_cfg.get('pct_start', 0.3),
            anneal_strategy=training_cfg.get('anneal_strategy', 'cos'),
            div_factor=training_cfg.get('div_factor', 10),
            final_div_factor=training_cfg.get('final_div_factor', 100),
            cycle_momentum=cycle_momentum
        )
    elif sched == 'plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=training_cfg.get('plateau_factor', 0.5),
            patience=training_cfg.get('plateau_patience', 5),
            min_lr=training_cfg.get('min_lr', 1e-6)
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched}")