import os
import time
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
import segmentation_models_pytorch as smp

from .losses import CombinedLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_loop(config, model, optimizer, train_dataloader, val_dataloader, lr_scheduler, resume=True):
    # 基础输出目录
    base_output = config['model']['output_dir']
    os.makedirs(base_output, exist_ok=True)
    # 每次训练创建唯一子目录
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output, run_id)
    os.makedirs(run_dir, exist_ok=True)

    logging_dir=os.path.join(run_dir, 'logs')
    os.makedirs(logging_dir, exist_ok=True)

    # Accelerate 项目配置，日志写入子目录/logs
    accelerator_project_config = ProjectConfiguration(
        project_dir = run_dir,
        logging_dir = logging_dir
    )
    accelerator = Accelerator(
        mixed_precision = config['training'].get('mixed_precision', 'fp16'),
        gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', 1),
        log_with = 'tensorboard',
        project_config = accelerator_project_config,
    )
    accelerator.init_trackers('train_example')

    # Loss
    criterion = CombinedLoss(alpha=config['training'].get('loss_alpha', 0.5))

    # 判断 Scheduler 类型 => step() different
    is_plateau = isinstance(lr_scheduler, ReduceLROnPlateau)

    # Checkpoint 状态变量
    start_epoch = 0
    best_val_loss = float('inf')
    global_step = 0
    if config["model"].get("check_ckpt") is not None:
        check_ckpt = os.path.join(base_output, config["model"].get("check_ckpt"))

    # 恢复训练
    if resume and os.path.exists(check_ckpt):
        print(f"[RESUME] Loading checkpoint from {check_ckpt}")
        ckpt = torch.load(check_ckpt, map_location=config['model'].get('device', 'cuda'))
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0) + 1
        global_step = ckpt.get('global_step', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
    else:
        print('[INIT] Starting from scratch')

    # Accelerator 包装
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = \
        accelerator.prepare(model, optimizer, train_dataloader, val_dataloader, lr_scheduler)

    # EarlyStopping 配置
    patience = config['training'].get('early_stopping_patience', 30)
    save_every = config['training'].get('save_every_n_epochs', 50)
    log_every = config['training'].get('log_every_n_steps', 10)
    num_epochs = config['training'].get('num_epochs', 100)
    patience_counter = 0

    if start_epoch == num_epochs:
        accelerator.end_training()
        # print("[RESUME] ERROR: num_epochs == start_epoch")
        raise ValueError("[RESUME] ERROR: num_epochs == start_epoch")

    total_start_time = time.time()

    for epoch in range(start_epoch, num_epochs):
        progress = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}")
        model.train()
        for batch_idx, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                inputs, targets = batch # (images, masks) = batch
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                accelerator.backward(loss)
                optimizer.step()
                # 仅对按批次更新的 scheduler 调用
                if not is_plateau:
                    lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            progress.update(1)
            current_lr = lr_scheduler.get_last_lr()[0] if not is_plateau else optimizer.param_groups[0]['lr']
            progress.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.6f}"})
            if (global_step % log_every) == 0:
                accelerator.log({'train_loss': loss.item(), 'lr': current_lr}, step=global_step)
        progress.close()

        # 验证
        model.eval()
        val_loss = 0.0
        tot = 0
        with torch.no_grad():
            for b in val_dataloader:
                inp, tgt = b
                out = model(inp)
                l = criterion(out, tgt)
                val_loss += l.item() * len(tgt)
                tot += len(tgt)
        val_loss /= tot
        accelerator.log({'val_loss': val_loss}, step=global_step)

        # 基于验证指标的 scheduler 更新
        if is_plateau:
            lr_scheduler.step(val_loss)

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'best_val_loss': best_val_loss
            }, os.path.join(run_dir, 'best_model.pth'))
        else:
            patience_counter += 1
            print(f"[EarlyStopping] No improvement for {patience_counter} epochs")

        # 周期 & last checkpoint
        if (epoch+1) % save_every == 0 or (epoch+1) == num_epochs:
            if (epoch+1) == num_epochs:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step,
                    'best_val_loss': best_val_loss
                }, os.path.join(run_dir, 'last_model.pth'))

            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'best_val_loss': best_val_loss
            }, os.path.join(run_dir, f'epoch_{epoch+1}.pth'))
            
        if patience_counter >= patience:
            print(f"[EarlyStopping] Stopped at epoch {epoch}")
            break
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"Total training time: {total_duration:.2f} seconds.")

    accelerator.end_training()