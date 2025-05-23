import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.init as init
from tqdm.auto import tqdm
from ruamel.yaml import YAML
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from util.losses import CombinedLoss, dice_score, iou_score

def init_weights(module):
    # 卷积层：Kaiming 正态
    if isinstance(module, nn.Conv2d):
        init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            init.zeros_(module.bias)

    # 线性层：Xavier 均匀
    elif isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)

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

    # TensorBoard writer
    tb_log_dir = os.path.join(run_dir, 'tb_logs')
    writer = SummaryWriter(log_dir=tb_log_dir)

    # save config
    yaml = YAML()
    with open(os.path.join(run_dir, 'config_used.yaml'), 'w') as f:
        yaml.dump(config, f)

    # 配置logging，日志输出到文件和控制台
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # 文件日志处理器
    file_handler = logging.FileHandler(os.path.join(logging_dir, 'train.log'))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台日志处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Loss
    criterion = CombinedLoss(alpha_focal=config["training"]["alpha_focal"],
                             alpha_tversky=config["training"]["alpha_tversky"],
                             alpha_boundary=config["training"]["alpha_boundary"],
                             alpha_hnm=config["training"]["alpha_hnm"])

    # 判断 Scheduler 类型 => step() different
    is_plateau = isinstance(lr_scheduler, ReduceLROnPlateau)

    # Checkpoint 状态变量
    start_epoch = 0
    best_val_loss = float('inf')
    global_step = 0
    check_ckpt = None
    if config["model"].get("check_ckpt") is not None:
        check_ckpt = os.path.join(base_output, config["model"].get("check_ckpt"))

    # 恢复训练
    if resume and check_ckpt is not None and os.path.exists(check_ckpt):
        logger.info(f"[RESUME] Loading checkpoint from {check_ckpt}")
        ckpt = torch.load(check_ckpt, map_location=config['model'].get('device', 'cuda'))
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0) + 1
        global_step = ckpt.get('global_step', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
    else:
        model.apply(init_weights)
        logger.info('[INIT] Starting from scratch')
        logger.info('[INIT] INIT WEIGHTS')

    device = config['model'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # EarlyStopping 配置
    patience = config['training'].get('early_stopping_patience', 4)
    min_epochs_before_stop = config['training'].get('min_epochs_before_stop', 30)
    save_every = config['training'].get('save_every_n_epochs', 50)
    log_every = config['training'].get('log_every_n_steps', 10)
    val_every = config['training'].get('val_every_n_epochs', 20)
    num_epochs = config['training'].get('num_epochs', 100)
    patience_counter = 0

    if start_epoch == num_epochs:
        raise ValueError("[RESUME] ERROR: num_epochs == start_epoch")

    total_start_time = time.time()

    scaler = GradScaler()

    try:
        for epoch in range(start_epoch, num_epochs):
            epoch_loss = 0.0
            epoch_steps = 0

            progress = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}")
            model.train()
            for batch_idx, batch in enumerate(train_dataloader):
                inputs, targets = batch # (images, masks) = batch
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                with autocast():
                    mask_logits, mask_prob = model(inputs)
                    loss = criterion(mask_logits, targets)

                # backward
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # scheduler step（not plateau）
                if not is_plateau:
                    lr_scheduler.step()

                global_step += 1
                epoch_loss += loss.item()
                epoch_steps += 1
                progress.update(1)

                # current_lr = lr_scheduler.get_last_lr()[0] if not is_plateau else optimizer.param_groups[0]['lr']
                current_lr = lr_scheduler.get_last_lr()[0]
                progress.set_postfix({
                    'global_step': global_step, 
                    'loss': f"{loss.item():.4f}", 
                    'lr': f"{current_lr:.8f}"})

                if (global_step % log_every) == 0:
                    # logger.info(f"[LOGGING] step={global_step}, loss={loss.item():.4f}")
                    writer.add_scalar('train/loss', loss.item(), global_step)
                    writer.add_scalar('train/lr', current_lr, global_step)

            progress.close()
            writer.add_scalar('train/epoch_loss', epoch_loss / max(1, epoch_steps), epoch)

            # 验证
            if (epoch+1) % val_every == 0:
                model.eval()
                val_loss = 0.0
                dice_sum = 0.0
                iou_sum = 0.0
                tot = 0
                with torch.no_grad():
                    for b in val_dataloader:
                        inp, tgt = b
                        inp = inp.to(device)
                        tgt = tgt.to(device)
                        
                        mask_logits, mask_prob = model(inp)
                        l = criterion(mask_logits, tgt)
                        val_loss += l.item() * len(tgt)

                        dice_sum += dice_score(mask_prob, tgt)
                        iou_sum += iou_score(mask_prob, tgt)

                        tot += len(tgt)
                val_loss /= tot
                val_dice = dice_sum / tot
                val_iou = iou_sum / tot

                current_lr = lr_scheduler.get_last_lr()[0]
                logger.info(f"[VALIDATION] epoch={epoch}, val_loss={val_loss:.4f}")
                logger.info(f"[VALIDATION] val_loss={val_loss:.4f}, Dice={val_dice:.4f}, IoU={val_iou:.4f}")
                writer.add_scalar('val/loss', val_loss, epoch)
                writer.add_scalar('val/lr', current_lr, epoch)

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
                    logger.info(f"[Checkpoint] New best model saved at epoch {epoch} with val_loss {val_loss:.4f}")
                else:
                    patience_counter += 1
                    logger.info(f"[EarlyStopping] Epoch {epoch}: val_loss = {val_loss:.4f} | best = {best_val_loss:.4f} | patience = {patience_counter}/{patience}")
                torch.cuda.empty_cache()

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
                
            if epoch >= min_epochs_before_stop and patience_counter >= patience:
                logger.info(f"[EarlyStopping] Stopped at epoch {epoch}")
                torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'global_step': global_step,
                        'best_val_loss': best_val_loss
                    }, os.path.join(run_dir, 'early_stop_model.pth'))
                break
        torch.cuda.empty_cache()

    finally:
        writer.close()
    total_end_time = time.time()
    logger.info(f"Total training time: {total_end_time - total_start_time:.2f}s")