import pytorch_lightning as pl
import torch
import os
from datetime import datetime
from ruamel.yaml import YAML
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from Lightning.lightning_module import LitModule
from Lightning.datamodule import CrackDataModule

torch.set_float32_matmul_precision('medium')

yaml = YAML()
with open('config/config_lightning.yaml', 'r') as f:
    config = yaml.load(f)

#TODO
class IntervalCheckpoint(pl.Callback):
    def __init__(self, save_dir, interval_epochs=50):
        super().__init__()
        self.save_dir = save_dir
        self.interval_epochs = interval_epochs
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.interval_epochs == 0:
            filename = f"epoch_{epoch+1}.ckpt"
            ckpt_path = os.path.join(self.save_dir, filename)
            trainer.save_checkpoint(ckpt_path)
            print(f"[IntervalCheckpoint] Saved checkpoint at epoch {epoch+1} to {ckpt_path}")

class DelayedEarlyStopping(EarlyStopping):
    def __init__(self, min_epochs=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_epochs = min_epochs

    def on_validation_end(self, trainer, pl_module):
        # 当前 epoch 小于 min_epochs，就跳过 EarlyStopping 检查
        if trainer.current_epoch < self.min_epochs:
            return
        super().on_validation_end(trainer, pl_module)

#TODO
checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True, 
        filename='{epoch:03d}-{val_loss:.4f}',
        verbose=True)

def calc_total_training_steps(num_samples, config):
    batch_size = config['data']['batch_size']
    accumulate_grad_batches = config['train']['accumulate_grad_batches']
    max_epochs = config['train']['max_epochs']

    steps_per_epoch = (num_samples + batch_size - 1) // batch_size
    optimizer_steps_per_epoch = (steps_per_epoch + accumulate_grad_batches - 1) // accumulate_grad_batches
    total_steps = optimizer_steps_per_epoch * max_epochs
    return total_steps

#!!! start training
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
unique_dir = f"checkpoints/{timestamp}"
logger = TensorBoardLogger(unique_dir, name="boards")

# 早停
early_stop = DelayedEarlyStopping(
    monitor='val_iou',
    mode='max',
    patience=config['train']['patience'],
    min_epochs=config['train']['min_epochs'],
    verbose=True)

# 模型保存
interval_checkpoint = IntervalCheckpoint(
    save_dir=unique_dir, 
    interval_epochs=config['train']['interval_epochs'])

# Trainer配置
trainer = pl.Trainer(
    logger=logger,
    max_epochs=config['train']['max_epochs'],
    accumulate_grad_batches=config['train']['accumulate_grad_batches'],
    callbacks=[early_stop, interval_checkpoint],
    check_val_every_n_epoch=config['train']['check_val_every_n_epoch'],
    devices=config['train']['devices'],
    accelerator=config['train']['accelerator'],
    precision=config['train']['precision'])

# 模型和数据
data_module = CrackDataModule(data_config=config['data'])

# 如果使用CosineAnnealingLR需要计算总step
if config['scheduler']['name'].lower() == 'cosine':
    data_module.setup()
    num_samples = len(data_module.train_dataset)
    total_steps = calc_total_training_steps(num_samples=num_samples, config=config)
    config['scheduler']['total_steps'] = total_steps

    model = LitModule(optimizer_config=config['optimizer'], scheduler_config=config['scheduler'])
else:
    model = LitModule(optimizer_config=config['optimizer'], scheduler_config=config['scheduler'])

# 训练启动
trainer.fit(model, datamodule=data_module)

