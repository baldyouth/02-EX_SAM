import os
import torch
from torch.utils.data import Dataset, DataLoader

from util.train_fun import train_loop
from util.optimizers import get_optimizer, get_scheduler
import segmentation_models_pytorch as smp

def build_model(model_cfg):
    """
    根据配置返回分割模型。
    model_cfg: dict, 包含'name', 'pretrained', 'in_channels', 'classes' 等键
    """
    name = model_cfg.get('name', 'unet')
    pretrained = model_cfg.get('pretrained', True)
    in_channels = model_cfg.get('in_channels', 3)
    classes = model_cfg.get('classes', 1)

    if name.lower() == 'unet':
        return smp.Unet(
            encoder_name='resnet34',
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=classes
        )
    elif name.lower() == 'fpn':
        return smp.FPN(
            encoder_name='resnet34',
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=classes
        )
    else:
        raise ValueError(f"Unsupported model name: {name}")

class RandomSegDataset(Dataset):
    def __init__(self, num_samples=20, img_size=(3, 64, 64)):
        self.num = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img = torch.randn(self.img_size)
        mask = (torch.rand(1, self.img_size[1], self.img_size[2]) > 0.5).float()
        return img, mask

def main():
    output_dir = os.path.join(os.getcwd(), 'test_checkpoints')
    os.makedirs(output_dir, exist_ok=True)

    config = {
        "model": {
            "output_dir": 'test_checkpoints',
            "name": "unet",
            "pretrained": False,
            "device": "cuda",
            "in_channels": 3,
            "classes": 1,
            "resume": True,
            "check_ckpt": '20250519_144230/last_model.pth'
        },
        "training": {
            "batch_size": 4,
            "num_epochs": 30,
            "mixed_precision": None,
            "gradient_accumulation_steps": 1,
            "early_stopping_patience": 11,
            "save_every_n_epochs": 3,
            "log_every_n_steps": 2,
            "loss_alpha": 0.5,
            "optimizer": "adamw",
            "base_lr": 1e-3,
            "weight_decay": 0.0,
            "momentum": 0.9,
            "scheduler": "plateau",
            "plateau_factor": 0.5,
            "plateau_patience": 1,
            "min_lr": 1e-6,
        },
        "data": {
            "batch_size": 4,
            "num_workers": 0,
        }
    }

    device = torch.device(config["model"]["device"])
    print(f"✅ USE {device}")
    model = build_model(config["model"]).to(device)

    train_ds = RandomSegDataset(num_samples=16)
    val_ds   = RandomSegDataset(num_samples=8)
    train_loader = DataLoader(train_ds,
                              batch_size=config["training"]["batch_size"],
                              shuffle=True,
                              num_workers=config["data"]["num_workers"])
    val_loader   = DataLoader(val_ds,
                              batch_size=config["training"]["batch_size"],
                              shuffle=False,
                              num_workers=config["data"]["num_workers"])

    optimizer    = get_optimizer(model, config["training"])
    total_steps  = len(train_loader) * config["training"]["num_epochs"]
    lr_scheduler = get_scheduler(optimizer, config["training"], total_steps)

    train_loop(config, model, optimizer, train_loader, val_loader, lr_scheduler, resume=config["model"]["resume"])

    print(f"✅ 检查点已保存至：{output_dir}")

if __name__ == "__main__":
    main()