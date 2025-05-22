from model.segModel import segModel
import torch
import os
import matplotlib.pyplot as plt
from ruamel.yaml import YAML
import random
import numpy as np
from util.datasets import load_data

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

def visualize_batch(images, masks, preds, idx=0):
    """
    images: torch.Tensor [B,3,H,W]（已 Normalize）
    masks:  torch.Tensor [B,1,H,W]（0/1）
    preds:  torch.Tensor [B,1,H,W]（0–1 概率）
    idx:    batch 中想画的样本下标
    """
    img = images[idx].cpu().permute(1,2,0).numpy()  # H,W,3
    # 反归一化回 [0,1]
    img = imagenet_std * img + imagenet_mean
    img = np.clip(img, 0, 1)

    gt  = masks[idx,0].cpu().numpy()    # H,W
    pr  = preds[idx,0].cpu().numpy()    # H,W

    fig, axes = plt.subplots(1,3, figsize=(12,4))
    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(gt, cmap="gray")
    axes[1].set_title("GT Mask")
    axes[1].axis("off")

    axes[2].imshow(pr, cmap="gray")
    axes[2].set_title("Pred Mask")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


yaml = YAML()
with open('config/config_01.yaml', 'r') as f:
    config = yaml.load(f)

device = config['model'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
print(f'Currently using "{device}" device.')

model = segModel(modeName='base', 
                device='cuda', 
                dims=config["model"]["dims"], 
                num_heads=config["model"]["num_heads"], 
                num_classes=config["model"]["num_classes"], 
                out_size=config["dataset"]["size"])

test_loader = load_data(config["dataset"]["root_path"], 
                        transforms = None,
                        image_size=config["dataset"]["size"], 
                        device = device, 
                        batch_size = config["dataset"]["batch_size"], 
                        train = False, 
                        shuffle = False,
                        drop_last = True)

ckpt_path = 'checkpoints/20250521_195037/epoch_50.pth'
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint['model'])
print(f'Loaded checkpoint from {ckpt_path}')
model.eval()

with torch.no_grad():
    for images, gt_masks in test_loader:
        images = images.to(device)
        gt_masks  = gt_masks.to(device)

        pre_logits, pre_masks = model(images)
        visualize_batch(images, gt_masks, pre_masks, idx=1)
        break

