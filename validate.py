# from model.segModel import segModel
# import torch
# import os
# import matplotlib.pyplot as plt
# from ruamel.yaml import YAML
# import random
# import numpy as np
# from util.datasets import load_data

# imagenet_mean = [0.485, 0.456, 0.406]
# imagenet_std  = [0.229, 0.224, 0.225]

# def visualize_batch(images, masks, preds, idx=0):
#     """
#     images: torch.Tensor [B,3,H,W]（已 Normalize）
#     masks:  torch.Tensor [B,1,H,W]（0/1）
#     preds:  torch.Tensor [B,1,H,W]（0–1 概率）
#     idx:    batch 中想画的样本下标
#     """
#     img = images[idx].cpu().permute(1,2,0).numpy()  # H,W,3
#     # 反归一化回 [0,1]
#     img = imagenet_std * img + imagenet_mean
#     img = np.clip(img, 0, 1)

#     gt  = masks[idx,0].cpu().numpy()    # H,W
#     pr  = preds[idx,0].cpu().numpy()    # H,W

#     fig, axes = plt.subplots(1,3, figsize=(12,4))
#     axes[0].imshow(img)
#     axes[0].set_title("Image")
#     axes[0].axis("off")

#     axes[1].imshow(gt, cmap="gray")
#     axes[1].set_title("GT Mask")
#     axes[1].axis("off")

#     axes[2].imshow(pr, cmap="gray")
#     axes[2].set_title("Pred Mask")
#     axes[2].axis("off")

#     plt.tight_layout()
#     plt.show()


# yaml = YAML()
# with open('config/config_01.yaml', 'r') as f:
#     config = yaml.load(f)

# device = config['model'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
# print(f'Currently using "{device}" device.')

# model = segModel(modeName=config["model"]["mode_name"],
#                 use_lora=config["model"]["use_lora"],
#                 lora_r=config["model"]["lora_r"], 
#                 lora_alpha=config["model"]["lora_alpha"],
#                 lora_dropout=config["model"]["lora_dropout"],
#                 device=device, 
#                 dims=config["model"]["dims"], 
#                 num_heads=config["model"]["num_heads"], 
#                 num_classes=config["model"]["num_classes"], 
#                 out_size=config["dataset"]["size"])

# test_loader = load_data(config["dataset"]["root_path"], 
#                         transforms = None,
#                         image_size=config["dataset"]["size"], 
#                         device = device, 
#                         batch_size = config["dataset"]["batch_size"], 
#                         train = False, 
#                         shuffle = False,
#                         drop_last = True)

# ckpt_path = 'checkpoints/20250522_175615/best_model.pth'
# checkpoint = torch.load(ckpt_path, map_location=device)
# model.load_state_dict(checkpoint['model'])
# model.to(device)
# print(f'Loaded checkpoint from {ckpt_path}')
# model.eval()

# with torch.no_grad():
#     for images, gt_masks in test_loader:
#         images = images.to(device)
#         gt_masks  = gt_masks.to(device)

#         pre_logits, pre_masks = model(images)
#         visualize_batch(images, gt_masks, pre_masks, idx=2)
#         break


# ====== SAVE ALL PIC ======
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ruamel.yaml import YAML
from model.img_model import ImgModel
from util.datasets import load_data

# ------------------- 配置读取 -------------------
yaml = YAML()
with open('config/config_01.yaml', 'r') as f:
    config = yaml.load(f)

device = config['model'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 输出目录，以及控制要保存多少样本
output_dir   = config.get('output_dir', 'outputs/test_vis')
num_to_save  = config.get('num_to_save', 10)
mask_thr     = config.get('mask_threshold', 0.9)

os.makedirs(output_dir, exist_ok=True)
print(f'Will save up to {num_to_save} composite images into {output_dir}')

# ImageNet 反归一化参数
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std  = np.array([0.229, 0.224, 0.225])

# ------------------- 模型加载 -------------------
model = ImgModel(device=device)

ckpt_path = 'checkpoints/20250609_145434/best_model.pth'
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model'])
model.to(device).eval()
print(f'Loaded checkpoint from {ckpt_path}')

# ------------------- 数据加载 -------------------
test_loader = load_data(config["dataset"]["root_path"], 
                        transforms = None,
                        image_size=config["dataset"]["size"], 
                        batch_size = 1, 
                        train = False, 
                        shuffle = False,
                        drop_last = True,
                        num_workers=1)

# ------------------- 推理 & 拼图保存 -------------------
saved_count = 0
with torch.no_grad():
    for images, gt_masks in test_loader:
        images   = images.to(device)            # [B,3,H,W]
        gt_masks = gt_masks.to(device)          # [B,1,H,W]
        logits = model(images)           # [B,1,H,W] logits & probs
        
        B = images.size(0)
        for i in range(B):
            if saved_count >= num_to_save:
                break

            # —— 1. 原图反归一化到 [0,1] —— #
            img_np = images[i].cpu().permute(1,2,0).numpy()
            img_np = imagenet_std * img_np + imagenet_mean
            img_np = np.clip(img_np, 0, 1)

            # —— 2. GT mask —— #
            gt_np = gt_masks[i,0].cpu().numpy()

            # —— 3. 预测 mask（阈值二值化） —— #
            prob_np = torch.sigmoid(logits[i,0]).cpu().numpy()
            pred_np = (prob_np > mask_thr).astype(np.float32)

            # —— 4. 拼接三图并保存 —— #
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(img_np)
            axes[0].set_title("Image")
            axes[0].axis("off")

            axes[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("GT Mask")
            axes[1].axis("off")

            axes[2].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("Pred Mask")
            axes[2].axis("off")

            plt.tight_layout()
            save_path = os.path.join(output_dir, f"{saved_count:03d}.png")
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            saved_count += 1

        if saved_count >= num_to_save:
            break

print("Done. Composite images saved.") 

