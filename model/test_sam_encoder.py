from mmpretrain import get_model

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision.utils import make_grid
from sam import sam_model_registry

def visualize_bilinear_two_rows(embeddings, target_size=(32, 32), top_k=6, figsize=(24, 6)):
    if not torch.is_tensor(embeddings):
        embeddings = torch.tensor(embeddings)
    
    # 关键修改：分离计算图
    embeddings = embeddings.detach().squeeze(0)  # [C, H, W]
    
    # 计算通道能量并筛选top_k
    channel_energy = torch.norm(embeddings, p=2, dim=(1, 2))
    top_channels = torch.argsort(channel_energy, descending=True)[:top_k]
    
    # 计算所有通道的平均特征
    avg_original = embeddings.mean(dim=0, keepdim=True)  # [1, H, W]
    avg_interpolated = F.interpolate(
        avg_original.unsqueeze(0),  # [1, 1, H, W]
        size=target_size, 
        mode='bilinear', 
        align_corners=False
    ).squeeze().detach().cpu().numpy()  # [target_H, target_W]
    
    # 创建2行(top_k+2)列的子图布局（新增两列用于显示平均值）
    plt.figure(figsize=figsize)
    
    # 第一行：原始特征
    for i, channel in enumerate(top_channels):
        ax = plt.subplot(2, top_k+2, i + 1)  # 注意列数变为top_k+2
        feat = embeddings[channel].cpu().numpy()
        ax.imshow(feat)
        ax.set_title(f'channel {channel}')
        ax.axis('off')
        if i == 0:
            ax.set_ylabel('ori', fontsize=12)
    
    # 第一行最后一列：所有通道平均原始特征
    ax_avg_original = plt.subplot(2, top_k+2, top_k + 1)
    ax_avg_original.imshow(avg_original.squeeze().cpu().numpy())
    ax_avg_original.set_title(f'all\naverage')
    ax_avg_original.axis('off')
    
    # 第二行：双线性插值特征
    for i, channel in enumerate(top_channels):
        feat = embeddings[channel].unsqueeze(0).unsqueeze(0)
        interpolated = F.interpolate(
            feat, size=target_size, mode='bilinear', align_corners=False
        ).squeeze().detach().cpu().numpy()
        
        ax = plt.subplot(2, top_k+2, i + top_k + 3)  # 注意索引计算
        ax.imshow(interpolated)
        ax.set_title(f'channel {channel}')
        ax.axis('off')
        if i == 0:
            ax.set_ylabel('bilinear', fontsize=12)
    
    # 第二行最后一列：所有通道平均插值特征
    ax_avg_interpolated = plt.subplot(2, top_k+2, 2*(top_k+2) - 1)
    ax_avg_interpolated.imshow(avg_interpolated)
    ax_avg_interpolated.set_title(f'all\naverage inter\n{target_size}')
    ax_avg_interpolated.axis('off')
    
    plt.tight_layout()
    plt.show()

def preprocess_image(image_path, target_size=448, device='cuda'):
    # 1. 读取图像
    original_image = Image.open(image_path).convert("RGB")
    original_size = original_image.size
    print(f"原始图像尺寸: {original_size}")
    
    # 2. 调整图像大小（保持比例，最长边为target_size）
    def resize_longest_side(image, target_length):
        width, height = image.size
        scale = target_length / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        return image.resize(new_size, resample=Image.BICUBIC), scale
    
    image, scale = resize_longest_side(original_image, target_length=target_size)
    print(f"调整后图像尺寸: {image.size}, 缩放比例: {scale:.4f}")
    
    # 3. 转换为张量并归一化
    image_np = np.array(image)
    image_tensor = torch.tensor(image_np).permute(2, 0, 1).float() / 255.0  # HWC -> CHW, 像素值归一化
    
    # 4. 添加batch维度
    image_tensor = image_tensor.unsqueeze(0)  # [1, C, H, W]
    
    # 5. 移至GPU（如果可用）
    
    image_tensor = image_tensor.to(device)
    
    return image_tensor, original_image, original_size, scale



def visualize_features(features, title=None, figsize=(16, 4), cmap='viridis', 
                       num_samples=4, show_stats=True, invert=True):
    if len(features) != 4:
        raise ValueError(f"Expected 4 features, got {len(features)}")
    
    if title is None:
        title = [f"Feature {i+1}" for i in range(4)]
    elif len(title) != 4:
        raise ValueError(f"Title list must have length 4, got {len(title)}")
    
    # Convert features to numpy
    features_np = []
    for feat in features:
        if isinstance(feat, torch.Tensor):
            feat = feat.detach().cpu().numpy()
        features_np.append(feat)

    # Determine batch size and shapes
    B = min([f.shape[0] for f in features_np])
    C_shapes = [f.shape[1] for f in features_np]
    H_shapes = [f.shape[2] for f in features_np]
    W_shapes = [f.shape[3] for f in features_np]
    
    samples = min(B, num_samples)
    fig, axes = plt.subplots(samples, 4, figsize=figsize)

    # Make axes 2D for consistent indexing if samples=1
    if samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, feat in enumerate(features_np):
        if show_stats:
            print(f"\n{title[i]} Features:")
            print(f"  Shape: {feat.shape}")
            print(f"  Channel: {C_shapes[i]}, Size: {H_shapes[i]}x{W_shapes[i]}")
            print(f"  Stats: min={feat.min():.4f}, max={feat.max():.4f}, mean={feat.mean():.4f}")

        for s in range(samples):
            avg_feat = np.mean(feat[s], axis=0)  # [C, H, W] -> [H, W]
            avg_feat_norm = (avg_feat - avg_feat.min()) / (avg_feat.max() - avg_feat.min() + 1e-8)

            if invert:
                avg_feat_norm = 1.0 - avg_feat_norm

            ax = axes[s, i]
            im = ax.imshow(avg_feat_norm, cmap=cmap)
            ax.set_title(f"{title[i]} (Sample {s+1})")
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = "/home/swjtu/workspace_01/data/crack_segmentation_dataset/images/CFD_002.jpg"
    model_type = "vit_b"
    sam_checkpoint = "/home/swjtu/workspace_01/02-EX_SAM/checkpoints_sam/sam_vit_b_01ec64.pth"

    SAM_encoder = get_model(
        'vit-large-p16_sam-pre_3rdparty_sa1b-1024px',
        backbone=dict(out_indices=(2, 5, 8, 11)), 
        pretrained=True, 
        device=device)
    
    print(SAM_encoder)
    # SAM = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    
    image_tensor, original_image, original_size, scale = preprocess_image(image_path, target_size=448, device=device)

    outputs = SAM_encoder(image_tensor)
    for output in outputs:
        print(output.shape)
    visualize_features(outputs, invert=False)
