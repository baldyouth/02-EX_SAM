# from mmpretrain import get_model
# from peft import get_peft_model, LoraConfig, TaskType

# def load_SAM_model(modeName='base', pretrain=True, device='cpu', use_lora=True, lora_r=4, lora_alpha=1.0, lora_dropout=0.0):
#     match modeName.strip().lower():
#         case 'base':
#             SAM = get_model('vit-base-p16_sam-pre_3rdparty_sa1b-1024px', pretrained=pretrain, device=device)
#         case 'large':
#             SAM = get_model('vit-large-p16_sam-pre_3rdparty_sa1b-1024px', pretrained=pretrain, device=device)
#         case 'huge':
#             SAM = get_model('vit-huge-p16_sam-pre_3rdparty_sa1b-1024px', pretrained=pretrain, device=device)
#         case _:
#             raise ValueError(f'Unsupported modeName {modeName}')
        
#     lora_config = LoraConfig(
#         task_type=TaskType.FEATURE_EXTRACTION,
#         r=lora_r,
#         lora_alpha=lora_alpha,
#         target_modules = ["attn.qkv", "attn.proj", "ffn.layers.0.0", "ffn.layers.1"], 
#         lora_dropout=lora_dropout,
#         bias="none",
#         modules_to_save=None,
#     )

#     SAM = get_peft_model(SAM, lora_config)

#     for name, param in SAM.named_parameters():
#         if "lora_" not in name:
#             param.requires_grad = False
#         else:
#             param.requires_grad = True

#     return SAM

import torch
import torch.nn as nn
import math
from sam import ImageEncoderViT
from mmpretrain import get_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import numpy as np
import torch.nn.functional as F

class LoRALinear(nn.Linear):
    def __init__(self, in_features, out_features, r=4, lora_alpha=1.0, lora_dropout=0.0, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
            self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
            
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        else:
            self.lora_A = None
            self.lora_B = None

    def forward(self, x):
        result = super().forward(x)
        if self.r > 0:
            lora_out = self.lora_dropout(x) @ self.lora_A.T
            lora_out = lora_out @ self.lora_B.T
            result = result + lora_out * self.scaling
        return result


def inject_lora_to_vitsam(model, target_keywords=('qkv', 'proj', 'ffn.layers.0.0', 'ffn.layers.1'), r=4, lora_alpha=1.0, lora_dropout=0.0):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(k in name for k in target_keywords):
            parent = model
            parts = name.split('.')
            for p in parts[:-1]:
                parent = getattr(parent, p)
            last_part = parts[-1]
            old_linear = getattr(parent, last_part)

            lora_linear = LoRALinear(
                in_features=old_linear.in_features,
                out_features=old_linear.out_features,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=old_linear.bias is not None
            )
            
            lora_linear.weight.data = old_linear.weight.data.clone()
            if old_linear.bias is not None:
                lora_linear.bias.data = old_linear.bias.data.clone()

            setattr(parent, last_part, lora_linear)


def load_SAM_model(modeName='base', freeze=True, use_rel_pos=True, pretrain=True, device='cuda', use_lora=True, lora_r=4, lora_alpha=1.0, lora_dropout=0.0):
    
    match modeName.strip().lower():
        case 'base':
            if use_rel_pos:
                SAM = ImageEncoderViT(use_rel_pos=True, window_size=14, global_attn_indexes=[2, 5, 8, 11])
            else:
                SAM = ImageEncoderViT(img_size=448, use_rel_pos=True, window_size=14, global_attn_indexes=[2, 5, 8, 11])
            state_dict = torch.load("/home/swjtu/workspace_01/02-EX_SAM/checkpoints_sam/sam_vit_b_01ec64.pth")
        case 'large':
            SAM = ImageEncoderViT(use_rel_pos=True, window_size=14, embed_dim=1024, depth=24, num_heads=16, global_attn_indexes=[5, 11, 17, 23]) #TODO
            state_dict = torch.load("/home/swjtu/workspace_01/02-EX_SAM/checkpoints_sam/sam_vit_l_0b3195.pth")
        case 'huge':
            SAM = ImageEncoderViT(use_rel_pos=True, window_size=14, embed_dim=1280, depth=32, num_heads=16, global_attn_indexes=[7, 15, 23, 31]) #TODO
            state_dict = torch.load("/home/swjtu/workspace_01/02-EX_SAM/checkpoints_sam/sam_vit_h_4b8939.pth")
        case _:
            raise ValueError(f'Unsupported modeName {modeName}')

    if pretrain:
        filtered_image_dict = {
            k.replace('image_encoder.', '', 1): v
            for k, v in state_dict.items()
            if k.startswith('image_encoder.')
        }

        SAM.load_state_dict(filtered_image_dict, strict=False)

    if freeze:
        for param in SAM.parameters():
            param.requires_grad = False

    # if use_lora:
    #     inject_lora_to_vitsam(
    #         SAM.backbone,
    #         target_keywords=('qkv', 'proj', 'ffn.layers.0.0', 'ffn.layers.1'),
    #         r=lora_r,
    #         lora_alpha=lora_alpha,
    #         lora_dropout=lora_dropout
    #     )
    
    # for name, param in SAM.backbone.named_parameters():
    #     if 'lora_' in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = False

    return SAM



def count_trainable_params(model):
    train_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sum_parameters = sum(p.numel() for p in model.parameters())
    print(f'train_parameters: {train_parameters}, sum_parameters: {sum_parameters}') 

def preprocess_image(image_path, target_size=1024):
    """
    读取并预处理图像，使其符合SAM模型的输入要求
    
    参数:
        image_path: 图像文件路径
        target_size: 调整后的最长边长度（SAM默认1024）
    
    返回:
        image_tensor: 预处理后的图像张量 [B, C, H, W]
        original_image: PIL格式的原始图像
        original_size: 原始图像尺寸 (width, height)
    """
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    
    return image_tensor, original_image, original_size, scale
    
def visualize_all_layers(features, channels=None, title_prefix=""):
    """
    可视化所有层的特征图
    
    参数:
        features: SAM模型输出的特征列表
        channels: 要可视化的通道列表（None表示平均激活）
        title_prefix: 标题前缀
    """
    n_layers = len(features)
    layer_names = ["Layer 2", "Layer 5", "Layer 8", "Layer 11", "Neck"]
    
    if channels is None:
        # 可视化平均激活
        channels = [None] * n_layers
        title_suffix = " (平均激活)"
    else:
        # 确保通道数与层数匹配
        channels = channels[:n_layers] + [None]*(n_layers - len(channels))
        title_suffix = ""
    
    # 创建子图
    fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 5))
    
    for i, (feat, channel) in enumerate(zip(features, channels)):
        feat_np = feat.detach().cpu().numpy()[0]  # [C, H, W]
        
        if channel is None:
            # 平均激活
            vis = np.mean(feat_np, axis=0)
            channel_text = "平均"
        else:
            # 特定通道
            vis = feat_np[channel]
            channel_text = f"通道 {channel}"
        
        # 显示特征图
        im = axes[i].imshow(vis, cmap='viridis')
        axes[i].set_title(f"{title_prefix}{layer_names[i]}\n{channel_text}")
        axes[i].axis('off')
        
        # 添加颜色条
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
    return fig
    

    """
    Visualize precomputed feature maps on CPU.

    Args:
        features (dict): mapping from layer name to feature tensor of shape (batch, channels, height, width) or (channels, height, width), possibly on CUDA.
        layer_names (list of str): list of layer names to plot
        cmap (str): matplotlib colormap
    """
    n = len(layer_names)
    fig, axes = plt.subplots(n, 1, figsize=(6, 3 * n))
    if n == 1:
        axes = [axes]

    for ax, lname in zip(axes, layer_names):
        feat = features.get(lname)
        if feat is None:
            ax.set_title(f"Layer '{lname}' not found in features.")
            ax.axis('off')
            continue
        # move to CPU and convert to numpy
        feat_cpu = feat.detach().cpu()
        # if batch dimension present, take first sample and average channels
        if feat_cpu.ndim == 4:
            fmap = feat_cpu[0].mean(dim=0).numpy()
        elif feat_cpu.ndim == 3:
            fmap = feat_cpu.mean(dim=0).numpy()
        else:
            ax.set_title(f"Layer '{lname}' has unsupported shape {feat_cpu.shape}.")
            ax.axis('off')
            continue
        im = ax.imshow(fmap, cmap=cmap)
        ax.set_title(lname)
        ax.axis('off')
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

def visualize_features(features, layer_names, cmap='viridis'):

    # Prepare mapping from names to tensors
    if isinstance(features, list):
        if len(features) != len(layer_names):
            raise ValueError("Length of features list must match length of layer_names list.")
        feat_map = {name: feat for name, feat in zip(layer_names, features)}
    elif isinstance(features, dict):
        feat_map = features
    else:
        raise TypeError("features must be a dict or a list of tensors.")

    # Precompute fmaps and find global min/max
    fmaps = []
    for lname in layer_names:
        feat = feat_map.get(lname)
        if feat is None:
            fmaps.append(None)
            continue
        feat_cpu = feat.detach().cpu()
        if feat_cpu.ndim == 4:
            fmap = feat_cpu[0].mean(dim=0)
        elif feat_cpu.ndim == 3:
            fmap = feat_cpu.mean(dim=0)
        else:
            fmaps.append(None)
            continue
        fmaps.append(fmap.numpy())
    # Determine global vmin/vmax
    valid_fmaps = [f for f in fmaps if f is not None]
    if not valid_fmaps:
        raise ValueError("No valid feature maps to display.")
    vmin = min(f.min() for f in valid_fmaps)
    vmax = max(f.max() for f in valid_fmaps)

    # Plot
    n = len(layer_names)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]

    for ax, lname, fmap in zip(axes, layer_names, fmaps):
        if fmap is None:
            ax.set_title(f"Layer '{lname}' not found or unsupported shape.")
            ax.axis('off')
            continue
        im = ax.imshow(fmap, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(lname)
        ax.axis('off')
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    layer_names = ["Layer 2", "Layer 5", "Layer 8", "Layer 11", "Neck"]
    img_encoder = load_SAM_model(modeName='base', use_rel_pos=True)

    # for name, param in img_encoder.named_parameters():
    #     print(f"参数: {name}, 可训练: {param.requires_grad}")

    # input01: image
    # image_path = "/home/swjtu/workspace_01/data/crack_segmentation_dataset/train/images/CFD_113.jpg"
    # image_tensor, original_image, original_size, scale = preprocess_image(image_path)

    # input02: test tensor
    test_tensor = torch.rand((1, 3, 448, 448), device='cuda')

    img_encoder.to('cuda')
    ouputs = img_encoder(test_tensor)

    for output in ouputs:
        print(output.shape)

    visualize_all_layers(ouputs)
    visualize_features(ouputs, layer_names)

    