from model.SS2D import SS2D
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from mmpretrain import get_model

#!!! ConvBNGELU
class ConvBNGELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.block(x)

#!!! 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced_channels = max(1, in_channels // reduction)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        scale = self.sigmoid(out)
        return x * scale

#!!! 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(pooled))
        return x * scale

#!!! CBAM
class CBAMBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, spatial_kernel=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

#!!! FPN
class FPN(nn.Module):
    def __init__(self, FPN_config):
        super().__init__()
        base_channels = FPN_config['base_channels']

        self.layer0 = nn.Sequential(
            ConvBNGELU(in_channels=3, out_channels=base_channels, kernel_size=3, stride=2, padding=1),
            ConvBNGELU(in_channels=base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            ConvBNGELU(in_channels=base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
        )
        self.layer1 = nn.Sequential(
            ConvBNGELU(in_channels=base_channels, out_channels=base_channels*2, kernel_size=3, stride=2, padding=1),
            ConvBNGELU(in_channels=base_channels*2, out_channels=base_channels*2, kernel_size=3, stride=1, padding=1),
            ConvBNGELU(in_channels=base_channels*2, out_channels=base_channels*2, kernel_size=3, stride=1, padding=1),
        )
        self.layer2 = nn.Sequential(
            ConvBNGELU(in_channels=base_channels*2, out_channels=base_channels*4, kernel_size=3, stride=2, padding=1),
            ConvBNGELU(in_channels=base_channels*4, out_channels=base_channels*4, kernel_size=3, stride=1, padding=1),
            ConvBNGELU(in_channels=base_channels*4, out_channels=base_channels*4, kernel_size=3, stride=1, padding=1),
            CBAMBlock(in_channels=base_channels*4, spatial_kernel=3)
        )
        self.layer3 = nn.Sequential(
            ConvBNGELU(in_channels=base_channels*4, out_channels=base_channels*8, kernel_size=3, stride=2, padding=1),
            ConvBNGELU(in_channels=base_channels*8, out_channels=base_channels*8, kernel_size=3, stride=1, padding=1),
            ConvBNGELU(in_channels=base_channels*8, out_channels=base_channels*8, kernel_size=3, stride=1, padding=1),
            CBAMBlock(in_channels=base_channels*8, spatial_kernel=3)
        )

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)

        return x0, x1, x2, x3

#!!! LoRALinear
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=16, alpha=32, dropout=0.1):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.lora_down = nn.Linear(in_features, r, bias=False)
        self.lora_up = nn.Linear(r, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.scaling = alpha / r

    def forward(self, x):
        return self.dropout(self.lora_up(self.lora_down(x))) * self.scaling
    
#!!! QKVLoRAWrapper
class QKVLoRAWrapper(nn.Module):
    def __init__(self, qkv_linear: nn.Linear, r=16, alpha=32, dropout=0.1):
        super().__init__()
        self.qkv = qkv_linear
        self.hidden_dim = qkv_linear.in_features
        self.total_dim = qkv_linear.out_features
        assert self.total_dim == 3 * self.hidden_dim

        self.lora_q = LoRALinear(self.hidden_dim, self.hidden_dim, r, alpha, dropout)
        self.lora_v = LoRALinear(self.hidden_dim, self.hidden_dim, r, alpha, dropout)

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.lora_q(x)
        v = v + self.lora_v(x)
        return torch.cat([q, k, v], dim=-1)

#!!! SAMEncoder
class SAMEncoder(nn.Module):
    def __init__(self, LoRA_config, SAM_config):
        super().__init__()
        self.r = LoRA_config['r']
        self.alpha = LoRA_config['alpha']
        self.dropout = LoRA_config['dropout']

        self.sam = get_model(
            model=SAM_config['name'],
            backbone=dict(
                img_size=SAM_config['img_size'],
                patch_size=SAM_config['patch_size'],
                window_size=SAM_config['window_size'],
                out_indices=SAM_config['out_indices'], 
                out_channels=SAM_config['out_channels']), # out_indices=(2, 5, 8, 11)
            pretrained=SAM_config['pretrained'])
            # device=device
        self._inject_lora()
    
    def _inject_lora(self):
        for i, blk in enumerate(self.sam.backbone.layers):
            old_qkv = blk.attn.qkv
            blk.attn.qkv = QKVLoRAWrapper(old_qkv, self.r, self.alpha, self.dropout)

    def forward(self, x):
        return self.sam(x)
    
    def set_trainable_params(self):
        for name, param in self.sam.named_parameters():
            if 'lora_' in name or "pos_embed" in name or "rel_pos" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

#!!! Cross Attention
class CrossAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cross_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)

    def forward(self, SAM_f, FPN_f):
        B, C, H, W = SAM_f.shape

        SAM_f_flat = SAM_f.flatten(2).permute(0, 2, 1)
        FPN_f_flat = FPN_f.flatten(2).permute(0, 2, 1)

        attn_output, _ = self.cross_attn(
            query = SAM_f_flat,
            key = FPN_f_flat,
            value = FPN_f_flat
        )

        attn_output = attn_output.permute(0, 2, 1).reshape(B, C, H, W)

        Fusion_f = SAM_f + attn_output

        return Fusion_f

#!!! ImgEncoder
class ImgEncoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.sam = SAMEncoder(LoRA_config=model_config['LoRA'], SAM_config=model_config['SAM'])
        self.sam.set_trainable_params()
        
        self.fpn = FPN(FPN_config=model_config['FPN'])

        self.sam_cbma = CBAMBlock(in_channels=256, spatial_kernel=3)
        self.fpn_cbma = CBAMBlock(in_channels=256)

        # self.ss2d = SS2D(SS2D_config=model_config['SS2D'])
        # self.cross_attention = CrossAttention()
    
    def compute_edge(self, x):
        x_gray = x.mean(dim=1, keepdim=True)

        sobel_x = torch.tensor([[[[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]]]], device=x.device, dtype=x.dtype)
        sobel_y = torch.tensor([[[[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]]]], device=x.device, dtype=x.dtype)

        edge_x = F.conv2d(x_gray, sobel_x, padding=1)
        edge_y = F.conv2d(x_gray, sobel_y, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

        edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-6)
        return edge

    def forward(self, x):
        edge = self.compute_edge(x) 
        
        (SAM_f,) = self.sam(x)
        SAM_f = self.sam_cbma(SAM_f)

        FPN_f = self.fpn(x)
        # FPN_f_3 = self.fpn_cbma(FPN_f[-1])

        FUSION_f = SAM_f + FPN_f[-1]
        # FUSION_f = self.cross_attention(SAM_f, FPN_f_3)

        return SAM_f, *FPN_f, FUSION_f, edge

if __name__ == '__main__':
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.rand((1, 3, 448, 448), device=device)
    model = ImgEncoder().to(device)
    for name, param in model.named_parameters():
        if 'sam' in name:
            print(name, param.requires_grad)
    y = model(x)

    print()

    for i in y:
        print(i.shape)