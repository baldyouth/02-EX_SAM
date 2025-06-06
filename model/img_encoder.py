from model.SS2D import SS2D
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from mmpretrain import get_model

from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm

from natten import NeighborhoodAttention2D, use_fused_na

use_fused_na()

#!!! CrossAttention
class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.2):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_feat, kv_feat):
        B, C, H, W = q_feat.shape
        q = self.to_q(q_feat.flatten(2).transpose(1, 2))  # (B, HW, C)
        k = self.to_k(kv_feat.flatten(2).transpose(1, 2))
        v = self.to_v(kv_feat.flatten(2).transpose(1, 2))

        q, k, v = map(lambda x: x.view(B, -1, self.heads, C // self.heads).transpose(1, 2), (q, k, v))
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.out(out)
        return out.transpose(1, 2).reshape(B, C, H, W)

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

#!!! MambaBlock
class MambaBlock(nn.Module):
    def __init__(self, dim, depth=8):
        super().__init__()
        self.mambalayers = nn.ModuleList([
            nn.Sequential(
                Mamba(d_model=dim),
                RMSNorm(dim)
            ) for _ in range(depth)
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1) # B, H*W, C

        for mamba_layer in self.mambalayers:
            x = mamba_layer(x)
        x = self.rmsnorm(x)

        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        return x

#!!! AttentionBlock
class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=3, depth=2):
        super().__init__()
        self.attentionlayers = nn.ModuleList([
            NeighborhoodAttention2D(dim=dim, num_heads=num_heads, kernel_size=kernel_size) for _ in range(depth)
        ])
        self.lnnorm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1) # BHWC

        for attn_layer in self.attentionlayers:
            x = attn_layer(x)
        
        x = self.lnnorm(x)
        x = x.permute(0, 3, 1, 2) #BCHW
        return x

#!!! FPN
class FPN(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        self.layer1 = nn.Sequential(
            ConvBNGELU(in_channels=in_channels, out_channels=base_channels, kernel_size=3, stride=2, padding=1),
            ConvBNGELU(in_channels=base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            ConvBNGELU(in_channels=base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1)
        ) # B, 3, H, W => B, 32, H/2, W/2
        self.layer2 = nn.Sequential(
            ConvBNGELU(in_channels=base_channels, out_channels=base_channels*2, kernel_size=3, stride=2, padding=1),
            ConvBNGELU(in_channels=base_channels*2, out_channels=base_channels*2, kernel_size=3, stride=1, padding=1),
            ConvBNGELU(in_channels=base_channels*2, out_channels=base_channels*2, kernel_size=3, stride=1, padding=1)
        ) # B, 32, H/2, W/2 => B, 64, H/4, W/4
        self.layer3 = nn.Sequential(
            ConvBNGELU(in_channels=base_channels*2, out_channels=base_channels*4, kernel_size=3, stride=2, padding=1),
            ConvBNGELU(in_channels=base_channels*4, out_channels=base_channels*4, kernel_size=3, stride=1, padding=1),
            ConvBNGELU(in_channels=base_channels*4, out_channels=base_channels*4, kernel_size=3, stride=1, padding=1)
        ) # B, 64, H/4, W/4 => B, 128, H/8, W/8
        # self.layer4 = nn.Sequential(
        #     ConvBNGELU(in_channels=base_channels*4, out_channels=base_channels*8, kernel_size=3, stride=2, padding=1),
        #     ConvBNGELU(in_channels=base_channels*8, out_channels=base_channels*8, kernel_size=3, stride=1, padding=1),
        #     ConvBNGELU(in_channels=base_channels*8, out_channels=base_channels*8, kernel_size=3, stride=1, padding=1)
        # ) # B, 128, H/8, W/8 => B, 256, H/16, W/16

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        # x4 = self.layer4(x3)

        return x1, x2, x3

# !!! SAMGuidedCrossAttention
class SAMGuidedCrossAttention(nn.Module):
    def __init__(self, dim, heads=4, window_size=8, dropout=0.2):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.window_size = window_size

        self.to_q = nn.Conv2d(dim, dim, 1)
        self.to_k = nn.Conv2d(dim, dim, 1)
        self.to_v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_feat, kv_feat):
        B, C, H, W = q_feat.shape
        ws = self.window_size
        assert H % ws == 0 and W % ws == 0, "Input must be divisible by window size"

        # Linear proj
        q = self.to_q(q_feat)
        k = self.to_k(kv_feat)
        v = self.to_v(kv_feat)

        # [B, heads, C//heads, H, W]
        def reshape_to_windows(x):
            x = x.view(B, self.heads, C // self.heads, H, W)
            x = x.unfold(3, ws, ws).unfold(4, ws, ws)  # [B, h, c, h/ws, w/ws, ws, ws]
            x = x.permute(0, 3, 4, 1, 5, 6, 2).contiguous()  # [B, h/ws, w/ws, heads, ws, ws, c']
            return x.view(-1, self.heads, ws * ws, C // self.heads)  # [B*num_windows, heads, win_len, c']

        q_w = reshape_to_windows(q)
        k_w = reshape_to_windows(k)
        v_w = reshape_to_windows(v)

        attn = (q_w @ k_w.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out_w = (attn @ v_w)  # [B*num_win, heads, win_len, c']

        # reshape back
        out_w = out_w.view(B, H // ws, W // ws, self.heads, ws, ws, C // self.heads)
        out_w = out_w.permute(0, 3, 6, 1, 4, 2, 5).contiguous()  # [B, heads, c', h/ws, ws, w/ws, ws]
        out_w = out_w.view(B, C, H, W)

        return self.proj(out_w)

# !!! 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return torch.sigmoid(out)

# !!! 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return torch.sigmoid(out)

# !!! MultiLevelDeconvFusion
class MultiLevelDeconvFusion(nn.Module):
    def __init__(self, 
                 sam_channels=256,
                 fpn_channels_list=[32,64,128],
                 reduction=4,
                 use_bilinear=False):# 是否使用双线性插值替代反卷积
        super().__init__()
        
        self.fpn_levels = len(fpn_channels_list)
        self.use_bilinear = use_bilinear
        
        self.deconv_modules = nn.ModuleList()
        self.channel_attn_modules = nn.ModuleList()
        self.spatial_attn_modules = nn.ModuleList()
        self.fusion_conv_modules = nn.ModuleList()
        
        for i in range(self.fpn_levels):
            # 计算需要的反卷积次数和步长
            scale_factor = 2 ** (self.fpn_levels - i - 1) # [8, 4, 2, 1]
            
            if use_bilinear: # 使用双线性插值+卷积
                deconv_module = nn.Sequential(
                    nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
                    nn.Conv2d(sam_channels, fpn_channels_list[i], kernel_size=1),
                    nn.BatchNorm2d(fpn_channels_list[i]),
                    nn.ReLU(inplace=True)
                )
            else: # 使用多级反卷积
                deconv_layers = []
                current_channels = sam_channels
                
                # 逐级反卷积
                while scale_factor > 1:
                    # 每次反卷积缩小一半
                    deconv_layers.append(nn.ConvTranspose2d(
                        current_channels, current_channels//2, 
                        kernel_size=4, stride=2, padding=1
                    ))
                    deconv_layers.append(nn.BatchNorm2d(current_channels//2))
                    deconv_layers.append(nn.GELU())
                    current_channels = current_channels // 2
                    scale_factor = scale_factor // 2
                
                # 调整最终通道数与FPN层匹配
                deconv_layers.append(nn.Conv2d(current_channels, fpn_channels_list[i], kernel_size=1))
                deconv_layers.append(nn.BatchNorm2d(fpn_channels_list[i]))
                deconv_layers.append(nn.GELU())
                
                deconv_module = nn.Sequential(*deconv_layers)
            
            # 通道注意力模块
            channel_attn = ChannelAttention(fpn_channels_list[i]*2, reduction)
            
            # 空间注意力模块
            spatial_attn = SpatialAttention()
            
            # 融合卷积
            # fusion_conv = nn.Sequential(
            #     nn.Conv2d(fpn_channels_list[i]*2, fpn_channels_list[i], kernel_size=3, padding=1),
            #     nn.BatchNorm2d(fpn_channels_list[i]),
            #     nn.ReLU(inplace=True)
            # )
            # fusion_conv = ConvBNGELU(in_channels=fpn_channels_list[i]*2, out_channels=fpn_channels_list[i], kernel_size=3, stride=1, padding=1)
            fusion_conv = nn.Sequential(
                SS2D(fpn_channels_list[i]*2, depth=2, fusion_method='attention', diag_mode='none'),
                ConvBNGELU(in_channels=fpn_channels_list[i]*2, out_channels=fpn_channels_list[i], kernel_size=3, stride=1, padding=1)
            )
            
            self.deconv_modules.append(deconv_module)
            self.channel_attn_modules.append(channel_attn)
            self.spatial_attn_modules.append(spatial_attn)
            self.fusion_conv_modules.append(fusion_conv)
    
    def forward(self, sam_features, fpn_features):
        fused_features = []
        
        for i in range(self.fpn_levels):
            # 1. 多级反卷积上采样SAM特征
            upsampled_sam = self.deconv_modules[i](sam_features[i])
            
            # 2. 特征拼接
            concat_feature = torch.cat([fpn_features[i], upsampled_sam], dim=1)
            
            # 3. 应用通道注意力
            channel_weight = self.channel_attn_modules[i](concat_feature)
            channel_enhanced = concat_feature * channel_weight
            
            # 4. 应用空间注意力
            spatial_weight = self.spatial_attn_modules[i](channel_enhanced)
            spatial_enhanced = channel_enhanced * spatial_weight
            
            # 5. 最终融合卷积
            fused = self.fusion_conv_modules[i](spatial_enhanced)
            
            fused_features.append(fused)
        
        return fused_features

# !!! LoRALinear
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=4.0, dropout=0.1):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        # self.lora_B = nn.Parameter(torch.zeros((out_features, r)))

        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_A.data *= 1e-4  # 缩放权重
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.bias = None  # 默认无 bias

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        lora_out = F.linear(self.dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B) * self.scaling
        return out + lora_out

# !!! --- 注入 LoRA 到 SAM ---
def apply_lora_to_sam(sam_model, r=4, alpha=4.0):
    for blk in sam_model.backbone.layers:
        if hasattr(blk.attn, 'qkv') and isinstance(blk.attn.qkv, nn.Linear):
            old_qkv = blk.attn.qkv
            new_qkv = LoRALinear(old_qkv.in_features, old_qkv.out_features, r=r, alpha=alpha)
            new_qkv.weight.data.copy_(old_qkv.weight.data)
            blk.attn.qkv = new_qkv

    for name, param in sam_model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

#!!! ImgEncoder
class ImgEncoder(nn.Module):
    def __init__(self, device='cuda', base_channels=64):
        super().__init__()
        self.sam = get_model(
            'vit-large-p16_sam-pre_3rdparty_sa1b-1024px',
            backbone=dict(patch_size=8, out_indices=11, out_channels=256), # out_indices=(2, 5, 8, 11)
            pretrained=True, 
            device=device)

        self.fpn = FPN(base_channels=base_channels)
        self.ch_atten = ChannelAttention(in_channels=256, reduction=16)
        self.sp_atten = SpatialAttention()
        self.ss2d = SS2D(channels=256, depth=4, fusion_method='average', use_residual=True, diag_mode='none')

        # 注入 LoRA
        apply_lora_to_sam(self.sam, r=4, alpha=4.0)

        # self.sam.eval()
        # for p in self.sam.parameters():
        #     p.requires_grad = False

    def forward(self, x):
        
        SAM_f = self.sam(x)

        FPN_f = self.fpn(x)
        # FUSION_f = torch.cat([SAM_f[-1], FPN_f[-1]], dim=1)
        FUSION_f = SAM_f[-1]+FPN_f[-1]

        ch_w = self.ch_atten(FUSION_f)
        FUSION_f = ch_w * FUSION_f

        sp_w = self.sp_atten(FUSION_f)
        FUSION_f = sp_w * FUSION_f

        FUSION_f = self.ss2d(FUSION_f)

        return *SAM_f, *FPN_f[0:2], FUSION_f

if __name__ == '__main__':
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.rand((1, 3, 448, 448), device=device)
    model = ImgEncoder().to(device)
    y = model(x)

    print()

    for i in y:
        print(i.shape)