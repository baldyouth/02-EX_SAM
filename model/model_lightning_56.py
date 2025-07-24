import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmpretrain import get_model

from model.SS2D import SS2D

#!!! ConvBNGELU
class ConvBNSiLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

#!!! 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduction_channels = max(1, in_channels // reduction)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduction_channels, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduction_channels, in_channels, 1, bias=False)
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

# MultiScaleFusionModule
class MultiScaleFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels, use_depthwise=False):
        super().__init__()
        padding_3 = 1
        padding_5 = 2
        padding_7 = 3

        self.down_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )

        if use_depthwise:
            Conv = lambda k, p: nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=k, padding=p, groups=out_channels, bias=False),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )
        else:
            Conv = lambda k, p: nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=k, padding=p, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True)
            )

        self.conv3 = Conv(3, padding_3)
        self.conv5 = Conv(5, padding_5)
        self.conv7 = Conv(7, padding_7)

        # 融合
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        x = self.down_conv(x)

        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)

        out = torch.cat([out3, out5, out7], dim=1)
        out = self.fuse(out)

        return out + x

#!!! FPN
class FPN(nn.Module):
    def __init__(self, FPN_config):
        super().__init__()
        base_channels = FPN_config['base_channels']

        self.layer0 = MultiScaleFusionModule(in_channels=3, out_channels=base_channels)
        self.layer1 = MultiScaleFusionModule(in_channels=base_channels, out_channels=base_channels*2)
        self.layer2 = MultiScaleFusionModule(in_channels=base_channels*2, out_channels=base_channels*4)

    def forward(self, x):
        x_fpn_0 = self.layer0(x)
        x_fpn_1 = self.layer1(x_fpn_0)
        x_fpn_2 = self.layer2(x_fpn_1)

        return (x_fpn_0, x_fpn_1, x_fpn_2)

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
                out_channels=SAM_config['out_channels']),
            pretrained=SAM_config['pretrained'])
        self._inject_lora()
    
    def _inject_lora(self):
        for i, blk in enumerate(self.sam.backbone.layers):
            old_qkv = blk.attn.qkv
            blk.attn.qkv = QKVLoRAWrapper(old_qkv, self.r, self.alpha, self.dropout)

    def forward(self, x):
        return self.sam(x)
    
    def set_trainable_params(self):
        for name, param in self.sam.named_parameters():
            if ('lora_' in name or "pos_embed" in name or "rel_pos" in name or 'channel_reduction' in name
                or 'layers.21' in name or 'layers.22' in name or 'layers.23' in name ):
                param.requires_grad = True
                # print(name)
            else:
                param.requires_grad = False

#!!! MultiScaleConv
class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.reduction = nn.Conv2d(in_channels=in_channels*2, out_channels=out_channels, kernel_size=1)
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.conv5x5 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2, stride=1)
        self.fusion = nn.Conv2d(in_channels=in_channels*3, out_channels=out_channels, kernel_size=1)

    def forward(self, feat_sam, feat_fpn):
        x_reduction = self.reduction(torch.cat([feat_sam, feat_fpn], dim=1))
        x_1 = self.conv1x1(x_reduction)
        x_3 = self.conv3x3(x_reduction)
        x_5 = self.conv5x5(x_reduction)
        x_fusion = self.fusion(torch.cat([x_1, x_3, x_5], dim=1))

        return x_fusion

#!!! UPSAM
class UpSAM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.up_block_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNSiLU(in_channels=256, out_channels=128)
        )
        self.up_block_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNSiLU(in_channels=128, out_channels=64)
        )

    def forward(self, x_sam_2):
        x_sam_1 = self.up_block_1(x_sam_2)
        x_sam_0 = self.up_block_2(x_sam_1)

        return (x_sam_0, x_sam_1)

#!!! Fusion
class Fusion(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cbam_2 = CBAMBlock(in_channels=256)

        self.multiscale_0 = MultiScaleConv(in_channels=64, out_channels=64)
        self.multiscale_1 = MultiScaleConv(in_channels=128, out_channels=128)
        self.multiscale_2 = MultiScaleConv(in_channels=256, out_channels=256)
    
    def forward(self, feat_sam_list, feat_fpn_list):
        x_fusion_0 = self.multiscale_0(feat_sam_list[0], feat_fpn_list[0])
        x_fusion_1 = self.multiscale_1(feat_sam_list[1], feat_fpn_list[1])
        x_fusion_2 = self.multiscale_2(self.cbam_2(feat_sam_list[2]), self.cbam_2(feat_fpn_list[2]))

        return (x_fusion_0, x_fusion_1, x_fusion_2)


class ImgDecoder(nn.Module):
    def __init__(self, in_channels=[256, 128, 64], out_channels=1):
        super().__init__()
        
        self.up_block2 = nn.Sequential(
            ConvBNSiLU(in_channels=in_channels[0]*2, out_channels=in_channels[0]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBNSiLU(in_channels=in_channels[0], out_channels=in_channels[1]),
        )
        self.up_block1 = nn.Sequential(
            ConvBNSiLU(in_channels=in_channels[1]*2, out_channels=in_channels[1]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBNSiLU(in_channels=in_channels[1], out_channels=in_channels[2])
        )
        self.up_block0 = nn.Sequential(
            ConvBNSiLU(in_channels=in_channels[2]*2, out_channels=in_channels[2]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBNSiLU(in_channels=in_channels[2], out_channels=out_channels)
        )
        self.edge_block = nn.Sequential(
            ConvBNSiLU(in_channels=out_channels*2, out_channels=out_channels),
            ConvBNSiLU(in_channels=out_channels, out_channels=out_channels)
        )

    def forward(self, x_de, x_fpn_list):
        x_de_1 = self.up_block2(torch.cat([x_fpn_list[2], x_de], dim=1))
        x_de_0 = self.up_block1(torch.cat([x_fpn_list[1], x_de_1], dim=1))
        logits = self.up_block0(torch.cat([x_fpn_list[0], x_de_0], dim=1))
        # prob = torch.sigmoid(logits)

        return logits

def compute_edge(x):
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

#!!! Model_Lightning
class Model_Lightning(nn.Module):
    def __init__(self, model_config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sam = SAMEncoder(LoRA_config=model_config['LoRA'], SAM_config=model_config['SAM'])
        self.sam.set_trainable_params()

        self.sam_reduction = ConvBNSiLU(in_channels=256*4, out_channels=256, kernel_size=1, padding=0)
        
        self.fpn = FPN(FPN_config=model_config['FPN'])

        self.cbam_sam = CBAMBlock(in_channels=256)

        self.decoder = ImgDecoder()

    def forward(self, img):
        # edge = compute_edge(img)

        (x_sam_0, x_sam_1, x_sam_2, x_sam_3) = self.sam(img)
        x_sam = self.sam_reduction(torch.cat([x_sam_0, x_sam_1, x_sam_2, x_sam_3], dim=1))

        (x_fpn_0, x_fpn_1, x_fpn_2) = self.fpn(img)
        
        x_de = self.cbam_sam(x_sam)

        logits = self.decoder(x_de, [x_fpn_0, x_fpn_1, x_fpn_2])

        return logits