import torch
import torch.nn as nn
from mamba_ssm import Mamba
import torch.nn.functional as F

class ConvBNGELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.block(x)

# concat MyNet_feat
class concatEncoder(nn.Module):
    def __init__(self, in_c=[128, 256, 512], out_c=256):
        super().__init__()
        self.block1 = nn.Sequential(
            ConvBNGELU(in_c[0], out_c),
            nn.MaxPool2d(2),
            ConvBNGELU(out_c, out_c),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            ConvBNGELU(in_c[1], out_c),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_c[2], out_c, kernel_size=1),
            nn.BatchNorm2d(out_c),
            nn.GELU()
        )
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(out_c*3, out_c, kernel_size=1),
            nn.BatchNorm2d(out_c),
            nn.GELU()
        )
        self.refine = nn.Sequential(
            ConvBNGELU(out_c, out_c),
            ConvBNGELU(out_c, out_c)
        )
    
    def forward(self, x):
        '''
        x(0).shape = (B, 128, H/4, W/4)
        x(1).shape = (B, 256, H/8, W/8)
        x(2).shape = (B, 512, H/16, W/16)
        '''
        x1 = torch.concat([self.block1(x[0]), self.block2(x[1]), self.block3(x[2])], dim=1)
        x1 = self.reduce_conv(x1)
        x1 = self.refine(x1)
        return x1

# !!! fused(Cross-Attn融合): SAM_feat + MyNet_feat
class crossAttention(nn.Module):
    def __init__(self, dim=256, heads=4, dropout=0.1):
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

# !!! enhanced(CCA空间增强): Criss-Cross Attention
class LightweightCCA(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        inter_channels = in_channels // reduction
        self.query_conv = nn.Conv2d(in_channels, inter_channels, 1)
        self.key_conv   = nn.Conv2d(in_channels, inter_channels, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        proj_query = self.query_conv(x).permute(0,2,3,1).reshape(B*H, W, -1)
        proj_key   = self.key_conv(x).permute(0,2,3,1).reshape(B*H, W, -1)
        attn_h = torch.bmm(proj_query, proj_key.transpose(1,2)).softmax(dim=-1)
        proj_value = self.value_conv(x).permute(0,2,3,1).reshape(B*H, W, -1)
        out_h = torch.bmm(attn_h, proj_value).reshape(B, H, W, C).permute(0,3,1,2)

        proj_query = self.query_conv(x).permute(0,3,2,1).reshape(B*W, H, -1)
        proj_key   = self.key_conv(x).permute(0,3,2,1).reshape(B*W, H, -1)
        attn_v = torch.bmm(proj_query, proj_key.transpose(1,2)).softmax(dim=-1)
        proj_value = self.value_conv(x).permute(0,3,2,1).reshape(B*W, H, -1)
        out_v = torch.bmm(attn_v, proj_value).reshape(B, W, H, C).permute(0,3,2,1)

        return self.gamma * (out_h + out_v) + x

# !!! Mamba
class MambaModule(nn.Module):
    def __init__(self, dim=256, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.to_seq = nn.Linear(dim, dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.from_seq = nn.Linear(dim, dim)

    def forward(self, x_feat):
        # x_feat: [B, C, H, W]
        B, C, H, W = x_feat.shape
        seq = x_feat.view(B, C, H * W).transpose(1, 2)  # [B, L, C]
        seq = self.to_seq(seq)
        seq = self.mamba(seq)                           # [B, L, C]
        seq = self.from_seq(seq)
        out = seq.transpose(1, 2).view(B, C, H, W)      # [B, C, H, W]
        return out

# !!! Fusion moudles
class fusionModule(nn.Module):
    """
    Fusion pipeline: cross-attention → CCA → decoupled MambaModule.
    """
    def __init__(self, dim=256, d_state=16, d_conv=4, expand=2, mamba_depth=2):
        super().__init__()
        self.cross_attn = crossAttention(dim=dim)
        self.cca = LightweightCCA(dim)
        self.mamba_module = nn.ModuleList([
            MambaModule(dim=dim, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(mamba_depth)
        ])

    def forward(self, sam_feat, mynet_feat):
        fusion = self.cross_attn(sam_feat, mynet_feat)
        spatial = self.cca(fusion)
        for layer in self.mamba_module:
            spatial = layer(spatial)
        return spatial
    
# !!! SegHead
class segHead(nn.Module):
    def __init__(self, fusion_channels = 256, skip_channels = 256, mid_channels = 128, num_classes = 1, out_size = (448, 448)):
        super().__init__()
        
        self.skip_conv = ConvBNGELU(skip_channels, mid_channels)
        self.fuse_conv = ConvBNGELU(fusion_channels*2, fusion_channels)
        self.res_conv = nn.Conv2d(mid_channels*2, fusion_channels, 1)
        self.cls_conv = nn.Conv2d(fusion_channels, num_classes, 1)
        self.out_size = out_size
    
    def forward(self, fused_feat, sam_feat, mynet_feat):
        sam_feat = self.skip_conv(sam_feat)
        mynet_feat = self.skip_conv(mynet_feat)
        res_feat = self.res_conv(torch.concat([sam_feat, mynet_feat], dim=1))

        x = self.fuse_conv(torch.cat([res_feat, fused_feat], dim=1))

        mask_logits = self.cls_conv(x)
        mask_logits = F.interpolate(
            mask_logits,
            size=self.out_size,
            mode='bilinear',
            align_corners=False
        )
        mask_prob = torch.sigmoid(mask_logits)
        
        return mask_logits, mask_prob
