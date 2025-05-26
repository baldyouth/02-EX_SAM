from loadSAM import load_SAM_model
import torch.nn as nn

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
            Mamba(d_model=dim) for _ in range(depth)
        ])
        self.rmsnorm = RMSNorm(dim)
    
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
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.layer1 = nn.Sequential(
            ConvBNGELU(in_channels=in_channels, out_channels=base_channels, kernel_size=3, stride=2, padding=1),
            ConvBNGELU(in_channels=base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1),
            ConvBNGELU(in_channels=base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1)
        )
        self.layer2 = nn.Sequential(
            ConvBNGELU(in_channels=base_channels, out_channels=base_channels*2, kernel_size=3, stride=2, padding=1),
            ConvBNGELU(in_channels=base_channels*2, out_channels=base_channels*2, kernel_size=3, stride=1, padding=1),
            ConvBNGELU(in_channels=base_channels*2, out_channels=base_channels*2, kernel_size=3, stride=1, padding=1)
        )
        self.layer3 = nn.Sequential(
            ConvBNGELU(in_channels=base_channels*2, out_channels=base_channels*4, kernel_size=3, stride=2, padding=1),
            ConvBNGELU(in_channels=base_channels*4, out_channels=base_channels*4, kernel_size=3, stride=1, padding=1),
            ConvBNGELU(in_channels=base_channels*4, out_channels=base_channels*4, kernel_size=3, stride=1, padding=1)
        )
        self.layer4 = nn.Sequential(
            ConvBNGELU(in_channels=base_channels*4, out_channels=base_channels*8, kernel_size=3, stride=2, padding=1),
            ConvBNGELU(in_channels=base_channels*8, out_channels=base_channels*8, kernel_size=3, stride=1, padding=1),
            ConvBNGELU(in_channels=base_channels*8, out_channels=base_channels*8, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x1, x2, x3, x4

#!!! SAMGuidedCrossAttention
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

#!!! ImgEncoder
class ImgEncoder(nn.Module):
    def __init__(self, dims=[768, 256], num_heads=[4, 8], kernel_size=3, depths=[2, 2, 8, 8], base_channels=32):
        super().__init__()
        self.attn1 = AttentionBlock(dim=dims[0], num_heads=num_heads[0], kernel_size=kernel_size, depth=depths[0])
        self.attn2 = AttentionBlock(dim=dims[0], num_heads=num_heads[1], kernel_size=kernel_size, depth=depths[1])
        self.mamba1 = MambaBlock(dim=dims[0], depth=depths[2])
        self.mamba2 = MambaBlock(dim=dims[0], depth=depths[3])

        self.SAM = load_SAM_model(modeName='base')
        self.fpn = FPN(base_channels=base_channels)

        self.samguide1 = SAMGuidedCrossAttention(dim=base_channels, heads=2, window_size=8)
        self.samguide2 = SAMGuidedCrossAttention(dim=base_channels*2, heads=2, window_size=8)
        self.samguide3 = SAMGuidedCrossAttention(dim=base_channels*4, heads=2, window_size=4)
        self.samguide4 = SAMGuidedCrossAttention(dim=base_channels*8, heads=2, window_size=4)
    
    def forward(self, x):
        with torch.no_grad():
            self.SAM.eval()
            SAM_f = self.SAM(x)

        Attn1_f = self.attn1(SAM_f[0])
        Attn2_f = self.attn2(SAM_f[1])
        Mamba1_f = self.mamba1(SAM_f[2])
        Mamba2_f = self.mamba2(SAM_f[3])

        fpn_1, fpn_2, fpn_3, fpn_4 = self.fpn(x)

        imgf_1 = self.samguide1(fpn_1, Attn1_f)
        imgf_2 = self.samguide2(fpn_2, Attn2_f)
        imgf_3 = self.samguide3(fpn_3, Mamba1_f)
        imgf_4 = self.samguide4(fpn_4, Mamba2_f)

        return imgf_1, imgf_2, imgf_3, imgf_4, SAM_f[-1]

if __name__ == '__main__':
    import torch

    # model = ImgEncoder()
    model = FPN()
    # print(model)

    input = torch.rand((1, 3, 1024, 1024)).to('cuda')
    model.to('cuda')
    output = model(input)
    for i in output:
        print(i.shape)



    