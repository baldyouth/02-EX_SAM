from loadSAM import load_SAM_model
import torch.nn as nn

from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm

from natten import NeighborhoodAttention2D, use_fused_na

use_fused_na()

#!!! ConvBNGELU
class ConvBNGELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
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

#!!! ImgEncoder
class ImgEncoder(nn.Module):
    def __init__(self, dims=[768, 256], num_heads=[8, 12], kernel_size=3, depths=[2, 4, 8, 8]):
        super().__init__()
        self.attn1 = AttentionBlock(dim=dims[0], num_heads=num_heads[0], kernel_size=kernel_size, depth=depths[0])
        self.attn2 = AttentionBlock(dim=dims[0], num_heads=num_heads[1], kernel_size=kernel_size, depth=depths[1])
        self.mamba1 = MambaBlock(dim=dims[0], depth=depths[2])
        self.mamba2 = MambaBlock(dim=dims[1], depth=depths[3])

        self.SAM = load_SAM_model(modeName='base')
    
    def forward(self, x):
        SAM_f = self.SAM(x)

        Attn1_f = self.attn1(SAM_f[0])
        Attn2_f = self.attn2(SAM_f[1])
        Mamba1_f = self.mamba1(SAM_f[2])
        Mamba2_f = self.mamba2(SAM_f[3])

        return Attn1_f, Attn2_f, Mamba1_f, Mamba2_f

if __name__ == '__main__':
    import torch

    model = ImgEncoder()
    print(model)
    input = torch.rand((1, 3, 1024, 1024)).to('cuda')
    model.to('cuda')
    output = model(input)
    for i in output:
        print(i.shape)



    