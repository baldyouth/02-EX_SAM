import torch
from torch import nn
from visionmamba import VisionMambaSeg
from natten import NeighborhoodAttention2D, use_fused_na
from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm

use_fused_na()

class AttentionMambaBlock(nn.Module):
    def __init__(self, dim, height, width, num_heads, kernel_size=3, depth=12):
        super().__init__()

        self.height = height
        self.width = width
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.depth = depth

        self.mambalayers = nn.ModuleList([
            Mamba(d_model=dim) for _ in range(depth)
        ])
        self.rmsnorm = RMSNorm(dim)
        self.attention = NeighborhoodAttention2D(dim=dim, num_heads=num_heads, kernel_size=kernel_size) # B, H, W, C

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.height and W == self.width and C == self.mambalayers[0].d_model, \
            f"Input size should be {C}×{H}×{W}, but got {x.shape}"

        x = x.view(B, C, H * W).permute(0, 2, 1)  # B×L×C

        for layer in self.mambalayers:
            x = layer(x)

        x = self.rmsnorm(x)
        x = x.view(B, H, W, C)
        x = self.attention(x)
        x = x.permute(0, 3, 1, 2)
        return x

class encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.nattenBlock_1 = nn.Sequential(
            nn.Conv2d(3, 32, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            AttentionMambaBlock(dim=32, height=224, width=224, num_heads=4, kernel_size=3)
        )
        self.nattenBlock_2 = nn.Sequential(
            nn.Conv2d(32, 64, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            AttentionMambaBlock(dim=64, height=112, width=112, num_heads=4, kernel_size=3)
        )
        self.nattenBlock_3 = nn.Sequential(
            nn.Conv2d(64, 128, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.GELU(),
            AttentionMambaBlock(dim=128, height=56, width=56, num_heads=8, kernel_size=3)
        )
        self.nattenBlock_4 = nn.Sequential(
            nn.Conv2d(128, 256, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.GELU(),
            AttentionMambaBlock(dim=256, height=28, width=28, num_heads=8, kernel_size=3)
        )
    def forward(self, x):
        x1 = self.nattenBlock_1(x)
        x2 = self.nattenBlock_2(x1)
        x3 = self.nattenBlock_3(x2)
        x4 = self.nattenBlock_4(x3)
        return (x1, x2, x3, x4)

if __name__ == '__main__':

    # input = torch.rand((1, 3, 448, 448), device='cuda')
    # model = encoder().to('cuda')
    # output = model(input)

    # print(output.shape)
    input = torch.rand((1, 3, 448, 448), device='cuda')
    layer1 = nn.Sequential(
            nn.Conv2d(3, 32, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            AttentionMambaBlock(dim=32, height=224, width=224, num_heads=4, kernel_size=3)
        ).to('cuda')
    output = layer1(input)
    print(output.shape)

