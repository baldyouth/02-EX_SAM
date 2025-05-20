import torch
from torch import nn
from natten import NeighborhoodAttention2D, use_fused_na
from mamba_ssm import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm

use_fused_na()

class AttentionMambaBlock(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=3, depth=2):
        super().__init__()

        self.mambalayers = nn.ModuleList([
            Mamba(d_model=dim) for _ in range(depth)
        ])
        self.rmsnorm = RMSNorm(dim)
        self.attention = NeighborhoodAttention2D(dim=dim, num_heads=num_heads, kernel_size=kernel_size) # B, H, W, C
        self.conv1 = nn.Conv2d(in_channels=2*dim, out_channels=2*dim, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x_atten = self.attention(x)
        x_atten = x_atten.permute(0, 3, 1, 2)

        B, C, H, W = x_atten.shape
        x_mamba = x_atten.view(B, C, H * W).permute(0, 2, 1)  # B×L×C
        for layer in self.mambalayers:
            x_mamba = layer(x_mamba)
        x_mamba = self.rmsnorm(x_mamba)
        x_mamba = x_mamba.view(B, H, W, C).permute(0, 3, 1, 2)

        x_out = torch.cat([x_atten, x_mamba], dim=1)
        x_out = self.conv1(x_out)

        return x_out

class encoder(nn.Module):
    def __init__(self, dims=[32, 64, 128, 256], num_heads=[4, 4, 8, 8]):
        super().__init__()

        self.nattenBlock_1 = nn.Sequential(
            nn.Conv2d(3, 32, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            AttentionMambaBlock(dim=dims[0], num_heads=num_heads[0], kernel_size=3, depth=2)
        )
        self.nattenBlock_2 = nn.Sequential(
            nn.Conv2d(64, 64, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            AttentionMambaBlock(dim=dims[1], num_heads=num_heads[1], kernel_size=3, depth=2)
        )
        self.nattenBlock_3 = nn.Sequential(
            nn.Conv2d(128, 128, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.GELU(),
            AttentionMambaBlock(dim=dims[2], num_heads=num_heads[2], kernel_size=3, depth=2)
        )
        self.nattenBlock_4 = nn.Sequential(
            nn.Conv2d(256, 256, stride=2, padding=1, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.GELU(),
            AttentionMambaBlock(dim=dims[3], num_heads=num_heads[3], kernel_size=3, depth=2)
        )
    def forward(self, x):
        x1 = self.nattenBlock_1(x)
        x2 = self.nattenBlock_2(x1)
        x3 = self.nattenBlock_3(x2)
        x4 = self.nattenBlock_4(x3)
        return (x2, x3, x4)

if __name__ == '__main__':

    # input = torch.rand((1, 3, 448, 448), device='cuda')
    # model = encoder().to('cuda')
    # output = model(input)

    # print(output.shape)
    input = torch.rand((1, 3, 448, 448), device='cuda')
    model = encoder().to('cuda')
    output = model(input)
    for i in output:
        print(i.shape)