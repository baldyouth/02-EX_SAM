import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNGELU(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channel),
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
        
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
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

#!!! ImgDecoder
class ImgDecoder(nn.Module):
    def __init__(self, in_channels=[256, 128, 64, 32], out_channels=1):
        super().__init__()
        
        self.reduction = nn.Sequential(
            ConvBNGELU(in_channel=in_channels[0]*3, out_channel=in_channels[0]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBNGELU(in_channel=in_channels[0], out_channel=in_channels[1]),
            CBAMBlock(in_channels=in_channels[1], spatial_kernel=3)
        )
        self.up_block1 = nn.Sequential(
            ConvBNGELU(in_channel=in_channels[1]*2, out_channel=in_channels[2]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBNGELU(in_channel=in_channels[2], out_channel=in_channels[2]),
        )
        self.up_block2 = nn.Sequential(
            ConvBNGELU(in_channel=in_channels[2]*2, out_channel=in_channels[3]),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBNGELU(in_channel=in_channels[3], out_channel=in_channels[3]),
        )
        self.up_block3 = nn.Sequential(
            ConvBNGELU(in_channel=in_channels[3]*2, out_channel=out_channels),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBNGELU(in_channel=out_channels, out_channel=out_channels),
        )
        self.logits = nn.Sequential(
            ConvBNGELU(in_channel=out_channels*2, out_channel=out_channels),
            ConvBNGELU(in_channel=out_channels, out_channel=out_channels),
        )

    def forward(self, x):
        de_x1 = self.reduction(torch.cat([x[0], x[4], x[5]], dim=1))
        de_x2 = self.up_block1(torch.cat([de_x1, x[3]], dim=1))
        de_x3 = self.up_block2(torch.cat([de_x2, x[2]], dim=1))
        de_x4 = self.up_block3(torch.cat([de_x3, x[1]], dim=1))
        logit = self.logits(torch.cat([de_x4, x[-1]], dim=1))

        return logit

if __name__ == '__main__':
    input = [
        torch.rand((1, 256, 56, 56), device='cuda'),
        torch.rand((1, 64, 224, 224), device='cuda'),
        torch.rand((1, 128, 112, 112), device='cuda'),
        torch.rand((1, 256, 56, 56), device='cuda')
    ]
    model = ImgDecoder().to('cuda')
    output = model(input)
    print(output.shape)
    
        