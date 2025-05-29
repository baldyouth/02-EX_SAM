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

#!!! ImgDecoder
class ImgDecoder(nn.Module):
    def __init__(self, in_channels=[256, 128, 64, 32], out_channels=1):
        super().__init__()
        
        self.up_block1 = nn.Sequential(
            ConvBNGELU(in_channel=in_channels[0]*2, out_channel=in_channels[0]),
            ConvBNGELU(in_channel=in_channels[0], out_channel=in_channels[0]),
            nn.Conv2d(in_channels[0], in_channels[1], kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.up_block2 = nn.Sequential(
            ConvBNGELU(in_channel=in_channels[1]*2, out_channel=in_channels[1]),
            ConvBNGELU(in_channel=in_channels[1], out_channel=in_channels[1]),
            nn.Conv2d(in_channels[1], in_channels[2], kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.up_block3 = nn.Sequential(
            ConvBNGELU(in_channel=in_channels[2]*2, out_channel=in_channels[2]),
            ConvBNGELU(in_channel=in_channels[2], out_channel=in_channels[2]),
            nn.Conv2d(in_channels[2], in_channels[3], kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.up_block4 = nn.Sequential(
            ConvBNGELU(in_channel=in_channels[3]*2, out_channel=in_channels[3]),
            ConvBNGELU(in_channel=in_channels[3], out_channel=in_channels[3]),
            nn.Conv2d(in_channels[3], in_channels[3], kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.head = nn.Conv2d(in_channels[3], out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        up_x1 = self.up_block1(torch.cat([x[4], x[3]], dim=1))
        up_x2 = self.up_block2(torch.cat([up_x1, x[2]], dim=1))
        up_x3 = self.up_block3(torch.cat([up_x2, x[1]], dim=1))
        up_x4 = self.up_block4(torch.cat([up_x3, x[0]], dim=1))

        logits = self.head(up_x4)
        # prob = torch.sigmoid(logits)

        return logits

if __name__ == '__main__':
    input = [
        torch.rand((1, 32, 224, 224), device='cuda'),
        torch.rand((1, 64, 112, 112), device='cuda'),
        torch.rand((1, 128, 56, 56), device='cuda'),
        torch.rand((1, 256, 28, 28), device='cuda'),
        torch.rand((1, 256, 28, 28), device='cuda'),
    ]
    model = ImgDecoder().to('cuda')
    output = model(input)
    print(output.shape)
    
        