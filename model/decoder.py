import torch
import torch.nn as nn

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

class concatEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            ConvBNGELU(128, 256),
            nn.MaxPool2d(2),
            ConvBNGELU(256, 256),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            ConvBNGELU(256, 256),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            ConvBNGELU(512, 256),
            nn.Upsample(2)
        )
        self.conv1 = nn.Conv2d(256, 256, kernel_size=1)
    
    def forward(self, x):
        x1 = torch.concat([self.block1(x(0)), self.block2(x(1)), self.block3(x(2))], dim=1)
        x1 = self.conv1(x1)
