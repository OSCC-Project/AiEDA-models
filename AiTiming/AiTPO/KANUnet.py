import torch
import torch.nn as nn
import torch.nn.functional as F
from kan import KAN

# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels, mid_channels=None):
#         super().__init__()
#         if not mid_channels:
#             mid_channels = out_channels
#         self.double_conv = nn.Sequential(
#             nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.double_conv(x)

class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, with_r=False):
        super(CoordConv, self).__init__()
        self.with_r = with_r  # Correctly initialize with_r attribute
        extra_channels = 2  # x and y coordinate channels
        if with_r:
            extra_channels += 1  # Add an additional channel for radius if with_r is True
        self.conv = nn.Conv2d(in_channels + extra_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        batch_size, _, height, width = x.size()

        # Create coordinate tensors
        xx_channel = torch.arange(width).repeat(1, height, 1)
        yy_channel = torch.arange(height).repeat(1, width, 1).transpose(1, 2)

        # Normalize to range [-1, 1]
        xx_channel = xx_channel.float() / (width - 1)
        yy_channel = yy_channel.float() / (height - 1)
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        # Expand and concatenate with input
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).to(x.device)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).to(x.device)
        coords = torch.cat([x, xx_channel, yy_channel], dim=1)

        if self.with_r:
            rr = torch.sqrt(xx_channel**2 + yy_channel**2)
            coords = torch.cat([coords, rr], dim=1)

        return self.conv(coords)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, with_r=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            CoordConv(in_channels, mid_channels, kernel_size=3, padding=1, with_r=with_r),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            CoordConv(mid_channels, out_channels, kernel_size=3, padding=1, with_r=with_r),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)



class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class EncoderUNet(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(EncoderUNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x5  # 只返回encoder的输出

class UNetEncoderMLP(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetEncoderMLP, self).__init__()
        self.encoder = EncoderUNet(n_channels, bilinear)
        self.pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc1 = nn.Linear(1024 // (2 if bilinear else 1), 512)
        self.fc2 = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class KANUnet(nn.Module):
    def __init__(self, input_size, n_channels, n_classes, bilinear=False):
        super(KANUnet, self).__init__()
        self.unet = UNetEncoderMLP(n_channels, n_classes)
        # self.kan = KAN([input_size, 16,64,64,256,512,1024,2048,2048,2048,512,64,8,1])
        # self.kan = KAN([input_size, 16,64,64,256,512,1024,2048,2048,2048,512,64,8,1])
        self.kan = KAN([input_size, 16,64,64,256,512,512,64,8,1])
        # self.kan = KAN([input_size, 16,64,64,256,512,1024,2048,2048,2048,2048,2048,512,64,64,8,1])
        # self.kan = KAN([input_size, 16,64,64,256,512,64,64,8,1])
        # self.kan = KAN([input_size, 16,32,32,64,32,32,32,16,1])

        
    def forward(self, x_path, x_map):
        x1 = self.unet(x_map)
        x2 = torch.cat((x_path, x1), dim=1)
        x3 = self.kan(x2)
       
        return x3
