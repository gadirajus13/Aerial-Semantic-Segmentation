# If you use the following architecture, please reference the following github
#  www.github.com/gadiraju.s13

import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 2
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)
        self.residual = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x) + self.residual(x)

class UNetWithYoloEncoder(nn.Module):
    def __init__(self, yolov5_encoder, n_classes):
        super(UNetWithYoloEncoder, self).__init__()
        self.encoder = yolov5_encoder
        self.up1 = Up(512, 512, 256)
        self.up2 = Up(256, 256, 128)
        self.up3 = Up(128, 128, 64)
        self.up4 = Up(64, 256, 128)
        self.up5 = Up(128, 512, 256)
        self.up6 = Up(256, 256, 128)
        self.up7 = Up(128, 128, 64)
        self.up8 = Up(64, 64, 32)
        self.out_conv = nn.Conv2d(32, n_classes, kernel_size=1)
        self.apply(self._init_decoder_weights)

    def _init_decoder_weights(self, m):
        if isinstance(m, (Up, OutConv)):
            initialize_weights(m)

    def forward(self, x):
        input_size = x.shape[2:]
        x, features = self.encoder(x)
        x = self.up1(x, features[23])
        x = self.up2(x, features[20])
        x = self.up3(x, features[17])
        x = self.up4(x, features[13])
        x = self.up5(x, features[9])
        x = self.up6(x, features[6])
        x = self.up7(x, features[4])
        x = self.up8(x, features[2])
        
        x = self.out_conv(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return x