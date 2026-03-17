import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------
# 2D decoder
# ------------------------------------------------

class MyDecoderBlock(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel + skip_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class MyUnetDecoder(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel):
        super().__init__()
        self.center = nn.Identity()

        i_channel = [in_channel] + out_channel[:-1]
        block = [
            MyDecoderBlock(i, s, o)
            for i, s, o in zip(i_channel, skip_channel, out_channel)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            s = skip[i]
            d = block(d, s)
            decode.append(d)
        last = d
        return last, decode


# ------------------------------------------------
# 3D decoder
# ------------------------------------------------

class MyDecoderBlock3d(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channel + skip_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention1 = nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )
        self.attention2 = nn.Identity()

    def forward(self, x, skip=None, depth_scaling=2):
        x = F.interpolate(x, scale_factor=(depth_scaling, 2, 2), mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class MyUnetDecoder3d(nn.Module):
    def __init__(self, in_channel, skip_channel, out_channel):
        super().__init__()
        self.center = nn.Identity()

        i_channel = [in_channel] + out_channel[:-1]
        block = [
            MyDecoderBlock3d(i, s, o)
            for i, s, o in zip(i_channel, skip_channel, out_channel)
        ]
        self.block = nn.ModuleList(block)

    def forward(self, feature, skip, depth_scaling=[2, 2, 2, 2, 2, 2]):
        d = self.center(feature)
        decode = []
        for i, block in enumerate(self.block):
            s = skip[i]
            d = block(d, s, depth_scaling[i])
            decode.append(d)
        last = d
        return last, decode