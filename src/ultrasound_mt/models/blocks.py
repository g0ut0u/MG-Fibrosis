"""Shared model building blocks: residual block, down/up sampling, CBAM attention.

These modules were duplicated in every model file in the original repository.
Here they are extracted into shared components and reused by models.py so that
every model's parameter count matches the original implementation exactly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, num_layers=2):
        super().__init__()
        assert num_layers >= 1
        padding = kernel // 2
        self.relu = nn.ReLU(inplace=True)

        layers = []
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride,
                                padding=padding, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
        if num_layers > 1:
            layers.append(nn.ReLU(inplace=True))

        for i in range(1, num_layers):
            layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=kernel, stride=1,
                                    padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(out_ch))
            if i != num_layers - 1:
                layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.main(x)
        out = out + identity
        out = self.relu(out)
        return out


class DownRes(nn.Module):
    def __init__(self, in_ch, out_ch, num_layers=1):
        super().__init__()
        self.block = ResidualBlock(in_ch, out_ch, kernel=3, stride=2, num_layers=num_layers)

    def forward(self, x):
        return self.block(x)


class UpRes(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, bilinear=True, num_res_blocks=2):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)

        blocks = [ResidualBlock(in_ch + skip_ch, out_ch, stride=1, num_layers=2)]
        for _ in range(num_res_blocks - 1):
            blocks.append(ResidualBlock(out_ch, out_ch, stride=1, num_layers=2))
        self.conv = nn.Sequential(*blocks)

    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)

        if diffY != 0 or diffX != 0:
            x = F.pad(x, [
                diffX // 2, diffX - diffX // 2,
                diffY // 2, diffY - diffY // 2
            ])

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x
