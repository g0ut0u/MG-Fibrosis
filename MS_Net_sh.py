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


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
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

        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


class MSNet_sh(nn.Module):
    def __init__(self, in_channels=1, bilinear=True, dropout=0.3):
        super().__init__()

        self.stage1 = ResidualBlock(in_channels, 8,   stride=1, num_layers=2)  # H
        self.down12 = DownRes(8,   16, num_layers=1)
        self.stage2 = ResidualBlock(16, 16, stride=1, num_layers=2)             # H/2

        self.down23 = DownRes(16,  32, num_layers=1)
        self.stage3 = ResidualBlock(32, 32, stride=1, num_layers=3)             # H/4

        self.down34 = DownRes(32,  64, num_layers=1)
        self.stage4 = ResidualBlock(64, 64, stride=1, num_layers=3)             # H/8

        self.down45 = DownRes(64, 128, num_layers=1)
        self.stage5 = ResidualBlock(128, 128, stride=1, num_layers=4)           # H/16

        self.cbam4 = CBAM(64, reduction=16, spatial_kernel=7)
        self.cbam5 = CBAM(128, reduction=16, spatial_kernel=7)


        self.up1 = UpRes(128, 64, 64, bilinear=bilinear, num_res_blocks=2)      # H/8 -> u1
        self.cbam_u1 = CBAM(64, reduction=16, spatial_kernel=7)


        total_dim = 64 + 128 + 64
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  
        )

        # 全局平均池化层
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # 编码器
        x1 = self.stage1(x)                     # [B, 8, H, W]
        x2 = self.stage2(self.down12(x1))       # [B,16, H/2,W/2]
        x3 = self.stage3(self.down23(x2))       # [B,32, H/4,W/4]
        x4 = self.stage4(self.down34(x3))       # [B,64, H/8,W/8]
        x4 = self.cbam4(x4)
        x5 = self.stage5(self.down45(x4))       # [B,128,H/16,W/16]
        x5 = self.cbam5(x5)

        # 仅第一个上采样块
        u1 = self.up1(x5, x4)                   # [B,64, H/8,W/8]
        u1 = self.cbam_u1(u1)

        # 全局平均池化
        feat_x4 = self.gap(x4).flatten(1)       # [B, 64]
        feat_x5 = self.gap(x5).flatten(1)       # [B,128]
        feat_u1 = self.gap(u1).flatten(1)       # [B, 64]

        # 融合并分类
        feat = torch.cat([feat_x4, feat_x5, feat_u1], dim=1)  # [B, 256]
        cls_logits = self.classifier(feat)                     # [B, 1]

        return {"cls_logits": cls_logits}


if __name__ == "__main__":
    model = MSNet_sh(in_channels=1, bilinear=True, dropout=0.3)
    x = torch.randn(2, 1, 768, 1024)
    out = model(x)
    print("cls_logits shape:", out["cls_logits"].shape)  # [2, 1]
    
    with torch.no_grad():
        x1 = model.stage1(x)
        x2 = model.stage2(model.down12(x1))
        x3 = model.stage3(model.down23(x2))
        x4 = model.stage4(model.down34(x3))
        x5 = model.stage5(model.down45(x4))
        u1 = model.up1(x5, x4)
        print(f"x4 shape: {x4.shape} -> expected [2,64,96,128]")
        print(f"x5 shape: {x5.shape} -> expected [2,128,48,64]")
        print(f"u1 shape: {u1.shape} -> expected [2,64,96,128]")