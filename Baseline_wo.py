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


class BaselineResUNetClassifier_wo(nn.Module):
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




        total_dim = 128
        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        x1 = self.stage1(x)                     # [B, 8, H, W]
        x2 = self.stage2(self.down12(x1))       # [B,16, H/2,W/2]
        x3 = self.stage3(self.down23(x2))       # [B,32, H/4,W/4]
        x4 = self.stage4(self.down34(x3))       # [B,64, H/8,W/8]
        x5 = self.stage5(self.down45(x4))       # [B,128,H/16,W/16]

        feat_x5 = self.gap(x5).flatten(1)       # [B,128]      # [B, 64]
        cls_logits = self.classifier(feat_x5)                     # [B, 1]

        return {"cls_logits": cls_logits}


if __name__ == "__main__":
    model = BaselineResUNetClassifier_wo(in_channels=1, bilinear=True, dropout=0.3)
    x = torch.randn(2, 1, 768, 1024)
    out = model(x)
    print("cls_logits shape:", out["cls_logits"].shape)  # [2, 1]
    
    with torch.no_grad():
        x1 = model.stage1(x)
        x2 = model.stage2(model.down12(x1))
        x3 = model.stage3(model.down23(x2))
        x4 = model.stage4(model.down34(x3))
        x5 = model.stage5(model.down45(x4))
        print(f"x4 shape: {x4.shape} -> expected [2,64,96,128]")
        print(f"x5 shape: {x5.shape} -> expected [2,128,48,64]")
