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

class ROIAwareMultiScaleClassifier(nn.Module):
    def __init__(self, c1, c2, c3, dim=128, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(c1 + c2 + c3, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, 1)
        )

    @staticmethod
    def downsample_mask_to_feat(mask_prob, feat):
        H, W = mask_prob.shape[-2:]
        h, w = feat.shape[-2:]
        assert H % h == 0 and W % w == 0, f"Mask size {(H,W)} not divisible by feat size {(h,w)}"

        sh = H // h
        sw = W // w

        mask_ds = F.avg_pool2d(mask_prob, kernel_size=(sh, sw), stride=(sh, sw))

        if mask_ds.shape[-2:] != (h, w):
            mask_ds = F.interpolate(mask_ds, size=(h, w), mode='nearest')

        return mask_ds

    @staticmethod
    def masked_gap(feat, mask_prob):
        """
        feat:      [B,C,h,w]
        mask_prob: [B,1,H,W]  (full-res probability map after sigmoid)
        """
        mask = ROIAwareMultiScaleClassifier.downsample_mask_to_feat(mask_prob, feat)

        denom = mask.sum(dim=(2, 3), keepdim=False).clamp_min(1e-6)  # [B,1]
        num = (feat * mask).sum(dim=(2, 3), keepdim=False)           # [B,C]
        pooled = num / denom                                         # [B,C]
        return pooled

    def forward(self, f1, f2, f3, seg_logits):
        mask_prob = torch.sigmoid(seg_logits)

        v1 = self.masked_gap(f1, mask_prob)
        v2 = self.masked_gap(f2, mask_prob)
        v3 = self.masked_gap(f3, mask_prob)

        v = torch.cat([v1, v2, v3], dim=1)
        return self.proj(v)


class MTMP(nn.Module):
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

        self.up1 = UpRes(128, 64, 64, bilinear=bilinear, num_res_blocks=2)      # H/8
        self.up2 = UpRes(64,  32, 32, bilinear=bilinear, num_res_blocks=2)      # H/4
        self.up3 = UpRes(32,  16, 16, bilinear=bilinear, num_res_blocks=2)      # H/2
        self.up4 = UpRes(16,   8,  8, bilinear=bilinear, num_res_blocks=3)      # H

        self.cbam4 = CBAM(64, reduction=16, spatial_kernel=7)
        self.cbam5 = CBAM(128, reduction=16, spatial_kernel=7)
        self.cbam_u1 = CBAM(64, reduction=16, spatial_kernel=7)

        self.seg_head = nn.Conv2d(8, 1, kernel_size=1)
        self.cls_head = ROIAwareMultiScaleClassifier(
            c1=64, c2=128, c3=64, dim=128, dropout=dropout
        )

    def forward(self, x, return_cls=True):
        x1 = self.stage1(x)                 # [B,  8, H,    W]
        x2 = self.stage2(self.down12(x1))   # [B, 16, H/2,  W/2]
        x3 = self.stage3(self.down23(x2))   # [B, 32, H/4,  W/4]
        x4 = self.stage4(self.down34(x3))   # [B, 64, H/8,  W/8]
        cbam4 = self.cbam4(x4)
        x5 = self.stage5(self.down45(x4))   # [B,128, H/16, W/16]
        cbam5 = self.cbam5(x5)

        u1 = self.up1(x5, x4)               # [B, 64, H/8,  W/8]
        cbam1 = self.cbam_u1(u1)
        u2 = self.up2(u1, x3)               # [B, 32, H/4,  W/4]
        u3 = self.up3(u2, x2)               # [B, 16, H/2,  W/2]
        u4 = self.up4(u3, x1)               # [B,  8, H,    W]


        seg_logits = self.seg_head(u4)      # [B,1,H,W]
        if not return_cls:
            return {"seg_logits": seg_logits, "cls_logits": None}
        seg_logits_detached = seg_logits.detach()
        cls_logits = self.cls_head(cbam4, cbam5, cbam1,seg_logits_detached)

        return {
            "seg_logits": seg_logits,
            "cls_logits": cls_logits
        }
