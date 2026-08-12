"""Classification heads: a global-pooling head and a segmentation-guided,
ROI-aware multi-scale head.

- ``GlobalPoolingClassifier``: global average pooling over multi-scale features
  followed by a classifier (no mask guidance).
- ``ROIAwareMultiScaleClassifier``: weights multi-scale features by the
  segmentation mask probability map before pooling for classification. Supports
  two pooling modes:
    * ``mode="roi_gap"`` : mask-weighted global average pooling (masked-GAP),
      formerly MT-MP.
    * ``mode="hybrid"``  : for each channel, half is ROI-pooled and half is
      globally pooled, then concatenated. Formerly MT-MIX.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalPoolingClassifier(nn.Module):
    """Global average pooling classification head (no mask guidance)."""
    def __init__(self, c1, c2, c3, dim=128, dropout=0.3):
        super().__init__()
        # Global average pooling layer
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Concatenate then reduce-dimension classifier
        self.proj = nn.Sequential(
            nn.Linear(c1 + c2 + c3, dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim, 1)
        )

    def forward(self, f1, f2, f3):
        # GAP each feature map, [B, C, 1, 1] -> squeeze to [B, C]
        v1 = self.gap(f1).flatten(1)
        v2 = self.gap(f2).flatten(1)
        v3 = self.gap(f3).flatten(1)
        # Concatenate all scale features
        v = torch.cat([v1, v2, v3], dim=1)
        return self.proj(v)


class ROIAwareMultiScaleClassifier(nn.Module):
    """Segmentation-guided, ROI-aware multi-scale classification head.

    Args:
        mode: "roi_gap" uses masked-GAP (formerly MT-MP), "hybrid" uses
              ROI + global mixed pooling (formerly MT-MIX).
    """
    def __init__(self, c1, c2, c3, dim=128, dropout=0.3, mode="roi_gap"):
        super().__init__()
        self.mode = mode
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

    @staticmethod
    def hybrid_pool(feat, mask_prob):
        B, C, h, w = feat.shape
        mask = ROIAwareMultiScaleClassifier.downsample_mask_to_feat(
            mask_prob, feat
        )

        c_roi = C // 2
        c_global = C - c_roi

        feat_roi = feat[:, :c_roi]
        feat_global = feat[:, c_roi:]

        # ROI pooling
        denom = mask.sum(dim=(2, 3), keepdim=False).clamp_min(1e-6)
        num = (feat_roi * mask).sum(dim=(2, 3), keepdim=False)
        pooled_roi = num / denom

        # Global pooling
        pooled_global = F.adaptive_avg_pool2d(
            feat_global, 1
        ).flatten(1)

        pooled = torch.cat([pooled_roi, pooled_global], dim=1)
        return pooled

    def forward(self, f1, f2, f3, seg_logits):
        mask_prob = torch.sigmoid(seg_logits)

        if self.mode == "roi_gap":
            v1 = self.masked_gap(f1, mask_prob)
            v2 = self.masked_gap(f2, mask_prob)
            v3 = self.masked_gap(f3, mask_prob)
        elif self.mode == "hybrid":
            v1 = self.hybrid_pool(f1, mask_prob)
            v2 = self.hybrid_pool(f2, mask_prob)
            v3 = self.hybrid_pool(f3, mask_prob)
        else:
            raise ValueError(f"Unknown ROI classifier mode: {self.mode}")

        v = torch.cat([v1, v2, v3], dim=1)
        return self.proj(v)
