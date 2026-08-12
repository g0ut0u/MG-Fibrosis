"""Model factory: assembles all model variants from configuration.

In the original repository each model was a standalone .py file with much
duplication. Here the models are assembled from the shared ``blocks.py`` /
``heads.py`` components, with modules attached **directly on the model object**
(no nested submodule), so the state_dict keys match the original
implementation byte-for-byte and all checkpoints in ``Result/`` can be loaded
strictly.

Model name -> config (each verified against the original implementation):
- Multi-task (classification + segmentation, MT_* family): head in {gap, roi_gap,
  hybrid}, decoder="full"
- Single-task (classification only): MS_Net / Baseline, decoder="u1" / "none"
"""
import torch
import torch.nn as nn

from .blocks import ResidualBlock, DownRes, UpRes, CBAM
from .heads import GlobalPoolingClassifier, ROIAwareMultiScaleClassifier


def _attach_encoder(self, in_channels, stage4_layers, stage5_layers):
    """Attach the encoder layers directly to self (keys match the original:
    stage1..stage5 / down12..down45)."""
    self.stage1 = ResidualBlock(in_channels, 8,   stride=1, num_layers=2)  # H
    self.down12 = DownRes(8,   16, num_layers=1)
    self.stage2 = ResidualBlock(16, 16, stride=1, num_layers=2)             # H/2
    self.down23 = DownRes(16,  32, num_layers=1)
    self.stage3 = ResidualBlock(32, 32, stride=1, num_layers=3)             # H/4
    self.down34 = DownRes(32,  64, num_layers=1)
    self.stage4 = ResidualBlock(64, 64, stride=1, num_layers=stage4_layers)  # H/8
    self.down45 = DownRes(64, 128, num_layers=1)
    self.stage5 = ResidualBlock(128, 128, stride=1, num_layers=stage5_layers)  # H/16


def _encode(self, x):
    x1 = self.stage1(x)                 # [B,  8, H,    W]
    x2 = self.stage2(self.down12(x1))   # [B, 16, H/2,  W/2]
    x3 = self.stage3(self.down23(x2))   # [B, 32, H/4,  W/4]
    x4 = self.stage4(self.down34(x3))   # [B, 64, H/8,  W/8]
    x5 = self.stage5(self.down45(x4))   # [B,128, H/16, W/16]
    return x1, x2, x3, x4, x5


# ---------------------------------------------------------------------------
# Multi-task model: segmentation + classification (MT_* family)
# ---------------------------------------------------------------------------
class MultiTaskResUNet(nn.Module):
    def __init__(self, in_channels=1, bilinear=True, dropout=0.3, head="gap",
                 stage4_layers=4, stage5_layers=5,
                 cbam_mode="parallel", cbam_in_state=True,
                 use_mask_guidance=False):
        super().__init__()
        _attach_encoder(self, in_channels, stage4_layers, stage5_layers)

        self.up1 = UpRes(128, 64, 64, bilinear=bilinear, num_res_blocks=2)      # H/8
        self.up2 = UpRes(64,  32, 32, bilinear=bilinear, num_res_blocks=2)      # H/4
        self.up3 = UpRes(32,  16, 16, bilinear=bilinear, num_res_blocks=2)      # H/2
        self.up4 = UpRes(16,   8,  8, bilinear=bilinear, num_res_blocks=3)      # H

        # cbam_in_state controls whether CBAM params exist in the state_dict
        # (mirrors the difference in the original _wo variants)
        if cbam_in_state:
            self.cbam4 = CBAM(64, reduction=16, spatial_kernel=7)
            self.cbam5 = CBAM(128, reduction=16, spatial_kernel=7)
            self.cbam_u1 = CBAM(64, reduction=16, spatial_kernel=7)
        else:
            self.register_module("cbam4", None)
            self.register_module("cbam5", None)
            self.register_module("cbam_u1", None)
        self.cbam_mode = cbam_mode

        feature_cfg = dict(c1=64, c2=128, c3=64, dim=128, dropout=dropout)
        if head == "gap":
            self.cls_head = GlobalPoolingClassifier(**feature_cfg)
        elif head == "roi_gap":
            self.cls_head = ROIAwareMultiScaleClassifier(mode="roi_gap", **feature_cfg)
        elif head == "hybrid":
            self.cls_head = ROIAwareMultiScaleClassifier(mode="hybrid", **feature_cfg)
        else:
            raise ValueError(f"Unknown head type: {head}")
        self.head = head
        self.use_mask_guidance = use_mask_guidance
        self.seg_head = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x, return_cls=True):
        x1 = self.stage1(x)                 # [B,  8, H,    W]
        x2 = self.stage2(self.down12(x1))   # [B, 16, H/2,  W/2]
        x3 = self.stage3(self.down23(x2))   # [B, 32, H/4,  W/4]
        x4 = self.stage4(self.down34(x3))   # [B, 64, H/8,  W/8]
        x5 = self.stage5(self.down45(x4))   # [B,128, H/16, W/16]

        if self.cbam_mode == "inplace":
            # _shcbam: CBAM replaces x4 in place, flowing into stage5 / up1 skip
            x4 = self.cbam4(x4)
            x5 = self.stage5(self.down45(x4))
            x5 = self.cbam5(x5)
            u1 = self.up1(x5, x4)
            u1 = self.cbam_u1(u1)
            c4, c5, cu1 = x4, x5, u1
        elif self.cbam_mode == "parallel":
            # Native MT_*: CBAM acts on a copy; the decode path uses the raw x4
            x5 = self.stage5(self.down45(x4))
            c4 = self.cbam4(x4)
            c5 = self.cbam5(x5)
            u1 = self.up1(x5, x4)
            cu1 = self.cbam_u1(u1)
        else:  # "none" (_wo)
            x5 = self.stage5(self.down45(x4))
            u1 = self.up1(x5, x4)
            c4, c5, cu1 = x4, x5, u1

        u2 = self.up2(u1, x3)
        u3 = self.up3(u2, x2)
        u4 = self.up4(u3, x1)
        seg_logits = self.seg_head(u4)          # [B,1,H,W]

        if not return_cls:
            return {"seg_logits": seg_logits, "cls_logits": None}

        if self.use_mask_guidance:
            cls_logits = self.cls_head(c4, c5, cu1, seg_logits.detach())
        else:
            cls_logits = self.cls_head(c4, c5, cu1)

        return {"seg_logits": seg_logits, "cls_logits": cls_logits}


# ---------------------------------------------------------------------------
# Single-task classifier (MS_Net / Baseline family)
# ---------------------------------------------------------------------------
class SingleTaskClassifier(nn.Module):
    def __init__(self, in_channels=1, bilinear=True, dropout=0.3,
                 stage4_layers=3, stage5_layers=4, cbam_mode="none",
                 cbam_in_state=True, feats=("x4", "x5", "u1")):
        super().__init__()
        _attach_encoder(self, in_channels, stage4_layers, stage5_layers)

        # MS_Net has cbam4/5 + cbam_u1 (incl. up1); Baseline only has cbam4/5, no up1
        if cbam_in_state:
            self.cbam4 = CBAM(64, reduction=16, spatial_kernel=7)
            self.cbam5 = CBAM(128, reduction=16, spatial_kernel=7)
        else:
            self.register_module("cbam4", None)
            self.register_module("cbam5", None)
        self.cbam_mode = cbam_mode
        self.feats = tuple(feats)

        if self.feats == ("x4", "x5", "u1"):
            self.up1 = UpRes(128, 64, 64, bilinear=bilinear, num_res_blocks=2)
            if cbam_in_state:
                self.cbam_u1 = CBAM(64, reduction=16, spatial_kernel=7)
            total_dim = 64 + 128 + 64
        elif feats == ("x5",):
            total_dim = 128
        else:
            raise ValueError(f"Unsupported feats: {feats}")

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(self.down12(x1))
        x3 = self.stage3(self.down23(x2))
        x4 = self.stage4(self.down34(x3))
        x5 = self.stage5(self.down45(x4))

        if self.feats == ("x5",):
            # Baseline: uses only the x5 feature (128 channels).
            # The native baseline applies CBAM in place
            if self.cbam_mode == "inplace":
                x4 = self.cbam4(x4)
                x5 = self.stage5(self.down45(x4))
                x5 = self.cbam5(x5)
            feat = self.gap(x5).flatten(1)      # [B,128]
        else:
            # MS_Net family
            if self.cbam_mode == "inplace":
                # _sh: CBAM replaces in place; stage5/u1 use CBAM'd x4, and
                # classification uses the CBAM'd features
                x4 = self.cbam4(x4)
                x5 = self.stage5(self.down45(x4))
                x5 = self.cbam5(x5)
                u1 = self.up1(x5, x4)
                u1 = self.cbam_u1(u1)
                fx4 = self.gap(x4).flatten(1)
                fx5 = self.gap(x5).flatten(1)
                fu1 = self.gap(u1).flatten(1)
            else:
                # Native MS_Net: the CBAM output is discarded (a quirk of the
                # original implementation); classification uses GAP of the raw
                # features
                x5 = self.stage5(self.down45(x4))
                u1 = self.up1(x5, x4)
                if self.cbam_mode == "parallel":
                    self.cbam4(x4); self.cbam5(x5); self.cbam_u1(u1)  # executed only, not used
                fx4 = self.gap(x4).flatten(1)
                fx5 = self.gap(x5).flatten(1)
                fu1 = self.gap(u1).flatten(1)
            feat = torch.cat([fx4, fx5, fu1], dim=1)  # [B,256]

        cls_logits = self.classifier(feat)      # [B,1]
        return {"cls_logits": cls_logits}


# ---------------------------------------------------------------------------
# Model registry: name -> config
# ---------------------------------------------------------------------------
def _parse_suffix(name):
    """Identify the _shcbam / _sh / _wo suffix. Returns (base_name, suffix)."""
    for s in ("_shcbam", "_sh", "_wo"):
        if name.endswith(s):
            return name[: -len(s)], s
    return name, ""


def resolve_model_cfg(name, extra=None):
    base, suffix = _parse_suffix(name)
    cfg = {"name": name, "task": "multi"}

    if base in ("MT_Net", "MT_MP", "MT_MIX"):
        head, stage45 = {
            "MT_Net": ("gap", (4, 5)),
            "MT_MP": ("roi_gap", (3, 4)),
            "MT_MIX": ("hybrid", (3, 4)),
        }[base]
        cfg.update(head=head, stage4_layers=stage45[0], stage5_layers=stage45[1],
                   task="multi", use_mask_guidance=(head != "gap"))
        if suffix == "_wo":
            cfg["cbam_mode"] = "none"
            # _wo cbam param presence: MT_MIX_wo keeps them, MT_Net_wo / MT_MP_wo don't
            cfg["cbam_in_state"] = (base == "MT_MIX")
        elif suffix in ("_shcbam", "_sh"):
            cfg["cbam_mode"] = "inplace"
            cfg["cbam_in_state"] = True
        else:
            cfg["cbam_mode"] = "parallel"
            cfg["cbam_in_state"] = True

    elif base == "MS_Net":
        cfg.update(task="single",
                   stage4_layers=3, stage5_layers=4,
                   feats=("x4", "x5", "u1"))
        if suffix == "_wo":
            cfg.update(cbam_mode="none", cbam_in_state=False)
        else:
            # MS_Net_sh applies CBAM in place; native MS_Net discards the CBAM
            # output (classification uses the raw features)
            cfg.update(cbam_mode="inplace" if suffix == "_sh" else "parallel",
                       cbam_in_state=True)
    elif base == "baseline":
        cfg.update(task="single",
                   stage4_layers=3, stage5_layers=4,
                   feats=("x5",))
        if suffix == "_wo":
            cfg.update(cbam_mode="none", cbam_in_state=False)
        else:
            cfg.update(cbam_mode="inplace", cbam_in_state=True)
    else:
        raise ValueError(f"Unknown model name: {name}. Valid: {ALL_MODEL_NAMES}")

    if extra:
        cfg.update(extra)
    return cfg


ALL_MODEL_NAMES = [
    "MT_Net", "MT_Net_shcbam", "MT_Net_wo",
    "MT_MP", "MT_MP_shcbam", "MT_MP_wo",
    "MT_MIX", "MT_MIX_shcbam", "MT_MIX_wo",
    "MS_Net", "MS_Net_sh", "MS_Net_wo",
    "baseline", "baseline_wo",
]


def build_model(name, in_channels=1, bilinear=True, dropout=0.3, extra=None,
                yaml_cfg=None):
    """Build a model instance by name.

    yaml_cfg: optional dict from configs/models/<name>.yaml (its ``model``
              section), used to override default configuration (e.g. cbam_mode /
              stage layer counts).
    """
    cfg = resolve_model_cfg(name, extra)
    if yaml_cfg:
        from ..config import _merge
        cfg = _merge(cfg, yaml_cfg.get("model", yaml_cfg))
    if "feats" in cfg and not isinstance(cfg["feats"], tuple):
        cfg["feats"] = tuple(cfg["feats"])
    if cfg["task"] == "multi":
        return MultiTaskResUNet(
            in_channels=in_channels, bilinear=bilinear, dropout=dropout,
            head=cfg["head"],
            stage4_layers=cfg["stage4_layers"], stage5_layers=cfg["stage5_layers"],
            cbam_mode=cfg.get("cbam_mode", "parallel"),
            cbam_in_state=cfg.get("cbam_in_state", True),
            use_mask_guidance=cfg.get("use_mask_guidance", False),
        )
    return SingleTaskClassifier(
        in_channels=in_channels, bilinear=bilinear, dropout=dropout,
        stage4_layers=cfg["stage4_layers"], stage5_layers=cfg["stage5_layers"],
        cbam_mode=cfg.get("cbam_mode", "none"),
        cbam_in_state=cfg.get("cbam_in_state", True),
        feats=cfg.get("feats", ("x4", "x5", "u1")),
    )
