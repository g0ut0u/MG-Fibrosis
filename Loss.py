import torch
import torch.nn as nn
import torch.nn.functional as F
def dice_loss_with_logits(logits, targets, smooth=1e-5):
    probs = torch.sigmoid(logits)

    probs = probs.contiguous().view(probs.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()

class WeightedFocalWithLogitsLoss(nn.Module):
    def __init__(self, alpha=273/624, gamma=1, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()

        return loss



#Loss=lambda_seg * Dice + lambda_cls * focal_cls
class MultiTaskLoss(nn.Module):
    def __init__(self,lambda_seg=1.0,lambda_cls=1.0,cls_alpha=273/624,cls_gamma=0.5):
        super().__init__()

        self.lambda_seg = lambda_seg
        self.lambda_cls = lambda_cls

        self.cls_focal = WeightedFocalWithLogitsLoss(alpha=cls_alpha,gamma=cls_gamma)

    def forward(self, outputs, batch):

        seg_logits = outputs["seg_logits"]     # [B,1,H,W]
        cls_logits = outputs["cls_logits"]     # [B,1]

        masks = batch["mask"].float()          # [B,1,H,W]
        labels = batch["label"].float().unsqueeze(1)  # [B,1]

        # ===== classification =====
        cls_loss = self.cls_focal(cls_logits, labels)

        # ===== segmentation (Dice only) =====
        if "has_mask" in batch:

            has_mask = batch["has_mask"]  # bool tensor

            if has_mask.any():
                seg_logits_valid = seg_logits[has_mask]
                masks_valid = masks[has_mask]
                seg_loss = dice_loss_with_logits(seg_logits_valid,masks_valid)
            else:
                seg_loss = torch.tensor(0.0,device=seg_logits.device)

        else:
            seg_loss = dice_loss_with_logits(seg_logits, masks)

        total_loss = (self.lambda_seg * seg_loss+ self.lambda_cls * cls_loss)

        return {
            "loss": total_loss,
            "seg_loss": seg_loss.detach(),
            "cls_loss": cls_loss.detach()
        }