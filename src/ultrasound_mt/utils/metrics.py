"""Segmentation evaluation metrics: Dice and IoU."""
import numpy as np
import torch


def dice_iou(pred, gt):
    """Compute Dice and IoU for a single sample. Input is a 0/1 array."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    dice = (2 * inter + 1e-7) / (pred.sum() + gt.sum() + 1e-7)
    union = np.logical_or(pred, gt).sum()
    iou = (inter + 1e-7) / (union + 1e-7)
    return dice, iou


def evaluate_segmentation(model, loader, device, threshold=0.5):
    """Evaluate mean segmentation Dice / IoU over a data loader.

    The model must output ``seg_logits`` ([B,1,H,W]), and each loader batch must contain ``image`` and ``mask``.
    """
    dice_list, iou_list = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            mask = batch["mask"]
            output = model(image)
            seg_logits = output["seg_logits"]
            pred = torch.sigmoid(seg_logits)
            pred = (pred > threshold).cpu().numpy()
            mask = mask.numpy()
            for p, g in zip(pred, mask):
                d, i = dice_iou(p[0], g[0])
                dice_list.append(d)
                iou_list.append(i)
    return float(np.mean(dice_list)), float(np.mean(iou_list))
