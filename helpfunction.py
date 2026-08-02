import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix
)

def tensor_to_gray_numpy(img_tensor):
    """
    img_tensor: [1,H,W] or [H,W]
    return: [H,W] float, 0~1
    """
    if img_tensor.ndim == 3:
        img = img_tensor[0].detach().cpu().numpy()
    else:
        img = img_tensor.detach().cpu().numpy()

    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)
    return img


def overlay_mask_on_gray(gray, mask, alpha=0.4):
    """
    gray: [H,W] 0~1
    mask: [H,W] bool or 0/1
    return: RGB [H,W,3]
    """
    rgb = np.stack([gray, gray, gray], axis=-1)
    overlay = rgb.copy()

    mask = mask.astype(bool)

    overlay[mask, 0] = (1 - alpha) * overlay[mask, 0] + alpha * 0.0
    overlay[mask, 1] = (1 - alpha) * overlay[mask, 1] + alpha * 1.0
    overlay[mask, 2] = (1 - alpha) * overlay[mask, 2] + alpha * 0.0

    return np.clip(overlay, 0, 1)


def overlay_cam_on_gray(gray, cam, alpha=0.45):
    """
    gray: [H,W] 0~1
    cam:  [H,W] 0~1
    return: RGB [H,W,3]
    """
    cmap = plt.get_cmap("jet")
    heat = cmap(cam)[..., :3]  # RGB

    base = np.stack([gray, gray, gray], axis=-1)
    out = (1 - alpha) * base + alpha * heat
    out = np.clip(out, 0, 1)
    return out

def compute_cls_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    y_pred = (y_prob >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = np.nan

    try:
        f1 = f1_score(y_true, y_pred)
    except:
        f1 = np.nan

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    except:
        sensitivity = np.nan
        specificity = np.nan

    return {
        "acc": acc,
        "auc": auc,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity
    }


def show_train_samples(dataset, num_samples=3):
    plt.figure(figsize=(12, 4 * num_samples))

    for i in range(num_samples):
        sample = dataset[i]
        img = sample["image"][0].numpy()   # [H, W]
        msk = sample["mask"][0].numpy()    # [H, W]
        lbl = sample["label"].item()

        plt.subplot(num_samples, 3, 3*i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"Image | label={lbl}")
        plt.axis("off")

        plt.subplot(num_samples, 3, 3*i + 2)
        plt.imshow(msk, cmap="gray")
        plt.title(f"Mask | has_mask={sample['has_mask'].item()}")
        plt.axis("off")

        plt.subplot(num_samples, 3, 3*i + 3)
        plt.imshow(img, cmap="gray")
        plt.imshow(msk, cmap="Greens", alpha=0.35)
        plt.title("Overlay")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def compute_seg_metrics(pred_mask, true_mask, eps=1e-8):

    pred_mask = (pred_mask > 0.5).astype(np.float32)
    true_mask = (true_mask > 0.5).astype(np.float32)

    intersection = np.sum(pred_mask * true_mask)

    dice = (2.0 * intersection + eps) / (
        np.sum(pred_mask) + np.sum(true_mask) + eps
    )

    jaccard = (intersection + eps) / (
        np.sum(pred_mask) +
        np.sum(true_mask) -
        intersection + eps
    )

    return dice, jaccard





def plot_loss_curve(history_df, fold_idx, save_dir):
    plt.figure()

    plt.plot(history_df["train_cls_loss"], label="Train CLS Loss")
    plt.plot(history_df["val_cls_loss"], label="Val CLS Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Classification Loss")
    plt.title(f"Fold {fold_idx+1} CLS Loss Curve")
    plt.legend()

    plt.savefig(os.path.join(save_dir, f"fold{fold_idx+1}_cls_loss.png"))
    plt.close()