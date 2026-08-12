"""Multi-task (segmentation + classification) 5-fold cross-validation training script.

Reads configs.yaml (data + model + train) and performs two-stage curriculum learning
(Phase 1 segmentation-only pretraining -> Phase 2 joint training).

Usage (from the repository root):
    python -m ultrasound_mt.train.train_multitask \\
        --model MT_MIX \\
        --data-config configs/data.yaml \\
        --train-config configs/train_multitask.yaml
"""
import argparse
import gc
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import load_config, resolve_model_yaml, CONFIGS_DIR
from ..data import MultiTaskUltrasoundDataset
from ..losses import MultiTaskLoss
from ..models import build_model, ALL_MODEL_NAMES
from ..utils import plot_loss_curve


def parse_args():
    p = argparse.ArgumentParser(description="Multi-task (classification + segmentation) 5-fold training")
    p.add_argument("--model", default="MT_MIX", help="Model name, e.g. MT_MIX / MT_MP / MT_Net")
    p.add_argument("--data-config", default=os.path.join(CONFIGS_DIR, "data.yaml"))
    p.add_argument("--train-config", default=os.path.join(CONFIGS_DIR, "train_multitask.yaml"))
    p.add_argument("--out-root", default=None, help="Override the output root directory")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics_with_probs(cls_probs, cls_labels, threshold=0.5):
    cls_preds = (cls_probs >= threshold).astype(int)
    acc = accuracy_score(cls_labels, cls_preds)
    auc = roc_auc_score(cls_labels, cls_probs)
    f1 = f1_score(cls_labels, cls_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(cls_labels, cls_preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return acc, auc, f1, sens, spec


def train_one_epoch(model, dataloader, optimizer, criterion,
                    lambda_seg, lambda_cls, device, phase):
    model.train()
    total_loss = total_seg = total_cls = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        images = batch["image"].to(device)
        batch["mask"] = batch["mask"].to(device)
        batch["label"] = batch["label"].to(device)
        if "has_mask" in batch:
            batch["has_mask"] = batch["has_mask"].to(device)

        optimizer.zero_grad()
        if phase == "seg_only":
            outputs = model(images, return_cls=False)
            outputs["cls_logits"] = torch.zeros(images.size(0), 1).to(device)
        else:
            outputs = model(images, return_cls=True)

        criterion.lambda_seg = lambda_seg
        criterion.lambda_cls = lambda_cls if phase != "seg_only" else 0.0

        loss_dict = criterion(outputs, batch)
        loss_dict["loss"].backward()
        optimizer.step()

        n = images.size(0)
        total_loss += loss_dict["loss"].item() * n
        total_seg += loss_dict["seg_loss"].item() * n
        if loss_dict.get("cls_loss") is not None:
            total_cls += loss_dict["cls_loss"].item() * n

    num = len(dataloader.dataset)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    return (total_loss / num, total_seg / num, total_cls / num)


def validate_loss(model, dataloader, criterion, device):
    model.eval()
    total_cls_loss = 0.0
    criterion.lambda_seg = 0.0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            nb = {"image": images, "label": batch["label"].to(device)}
            if "mask" in batch:
                nb["mask"] = batch["mask"].to(device)
            if "has_mask" in batch:
                nb["has_mask"] = batch["has_mask"].to(device)
            outputs = model(images, return_cls=True)
            total_cls_loss += criterion(outputs, nb)["cls_loss"].item() * images.size(0)
    return total_cls_loss / len(dataloader.dataset)


def validate_and_collect(model, dataloader, device):
    model.eval()
    probs, labels, pids, paths = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch["image"].to(device)
            labels.extend(batch["label"].cpu().numpy().reshape(-1))
            pids.extend(batch["patient_id"])
            paths.extend(batch["image_path"])
            outputs = model(images)
            probs.extend(torch.sigmoid(outputs["cls_logits"]).cpu().numpy().flatten())

    probs = np.array(probs)
    labels = np.array(labels)
    acc, auc, f1, sens, spec = compute_metrics_with_probs(probs, labels)
    results_df = pd.DataFrame({
        "image_path": paths, "patient_id": pids, "true_label": labels,
        "pred_prob": probs, "pred_label": (probs >= 0.5).astype(int),
    })
    return ({"acc": acc, "auc": auc, "f1": f1, "sensitivity": sens,
             "specificity": spec, "probs": probs, "labels": labels}, results_df)


def train_model_for_fold(model_name, device, cfg, train_samples, val_samples, fold_idx):
    d = cfg["data"]
    t = cfg["train"]
    save_csv_dir = cfg["paths"]["save_csv_dir"]
    save_model_dir = cfg["paths"]["save_model_dir"]
    os.makedirs(save_csv_dir, exist_ok=True)
    os.makedirs(save_model_dir, exist_ok=True)

    p1_lambda_seg = t["phase1_lambda_seg"]
    p1_lambda_cls = t.get("phase1_lambda_cls", 0.0)
    p2_lambda_seg = t["phase2_lambda_seg"]
    p2_lambda_cls = t.get("phase2_lambda_cls", 0.2)

    train_dataset = MultiTaskUltrasoundDataset(
        root_dir=cfg["paths"]["dataset_dir"], image_size=d["image_size"],
        require_mask=True,
        use_translation_aug=d.get("use_translation_aug", True),
        use_rotation_aug=d.get("use_rotation_aug", True),
        max_shift_x=d.get("max_shift_x", 20), max_shift_y=d.get("max_shift_y", 20),
        max_rotate_angle=d.get("max_rotate_angle", 10),
        translation_p=d.get("translation_p", 1.0), rotation_p=d.get("rotation_p", 1.0),
        aug_repeat_count=d.get("aug_repeat_count", 2))
    train_dataset.samples = train_samples

    val_dataset = MultiTaskUltrasoundDataset(
        root_dir=cfg["paths"]["dataset_dir"], image_size=d["image_size"],
        require_mask=False, use_translation_aug=False, use_rotation_aug=False,
        aug_repeat_count=1)
    val_dataset.samples = val_samples

    train_loader = DataLoader(train_dataset, batch_size=d["batch_size"], shuffle=True,
                              num_workers=d.get("num_workers", 0),
                              pin_memory=d.get("pin_memory", True))
    val_loader = DataLoader(val_dataset, batch_size=d["batch_size"], shuffle=False,
                            num_workers=d.get("num_workers", 0),
                            pin_memory=d.get("pin_memory", True))

    model = build_model(model_name, in_channels=d.get("in_channels", 1),
                        bilinear=cfg["model"].get("bilinear", True),
                        dropout=cfg["model"].get("dropout", 0.3),
                        yaml_cfg={"model": cfg["model"]}).to(device)
    criterion = MultiTaskLoss(lambda_seg=p1_lambda_seg, lambda_cls=p1_lambda_cls,
                              cls_alpha=t.get("cls_alpha", 273 / 624),
                              cls_gamma=t.get("cls_gamma", 0.5))
    history = {"train_cls_loss": [], "val_cls_loss": []}
    phase1_epochs = t["phase1_epochs"]
    phase2_epochs = t.get("phase2_epochs", 0)

    # ---- Phase 1: segmentation only ----
    opt1 = optim.AdamW(model.parameters(), lr=t["phase1_lr"], weight_decay=t.get("weight_decay", 1e-4))
    for epoch in range(1, phase1_epochs + 1):
        tt = (epoch - 1) / (phase1_epochs - 1) if phase1_epochs > 1 else 0.0
        cur_lr = t["phase1_lr"] + (t["phase1_lr"] / 2 - t["phase1_lr"]) * tt
        for g in opt1.param_groups:
            g["lr"] = cur_lr
        train_loss, seg_loss, _ = train_one_epoch(
            model, train_loader, opt1, criterion, p1_lambda_seg, p1_lambda_cls,
            device, phase="seg_only")
        if epoch % 10 == 0 or epoch == phase1_epochs:
            vm, _ = validate_and_collect(model, val_loader, device)
            print(f"Epoch {epoch:2d}/{phase1_epochs} | Train Loss: {train_loss:.4f} | "
                  f"Val Acc: {vm['acc']:.4f} AUC: {vm['auc']:.4f}")
        else:
            print(f"Epoch {epoch:2d}/{phase1_epochs} | Train Loss: {train_loss:.4f}")

    # ---- Phase 2: joint training (if epochs > 0) ----
    if phase2_epochs > 0:
        opt2 = optim.AdamW(model.parameters(), lr=t["phase2_lr"], weight_decay=t.get("weight_decay", 1e-4))
        for epoch in range(1, phase2_epochs + 1):
            tt = (epoch - 1) / (phase2_epochs - 1) if phase2_epochs > 1 else 0.0
            cur_lr = t["phase2_lr"] + (t["phase2_lr"] / 2 - t["phase2_lr"]) * tt
            cur_lambda_seg = 1.0 + (p2_lambda_seg - 1.0) * tt
            cur_lambda_cls = 1.0 - cur_lambda_seg
            for g in opt2.param_groups:
                g["lr"] = cur_lr
            train_loss, _, train_cls = train_one_epoch(
                model, train_loader, opt2, criterion, cur_lambda_seg, cur_lambda_cls,
                device, phase="joint")
            val_cls = validate_loss(model, val_loader, criterion, device)
            history["train_cls_loss"].append(train_cls)
            history["val_cls_loss"].append(val_cls)
            vm, _ = validate_and_collect(model, val_loader, device)
            print(f"[Phase2] Epoch {epoch:2d}/{phase2_epochs} | Train CLS: {train_cls:.4f} | "
                  f"Val CLS: {val_cls:.4f} | Val Acc: {vm['acc']:.4f} AUC: {vm['auc']:.4f} "
                  f"F1: {vm['f1']:.4f} Sens: {vm['sensitivity']:.4f} Spec: {vm['specificity']:.4f}")

    pd.DataFrame(history).to_csv(os.path.join(save_csv_dir, f"fold{fold_idx+1}_loss_curve.csv"),
                                 index=False)
    plot_loss_curve(pd.DataFrame(history), fold_idx, save_csv_dir)

    model_save_path = os.path.join(save_model_dir, f"{model_name}_fold{fold_idx+1}.pth")
    torch.save({"fold": fold_idx + 1, "model_state_dict": model.state_dict()}, model_save_path)
    print(f"Model saved to: {model_save_path}")

    final_metrics, results_df = validate_and_collect(model, val_loader, device)
    error_df = results_df[results_df["true_label"] != results_df["pred_label"]].copy()
    error_df.insert(0, "fold", fold_idx + 1)
    return final_metrics, error_df, results_df


def main():
    args = parse_args()
    cfg = load_config(
        data_cfg=args.data_config,
        model_cfg=resolve_model_yaml(args.model),
        train_cfg=args.train_config,
    )
    model_name = cfg["model"].get("name", args.model)
    if model_name not in ALL_MODEL_NAMES:
        raise ValueError(f"Unknown model: {model_name}. Valid: {ALL_MODEL_NAMES}")

    # Output directory
    out = args.out_root or cfg["paths"].get("output_dir", "/data/Li/results")
    cfg["paths"]["save_csv_dir"] = os.path.join(out, f"{model_name}_results")
    cfg["paths"]["save_model_dir"] = os.path.join(out, f"{model_name}_models")
    os.makedirs(cfg["paths"]["save_csv_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["save_model_dir"], exist_ok=True)

    seed = args.seed or cfg["train"].get("random_state", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {model_name} | Seed: {seed} | Device: {device}")

    d = cfg["data"]
    temp = MultiTaskUltrasoundDataset(
        root_dir=cfg["paths"]["dataset_dir"], image_size=d["image_size"],
        require_mask=False, use_translation_aug=False, use_rotation_aug=False,
        aug_repeat_count=1)
    samples_original = temp.samples

    patient_to_label = {}
    for s in samples_original:
        if s["patient_id"] not in patient_to_label:
            patient_to_label[s["patient_id"]] = s["label"]
    patient_ids = list(patient_to_label.keys())
    patient_labels = [patient_to_label[p] for p in patient_ids]
    print(f"Total original images: {len(samples_original)} | Patients: {len(patient_ids)}")

    n_splits = cfg["train"]["n_splits"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    all_metrics, all_errors, all_preds = [], [], []

    for fold_idx, (train_pidx, val_pidx) in enumerate(skf.split(patient_ids, patient_labels)):
        train_patients = set(patient_ids[i] for i in train_pidx)
        val_patients = set(patient_ids[i] for i in val_pidx)
        train_samples = [s for s in samples_original if s["patient_id"] in train_patients]
        val_samples = [s for s in samples_original if s["patient_id"] in val_patients]
        print(f"\n{'='*50}\nFold {fold_idx+1}/{n_splits}\n{'='*50}")

        metrics, error_df, results_df = train_model_for_fold(
            model_name, device, cfg, train_samples, val_samples, fold_idx)
        metrics["fold"] = fold_idx + 1
        all_metrics.append(metrics)
        all_errors.append(error_df)
        all_preds.append(results_df)

    metrics_df = pd.DataFrame([
        {"fold": m["fold"], "Accuracy": m["acc"], "AUC": m["auc"], "F1": m["f1"],
         "Sensitivity": m["sensitivity"], "Specificity": m["specificity"]}
        for m in all_metrics])
    print("\nPerformance (each fold):\n", metrics_df.to_string(index=False))
    mean_metrics = metrics_df.drop("fold", axis=1).mean()
    std_metrics = metrics_df.drop("fold", axis=1).std()
    print("\nMean Performance (5-fold):")
    for metric in mean_metrics.index:
        print(f"{metric}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

    all_preds_with_fold = []
    for i, df in enumerate(all_preds):
        df_copy = df.copy()
        df_copy.insert(0, "fold", i + 1)
        all_preds_with_fold.append(df_copy)
    pd.concat(all_preds_with_fold, ignore_index=True).to_csv(
        os.path.join(cfg["paths"]["save_csv_dir"], "all_predictions.csv"), index=False)


if __name__ == "__main__":
    main()
