import os
import random
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from helpfunction import plot_loss_curve

from MT_MP import MTMP
from MT_Net import MTNet
from MT_MIX import MTMIX
from MT_MP_shcbam import MTMP_sh
from MT_Net_shcbam import MTNet_sh
from MT_MIX_shcbam import MTMIX_sh
from MT_MP_wo import MTMP_wo
from MT_Net_wo import MTNet_wo
from MT_MIX_wo import MTMIX_wo
from dataloader import MultiTaskUltrasoundDataset
from Loss import MultiTaskLoss
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

#use different model_name for different models

#model_name='MT_MP'
#model_name='MT_MIX'
#model_name='MT_Net'
#model_name='MT_MP_shcbam'
#model_name='MT_MIX_shcbam'
#model_name='MT_Net_shcbam'
#model_name='MT_MP_wo'
#model_name='MT_MIX_wo'
model_name='MT_Net_wo'
result_root = config["paths"]["result_root"]
root_dir = config["dataset"]["root_dir"]
image_size = (768, 1024)          # (H, W)
batch_size = 48
num_workers = 0
pin_memory = True

use_translation_aug = True
use_rotation_aug = True
max_shift_x = 20
max_shift_y = 20
max_rotate_angle = 10
translation_p = 1
rotation_p = 1
aug_repeat_count = 2            

phase1_epochs = 45       
phase1_lr = 8e-4
phase1_lambda_seg = 1.0
phase1_lambda_cls = 0

phase2_epochs = 15          
phase2_lr = 2e-4
phase2_lambda_seg = 0.8
phase2_lambda_cls = 0.2

n_splits = 5
random_state = 42

save_csv_dir = os.path.join(result_root,f"{model_name}_results")
os.makedirs(save_csv_dir,exist_ok=True)

save_model_dir = os.path.join(result_root,f"{model_name}_models")
os.makedirs(save_model_dir,exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(random_state)
print(f"Random seed set to {random_state}")

def compute_metrics_with_probs(cls_probs, cls_labels, threshold=0.5):
    cls_preds = (cls_probs >= threshold).astype(int)
    acc = accuracy_score(cls_labels, cls_preds)
    auc = roc_auc_score(cls_labels, cls_probs)
    f1 = f1_score(cls_labels, cls_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(cls_labels, cls_preds, labels=[0,1]).ravel()
    sensitivity = tp / (tp + fn) if (tp+fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn+fp) > 0 else 0.0
    return acc, auc, f1, sensitivity, specificity

def train_one_epoch(model, dataloader, optimizer, criterion, lambda_seg, lambda_cls, device, phase='joint'):
    model.train()
    total_loss = 0.0
    total_seg_loss = 0.0
    total_cls_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        labels = batch["label"].to(device)

        batch["mask"] = masks
        batch["label"] = labels
        if "has_mask" in batch:
            batch["has_mask"] = batch["has_mask"].to(device)

        optimizer.zero_grad()

        if phase == 'seg_only':
            outputs = model(images, return_cls=False)
            outputs["cls_logits"] = torch.zeros(images.size(0), 1).to(device)
        else:
            outputs = model(images, return_cls=True)

        criterion.lambda_seg = lambda_seg
        criterion.lambda_cls = lambda_cls if phase != 'seg_only' else 0.0

        loss_dict = criterion(outputs, batch)
        loss = loss_dict["loss"]

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_seg_loss += loss_dict["seg_loss"].item() * images.size(0)

        cls_loss_val = loss_dict.get("cls_loss")
        if cls_loss_val is not None:
            total_cls_loss += cls_loss_val.item() * images.size(0)

    num_samples = len(dataloader.dataset)
    avg_loss = total_loss / num_samples
    avg_seg_loss = total_seg_loss / num_samples
    avg_cls_loss = total_cls_loss / num_samples

    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()

    return avg_loss, avg_seg_loss, avg_cls_loss


def validate_loss(model, dataloader, criterion, lambda_seg, lambda_cls, device):
    model.eval()
    total_cls_loss = 0.0

    criterion.lambda_cls = lambda_cls
    criterion.lambda_seg = 0.0   
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            new_batch = {
                "image": images,
                "label": labels
            }

            if "mask" in batch:
                new_batch["mask"] = batch["mask"].to(device)

            if "has_mask" in batch:
                new_batch["has_mask"] = batch["has_mask"].to(device)

            outputs = model(images, return_cls=True)
            loss_dict = criterion(outputs, new_batch)

            cls_loss_val = loss_dict.get("cls_loss")
            if cls_loss_val is not None:
                total_cls_loss += cls_loss_val.item() * images.size(0)

    n = len(dataloader.dataset)
    return total_cls_loss / n   

def validate_and_collect(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []
    all_patient_ids = []
    all_image_paths = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy().reshape(-1)
            patient_ids = list(batch["patient_id"])
            image_paths = list(batch["image_path"])

            outputs = model(images)
            cls_logits = outputs["cls_logits"]
            probs = torch.sigmoid(cls_logits).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_labels.extend(labels)
            all_patient_ids.extend(patient_ids)
            all_image_paths.extend(image_paths)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    acc, auc, f1, sens, spec = compute_metrics_with_probs(all_probs, all_labels)

    results_df = pd.DataFrame({
        "image_path": all_image_paths,
        "patient_id": all_patient_ids,
        "true_label": all_labels,
        "pred_prob": all_probs,
        "pred_label": (all_probs >= 0.5).astype(int)
    })

    metrics = {
        "acc": acc, "auc": auc, "f1": f1,
        "sensitivity": sens, "specificity": spec,
        "probs": all_probs, "labels": all_labels
    }
    return metrics, results_df




def train_model_for_fold(train_samples, val_samples, fold_idx):
    train_dataset = MultiTaskUltrasoundDataset(
        root_dir=root_dir, image_size=image_size,
        require_mask=True,
        use_translation_aug=use_translation_aug,
        use_rotation_aug=use_rotation_aug,
        max_shift_x=max_shift_x, max_shift_y=max_shift_y,
        max_rotate_angle=max_rotate_angle,
        translation_p=translation_p, rotation_p=rotation_p,
        aug_repeat_count=aug_repeat_count
    )
    train_dataset.samples = train_samples
    
    val_dataset = MultiTaskUltrasoundDataset(
        root_dir=root_dir, image_size=image_size,
        require_mask=False,
        use_translation_aug=False, use_rotation_aug=False,
        aug_repeat_count=1
    )
    val_dataset.samples = val_samples

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    history = {
        "train_cls_loss": [],
        "val_cls_loss": [],
    }
    if model_name == 'MT_MP':
        model = MTMP(in_channels=1, bilinear=True, dropout=0.3).to(device)
    elif model_name == 'MT_MIX':
        model = MTMIX(in_channels=1, bilinear=True, dropout=0.3).to(device)
    elif model_name == 'MT_Net':
        model = MTNet(in_channels=1, bilinear=True, dropout=0.3).to(device)
    elif model_name == 'MT_MP_shcbam':
        model = MTMP_sh(in_channels=1, bilinear=True, dropout=0.3).to(device)
    elif model_name == 'MT_MIX_cbam':
        model = MTMIX_sh(in_channels=1, bilinear=True, dropout=0.3).to(device)
    elif model_name == 'MT_Net_cbam':
        model = MTNet_sh(in_channels=1, bilinear=True, dropout=0.3).to(device)
    elif model_name == 'MT_MP_wo':
        model = MTMP_wo(in_channels=1, bilinear=True, dropout=0.3).to(device)
    elif model_name == 'MT_MIX_wo':
        model = MTMIX_wo(in_channels=1, bilinear=True, dropout=0.3).to(device)
    elif model_name == 'MT_Net_wo':
        model = MTNet_wo(in_channels=1, bilinear=True, dropout=0.3).to(device)
    criterion = MultiTaskLoss(lambda_seg=phase1_lambda_seg, lambda_cls=phase1_lambda_cls, cls_alpha= 273/624)

    #phase1
    optimizer1 = optim.AdamW(model.parameters(), lr=phase1_lr, weight_decay=1e-4)
    print(f"\n--- Fold {fold_idx+1} - Phase 1 (lr={phase1_lr}, λ_seg={phase1_lambda_seg}) ---")
    for epoch in range(1, phase1_epochs + 1):
        t = (epoch - 1) / (phase1_epochs - 1) if phase1_epochs > 1 else 0.0
        current_lr = phase1_lr + (phase1_lr/2 - phase1_lr) * t
        for param_group in optimizer1.param_groups:
            param_group['lr'] = current_lr
        train_loss, seg_loss, cls_loss = train_one_epoch(
            model, train_loader, optimizer1, criterion,
            phase1_lambda_seg, phase1_lambda_cls, device,
            phase='seg_only'
        )

        if epoch % 10 == 0 or epoch == phase1_epochs:
            val_metrics, _ = validate_and_collect(model, val_loader, device)
            print(f"Epoch {epoch:2d}/{phase1_epochs} | Train Loss: {train_loss:.4f} | "
                  f"Val Acc: {val_metrics['acc']:.4f} AUC: {val_metrics['auc']:.4f}")
        else:
            print(f"Epoch {epoch:2d}/{phase1_epochs} | Train Loss: {train_loss:.4f}")
    #phase2
    optimizer2 = optim.AdamW(model.parameters(), lr=phase2_lr, weight_decay=1e-4)
    print(f"\n--- Fold {fold_idx+1} - Phase 2 (lr={phase2_lr}, λ_seg={phase2_lambda_seg}) ---")
    for epoch in range(1, phase2_epochs + 1):
        t = (epoch - 1) / (phase2_epochs - 1) if phase2_epochs > 1 else 0.0
        current_lr = phase2_lr + (phase2_lr/2 - phase2_lr) * t
        current_lambda_seg = 1 + (phase2_lambda_seg - 1) * t
        current_lambda_cls = 1 - current_lambda_seg

        for param_group in optimizer2.param_groups:
            param_group['lr'] = current_lr


        train_loss, train_seg, train_cls = train_one_epoch(
            model, train_loader, optimizer2, criterion,
            current_lambda_seg, current_lambda_cls, device,
            phase='joint'
        )
        val_cls = validate_loss(
            model, val_loader, criterion,
            current_lambda_seg, current_lambda_cls, device
        )
        history["train_cls_loss"].append(train_cls)
        history["val_cls_loss"].append(val_cls)

        print(f"[Phase2] Epoch {epoch:2d} "
          f"Train CLS: {train_cls:.4f} | Val CLS: {val_cls:.4f}")
        val_metrics, _ = validate_and_collect(model, val_loader, device)
        print(f"Epoch {epoch:2d}/{phase2_epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Acc: {val_metrics['acc']:.4f} AUC: {val_metrics['auc']:.4f} F1: {val_metrics['f1']:.4f} "
            f"Sens: {val_metrics['sensitivity']:.4f} Spec: {val_metrics['specificity']:.4f}")
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(save_csv_dir, f"fold{fold_idx+1}_loss_curve.csv"),index=False)
    history_df = pd.DataFrame(history)
    plot_loss_curve(history_df, fold_idx, save_csv_dir)
    #save
    model_save_path = os.path.join(
    save_model_dir,
    f"{model_name}_fold{fold_idx+1}.pth"
    )

    torch.save({
        "fold": fold_idx + 1,
        "model_state_dict": model.state_dict(),
    }, model_save_path)

    print(f"Model saved to: {model_save_path}")

    final_metrics, results_df = validate_and_collect(model, val_loader, device)
    error_df = results_df[results_df["true_label"] != results_df["pred_label"]].copy()
    error_df.insert(0, "fold", fold_idx + 1)

    return final_metrics, error_df, results_df

print("\nLoading dataset...")
temp_dataset = MultiTaskUltrasoundDataset(
    root_dir=root_dir, image_size=image_size,
    require_mask=False,
    use_translation_aug=False, use_rotation_aug=False,
    aug_repeat_count=1
)
samples_original = temp_dataset.samples

patient_to_label = {}
for s in samples_original:
    pid = s["patient_id"]
    if pid not in patient_to_label:
        patient_to_label[pid] = s["label"]
patient_ids = list(patient_to_label.keys())
patient_labels = [patient_to_label[pid] for pid in patient_ids]

print(f"Total original images: {len(samples_original)}")
print(f"Total patients: {len(patient_ids)}")
print(f"Patient class distribution: {np.bincount(patient_labels)}")


skf_patient = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

all_folds_metrics = []
all_errors = []
all_predictions = []  

for fold_idx, (train_patient_idx, val_patient_idx) in enumerate(skf_patient.split(patient_ids, patient_labels)):
    train_patients = set([patient_ids[i] for i in train_patient_idx])
    val_patients = set([patient_ids[i] for i in val_patient_idx])

    train_samples = [s for s in samples_original if s["patient_id"] in train_patients]
    val_samples = [s for s in samples_original if s["patient_id"] in val_patients]

    print(f"\n{'='*50}")
    print(f"Fold {fold_idx+1}/{n_splits}")
    print(f"Train patients: {len(train_patients)}, Train images: {len(train_samples)}")
    print(f"Val patients: {len(val_patients)}, Val images: {len(val_samples)}")
    print(f"{'='*50}")

    metrics, error_df, results_df = train_model_for_fold(train_samples, val_samples, fold_idx)

    metrics["fold"] = fold_idx + 1
    all_folds_metrics.append(metrics)
    all_errors.append(error_df)
    all_predictions.append(results_df)

    print(f"\nFold {fold_idx+1} Results:")
    print(f"  Acc: {metrics['acc']:.4f}, AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}, Specificity: {metrics['specificity']:.4f}")

metrics_df = pd.DataFrame([
    {
        "fold": m["fold"],
        "Accuracy": m["acc"],
        "AUC": m["auc"],
        "F1": m["f1"],
        "Sensitivity": m["sensitivity"],
        "Specificity": m["specificity"]
    }
    for m in all_folds_metrics
])

print("\n Performance (each fold)")
print(metrics_df.to_string(index=False))

mean_metrics = metrics_df.drop("fold", axis=1).mean()
std_metrics = metrics_df.drop("fold", axis=1).std()

print("\n Mean Performance (5-fold)")
for metric in mean_metrics.index:
    print(f"{metric}: {mean_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")



all_predictions_with_fold = []
for i, df in enumerate(all_predictions):
    df_copy = df.copy()
    df_copy.insert(0, "fold", i+1)
    all_predictions_with_fold.append(df_copy)
all_predictions_final = pd.concat(all_predictions_with_fold, ignore_index=True)
all_predictions_final.to_csv(os.path.join(save_csv_dir, "all_predictions.csv"), index=False)

