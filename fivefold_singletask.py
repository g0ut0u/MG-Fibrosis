import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from Loss import WeightedFocalWithLogitsLoss
import gc
from MS_Net import MSNet
from MS_Net_sh import MSNet_sh
from Baseline import BaselineResUNetClassifier
from MS_Net_wo import MSNet_wo
from Baseline_wo import BaselineResUNetClassifier_wo
from dataloader import MultiTaskUltrasoundDataset
from helpfunction import plot_loss_curve
import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

#model_name = 'baseline'  #use different model_name for different models
#model_name = 'MS_Net'
model_name = 'MS_Net'
#model_name = 'baseline_wo'  
#model_name = 'MS_Net_wo'


result_root = config["paths"]["result_root"]
root_dir = config["dataset"]["root_dir"]  
image_size = (768, 1024)
batch_size = 48
num_workers = 0
pin_memory = True
in_channels = 1
bilinear = True
dropout = 0.3
lr = 2e-4
weight_decay = 1e-4
epochs = 20
n_splits = 5
random_state = 42
save_csv_dir = os.path.join(result_root,f"{model_name}_results")
save_model_dir = os.path.join(result_root,f"{model_name}_models")
use_translation_aug = True
use_rotation_aug = True
max_shift_x = 20
max_shift_y = 20
max_rotate_angle = 10
translation_p = 1
rotation_p = 1
aug_repeat_count = 2

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_metrics_with_probs(probs, labels, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.5
    f1 = f1_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0,1]).ravel()
    sensitivity = tp / (tp + fn) if (tp+fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn+fp) > 0 else 0.0
    return acc, auc, f1, sensitivity, specificity

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device).float().view(-1, 1)
        optimizer.zero_grad()
        outputs = model(images)
        logits = outputs["cls_logits"]
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        gc.collect()
    return avg_loss

def validate(model, dataloader, device):

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

            probs = torch.sigmoid(
                outputs["cls_logits"]
            ).cpu().numpy().flatten()

            all_probs.extend(probs)
            all_labels.extend(labels)

            all_patient_ids.extend(patient_ids)
            all_image_paths.extend(image_paths)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    acc, auc, f1, sens, spec = compute_metrics_with_probs(
        all_probs,
        all_labels
    )

    results_df = pd.DataFrame({
        "image_path": all_image_paths,
        "patient_id": all_patient_ids,
        "true_label": all_labels,
        "pred_prob": all_probs,
        "pred_label": (all_probs >= 0.5).astype(int)
    })

    metrics = {
        "acc": acc,
        "auc": auc,
        "f1": f1,
        "sensitivity": sens,
        "specificity": spec,
        "probs": all_probs,
        "labels": all_labels
    }

    return metrics, results_df

def validate_loss(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:

            images = batch["image"].to(device)
            labels = batch["label"].float().view(-1,1).to(device)

            outputs = model(images)

            logits = outputs["cls_logits"]

            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)

    return total_loss / len(dataloader.dataset)

def train_model_for_fold(train_samples, val_samples, fold_idx, device):
    train_dataset = MultiTaskUltrasoundDataset(
        root_dir=root_dir, image_size=image_size,
        require_mask=False,
        use_translation_aug=use_translation_aug,
        use_rotation_aug=use_rotation_aug,
        max_shift_x=max_shift_x, max_shift_y=max_shift_y,
        max_rotate_angle=max_rotate_angle,
        translation_p=translation_p, rotation_p=rotation_p,
        aug_repeat_count=aug_repeat_count
    )
    train_dataset.samples = train_samples
    train_dataset.__len__ = lambda self: len(self.samples) * self.aug_repeat_count

    val_dataset = MultiTaskUltrasoundDataset(
        root_dir=root_dir, image_size=image_size,
        require_mask=False,
        use_translation_aug=False, use_rotation_aug=False,
        aug_repeat_count=1
    )
    val_dataset.samples = val_samples
    val_dataset.__len__ = lambda self: len(self.samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    history = {"train_cls_loss": [],"val_cls_loss": []}

    if model_name == 'baseline':
        model = BaselineResUNetClassifier(in_channels=in_channels, bilinear=bilinear, dropout=dropout).to(device)
    elif model_name == 'MS_Net':
        model = MSNet(in_channels=in_channels, bilinear=bilinear, dropout=dropout).to(device)
    elif model_name == 'MS_Net_sh':
        model = MSNet(in_channels=in_channels, bilinear=bilinear, dropout=dropout).to(device)
    elif model_name == 'baseline_wo':
        model = BaselineResUNetClassifier_wo(in_channels=in_channels, bilinear=bilinear, dropout=dropout).to(device)
    elif model_name == 'MS_Net_wo':
        model = MSNet_wo(in_channels=in_channels, bilinear=bilinear, dropout=dropout).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = WeightedFocalWithLogitsLoss(gamma=1)

    print(f"\n--- Fold {fold_idx+1} / {n_splits} ---")
    for epoch in range(1, epochs+1):
        t = (epoch - 1) / (epochs - 1) if epochs > 1 else 0.0
        current_lr = lr + (lr/2 - lr) * t
        optimizer = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=weight_decay)
        criterion = WeightedFocalWithLogitsLoss(gamma=1)
        train_loss = train_one_epoch(model,train_loader,optimizer,criterion,device)

        val_loss = validate_loss(model,val_loader,criterion,device)

        val_metrics, _ = validate(model,val_loader,device)

        history["train_cls_loss"].append(train_loss)
        history["val_cls_loss"].append(val_loss)
        print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['acc']:.4f} AUC: {val_metrics['auc']:.4f} F1: {val_metrics['f1']:.4f} "
            f"Sens: {val_metrics['sensitivity']:.4f} Spec: {val_metrics['specificity']:.4f}")

    history_df = pd.DataFrame(history)

    history_df.to_csv(
    os.path.join(save_csv_dir,f"fold{fold_idx+1}_loss_curve.csv"),index=False)
    plot_loss_curve(history_df,fold_idx,save_csv_dir)
    final_metrics, results_df = validate(model,val_loader,device)
    model_save_path = os.path.join(save_model_dir,f"{model_name}_fold{fold_idx+1}.pth")
    torch.save({"fold": fold_idx + 1,"model_state_dict": model.state_dict(),},model_save_path)
    print(f"Model saved to: {model_save_path}")
    return final_metrics, results_df

if __name__ == "__main__":
    set_seed(random_state)
    print(f"Random seed set to {random_state}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(save_csv_dir, exist_ok=True)
    os.makedirs(save_model_dir, exist_ok=True)

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

        metrics, results_df = train_model_for_fold(train_samples,val_samples,fold_idx,device)
        metrics["fold"] = fold_idx + 1
        all_folds_metrics.append(metrics)

        results_df.insert(0,"fold",fold_idx + 1)
        all_predictions.append(results_df)

        print(f"\nFold {fold_idx+1} Results:")
        print(f"  Acc: {metrics['acc']:.4f}, AUC: {metrics['auc']:.4f}, F1: {metrics['f1']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}, Specificity: {metrics['specificity']:.4f}")

    avg_acc = np.mean([m["acc"] for m in all_folds_metrics])
    avg_auc = np.mean([m["auc"] for m in all_folds_metrics])
    avg_f1 = np.mean([m["f1"] for m in all_folds_metrics])
    avg_sens = np.mean([m["sensitivity"] for m in all_folds_metrics])
    avg_spec = np.mean([m["specificity"] for m in all_folds_metrics])

    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    all_predictions_df.to_csv(os.path.join(save_csv_dir, "all_predictions.csv"), index=False)

    print(f"\nResults saved to {save_csv_dir}")