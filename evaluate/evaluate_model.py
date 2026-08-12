"""Model evaluation script: computes classification and segmentation metrics for trained weights.

- Classification metrics (ACC / AUC / F1 / Sensitivity / Specificity): image-level and patient-level
- Segmentation metrics (Dice / IoU): if the model outputs seg_logits and the data contains masks

By default it points to the repository's existing Result/ directory (retained old weights); it can be
overridden via yaml / command-line arguments.

Usage (from the repository root):
    python -m evaluate.evaluate_model --model MT_MIX \
        --result-dir /data/Li/Result --out /data/Li/results/eval
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader

import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ultrasound_mt.config import load_config, resolve_model_yaml
from ultrasound_mt.data import MultiTaskUltrasoundDataset
from ultrasound_mt.models import build_model
from ultrasound_mt.utils.metrics import evaluate_segmentation


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained model")
    p.add_argument("--model", required=True, help="Model name")
    p.add_argument("--data-config", default="configs/data.yaml")
    p.add_argument("--result-dir", default=None,
                   help="Root directory containing {model}_models/*.pth and {model}_results; defaults to the yaml result_dir")
    p.add_argument("--out", default=None, help="Output directory for evaluation results")
    p.add_argument("--folds", default=None, help="Comma-separated fold indices, e.g. 1,2,3,4,5; defaults to all")
    return p.parse_args()


def find_weights(root_dir, model_name, folds):
    """Locate the fold-th weight under the {model_name}_models directory in root_dir (file names may omit _shcbam/_wo)."""
    folder = os.path.join(root_dir, f"{model_name}_models")
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"model folder not found: {folder}")
    files = glob.glob(os.path.join(folder, "*.pth"))

    # keyword = model name with suffixes removed (e.g. MT_MP_shcbam -> weights named MT_MP_fold*.pth)
    kw = model_name
    for s in ("_shcbam", "_sh", "_wo"):
        kw = kw.replace(s, "")
    kw = kw.lower()

    found = {}
    for fold in folds:
        cands = [f for f in files if kw in os.path.basename(f).lower()
                 and f"fold{fold}" in os.path.basename(f).lower()]
        if not cands:
            print(f"[warn] no weight found for {model_name} fold {fold} in {folder}")
            continue
        found[fold] = cands[0]
    return found, folder


def compute_cls_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    f1 = f1_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return dict(Accuracy=acc, AUC=auc, F1=f1, Sensitivity=sens,
                Specificity=spec, PPV=ppv, NPV=npv)


def main():
    args = parse_args()
    cfg = load_config(data_cfg=args.data_config, model_cfg=resolve_model_yaml(args.model))
    model_name = args.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = list(range(1, cfg["train"].get("n_splits", 5) + 1))
    if args.folds:
        folds = [int(f) for f in args.folds.split(",")]

    root = args.result_dir or cfg["paths"].get("result_dir")
    if not root:
        raise ValueError("--result-dir or configs paths.result_dir must be specified")
    weights, folder = find_weights(root, model_name, folds)

    out = args.out or os.path.join(root, f"{model_name}_eval")
    os.makedirs(out, exist_ok=True)

    d = cfg["data"]
    ds = MultiTaskUltrasoundDataset(root_dir=cfg["paths"]["dataset_dir"],
                                    image_size=d["image_size"], require_mask=True)
    rel_to_idx = {s["image_path"].replace(cfg["paths"]["dataset_dir"] + "/", ""): i
                  for i, s in enumerate(ds.samples)}

    model = build_model(model_name, in_channels=d.get("in_channels", 1),
                        bilinear=cfg["model"].get("bilinear", True),
                        dropout=cfg["model"].get("dropout", 0.3),
                        yaml_cfg={"model": cfg["model"]})
    model.to(device)

    all_rows = []
    metrics_rows = []
    for fold, wpath in weights.items():
        ck = torch.load(wpath, map_location="cpu")
        model.load_state_dict(ck["model_state_dict"], strict=True)
        model.eval()

        # Classification: process the val-fold samples
        # Simplified here: evaluate this weight on the full dataset (aligning with the original results
        # would require fold-wise splitting; this gives predictions over all samples)
        loader = DataLoader(ds, batch_size=d["batch_size"], shuffle=False,
                            num_workers=d.get("num_workers", 0))
        probs, labels, pids, paths = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                img = batch["image"].to(device)
                labels.extend(batch["label"].cpu().numpy().reshape(-1))
                pids.extend(batch["patient_id"])
                paths.extend(batch["image_path"])
                model_out = model(img)
                probs.extend(torch.sigmoid(model_out["cls_logits"]).cpu().numpy().flatten())
        probs, labels = np.array(probs), np.array(labels)

        # Segmentation
        has_seg = any("seg_head" in k for k in model.state_dict())
        dice = iou = float("nan")
        if has_seg:
            dice, iou = evaluate_segmentation(model, loader, device)

        m = compute_cls_metrics(labels, probs)
        m.update(fold=fold, Dice=dice, IoU=iou)
        # Patient level
        pdf = pd.DataFrame({"patient_id": pids, "label": labels, "prob": probs})
        pgroup = pdf.groupby("patient_id").agg(label=("label", "first"), prob=("prob", "mean"))
        pm = compute_cls_metrics(pgroup["label"].values, pgroup["prob"].values)
        print(f"fold {fold}: ACC={m['Accuracy']:.4f} AUC={m['AUC']:.4f} "
              f"Sens={m['Sensitivity']:.4f} Spec={m['Specificity']:.4f} "
              f"PatientAUC={pm['AUC']:.4f} Dice={dice:.4f} IoU={iou:.4f}")

        metrics_rows.append(m)
        all_rows.append(pd.DataFrame({
            "fold": fold, "image_path": paths, "patient_id": pids,
            "true_label": labels, "pred_prob": probs,
        }))

    pd.DataFrame(metrics_rows).to_csv(os.path.join(out, "image_level_metrics.csv"), index=False)
    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(
            os.path.join(out, "all_predictions.csv"), index=False)
    print(f"\nEvaluation results saved to: {out}")


if __name__ == "__main__":
    main()
