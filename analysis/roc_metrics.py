"""Cross-model evaluation summary: compute metrics from each model's all_predictions.csv and plot mean ROC.

Mirrors the logic of the original AUC.ipynb, outputting:
- Image-level/patient-level metric tables (ACC / AUC / F1 / Sensitivity / Specificity / PPV / NPV)
- Image-level/patient-level mean ROC curves (5 folds combined)

Usage (from the repository root):
    python analysis/roc_metrics.py --models MT_MIX_shcbam MT_MP_shcbam MT_Net_shcbam MS_Net_sh
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ultrasound_mt.config import load_config


def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return acc, sens, spec, ppv, npv, f1


def patient_level(df):
    """Aggregate by (fold, patient_id), taking the first true_label and the mean pred_prob."""
    return (df.dropna(subset=["true_label", "pred_prob"])
              .groupby(["fold", "patient_id"], as_index=False)
              .agg({"true_label": "first", "pred_prob": "mean"}))


def evaluate(df, threshold=0.5):
    df = df.dropna(subset=["true_label", "pred_prob"])
    y_true = df["true_label"].values
    y_prob = df["pred_prob"].values
    y_pred = (y_prob >= threshold).astype(int)
    acc, sens, spec, ppv, npv, f1 = compute_metrics(y_true, y_pred)
    auc_val = auc(*roc_curve(y_true, y_prob)[:2]) if len(np.unique(y_true)) > 1 else float("nan")
    return dict(AUC=auc_val, Accuracy=acc, F1=f1, Sensitivity=sens,
                Specificity=spec, PPV=ppv, NPV=npv)


def plot_mean_roc(dfs, labels, out_path, level="image"):
    plt.figure(figsize=(6, 6))
    # Concatenate (y_true, y_prob) across all folds for each model, then plot a single ROC
    for (name, sub), label in zip(dfs, labels):
        sub = sub.dropna(subset=["true_label", "pred_prob"])
        y_true = sub["true_label"].values
        y_prob = sub["pred_prob"].values
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        a = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC={a:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1.01])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"Mean ROC ({level}-level)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"ROC saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="List of model names to summarize")
    p.add_argument("--result-dir", default=None, help="Root directory of results; defaults to configs/data.yaml")
    p.add_argument("--out", default=None, help="Output directory")
    args = p.parse_args()

    cfg = load_config(data_cfg="configs/data.yaml")
    root = args.result_dir or cfg["paths"].get("result_dir")
    out = args.out or os.path.join(cfg["paths"].get("output_dir", "results"), "analysis")
    os.makedirs(out, exist_ok=True)

    img_dfs, pat_dfs = [], []
    metrics_img, metrics_pat = [], []
    for name in args.models:
        pred_csv = os.path.join(root, f"{name}_results", "all_predictions.csv")
        if not os.path.exists(pred_csv):
            print(f"[warn] missing {pred_csv}, skip")
            continue
        df = pd.read_csv(pred_csv)
        img_dfs.append((name, df))
        pdf = patient_level(df)
        pat_dfs.append((name, pdf))
        metrics_img.append({"Model": name, **evaluate(df)})
        metrics_pat.append({"Model": name, **evaluate(pdf)})

    if not img_dfs:
        print("no data to plot")
        return

    img_metrics = pd.DataFrame(metrics_img)
    pat_metrics = pd.DataFrame(metrics_pat)
    img_metrics.to_csv(os.path.join(out, "image_level_results.csv"), index=False)
    pat_metrics.to_csv(os.path.join(out, "patient_level_results.csv"), index=False)
    print("\nImage-level:\n", img_metrics.to_string(index=False))
    print("\nPatient-level:\n", pat_metrics.to_string(index=False))

    plot_mean_roc(img_dfs, args.models, os.path.join(out, "Mean_ROC_image.png"), level="image")
    plot_mean_roc(pat_dfs, args.models, os.path.join(out, "Mean_ROC_patient.png"), level="patient")


if __name__ == "__main__":
    main()
