# Segmentation-Guided Liver Fibrosis Classification

This repository contains the implementation of our segmentation-guided framework for liver fibrosis classification from B-mode ultrasound images.

---

## Repository layout

```
.
├── src/ultrasound_mt/     # Core package: models, losses, data, utils, training
├── configs/               # YAML configuration (data, training, per-model)
├── evaluate/              # Evaluate trained checkpoints (classification + segmentation)
├── analysis/              # Metrics summary and mean-ROC plotting
├── dataset/               # Ultrasound images + segmentation masks (see below)
├── results/               # Output of training / evaluation runs
├── pyproject.toml         # Package definition (pip install -e .)
├── requirements.txt
└── README.md
```

> `src/ultrasound_mt/` is the single source of truth for the model
> implementation. It contains:
>
> ```
> src/ultrasound_mt/
> ├── config.py          # YAML loading and merging
> ├── models/            # blocks / heads / model factory (models.py)
> ├── losses/            # Dice + weighted Focal + MultiTaskLoss
> ├── data/              # dataloader (loading + augmentation)
> ├── utils/             # helpers + segmentation metrics
> └── train/             # train_multitask / train_singletask
> ```



---

## Dataset format

The dataset is organized into **two class groups**, each containing one folder
per patient. Every patient folder holds paired images and masks with matching
base names: a `.png` image and a `.nrrd` segmentation mask of the same name.

```
dataset/
├── 0/                      # class 0 (e.g. benign)
│   ├── <patient_id>/
│   │   ├── 3.png           # ultrasound image
│   │   ├── 3.nrrd          # lesion segmentation mask
│   │   ├── 4.png
│   │   ├── 4.nrrd
│   │   └── ...
│   └── ...
└── 1/                      # class 1 (e.g. malignant)
    ├── <patient_id>/
    │   ├── 3.png
    │   ├── 3.nrrd
    │   └── ...
    └── ...
```

Details:

- **Class label** is the top-level folder name (`0` / `1`); the dataset is used
  for binary classification.
- **Image**: grayscale ultrasound image, `.png`.
- **Segmentation mask**: `.nrrd` file with the **same base name** as its image
  (e.g. `3.png` ↔ `3.nrrd`), stored as a binary lesion mask. Some samples may
  lack a mask; those are used for classification only.
- Images are resized to `(768, 1024)` (H, W) by the dataloader.
- Data paths are configured in `configs/data.yaml` (the `paths` section).
- The clinical dataset used in this study is not publicly available due to privacy and ethical restrictions.
---

## Models

All 14 model variants are produced by the single factory
`ultrasound_mt.models.build_model(name)` (see `configs/models/*.yaml`):

| Family | Classifier head | Variants | Description |
|--------|-----------------|----------|-------------|
| MT_Net | Global pooling (gap) | MT_Net / MT_Net_shcbam / MT_Net_wo | Multi-task (seg + cls) |
| MT_MP | ROI masked-GAP (roi_gap) | MT_MP / MT_MP_shcbam / MT_MP_wo | Segmentation-guided weighted pooling |
| MT_MIX | ROI + global hybrid (hybrid) | MT_MIX / MT_MIX_shcbam / MT_MIX_wo | Paper's main method (MG-MIX) |
| MS_Net | Global pooling | MS_Net / MS_Net_sh / MS_Net_wo | Single-task classification baseline |
| baseline | Global pooling (x5) | baseline / baseline_wo | Encoder-last-layer only |

- `_wo` are CBAM-removed ablation counterparts.
- Each model's structure is described in `configs/models/<name>.yaml`.

---

## Training

Run from the repository root (the scripts use relative imports and must be
launched as modules — see "How to start training (Method A)" above):

```bash
# Multi-task: two-stage curriculum learning
#   Phase 1 - segmentation pretraining, Phase 2 - joint training
python -m ultrasound_mt.train.train_multitask --model MT_MIX

# Single-task (classification only)
python -m ultrasound_mt.train.train_singletask --model MS_Net

# With explicit config / output dir / seed
python -m ultrasound_mt.train.train_multitask --model MT_MP_shcbam \
    --data-config configs/data.yaml --train-config configs/train_multitask.yaml \
    --out-root results/my_run --seed 42
```

Outputs are written to `<output_dir>/{model}_models/*.pth` and
`<output_dir>/{model}_results/` (loss curves, per-fold metrics,
`all_predictions.csv`).

---

## Evaluation & analysis

```bash
# Evaluate trained weights (classification + segmentation metrics)
python -m evaluate.evaluate_model --model MT_MIX --result-dir /data/Li/Result \
    --out results/eval

# Summarize multiple models and plot mean ROC (image-level / patient-level)
python analysis/roc_metrics.py \
    --models MT_MIX_shcbam MT_MP_shcbam MT_Net_shcbam MS_Net_sh
```

---

## Reproducibility notes

- **Patient-level stratified 5-fold cross-validation** is used so that images of
  the same patient never leak across folds.
---
