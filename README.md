# Segmentation-Guided Liver Fibrosis Classification

This repository contains the implementation of our segmentation-guided framework for liver fibrosis classification from B-mode ultrasound images.

## Environment

The code was developed and tested under the following environment:

- Operating system: Linux
- Python: 3.11
- PyTorch: 2.8.0
- Torchvision: 0.23.0
- CUDA: 12.6

Main dependencies:

- numpy==2.0.2
- pandas==2.3.3
- scikit-learn==1.6.1
- opencv-python==4.13.0.92
- scikit-image==0.24.0
- SimpleITK==2.5.3
- pynrrd==1.1.3
- PyYAML==6.0.3
- matplotlib==3.9.4
- tqdm==4.67.3


Before running the code, modify the paths and batch size in config.yaml.

## Data Organization

The dataset should be organized as follows:
Dataset/
```text
│
├── 0/
│   ├── Patient_001/
│   │   ├── image_001.png
│   │   └── image_001.nrrd
│
└── 1/
    ├── Patient_002/
    │   ├── image_002.png
    │   └── image_002.nrrd
```

0 and 1 indicate fibrosis classification labels and 0 is for non-significant fibrosis.
.png files are ultrasound images.
.nrrd files are corresponding liver segmentation masks.

The clinical dataset used in this study is not publicly available due to privacy and ethical restrictions.
