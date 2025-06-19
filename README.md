# Uncertanity-Aware Semi-Supervised Learning for Neurosurgical Navigation

Weights: https://drive.google.com/file/d/1wGZ4kIfj_DF5r-dL7LkxjKWQbPsqTWQT/

## Overview

Train a segmentation model with a combination of labeled and unlabeled data. The system estimate uncertainty and generates pseudolabels for confident predictions.

Key steps:
- Base supervised training on a small labeled set
- Class-wise calibration of uncertainty thresholds
- Pseudolabel generation for the unlabeled set
- Iterative retraining with new pseudolabels

## Folder Structure

The expected dataset layout is as follows:

```
DATA/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── Unlabeled/
    └── images/
```
## Configuration file

Training is controlled by a YAML file (example: ./configs/custom.yaml).
You should include at least number of classes and name of classes.


