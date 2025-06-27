# Uncertanity-Aware Semi-Supervised Learning for Neurosurgical Navigation

## 🚧 **Repository under construction**
This repository contains the implementation of our method  
**"Uncertainty-Aware Semi-Supervised Learning for Neurosurgical Navigation"**
The full README, and documentation will be made available once the editorial process progresses.

For any inquiries in the meantime, feel free to reach out @francesco.nitti@polito.it

## Pretrained Weights

[Download weights from Google Drive](https://drive.google.com/file/d/1wGZ4kIfj_DF5r-dL7LkxjKWQbPsqTWQT/)

## Overview

Train a segmentation model with a combination of labeled and unlabeled data. The system estimate uncertainty and generates pseudolabels for confident predictions.

Key steps:
- Base supervised training on a small labeled set
- Class-wise calibration of uncertainty thresholds
- Pseudolabel generation for the unlabeled set
- Iterative retraining with new pseudolabels

## Dataset Structure

The expected dataset layout is as follows. 

```
dataset/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── unlabeled/
    └── images/
```
A few annotated samples are provided to understand annotation format.


## Configuration file

Training is controlled by a YAML file (example: ./configs/custom.yaml)
Classes should be defined in the YAML file.
