# Semi-Supervised Segmentation with Uncertainty Calibration

Author: Francesco Nitti  
Affiliation: Politecnico di Torino – Master Thesis

## Overview

The core idea is to train a segmentation model with a combination of labeled and unlabeled data. The system estimate uncertainty and generates pseudolabels for confident predictions.

Key steps:
- Base supervised training on a small labeled set
- Class-wise calibration of uncertainty thresholds
- Pseudolabel generation for the unlabeled set
- Iterative retraining with new pseudolabels

## Folder Structure

The expected dataset layout is as follows:

DATA/
├── train/
│ ├── images/
│ └── masks/
├── val/
│ ├── images/
│ └── masks/
└── unlabeled/
│ ├── images/

## Configuration file

Training is controlled by a YAML file (example: ./configs/custom.yaml).
You should include at least number of classes and name of classes.


