# General information

This repository contains the implementation of our method 
**"Uncertainty-Aware Semi-Supervised Learning for Neurosurgical Navigation"**, 
which has been submitted for possible publication to *Applied Soft Computing*.
The full README and documentation will be made available once the editorial process progresses.

For any inquiries in the meantime, feel free to reach out at francesco.nitti@polito.it

This project trains a segmentation model using a combination of labeled and unlabeled data.
The system estimates uncertainty and generates pseudolabels for confident predictions.

**Key steps:**
- Supervised training on a small labeled set
- Class-wise calibration of uncertainty thresholds
- Pseudolabel generation for the unlabeled set
- Iterative retraining with new pseudolabels


# Dataset availablility

You can download the dataset at the following link: https://www.kaggle.com/datasets/artemis90/neurosurgical-navigation-semisupervised-dataset

## Dataset Structure

The expected dataset layout is as follows:

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


## Configuration

Training is controlled by a YAML file (e.g., `./configs/custom.yaml`). 
The list of classes should be defined in the YAML configuration.
