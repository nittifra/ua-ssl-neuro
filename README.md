# General information
This repository contains the implementation of our method 
**"Uncertainty-aware semi-supervised learning for neurosurgical navigation"**, 
published open access in *Applied Soft Computing* (CC BY 4.0).

Paper: https://doi.org/10.1016/j.asoc.2026.115252
Volume 197 (2026) 115252.

If you use this code, please cite:
> F. Nitti, S. Seoni, A. Morello, L. Dolci, A. Piazza, V. Esposito, L. Rosito, 
> D. Garbossa, F. Cofano, A. Sengur, M. Salvi. 
> Uncertainty-aware semi-supervised learning for neurosurgical navigation. 
> *Applied Soft Computing* 197 (2026) 115252. 
> https://doi.org/10.1016/j.asoc.2026.115252

For any inquiries, feel free to reach out at francesco.nitti@polito.it

This project trains a segmentation model using a combination of labeled and unlabeled data.
The system estimates uncertainty (Semantic Spatial Uncertainty, SSU) via Monte Carlo Dropout 
and generates pseudolabels for confident predictions.

**Key steps:**
- Supervised training on a small labeled set
- Class-wise calibration of uncertainty thresholds
- Pseudolabel generation for the unlabeled set
- Iterative retraining with new pseudolabels

# Dataset availablility
You can download the dataset at the following link: https://www.kaggle.com/datasets/artemis90/neurosurgical-navigation-semisupervised-dataset

## Dataset Structure
The expected dataset layout is as follows:

# Dataset availablility

You can download the dataset at the following link: https://www.kaggle.com/datasets/artemis90/neurosurgical-navigation-semisupervised-dataset


## Configuration

Training is controlled by a YAML file (e.g., `./configs/custom.yaml`). 
The list of classes should be defined in the YAML configuration.
