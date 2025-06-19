import os

import numpy as np

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
import cv2

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return param_num / 1e6, trainable_params / 1e6


def entropy_minimization(prediction, lambda_entropy = 0.1):
    softmax_pred = F.softmax(prediction, dim=1)
    entropy = -torch.sum(softmax_pred * torch.log(softmax_pred + 1e-10), dim=1)
    entropy_loss = torch.mean(entropy) * lambda_entropy
    return entropy_loss

def normalize_image(img, resize_size):
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize([0.45, 0.27, 0.25], [0.30, 0.23, 0.21]),
    ])
    return transform(img)

def load_image(image_path, resize_size):
    img = Image.open(image_path).convert('RGB')
    return normalize_image(img, resize_size).unsqueeze(0)

def load_mask(mask_path, resize_size):
    mask = Image.open(mask_path).convert('L')
    mask = transforms.Resize(resize_size, interpolation=Image.NEAREST)(mask)
    return torch.from_numpy(np.array(mask)).long().unsqueeze(0)

def remove_small_components(pred,cfg,min_size=1):
    pred = pred.astype(np.uint8)
    for class_idx in range(1, cfg['nclass']):
        class_mask = (pred == class_idx).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] < min_size:
                pred[labels == label] = 0

    return pred

def calculate_dice(pred, mask, num_classes):
    dice_scores = np.full(num_classes, np.nan)
    for i in range(num_classes):
        pred_i = (pred == i)
        mask_i = (mask == i)
        tp = np.sum(pred_i & mask_i)
        fp = np.sum(pred_i & ~mask_i)
        fn = np.sum(~pred_i & mask_i)
        if tp + fp + fn > 0:
            dice_scores[i] = (2. * tp) / (2. * tp + fp + fn + 1e-10)
    return dice_scores
