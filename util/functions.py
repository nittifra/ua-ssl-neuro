import os
import json
import shutil
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt
import cv2
from colorama import Fore, Style
from alive_progress import alive_bar
import albumentations as A
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy.stats import spearmanr

from util.utils import (
    calculate_dice,
    load_image,
    load_mask,
    remove_small_components,
)


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_params(model):
    """Count the number of parameters in a model"""
    param_num = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return param_num / 1e6, trainable_params / 1e6


def entropy_minimization(prediction, lambda_entropy=0.1):
    """Compute entropy minimization loss for semi-supervised learning"""
    softmax_pred = F.softmax(prediction, dim=1)
    entropy = -torch.sum(softmax_pred * torch.log(softmax_pred + 1e-10), dim=1)
    entropy_loss = torch.mean(entropy) * lambda_entropy
    return entropy_loss


def inference_SSU(
        model,
        input_image,
        n_samples,
        device
        ):

    """
    auth: nittifra
    """
    model.eval()  

    def enable_dropout(m):
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    model.apply(enable_dropout)
    input_image = input_image.to(device)

    outputs = []
    for _ in range(n_samples):
        with torch.no_grad():
            output = model(input_image)
            output = F.softmax(output, dim=1)
            output = output.squeeze(0)  
            outputs.append(output)
    

    outputs = torch.stack(outputs)                                                 # Shape: [n_samples, num_classes, W, H]
    mean_output = torch.mean(outputs, dim=0)                                       # Shape: [num_classes, H, W]
    pred = torch.argmax(mean_output, dim=0)                                        # Shape: [H, W]  -> OUTPUT "pred"
    prediction_one_hot = F.one_hot(pred, num_classes=mean_output.shape[0]).bool()  # Shape: [H, W, num_classes]

    # Compute per-sample predictions: argmax over num_classes dimension
    preds_per_sample = torch.argmax(outputs, dim=1)                                # Shape: [n_samples, H, W]
    preds_one_hot = F.one_hot(preds_per_sample, num_classes=mean_output.shape[0])  # Shape: [n_samples, H, W, num_classes]

    # Compute logical OR over samples
    logical_or = torch.any(preds_one_hot, dim=0)                                   # Shape: [H, W, num_classes]

    # Computer uncertainty
    mask1 = prediction_one_hot                                                     # Shape: [H, W, num_classes]
    mask2 = logical_or                                                             # Shape: [H, W, num_classes]
    
    epsilon = 1e-10 
    num = (mask1 & mask2).sum(dim=(0, 1)).float()                                  # Shape: [num_classes]
    den = mask1.sum(dim=(0, 1)).float() + mask2.sum(dim=(0, 1)).float()            # Shape: [num_classes]
    uncertainty = 1 - (2 * num / den + epsilon)                                    # Shape: [num_classes] -> OUTPUT "uncertainty"

    return pred.cpu().numpy(), uncertainty.cpu().numpy()


def calibration_thresholds(
        model,
        images_path,
        masks_path,
        cfg,
        device
    ):

    n_samples = cfg['montecarlo_runs']
    runs = cfg['calibration_runs']
    cap = cfg['calibration_cap']

    image_files = sorted(os.listdir(images_path))
    mask_files = sorted(os.listdir(masks_path))

    len_images = len(image_files)
    expected_iterations = len_images 

    if cap and cap < expected_iterations:       # Cap is used when calibrating on the training set, to avoid the calibration on the whole set.
        expected_iterations = cap
        
    expected_iterations = expected_iterations * runs

    num_classes = cfg['nclass']

    r_values_total = []
    s_values_total = []

    all_thresholds = {i: [] for i in range(1, num_classes)} 
    acceptable_region = {i: [] for i in range(1, num_classes)}

    dice_raw_list = []

    if "train" in images_path.lower():
        dataset_type = "train"
    elif "val" in images_path.lower():
        dataset_type = "val"
    else:
        dataset_type = "unknown"

    CLASSES = cfg['classes']



    with alive_bar(expected_iterations + 1, spinner='pulse', bar='solid',title=f"{'       Uncertainty Calibration':35}") as bar:
        for _ in range(runs):

            class_uncertainty = {i: [] for i in range(1, num_classes)}
            class_dice = {i: [] for i in range(num_classes)}

            combined = list(zip(image_files, mask_files))
            random.shuffle(combined)             # When calibration is performed on the training set, this is needed. (otherwise the calibration is biased)
            image_files, mask_files = zip(*combined)

            j = 0

            for img_file, mask_file in (zip(image_files, mask_files)):
                img_path = os.path.join(images_path, img_file)
                mask_path = os.path.join(masks_path, mask_file)

                img = load_image(img_path, cfg['crop_size']).to(device)
                mask = load_mask(mask_path, cfg['crop_size']).cpu().numpy()[0]        

                pred, uncertainty = inference_SSU(model, img, n_samples, device)
                dice = calculate_dice(pred, mask, num_classes)

                dice_raw_list.append(dice)
                for i in range(1, num_classes):
                    if np.sum(mask == i) > 0 and not np.isnan(dice[i]) and np.sum(pred == i) > 0:   # Some constraints that are needed when calibrating."""np.sum(mask == i) > 0"""
                        class_dice[i].append(dice[i])
                        class_uncertainty[i].append(uncertainty[i])
                        if np.isnan(uncertainty[i]):
                            print(f"Hopefully this message will never appear.")

                bar()
                j += 1
                if j == cap:
                    break

            mean_raw_dice = np.nanmean(dice_raw_list, axis=0)
            bar.title(f"{'      We are saving the results...':35}")

            # Number of classes (excluding background)
            n_plots = cfg['nclass'] - 1  # Subtract 1 to ignore the background

            # Automatically compute number of rows and columns
            n_cols = int(np.ceil(np.sqrt(n_plots)))
            n_rows = int(np.ceil(n_plots / n_cols))

            # Create the subplots
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(32, 42))
            fig.subplots_adjust(hspace=0.2, wspace=0.2)
            axes = axes.ravel()  # Flatten the axes array for easy iteration

            for i in range(1, num_classes):
                # Extract the uncertainty and Dice score data
                uncertainties = np.array(class_uncertainty[i])
                dice_scores = np.array(class_dice[i])

                data = np.column_stack((uncertainties, dice_scores))

                if data.shape[0] > 10:
                    isolation_forest = IsolationForest(contamination=0.01, random_state=0)  
                    outliers = isolation_forest.fit_predict(data) == -1  # Outliers are marked as -1
                    data_no_outliers = data[~outliers]
                    dice_scores_no_outliers = dice_scores[~outliers]
                else:
                    data_no_outliers = data
                    dice_scores_no_outliers = dice_scores

                ax = axes[i - 1]  

                if len(data_no_outliers) > 2:

                    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(data_no_outliers)
                    centroids = kmeans.cluster_centers_
                    labels = kmeans.labels_
                    optimal_threshold = min(centroids[0, 0], centroids[1, 0])
                    acceptable = (centroids[0, 0] + centroids[1, 0]) / 2
                    all_thresholds[i].append(optimal_threshold)
                    acceptable_region[i].append(acceptable)

                    cluster_0_mean_dice = np.mean(dice_scores_no_outliers[labels == 0])
                    cluster_1_mean_dice = np.mean(dice_scores_no_outliers[labels == 1])

                    # Ensure the high Dice cluster is always labeled as 1
                    if cluster_0_mean_dice > cluster_1_mean_dice:
                        labels = np.where(labels == 0, 1, 0)  # Swap labels

                    # Calculate correlation coefficients
                    r_value = np.corrcoef(data_no_outliers[:, 0], data_no_outliers[:, 1])[0, 1]
                    s_value, _ = spearmanr(data_no_outliers[:, 0], data_no_outliers[:, 1])

                    r_values_total.append(r_value)
                    s_values_total.append(s_value)

                    # Plot the data points
                    ax.scatter(data_no_outliers[:, 0], data_no_outliers[:, 1], c=labels, cmap='viridis', edgecolors='black')
                    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100)

                    # Set the legend and title
                    ax.text(0.70, 0.95, f'R = {r_value:.2f}\nS = {s_value:.2f}', transform=ax.transAxes, fontsize=20, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
                    ax.set_title(f'{CLASSES[i]}', fontsize=30, fontweight='bold')
                else:
                    ax.scatter(data[:, 0], data[:, 1], c='red', edgecolors='black') 

                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel('Semantic Spatial Uncertainty', fontsize=30, fontstyle='italic')
                ax.set_ylabel('Dice Similarity Coefficient', fontsize=30, fontstyle='italic')
                average_r_value = np.nanmean(r_values_total)
                average_s_value = np.nanmean(s_values_total)
                # Add average R and S values as a superimposed text
                fig.suptitle(f'Average R = {average_r_value:.2f}, Average S = {average_s_value:.2f}', fontsize=30, fontweight='bold', y=0.92)

        output_dir = f'./results/{cfg["dataset"]}/{cfg["save_name"]}'
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, 'calibration.png')
        counter = 1

        while os.path.exists(output_path):
            output_path = os.path.join(output_dir, f'calibration_{counter}.png')
            counter += 1

        plt.savefig(output_path)  # Save in reduced quality to decrease file size
        plt.close()
        os.makedirs(output_dir, exist_ok=True)
        
        mean_thresholds = {i: np.mean(all_thresholds[i]) for i in range(1, num_classes) if all_thresholds[i]}
        mean_thresholds.update({i: None for i in range(1, num_classes) if not all_thresholds[i]})
        mean_acceptable = {i: np.mean(acceptable_region[i]) for i in range(1, num_classes) if acceptable_region[i]}
        mean_acceptable.update({i: None for i in range(1, num_classes) if not acceptable_region[i]})
        
        save_path = os.path.join(f'./results/{cfg["dataset"]}/{cfg["save_name"]}/calibration_thresholds.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if os.path.exists(save_path):
            with open(save_path, 'r') as json_file:
                data = json.load(json_file)
        else:
            data = {"set": [], "thresholds": {CLASSES[i]: [] for i in range(1, num_classes)}, "acceptable": {CLASSES[i]: [] for i in range(1, num_classes)}}

        data["set"].append(dataset_type)

        for i in range(1, num_classes):  # Use the index to get class name
            class_name = CLASSES[i]
            if mean_thresholds[i] is not None:
                data["thresholds"][class_name].append(mean_thresholds[i])
                data["acceptable"][class_name].append(mean_acceptable[i])

        # Write the updated data back to the JSON file
        with open(save_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
                
        bar()
        bar.title(f"{'      Calibration completed':35}")
    
    return mean_thresholds, mean_acceptable, mean_raw_dice


def generate_pseudolabels(
    model,
    cfg,
    device,
    unlabeled_images_dir,
    labeled_images_dir,
    labeled_masks_dir,
    uncertainty_thresholds,
    acceptable_region,
    unlabeled_to_evaluate,
    class_to_pseudolabel,
    class_acceptable,
):
    """Create pseudo‑labels from images in *Unlabeled/images* and move them to train/.
    Returns the number of new pseudo‑labels generated in this round.
    """

    monte_carlo_runs = cfg["montecarlo_runs"]

    # pool of available unlabeled images
    unlabeled_pool = [
        file for file in os.listdir(unlabeled_images_dir)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not unlabeled_pool:
        print("No images found in Unlabeled/images – skipping pseudo‑label stage.")
        return 0

    images_to_process = random.sample(unlabeled_pool, min(unlabeled_to_evaluate, len(unlabeled_pool)))

    # albumentations pipeline (unchanged)
    transform = A.Compose([
        A.Rotate(limit=10, p=0.2),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, p=0.2),
        A.RGBShift(r_shift_limit=8, g_shift_limit=5, b_shift_limit=5, p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.Sharpen(p=0.2),
        A.OpticalDistortion(p=0.2),
        A.ElasticTransform(p=0.2),
        A.GridDistortion(p=0.2),
        A.Perspective(p=0.2),
    ], p=1.0)

    # round log setup
    results_dir = os.path.join("./results", cfg["dataset"], cfg["save_name"])
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "pseudolabel_log.json")

    if os.path.exists(log_path):
        with open(log_path, "r") as file:
            log_data = json.load(file)
        round_id = len(log_data["rounds"]) + 1
    else:
        log_data = {"rounds": []}
        round_id = 1

    round_log = {"round_id": round_id, "pseudolabels": [], "count": 0}

    print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
    with alive_bar(len(images_to_process), spinner="flowers", bar="smooth", title="       Pseudolabeling".ljust(35)) as bar:
        for image_name in images_to_process:
            bar()
            image_path = os.path.join(unlabeled_images_dir, image_name)

            # ------------- model inference + uncertainty --------------------
            try:
                image_tensor = load_image(image_path, cfg["crop_size"]).to(device)
                prediction, uncertainty = inference_SSU(
                    model, image_tensor, monte_carlo_runs, device
                )
            except Exception as exc:
                print(f"            Inference error on {image_name}: {exc}")
                continue

            prediction = remove_small_components(prediction, cfg, min_size=100)

            # ------------- decision logic per class ------------------------
            num_pseudo      = 0
            num_ignored     = 0
            classes_in_mask = []
            acceptable_cls  = []
            pseudolabel_target_present = False  # at least one class in class_to_pseudolabel
            acceptable_criteria_ok     = True   # all low‑unc classes must be acceptable

            for class_idx in range(1, cfg["nclass"]):
                if not np.any(prediction == class_idx):
                    continue

                classes_in_mask.append(class_idx)

                if uncertainty[class_idx] < uncertainty_thresholds[class_idx]:
                    # confident enough ⇒ pseudo‑label candidate
                    num_pseudo += 1
                    if class_idx in class_to_pseudolabel:
                        pseudolabel_target_present = True

                elif uncertainty[class_idx] < acceptable_region[class_idx]:
                    # moderately confident ⇒ acceptable region
                    acceptable_cls.append(class_idx)
                    if class_idx not in class_acceptable:
                        acceptable_criteria_ok = False
                        break
                else:
                    # too uncertain ⇒ ignore
                    num_ignored += 1
                    if num_ignored > cfg["max_ignored"]:
                        break

            keep_image = (
                num_pseudo >= cfg["min_pseudol"]
                and num_ignored <= cfg["max_ignored"]
                and pseudolabel_target_present
                and acceptable_criteria_ok
            )
            if not keep_image:
                continue

            # ------------- save pseudo‑label + augmented image -------------
            augmented = transform(
                image=np.array(Image.open(image_path)),
                mask=prediction.astype(np.uint8)
            )
            pseudo_name = f"pseudo_{image_name}"
            out_image_path = os.path.join(labeled_images_dir,  pseudo_name)
            out_mask_path  = os.path.join(labeled_masks_dir,   pseudo_name)
            os.makedirs(labeled_images_dir, exist_ok=True)
            os.makedirs(labeled_masks_dir,  exist_ok=True)
            Image.fromarray(augmented["image"]).save(out_image_path)
            Image.fromarray(augmented["mask"]).save(out_mask_path)

            # ------------- optional inspection copy ------------------------
            inspection_img = os.path.join(cfg["data_root"], "OnlyPseudo", "images", pseudo_name)
            inspection_msk = os.path.join(cfg["data_root"], "OnlyPseudo", "masks",  pseudo_name)
            os.makedirs(os.path.dirname(inspection_img), exist_ok=True)
            os.makedirs(os.path.dirname(inspection_msk), exist_ok=True)
            shutil.copy(out_image_path, inspection_img)
            shutil.copy(out_mask_path,  inspection_msk)

            # move processed unlabeled image away from pool
            processed_dir = os.path.join(unlabeled_images_dir, "processed")
            os.makedirs(processed_dir, exist_ok=True)
            shutil.move(image_path, os.path.join(processed_dir, image_name))

            # ------------- update round log -------------------------------
            round_log["pseudolabels"].append(
                {
                    "image": out_image_path,
                    "mask": out_mask_path,
                    "classes_included": classes_in_mask,
                    "acceptable_classes": acceptable_cls,
                    "uncertainty": [float(uncertainty[c]) for c in classes_in_mask],
                }
            )
            round_log["count"] += 1
            bar.text(f"Generated {round_log['count']} pseudolabels")

    # save log
    log_data["rounds"].append(round_log)
    with open(log_path, "w") as file:
        json.dump(log_data, file, indent=4)

    print(f"Generated {round_log['count']} pseudo‑labels.")
    return round_log["count"]


def _gather_mask_paths(data_root, split):
    masks_dir = os.path.join(data_root, split, "masks")
    return [
        os.path.join(masks_dir, file)
        for file in os.listdir(masks_dir)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]


def compute_class_presence(
        cfg,
        validation=False,
        verbose=False
        ):

    split = "val" if validation else "train"
    mask_paths = _gather_mask_paths(cfg["data_root"], split)

    num_classes = cfg["nclass"]
    class_counts = [0] * num_classes

    with alive_bar(len(mask_paths), title="Computing Class Presence".ljust(35)) as bar:
        for mask_path in mask_paths:
            mask = np.array(Image.open(mask_path))
            for class_idx in np.unique(mask):
                if 0 <= class_idx < num_classes:
                    class_counts[class_idx] += 1
            bar()

    total_images = len(mask_paths) or 1  # prevent division by zero
    class_presence_fraction = [count / total_images for count in class_counts]

    if verbose:
        for idx, (name, count, presence) in enumerate(zip(cfg["classes"], class_counts, class_presence_fraction)):
            if idx == 0:  # skip background
                continue
            print(f"        {name}: {count} images ({presence*100:.2f}%)")

    presence_dict = {i: class_presence_fraction[i] for i in range(1, num_classes)}
    counts_dict   = {i: class_counts[i] for i in range(1, num_classes)}

    return presence_dict, counts_dict


def calibration_classes(
        mean_raw_dice,
        class_presence_dict,
        class_counts_dict,
        cfg
        ):

    class_indices = list(range(1, cfg['nclass']))  

    class_presence = [class_presence_dict[i] for i in class_indices]
    class_dice = [mean_raw_dice[i] for i in class_indices]
    class_labels = [cfg['classes'][i] for i in class_indices]

    output_dir = f'./results/{cfg["dataset"]}/{cfg["save_name"]}'
    os.makedirs(output_dir, exist_ok=True)

    class_dice = np.array(class_dice)
    class_presence = np.array(class_presence)

    # Perform K-Means clustering on Class Presence
    n_clusters = 3  # Clustering into thirds
    kmeans_presence = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(class_presence.reshape(-1, 1))
    presence_cluster_labels = kmeans_presence.labels_
 
    # Perform K-Means clustering on Dice scores
    kmeans_dice = KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit(class_dice.reshape(-1, 1))
    dice_cluster_labels = kmeans_dice.labels_

    # Get the indices for the lowest 1/3rd Class Presence (always pseudolabel)
    presence_cluster_means = [np.mean(class_presence[presence_cluster_labels == i]) for i in range(n_clusters)]
    presence_sorted_clusters = np.argsort(presence_cluster_means)
    low_presence_cluster = presence_sorted_clusters[0]  # Lowest 1/3rd in Presence

    # Get the indices for the lowest 1/3rd Dice scores (additional pseudolabeling)
    dice_cluster_means = [np.mean(class_dice[dice_cluster_labels == i]) for i in range(n_clusters)]
    dice_sorted_clusters = np.argsort(dice_cluster_means)
    low_dice_cluster = dice_sorted_clusters[0]  # Lowest 1/3rd in Dice scores

    # Classes that are in the lowest presence cluster (always pseudolabel)
    class_to_pseudolabel = [
        class_indices[i] for i in range(len(class_indices)) 
        if presence_cluster_labels[i] == low_presence_cluster
    ]

    # Add classes from the lowest dice cluster to pseudolabeling (even if not in low presence)
    additional_pseudolabel_classes = [
        class_indices[i] for i in range(len(class_indices)) 
        if dice_cluster_labels[i] == low_dice_cluster and class_indices[i] not in class_to_pseudolabel
    ]
    class_to_pseudolabel.extend(additional_pseudolabel_classes)

    # Classes that are in the top third for both presence and dice (acceptable)
    high_dice_cluster = dice_sorted_clusters[-1]
    high_presence_cluster = presence_sorted_clusters[-1]
    class_acceptable = [
        class_indices[i] for i in range(len(class_indices)) 
        if dice_cluster_labels[i] == high_dice_cluster and presence_cluster_labels[i] == high_presence_cluster
    ]

    # Save categorization
    categorization_save_path = os.path.join(output_dir, 'class_categorization.json')
    categorization_data = {
        "class_to_pseudolabel": class_to_pseudolabel,
        "class_acceptable": class_acceptable
    }

    if os.path.exists(categorization_save_path):
        with open(categorization_save_path, 'r') as json_file:
            existing_data = json.load(json_file)
        existing_data.append(categorization_data)
    else:
        existing_data = [categorization_data]

    with open(categorization_save_path, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)

    # Define fixed colors for the clusters (lowest to highest presence and dice)
    pseudolabel_color = '#ff0000'  # Red for classes to pseudolabel
    acceptable_color = '#008000'   # Green for acceptable classes
    other_color = '#ffa500'        # Orange for others

    # Create a color array
    cluster_colors = [
        pseudolabel_color if i in class_to_pseudolabel else 
        acceptable_color if i in class_acceptable else 
        other_color 
        for i in class_indices
    ]

    # Plot the clustered data with consistent colors for the categorization
    plt.figure(figsize=(10, 10))
    plt.scatter(class_presence, class_dice, c=cluster_colors, marker='o', alpha=0.5, edgecolors='black')

    for i, label in enumerate(class_labels):
        plt.annotate(label, (class_presence[i], class_dice[i]), fontsize=8, ha='center')

    plt.xlabel('Class Presence (%)')
    plt.ylabel('Average Dice Score')
    plt.title('Class Presence vs Average Dice Score with Double Clustering')
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Save the clustered scatter plot
    plot_save_path = os.path.join(output_dir, 'dice_vs_presence_double_clustering.png')
    counter = 1
    while os.path.exists(plot_save_path):
        plot_save_path = os.path.join(output_dir, f'dice_vs_presence_double_clustering_{counter}.png')
        counter += 1

    plt.savefig(plot_save_path)
    plt.close()

    return class_to_pseudolabel, class_acceptable
