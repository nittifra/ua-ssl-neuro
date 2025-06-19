"""
@ Nitti Francesco
Folder structure expected:
DATA/
 ├─ train/  ── images/ , masks/
 ├─ val/    ── images/ , masks/
 └─ unlabeled/ ── images/
"""

import os
import shutil
import yaml
import torch

from util.train import main as train_main  
from util.functions import (
    generate_pseudolabels,       
    calibration_thresholds,
    compute_class_presence,       
    calibration_classes,
)
from model.deeplabv3vit import DeepLabV3Plus


def execute_training_schedule(cfg: dict, save_path: str):
    """Run the multi‑stage schedule defined in ``cfg['training_config']['schedule']``."""

    training_counter = 0
    pseudolabeling_counter = 0

    schedule = cfg["training_config"]["schedule"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepLabV3Plus(cfg).to(device)
    best_model_path = os.path.join(save_path, "best.pth")

    # ------------------------------------------------ Resume if possible ----
    latest_models = sorted([f for f in os.listdir(save_path) if f.startswith("best_training")],reverse=True,)
    latest_checkpoint = (os.path.join(save_path, latest_models[0]) if latest_models else best_model_path if os.path.exists(best_model_path) else None)
    if latest_checkpoint:
        print(f"Resuming model weights from {latest_checkpoint}")
        ckpt = torch.load(latest_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    # ------------------------------------------------ Iterate over stages ---
    for stage in schedule:
        stage_type = stage["type"].upper()

        # BASE SUPERVISION 
        if stage_type == "BS":
            epochs = stage["epochs"]
            restart_flag = stage.get("restart", False)

            print(f"\n▶ Base Supervision for {epochs} epochs (restart={restart_flag})")
            train_main(cfg, save_path, stopper=epochs, restart=restart_flag)

            # Snapshot best.pth for bookkeeping
            if os.path.exists(best_model_path):
                snapshot = os.path.join(save_path, f"best_training_{training_counter}.pth")
                shutil.copyfile(best_model_path, snapshot)
                print(f"Saved snapshot → {snapshot}")
            else:
                print("Warning: best.pth not found after BS stage.")
            training_counter += 1

            # Reload best weights for next stage
            if os.path.exists(best_model_path):
                ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model"])

        # ONLY PSEUDOLABELING 
        elif stage_type == "OPL":
            samples = stage["samples"]              # how many unlabeled images to try
            set_type = cfg["calibrate_on"]          # "train" or "val" -> used only for calibration split

            # Folder paths in new structure
            data_root = cfg["data_root"]
            labeled_images_dir = os.path.join(data_root, "train", "images")
            labeled_masks_dir  = os.path.join(data_root, "train", "masks")
            unlabeled_images_dir = os.path.join(data_root, "unlabeled", "images")

            # choose calibration images/masks (only affects threshold computation)
            if set_type == "val":
                calib_images_dir = os.path.join(data_root, "val", "images")
                calib_masks_dir  = os.path.join(data_root, "val", "masks")
            elif set_type == "train":
                calib_images_dir = labeled_images_dir
                calib_masks_dir  = labeled_masks_dir
            else:
                raise ValueError(f"Invalid 'calibrate_on' value: {set_type}")

            # --- optional per‑class stats on current labeled masks
            try:
                class_presence, class_counts = compute_class_presence(cfg, verbose=True)
            except Exception as e:
                print(f"Class‑presence computation failed: {e}")
                class_presence, class_counts = {}, {}

            # --- Calibration
            print(f"Calibrating on {calib_images_dir} …")

            unc_thr, acc_region, mean_raw_dice = calibration_thresholds(model, calib_images_dir, calib_masks_dir, cfg, device)

            class_to_pl, class_acceptable = calibration_classes(mean_raw_dice, class_presence, class_counts, cfg)

            # Fallback if thresholds contain None
            if None in unc_thr.values() and set_type == "val":
                print("Thresholds contained None, recalibrating on train set")
                unc_thr, acc_region, _ = calibration_thresholds(model, labeled_images_dir, labeled_masks_dir, cfg, device)
                if None in unc_thr.values():
                    print("Calibration failed again, skipping OPL stage")
                    continue

            generate_pseudolabels(
                model=model,
                cfg=cfg,
                device=device,
                unlabeled_images_dir=unlabeled_images_dir,
                labeled_images_dir=labeled_images_dir,
                labeled_masks_dir=labeled_masks_dir,
                uncertainty_thresholds=unc_thr,
                acceptable_region=acc_region,
                unlabeled_to_evaluate=samples,
                class_to_pseudolabel=class_to_pl,
                class_acceptable=class_acceptable,
            )
            pseudolabeling_counter += 1

        # -------------------------------------------------------------------
        else:
            raise ValueError(f"Unknown stage type: {stage_type}")

    print("\n✔ Training schedule completed.")


# ───────────────────────────── YAML Loader ──────────────────────────────────

def load_config(yaml_path: str):
    with open(yaml_path, "r") as fp:
        return yaml.safe_load(fp)


# ─────────────────────────────────── CLI ─────────────────────────────────────

def main():
    dataset = "custom"
    cfg = load_config(f"./configs/{dataset}.yaml")

    save_path = os.path.join("./results", dataset, cfg["save_name"])
    os.makedirs(save_path, exist_ok=True)

    execute_training_schedule(cfg, save_path)


if __name__ == "__main__":
    main()
