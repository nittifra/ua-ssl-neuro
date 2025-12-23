import os
import json
import pprint

import numpy as np
from PIL import Image
from alive_progress import alive_bar
from colorama import Fore, Back, Style

import torch
from torch import nn
from torch.backends import cudnn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms

from util.functions import entropy_minimization, count_params, AverageMeter
from util.utils import calculate_dice
from model.deeplabv3vit import DeepLabV3Plus

MEAN = [0.45, 0.27, 0.25]
STD = [0.30, 0.23, 0.21]
NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

IMG_EXT = (".png", ".jpg", ".jpeg")

class SupervisedDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir: str, mask_dir: str, nclass: int = None, ignore_index: int = 255):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(IMG_EXT)])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.lower().endswith(IMG_EXT)])
        assert len(self.image_paths) == len(self.mask_paths), "Unequal number of images and masks."
        self.nclass = nclass
        self.ignore_index = ignore_index
        self._warned_invalid = False

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = NORMALIZE(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = torch.from_numpy(np.array(Image.open(self.mask_paths[idx]))).long()
        
        # Validate mask values if nclass is provided
        if self.nclass is not None:
            # Check for values >= nclass (excluding ignore_index) or negative values
            invalid_mask = ((mask >= self.nclass) | (mask < 0)) & (mask != self.ignore_index)
            if invalid_mask.any():
                if not self._warned_invalid:
                    invalid_values = torch.unique(mask[invalid_mask]).tolist()
                    print(f"{Fore.YELLOW}Warning: Found invalid mask values {invalid_values} in {self.mask_paths[idx]}. "
                          f"Expected values in [0, {self.nclass-1}] or {self.ignore_index}. Clamping to valid range.{Style.RESET_ALL}")
                    self._warned_invalid = True
                # Clamp invalid values to ignore_index
                mask[invalid_mask] = self.ignore_index
        
        return img, mask



def evaluate(model, loader, cfg, verbose=False):

    model.eval()
    dice_list = []

    with torch.no_grad():
        with alive_bar(len(loader), spinner="dots_waves2", bar="classic", title="       Evaluating".ljust(35)) as bar:
            for img, mask in loader:
                img = img.cuda()
                pred = model(img).argmax(dim=1).cpu().numpy()
                dice_list.append(calculate_dice(pred, mask.numpy(), cfg["nclass"]))
                bar()

    mean_dice = np.nanmean(dice_list, axis=0)
    overall = np.nanmean(mean_dice[1:])
    print(f"{Fore.GREEN}{Back.BLACK}{Style.BRIGHT} {overall:.4f}\n{Style.RESET_ALL}")
    if verbose:
        for i, c in enumerate(cfg["classes"]):
            print(f"       {c.ljust(35)}: {mean_dice[i]:.4f}")

    return overall



def forward_pass(model, img, mask, criterion, lambda_entropy):

    pred = model(img)
    ce_loss = criterion(pred, mask)
    ent_loss = entropy_minimization(pred, lambda_entropy)

    return ce_loss + ent_loss, ce_loss, ent_loss


def run_epoch(model, loader, optimizer, criterion, epoch_idx, ent_cfg):

    model.train()
    loss_meter, ce_meter, ent_meter = AverageMeter(), AverageMeter(), AverageMeter()
    lambda_base = ent_cfg["start_weight"]
    growth = ent_cfg["growth_rate"]
    max_w = ent_cfg["max_weight"]

    with alive_bar(len(loader), spinner="radioactive", bar="classic2", title=f"\n      Epoch {epoch_idx+1}/{ent_cfg['epochs']}".ljust(35)) as bar:
        for img, mask in loader:
            img, mask = img.cuda(), mask.cuda()
            lam = min(lambda_base * (1 + growth) ** epoch_idx, max_w)
            loss, ce, ent = forward_pass(model, img, mask, criterion, lam)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            loss_meter.update(loss.item()); ce_meter.update(ce.item()); ent_meter.update(ent.item())
            bar.text = f"Loss: {loss_meter.avg:.4f}" 
            bar()

    return loss_meter.avg, ce_meter.avg, ent_meter.avg


def save_checkpoint(state: dict, path: str, is_best: bool):
    torch.save(state, os.path.join(path, "latest.pth"))
    if is_best:
        torch.save(state, os.path.join(path, "best.pth"))
    if state["epoch"] % state["save_every"] == 0:
        torch.save(state, os.path.join(path, f"Epoch_{state['epoch']}.pth"))



def main(config, save_path, stopper=None, restart=False):
    os.makedirs(save_path, exist_ok=True)
    cudnn.enabled = cudnn.benchmark = True

    # Model & Optimiser
    model = DeepLabV3Plus(config).cuda()
    optimizer = SGD([
        {"params": model.backbone.parameters(), "lr": config["lr"], "name": "backbone"},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n],
            "lr": config["lr"] * config["lr_multi"],
            "name": "segmentation head",
        },
    ], lr=config["lr"], momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["lr_min"])

    # Dataloaders
    ignore_idx = config["criterion"]["kwargs"].get("ignore_index", 255)
    trainloader = DataLoader(SupervisedDataset(
        os.path.join(config["data_root"], "train/images"), 
        os.path.join(config["data_root"], "train/masks"),
        nclass=config["nclass"],
        ignore_index=ignore_idx
    ), batch_size=config["batch_size"], pin_memory=True, num_workers=2, drop_last=True,)
    valloader = DataLoader(SupervisedDataset(
        os.path.join(config["data_root"], "val/images"), 
        os.path.join(config["data_root"], "val/masks"),
        nclass=config["nclass"],
        ignore_index=ignore_idx
    ), batch_size=1, pin_memory=True, num_workers=1, drop_last=False,)

    # Criterion
    crit_cfg = config["criterion"]
    if crit_cfg["name"] == "CELoss":
        w = torch.tensor(np.array(crit_cfg["kwargs"].get("weight", [])), dtype=torch.float32) if "weight" in crit_cfg["kwargs"] else None
        criterion = nn.CrossEntropyLoss(ignore_index=crit_cfg["kwargs"]["ignore_index"], weight=w).cuda()
    else:
        raise NotImplementedError(f"Criterion {crit_cfg['name']} not implemented")

    # Resume if wanted
    best_dice, start_epoch = 0.0, -1
    if os.path.exists(os.path.join(save_path, "latest.pth")) and not restart:
        ckpt = torch.load(os.path.join(save_path, "latest.pth"), weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        best_dice = ckpt["previous_best_dice"]
        print(f"\n\nResumed from epoch {start_epoch + 1}\n")

    # ---------------- Book-keeping ----------------
    total_losses, ce_losses, entropy_losses = [], [], []

    print("Arguments:", pprint.pformat({"config": config, "save_path": save_path, "stopper": stopper}))
    n_params, n_trainable = count_params(model)
    print(f"Total params: {n_params:.1f}M | Trainable: {n_trainable:.1f}M\n")

    ent_cfg = {
        "start_weight": config["entropy_minimization"]["start_weight"],
        "growth_rate": config["entropy_minimization"]["growth_rate"],
        "max_weight": config["entropy_minimization"]["max_weight"],
        "epochs": config["epochs"],
    }

    # ---------------- Training loop ----------------
    for epoch in range(start_epoch + 1, config["epochs"]):
        total_loss_avg, ce_loss_avg, entropy_loss_avg = run_epoch(
            model, trainloader, optimizer, criterion, epoch, ent_cfg
        )
        scheduler.step()

        total_losses.append(total_loss_avg)
        ce_losses.append(ce_loss_avg)
        entropy_losses.append(entropy_loss_avg)

        # Validate every epoch
        verbose = epoch % 5 == 0
        mean_dice = evaluate(model, valloader, config, verbose)

        is_best = mean_dice > best_dice
        best_dice = max(best_dice, mean_dice)

        save_checkpoint(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best_dice": best_dice,
                "save_every": config["save_every"],
            },
            save_path,
            is_best,
        )

        if stopper is not None and (epoch - start_epoch) >= stopper:
            break

    # ---------------- Save loss curves ----------------
    with open(os.path.join(save_path, "losses.json"), "w") as fp:
        json.dump(
            {
                "total_losses": total_losses,
                "ce_losses": ce_losses,
                "entropy_losses": entropy_losses,
            },
            fp,
            indent=4,
        )


if __name__ == "__main__":
    print("This should not be run directly; use 'main.py' instead.")
