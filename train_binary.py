#training the binary models on combined dataset

"""
Binary Segmentation Training Script
=====================================
Trains binary models for building or woodland detection.
Supports two mask formats:
  - Class index masks (LandCover.ai): target_class=1 means building
  - Binary masks (INRIA/combined): 0=background, 255=positive

Usage:
    # Train on combined dataset (binary 0/255 masks):
    python train_binary.py --data_dir ./combined_building --time_budget 180 --task building --binary_masks

    # Train on LandCover.ai (class index masks):
    python train_binary.py --data_dir ./dataset --time_budget 180 --task building

    # Train both tasks on LandCover.ai:
    python train_binary.py --data_dir ./dataset --time_budget 180

Requirements:
    pip install segmentation-models-pytorch torch torchvision albumentations
"""

import argparse
import gc
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# DATASET
# ============================================================

class BinarySegDataset(Dataset):
    """
    Binary segmentation dataset. Supports two mask formats:
      - binary_masks=False: mask values are class indices, use target_class to select
      - binary_masks=True: mask values are 0/255, threshold to get binary
    """

    def __init__(self, data_dir, split, target_class=1, transform=None, binary_masks=False):
        self.data_dir = Path(data_dir) / split
        self.target_class = target_class
        self.transform = transform
        self.binary_masks = binary_masks

        self.images = sorted(
            list((self.data_dir / "images").glob("*.jpg")) +
            list((self.data_dir / "images").glob("*.png"))
        )
        self.masks = sorted((self.data_dir / "masks").glob("*.png"))

        assert len(self.images) == len(self.masks), \
            f"Mismatch: {len(self.images)} images vs {len(self.masks)} masks"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert("RGB"))
        mask = np.array(Image.open(self.masks[idx]))

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        if self.binary_masks:
            # Masks are 0/255 → convert to 0/1
            mask = (mask > 127).astype(np.uint8)
        else:
            # Masks are class indices → select target class
            mask = (mask == self.target_class).astype(np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0).float()


# ============================================================
# AUGMENTATIONS
# ============================================================

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ============================================================
# LOSS: BCE + DICE
# ============================================================

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, pos_weight=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        smooth = 1.0
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)
        dice_loss = 1.0 - dice
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


# ============================================================
# METRICS
# ============================================================

class BinaryMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, preds, targets):
        preds = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten().astype(int)
        self.tp += ((preds == 1) & (targets == 1)).sum()
        self.fp += ((preds == 1) & (targets == 0)).sum()
        self.fn += ((preds == 0) & (targets == 1)).sum()
        self.tn += ((preds == 0) & (targets == 0)).sum()

    def compute(self):
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-6)
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + 1e-6)
        return {"iou": iou, "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


# ============================================================
# MODEL BUILDER
# ============================================================

def build_model(arch, encoder):
    arch_map = {
        "Unet": smp.Unet,
        "DeepLabV3Plus": smp.DeepLabV3Plus,
        "FPN": smp.FPN,
        "MAnet": smp.MAnet,
        "LinkNet": smp.Linknet,
        "PSPNet": smp.PSPNet,
    }
    return arch_map[arch](
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )


# ============================================================
# TRAIN / VALIDATE
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            print(f"    batch {batch_idx+1}/{len(loader)}, loss: {loss.item():.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, metrics):
    model.eval()
    total_loss = 0
    metrics.reset()

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)

        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).long()
        metrics.update(preds, masks.long())

    return total_loss / len(loader), metrics.compute()


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ============================================================
# TRAIN ONE TASK
# ============================================================

def train_task(task_name, target_class, data_dir, arch, encoder, lr, batch_size,
               time_budget_seconds, save_dir, device, num_workers=4, patience=7,
               binary_masks=False):

    print("\n" + "=" * 60)
    print(f"TRAINING: {task_name.upper()} DETECTOR")
    print(f"  Target class: {target_class} ({task_name})")
    print(f"  Binary masks: {binary_masks}")
    print(f"  Architecture: {arch} + {encoder}")
    print(f"  LR: {lr}, Batch size: {batch_size}")
    print(f"  Time budget: {time_budget_seconds/60:.1f} min")
    print("=" * 60)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_dataset = BinarySegDataset(data_dir, "train", target_class,
                                     get_train_transforms(), binary_masks)
    val_dataset = BinarySegDataset(data_dir, "val", target_class,
                                   get_val_transforms(), binary_masks)
    print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Model
    clear_gpu()
    model = build_model(arch, encoder).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")

    # Loss
    if task_name == "building":
        pos_weight = torch.tensor([10.0]).to(device)
    else:
        pos_weight = torch.tensor([2.0]).to(device)

    criterion = BCEDiceLoss(bce_weight=0.5, pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scaler = torch.amp.GradScaler('cuda')
    metrics = BinaryMetrics()

    best_iou = 0.0
    epochs_without_improvement = 0
    training_start = time.time()
    epoch = 0

    while True:
        elapsed = time.time() - training_start
        remaining = time_budget_seconds - elapsed

        if remaining < 120:
            print(f"\n  Time budget reached after {epoch} epochs")
            break
        if epochs_without_improvement >= patience:
            print(f"\n  Early stopping after {epoch} epochs")
            break

        epoch += 1
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n  Epoch {epoch} (lr: {current_lr:.6f}, remaining: {remaining/60:.1f} min)")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_results = validate(model, val_loader, criterion, device, metrics)
        scheduler.step()

        iou = val_results["iou"]
        print(f"    Train Loss: {train_loss:.4f}")
        print(f"    Val Loss: {val_loss:.4f} | IoU: {iou:.4f} | "
              f"Precision: {val_results['precision']:.4f} | "
              f"Recall: {val_results['recall']:.4f} | "
              f"F1: {val_results['f1']:.4f}")

        if iou > best_iou:
            best_iou = iou
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou": best_iou,
                "arch": arch,
                "encoder": encoder,
                "task": task_name,
                "target_class": target_class,
            }, save_dir / f"best_{task_name}.pth")
            print(f"    *** New best {task_name} model! IoU: {best_iou:.4f} ***")
        else:
            epochs_without_improvement += 1
            print(f"    No improvement ({epochs_without_improvement}/{patience})")

    total_time = time.time() - training_start
    print(f"\n  {task_name} training done in {total_time/60:.1f} min | Best IoU: {best_iou:.4f}")

    del model, optimizer, scaler
    clear_gpu()

    return best_iou


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train binary segmentation models")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--time_budget", type=int, default=180, help="Total time in MINUTES")
    parser.add_argument("--arch", type=str, default="FPN")
    parser.add_argument("--encoder", type=str, default="efficientnet-b3")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="checkpoints_binary")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--task", type=str, default="both",
                        choices=["both", "building", "woodland"],
                        help="Which model(s) to train")
    parser.add_argument("--binary_masks", action="store_true",
                        help="Use if masks are binary 0/255 (combined dataset)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    total_budget = args.time_budget * 60
    global_start = time.time()

    results = {}

    if args.task in ["both", "building"]:
        budget = total_budget // 2 if args.task == "both" else total_budget - 60
        building_iou = train_task(
            "building", target_class=1,
            data_dir=args.data_dir, arch=args.arch, encoder=args.encoder,
            lr=args.lr, batch_size=args.batch_size,
            time_budget_seconds=budget, save_dir=args.save_dir,
            device=device, num_workers=args.num_workers, patience=args.patience,
            binary_masks=args.binary_masks,
        )
        results["building"] = building_iou

    if args.task in ["both", "woodland"]:
        elapsed = time.time() - global_start
        budget = total_budget - elapsed - 60 if args.task == "both" else total_budget - 60
        woodland_iou = train_task(
            "woodland", target_class=2,
            data_dir=args.data_dir, arch=args.arch, encoder=args.encoder,
            lr=args.lr, batch_size=args.batch_size,
            time_budget_seconds=budget, save_dir=args.save_dir,
            device=device, num_workers=args.num_workers, patience=args.patience,
            binary_masks=args.binary_masks,
        )
        results["woodland"] = woodland_iou

    total_time = time.time() - global_start
    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print(f"Total time: {total_time/60:.1f} minutes")
    for task, iou in results.items():
        print(f"  {task}: Best IoU = {iou:.4f}")
        print(f"    Checkpoint: {args.save_dir}/best_{task}.pth")


if __name__ == "__main__":
    main()