# for finding the best model/approach using some hyperparam tuning + comp analysis of different models




"""
Auto-Tuning Training Script for LandCover.ai
==============================================
Two-phase approach that fits within a time budget:

Phase 1 (SEARCH): Try many architecture/encoder/hyperparam combos using a
    subset of training data and 3 quick epochs per config. (~2 hours)

Phase 2 (FULL TRAIN): Take the best config from Phase 1 and train on the
    full dataset for the remaining time budget. (~1 hour)

Architectures tested: U-Net, DeepLabV3+, FPN, MAnet, LinkNet, PSPNet
Encoders tested: resnet34, resnet50, efficientnet-b3, mobilenet_v2
Learning rates tested: 1e-3, 1e-4, 5e-5

Usage:
    python auto_tune.py --data_dir ./dataset --time_budget 180

    # Custom time budget in minutes:
    python auto_tune.py --data_dir ./dataset --time_budget 120

    # Skip search, train best config directly:
    python auto_tune.py --data_dir ./dataset --time_budget 180 --skip_search --arch Unet --encoder resnet34 --lr 1e-4

Requirements:
    pip install segmentation-models-pytorch torch torchvision albumentations
"""

import argparse
import csv
import gc
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# DATASET
# ============================================================

class LandCoverDataset(Dataset):
    CLASS_NAMES = ["background", "building", "woodland", "water", "road"]
    NUM_CLASSES = 5

    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.images = sorted((self.data_dir / "images").glob("*.jpg"))
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
        mask = np.clip(mask, 0, self.NUM_CLASSES - 1)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        return image, mask.long()


def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ============================================================
# METRICS
# ============================================================

class SegmentationMetrics:
    def __init__(self, num_classes, class_names):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()

    def reset(self):
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds, targets):
        preds = preds.cpu().numpy().flatten()
        targets = targets.cpu().numpy().flatten()
        valid = targets < self.num_classes
        preds = preds[valid]
        targets = targets[valid]
        for t, p in zip(targets, preds):
            self.confusion[t, p] += 1

    def compute(self):
        iou_per_class = {}
        for i in range(self.num_classes):
            tp = self.confusion[i, i]
            fp = self.confusion[:, i].sum() - tp
            fn = self.confusion[i, :].sum() - tp
            denom = tp + fp + fn
            iou_per_class[self.class_names[i]] = tp / denom if denom > 0 else float("nan")

        valid_ious = [v for v in iou_per_class.values() if not np.isnan(v)]
        mean_iou = np.mean(valid_ious) if valid_ious else 0.0
        accuracy = np.diag(self.confusion).sum() / (self.confusion.sum() + 1e-6)
        return {"mean_iou": mean_iou, "accuracy": accuracy, "per_class_iou": iou_per_class}


# ============================================================
# MODEL BUILDER
# ============================================================

def build_model(arch, encoder, num_classes=5):
    """Build a segmentation model from architecture name and encoder."""
    arch_map = {
        "Unet": smp.Unet,
        "DeepLabV3Plus": smp.DeepLabV3Plus,
        "FPN": smp.FPN,
        "MAnet": smp.MAnet,
        "LinkNet": smp.Linknet,
        "PSPNet": smp.PSPNet,
    }

    if arch not in arch_map:
        raise ValueError(f"Unknown architecture: {arch}. Choose from: {list(arch_map.keys())}")

    model = arch_map[arch](
        encoder_name=encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
    return model


# ============================================================
# TRAIN / VALIDATE FUNCTIONS
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0

    for images, masks in loader:
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
        preds = outputs.argmax(dim=1)
        metrics.update(preds, masks)

    return total_loss / len(loader), metrics.compute()


def clear_gpu():
    """Free GPU memory between configs."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ============================================================
# PHASE 1: SEARCH
# ============================================================

def run_search(data_dir, device, search_time_budget, num_workers=4):
    """
    Try many configs with a subset of training data.
    Returns sorted list of (config, val_miou) results.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: ARCHITECTURE & HYPERPARAMETER SEARCH")
    print("=" * 60)

    # Load full datasets
    full_train = LandCoverDataset(data_dir, "train", get_train_transforms())
    val_dataset = LandCoverDataset(data_dir, "val", get_val_transforms())
    print(f"Full training set: {len(full_train)} samples")
    print(f"Validation set: {len(val_dataset)} samples")

    # Use 20% subset for search (faster iterations)
    subset_size = len(full_train) // 5
    subset_indices = random.sample(range(len(full_train)), subset_size)
    train_subset = Subset(full_train, subset_indices)
    print(f"Search subset: {subset_size} samples")

    # Also use a smaller val subset for speed
    val_subset_size = min(400, len(val_dataset))
    val_indices = random.sample(range(len(val_dataset)), val_subset_size)
    val_subset = Subset(val_dataset, val_indices)

    # Define search space
    # Ordered by priority — most promising configs first
    configs = [
        # Architecture     Encoder            LR     Batch
        ("Unet",           "resnet34",        1e-4,  4),
        ("Unet",           "resnet34",        1e-3,  4),
        ("DeepLabV3Plus",  "resnet34",        1e-4,  2),
        ("Unet",           "resnet50",        1e-4,  2),
        ("FPN",            "resnet34",        1e-4,  4),
        ("DeepLabV3Plus",  "resnet50",        1e-4,  2),
        ("Unet",           "efficientnet-b3", 1e-4,  2),
        ("MAnet",          "resnet34",        1e-4,  2),
        ("Unet",           "resnet34",        5e-5,  4),
        ("FPN",            "resnet50",        1e-4,  2),
        ("LinkNet",        "resnet34",        1e-4,  4),
        ("PSPNet",         "resnet34",        1e-4,  4),
        ("DeepLabV3Plus",  "efficientnet-b3", 1e-4,  2),
        ("Unet",           "mobilenet_v2",    1e-4,  4),
        ("FPN",            "efficientnet-b3", 1e-4,  2),
    ]

    search_epochs = 3  # Quick eval per config
    class_weights = torch.tensor([0.5, 5.0, 1.0, 3.0, 3.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    metrics = SegmentationMetrics(5, LandCoverDataset.CLASS_NAMES)

    results = []
    search_start = time.time()

    for i, (arch, encoder, lr, batch_size) in enumerate(configs):
        # Check time budget
        elapsed = time.time() - search_start
        remaining = search_time_budget - elapsed
        if remaining < 120:  # Need at least 2 min for a config
            print(f"\nTime budget reached after {i} configs ({elapsed/60:.1f} min)")
            break

        config_name = f"{arch}_{encoder}_lr{lr}_bs{batch_size}"
        print(f"\n[{i+1}/{len(configs)}] Testing: {config_name}")
        print(f"  Time elapsed: {elapsed/60:.1f} min, remaining: {remaining/60:.1f} min")

        try:
            clear_gpu()

            # Build model
            model = build_model(arch, encoder).to(device)
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {param_count:,}")

            # Data loaders
            train_loader = DataLoader(
                train_subset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=True, drop_last=True,
            )
            val_loader = DataLoader(
                val_subset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True,
            )

            # Optimizer and scaler
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            scaler = torch.amp.GradScaler('cuda')

            # Quick training
            best_val_miou = 0.0
            config_start = time.time()

            for epoch in range(search_epochs):
                train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
                val_loss, val_results = validate(model, val_loader, criterion, device, metrics)
                val_miou = val_results["mean_iou"]

                if val_miou > best_val_miou:
                    best_val_miou = val_miou

                print(f"  Epoch {epoch+1}/{search_epochs}: "
                      f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                      f"val_mIoU={val_miou:.4f}")

            config_time = time.time() - config_start

            result = {
                "arch": arch,
                "encoder": encoder,
                "lr": lr,
                "batch_size": batch_size,
                "best_val_miou": best_val_miou,
                "val_accuracy": val_results["accuracy"],
                "per_class_iou": val_results["per_class_iou"],
                "params": param_count,
                "time_seconds": config_time,
            }
            results.append(result)

            building_iou = val_results["per_class_iou"].get("building", 0)
            woodland_iou = val_results["per_class_iou"].get("woodland", 0)
            print(f"  RESULT: mIoU={best_val_miou:.4f} | "
                  f"building={building_iou:.4f} | woodland={woodland_iou:.4f} | "
                  f"time={config_time:.0f}s")

            # Cleanup
            del model, optimizer, scaler, train_loader, val_loader
            clear_gpu()

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM! Skipping {config_name}")
            clear_gpu()
            continue
        except Exception as e:
            print(f"  ERROR: {e}. Skipping {config_name}")
            clear_gpu()
            continue

    # Sort by best val mIoU
    results.sort(key=lambda x: x["best_val_miou"], reverse=True)

    # Print leaderboard
    print("\n" + "=" * 60)
    print("SEARCH RESULTS (ranked by val mIoU)")
    print("=" * 60)
    print(f"{'Rank':<5} {'Architecture':<16} {'Encoder':<18} {'LR':<10} {'BS':<4} {'mIoU':<8} {'Build IoU':<10} {'Wood IoU':<10} {'Time':<8}")
    print("-" * 99)

    for rank, r in enumerate(results, 1):
        b_iou = r["per_class_iou"].get("building", float("nan"))
        w_iou = r["per_class_iou"].get("woodland", float("nan"))
        b_str = f"{b_iou:.4f}" if not np.isnan(b_iou) else "N/A"
        w_str = f"{w_iou:.4f}" if not np.isnan(w_iou) else "N/A"
        print(f"{rank:<5} {r['arch']:<16} {r['encoder']:<18} {r['lr']:<10.0e} {r['batch_size']:<4} "
              f"{r['best_val_miou']:<8.4f} {b_str:<10} {w_str:<10} {r['time_seconds']:<8.0f}s")

    total_search_time = time.time() - search_start
    print(f"\nSearch completed in {total_search_time/60:.1f} minutes")

    return results


# ============================================================
# PHASE 2: FULL TRAINING
# ============================================================

def run_full_training(data_dir, config, device, time_budget_seconds, save_dir, num_workers=4):
    """
    Train the best config on full dataset for remaining time budget.
    """
    print("\n" + "=" * 60)
    print("PHASE 2: FULL TRAINING WITH BEST CONFIG")
    print("=" * 60)
    print(f"Architecture: {config['arch']}")
    print(f"Encoder: {config['encoder']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Time budget: {time_budget_seconds/60:.1f} minutes")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Full datasets
    train_dataset = LandCoverDataset(data_dir, "train", get_train_transforms())
    val_dataset = LandCoverDataset(data_dir, "val", get_val_transforms())

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # Build model
    clear_gpu()
    model = build_model(config["arch"], config["encoder"]).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss, optimizer, scheduler, scaler
    class_weights = torch.tensor([0.5, 5.0, 1.0, 3.0, 3.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    scaler = torch.amp.GradScaler('cuda')

    metrics = SegmentationMetrics(5, LandCoverDataset.CLASS_NAMES)

    best_miou = 0.0
    patience = 5
    epochs_without_improvement = 0
    training_start = time.time()
    epoch = 0

    while True:
        epoch_start = time.time()
        elapsed_total = time_start = time.time() - training_start
        remaining = time_budget_seconds - elapsed_total

        # Stop if less than 2 minutes remaining or patience exhausted
        if remaining < 120:
            print(f"\nTime budget reached after {epoch} epochs")
            break

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping after {epoch} epochs (no improvement for {patience} epochs)")
            break

        epoch += 1
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch} (lr: {current_lr:.6f}, remaining: {remaining/60:.1f} min)")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)

        # Validate
        val_loss, val_results = validate(model, val_loader, criterion, device, metrics)
        val_miou = val_results["mean_iou"]

        scheduler.step()

        # Print results
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Accuracy: {val_results['accuracy']:.4f} | mIoU: {val_miou:.4f}")
        for cls_name, iou in val_results["per_class_iou"].items():
            if not np.isnan(iou):
                print(f"    {cls_name:12s}: IoU = {iou:.4f}")
            else:
                print(f"    {cls_name:12s}: IoU = N/A")

        # Save best
        if val_miou > best_miou:
            best_miou = val_miou
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_miou": best_miou,
                "arch": config["arch"],
                "encoder": config["encoder"],
                "lr": config["lr"],
                "batch_size": config["batch_size"],
            }, save_dir / "best_model.pth")
            print(f"  *** New best model! mIoU: {best_miou:.4f} ***")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement ({epochs_without_improvement}/{patience})")

        # Save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_miou": best_miou,
            "arch": config["arch"],
            "encoder": config["encoder"],
            "lr": config["lr"],
            "batch_size": config["batch_size"],
        }, save_dir / "latest.pth")

        epoch_time = time.time() - epoch_start
        print(f"  Epoch time: {epoch_time:.1f}s")

    total_time = time.time() - training_start
    print(f"\nFull training completed in {total_time/60:.1f} minutes")
    print(f"Best val mIoU: {best_miou:.4f}")
    print(f"Best model saved to: {save_dir / 'best_model.pth'}")

    return best_miou


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Auto-tune and train on LandCover.ai")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset/")
    parser.add_argument("--time_budget", type=int, default=180, help="Total time budget in MINUTES")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--search_fraction", type=float, default=0.6,
                        help="Fraction of time budget for search (default 0.6 = 60%%)")

    # Skip search and train directly
    parser.add_argument("--skip_search", action="store_true",
                        help="Skip search, train with specified config")
    parser.add_argument("--arch", type=str, default="Unet")
    parser.add_argument("--encoder", type=str, default="resnet34")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    total_budget_seconds = args.time_budget * 60
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    global_start = time.time()

    if args.skip_search:
        # Skip search, train directly
        best_config = {
            "arch": args.arch,
            "encoder": args.encoder,
            "lr": args.lr,
            "batch_size": args.batch_size,
        }
        train_budget = total_budget_seconds - 60  # 1 min buffer
    else:
        # Phase 1: Search
        search_budget = total_budget_seconds * args.search_fraction
        print(f"\nTotal time budget: {args.time_budget} minutes")
        print(f"Search budget: {search_budget/60:.0f} minutes")
        print(f"Training budget: {(total_budget_seconds - search_budget)/60:.0f} minutes")

        results = run_search(args.data_dir, device, search_budget, args.num_workers)

        # Save search results to CSV
        csv_path = save_dir / "search_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "arch", "encoder", "lr", "batch_size",
                             "val_miou", "val_accuracy", "building_iou",
                             "woodland_iou", "params", "time_seconds"])
            for rank, r in enumerate(results, 1):
                b_iou = r["per_class_iou"].get("building", "")
                w_iou = r["per_class_iou"].get("woodland", "")
                writer.writerow([rank, r["arch"], r["encoder"], r["lr"],
                                 r["batch_size"], f"{r['best_val_miou']:.4f}",
                                 f"{r['val_accuracy']:.4f}",
                                 f"{b_iou:.4f}" if isinstance(b_iou, float) and not np.isnan(b_iou) else "N/A",
                                 f"{w_iou:.4f}" if isinstance(w_iou, float) and not np.isnan(w_iou) else "N/A",
                                 r["params"], f"{r['time_seconds']:.0f}"])
        print(f"\nSearch results saved to: {csv_path}")

        # Save results as JSON too
        json_results = []
        for r in results:
            jr = dict(r)
            jr["per_class_iou"] = {k: (v if not np.isnan(v) else None) for k, v in r["per_class_iou"].items()}
            json_results.append(jr)
        with open(save_dir / "search_results.json", "w") as f:
            json.dump(json_results, f, indent=2)

        if not results:
            print("No valid configs found! Check your GPU memory.")
            return

        best_config = {
            "arch": results[0]["arch"],
            "encoder": results[0]["encoder"],
            "lr": results[0]["lr"],
            "batch_size": results[0]["batch_size"],
        }

        elapsed_search = time.time() - global_start
        train_budget = total_budget_seconds - elapsed_search - 60  # 1 min buffer

    # Phase 2: Full training
    if train_budget > 120:  # At least 2 minutes
        best_miou = run_full_training(
            args.data_dir, best_config, device, train_budget,
            save_dir, args.num_workers
        )
    else:
        print("\nNot enough time remaining for full training.")
        print("Use --skip_search with the best config to do a dedicated training run.")

    # Final summary
    total_time = time.time() - global_start
    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best config: {best_config}")
    print(f"Checkpoints saved in: {save_dir}/")
    print(f"\nNext steps:")
    print(f"  1. Check search_results.csv for full comparison")
    print(f"  2. Evaluate: python eval_unet.py --checkpoint {save_dir}/best_model.pth --mode test --data_dir {args.data_dir}")
    print(f"  3. Infer on Cumberland: python eval_unet.py --checkpoint {save_dir}/best_model.pth --mode infer --tiles_dir ./filtered/has_structures")


if __name__ == "__main__":
    main()