"""
Binary Segmentation Training — ConvNeXt-Base + MLP Decoder
==========================================================
ConvNeXt-Base encoder (88M params) with lightweight MLP decoder,
differential learning rates, and EMA weight averaging.

Optimized for RTX 2060 (6GB VRAM) with mixed precision training.

Features:
  - ConvNeXt-Base encoder (ImageNet pretrained, 88M params)
  - Lightweight MLP decoder with spatial dropout
  - Differential learning rates (encoder lr/10, decoder full lr)
  - EMA (Exponential Moving Average) of model weights
  - Focal + Dice loss for class imbalance
  - Strong data augmentation (13 transforms)
  - OneCycleLR with cosine annealing + warmup
  - Gradient accumulation (effective batch size = 8)
  - Training AND validation metrics per epoch
  - CSV training log for plotting learning curves
  - Held-out test set evaluation after training

Usage:
    # Train building detector on combined dataset:
    python train_segformer.py --data_dir ./combined_building --task building --binary_masks

    # Train woodland detector on LandCover.ai:
    python train_segformer.py --data_dir ./dataset --task woodland

Requirements:
    pip install torch torchvision timm albumentations
"""

import argparse
import copy
import csv
import gc
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# MODEL: ConvNeXt-Base + MLP Decoder
# ============================================================

class MLPDecoder(nn.Module):
    """
    Lightweight MLP decoder.
    Takes multi-scale features from encoder, projects each to embed_dim,
    upsamples all to 1/4 resolution, concatenates, fuses, and classifies.
    """

    def __init__(self, encoder_channels, embed_dim=256, num_classes=1, dropout=0.1):
        super().__init__()

        # Project each scale to embed_dim
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            ) for ch in encoder_channels
        ])

        # Fuse concatenated features
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * len(encoder_channels), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

        # Final classifier
        self.classifier = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, features):
        target_size = features[0].shape[2:]  # 1/4 resolution

        projected = []
        for feat, proj in zip(features, self.projections):
            x = proj(feat)
            if x.shape[2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            projected.append(x)

        x = torch.cat(projected, dim=1)
        x = self.fusion(x)
        return self.classifier(x)


class SegModel(nn.Module):
    """
    ConvNeXt-Base encoder + MLP decoder.

    ConvNeXt-Base outputs multi-scale features at [1/4, 1/8, 1/16, 1/32]
    with channels [128, 256, 512, 1024].
    """

    def __init__(self, encoder_name='convnext_base', num_classes=1, dropout=0.1, pretrained=True):
        super().__init__()

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=True,
        )
        encoder_channels = self.encoder.feature_info.channels()

        self.decoder = MLPDecoder(
            encoder_channels=encoder_channels,
            embed_dim=256,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, x):
        input_size = x.shape[2:]
        features = self.encoder(x)
        logits = self.decoder(features)
        logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        return logits


# ============================================================
# EMA (Exponential Moving Average)
# ============================================================

class ModelEMA:
    """
    Maintains an exponential moving average of model parameters.
    The EMA model is used for evaluation — it produces smoother,
    more stable predictions than the raw training model.
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)


# ============================================================
# DATASET
# ============================================================

class BinarySegDataset(Dataset):
    """
    Binary segmentation dataset.
      - binary_masks=False: mask values are class indices, use target_class
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
            f"Mismatch: {len(self.images)} images vs {len(self.masks)} masks in {self.data_dir}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert("RGB"))
        mask = np.array(Image.open(self.masks[idx]))

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        if self.binary_masks:
            mask = (mask > 127).astype(np.uint8)
        else:
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
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(-0.1, 0.1),
            rotate=(-20, 20),
            shear=(-5, 5),
            p=0.5,
        ),
        A.ElasticTransform(p=0.2),
        A.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0), p=0.3),

        # Color / intensity
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),

        # Noise / blur / dropout
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.CoarseDropout(p=0.2),

        # Normalize
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ============================================================
# LOSS: FOCAL + DICE
# ============================================================

class FocalDiceLoss(nn.Module):
    """
    Combined Focal Loss + Dice Loss.
    - Focal loss (gamma=2.0): downweights easy pixels, focuses on hard ones
    - Dice loss: directly optimizes region overlap (IoU-like)
    """

    def __init__(self, focal_weight=0.5, alpha=0.25, gamma=2.0, smooth=1.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        # Focal loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        focal_loss = focal.mean()

        # Dice loss
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        dice_loss = 1.0 - dice

        return self.focal_weight * focal_loss + (1 - self.focal_weight) * dice_loss


# ============================================================
# METRICS
# ============================================================

class BinaryMetrics:
    """Tracks TP/FP/FN/TN across batches, computes IoU, accuracy, etc."""

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
        total = self.tp + self.fp + self.fn + self.tn + 1e-8
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-8)
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (self.tp + self.tn) / total
        return {
            "iou": iou,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


# ============================================================
# TRAIN / VALIDATE
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler,
                    device, metrics, ema=None, accumulate_steps=1):
    model.train()
    metrics.reset()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()

    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks) / accumulate_steps

        scaler.scale(loss).backward()

        # Track training metrics
        with torch.no_grad():
            preds = (torch.sigmoid(outputs) > 0.5).long()
            metrics.update(preds, masks.long())

        total_loss += loss.item() * accumulate_steps
        num_batches += 1

        # Optimizer step every accumulate_steps batches
        if (batch_idx + 1) % accumulate_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)

        if (batch_idx + 1) % 200 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"    batch {batch_idx+1}/{len(loader)}, "
                  f"loss: {loss.item() * accumulate_steps:.4f}, lr: {current_lr:.2e}")

    # Handle remaining accumulated gradients
    if num_batches % accumulate_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)

    return total_loss / num_batches


@torch.no_grad()
def validate(model, loader, criterion, device, metrics):
    model.eval()
    metrics.reset()
    total_loss = 0
    num_batches = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, masks)

        total_loss += loss.item()
        num_batches += 1
        preds = (torch.sigmoid(outputs) > 0.5).long()
        metrics.update(preds, masks.long())

    return total_loss / num_batches, metrics.compute()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="ConvNeXt-Base Binary Segmentation Training")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Dataset directory with train/val/test splits")
    parser.add_argument("--task", type=str, required=True, choices=["building", "woodland"],
                        help="Which task to train")
    parser.add_argument("--binary_masks", action="store_true",
                        help="Use if masks are binary 0/255 (combined dataset)")
    parser.add_argument("--encoder", type=str, default="convnext_base",
                        help="Encoder name from timm (default: convnext_base)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=6e-5,
                        help="Peak learning rate (decoder). Encoder gets lr/10.")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="AdamW weight decay (L2 regularization)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Spatial dropout rate in decoder")
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay rate (0 to disable)")
    parser.add_argument("--accumulate_steps", type=int, default=2,
                        help="Gradient accumulation steps (effective batch = batch_size * this)")
    parser.add_argument("--warmup_pct", type=float, default=0.05,
                        help="Fraction of training for LR warmup")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (epochs without val IoU improvement)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--save_dir", type=str, default="checkpoints_segformer",
                        help="Directory to save checkpoints and logs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory
        print(f"VRAM: {total_mem / 1024**3:.1f} GB")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Target class for class-index masks
    target_class = 1 if args.task == "building" else 2

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------
    train_dataset = BinarySegDataset(
        args.data_dir, "train", target_class, get_train_transforms(), args.binary_masks)
    val_dataset = BinarySegDataset(
        args.data_dir, "val", target_class, get_val_transforms(), args.binary_masks)
    test_dataset = BinarySegDataset(
        args.data_dir, "test", target_class, get_val_transforms(), args.binary_masks)

    total_samples = len(train_dataset) + len(val_dataset) + len(test_dataset)
    print(f"\nDataset: {args.data_dir}")
    print(f"  Task: {args.task} (target_class={target_class}, binary_masks={args.binary_masks})")
    print(f"  Train: {len(train_dataset):,} ({len(train_dataset)/total_samples*100:.0f}%)")
    print(f"  Val:   {len(val_dataset):,} ({len(val_dataset)/total_samples*100:.0f}%)")
    print(f"  Test:  {len(test_dataset):,} ({len(test_dataset)/total_samples*100:.0f}%)")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    gc.collect()
    torch.cuda.empty_cache()

    model = SegModel(
        encoder_name=args.encoder,
        num_classes=1,
        dropout=args.dropout,
        pretrained=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    encoder_channels = model.encoder.feature_info.channels()

    print(f"\nModel: ConvNeXt-Base + MLP Decoder")
    print(f"  Encoder: {args.encoder} ({encoder_params:,} params)")
    print(f"  Encoder channels: {encoder_channels}")
    print(f"  Decoder: MLP (embed_dim=256, {decoder_params:,} params)")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Dropout: {args.dropout}")

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------
    ema = None
    if args.ema_decay > 0:
        ema = ModelEMA(model, decay=args.ema_decay)
        print(f"  EMA: enabled (decay={args.ema_decay})")

    # --------------------------------------------------------
    # Loss, optimizer, scheduler
    # --------------------------------------------------------
    criterion = FocalDiceLoss(focal_weight=0.5, alpha=0.25, gamma=2.0)

    # Differential learning rates: encoder gets lr/10
    encoder_lr = args.lr / 10.0
    decoder_lr = args.lr
    param_groups = [
        {"params": model.encoder.parameters(), "lr": encoder_lr},
        {"params": model.decoder.parameters(), "lr": decoder_lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / args.accumulate_steps)
    total_steps = steps_per_epoch * args.epochs

    # OneCycleLR needs max_lr per param group
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[encoder_lr, decoder_lr],
        total_steps=total_steps,
        pct_start=args.warmup_pct,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    scaler = torch.amp.GradScaler('cuda')
    train_metrics = BinaryMetrics()
    val_metrics = BinaryMetrics()

    effective_batch = args.batch_size * args.accumulate_steps
    print(f"\nTraining config:")
    print(f"  Epochs: {args.epochs} (patience: {args.patience})")
    print(f"  Batch size: {args.batch_size} x {args.accumulate_steps} accumulation = {effective_batch} effective")
    print(f"  Encoder LR: {encoder_lr:.2e} | Decoder LR: {decoder_lr:.2e}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Loss: Focal (alpha=0.25, gamma=2.0) + Dice (50/50)")
    print(f"  Scheduler: OneCycleLR ({total_steps} total steps)")
    print(f"  Mixed precision: fp16")

    # --------------------------------------------------------
    # Training log CSV
    # --------------------------------------------------------
    log_path = save_dir / f"training_log_{args.task}.csv"
    log_fields = [
        "epoch", "enc_lr", "dec_lr",
        "train_loss", "train_acc", "train_iou", "train_precision", "train_recall", "train_f1",
        "val_loss", "val_acc", "val_iou", "val_precision", "val_recall", "val_f1",
        "ema_val_loss", "ema_val_iou",
        "epoch_time_s",
    ]
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(log_fields)

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    print(f"\n{'='*120}")
    print(f"{'EP':>3} | {'Enc LR':>9} {'Dec LR':>9} | "
          f"{'TrLoss':>7} {'TrAcc':>6} {'TrIoU':>6} {'TrPrec':>6} {'TrRec':>6} {'TrF1':>6} | "
          f"{'VLoss':>7} {'VAcc':>6} {'VIoU':>6} {'VPrec':>6} {'VRec':>6} {'VF1':>6} | "
          f"{'EMA':>6} | {'T':>4}")
    print(f"{'='*120}")

    best_iou = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler,
            device, train_metrics, ema, args.accumulate_steps,
        )
        train_results = train_metrics.compute()

        # Validate (raw model)
        val_loss, val_results = validate(model, val_loader, criterion, device, val_metrics)

        # Validate (EMA model)
        ema_val_iou = 0.0
        ema_val_loss = 0.0
        if ema is not None:
            ema_val_loss, ema_val_results = validate(
                ema.ema_model, val_loader, criterion, device, val_metrics)
            ema_val_iou = ema_val_results["iou"]

        epoch_time = time.time() - epoch_start
        enc_lr = optimizer.param_groups[0]["lr"]
        dec_lr = optimizer.param_groups[1]["lr"]

        # Use EMA IoU for model selection if available, otherwise raw
        select_iou = ema_val_iou if ema is not None else val_results["iou"]

        # Print compact results
        tr = train_results
        vr = val_results
        ema_str = f"{ema_val_iou:>6.4f}" if ema else "   N/A"
        print(f"{epoch:>3} | {enc_lr:>9.2e} {dec_lr:>9.2e} | "
              f"{train_loss:>7.4f} {tr['accuracy']:>6.4f} {tr['iou']:>6.4f} {tr['precision']:>6.4f} {tr['recall']:>6.4f} {tr['f1']:>6.4f} | "
              f"{val_loss:>7.4f} {vr['accuracy']:>6.4f} {vr['iou']:>6.4f} {vr['precision']:>6.4f} {vr['recall']:>6.4f} {vr['f1']:>6.4f} | "
              f"{ema_str} | {epoch_time:>3.0f}s", end="")

        # Log to CSV
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch, f"{enc_lr:.2e}", f"{dec_lr:.2e}",
                f"{train_loss:.6f}", f"{tr['accuracy']:.6f}", f"{tr['iou']:.6f}",
                f"{tr['precision']:.6f}", f"{tr['recall']:.6f}", f"{tr['f1']:.6f}",
                f"{val_loss:.6f}", f"{vr['accuracy']:.6f}", f"{vr['iou']:.6f}",
                f"{vr['precision']:.6f}", f"{vr['recall']:.6f}", f"{vr['f1']:.6f}",
                f"{ema_val_loss:.6f}", f"{ema_val_iou:.6f}",
                f"{epoch_time:.1f}",
            ])

        # Check for improvement (use EMA if available)
        if select_iou > best_iou:
            best_iou = select_iou
            best_epoch = epoch
            epochs_without_improvement = 0

            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_iou": best_iou,
                "arch": "ConvNeXt-Base + MLP",
                "encoder": args.encoder,
                "task": args.task,
                "target_class": target_class,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "dropout": args.dropout,
                "train_results": {k: float(v) for k, v in tr.items()},
                "val_results": {k: float(v) for k, v in vr.items()},
            }
            if ema is not None:
                save_dict["ema_state_dict"] = ema.state_dict()
                save_dict["ema_val_results"] = {k: float(v) for k, v in ema_val_results.items()}

            torch.save(save_dict, save_dir / f"best_{args.task}.pth")
            print(f" ★ BEST")
        else:
            epochs_without_improvement += 1
            print(f" ({epochs_without_improvement}/{args.patience})")

        # Early stopping
        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping after {epoch} epochs (no improvement for {args.patience} epochs)")
            break

    total_training_time = time.time() - training_start

    # --------------------------------------------------------
    # Test set evaluation
    # --------------------------------------------------------
    print(f"\n{'='*120}")
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print(f"{'='*120}")

    checkpoint = torch.load(save_dir / f"best_{args.task}.pth", map_location=device, weights_only=False)

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Evaluate raw model
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_results = validate(model, test_loader, criterion, device, val_metrics)

    # Evaluate EMA model if available
    ema_test_results = None
    if "ema_state_dict" in checkpoint and ema is not None:
        ema.load_state_dict(checkpoint["ema_state_dict"])
        ema_test_loss, ema_test_results = validate(
            ema.ema_model, test_loader, criterion, device, val_metrics)

    best_train = checkpoint.get("train_results", {})
    best_val = checkpoint.get("val_results", {})

    print(f"\n  Best model from epoch {best_epoch}")
    if ema_test_results:
        print(f"  {'Metric':<15} {'Train':>10} {'Val':>10} {'Test':>10} {'EMA Test':>10}")
        print(f"  {'-'*58}")
        for metric in ["accuracy", "iou", "precision", "recall", "f1"]:
            print(f"  {metric:<15} {best_train.get(metric, 0):>10.4f} {best_val.get(metric, 0):>10.4f} "
                  f"{test_results[metric]:>10.4f} {ema_test_results[metric]:>10.4f}")
    else:
        print(f"  {'Metric':<15} {'Train':>10} {'Val':>10} {'Test':>10}")
        print(f"  {'-'*45}")
        for metric in ["accuracy", "iou", "precision", "recall", "f1"]:
            print(f"  {metric:<15} {best_train.get(metric, 0):>10.4f} {best_val.get(metric, 0):>10.4f} "
                  f"{test_results[metric]:>10.4f}")

    print(f"\n  Total training time: {total_training_time/60:.1f} minutes")
    print(f"  Checkpoint: {save_dir / f'best_{args.task}.pth'}")
    print(f"  Training log: {log_path}")

    # Save test results
    results_path = save_dir / f"test_results_{args.task}.txt"
    with open(results_path, 'w') as f:
        f.write(f"Task: {args.task}\n")
        f.write(f"Model: ConvNeXt-Base + MLP Decoder ({args.encoder})\n")
        f.write(f"Parameters: {total_params:,}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Training time: {total_training_time/60:.1f} minutes\n\n")
        if ema_test_results:
            f.write(f"{'Metric':<15} {'Train':>10} {'Val':>10} {'Test':>10} {'EMA Test':>10}\n")
            f.write(f"{'-'*58}\n")
            for metric in ["accuracy", "iou", "precision", "recall", "f1"]:
                f.write(f"{metric:<15} {best_train.get(metric, 0):>10.4f} {best_val.get(metric, 0):>10.4f} "
                        f"{test_results[metric]:>10.4f} {ema_test_results[metric]:>10.4f}\n")
        else:
            f.write(f"{'Metric':<15} {'Train':>10} {'Val':>10} {'Test':>10}\n")
            f.write(f"{'-'*45}\n")
            for metric in ["accuracy", "iou", "precision", "recall", "f1"]:
                f.write(f"{metric:<15} {best_train.get(metric, 0):>10.4f} {best_val.get(metric, 0):>10.4f} "
                        f"{test_results[metric]:>10.4f}\n")

    print(f"  Test results: {results_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
