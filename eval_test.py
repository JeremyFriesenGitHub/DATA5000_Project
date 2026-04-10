"""Standalone test-set evaluation for building and woodland models."""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---- Model ----
class MLPDecoder(nn.Module):
    def __init__(self, encoder_channels, embed_dim=256, num_classes=1, dropout=0.1):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            ) for ch in encoder_channels
        ])
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
        self.classifier = nn.Conv2d(embed_dim, num_classes, 1)

    def forward(self, features):
        target_size = features[0].shape[2:]
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
    def __init__(self, encoder_name='convnext_xlarge', num_classes=1, dropout=0.1, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True)
        encoder_channels = self.encoder.feature_info.channels()
        self.decoder = MLPDecoder(encoder_channels=encoder_channels, embed_dim=256,
                                  num_classes=num_classes, dropout=dropout)

    def forward(self, x):
        input_size = x.shape[2:]
        features = self.encoder(x)
        logits = self.decoder(features)
        return F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)


# ---- Dataset ----
class BinarySegDataset(Dataset):
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
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = (mask == self.target_class).astype(np.uint8)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        return image, mask.unsqueeze(0).float()


# ---- Metrics ----
class BinaryMetrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0

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
        return {"iou": iou, "accuracy": accuracy, "precision": precision,
                "recall": recall, "f1": f1}


# ---- Loss ----
class FocalDiceLoss(nn.Module):
    def __init__(self, focal_weight=0.5, alpha=0.25, gamma=2.0, smooth=1.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        focal_loss = focal.mean()
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice_loss = 1 - (2 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return self.focal_weight * focal_loss + (1 - self.focal_weight) * dice_loss


@torch.no_grad()
def evaluate(model, loader, criterion, device, metrics):
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


def run_eval(task, data_dir, binary_masks, encoder, checkpoint_path, device):
    print(f"\n{'='*60}")
    print(f"  TEST EVALUATION: {task.upper()}")
    print(f"{'='*60}")

    target_class = {"building": 1, "woodland": 3}.get(task, 1)
    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    test_dataset = BinarySegDataset(data_dir, "test", target_class, val_transform, binary_masks)
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False, num_workers=4, pin_memory=True)
    print(f"  Test samples: {len(test_dataset)}")

    model = SegModel(encoder_name=encoder, pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded checkpoint: {checkpoint_path}")
    print(f"  Best epoch: {checkpoint.get('epoch', 'N/A')}")

    criterion = FocalDiceLoss()
    metrics = BinaryMetrics()
    test_loss, results = evaluate(model, test_loader, criterion, device, metrics)

    val_results = checkpoint.get("val_results", {})
    train_results = checkpoint.get("train_results", {})

    print(f"\n  {'Metric':<15} {'Train':>10} {'Val':>10} {'Test':>10}")
    print(f"  {'-'*45}")
    for metric in ["accuracy", "iou", "precision", "recall", "f1"]:
        print(f"  {metric:<15} {train_results.get(metric, 0):>10.4f} "
              f"{val_results.get(metric, 0):>10.4f} {results[metric]:>10.4f}")
    print(f"  {'test_loss':<15} {'':>10} {'':>10} {test_loss:>10.4f}")
    print()
    return results


if __name__ == "__main__":
    device = torch.device("cuda:0")

    # Building eval
    building_results = run_eval(
        task="building",
        data_dir="./combined_building_v2",
        binary_masks=True,
        encoder="convnext_xlarge",
        checkpoint_path="checkpoints_segformer/best_building.pth",
        device=device,
    )

    # Woodland eval (binary_masks=True, target_class not used when binary)
    woodland_results = run_eval(
        task="woodland",
        data_dir="./woodland_data",
        binary_masks=True,
        encoder="convnext_xlarge",
        checkpoint_path="checkpoints_segformer/best_woodland.pth",
        device=device,
    )

    # Summary comparison
    print("=" * 60)
    print("  SUMMARY: NEW (ConvNeXt-XLarge) vs OLD (ConvNeXt-Base)")
    print("=" * 60)
    print(f"\n  Building Detection:")
    print(f"    Old Test IoU: 0.7761    New Test IoU: {building_results['iou']:.4f}  "
          f"({'+'if building_results['iou']>0.7761 else ''}{(building_results['iou']-0.7761)*100:.2f}%)")
    print(f"    Old Test F1:  0.8742    New Test F1:  {building_results['f1']:.4f}  "
          f"({'+'if building_results['f1']>0.8742 else ''}{(building_results['f1']-0.8742)*100:.2f}%)")
    print(f"\n  Woodland Detection:")
    print(f"    Old Test IoU: 0.8985    New Test IoU: {woodland_results['iou']:.4f}  "
          f"({'+'if woodland_results['iou']>0.8985 else ''}{(woodland_results['iou']-0.8985)*100:.2f}%)")
    print(f"    Old Test F1:  0.9466    New Test F1:  {woodland_results['f1']:.4f}  "
          f"({'+'if woodland_results['f1']>0.9466 else ''}{(woodland_results['f1']-0.9466)*100:.2f}%)")
    print()
