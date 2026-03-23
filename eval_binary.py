# final script for evaluating binary models

"""
Evaluate & Infer with Binary Segmentation Models
==================================================
Works with models trained by train_binary.py.
Includes post-processing: morphological closing to fill patchy roofs,
and small object removal to filter cars/debris.

Usage:
    # Evaluate on test set:
    python eval_binary.py --checkpoint checkpoints_binary/best_building.pth --mode test --data_dir ./dataset

    # Infer on Cumberland:
    python eval_binary.py --checkpoint checkpoints_binary/best_building.pth --mode infer --tiles_dir ~/COMP4900/filtered/has_structures --output_dir ./cumberland_buildings

    # Run both models and combine:
    python eval_binary.py --building_ckpt checkpoints_binary/best_building.pth --woodland_ckpt checkpoints_binary/best_woodland.pth --mode combined --tiles_dir ~/COMP4900/filtered/has_structures --output_dir ./cumberland_combined
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import segmentation_models_pytorch as smp
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


# ============================================================
# DATASETS
# ============================================================

class BinaryTestDataset(Dataset):
    def __init__(self, data_dir, split, target_class, transform=None):
        self.data_dir = Path(data_dir) / split
        self.target_class = target_class
        self.transform = transform
        self.images = sorted((self.data_dir / "images").glob("*.jpg"))
        self.masks = sorted((self.data_dir / "masks").glob("*.png"))
        assert len(self.images) == len(self.masks)
        print(f"  {split}: {len(self.images)} samples")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert("RGB"))
        mask = np.array(Image.open(self.masks[idx]))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = (mask == self.target_class).astype(np.uint8)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        return image, mask.unsqueeze(0).float()


class InferenceDataset(Dataset):
    def __init__(self, tiles_dir, transform=None):
        self.tiles_dir = Path(tiles_dir)
        self.tiles = sorted(
            list(self.tiles_dir.glob("*.png")) +
            list(self.tiles_dir.glob("*.jpg")) +
            list(self.tiles_dir.glob("*.tif"))
        )
        self.transform = transform
        print(f"Found {len(self.tiles)} tiles")

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.tiles[idx]).convert("RGB"))
        name = self.tiles[idx].stem
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image, name


# ============================================================
# POST-PROCESSING
# ============================================================

def postprocess_building(mask, min_area=3000):
    """
    Clean up building predictions:
    1. Erode to break thin connections (roads touching buildings)
    2. Filter by shape — keep compact blobs, remove elongated ones
    3. Dilate back to restore building size
    4. Close small holes in roofs
    """
    # Step 1: Erode to separate touching blobs
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    eroded = cv2.morphologyEx(mask, cv2.MORPH_ERODE, erode_kernel)

    # Step 2: Filter by shape on separated blobs
    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shape_filtered = np.zeros_like(mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / (min(w, h) + 1e-6)

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # Buildings: aspect < 4, solidity > 0.4
        # Roads after erosion: still elongated or very low solidity
        if aspect < 4.0 and solidity > 0.4:
            cv2.drawContours(shape_filtered, [cnt], -1, 1, -1)

    # Step 3: Dilate back to restore building size
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    dilated = cv2.morphologyEx(shape_filtered, cv2.MORPH_DILATE, dilate_kernel)

    # Step 4: Mask with original predictions (don't expand beyond what model predicted)
    result = dilated & mask

    # Step 5: Close small holes in roofs
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, close_kernel)

    return result


def postprocess_woodland(mask, close_kernel=11, min_area=300):
    """
    Clean up woodland predictions:
    1. Morphological closing connects nearby tree canopy
    2. Remove tiny fragments
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(closed)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            cv2.drawContours(cleaned, [cnt], -1, 1, -1)

    return cleaned


def blend_overlay(orig, mask, color, alpha=0.4):
    """Blend a colored mask onto an image using float math (no cv2.addWeighted issues)."""
    overlay = orig.copy().astype(np.float32)
    where = mask == 1
    overlay[where] = overlay[where] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


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
# CONVNEXT-BASE + MLP DECODER (from train_segformer.py)
# ============================================================

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
    def __init__(self, encoder_name='convnext_base', num_classes=1, dropout=0.1, pretrained=False):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, features_only=True)
        encoder_channels = self.encoder.feature_info.channels()
        self.decoder = MLPDecoder(encoder_channels=encoder_channels, embed_dim=256,
                                  num_classes=num_classes, dropout=dropout)

    def forward(self, x):
        input_size = x.shape[2:]
        features = self.encoder(x)
        logits = self.decoder(features)
        logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        return logits


# ============================================================
# MODEL LOADING
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
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = checkpoint.get("arch", "FPN")
    encoder = checkpoint.get("encoder", "efficientnet-b3")
    task = checkpoint.get("task", "unknown")
    dropout = checkpoint.get("dropout", 0.1)

    if "ConvNeXt" in arch or "convnext" in encoder:
        # New ConvNeXt-Base + MLP decoder model
        model = SegModel(encoder_name=encoder, num_classes=1, dropout=dropout, pretrained=False)
        # Prefer EMA weights if available (smoother predictions)
        if "ema_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["ema_state_dict"])
            print(f"  (using EMA weights)")
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Legacy smp model
        model = build_model(arch, encoder)
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()
    iou = checkpoint.get("best_iou", "?")
    epoch = checkpoint.get("epoch", "?")
    print(f"Loaded: {arch} + {encoder} | task={task} | epoch={epoch} | IoU={iou}")
    return model, checkpoint


# ============================================================
# EVALUATE
# ============================================================

@torch.no_grad()
def evaluate_test(model, data_dir, target_class, task_name, device, batch_size=4):
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    dataset = BinaryTestDataset(data_dir, "test", target_class, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    metrics = BinaryMetrics()

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        with torch.amp.autocast('cuda'):
            outputs = model(images)
        preds = (torch.sigmoid(outputs) > 0.5).long()
        metrics.update(preds, masks.long())

    results = metrics.compute()
    print(f"\n{'=' * 50}")
    print(f"TEST SET EVALUATION: {task_name.upper()}")
    print(f"{'=' * 50}")
    print(f"  IoU:       {results['iou']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1:        {results['f1']:.4f}")
    print(f"  Accuracy:  {results['accuracy']:.4f}")
    return results


# ============================================================
# INFERENCE — SINGLE MODEL
# ============================================================

@torch.no_grad()
def run_inference(model, task_name, tiles_dir, output_dir, device):
    output_dir = Path(output_dir)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    (output_dir / "overlay").mkdir(parents=True, exist_ok=True)

    color = [255, 0, 0] if task_name == "building" else [0, 180, 0]

    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    dataset = InferenceDataset(tiles_dir, transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

    positive_pixels = 0
    total_pixels = 0

    for images, names in loader:
        images = images.to(device)
        with torch.amp.autocast('cuda'):
            outputs = model(images)
        preds = (torch.sigmoid(outputs) > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)

        for pred, name in zip(preds, names):
            # Post-process
            if task_name == "building":
                pred = postprocess_building(pred)
            else:
                pred = postprocess_woodland(pred)

            # Save binary mask (0 or 255)
            Image.fromarray(pred * 255).save(output_dir / "masks" / f"{name}.png")

            # Create overlay
            orig_path = Path(tiles_dir) / f"{name}.png"
            if not orig_path.exists():
                orig_path = Path(tiles_dir) / f"{name}.jpg"
            if orig_path.exists():
                orig = np.array(Image.open(orig_path).convert("RGB"))
                if orig.shape[:2] != pred.shape:
                    pred = cv2.resize(pred, (orig.shape[1], orig.shape[0]),
                                      interpolation=cv2.INTER_NEAREST)
                overlay = blend_overlay(orig, pred, color)
                Image.fromarray(overlay).save(output_dir / "overlay" / f"{name}.png")

            positive_pixels += pred.sum()
            total_pixels += pred.size

    pct = positive_pixels / total_pixels * 100 if total_pixels > 0 else 0
    print(f"\n{task_name.upper()} inference complete:")
    print(f"  {task_name} coverage: {pct:.1f}%")
    print(f"  Results saved to {output_dir}")


# ============================================================
# INFERENCE — COMBINED (both models)
# ============================================================

@torch.no_grad()
def run_combined_inference(building_model, woodland_model, tiles_dir, output_dir, device,
                           building_model2=None):
    output_dir = Path(output_dir)
    for sub in ["masks", "overlay", "building_masks", "woodland_masks"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    dataset = InferenceDataset(tiles_dir, transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

    building_pixels = 0
    woodland_pixels = 0
    total_pixels = 0

    for images, names in loader:
        images = images.to(device)

        with torch.amp.autocast('cuda'):
            building_out = building_model(images)
            woodland_out = woodland_model(images)

        building_probs = torch.sigmoid(building_out).squeeze(1).cpu().numpy()
        woodland_preds = (torch.sigmoid(woodland_out) > 0.5).squeeze(1).cpu().numpy().astype(np.uint8)

        # Ensemble: average probabilities with second building model
        if building_model2 is not None:
            with torch.amp.autocast('cuda'):
                building_out2 = building_model2(images)
            building_probs2 = torch.sigmoid(building_out2).squeeze(1).cpu().numpy()
            building_probs = (building_probs + building_probs2) / 2.0

        building_preds = (building_probs > 0.4).astype(np.uint8)

        for b_pred, w_pred, name in zip(building_preds, woodland_preds, names):
            # Post-process both masks
            b_pred = postprocess_building(b_pred)
            w_pred = postprocess_woodland(w_pred)

            # Save individual masks
            Image.fromarray(b_pred * 255).save(output_dir / "building_masks" / f"{name}.png")
            Image.fromarray(w_pred * 255).save(output_dir / "woodland_masks" / f"{name}.png")

            # Combined mask: 0=background, 1=building, 2=woodland
            combined = np.zeros_like(b_pred)
            combined[w_pred == 1] = 2
            combined[b_pred == 1] = 1  # Building overwrites
            Image.fromarray(combined).save(output_dir / "masks" / f"{name}.png")

            # Combined overlay
            orig_path = Path(tiles_dir) / f"{name}.png"
            if not orig_path.exists():
                orig_path = Path(tiles_dir) / f"{name}.jpg"
            if orig_path.exists():
                orig = np.array(Image.open(orig_path).convert("RGB"))
                if orig.shape[:2] != b_pred.shape:
                    b_pred = cv2.resize(b_pred, (orig.shape[1], orig.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                    w_pred = cv2.resize(w_pred, (orig.shape[1], orig.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)

                # Woodland green, then building red on top
                overlay = blend_overlay(orig, w_pred, [0, 180, 0])
                overlay = blend_overlay(overlay, b_pred, [255, 0, 0])

                Image.fromarray(overlay).save(output_dir / "overlay" / f"{name}.png")

            building_pixels += b_pred.sum()
            woodland_pixels += w_pred.sum()
            total_pixels += b_pred.size

    b_pct = building_pixels / total_pixels * 100 if total_pixels > 0 else 0
    w_pct = woodland_pixels / total_pixels * 100 if total_pixels > 0 else 0
    bg_pct = 100 - b_pct - w_pct

    print(f"\nCOMBINED INFERENCE COMPLETE:")
    print(f"  Building:   {b_pct:.1f}%")
    print(f"  Woodland:   {w_pct:.1f}%")
    print(f"  Background: {bg_pct:.1f}%")
    print(f"  Results saved to {output_dir}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None, help="Single model checkpoint")
    parser.add_argument("--building_ckpt", type=str, default=None, help="Building model checkpoint")
    parser.add_argument("--woodland_ckpt", type=str, default=None, help="Woodland model checkpoint")
    parser.add_argument("--mode", type=str, required=True, choices=["test", "infer", "combined"])
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--tiles_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="predictions_binary")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--building_ckpt2", type=str, default=None,
                        help="Second building model for ensemble (AND both predictions)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "test":
        model, ckpt = load_model(args.checkpoint, device)
        task = ckpt.get("task", "unknown")
        target_class = ckpt.get("target_class", 1)
        evaluate_test(model, args.data_dir, target_class, task, device, args.batch_size)

    elif args.mode == "infer":
        if not args.tiles_dir:
            print("ERROR: --tiles_dir required")
            return
        model, ckpt = load_model(args.checkpoint, device)
        task = ckpt.get("task", "unknown")
        run_inference(model, task, args.tiles_dir, args.output_dir, device)

    elif args.mode == "combined":
        if not args.building_ckpt or not args.woodland_ckpt:
            print("ERROR: --building_ckpt and --woodland_ckpt required for combined mode")
            return
        if not args.tiles_dir:
            print("ERROR: --tiles_dir required")
            return
        building_model, _ = load_model(args.building_ckpt, device)
        building_model2 = None
        if args.building_ckpt2:
            building_model2, _ = load_model(args.building_ckpt2, device)
        woodland_model, _ = load_model(args.woodland_ckpt, device)
        run_combined_inference(building_model, woodland_model, args.tiles_dir, args.output_dir, device,
                               building_model2=building_model2)


if __name__ == "__main__":
    main()