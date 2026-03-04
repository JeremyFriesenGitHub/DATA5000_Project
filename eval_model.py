# initial eval script for evaluating on test data/unlabeled

"""
Evaluate & Infer with Trained Model
=====================================
Works with any architecture saved by auto_tune.py or train_unet.py.

Usage:
    # Evaluate on test set:
    python eval_model.py --checkpoint checkpoints/best_model.pth --mode test --data_dir ./dataset

    # Run on Cumberland tiles:
    python eval_model.py --checkpoint checkpoints/best_model.pth --mode infer --tiles_dir ./filtered/has_structures --output_dir ./cumberland_predictions

Requirements:
    pip install segmentation-models-pytorch torch torchvision albumentations opencv-python
"""

import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


CLASS_NAMES = ["background", "building", "woodland", "water", "road"]
NUM_CLASSES = 5

CLASS_COLORS = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [0, 180, 0],
    3: [0, 0, 255],
    4: [180, 180, 180],
}


class LandCoverDataset(Dataset):
    CLASS_NAMES = CLASS_NAMES
    NUM_CLASSES = NUM_CLASSES

    def __init__(self, data_dir, split="test", transform=None):
        self.data_dir = Path(data_dir) / split
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
        mask = np.clip(mask, 0, self.NUM_CLASSES - 1)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        return image, mask.long()


class InferenceDataset(Dataset):
    def __init__(self, tiles_dir, transform=None):
        self.tiles_dir = Path(tiles_dir)
        self.tiles = sorted(
            list(self.tiles_dir.glob("*.png")) +
            list(self.tiles_dir.glob("*.jpg"))
        )
        self.transform = transform
        print(f"Found {len(self.tiles)} tiles for inference")

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.tiles[idx]).convert("RGB"))
        name = self.tiles[idx].stem
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image, name


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
        for t, p in zip(targets[valid], preds[valid]):
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


def build_model(arch, encoder, num_classes=5):
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
        classes=num_classes,
    )


def colorize_mask(mask):
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in CLASS_COLORS.items():
        colored[mask == val] = color
    return colored


def create_overlay(image, mask, alpha=0.4):
    colored_mask = colorize_mask(mask)
    overlay = image.copy()
    non_bg = mask > 0
    overlay[non_bg] = cv2.addWeighted(
        image[non_bg], 1 - alpha,
        colored_mask[non_bg], alpha, 0
    )
    return overlay


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = checkpoint.get("arch", "Unet")
    encoder = checkpoint.get("encoder", "resnet34")
    model = build_model(arch, encoder)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    miou = checkpoint.get("best_miou", "?")
    epoch = checkpoint.get("epoch", "?")
    print(f"Loaded: {arch} + {encoder} (epoch={epoch}, best_mIoU={miou})")
    return model
   


@torch.no_grad()
def evaluate_test(model, data_dir, device, batch_size=4):
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    test_dataset = LandCoverDataset(data_dir, "test", transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    metrics = SegmentationMetrics(NUM_CLASSES, CLASS_NAMES)

    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)
        with torch.amp.autocast('cuda'):
            outputs = model(images)
        preds = outputs.argmax(dim=1)
        metrics.update(preds, masks)

    results = metrics.compute()
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)
    print(f"Mean IoU:  {results['mean_iou']:.4f}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"\nPer-class IoU:")
    for cls_name, iou in results['per_class_iou'].items():
        print(f"  {cls_name:12s}: {iou:.4f}" if not np.isnan(iou) else f"  {cls_name:12s}: N/A")
    return results


@torch.no_grad()
def run_inference(model, tiles_dir, output_dir, device):
    output_dir = Path(output_dir)
    for sub in ["masks", "colored", "overlay"]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    dataset = InferenceDataset(tiles_dir, transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

    stats = {name: 0 for name in CLASS_NAMES}
    total_pixels = 0

    for images, names in loader:
        images = images.to(device)
        with torch.amp.autocast('cuda'):
            outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()

        for pred, name in zip(preds, names):
            Image.fromarray(pred.astype(np.uint8)).save(output_dir / "masks" / f"{name}.png")
            Image.fromarray(colorize_mask(pred)).save(output_dir / "colored" / f"{name}.png")

            orig_path = Path(tiles_dir) / f"{name}.png"
            if not orig_path.exists():
                orig_path = Path(tiles_dir) / f"{name}.jpg"
            if orig_path.exists():
                orig = np.array(Image.open(orig_path).convert("RGB"))
                if orig.shape[:2] != pred.shape:
                    pred_r = cv2.resize(pred, (orig.shape[1], orig.shape[0]),
                                        interpolation=cv2.INTER_NEAREST)
                    overlay = create_overlay(orig, pred_r)
                else:
                    overlay = create_overlay(orig, pred)
                Image.fromarray(overlay).save(output_dir / "overlay" / f"{name}.png")

            for i, cls in enumerate(CLASS_NAMES):
                stats[cls] += np.sum(pred == i)
            total_pixels += pred.size

    print("\n" + "=" * 50)
    print("INFERENCE COMPLETE")
    print("=" * 50)
    for cls, count in stats.items():
        pct = count / total_pixels * 100 if total_pixels > 0 else 0
        print(f"  {cls:12s}: {pct:.1f}%")
    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["test", "infer"])
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--tiles_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="predictions")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    if args.mode == "test":
        evaluate_test(model, args.data_dir, device, args.batch_size)
    elif args.mode == "infer":
        if not args.tiles_dir:
            print("ERROR: --tiles_dir required for infer mode")
            return
        run_inference(model, args.tiles_dir, args.output_dir, device)


if __name__ == "__main__":
    main()