# combines datasets for training prep


"""
Prepare Combined Building Dataset
====================================
1. Tiles INRIA 5000x5000 images into 512x512 patches
2. Extracts building class from LandCover.ai masks
3. Combines both into one unified binary building dataset
4. Creates train/val/test splits

INRIA format:
  - Images: AerialImageDataset/train/images/*.tif (5000x5000 RGB)
  - Masks:  AerialImageDataset/train/gt/*.tif (5000x5000, 255=building)
  - Cities: austin, chicago, kitsap, tyrol, vienna (36 tiles each)

LandCover.ai format:
  - Images: dataset/{split}/images/*.jpg (512x512 RGB)
  - Masks:  dataset/{split}/masks/*.png (512x512, class 1=building)

Output:
  combined_building/
  ├── train/
  │   ├── images/   (jpg)
  │   └── masks/    (png, 0=not building, 255=building)
  ├── val/
  │   ├── images/
  │   └── masks/
  └── test/
      ├── images/
      └── masks/

Usage:
    python prep_combined.py \
        --inria_dir ~/Downloads/AerialImageDataset \
        --landcover_dir ~/Downloads/landcover.ai.v1/dataset \
        --output_dir ./combined_building \
        --tile_size 512
"""

import argparse
import random
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import cv2


def tile_inria(inria_dir, output_dir, tile_size=512, overlap=64):
    """
    Tile INRIA images and masks into 512x512 patches.
    Only keeps tiles that contain at least some building pixels.
    """
    inria_dir = Path(inria_dir)
    img_dir = inria_dir / "train" / "images"
    gt_dir = inria_dir / "train" / "gt"

    output_dir = Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)

    tif_files = sorted(img_dir.glob("*.tif"))
    print(f"Found {len(tif_files)} INRIA images")

    total_tiles = 0
    tiles_with_buildings = 0
    stride = tile_size - overlap

    for tif_path in tif_files:
        name = tif_path.stem  # e.g., "austin1"
        gt_path = gt_dir / tif_path.name

        if not gt_path.exists():
            print(f"  Skipping {name} — no ground truth")
            continue

        print(f"  Processing {name}...", end=" ")

        # Load image and mask
        img = np.array(Image.open(tif_path).convert("RGB"))
        mask = np.array(Image.open(gt_path))

        # INRIA masks: 255=building, 0=not building
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        h, w = img.shape[:2]
        tile_count = 0

        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                img_tile = img[y:y+tile_size, x:x+tile_size]
                mask_tile = mask[y:y+tile_size, x:x+tile_size]

                total_tiles += 1

                # Keep all tiles (even without buildings) for balanced training
                # But track which have buildings
                has_building = mask_tile.max() > 0
                if has_building:
                    tiles_with_buildings += 1

                tile_name = f"inria_{name}_{y:05d}_{x:05d}"

                Image.fromarray(img_tile).save(
                    output_dir / "images" / f"{tile_name}.jpg", quality=95
                )
                # Save as binary mask: 0 or 255
                Image.fromarray(mask_tile).save(
                    output_dir / "masks" / f"{tile_name}.png"
                )
                tile_count += 1

        print(f"{tile_count} tiles")

    print(f"\nINRIA tiling complete:")
    print(f"  Total tiles: {total_tiles}")
    print(f"  Tiles with buildings: {tiles_with_buildings}")

    return total_tiles


def extract_landcover_buildings(landcover_dir, output_dir):
    """
    Extract building masks from LandCover.ai dataset.
    Converts multi-class masks (class 1 = building) to binary (0/255).
    """
    landcover_dir = Path(landcover_dir)
    output_dir = Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)

    total = 0

    for split in ["train", "val", "test"]:
        img_dir = landcover_dir / split / "images"
        mask_dir = landcover_dir / split / "masks"

        if not img_dir.exists():
            continue

        images = sorted(img_dir.glob("*.jpg"))
        print(f"  LandCover.ai {split}: {len(images)} images")

        for img_path in images:
            name = img_path.stem
            mask_path = mask_dir / f"{name}_m.png"

            if not mask_path.exists():
                # Try without _m suffix
                mask_path = mask_dir / f"{name}.png"
                if not mask_path.exists():
                    continue

            mask = np.array(Image.open(mask_path))
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            # Convert: class 1 (building) → 255, everything else → 0
            binary_mask = ((mask == 1) * 255).astype(np.uint8)

            # Copy image
            out_name = f"lc_{name}"
            shutil.copy(img_path, output_dir / "images" / f"{out_name}.jpg")
            Image.fromarray(binary_mask).save(output_dir / "masks" / f"{out_name}.png")
            total += 1

    print(f"  LandCover.ai total: {total} tiles")
    return total


def create_splits(combined_dir, output_dir, train_ratio=0.7, val_ratio=0.15):
    """
    Shuffle all tiles and create train/val/test splits.
    """
    combined_dir = Path(combined_dir)
    output_dir = Path(output_dir)

    # Get all image files
    images = sorted((combined_dir / "images").glob("*.jpg"))
    print(f"\nTotal combined tiles: {len(images)}")

    # Pair images with masks
    pairs = []
    for img_path in images:
        name = img_path.stem
        mask_path = combined_dir / "masks" / f"{name}.png"
        if mask_path.exists():
            pairs.append((img_path, mask_path))

    print(f"Valid pairs: {len(pairs)}")

    # Shuffle
    random.seed(42)
    random.shuffle(pairs)

    # Split
    n = len(pairs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    for split_name, split_pairs in splits.items():
        img_out = output_dir / split_name / "images"
        mask_out = output_dir / split_name / "masks"
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        for img_path, mask_path in split_pairs:
            shutil.copy(img_path, img_out / img_path.name)
            shutil.copy(mask_path, mask_out / mask_path.name)

        # Count tiles with buildings
        building_count = 0
        for _, mask_path in split_pairs:
            m = np.array(Image.open(mask_path))
            if m.max() > 0:
                building_count += 1

        print(f"  {split_name}: {len(split_pairs)} tiles ({building_count} with buildings)")


def main():
    parser = argparse.ArgumentParser(description="Prepare combined building dataset")
    parser.add_argument("--inria_dir", type=str, required=True,
                        help="Path to AerialImageDataset/")
    parser.add_argument("--landcover_dir", type=str, required=True,
                        help="Path to landcover.ai dataset/")
    parser.add_argument("--output_dir", type=str, default="./combined_building",
                        help="Output directory for combined dataset")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    temp_dir = output_dir / "_temp_all"
    temp_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Tile INRIA dataset")
    print("=" * 60)
    inria_temp = temp_dir / "inria"
    tile_inria(args.inria_dir, inria_temp, args.tile_size, args.overlap)

    print("\n" + "=" * 60)
    print("STEP 2: Extract LandCover.ai buildings")
    print("=" * 60)
    lc_temp = temp_dir / "landcover"
    extract_landcover_buildings(args.landcover_dir, lc_temp)

    print("\n" + "=" * 60)
    print("STEP 3: Merge into combined directory")
    print("=" * 60)
    merged = temp_dir / "merged"
    (merged / "images").mkdir(parents=True, exist_ok=True)
    (merged / "masks").mkdir(parents=True, exist_ok=True)

    # Copy INRIA tiles
    inria_images = list((inria_temp / "images").glob("*.jpg"))
    print(f"  Copying {len(inria_images)} INRIA tiles...")
    for img in inria_images:
        shutil.copy(img, merged / "images" / img.name)
        mask = inria_temp / "masks" / f"{img.stem}.png"
        if mask.exists():
            shutil.copy(mask, merged / "masks" / mask.name)

    # Copy LandCover tiles
    lc_images = list((lc_temp / "images").glob("*.jpg"))
    print(f"  Copying {len(lc_images)} LandCover.ai tiles...")
    for img in lc_images:
        shutil.copy(img, merged / "images" / img.name)
        mask = lc_temp / "masks" / f"{img.stem}.png"
        if mask.exists():
            shutil.copy(mask, merged / "masks" / mask.name)

    print("\n" + "=" * 60)
    print("STEP 4: Create train/val/test splits")
    print("=" * 60)
    create_splits(merged, output_dir)

    # Cleanup temp
    print("\nCleaning up temp files...")
    shutil.rmtree(temp_dir)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Combined dataset saved to: {output_dir}")
    print(f"\nTo train:")
    print(f"  python train_binary.py --data_dir {output_dir} --time_budget 180 --task building")


if __name__ == "__main__":
    main()