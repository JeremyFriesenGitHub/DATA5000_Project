"""
Data Preparation Script
========================
Properly combines INRIA and LandCover.ai datasets for training.
Splits by SOURCE IMAGE to prevent data leakage from overlapping tiles.

Building dataset: INRIA (binary 0/255) + LandCover.ai (class 1 extracted)
Woodland dataset: LandCover.ai (class 2 extracted) — separate output

Usage:
    python prep_data.py

Output:
    ./combined_building_v2/     (building: INRIA + LandCover, split by source)
    ./woodland_data/            (woodland: LandCover only, split by source)
"""

import random
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

COMBINED_SRC = Path("./combined_building")
LANDCOVER_SRC = Path("./dataset")

BUILDING_OUT = Path("./combined_building_v2")
WOODLAND_OUT = Path("./woodland_data")


def parse_source(filename):
    """Extract source image identifier from tile filename."""
    name = filename.stem
    if name.startswith("inria_"):
        # inria_austin1_00000_00000 -> austin1
        parts = name.split("_")
        return parts[1]
    elif name.startswith("lc_"):
        # lc_M-33-20-D-c-4-2_101 -> lc_M-33-20-D-c-4-2
        rest = name[3:]
        last_underscore = rest.rfind("_")
        return "lc_" + rest[:last_underscore]
    return name


def split_sources(sources, seed=SEED):
    """Split source images into train/val/test, stratified by dataset origin."""
    random.seed(seed)

    inria_sources = sorted([s for s in sources if not s.startswith("lc_")])
    lc_sources = sorted([s for s in sources if s.startswith("lc_")])

    def do_split(src_list):
        shuffled = src_list.copy()
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        return {
            "train": shuffled[:n_train],
            "val": shuffled[n_train:n_train + n_val],
            "test": shuffled[n_train + n_val:],
        }

    inria_split = do_split(inria_sources)
    lc_split = do_split(lc_sources)

    # Merge
    split = {
        "train": inria_split["train"] + lc_split["train"],
        "val": inria_split["val"] + lc_split["val"],
        "test": inria_split["test"] + lc_split["test"],
    }

    return split, inria_split, lc_split


def prepare_building_dataset():
    """
    Re-split combined_building by source image.
    All tiles from the same source go into the same split.
    """
    print("=" * 60)
    print("BUILDING DATASET PREPARATION")
    print("=" * 60)

    # Gather all tiles from all existing splits
    source_tiles = defaultdict(list)
    for split in ["train", "val", "test"]:
        img_dir = COMBINED_SRC / split / "images"
        mask_dir = COMBINED_SRC / split / "masks"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*")):
            source = parse_source(img_path)
            # Find corresponding mask
            mask_name = img_path.stem + ".png"
            mask_path = mask_dir / mask_name
            if mask_path.exists():
                source_tiles[source].append((img_path, mask_path))

    total_tiles = sum(len(v) for v in source_tiles.values())
    print(f"\n  Total source images: {len(source_tiles)}")
    print(f"  Total tiles: {total_tiles}")

    # Split by source
    split_assignment, inria_split, lc_split = split_sources(source_tiles)

    # Create output directory
    for split in ["train", "val", "test"]:
        (BUILDING_OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (BUILDING_OUT / split / "masks").mkdir(parents=True, exist_ok=True)

    # Copy tiles to new splits
    stats = {"train": {"inria": 0, "lc": 0, "with_building": 0},
             "val": {"inria": 0, "lc": 0, "with_building": 0},
             "test": {"inria": 0, "lc": 0, "with_building": 0}}

    for split_name, sources in split_assignment.items():
        for source in sources:
            for img_path, mask_path in source_tiles[source]:
                # Copy image
                dst_img = BUILDING_OUT / split_name / "images" / img_path.name
                dst_mask = BUILDING_OUT / split_name / "masks" / mask_path.name
                shutil.copy2(img_path, dst_img)
                shutil.copy2(mask_path, dst_mask)

                # Track stats
                is_inria = not source.startswith("lc_")
                if is_inria:
                    stats[split_name]["inria"] += 1
                else:
                    stats[split_name]["lc"] += 1

                # Check if mask has buildings
                mask = np.array(Image.open(mask_path))
                if mask.max() > 0:
                    stats[split_name]["with_building"] += 1

    # Print results
    print(f"\n  Split by source image (no data leakage):")
    print(f"  INRIA sources:     train={len(inria_split['train'])} / val={len(inria_split['val'])} / test={len(inria_split['test'])}")
    print(f"  LandCover sources: train={len(lc_split['train'])} / val={len(lc_split['val'])} / test={len(lc_split['test'])}")

    print(f"\n  {'Split':<8} {'Total':>8} {'INRIA':>8} {'LandCov':>8} {'w/Build':>8} {'%':>6}")
    print(f"  {'-'*46}")
    for split_name in ["train", "val", "test"]:
        s = stats[split_name]
        total = s["inria"] + s["lc"]
        pct = total / sum(s2["inria"] + s2["lc"] for s2 in stats.values()) * 100
        print(f"  {split_name:<8} {total:>8} {s['inria']:>8} {s['lc']:>8} {s['with_building']:>8} {pct:>5.1f}%")

    grand_total = sum(s["inria"] + s["lc"] for s in stats.values())
    print(f"  {'TOTAL':<8} {grand_total:>8}")
    print(f"\n  Output: {BUILDING_OUT}")


def prepare_woodland_dataset():
    """
    Create woodland binary masks from LandCover.ai (class 2).
    Split by source map sheet to prevent leakage.
    """
    print(f"\n{'=' * 60}")
    print("WOODLAND DATASET PREPARATION")
    print("=" * 60)

    # Gather all LandCover tiles from all existing splits
    source_tiles = defaultdict(list)
    for split in ["train", "val", "test"]:
        img_dir = LANDCOVER_SRC / split / "images"
        mask_dir = LANDCOVER_SRC / split / "masks"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*")):
            # Parse source: M-33-20-D-c-4-2_101.jpg -> M-33-20-D-c-4-2
            name = img_path.stem
            last_underscore = name.rfind("_")
            if last_underscore > 0:
                source = name[:last_underscore]
            else:
                source = name

            # Find mask (has _m suffix in LandCover.ai)
            mask_name = name + "_m.png"
            mask_path = mask_dir / mask_name
            if not mask_path.exists():
                # Try without _m
                mask_name = name + ".png"
                mask_path = mask_dir / mask_name
            if mask_path.exists():
                source_tiles[source].append((img_path, mask_path))

    total_tiles = sum(len(v) for v in source_tiles.values())
    print(f"\n  Total source map sheets: {len(source_tiles)}")
    print(f"  Total tiles: {total_tiles}")

    # Split by source
    random.seed(SEED)
    sources_list = sorted(source_tiles.keys())
    random.shuffle(sources_list)
    n = len(sources_list)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    split_assignment = {
        "train": sources_list[:n_train],
        "val": sources_list[n_train:n_train + n_val],
        "test": sources_list[n_train + n_val:],
    }

    # Create output directory
    for split in ["train", "val", "test"]:
        (WOODLAND_OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (WOODLAND_OUT / split / "masks").mkdir(parents=True, exist_ok=True)

    # Copy tiles with extracted woodland masks
    stats = {"train": {"total": 0, "with_woodland": 0},
             "val": {"total": 0, "with_woodland": 0},
             "test": {"total": 0, "with_woodland": 0}}

    for split_name, sources in split_assignment.items():
        for source in sources:
            for img_path, mask_path in source_tiles[source]:
                # Copy image
                dst_img = WOODLAND_OUT / split_name / "images" / img_path.name
                shutil.copy2(img_path, dst_img)

                # Extract woodland (class 2) as binary mask
                mask = np.array(Image.open(mask_path))
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                woodland_mask = ((mask == 2) * 255).astype(np.uint8)

                dst_mask = WOODLAND_OUT / split_name / "masks" / (img_path.stem + ".png")
                Image.fromarray(woodland_mask).save(dst_mask)

                stats[split_name]["total"] += 1
                if woodland_mask.max() > 0:
                    stats[split_name]["with_woodland"] += 1

    # Print results
    print(f"\n  Split by source map sheet (no data leakage):")
    print(f"  Sources: train={len(split_assignment['train'])} / val={len(split_assignment['val'])} / test={len(split_assignment['test'])}")

    print(f"\n  {'Split':<8} {'Total':>8} {'w/Trees':>8} {'%':>6}")
    print(f"  {'-'*30}")
    for split_name in ["train", "val", "test"]:
        s = stats[split_name]
        pct = s["total"] / total_tiles * 100
        print(f"  {split_name:<8} {s['total']:>8} {s['with_woodland']:>8} {pct:>5.1f}%")

    print(f"  {'TOTAL':<8} {total_tiles:>8}")
    print(f"\n  Output: {WOODLAND_OUT}")


def main():
    print("Data Preparation — Source-Level Splitting")
    print("Prevents data leakage from overlapping/adjacent tiles\n")
    print(f"Split ratio: {TRAIN_RATIO*100:.0f}% / {VAL_RATIO*100:.0f}% / {TEST_RATIO*100:.0f}%")
    print(f"Random seed: {SEED}")

    prepare_building_dataset()
    prepare_woodland_dataset()

    print(f"\n{'=' * 60}")
    print("DONE!")
    print(f"{'=' * 60}")
    print(f"\nTraining commands:")
    print(f"  # Building:")
    print(f"  python train_segformer.py --data_dir ./combined_building_v2 --task building --binary_masks")
    print(f"\n  # Woodland:")
    print(f"  python train_segformer.py --data_dir ./woodland_data --task woodland --binary_masks")


if __name__ == "__main__":
    main()
