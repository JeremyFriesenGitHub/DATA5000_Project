#!/usr/bin/env python3
"""
Generate BC woodland training data by combining:
- BC Government WMS aerial imagery
- VRI (Vegetation Resource Inventory) polygons as vegetation masks

This produces training tiles to augment existing LandCover.ai woodland data.
"""

import os
import sys
import time
import shutil
import json
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from collections import defaultdict
import traceback

# Geospatial imports
from shapely.geometry import shape, box, mapping
from shapely.ops import unary_union
from rasterio.features import rasterize
from rasterio.transform import from_bounds

# ============================================================
# Configuration
# ============================================================

BASE_DIR = "/home/jeremy/COMP4900_Project_backup_files"
EXISTING_DATA = os.path.join(BASE_DIR, "woodland_data")
OUTPUT_DATA = os.path.join(BASE_DIR, "woodland_data_v2")
BC_TILES_DIR = os.path.join(BASE_DIR, "bc_tiles_temp")

TILE_SIZE = 512
TILE_SPACING = 0.005  # ~500m in degrees
REQUEST_DELAY = 0.2   # seconds between WMS/WFS requests

# WMS Configuration
WMS_ENDPOINT = "https://openmaps.gov.bc.ca/imagex/ecw_wms.dll?"
WMS_LAYER = "bc_bc_bc_xc1m_bcalb_1995_2004"

# WFS Configuration
WFS_ENDPOINT = "https://openmaps.gov.bc.ca/geo/pub/WHSE_FOREST_VEGETATION.VEG_COMP_LYR_R1_POLY/ows"

# Vegetation codes: T=Treed, S=Shrub, H=Herb
VEG_CODES = ('T', 'S', 'H')

# White pixel threshold for skipping blank tiles
WHITE_THRESHOLD = 0.95

# ============================================================
# BC Locations - Diverse vegetation types
# Excluded: West Kelowna, Logan Lake (test areas), Cumberland (study area)
# ============================================================

LOCATIONS = {
    # Interior dry forests
    "kamloops": {
        "desc": "Interior dry forest - WUI area",
        "bbox": (-120.39, 50.67, -120.34, 50.72),
    },
    "penticton": {
        "desc": "Interior dry forest - Okanagan WUI",
        "bbox": (-119.65, 49.48, -119.60, 49.53),
    },
    # Coastal wet forests
    "nanaimo": {
        "desc": "Coastal wet forest - Vancouver Island",
        "bbox": (-124.02, 49.14, -123.97, 49.19),
    },
    "sechelt": {
        "desc": "Coastal wet forest - Sunshine Coast",
        "bbox": (-123.78, 49.45, -123.73, 49.50),
    },
    # Northern forests
    "prince_george": {
        "desc": "Northern boreal/sub-boreal forest",
        "bbox": (-122.82, 53.88, -122.77, 53.93),
    },
    "quesnel": {
        "desc": "Northern interior forest",
        "bbox": (-122.52, 52.96, -122.47, 53.01),
    },
    # Mountain forests
    "revelstoke": {
        "desc": "Mountain interior wet forest",
        "bbox": (-118.22, 50.97, -118.17, 51.02),
    },
    "salmon_arm": {
        "desc": "Interior transition forest",
        "bbox": (-119.30, 50.68, -119.25, 50.73),
    },
}


def get_tile_grid(bbox, spacing=TILE_SPACING):
    """Generate grid of tile bounding boxes within a location bbox."""
    lon_min, lat_min, lon_max, lat_max = bbox
    tiles = []
    lon = lon_min
    col = 0
    while lon + spacing <= lon_max + 1e-9:
        lat = lat_min
        row = 0
        while lat + spacing <= lat_max + 1e-9:
            tile_bbox = (lon, lat, lon + spacing, lat + spacing)
            tiles.append((row, col, tile_bbox))
            lat += spacing
            row += 1
        lon += spacing
        col += 1
    return tiles


def download_wms_tile(tile_bbox):
    """Download a single WMS tile as a PIL Image."""
    lon_min, lat_min, lon_max, lat_max = tile_bbox

    params = {
        "SERVICE": "WMS",
        "VERSION": "1.1.1",
        "REQUEST": "GetMap",
        "LAYERS": WMS_LAYER,
        "SRS": "EPSG:4326",
        "BBOX": f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "WIDTH": str(TILE_SIZE),
        "HEIGHT": str(TILE_SIZE),
        "FORMAT": "image/jpeg",
        "STYLES": "",
    }

    try:
        resp = requests.get(WMS_ENDPOINT, params=params, timeout=30)
        resp.raise_for_status()

        content_type = resp.headers.get('Content-Type', '')
        if 'image' not in content_type and 'jpeg' not in content_type:
            if b'<?xml' in resp.content[:100] or b'ServiceException' in resp.content[:500]:
                return None, "WMS returned error XML"

        img = Image.open(BytesIO(resp.content))
        img = img.convert("RGB")

        if img.size != (TILE_SIZE, TILE_SIZE):
            img = img.resize((TILE_SIZE, TILE_SIZE), Image.LANCZOS)

        return img, None

    except requests.exceptions.RequestException as e:
        return None, f"Request error: {e}"
    except Exception as e:
        return None, f"Error: {e}"


def is_blank_tile(img, threshold=WHITE_THRESHOLD):
    """Check if tile is mostly white/blank."""
    arr = np.array(img)
    white_pixels = np.all(arr > 240, axis=2)
    white_ratio = white_pixels.sum() / (arr.shape[0] * arr.shape[1])
    return white_ratio > threshold


def query_vri_wfs_for_location(location_bbox, max_retries=3):
    """
    Query VRI WFS for vegetation polygons using BBOX parameter.
    Uses BBOX parameter (not CQL_FILTER BBOX, as they are mutually exclusive).
    Filters for vegetation types (T/S/H) client-side.
    """
    lon_min, lat_min, lon_max, lat_max = location_bbox

    # BBOX parameter format: minx,miny,maxx,maxy,CRS (lon,lat order for EPSG:4326 in WFS 2.0)
    bbox_str = f"{lon_min},{lat_min},{lon_max},{lat_max},EPSG:4326"

    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "typeName": "pub:WHSE_FOREST_VEGETATION.VEG_COMP_LYR_R1_POLY",
        "outputFormat": "json",
        "SRSNAME": "EPSG:4326",
        "BBOX": bbox_str,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.get(WFS_ENDPOINT, params=params, timeout=120)
            resp.raise_for_status()

            data = resp.json()
            if "features" not in data:
                return [], "No features key in response"

            all_features = data["features"]

            # Client-side filter for vegetation types
            veg_features = [
                f for f in all_features
                if f.get("properties", {}).get("BCLCS_LEVEL_2") in VEG_CODES
            ]

            return veg_features, None

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"    WFS timeout, retrying ({attempt+1}/{max_retries})...")
                time.sleep(3)
                continue
            return [], "WFS timeout after retries"
        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return [], "JSON decode error"
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return [], f"Request error: {e}"

    return [], "Max retries exceeded"


def query_vri_wfs_quadrants(location_bbox, max_retries=3):
    """Fallback: query VRI in 4 quadrants if full bbox fails or returns too many."""
    lon_min, lat_min, lon_max, lat_max = location_bbox
    lon_mid = (lon_min + lon_max) / 2
    lat_mid = (lat_min + lat_max) / 2

    quadrants = [
        (lon_min, lat_min, lon_mid, lat_mid),
        (lon_mid, lat_min, lon_max, lat_mid),
        (lon_min, lat_mid, lon_mid, lat_max),
        (lon_mid, lat_mid, lon_max, lat_max),
    ]

    all_features = []
    for qi, qbbox in enumerate(quadrants):
        feats, err = query_vri_wfs_for_location(qbbox, max_retries=max_retries)
        if feats:
            all_features.extend(feats)
            print(f"    Quadrant {qi+1}: {len(feats)} veg features")
        elif err:
            print(f"    Quadrant {qi+1}: error - {err}")
        else:
            print(f"    Quadrant {qi+1}: 0 veg features")
        time.sleep(REQUEST_DELAY)

    return all_features


def rasterize_veg_mask(features, tile_bbox, size=TILE_SIZE):
    """Rasterize VRI vegetation polygons into a binary mask for a tile."""
    if not features:
        return np.zeros((size, size), dtype=np.uint8)

    lon_min, lat_min, lon_max, lat_max = tile_bbox
    tile_box = box(lon_min, lat_min, lon_max, lat_max)

    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, size, size)

    shapes = []
    for feat in features:
        try:
            geom = shape(feat["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)
            clipped = geom.intersection(tile_box)
            if not clipped.is_empty and clipped.area > 0:
                shapes.append((mapping(clipped), 255))
        except Exception:
            continue

    if not shapes:
        return np.zeros((size, size), dtype=np.uint8)

    mask = rasterize(
        shapes,
        out_shape=(size, size),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    return mask


def process_location(name, config, output_img_dir, output_mask_dir):
    """Process a single BC location: download imagery + generate masks."""
    bbox = config["bbox"]
    desc = config["desc"]
    print(f"\n{'='*60}")
    print(f"Processing: {name} - {desc}")
    print(f"  BBOX: {bbox}")

    tiles = get_tile_grid(bbox)
    print(f"  Grid: {len(tiles)} tiles")

    # Query VRI for the entire location
    print(f"  Querying VRI WFS for vegetation polygons...")
    vri_features, vri_error = query_vri_wfs_for_location(bbox)

    if vri_error and not vri_features:
        print(f"  Full query failed ({vri_error}), trying quadrants...")
        vri_features = query_vri_wfs_quadrants(bbox)

    print(f"  VRI vegetation features: {len(vri_features)}")
    time.sleep(REQUEST_DELAY)

    # Process each tile
    stats = {"downloaded": 0, "blank": 0, "saved": 0, "veg_tiles": 0, "errors": 0}

    for idx, (row, col, tile_bbox) in enumerate(tiles):
        tile_name = f"bc_{name}_{col:03d}_{row:03d}"
        img_path = os.path.join(output_img_dir, f"{tile_name}.jpg")
        mask_path = os.path.join(output_mask_dir, f"{tile_name}.png")

        # Download WMS tile
        img, err = download_wms_tile(tile_bbox)
        time.sleep(REQUEST_DELAY)

        if err or img is None:
            stats["errors"] += 1
            continue

        stats["downloaded"] += 1

        # Check for blank
        if is_blank_tile(img):
            stats["blank"] += 1
            continue

        # Generate vegetation mask
        mask = rasterize_veg_mask(vri_features, tile_bbox)

        # Check vegetation coverage
        veg_coverage = (mask > 0).sum() / (TILE_SIZE * TILE_SIZE)
        if veg_coverage > 0.01:
            stats["veg_tiles"] += 1

        # Save image and mask
        img.save(img_path, "JPEG", quality=95)
        mask_img = Image.fromarray(mask, mode='L')
        mask_img.save(mask_path, "PNG")
        stats["saved"] += 1

        # Progress update every 25 tiles
        if (idx + 1) % 25 == 0:
            print(f"    Progress: {idx+1}/{len(tiles)} tiles processed")

    print(f"  Results: downloaded={stats['downloaded']}, blank={stats['blank']}, "
          f"saved={stats['saved']}, with_veg={stats['veg_tiles']}, errors={stats['errors']}")

    return stats


def copy_existing_data():
    """Copy existing woodland_data into woodland_data_v2."""
    print("\n" + "="*60)
    print("Copying existing woodland_data to woodland_data_v2...")

    for split in ["train", "val", "test"]:
        for subdir in ["images", "masks"]:
            src = os.path.join(EXISTING_DATA, split, subdir)
            dst = os.path.join(OUTPUT_DATA, split, subdir)
            os.makedirs(dst, exist_ok=True)

            if not os.path.exists(src):
                print(f"  WARNING: Source not found: {src}")
                continue

            count = 0
            for fname in os.listdir(src):
                src_file = os.path.join(src, fname)
                dst_file = os.path.join(dst, fname)
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)
                count += 1

            print(f"  Copied {count} files: {split}/{subdir}")


def main():
    print("=" * 60)
    print("BC Woodland Training Data Generator")
    print("=" * 60)

    # Create output directories
    bc_img_dir = os.path.join(BC_TILES_DIR, "images")
    bc_mask_dir = os.path.join(BC_TILES_DIR, "masks")
    os.makedirs(bc_img_dir, exist_ok=True)
    os.makedirs(bc_mask_dir, exist_ok=True)

    # Process each BC location
    all_stats = {}
    total_saved = 0
    total_veg = 0

    for name, config in LOCATIONS.items():
        try:
            stats = process_location(name, config, bc_img_dir, bc_mask_dir)
            all_stats[name] = stats
            total_saved += stats["saved"]
            total_veg += stats["veg_tiles"]
        except Exception as e:
            print(f"  ERROR processing {name}: {e}")
            traceback.print_exc()
            all_stats[name] = {"downloaded": 0, "blank": 0, "saved": 0, "veg_tiles": 0, "errors": -1}

    print(f"\n{'='*60}")
    print(f"BC tile generation complete: {total_saved} tiles saved, {total_veg} with vegetation")

    # Copy existing data
    copy_existing_data()

    # Copy BC tiles to train split
    print("\nAdding BC tiles to training set...")
    train_img_dir = os.path.join(OUTPUT_DATA, "train", "images")
    train_mask_dir = os.path.join(OUTPUT_DATA, "train", "masks")
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)

    bc_added = 0
    for fname in sorted(os.listdir(bc_img_dir)):
        if fname.endswith(".jpg"):
            base = fname[:-4]
            mask_fname = base + ".png"

            src_img = os.path.join(bc_img_dir, fname)
            src_mask = os.path.join(bc_mask_dir, mask_fname)

            if os.path.exists(src_mask):
                shutil.copy2(src_img, os.path.join(train_img_dir, fname))
                shutil.copy2(src_mask, os.path.join(train_mask_dir, mask_fname))
                bc_added += 1

    print(f"  Added {bc_added} BC tiles to train split")

    # ============================================================
    # Print Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\nNew BC tiles per location:")
    print(f"  {'Location':<18} {'Saved':>6} {'With Veg':>10} {'Blank':>7} {'Errors':>7}")
    print(f"  {'-'*50}")
    for name, stats in all_stats.items():
        print(f"  {name:<18} {stats['saved']:>6} {stats['veg_tiles']:>10} "
              f"{stats['blank']:>7} {stats['errors']:>7}")

    # Count final dataset
    final_counts = {}
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(OUTPUT_DATA, split, "images")
        mask_dir = os.path.join(OUTPUT_DATA, split, "masks")
        n_img = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")]) if os.path.exists(img_dir) else 0
        n_mask = len([f for f in os.listdir(mask_dir) if f.endswith(".png")]) if os.path.exists(mask_dir) else 0
        final_counts[split] = {"images": n_img, "masks": n_mask}

    # Count BC tiles specifically
    bc_train_count = len([f for f in os.listdir(train_img_dir) if f.startswith("bc_") and f.endswith(".jpg")])

    # Check vegetation coverage in new BC tiles
    veg_tiles_count = 0
    total_bc_tiles = 0
    for fname in sorted(os.listdir(train_mask_dir)):
        if fname.startswith("bc_") and fname.endswith(".png"):
            total_bc_tiles += 1
            mask = np.array(Image.open(os.path.join(train_mask_dir, fname)))
            if (mask > 0).sum() > 0.01 * mask.size:
                veg_tiles_count += 1

    existing_train = final_counts["train"]["images"] - bc_train_count

    print(f"\nDataset composition (woodland_data_v2):")
    print(f"  Train:")
    print(f"    Existing LandCover.ai tiles: {existing_train}")
    print(f"    New BC tiles:                {bc_train_count}")
    print(f"    Total train:                 {final_counts['train']['images']}")
    print(f"  Val (unchanged):               {final_counts['val']['images']}")
    print(f"  Test (unchanged):              {final_counts['test']['images']}")
    print(f"  TOTAL:                         {sum(v['images'] for v in final_counts.values())}")

    if total_bc_tiles > 0:
        pct = 100.0 * veg_tiles_count / total_bc_tiles
        print(f"\nBC tile vegetation coverage:")
        print(f"  Tiles with >1% vegetation: {veg_tiles_count}/{total_bc_tiles} ({pct:.1f}%)")

    print(f"\nOutput directory: {OUTPUT_DATA}")
    print("Done!")


if __name__ == "__main__":
    main()
