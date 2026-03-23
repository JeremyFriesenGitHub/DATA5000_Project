"""
Generate vegetation masks from Meta's Global Canopy Height Maps.

Replaces VRI-based vegetation masks with satellite-derived canopy height data.
Threshold: > 1m canopy height = vegetation (potential fire fuel).

Usage:
    python generate_meta_vegetation_masks.py
"""

import os
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
from pyproj import Transformer
import shutil

# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path("/home/jeremy/COMP4900_Project_backup_files")
CANOPY_URL = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float/chm/{tile_id}.tif"
CANOPY_THRESHOLD_M = 1  # meters - anything taller is vegetation

# Area definitions
AREAS = {
    "cumberland": {
        "tile_id": "021203330",
        "bounds_lonlat": (-125.05, 49.60, -124.97, 49.64),
    },
    "cumberland_lowres": {
        "tile_id": "021203330",
        "bounds_lonlat": (-125.06, 49.61, -125.02, 49.64),
    },
    "west_kelowna": {
        "tile_id": "021212303",
        "bounds_lonlat": (-119.66, 49.82, -119.48, 49.92),
    },
    "logan_lake": {
        "tile_id": "021212300",
        "bounds_lonlat": (-120.85, 50.45, -120.68, 50.55),
    },
    "silver_star": {
        "tile_id": "021212310",
        "bounds_lonlat": (-119.12, 50.33, -119.02, 50.39),
    },
}

# GeoTIFF info for Cumberland tiles - maps short ID prefix to full tif filename
CUMBERLAND_TIFS = {
    "68899581": "68899581e2874a0909754caa.tif",
    "688efdf3": "688efdf37163f9907394043a.tif",
    "688f0868": "688f08687163f99073940cba.tif",
    "688f8a1f": "688f8a1f7163f99073946297.tif",
    "688f9793": "688f97937163f99073946e64.tif",
    "689661fb": "689661fb4c41cdb5244a0f21.tif",
    "68a13b5f": "68a13b5f5923f304f0cb25fa.tif",
    "68be363e": "68be363e50956bf80eb8fcf9.tif",
}


def lonlat_to_3857(lon, lat):
    """Convert lon/lat to EPSG:3857."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return transformer.transform(lon, lat)


def download_canopy_window(tile_id, bounds_lonlat):
    """
    Open the Meta canopy height COG and read the window for the given bounds.
    Returns (canopy_data, canopy_transform, canopy_crs).
    """
    url = CANOPY_URL.format(tile_id=tile_id)
    print(f"  Opening COG: {url}")

    # Convert bounds from lon/lat to EPSG:3857
    xmin, ymin = lonlat_to_3857(bounds_lonlat[0], bounds_lonlat[1])
    xmax, ymax = lonlat_to_3857(bounds_lonlat[2], bounds_lonlat[3])
    print(f"  EPSG:3857 bounds: ({xmin:.1f}, {ymin:.1f}) to ({xmax:.1f}, {ymax:.1f})")

    with rasterio.open(url) as ds:
        print(f"  COG opened: {ds.width}x{ds.height}, CRS={ds.crs}")
        window = from_bounds(xmin, ymin, xmax, ymax, ds.transform)
        print(f"  Window: {window}")
        data = ds.read(1, window=window)
        win_transform = ds.window_transform(window)
        print(f"  Read {data.shape[0]}x{data.shape[1]} pixels")
        return data, win_transform, ds.crs


def save_canopy_cache(area_name, data, transform, crs):
    """Save downloaded canopy data to local GeoTIFF for reuse."""
    cache_dir = BASE_DIR / "meta_canopy_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"{area_name}_canopy.tif"

    with rasterio.open(
        cache_path, "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)
    print(f"  Cached to {cache_path}")
    return cache_path


def load_canopy_cache(area_name):
    """Load cached canopy data if available."""
    cache_path = BASE_DIR / "meta_canopy_cache" / f"{area_name}_canopy.tif"
    if cache_path.exists():
        with rasterio.open(cache_path) as ds:
            data = ds.read(1)
            return data, ds.transform, ds.crs
    return None, None, None


def extract_canopy_for_bounds(canopy_data, canopy_transform, canopy_crs,
                                tile_bounds_utm, tile_crs, tile_size):
    """
    Extract and threshold canopy data for a specific tile's bounds.

    Args:
        canopy_data: full area canopy array (EPSG:3857)
        canopy_transform: affine transform for canopy data
        canopy_crs: CRS of canopy data (EPSG:3857)
        tile_bounds_utm: (left, bottom, right, top) in tile's CRS
        tile_crs: CRS of the tile (e.g., EPSG:32610)
        tile_size: (width, height) of output mask

    Returns:
        Binary mask array (0 or 255), sized to tile_size
    """
    # Convert tile bounds from tile CRS to canopy CRS (EPSG:3857)
    transformer = Transformer.from_crs(tile_crs, canopy_crs, always_xy=True)

    left, bottom = transformer.transform(tile_bounds_utm[0], tile_bounds_utm[1])
    right, top = transformer.transform(tile_bounds_utm[2], tile_bounds_utm[3])

    # Get pixel coordinates in the canopy data
    inv_transform = ~canopy_transform
    col_start, row_start = inv_transform * (left, top)
    col_end, row_end = inv_transform * (right, bottom)

    # Ensure correct ordering
    col_start, col_end = sorted([col_start, col_end])
    row_start, row_end = sorted([row_start, row_end])

    # Clamp to data bounds
    row_start = max(0, int(row_start))
    row_end = min(canopy_data.shape[0], int(row_end) + 1)
    col_start = max(0, int(col_start))
    col_end = min(canopy_data.shape[1], int(col_end) + 1)

    if row_end <= row_start or col_end <= col_start:
        # No overlap
        return np.zeros((tile_size[1], tile_size[0]), dtype=np.uint8)

    # Extract the canopy window
    canopy_window = canopy_data[row_start:row_end, col_start:col_end]

    # Threshold: > 1m = vegetation
    veg_mask = (canopy_window > CANOPY_THRESHOLD_M).astype(np.uint8) * 255

    # Resize to tile dimensions using nearest neighbor
    veg_pil = Image.fromarray(veg_mask)
    veg_resized = veg_pil.resize((tile_size[0], tile_size[1]), Image.NEAREST)

    return np.array(veg_resized)


# ============================================================
# Cumberland Processing
# ============================================================

def process_cumberland(canopy_data, canopy_transform, canopy_crs):
    """Generate vegetation masks for all Cumberland sub-tiles."""
    print("\n=== Processing Cumberland ===")

    output_dir = BASE_DIR / "cumberland_meta_masks" / "woodland_masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    tiles_dir = BASE_DIR / "cumberland_geotiff_tiles"
    building_masks_dir = BASE_DIR / "cumberland_ms_masks" / "building_masks"
    tif_dir = BASE_DIR / "tif"

    # Get list of tiles from building masks (these are the tiles we need)
    building_mask_files = sorted(building_masks_dir.glob("*.png"))
    print(f"  Found {len(building_mask_files)} building mask tiles")

    # Cache GeoTIFF metadata (transform, size) for each source TIF
    tif_meta = {}
    for short_id, tif_name in CUMBERLAND_TIFS.items():
        tif_path = tif_dir / tif_name
        if tif_path.exists():
            with rasterio.open(tif_path) as ds:
                tif_meta[short_id] = {
                    "transform": ds.transform,
                    "crs": ds.crs,
                    "width": ds.width,
                    "height": ds.height,
                    "bounds": ds.bounds,
                }

    processed = 0
    skipped = 0

    for mask_path in building_mask_files:
        tile_name = mask_path.stem  # e.g., "68899581_00000_00512"
        parts = tile_name.split("_")
        short_id = parts[0]
        row_offset = int(parts[1])
        col_offset = int(parts[2])

        if short_id not in tif_meta:
            skipped += 1
            continue

        meta = tif_meta[short_id]
        transform = meta["transform"]
        tile_crs = meta["crs"]

        # Compute geographic bounds for this 512x512 sub-tile
        # The sub-tile starts at pixel (col_offset, row_offset) in the source GeoTIFF
        tile_left = transform.c + col_offset * transform.a
        tile_top = transform.f + row_offset * transform.e
        tile_right = transform.c + (col_offset + 512) * transform.a
        tile_bottom = transform.f + (row_offset + 512) * transform.e

        # Ensure correct ordering
        tile_left, tile_right = min(tile_left, tile_right), max(tile_left, tile_right)
        tile_bottom, tile_top = min(tile_bottom, tile_top), max(tile_bottom, tile_top)

        tile_bounds = (tile_left, tile_bottom, tile_right, tile_top)

        # Extract and threshold canopy data
        veg_mask = extract_canopy_for_bounds(
            canopy_data, canopy_transform, canopy_crs,
            tile_bounds, tile_crs, (512, 512)
        )

        # Save mask
        out_path = output_dir / f"{tile_name}.png"
        Image.fromarray(veg_mask, mode="L").save(out_path)
        processed += 1

        if processed % 200 == 0:
            print(f"  Processed {processed}/{len(building_mask_files)} tiles...")

    print(f"  Done: {processed} masks generated, {skipped} skipped")
    return processed


# ============================================================
# Case Study Processing (West Kelowna / Logan Lake)
# ============================================================

def process_case_study(area_name, canopy_data, canopy_transform, canopy_crs,
                       imagery_dir_name, masks_dir_name, output_dir_name):
    """Generate vegetation masks for case study tiles (GeoTIFF-based)."""
    print(f"\n=== Processing {area_name} ===")

    imagery_dir = BASE_DIR / imagery_dir_name
    output_dir = BASE_DIR / output_dir_name / "woodland_masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy building masks
    src_building = BASE_DIR / masks_dir_name / "building_masks"
    dst_building = BASE_DIR / output_dir_name / "building_masks"
    if src_building.exists() and not dst_building.exists():
        print(f"  Copying building masks from {src_building}")
        shutil.copytree(src_building, dst_building)
        print(f"  Copied {len(list(dst_building.glob('*.png')))} building masks")
    elif dst_building.exists():
        print(f"  Building masks already exist at {dst_building}")

    # Process each GeoTIFF tile
    tile_files = sorted(imagery_dir.glob("*.tif"))
    print(f"  Found {len(tile_files)} imagery tiles")

    processed = 0
    for tile_path in tile_files:
        tile_name = tile_path.stem

        with rasterio.open(tile_path) as ds:
            tile_bounds = (ds.bounds.left, ds.bounds.bottom,
                          ds.bounds.right, ds.bounds.top)
            tile_crs = ds.crs
            tile_width = ds.width
            tile_height = ds.height

        # Extract and threshold canopy data
        veg_mask = extract_canopy_for_bounds(
            canopy_data, canopy_transform, canopy_crs,
            tile_bounds, tile_crs, (tile_width, tile_height)
        )

        # Save mask as PNG
        out_path = output_dir / f"{tile_name}.png"
        Image.fromarray(veg_mask, mode="L").save(out_path)
        processed += 1

    print(f"  Done: {processed} masks generated")
    return processed


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Meta Global Canopy Height → Vegetation Masks")
    print("=" * 60)

    # Process each area
    for area_name, config in AREAS.items():
        print(f"\n--- Downloading canopy data for {area_name} ---")

        # Try cache first
        canopy_data, canopy_transform, canopy_crs = load_canopy_cache(area_name)
        if canopy_data is not None:
            print(f"  Loaded from cache: {canopy_data.shape}")
        else:
            canopy_data, canopy_transform, canopy_crs = download_canopy_window(
                config["tile_id"], config["bounds_lonlat"]
            )
            save_canopy_cache(area_name, canopy_data, canopy_transform, canopy_crs)

        # Generate masks
        if area_name == "cumberland":
            process_cumberland(canopy_data, canopy_transform, canopy_crs)
        elif area_name == "west_kelowna":
            process_case_study(
                "West Kelowna",
                canopy_data, canopy_transform, canopy_crs,
                "west_kelowna_imagery",
                "west_kelowna_masks",
                "west_kelowna_meta_masks",
            )
        elif area_name == "cumberland_lowres":
            process_case_study(
                "Cumberland (1m)",
                canopy_data, canopy_transform, canopy_crs,
                "cumberland_imagery",
                "cumberland_meta_masks",
                "cumberland_meta_masks",
            )
        elif area_name == "logan_lake":
            process_case_study(
                "Logan Lake",
                canopy_data, canopy_transform, canopy_crs,
                "logan_lake_imagery",
                "logan_lake_masks",
                "logan_lake_meta_masks",
            )
        elif area_name == "silver_star":
            process_case_study(
                "Silver Star",
                canopy_data, canopy_transform, canopy_crs,
                "silver_star_imagery",
                "silver_star_meta_masks",
                "silver_star_meta_masks",
            )

    print("\n" + "=" * 60)
    print("All vegetation masks generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
