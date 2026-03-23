#!/usr/bin/env python3
"""
Generate vegetation/woodland masks for West Kelowna and Logan Lake study areas.

Uses BC VRI (Vegetation Resource Inventory) data via WFS.
Falls back to Excess Green Index from RGB imagery if WFS fails.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import urllib.request
import urllib.parse
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box as shapely_box
from shapely import STRtree
from pyproj import Transformer
from PIL import Image

BASE_DIR = Path("/home/jeremy/COMP4900_Project_backup_files")

# Study area bounds in EPSG:4326 (lon, lat)
AREAS = {
    "west_kelowna": {
        "bbox_4326": (-119.66, 49.82, -119.48, 49.92),
        "imagery_dir": BASE_DIR / "west_kelowna_imagery",
        "mask_dir": BASE_DIR / "west_kelowna_masks" / "woodland_masks",
        "prefix": "wk_",
    },
    "logan_lake": {
        "bbox_4326": (-120.85, 50.45, -120.68, 50.55),
        "imagery_dir": BASE_DIR / "logan_lake_imagery",
        "mask_dir": BASE_DIR / "logan_lake_masks" / "woodland_masks",
        "prefix": "ll_",
    },
}

WFS_BASE = "https://openmaps.gov.bc.ca/geo/pub/WHSE_FOREST_VEGETATION.VEG_COMP_LYR_R1_POLY/ows"


def fetch_vri_wfs(area_name, bbox_4326):
    """
    Fetch VRI vegetation polygons from BC WFS for the given bounding box.
    Returns GeoDataFrame in EPSG:32610 with treed polygons, or None on failure.
    """
    minlon, minlat, maxlon, maxlat = bbox_4326

    # WFS 2.0 BBOX is: lower_corner_lat, lower_corner_lon, upper_corner_lat, upper_corner_lon, CRS
    bbox_str = f"{minlat},{minlon},{maxlat},{maxlon},urn:ogc:def:crs:EPSG::4326"

    params = {
        'service': 'WFS',
        'version': '2.0.0',
        'request': 'GetFeature',
        'typeName': 'pub:WHSE_FOREST_VEGETATION.VEG_COMP_LYR_R1_POLY',
        'outputFormat': 'application/json',
        'srsName': 'EPSG:4326',
        'BBOX': bbox_str,
        'count': '10000',
    }

    url = WFS_BASE + '?' + urllib.parse.urlencode(params)
    print(f"  Fetching VRI data from WFS...")

    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0')

    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()
        print(f"    Got {len(data) / (1024*1024):.1f} MB")

    tmp_path = BASE_DIR / f"vri_{area_name}.geojson"
    with open(tmp_path, 'wb') as f:
        f.write(data)

    gdf = gpd.read_file(tmp_path)
    tmp_path.unlink()

    if len(gdf) == 0:
        return None

    print(f"  Total VRI polygons fetched: {len(gdf)}")

    # Filter for treed/vegetated polygons
    # BCLCS_LEVEL_2: T=Treed, S=Shrub, H=Herb, (N=Non-vegetated, W=Water)
    # We want vegetation, so include T, S, H
    if 'BCLCS_LEVEL_2' in gdf.columns:
        veg_types = ['T', 'S', 'H']
        gdf_veg = gdf[gdf['BCLCS_LEVEL_2'].isin(veg_types)].copy()
        print(f"  Filtered to vegetation (T/S/H): {len(gdf_veg)} polygons")
        print(f"    Breakdown: {gdf['BCLCS_LEVEL_2'].value_counts().to_dict()}")
    else:
        print(f"  WARNING: BCLCS_LEVEL_2 column not found. Using all polygons.")
        print(f"  Available columns: {list(gdf.columns)[:20]}")
        gdf_veg = gdf

    if len(gdf_veg) == 0:
        return None

    # Reproject to EPSG:32610
    gdf_veg = gdf_veg.to_crs(epsg=32610)
    return gdf_veg


def rasterize_veg_for_tiles(area_name, veg_geometries, cfg):
    """Create vegetation mask PNGs for each imagery tile from VRI polygons."""
    imagery_dir = cfg["imagery_dir"]
    mask_dir = cfg["mask_dir"]
    mask_dir.mkdir(parents=True, exist_ok=True)

    tile_files = sorted(imagery_dir.glob("*.tif"))
    print(f"\n  Rasterizing {len(veg_geometries)} VRI polygons onto {len(tile_files)} tiles...")

    tree = STRtree(veg_geometries)

    tiles_with_veg = 0
    total_veg_pixels = 0
    total_pixels = 0

    for i, tile_path in enumerate(tile_files):
        with rasterio.open(tile_path) as src:
            tile_transform = src.transform
            tile_width = src.width
            tile_height = src.height
            tile_bounds = src.bounds

        tile_box = shapely_box(tile_bounds.left, tile_bounds.bottom,
                               tile_bounds.right, tile_bounds.top)

        candidate_indices = tree.query(tile_box)
        tile_veg = []
        for idx in candidate_indices:
            geom = veg_geometries[idx]
            if geom.intersects(tile_box):
                tile_veg.append(geom)

        if tile_veg:
            shapes = [(geom, 255) for geom in tile_veg]
            mask = rasterize(
                shapes,
                out_shape=(tile_height, tile_width),
                transform=tile_transform,
                fill=0,
                dtype=np.uint8,
            )
            n_veg = int(np.sum(mask == 255))
            if n_veg > 0:
                tiles_with_veg += 1
            total_veg_pixels += n_veg
        else:
            mask = np.zeros((tile_height, tile_width), dtype=np.uint8)

        total_pixels += tile_height * tile_width

        tile_stem = tile_path.stem
        mask_path = mask_dir / f"{tile_stem}.png"
        img = Image.fromarray(mask, mode='L')
        img.save(mask_path)

        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(tile_files)} tiles...")

    pct_veg = (total_veg_pixels / total_pixels * 100) if total_pixels > 0 else 0
    print(f"  Completed {area_name} (VRI approach):")
    print(f"    Total tiles: {len(tile_files)}")
    print(f"    Tiles with vegetation: {tiles_with_veg}")
    print(f"    Total vegetation pixels: {total_veg_pixels}")
    print(f"    Approximate vegetation coverage: {pct_veg:.1f}%")

    return tiles_with_veg, len(tile_files), pct_veg


def excess_green_approach(area_name, cfg):
    """
    Fallback: Use Excess Green Index (ExG = 2*G - R - B) to classify vegetation.
    """
    imagery_dir = cfg["imagery_dir"]
    mask_dir = cfg["mask_dir"]
    mask_dir.mkdir(parents=True, exist_ok=True)

    tile_files = sorted(imagery_dir.glob("*.tif"))
    print(f"\n  Processing {len(tile_files)} tiles for {area_name} using ExG index...")

    tiles_with_veg = 0
    total_veg_pixels = 0
    total_pixels = 0

    for i, tile_path in enumerate(tile_files):
        with rasterio.open(tile_path) as src:
            r = src.read(1).astype(np.float32)
            g = src.read(2).astype(np.float32)
            b = src.read(3).astype(np.float32)

        total_rgb = r + g + b
        total_rgb = np.where(total_rgb == 0, 1, total_rgb)
        r_norm = r / total_rgb
        g_norm = g / total_rgb
        b_norm = b / total_rgb

        exg = 2.0 * g_norm - r_norm - b_norm

        threshold = 0.1
        min_brightness = 30
        veg_mask = (exg > threshold) & (g > min_brightness) & (total_rgb > 90)

        mask = np.zeros_like(r, dtype=np.uint8)
        mask[veg_mask] = 255

        n_veg = int(np.sum(veg_mask))
        if n_veg > 0:
            tiles_with_veg += 1
        total_veg_pixels += n_veg
        total_pixels += mask.shape[0] * mask.shape[1]

        tile_stem = tile_path.stem
        mask_path = mask_dir / f"{tile_stem}.png"
        img = Image.fromarray(mask, mode='L')
        img.save(mask_path)

        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(tile_files)} tiles...")

    pct_veg = (total_veg_pixels / total_pixels * 100) if total_pixels > 0 else 0
    print(f"  Completed {area_name} (ExG approach):")
    print(f"    Total tiles: {len(tile_files)}")
    print(f"    Tiles with vegetation: {tiles_with_veg}")
    print(f"    Total vegetation pixels: {total_veg_pixels}")
    print(f"    Approximate vegetation coverage: {pct_veg:.1f}%")

    return tiles_with_veg, len(tile_files), pct_veg


def main():
    print("=" * 60)
    print("TASK 2: Vegetation/Woodland Masks")
    print("=" * 60)

    for area_name, cfg in AREAS.items():
        print(f"\n{'='*40}")
        print(f"  {area_name.upper()}")
        print(f"{'='*40}")

        veg_gdf = None
        try:
            veg_gdf = fetch_vri_wfs(area_name, cfg["bbox_4326"])
        except Exception as e:
            print(f"  WFS approach error: {e}")
            import traceback
            traceback.print_exc()

        if veg_gdf is not None and len(veg_gdf) > 0:
            print(f"  Using VRI data ({len(veg_gdf)} vegetation polygons)")
            veg_geometries = list(veg_gdf.geometry)
            rasterize_veg_for_tiles(area_name, veg_geometries, cfg)
        else:
            print(f"  WFS did not return usable vegetation data. Using ExG index approach.")
            excess_green_approach(area_name, cfg)

    print("\n" + "=" * 60)
    print("Vegetation mask generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
