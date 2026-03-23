#!/usr/bin/env python3
"""
Download Microsoft Building Footprints for BC and generate building masks
for West Kelowna and Logan Lake study areas.
"""

import os
import sys
import json
import zipfile
import urllib.request
import numpy as np
from pathlib import Path
from shapely.geometry import shape, box
from shapely.ops import transform as shapely_transform
import rasterio
from rasterio.features import rasterize
from pyproj import Transformer
from PIL import Image

BASE_DIR = Path("/home/jeremy/COMP4900_Project_backup_files")

# Study area bounds in EPSG:4326 (lon, lat)
AREAS = {
    "west_kelowna": {
        "bbox_4326": (-119.66, 49.82, -119.48, 49.92),
        "imagery_dir": BASE_DIR / "west_kelowna_imagery",
        "mask_dir": BASE_DIR / "west_kelowna_masks" / "building_masks",
        "prefix": "wk_",
    },
    "logan_lake": {
        "bbox_4326": (-120.85, 50.45, -120.68, 50.55),
        "imagery_dir": BASE_DIR / "logan_lake_imagery",
        "mask_dir": BASE_DIR / "logan_lake_masks" / "building_masks",
        "prefix": "ll_",
    },
    "cumberland": {
        "bbox_4326": (-125.0522, 49.6149, -125.0259, 49.6315),
        "imagery_dir": BASE_DIR / "cumberland_imagery",
        "mask_dir": BASE_DIR / "cumberland_meta_masks" / "building_masks",
        "prefix": "cb_",
    },
    "silver_star": {
        "bbox_4326": (-119.10, 50.34, -119.04, 50.38),
        "imagery_dir": BASE_DIR / "silver_star_imagery",
        "mask_dir": BASE_DIR / "silver_star_meta_masks" / "building_masks",
        "prefix": "ss_",
    },
}

BC_URL = "https://minedbuildings.z5.web.core.windows.net/legacy/canadian-buildings-v2/BritishColumbia.zip"
ZIP_PATH = BASE_DIR / "BritishColumbia.zip"


def download_bc_buildings():
    """Download BC building footprints zip if not already present."""
    if ZIP_PATH.exists():
        print(f"Zip already exists: {ZIP_PATH}")
        return
    print(f"Downloading BC building footprints...")
    print(f"  URL: {BC_URL}")
    urllib.request.urlretrieve(BC_URL, ZIP_PATH)
    print(f"  Downloaded to {ZIP_PATH}")


def extract_all_buildings():
    """
    Extract buildings for both areas using a streaming JSON approach.
    The file is a FeatureCollection with features array.
    We stream it to avoid loading the entire ~300MB+ uncompressed file.
    """
    print("Extracting buildings for both areas...")

    area_boxes = {}
    area_buildings = {}
    area_bounds = {}
    for area_name, cfg in AREAS.items():
        minlon, minlat, maxlon, maxlat = cfg["bbox_4326"]
        area_boxes[area_name] = box(minlon, minlat, maxlon, maxlat)
        area_buildings[area_name] = []
        area_bounds[area_name] = cfg["bbox_4326"]

    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        names = zf.namelist()
        geojson_file = names[0]
        print(f"  Reading from: {geojson_file}")

        with zf.open(geojson_file) as f:
            # Read the entire content -- it's about 300MB uncompressed but
            # we need to parse it. Use ijson if available, otherwise load in chunks.
            # Actually, the format has one feature per line after the header.
            # Let's try a line-by-line approach with proper cleanup.

            feature_count = 0

            for raw_line in f:
                line = raw_line.decode('utf-8').strip()

                # Skip structural lines
                if not line or line in ('{', '}', '[', ']',
                                        '"type":"FeatureCollection",',
                                        '"features":'):
                    continue

                # Strip trailing comma if present
                if line.endswith(','):
                    line = line[:-1]

                # Skip non-feature lines
                if not line.startswith('{"type":"Feature"'):
                    continue

                feature_count += 1
                if feature_count % 200000 == 0:
                    counts = {k: len(v) for k, v in area_buildings.items()}
                    print(f"    Processed {feature_count} features, found: {counts}")

                try:
                    feature = json.loads(line)
                    coords = feature["geometry"]["coordinates"][0]

                    # Quick bounding box from coordinates
                    lons = [c[0] for c in coords]
                    lats = [c[1] for c in coords]
                    min_lon, max_lon = min(lons), max(lons)
                    min_lat, max_lat = min(lats), max(lats)

                    for area_name, (amin_lon, amin_lat, amax_lon, amax_lat) in area_bounds.items():
                        if max_lon < amin_lon or min_lon > amax_lon or max_lat < amin_lat or min_lat > amax_lat:
                            continue

                        geom = shape(feature["geometry"])
                        if geom.intersects(area_boxes[area_name]):
                            area_buildings[area_name].append(geom)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        print(f"  Total features processed: {feature_count}")
        for area_name, bldgs in area_buildings.items():
            print(f"  {area_name}: {len(bldgs)} buildings found")

    return area_buildings


def reproject_buildings(buildings_4326):
    """Reproject building geometries from EPSG:4326 to EPSG:32610."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)

    reprojected = []
    for geom in buildings_4326:
        reprojected_geom = shapely_transform(transformer.transform, geom)
        reprojected.append(reprojected_geom)

    return reprojected


def rasterize_buildings_for_tiles(area_name, buildings_32610, cfg):
    """Create building mask PNGs for each imagery tile."""
    imagery_dir = cfg["imagery_dir"]
    mask_dir = cfg["mask_dir"]
    mask_dir.mkdir(parents=True, exist_ok=True)

    tile_files = sorted(imagery_dir.glob("*.tif"))
    print(f"\n  Processing {len(tile_files)} tiles for {area_name}...")

    # Build a spatial index
    from shapely import STRtree
    if buildings_32610:
        tree = STRtree(buildings_32610)

    tiles_with_buildings = 0
    total_building_pixels = 0

    for i, tile_path in enumerate(tile_files):
        with rasterio.open(tile_path) as src:
            tile_transform = src.transform
            tile_width = src.width
            tile_height = src.height
            tile_bounds = src.bounds

        tile_box = box(tile_bounds.left, tile_bounds.bottom,
                       tile_bounds.right, tile_bounds.top)

        if buildings_32610:
            candidate_indices = tree.query(tile_box)
            tile_buildings = []
            for idx in candidate_indices:
                bldg = buildings_32610[idx]
                if bldg.intersects(tile_box):
                    tile_buildings.append(bldg)
        else:
            tile_buildings = []

        if tile_buildings:
            shapes = [(geom, 255) for geom in tile_buildings]
            mask = rasterize(
                shapes,
                out_shape=(tile_height, tile_width),
                transform=tile_transform,
                fill=0,
                dtype=np.uint8,
            )
            tiles_with_buildings += 1
            total_building_pixels += int(np.sum(mask == 255))
        else:
            mask = np.zeros((tile_height, tile_width), dtype=np.uint8)

        tile_stem = tile_path.stem
        mask_path = mask_dir / f"{tile_stem}.png"
        img = Image.fromarray(mask, mode='L')
        img.save(mask_path)

        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1}/{len(tile_files)} tiles...")

    print(f"  Completed {area_name}:")
    print(f"    Total tiles: {len(tile_files)}")
    print(f"    Tiles with buildings: {tiles_with_buildings}")
    print(f"    Total building pixels: {total_building_pixels}")

    return tiles_with_buildings, len(tile_files)


def cleanup():
    """Remove downloaded BC-wide zip file."""
    if ZIP_PATH.exists():
        print(f"Removing {ZIP_PATH}...")
        ZIP_PATH.unlink()
        print("  Cleaned up.")


def main():
    print("=" * 60)
    print("TASK 1: Microsoft Building Footprints -> Building Masks")
    print("=" * 60)

    download_bc_buildings()
    area_buildings = extract_all_buildings()

    for area_name, cfg in AREAS.items():
        print(f"\n--- {area_name.upper()} ---")
        buildings_4326 = area_buildings[area_name]

        if not buildings_4326:
            print(f"  No buildings found for {area_name}! Creating empty masks...")
            imagery_dir = cfg["imagery_dir"]
            mask_dir = cfg["mask_dir"]
            mask_dir.mkdir(parents=True, exist_ok=True)
            tile_files = sorted(imagery_dir.glob("*.tif"))
            for tile_path in tile_files:
                with rasterio.open(tile_path) as src:
                    mask = np.zeros((src.height, src.width), dtype=np.uint8)
                mask_path = mask_dir / f"{tile_path.stem}.png"
                Image.fromarray(mask, mode='L').save(mask_path)
            print(f"  Created {len(tile_files)} empty masks.")
            continue

        print(f"  Reprojecting {len(buildings_4326)} buildings to EPSG:32610...")
        buildings_32610 = reproject_buildings(buildings_4326)

        rasterize_buildings_for_tiles(area_name, buildings_32610, cfg)

    cleanup()

    print("\n" + "=" * 60)
    print("Building mask generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
