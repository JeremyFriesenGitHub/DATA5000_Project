#!/usr/bin/env python3
"""
Generate Microsoft Building Footprint masks and VRI vegetation masks
for Cumberland, BC study area.

Creates masks matching the cumberland_geotiff_tiles/ PNG tile structure.
Each GeoTIFF in tif/ was cut into 512x512 PNG sub-tiles named:
    {tif_prefix_8chars}_{row:05d}_{col:05d}.png

This script:
1. Downloads BC building footprints from Microsoft
2. Fetches VRI vegetation data from BC WFS
3. Rasterizes both onto each 512x512 tile using the parent GeoTIFF's transform
4. Saves masks to cumberland_ms_masks/building_masks/ and woodland_masks/
"""

import os
import sys
import json
import zipfile
import urllib.request
import urllib.parse
import numpy as np
from pathlib import Path
from PIL import Image
import rasterio
from rasterio.transform import Affine
from rasterio.features import rasterize
from pyproj import Transformer
from shapely.geometry import shape, box as shapely_box
from shapely.ops import transform as shapely_transform
from shapely import STRtree
import geopandas as gpd
import gc

BASE_DIR = Path("/home/jeremy/COMP4900_Project_backup_files")
TIF_DIR = BASE_DIR / "tif"
TILES_DIR = BASE_DIR / "cumberland_geotiff_tiles"
OUTPUT_DIR = BASE_DIR / "cumberland_ms_masks"
BUILDING_MASK_DIR = OUTPUT_DIR / "building_masks"
WOODLAND_MASK_DIR = OUTPUT_DIR / "woodland_masks"

# Cumberland bounds in EPSG:4326 (with padding)
CUMBERLAND_BBOX_4326 = (-125.06, 49.61, -125.02, 49.64)

BC_URL = "https://minedbuildings.z5.web.core.windows.net/legacy/canadian-buildings-v2/BritishColumbia.zip"
ZIP_PATH = BASE_DIR / "BritishColumbia.zip"

WFS_BASE = "https://openmaps.gov.bc.ca/geo/pub/WHSE_FOREST_VEGETATION.VEG_COMP_LYR_R1_POLY/ows"

TILE_SIZE = 512


def get_tif_info():
    """Load GeoTIFF metadata: transform, size, CRS for each file."""
    tif_info = {}
    for f in sorted(os.listdir(TIF_DIR)):
        if not f.endswith('.tif'):
            continue
        path = TIF_DIR / f
        with rasterio.open(path) as src:
            tif_info[f] = {
                'path': path,
                'transform': src.transform,
                'width': src.width,
                'height': src.height,
                'bounds': src.bounds,
                'crs': src.crs,
                'res': src.res,
            }
    return tif_info


def get_tile_geobounds(tif_transform, tif_res, row_offset, col_offset, tile_h=512, tile_w=512):
    """
    Compute the geospatial bounds (EPSG:32610) of a tile given its pixel offset
    within the parent GeoTIFF.
    """
    # Top-left corner of tile in projected coordinates
    x_left, y_top = tif_transform * (col_offset, row_offset)
    # Bottom-right corner
    x_right, y_bottom = tif_transform * (col_offset + tile_w, row_offset + tile_h)

    return {
        'left': min(x_left, x_right),
        'bottom': min(y_top, y_bottom),
        'right': max(x_left, x_right),
        'top': max(y_top, y_bottom),
    }


def build_tile_catalog(tif_info):
    """
    Build a catalog of all tiles with their geospatial bounds.
    Returns dict: tile_name -> {tif_file, row, col, bounds_32610, transform}
    """
    catalog = {}

    for tif_file, info in tif_info.items():
        prefix = tif_file[:8]  # First 8 chars of tif filename

        # Find all tiles from this GeoTIFF
        for tile_path in sorted(TILES_DIR.glob(f"{prefix}_*.png")):
            tile_name = tile_path.stem
            parts = tile_name.split('_')
            if len(parts) != 3:
                continue
            row_offset = int(parts[1])
            col_offset = int(parts[2])

            bounds = get_tile_geobounds(
                info['transform'], info['res'],
                row_offset, col_offset,
                TILE_SIZE, TILE_SIZE
            )

            # Create a sub-tile transform
            x_left, y_top = info['transform'] * (col_offset, row_offset)
            tile_transform = Affine(
                info['transform'].a, info['transform'].b, x_left,
                info['transform'].d, info['transform'].e, y_top
            )

            catalog[tile_name] = {
                'tif_file': tif_file,
                'row': row_offset,
                'col': col_offset,
                'bounds': bounds,
                'transform': tile_transform,
                'path': tile_path,
            }

    return catalog


# ============================================================
# STEP 1: Microsoft Building Footprints
# ============================================================

def download_bc_buildings():
    """Download BC building footprints zip if not already present."""
    if ZIP_PATH.exists():
        print(f"  Zip already exists: {ZIP_PATH}")
        return
    print(f"  Downloading BC building footprints...")
    print(f"  URL: {BC_URL}")
    urllib.request.urlretrieve(BC_URL, ZIP_PATH)
    size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
    print(f"  Downloaded: {size_mb:.1f} MB")


def extract_cumberland_buildings():
    """
    Extract buildings within Cumberland bounds from the BC zip.
    Returns list of shapely geometries in EPSG:4326.
    """
    minlon, minlat, maxlon, maxlat = CUMBERLAND_BBOX_4326
    print(f"  Extracting buildings within bounds: {CUMBERLAND_BBOX_4326}")

    buildings = []
    feature_count = 0

    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        names = zf.namelist()
        geojson_file = names[0]
        print(f"  Reading: {geojson_file}")

        with zf.open(geojson_file) as f:
            for raw_line in f:
                line = raw_line.decode('utf-8').strip()

                if not line or line in ('{', '}', '[', ']',
                                        '"type":"FeatureCollection",',
                                        '"features":'):
                    continue

                if line.endswith(','):
                    line = line[:-1]

                if not line.startswith('{"type":"Feature"'):
                    continue

                feature_count += 1
                if feature_count % 200000 == 0:
                    print(f"    Processed {feature_count} features, found {len(buildings)} in Cumberland...")

                try:
                    feature = json.loads(line)
                    coords = feature["geometry"]["coordinates"][0]

                    lons = [c[0] for c in coords]
                    lats = [c[1] for c in coords]
                    fmin_lon, fmax_lon = min(lons), max(lons)
                    fmin_lat, fmax_lat = min(lats), max(lats)

                    # Quick bbox check
                    if fmax_lon < minlon or fmin_lon > maxlon or fmax_lat < minlat or fmin_lat > maxlat:
                        continue

                    geom = shape(feature["geometry"])
                    area_box = shapely_box(minlon, minlat, maxlon, maxlat)
                    if geom.intersects(area_box):
                        buildings.append(geom)
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    print(f"  Total features processed: {feature_count}")
    print(f"  Buildings found in Cumberland: {len(buildings)}")
    return buildings


def reproject_buildings(buildings_4326):
    """Reproject building geometries from EPSG:4326 to EPSG:32610."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
    reprojected = []
    for geom in buildings_4326:
        reprojected_geom = shapely_transform(transformer.transform, geom)
        reprojected.append(reprojected_geom)
    return reprojected


def generate_building_masks(buildings_32610, tile_catalog):
    """Generate building mask PNGs for each tile."""
    BUILDING_MASK_DIR.mkdir(parents=True, exist_ok=True)

    if buildings_32610:
        tree = STRtree(buildings_32610)
    else:
        tree = None

    tiles_with_buildings = 0
    total_building_pixels = 0
    total_tiles = len(tile_catalog)

    for i, (tile_name, tile_info) in enumerate(sorted(tile_catalog.items())):
        bounds = tile_info['bounds']
        tile_box = shapely_box(bounds['left'], bounds['bottom'],
                               bounds['right'], bounds['top'])

        tile_buildings = []
        if tree is not None:
            candidates = tree.query(tile_box)
            for idx in candidates:
                bldg = buildings_32610[idx]
                if bldg.intersects(tile_box):
                    tile_buildings.append(bldg)

        if tile_buildings:
            shapes = [(geom, 255) for geom in tile_buildings]
            mask = rasterize(
                shapes,
                out_shape=(TILE_SIZE, TILE_SIZE),
                transform=tile_info['transform'],
                fill=0,
                dtype=np.uint8,
            )
            n_pixels = int(np.sum(mask == 255))
            if n_pixels > 0:
                tiles_with_buildings += 1
            total_building_pixels += n_pixels
        else:
            mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)

        mask_path = BUILDING_MASK_DIR / f"{tile_name}.png"
        Image.fromarray(mask, mode='L').save(mask_path)

        if (i + 1) % 200 == 0:
            print(f"    Building masks: {i + 1}/{total_tiles} tiles...")

    print(f"  Building mask generation complete:")
    print(f"    Total tiles: {total_tiles}")
    print(f"    Tiles with buildings: {tiles_with_buildings}")
    print(f"    Total building pixels: {total_building_pixels}")
    return tiles_with_buildings


# ============================================================
# STEP 2: VRI Vegetation Masks
# ============================================================

def fetch_vri_data():
    """
    Fetch VRI vegetation polygons from BC WFS for Cumberland.
    Returns list of shapely geometries in EPSG:32610.
    """
    minlon, minlat, maxlon, maxlat = CUMBERLAND_BBOX_4326

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
    print(f"  BBOX: {bbox_str}")

    req = urllib.request.Request(url)
    req.add_header('User-Agent', 'Mozilla/5.0')

    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()
        print(f"  Got {len(data) / (1024*1024):.1f} MB")

    tmp_path = BASE_DIR / "vri_cumberland.geojson"
    with open(tmp_path, 'wb') as f:
        f.write(data)

    gdf = gpd.read_file(tmp_path)
    tmp_path.unlink()

    if len(gdf) == 0:
        print("  WARNING: No VRI data returned!")
        return []

    print(f"  Total VRI polygons fetched: {len(gdf)}")

    if 'BCLCS_LEVEL_2' in gdf.columns:
        veg_types = ['T', 'S', 'H']
        gdf_veg = gdf[gdf['BCLCS_LEVEL_2'].isin(veg_types)].copy()
        print(f"  Filtered to vegetation (T/S/H): {len(gdf_veg)} polygons")
        print(f"    Breakdown: {gdf['BCLCS_LEVEL_2'].value_counts().to_dict()}")
    else:
        print(f"  WARNING: BCLCS_LEVEL_2 not found. Columns: {list(gdf.columns)[:20]}")
        gdf_veg = gdf

    if len(gdf_veg) == 0:
        return []

    gdf_veg = gdf_veg.to_crs(epsg=32610)
    return list(gdf_veg.geometry)


def generate_woodland_masks(veg_geometries, tile_catalog):
    """Generate vegetation mask PNGs for each tile."""
    WOODLAND_MASK_DIR.mkdir(parents=True, exist_ok=True)

    if veg_geometries:
        tree = STRtree(veg_geometries)
    else:
        tree = None

    tiles_with_veg = 0
    total_veg_pixels = 0
    total_tiles = len(tile_catalog)

    for i, (tile_name, tile_info) in enumerate(sorted(tile_catalog.items())):
        bounds = tile_info['bounds']
        tile_box = shapely_box(bounds['left'], bounds['bottom'],
                               bounds['right'], bounds['top'])

        tile_veg = []
        if tree is not None:
            candidates = tree.query(tile_box)
            for idx in candidates:
                geom = veg_geometries[idx]
                if geom.intersects(tile_box):
                    tile_veg.append(geom)

        if tile_veg:
            shapes = [(geom, 255) for geom in tile_veg]
            mask = rasterize(
                shapes,
                out_shape=(TILE_SIZE, TILE_SIZE),
                transform=tile_info['transform'],
                fill=0,
                dtype=np.uint8,
            )
            n_pixels = int(np.sum(mask == 255))
            if n_pixels > 0:
                tiles_with_veg += 1
            total_veg_pixels += n_pixels
        else:
            mask = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)

        mask_path = WOODLAND_MASK_DIR / f"{tile_name}.png"
        Image.fromarray(mask, mode='L').save(mask_path)

        if (i + 1) % 200 == 0:
            print(f"    Woodland masks: {i + 1}/{total_tiles} tiles...")

    print(f"  Woodland mask generation complete:")
    print(f"    Total tiles: {total_tiles}")
    print(f"    Tiles with vegetation: {tiles_with_veg}")
    print(f"    Total vegetation pixels: {total_veg_pixels}")
    return tiles_with_veg


def cleanup():
    """Remove downloaded BC-wide zip file."""
    if ZIP_PATH.exists():
        print(f"Removing {ZIP_PATH}...")
        ZIP_PATH.unlink()
        print("  Cleaned up.")


def main():
    print("=" * 60)
    print("Cumberland MS Building + VRI Vegetation Mask Generation")
    print("=" * 60)

    # Build tile catalog from GeoTIFFs
    print("\n--- Building tile catalog ---")
    tif_info = get_tif_info()
    print(f"  Found {len(tif_info)} GeoTIFF files")
    for name, info in tif_info.items():
        print(f"    {name}: {info['width']}x{info['height']} @ {info['res'][0]:.4f}m")

    tile_catalog = build_tile_catalog(tif_info)
    print(f"  Total tiles cataloged: {len(tile_catalog)}")

    # Step 1: Microsoft Building Footprints
    print("\n" + "=" * 60)
    print("STEP 1: Microsoft Building Footprints")
    print("=" * 60)

    download_bc_buildings()
    buildings_4326 = extract_cumberland_buildings()

    if buildings_4326:
        print(f"  Reprojecting {len(buildings_4326)} buildings to EPSG:32610...")
        buildings_32610 = reproject_buildings(buildings_4326)
        print(f"  Generating building masks...")
        generate_building_masks(buildings_32610, tile_catalog)
    else:
        print("  WARNING: No buildings found! Creating empty masks...")
        buildings_32610 = []
        generate_building_masks([], tile_catalog)

    # Cleanup zip
    cleanup()
    gc.collect()

    # Step 2: VRI Vegetation
    print("\n" + "=" * 60)
    print("STEP 2: VRI Vegetation Masks")
    print("=" * 60)

    try:
        veg_geometries = fetch_vri_data()
    except Exception as e:
        print(f"  WFS error: {e}")
        import traceback
        traceback.print_exc()
        veg_geometries = []

    if veg_geometries:
        print(f"  Generating woodland masks from {len(veg_geometries)} polygons...")
        generate_woodland_masks(veg_geometries, tile_catalog)
    else:
        print("  WARNING: No VRI data! Creating empty woodland masks...")
        generate_woodland_masks([], tile_catalog)

    print("\n" + "=" * 60)
    print("Mask generation complete!")
    print(f"  Building masks: {BUILDING_MASK_DIR}")
    print(f"  Woodland masks: {WOODLAND_MASK_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
