"""
Download Cumberland aerial imagery from BC Government WMS
==========================================================
Downloads 1m resolution orthophoto tiles covering the residential
areas of Cumberland, BC for FireSmart case study (Tier 1 comparison).

Saves as georeferenced GeoTIFFs in EPSG:32610 (UTM Zone 10N).

Usage:
    python download_cumberland.py

Output:
    ./cumberland_imagery/  (GeoTIFF files)
"""

import requests
import numpy as np
from pathlib import Path
from PIL import Image
import io
import time
import rasterio
from rasterio.transform import from_bounds

# WMS endpoint
WMS_URL = "https://openmaps.gov.bc.ca/imagex/ecw_wms.dll?"
LAYER = "bc_bc_bc_xc1m_bcalb_1995_2004"  # Province-wide 1m color ortho

# Output directory
OUT_DIR = Path("/home/jeremy/COMP4900_Project_backup_files/cumberland_imagery")

# Cumberland, BC bounds (lat/lon)
# Covers the village and surrounding residential areas
AREA_BOUNDS = {
    "west": -125.0522,
    "east": -125.0259,
    "south": 49.6149,
    "north": 49.6315,
}

# Tile size: each tile covers this many degrees
TILE_SIZE_DEG = 0.005  # smaller tiles since area is small (~500m per tile)
TILE_PIXELS = 1024  # pixels per tile


def download_tile(lon_min, lat_min, lon_max, lat_max, retries=3):
    """Download a single tile from the WMS."""
    url = (
        f"{WMS_URL}"
        f"SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1"
        f"&LAYERS={LAYER}"
        f"&STYLES="
        f"&SRS=EPSG:4326"
        f"&BBOX={lon_min},{lat_min},{lon_max},{lat_max}"
        f"&WIDTH={TILE_PIXELS}&HEIGHT={TILE_PIXELS}"
        f"&FORMAT=image/jpeg"
    )

    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
                return r.content
        except requests.RequestException:
            pass
        if attempt < retries - 1:
            time.sleep(2)

    return None


def is_blank(img_data, threshold=0.95):
    """Check if an image is mostly blank/white."""
    img = Image.open(io.BytesIO(img_data))
    arr = np.array(img)
    white_frac = (arr > 250).all(axis=2).mean()
    return white_frac > threshold


def latlon_to_utm10n(lon, lat):
    """Approximate conversion from WGS84 lat/lon to UTM Zone 10N."""
    import math
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = 2 * f - f * f
    k0 = 0.9996
    lon0 = -123.0  # UTM Zone 10N central meridian

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lon0_rad = math.radians(lon0)

    N = a / math.sqrt(1 - e2 * math.sin(lat_rad) ** 2)
    T = math.tan(lat_rad) ** 2
    C = (e2 / (1 - e2)) * math.cos(lat_rad) ** 2
    A = (lon_rad - lon0_rad) * math.cos(lat_rad)
    e4 = e2 * e2
    e6 = e4 * e2

    M = a * (
        (1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * lat_rad
        - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * math.sin(2 * lat_rad)
        + (15 * e4 / 256 + 45 * e6 / 1024) * math.sin(4 * lat_rad)
        - (35 * e6 / 3072) * math.sin(6 * lat_rad)
    )

    easting = k0 * N * (
        A + (1 - T + C) * A ** 3 / 6
        + (5 - 18 * T + T ** 2 + 72 * C - 58 * (e2 / (1 - e2))) * A ** 5 / 120
    ) + 500000.0

    northing = k0 * (
        M + N * math.tan(lat_rad) * (
            A ** 2 / 2
            + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
            + (61 - 58 * T + T ** 2 + 600 * C - 330 * (e2 / (1 - e2))) * A ** 6 / 720
        )
    )

    return easting, northing


def save_geotiff(img_data, filepath, lon_min, lat_min, lon_max, lat_max):
    """Save image as a GeoTIFF with UTM Zone 10N CRS."""
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    arr = np.array(img)

    # Convert corners to UTM 10N
    ul_e, ul_n = latlon_to_utm10n(lon_min, lat_max)  # upper-left
    lr_e, lr_n = latlon_to_utm10n(lon_max, lat_min)  # lower-right

    height, width = arr.shape[:2]
    transform = from_bounds(ul_e, lr_n, lr_e, ul_n, width, height)

    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=arr.dtype,
        crs="EPSG:32610",
        transform=transform,
    ) as dst:
        for band in range(3):
            dst.write(arr[:, :, band], band + 1)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Calculate tile grid
    lon_start = AREA_BOUNDS["west"]
    lon_end = AREA_BOUNDS["east"]
    lat_start = AREA_BOUNDS["south"]
    lat_end = AREA_BOUNDS["north"]

    lon_steps = int(np.ceil((lon_end - lon_start) / TILE_SIZE_DEG))
    lat_steps = int(np.ceil((lat_end - lat_start) / TILE_SIZE_DEG))
    total = lon_steps * lat_steps

    print(f"Cumberland WMS Imagery Downloader")
    print(f"==================================")
    print(f"Layer: {LAYER} (1m resolution)")
    print(f"Area: {lon_start}W to {lon_end}W, {lat_start}N to {lat_end}N")
    print(f"Tile grid: {lon_steps} x {lat_steps} = {total} tiles")
    print(f"Tile size: {TILE_PIXELS}x{TILE_PIXELS} px")
    print(f"Output: {OUT_DIR}")
    print()

    downloaded = 0
    skipped = 0

    for lon_idx in range(lon_steps):
        for lat_idx in range(lat_steps):
            lon_min = lon_start + lon_idx * TILE_SIZE_DEG
            lat_min = lat_start + lat_idx * TILE_SIZE_DEG
            lon_max = min(lon_min + TILE_SIZE_DEG, lon_end)
            lat_max = min(lat_min + TILE_SIZE_DEG, lat_end)

            tile_name = f"cb_{lon_idx:03d}_{lat_idx:03d}.tif"
            tile_path = OUT_DIR / tile_name

            if tile_path.exists():
                downloaded += 1
                continue

            tile_num = lon_idx * lat_steps + lat_idx + 1
            print(f"  [{tile_num}/{total}] Downloading {tile_name}...", end=" ", flush=True)

            img_data = download_tile(lon_min, lat_min, lon_max, lat_max)
            if img_data is None:
                print("FAILED")
                skipped += 1
                continue

            if is_blank(img_data):
                print("blank (outside coverage)")
                skipped += 1
                continue

            save_geotiff(img_data, tile_path, lon_min, lat_min, lon_max, lat_max)
            downloaded += 1
            print(f"OK ({len(img_data)//1024} KB)")

            time.sleep(0.3)

    print(f"\nDone: {downloaded} tiles downloaded, {skipped} skipped")
    print(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
