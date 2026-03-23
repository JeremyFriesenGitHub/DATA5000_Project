#!/usr/bin/env python3
"""
Download ParcelMap BC province-wide parcel polygons, extract parcels
covering West Kelowna, reproject to EPSG:32610, and save as shapefile.
"""

import os
import sys
import tempfile
import zipfile
import urllib.request
import shutil

import geopandas as gpd
import fiona
from shapely.geometry import box

# ---- Configuration ----
URL = "https://pub.data.gov.bc.ca/datasets/2f4117d9-41fc-44db-87d4-dbdb77f14086/pmbc_parcel_poly_sv.zip"
OUTPUT_DIR = "/home/jeremy/COMP4900_Project_backup_files/west_kelowna_parcels"
OUTPUT_SHP = os.path.join(OUTPUT_DIR, "west_kelowna_parcels.shp")

# West Kelowna bounding box in WGS84 (EPSG:4326)
LON_MIN, LON_MAX = -119.66, -119.48
LAT_MIN, LAT_MAX = 49.82, 49.92

TARGET_CRS = "EPSG:32610"  # UTM Zone 10N

def download_with_progress(url, dest_path):
    """Download a file with progress reporting."""
    print(f"Downloading: {url}")
    print(f"Destination: {dest_path}")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    response = urllib.request.urlopen(req, timeout=600)

    total_size = response.headers.get('Content-Length')
    if total_size:
        total_size = int(total_size)
        print(f"File size: {total_size / (1024**3):.2f} GB")
    else:
        print("File size: unknown")

    downloaded = 0
    chunk_size = 1024 * 1024  # 1MB chunks
    last_pct = -1

    with open(dest_path, 'wb') as f:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = int(downloaded / total_size * 100)
                if pct != last_pct and pct % 5 == 0:
                    print(f"  {pct}% ({downloaded / (1024**3):.2f} GB / {total_size / (1024**3):.2f} GB)")
                    last_pct = pct
            else:
                if downloaded % (100 * 1024 * 1024) == 0:
                    print(f"  Downloaded {downloaded / (1024**3):.2f} GB...")

    print(f"Download complete: {downloaded / (1024**3):.2f} GB")
    return dest_path


def main():
    tmpdir = tempfile.mkdtemp(prefix="pmbc_")
    print(f"Temp directory: {tmpdir}")

    try:
        # Step 1: Download the zip
        zip_path = os.path.join(tmpdir, "pmbc_parcel_poly_sv.zip")
        download_with_progress(URL, zip_path)

        # Step 2: Extract the zip
        print("\nExtracting zip file...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(tmpdir)
        print("Extraction complete.")

        # Remove zip to free disk space
        os.remove(zip_path)
        print("Removed zip file to free space.")

        # Step 3: Find the .gdb directory
        gdb_path = None
        for item in os.listdir(tmpdir):
            if item.endswith('.gdb'):
                gdb_path = os.path.join(tmpdir, item)
                break

        if not gdb_path:
            # Search recursively
            for root, dirs, files in os.walk(tmpdir):
                for d in dirs:
                    if d.endswith('.gdb'):
                        gdb_path = os.path.join(root, d)
                        break
                if gdb_path:
                    break

        if not gdb_path:
            print("ERROR: Could not find .gdb directory in extracted files.")
            print("Contents of temp dir:")
            for item in os.listdir(tmpdir):
                print(f"  {item}")
            sys.exit(1)

        print(f"\nFound GDB: {gdb_path}")

        # List layers in the GDB
        layers = fiona.listlayers(gdb_path)
        print(f"Layers in GDB: {layers}")

        # Step 4: Read with bounding box filter
        # First, determine the CRS of the source data
        layer_name = layers[0] if len(layers) == 1 else None
        if layer_name is None:
            # Try to find a parcel polygon layer
            for l in layers:
                if 'parcel' in l.lower() or 'poly' in l.lower():
                    layer_name = l
                    break
            if layer_name is None:
                layer_name = layers[0]

        print(f"\nUsing layer: {layer_name}")

        # Get source CRS info
        with fiona.open(gdb_path, layer=layer_name) as src:
            src_crs = src.crs
            print(f"Source CRS: {src_crs}")
            print(f"Total features: {len(src)}")

        # Create bounding box in WGS84 and transform to source CRS if needed
        bbox_wgs84 = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
        bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_wgs84], crs="EPSG:4326")

        # Transform bbox to source CRS for spatial filtering
        bbox_src = bbox_gdf.to_crs(src_crs)
        src_bounds = bbox_src.total_bounds  # [minx, miny, maxx, maxy]
        print(f"\nBounding box in source CRS: {src_bounds}")

        # Read with bbox filter - much faster than reading everything
        print("\nReading parcels within bounding box...")
        parcels = gpd.read_file(
            gdb_path,
            layer=layer_name,
            bbox=tuple(src_bounds)
        )
        print(f"Parcels found in bbox: {len(parcels)}")

        if len(parcels) == 0:
            print("WARNING: No parcels found! Trying alternative approach...")
            # Try reading with WGS84 bbox directly
            parcels = gpd.read_file(
                gdb_path,
                layer=layer_name,
                bbox=(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
            )
            print(f"Parcels found (alt): {len(parcels)}")

        if len(parcels) == 0:
            print("ERROR: No parcels found in the West Kelowna area.")
            sys.exit(1)

        # Step 5: Clip to exact bounding box
        # Transform parcels to WGS84 for clipping
        parcels_wgs84 = parcels.to_crs("EPSG:4326")
        clip_box = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
        clipped = parcels_wgs84[parcels_wgs84.intersects(clip_box)]
        print(f"Parcels after intersection filter: {len(clipped)}")

        # Step 6: Reproject to EPSG:32610
        print(f"\nReprojecting to {TARGET_CRS}...")
        clipped_utm = clipped.to_crs(TARGET_CRS)

        # Step 7: Save as shapefile
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"\nSaving to: {OUTPUT_SHP}")
        clipped_utm.to_file(OUTPUT_SHP)
        print(f"Saved {len(clipped_utm)} parcels.")

        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total parcels extracted: {len(clipped_utm)}")
        print(f"CRS: {clipped_utm.crs}")
        print(f"Bounds (UTM): {clipped_utm.total_bounds}")
        print(f"Columns: {list(clipped_utm.columns)}")
        print(f"Output: {OUTPUT_SHP}")

        # List output files
        print("\nOutput files:")
        for f in sorted(os.listdir(OUTPUT_DIR)):
            fpath = os.path.join(OUTPUT_DIR, f)
            size = os.path.getsize(fpath)
            print(f"  {f}: {size / 1024:.1f} KB")

    finally:
        # Cleanup temp directory
        print(f"\nCleaning up temp directory: {tmpdir}")
        shutil.rmtree(tmpdir, ignore_errors=True)
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
