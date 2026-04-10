"""
Parcel-Level FireSmart Risk Assessment Pipeline
=================================================
Stitches inference masks back to georeferenced GeoTIFF coordinates,
clips to property parcels, and computes per-property wildfire risk.

Inputs:
  - GeoTIFF aerial images (with tiepoint + pixel scale metadata)
  - Inference masks from eval_binary.py (building + woodland per tile)
  - Parcel shapefile (ParcelMap BC subdivision polygons)

Output:
  - CSV with one row per property (PID), including risk score and overlay path
  - Per-parcel overlay images showing FireSmart zones

Usage:
    python parcel_risk.py \
        --tif_dir ./tif \
        --masks_dir ./cumberland_geotiff_results \
        --parcels ./ParcelsSubdivisonOnly/ParcelsSubdivisonOnly.shp \
        --output_dir ./firesmart_property_results

Requirements:
    pip install numpy opencv-python Pillow scipy pyshp
"""

import argparse
import csv
import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from scipy import ndimage
import shapefile

Image.MAX_IMAGE_PIXELS = 300000000

TILE_SIZE = 512

ALIGNMENT_OFFSETS = {}

# ============================================================
# FIRESMART ZONES (same as firesmart_risk.py)
# ============================================================

ZONES = {
    "zone_1a": {"min_m": 0.0,  "max_m": 1.5,  "color": [255, 0, 0],     "label": "Critical (0-1.5m)"},
    "zone_1b": {"min_m": 1.5,  "max_m": 10.0,  "color": [255, 128, 0],   "label": "High (1.5-10m)"},
    "zone_2":  {"min_m": 10.0, "max_m": 30.0,  "color": [255, 255, 0],   "label": "Moderate (10-30m)"},
    "zone_3":  {"min_m": 30.0, "max_m": 999.0, "color": [0, 200, 0],     "label": "Low (30m+)"},
}

BUFFER_M = 35.0  # buffer beyond parcel boundary for FireSmart zone computation


# ============================================================
# GEOTIFF METADATA
# ============================================================

def load_geotiff_meta(tif_path):
    """Extract georeferencing from a GeoTIFF: origin, pixel scale, dimensions."""
    img = Image.open(tif_path)
    w, h = img.size
    tags = img.tag_v2

    # ModelPixelScaleTag (33550): [scaleX, scaleY, scaleZ]
    pixel_scale = tags.get(33550, None)
    # ModelTiepointTag (33922): [i, j, k, x, y, z] — pixel (i,j) maps to UTM (x,y)
    tiepoint = tags.get(33922, None)

    if pixel_scale is None or tiepoint is None:
        raise ValueError(f"Missing GeoTIFF tags in {tif_path}")

    gsd_x = pixel_scale[0]  # meters per pixel in X
    gsd_y = pixel_scale[1]  # meters per pixel in Y

    # UTM origin of pixel (0, 0) = top-left corner
    origin_x = tiepoint[3]  # UTM easting
    origin_y = tiepoint[4]  # UTM northing

    img.close()

    return {
        "path": str(tif_path),
        "width": w,
        "height": h,
        "gsd_x": gsd_x,
        "gsd_y": gsd_y,
        "origin_x": origin_x,  # UTM easting of top-left
        "origin_y": origin_y,  # UTM northing of top-left
        # Bounding box in UTM
        "bbox": (
            origin_x,                  # min easting (left)
            origin_y - h * gsd_y,      # min northing (bottom)
            origin_x + w * gsd_x,      # max easting (right)
            origin_y,                  # max northing (top)
        ),
        "short_name": Path(tif_path).stem[:8],
    }


def utm_to_pixel(utm_x, utm_y, meta, apply_offset=True):
    """Convert UTM coordinates to pixel coordinates in a GeoTIFF."""
    px = (utm_x - meta["origin_x"]) / meta["gsd_x"]
    py = (meta["origin_y"] - utm_y) / meta["gsd_y"]  # Y is flipped
    if apply_offset:
        sn = meta["short_name"]
        if sn in ALIGNMENT_OFFSETS:
            dx, dy = ALIGNMENT_OFFSETS[sn]
            px += dx
            py += dy
    return px, py


def pixel_to_utm(px, py, meta):
    """Convert pixel coordinates to UTM coordinates."""
    utm_x = meta["origin_x"] + px * meta["gsd_x"]
    utm_y = meta["origin_y"] - py * meta["gsd_y"]
    return utm_x, utm_y


# ============================================================
# MASK STITCHING
# ============================================================

def stitch_masks(masks_dir, short_name, full_h, full_w, mask_type="building_masks"):
    """
    Reconstruct full-size mask from 512x512 tiles.
    Tile names: {short_name}_{row:05d}_{col:05d}.png
    """
    mask = np.zeros((full_h, full_w), dtype=np.uint8)
    mask_path = Path(masks_dir) / mask_type

    for tile_file in mask_path.glob(f"{short_name}_*.png"):
        parts = tile_file.stem.split("_")
        row = int(parts[1])
        col = int(parts[2])

        tile = np.array(Image.open(tile_file))
        if tile.ndim == 3:
            tile = tile[:, :, 0]

        # Binary: threshold at 127 for building/woodland masks saved as 0/255
        tile_bin = (tile > 127).astype(np.uint8)

        th, tw = tile_bin.shape
        # Clip to image bounds
        end_row = min(row + th, full_h)
        end_col = min(col + tw, full_w)
        mask[row:end_row, col:end_col] = tile_bin[:end_row - row, :end_col - col]

    return mask


def load_image_region(tif_path, row_start, row_end, col_start, col_end):
    """Load a region of the original GeoTIFF as RGB."""
    img = Image.open(tif_path).convert("RGB")
    arr = np.array(img)
    img.close()

    h, w = arr.shape[:2]
    # Clamp to image bounds
    r0 = max(0, row_start)
    r1 = min(h, row_end)
    c0 = max(0, col_start)
    c1 = min(w, col_end)

    # Create output with padding if needed
    out_h = row_end - row_start
    out_w = col_end - col_start
    result = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    result[r0 - row_start:r1 - row_start, c0 - col_start:c1 - col_start] = arr[r0:r1, c0:c1]

    return result


# ============================================================
# PARCEL PROCESSING
# ============================================================

def parcel_polygon_to_pixel(shape, meta):
    """Convert parcel polygon (UTM coords) to pixel coords in a GeoTIFF."""
    pixel_points = []
    for x, y in shape.points:
        px, py = utm_to_pixel(x, y, meta)
        pixel_points.append((px, py))
    return pixel_points


def create_parcel_mask(pixel_points, crop_offset_x, crop_offset_y, crop_h, crop_w):
    """Create a binary mask of the parcel polygon within the crop region."""
    adjusted = []
    for px, py in pixel_points:
        adjusted.append((int(px - crop_offset_x), int(py - crop_offset_y)))

    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    pts = np.array(adjusted, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def find_covering_geotiff(parcel_bbox, geotiff_metas):
    """Find which GeoTIFF best covers a parcel (by overlap area)."""
    px_min, py_min, px_max, py_max = parcel_bbox
    best = None
    best_overlap = 0

    for meta in geotiff_metas:
        gx_min, gy_min, gx_max, gy_max = meta["bbox"]

        # Overlap rectangle
        ox_min = max(px_min, gx_min)
        oy_min = max(py_min, gy_min)
        ox_max = min(px_max, gx_max)
        oy_max = min(py_max, gy_max)

        if ox_min < ox_max and oy_min < oy_max:
            overlap = (ox_max - ox_min) * (oy_max - oy_min)
            if overlap > best_overlap:
                best_overlap = overlap
                best = meta

    return best


def meters_to_pixels(meters, gsd):
    return int(meters / gsd)


# ============================================================
# RISK SCORING (same as firesmart_risk.py)
# ============================================================

def compute_building_risk(building_mask_single, woodland_mask, gsd, parcel_mask=None):
    h, w = building_mask_single.shape

    building_border = cv2.dilate(building_mask_single, np.ones((3, 3), np.uint8)) - building_mask_single
    if building_border.sum() == 0:
        building_border = building_mask_single

    dist_from_building = ndimage.distance_transform_edt(1 - building_border)

    veg_pixels = woodland_mask > 0

    # Split vegetation into on-parcel and off-parcel
    if parcel_mask is not None:
        on_parcel = parcel_mask > 0
        veg_on_parcel = veg_pixels & on_parcel
        veg_off_parcel = veg_pixels & ~on_parcel
    else:
        veg_on_parcel = veg_pixels
        veg_off_parcel = np.zeros_like(veg_pixels)

    if veg_pixels.any():
        veg_distances = dist_from_building[veg_pixels]
        min_distance_m = float(veg_distances.min()) * gsd
    else:
        min_distance_m = 999.0

    # Min distance split by on/off parcel
    if veg_on_parcel.any():
        min_dist_on = float(dist_from_building[veg_on_parcel].min()) * gsd
    else:
        min_dist_on = 999.0
    if veg_off_parcel.any():
        min_dist_off = float(dist_from_building[veg_off_parcel].min()) * gsd
    else:
        min_dist_off = 999.0

    zone_vegetation = {}
    for zone_name, zone_def in ZONES.items():
        min_px = meters_to_pixels(zone_def["min_m"], gsd)
        max_px = meters_to_pixels(zone_def["max_m"], gsd)

        zone_ring = (dist_from_building >= min_px) & (dist_from_building < max_px)
        zone_ring = zone_ring & (~(building_mask_single > 0))

        zone_total = zone_ring.sum()
        zone_veg = (zone_ring & veg_pixels).sum()

        # On/off parcel vegetation within this zone
        zone_veg_on = int((zone_ring & veg_on_parcel).sum())
        zone_veg_off = int((zone_ring & veg_off_parcel).sum())

        # On/off parcel area within this zone
        if parcel_mask is not None:
            zone_on_total = int((zone_ring & on_parcel).sum())
            zone_off_total = int((zone_ring & ~on_parcel).sum())
        else:
            zone_on_total = int(zone_total)
            zone_off_total = 0

        zone_vegetation[zone_name] = {
            "veg_pixels": int(zone_veg),
            "total_pixels": int(zone_total),
            "veg_density": float(zone_veg / zone_total) if zone_total > 0 else 0.0,
            "on_parcel_veg": zone_veg_on,
            "on_parcel_total": zone_on_total,
            "on_parcel_density": float(zone_veg_on / zone_on_total) if zone_on_total > 0 else 0.0,
            "off_parcel_veg": zone_veg_off,
            "off_parcel_total": zone_off_total,
            "off_parcel_density": float(zone_veg_off / zone_off_total) if zone_off_total > 0 else 0.0,
        }

    risk_score = compute_risk_score(min_distance_m, zone_vegetation)
    return min_distance_m, min_dist_on, min_dist_off, zone_vegetation, risk_score


def compute_risk_score(min_distance_m, zone_vegetation):
    # Density-only score: weighted sum of vegetation density per zone
    density_score = 0.0
    weights = {"zone_1a": 5.0, "zone_1b": 3.0, "zone_2": 1.25, "zone_3": 0.75}
    for zone_name, weight in weights.items():
        if zone_name in zone_vegetation:
            density = zone_vegetation[zone_name]["veg_density"]
            density_score += weight * density

    return round(density_score, 2)


# ============================================================
# OVERLAY VISUALIZATION
# ============================================================

def create_parcel_overlay(orig_crop, building_mask, woodland_mask, parcel_mask,
                          buildings, gsd, parcel_boundary_pts):
    overlay = orig_crop.copy().astype(np.float32)

    # Distance from buildings for zone rings
    building_border = cv2.dilate(building_mask, np.ones((3, 3), np.uint8)) - building_mask
    if building_border.sum() == 0:
        building_border = building_mask
    dist_from_building = ndimage.distance_transform_edt(1 - building_border)

    # Draw zone rings
    zone_order = ["zone_3", "zone_2", "zone_1b", "zone_1a"]
    for zone_name in zone_order:
        zone_def = ZONES[zone_name]
        min_px = meters_to_pixels(zone_def["min_m"], gsd)
        max_px = meters_to_pixels(zone_def["max_m"], gsd)
        zone_ring = (dist_from_building >= min_px) & (dist_from_building < max_px)
        zone_ring = zone_ring & (~(building_mask > 0))
        if zone_ring.any():
            color = np.array(zone_def["color"], dtype=np.float32)
            overlay[zone_ring] = overlay[zone_ring] * 0.7 + color * 0.3

    # Vegetation
    veg_mask = woodland_mask > 0
    overlay[veg_mask] = overlay[veg_mask] * 0.6 + np.array([0, 180, 0], dtype=np.float32) * 0.4

    # Buildings
    bld_mask = building_mask > 0
    overlay[bld_mask] = overlay[bld_mask] * 0.5 + np.array([0, 80, 255], dtype=np.float32) * 0.5

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Draw parcel boundary
    if parcel_boundary_pts:
        pts = np.array(parcel_boundary_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=True, color=(255, 255, 0), thickness=2)

    # Draw risk scores
    for bld in buildings:
        cx, cy = bld["centroid"]
        risk = bld.get("risk_score", 0)
        if risk >= 7:
            txt_color = (255, 0, 0)
        elif risk >= 4:
            txt_color = (255, 165, 0)
        else:
            txt_color = (0, 200, 0)
        cv2.putText(overlay, f"{risk}", (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(overlay, f"{risk}", (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 2)

    return overlay


# ============================================================
# BUILDING EXTRACTION
# ============================================================

def extract_buildings(building_mask, min_area=100):
    contours, _ = cv2.findContours(building_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    buildings = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        buildings.append({
            "id": i,
            "contour": cnt,
            "centroid": (cx, cy),
            "area_px": area,
        })
    return buildings


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Parcel-Level FireSmart Risk Assessment")
    parser.add_argument("--tif_dir", type=str, required=True, help="Directory with GeoTIFF files")
    parser.add_argument("--masks_dir", type=str, required=True, help="Directory with inference results")
    parser.add_argument("--parcels", type=str, required=True, help="Path to parcel shapefile (.shp)")
    parser.add_argument("--output_dir", type=str, default="./firesmart_property_results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    (output_dir / "overlay").mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Step 1: Load GeoTIFF metadata
    # --------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Loading GeoTIFF metadata")
    print("=" * 60)

    tif_dir = Path(args.tif_dir)
    geotiff_metas = []
    for tif_path in sorted(tif_dir.glob("*.tif")):
        meta = load_geotiff_meta(tif_path)
        geotiff_metas.append(meta)
        print(f"  {meta['short_name']}: {meta['width']}x{meta['height']} "
              f"GSD={meta['gsd_x']:.4f}m bbox=({meta['bbox'][0]:.0f}, {meta['bbox'][1]:.0f}, "
              f"{meta['bbox'][2]:.0f}, {meta['bbox'][3]:.0f})")

    print(f"\n  Loaded {len(geotiff_metas)} GeoTIFFs")

    # --------------------------------------------------------
    # Step 2: Stitch masks per GeoTIFF
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Stitching inference masks")
    print("=" * 60)

    stitched = {}
    for meta in geotiff_metas:
        sn = meta["short_name"]
        print(f"  Stitching {sn}...", end=" ", flush=True)
        b_mask = stitch_masks(args.masks_dir, sn, meta["height"], meta["width"], "building_masks")
        w_mask = stitch_masks(args.masks_dir, sn, meta["height"], meta["width"], "woodland_masks")
        stitched[sn] = {"building": b_mask, "woodland": w_mask}
        print(f"building={b_mask.sum()} px, woodland={w_mask.sum()} px")

    # --------------------------------------------------------
    # Step 3: Load parcels
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Loading parcel shapefile")
    print("=" * 60)

    sf = shapefile.Reader(args.parcels)
    fields = [f[0] for f in sf.fields[1:]]
    print(f"  {len(sf)} parcels loaded")
    print(f"  Fields: {fields}")

    # --------------------------------------------------------
    # Step 4: Process each parcel
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Computing per-parcel risk")
    print("=" * 60)

    all_parcel_results = []
    parcels_processed = 0
    parcels_skipped = 0
    parcels_no_buildings = 0

    for idx in range(len(sf)):
        shape = sf.shape(idx)
        rec = sf.record(idx)
        rec_dict = dict(zip(fields, rec))

        pid = rec_dict.get("PID", "")
        pid_fmt = rec_dict.get("PID_FORMAT", pid)
        owner_type = rec_dict.get("OWNER_TYPE", "")
        parcel_area_m2 = rec_dict.get("FEATURE_AR", 0)

        # Parcel bounding box in UTM
        p_bbox = shape.bbox  # (min_x, min_y, max_x, max_y)

        # Add buffer for FireSmart zones
        buffered_bbox = (
            p_bbox[0] - BUFFER_M,
            p_bbox[1] - BUFFER_M,
            p_bbox[2] + BUFFER_M,
            p_bbox[3] + BUFFER_M,
        )

        # Find covering GeoTIFF
        meta = find_covering_geotiff(buffered_bbox, geotiff_metas)
        if meta is None:
            parcels_skipped += 1
            continue

        sn = meta["short_name"]
        gsd = (meta["gsd_x"] + meta["gsd_y"]) / 2.0  # average GSD

        # Convert buffered bbox to pixel coords
        px_min, py_min = utm_to_pixel(buffered_bbox[0], buffered_bbox[3], meta)  # top-left
        px_max, py_max = utm_to_pixel(buffered_bbox[2], buffered_bbox[1], meta)  # bottom-right

        # Clamp to image bounds
        px_min = max(0, int(px_min))
        py_min = max(0, int(py_min))
        px_max = min(meta["width"], int(px_max))
        py_max = min(meta["height"], int(py_max))

        crop_w = px_max - px_min
        crop_h = py_max - py_min
        if crop_w < 10 or crop_h < 10:
            parcels_skipped += 1
            continue

        # Crop masks
        b_crop = stitched[sn]["building"][py_min:py_max, px_min:px_max].copy()
        w_crop = stitched[sn]["woodland"][py_min:py_max, px_min:px_max].copy()

        # Create parcel boundary mask (for clipping buildings to parcel)
        pixel_points = parcel_polygon_to_pixel(shape, meta)
        adjusted_pts = [(int(px - px_min), int(py - py_min)) for px, py in pixel_points]
        parcel_mask = create_parcel_mask(pixel_points, px_min, py_min, crop_h, crop_w)

        # Assign buildings to parcel if any building pixels overlap the parcel
        buildings = extract_buildings(b_crop)
        parcel_buildings = []
        for bld in buildings:
            bld_mask_single = np.zeros((crop_h, crop_w), dtype=np.uint8)
            cv2.drawContours(bld_mask_single, [bld["contour"]], -1, 1, -1)
            overlap = (bld_mask_single & parcel_mask).sum()
            if overlap > 0:
                parcel_buildings.append(bld)

        if not parcel_buildings:
            parcels_no_buildings += 1
            # Still record the parcel with no buildings
            all_parcel_results.append({
                "pid": pid,
                "pid_format": pid_fmt,
                "owner_type": owner_type,
                "parcel_area_m2": round(parcel_area_m2, 1),
                "num_buildings": 0,
                "max_risk_score": None,
                "mean_risk_score": None,
                "min_veg_distance_m": None,
                "min_veg_dist_on_parcel": None,
                "min_veg_dist_off_parcel": None,
                "zone_1a_veg_density": None,
                "zone_1b_veg_density": None,
                "zone_2_veg_density": None,
                "zone_1a_on_parcel": None,
                "zone_1a_off_parcel": None,
                "zone_1b_on_parcel": None,
                "zone_1b_off_parcel": None,
                "zone_2_on_parcel": None,
                "zone_2_off_parcel": None,
                "overlay_image": "",
            })
            continue

        # Compute risk for each building in this parcel
        building_results = []
        for bld in parcel_buildings:
            bld_mask = np.zeros_like(b_crop)
            cv2.drawContours(bld_mask, [bld["contour"]], -1, 1, -1)

            min_dist, min_dist_on, min_dist_off, zone_veg, risk_score = \
                compute_building_risk(bld_mask, w_crop, gsd, parcel_mask)
            bld["min_distance_m"] = min_dist
            bld["min_dist_on_parcel"] = min_dist_on
            bld["min_dist_off_parcel"] = min_dist_off
            bld["zone_vegetation"] = zone_veg
            bld["risk_score"] = risk_score
            bld["area_m2"] = bld["area_px"] * gsd * gsd

            building_results.append({
                "risk_score": risk_score,
                "min_distance_m": min_dist,
                "min_dist_on_parcel": min_dist_on,
                "min_dist_off_parcel": min_dist_off,
                "zone_1a": zone_veg["zone_1a"]["veg_density"],
                "zone_1b": zone_veg["zone_1b"]["veg_density"],
                "zone_2": zone_veg["zone_2"]["veg_density"],
                "zone_1a_on": zone_veg["zone_1a"]["on_parcel_density"],
                "zone_1a_off": zone_veg["zone_1a"]["off_parcel_density"],
                "zone_1b_on": zone_veg["zone_1b"]["on_parcel_density"],
                "zone_1b_off": zone_veg["zone_1b"]["off_parcel_density"],
                "zone_2_on": zone_veg["zone_2"]["on_parcel_density"],
                "zone_2_off": zone_veg["zone_2"]["off_parcel_density"],
            })

        # Aggregate per-property
        risk_scores = [b["risk_score"] for b in building_results]
        min_dists = [b["min_distance_m"] for b in building_results if b["min_distance_m"] < 999]
        min_dists_on = [b["min_dist_on_parcel"] for b in building_results if b["min_dist_on_parcel"] < 999]
        min_dists_off = [b["min_dist_off_parcel"] for b in building_results if b["min_dist_off_parcel"] < 999]

        overlay_name = f"parcel_{pid}.png"

        # Generate overlay
        orig_crop = load_image_region(meta["path"], py_min, py_max, px_min, px_max)
        overlay = create_parcel_overlay(
            orig_crop, b_crop, w_crop, parcel_mask,
            parcel_buildings, gsd, adjusted_pts
        )
        Image.fromarray(overlay).save(output_dir / "overlay" / overlay_name)

        all_parcel_results.append({
            "pid": pid,
            "pid_format": pid_fmt,
            "owner_type": owner_type,
            "parcel_area_m2": round(parcel_area_m2, 1),
            "num_buildings": len(parcel_buildings),
            "max_risk_score": max(risk_scores),
            "mean_risk_score": round(np.mean(risk_scores), 1),
            "min_veg_distance_m": round(min(min_dists), 2) if min_dists else 999.0,
            "min_veg_dist_on_parcel": round(min(min_dists_on), 2) if min_dists_on else 999.0,
            "min_veg_dist_off_parcel": round(min(min_dists_off), 2) if min_dists_off else 999.0,
            "zone_1a_veg_density": round(np.mean([b["zone_1a"] for b in building_results]), 3),
            "zone_1b_veg_density": round(np.mean([b["zone_1b"] for b in building_results]), 3),
            "zone_2_veg_density": round(np.mean([b["zone_2"] for b in building_results]), 3),
            "zone_1a_on_parcel": round(np.mean([b["zone_1a_on"] for b in building_results]), 3),
            "zone_1a_off_parcel": round(np.mean([b["zone_1a_off"] for b in building_results]), 3),
            "zone_1b_on_parcel": round(np.mean([b["zone_1b_on"] for b in building_results]), 3),
            "zone_1b_off_parcel": round(np.mean([b["zone_1b_off"] for b in building_results]), 3),
            "zone_2_on_parcel": round(np.mean([b["zone_2_on"] for b in building_results]), 3),
            "zone_2_off_parcel": round(np.mean([b["zone_2_off"] for b in building_results]), 3),
            "overlay_image": f"overlay/{overlay_name}",
        })
        parcels_processed += 1

        if (parcels_processed % 50) == 0:
            print(f"  Processed {parcels_processed} parcels with buildings...")

    # --------------------------------------------------------
    # Step 5: Summary and output
    # --------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PARCEL-LEVEL RISK ASSESSMENT SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total parcels in shapefile: {len(sf)}")
    print(f"  Parcels with buildings: {parcels_processed}")
    print(f"  Parcels without buildings: {parcels_no_buildings}")
    print(f"  Parcels outside imagery: {parcels_skipped}")

    parcels_with_risk = [p for p in all_parcel_results if p["max_risk_score"] is not None]

    if parcels_with_risk:
        scores = [p["max_risk_score"] for p in parcels_with_risk]
        high = sum(1 for s in scores if s >= 7)
        moderate = sum(1 for s in scores if 4 <= s < 7)
        low = sum(1 for s in scores if s < 4)

        print(f"\n  Risk Distribution (by max building risk per property):")
        print(f"    High (7-10):     {high} properties")
        print(f"    Moderate (4-6.9): {moderate} properties")
        print(f"    Low (1-3.9):     {low} properties")
        print(f"\n    Mean max risk: {np.mean(scores):.1f}")
        print(f"    Median max risk: {np.median(scores):.1f}")

    # Save CSV
    csv_path = output_dir / "parcel_risk_scores.csv"
    fieldnames = [
        "pid", "pid_format", "owner_type", "parcel_area_m2", "num_buildings",
        "max_risk_score", "mean_risk_score", "min_veg_distance_m",
        "min_veg_dist_on_parcel", "min_veg_dist_off_parcel",
        "zone_1a_veg_density", "zone_1b_veg_density", "zone_2_veg_density",
        "zone_1a_on_parcel", "zone_1a_off_parcel",
        "zone_1b_on_parcel", "zone_1b_off_parcel",
        "zone_2_on_parcel", "zone_2_off_parcel",
        "overlay_image"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_parcel_results:
            writer.writerow(row)
    print(f"\n  Per-property results: {csv_path}")

    # Save JSON summary
    summary = {
        "total_parcels": len(sf),
        "parcels_with_buildings": parcels_processed,
        "parcels_no_buildings": parcels_no_buildings,
        "parcels_outside_imagery": parcels_skipped,
        "risk_distribution": {
            "high": high if parcels_with_risk else 0,
            "moderate": moderate if parcels_with_risk else 0,
            "low": low if parcels_with_risk else 0,
        } if parcels_with_risk else {},
    }
    json_path = output_dir / "parcel_risk_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {json_path}")

    # Top 10 highest risk
    if parcels_with_risk:
        parcels_with_risk.sort(key=lambda x: x["max_risk_score"], reverse=True)
        print(f"\n  Top 10 Highest Risk Properties:")
        print(f"  {'PID':<15} {'Risk':<6} {'Buildings':<10} {'Min Dist':<10} {'Area m²':<10}")
        print(f"  {'-' * 51}")
        for p in parcels_with_risk[:10]:
            print(f"  {p['pid_format']:<15} {p['max_risk_score']:<6} "
                  f"{p['num_buildings']:<10} {p['min_veg_distance_m']:<10} "
                  f"{p['parcel_area_m2']:<10}")

    print(f"\n  Overlays: {output_dir / 'overlay'}")
    print("  Done!")


if __name__ == "__main__":
    main()
