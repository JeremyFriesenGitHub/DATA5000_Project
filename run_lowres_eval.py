#!/usr/bin/env python3
"""
Low-Resolution FireSmart Evaluation Pipeline
=============================================
Runs tile-level and parcel-level risk assessment for BC case study regions
using MS Building Footprints + Meta Canopy Height masks.

Scoring: density-only formula
  Risk = (Zone1a_density * 5.0) + (Zone1b_density * 3.0)
       + (Zone2_density * 1.25) + (Zone3_density * 0.75)

Output structure:
  eval/{region}_tiles/    — tile-level assessment (without parcels)
  eval/{region}_parcels/  — parcel-level assessment (with property lines)

Usage:
  python run_lowres_eval.py
  python run_lowres_eval.py --regions cumberland west_kelowna
  python run_lowres_eval.py --skip-parcels
"""

import argparse
import csv
import json
import sys
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from scipy import ndimage

Image.MAX_IMAGE_PIXELS = 300000000

BASE_DIR = Path(__file__).parent

# ============================================================
# CONFIGURATION
# ============================================================

ZONES = {
    "zone_1a": {"min_m": 0.0,  "max_m": 1.5,  "color": [255, 0, 0],     "label": "Critical (0-1.5m)"},
    "zone_1b": {"min_m": 1.5,  "max_m": 10.0, "color": [255, 128, 0],   "label": "High (1.5-10m)"},
    "zone_2":  {"min_m": 10.0, "max_m": 30.0, "color": [255, 255, 0],   "label": "Moderate (10-30m)"},
    "zone_3":  {"min_m": 30.0, "max_m": 999.0,"color": [100, 180, 255],  "label": "Low (30m+)"},
}

DENSITY_WEIGHTS = {"zone_1a": 5.0, "zone_1b": 3.0, "zone_2": 1.25, "zone_3": 0.75}

REGIONS = {
    "cumberland": {
        "imagery": "cumberland_imagery",
        "masks": "cumberland_meta_masks",
        "parcels": "ParcelsSubdivisonOnly/ParcelsSubdivisonOnly.shp",
    },
    "west_kelowna": {
        "imagery": "west_kelowna_imagery",
        "masks": "west_kelowna_meta_masks",
        "parcels": "west_kelowna_parcels/west_kelowna_parcels.shp",
    },
    "logan_lake": {
        "imagery": "logan_lake_imagery",
        "masks": "logan_lake_meta_masks",
        "parcels": "logan_lake_parcels/logan_lake_parcels.shp",
    },
    "silver_star": {
        "imagery": "silver_star_imagery",
        "masks": "silver_star_meta_masks",
        "parcels": "silver_star_parcels/silver_star_parcels.shp",
    },
}

BUFFER_M = 35.0  # buffer around parcel for FireSmart zone computation


# ============================================================
# GEOTIFF METADATA
# ============================================================

def load_geotiff_meta(tif_path):
    """Extract georeferencing from a GeoTIFF tile. Returns None if tags missing."""
    img = Image.open(tif_path)
    w, h = img.size
    tags = img.tag_v2
    pixel_scale = tags.get(33550)
    tiepoint = tags.get(33922)
    img.close()
    if pixel_scale is None or tiepoint is None:
        return None
    gsd_x = pixel_scale[0]
    gsd_y = pixel_scale[1]
    origin_x = tiepoint[3]
    origin_y = tiepoint[4]
    return {
        "path": str(tif_path),
        "name": Path(tif_path).stem,
        "width": w,
        "height": h,
        "gsd_x": gsd_x,
        "gsd_y": gsd_y,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "bbox": (
            origin_x,
            origin_y - h * gsd_y,
            origin_x + w * gsd_x,
            origin_y,
        ),
    }


def utm_to_pixel(utm_x, utm_y, origin_x, origin_y, gsd_x, gsd_y):
    """Convert UTM coordinates to pixel coordinates."""
    px = (utm_x - origin_x) / gsd_x
    py = (origin_y - utm_y) / gsd_y
    return px, py


# ============================================================
# MASK LOADING
# ============================================================

def load_mask(mask_path):
    """Load a mask image as binary uint8."""
    img = np.array(Image.open(mask_path))
    if img.ndim == 3:
        img = img[:, :, 0]
    return (img > 127).astype(np.uint8)


# ============================================================
# BUILDING EXTRACTION
# ============================================================

def extract_buildings(building_mask, min_area=100):
    """Find individual buildings as connected components."""
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
# RISK SCORING
# ============================================================

def compute_risk_score(zone_vegetation):
    """
    Density-only risk score:
      (Zone1a_density * 5.0) + (Zone1b_density * 3.0)
      + (Zone2_density * 1.25) + (Zone3_density * 0.75)
    """
    score = 0.0
    for zone_name, weight in DENSITY_WEIGHTS.items():
        if zone_name in zone_vegetation:
            score += weight * zone_vegetation[zone_name]["veg_density"]
    return round(score, 2)


def compute_building_risk(building_mask_single, woodland_mask, gsd_x, gsd_y, parcel_mask=None):
    """
    Compute risk for a single building using anisotropic distance transform.
    Returns: (min_distance_m, min_dist_on, min_dist_off, zone_vegetation, risk_score)
    """
    building_border = cv2.dilate(building_mask_single, np.ones((3, 3), np.uint8)) - building_mask_single
    if building_border.sum() == 0:
        building_border = building_mask_single

    # Anisotropic distance transform: gives distances in meters directly
    dist_m = ndimage.distance_transform_edt(1 - building_border, sampling=(gsd_y, gsd_x))

    veg_pixels = woodland_mask > 0

    # On/off parcel vegetation
    if parcel_mask is not None:
        on_parcel = parcel_mask > 0
        veg_on_parcel = veg_pixels & on_parcel
        veg_off_parcel = veg_pixels & ~on_parcel
    else:
        veg_on_parcel = veg_pixels
        veg_off_parcel = np.zeros_like(veg_pixels)

    min_dist_on = float(dist_m[veg_on_parcel].min()) if veg_on_parcel.any() else 999.0
    min_dist_off = float(dist_m[veg_off_parcel].min()) if veg_off_parcel.any() else 999.0

    # Main min distance uses on-parcel veg only when parcel boundary is provided
    if parcel_mask is not None:
        min_distance_m = min_dist_on
    elif veg_pixels.any():
        min_distance_m = float(dist_m[veg_pixels].min())
    else:
        min_distance_m = 999.0

    # Vegetation density per zone
    zone_vegetation = {}
    for zone_name, zone_def in ZONES.items():
        zone_ring = (dist_m >= zone_def["min_m"]) & (dist_m < zone_def["max_m"])
        zone_ring = zone_ring & (~(building_mask_single > 0))

        zone_veg_on = int((zone_ring & veg_on_parcel).sum())
        zone_veg_off = int((zone_ring & veg_off_parcel).sum())

        if parcel_mask is not None:
            zone_on_total = int((zone_ring & on_parcel).sum())
            zone_off_total = int((zone_ring & ~on_parcel).sum())
            # Main density uses only on-parcel area when parcel boundary is provided
            zone_total = zone_on_total
            zone_veg = zone_veg_on
        else:
            zone_total = int(zone_ring.sum())
            zone_veg = int((zone_ring & veg_pixels).sum())
            zone_on_total = zone_total
            zone_off_total = 0

        zone_vegetation[zone_name] = {
            "veg_pixels": zone_veg,
            "total_pixels": zone_total,
            "veg_density": float(zone_veg / zone_total) if zone_total > 0 else 0.0,
            "on_parcel_veg": zone_veg_on,
            "on_parcel_total": zone_on_total,
            "on_parcel_density": float(zone_veg_on / zone_on_total) if zone_on_total > 0 else 0.0,
            "off_parcel_veg": zone_veg_off,
            "off_parcel_total": zone_off_total,
            "off_parcel_density": float(zone_veg_off / zone_off_total) if zone_off_total > 0 else 0.0,
        }

    risk_score = compute_risk_score(zone_vegetation)
    return min_distance_m, min_dist_on, min_dist_off, zone_vegetation, risk_score


# ============================================================
# VISUALIZATION
# ============================================================

def create_risk_overlay(orig_image, building_mask, woodland_mask, buildings, gsd_x, gsd_y):
    """Create overlay with FireSmart zone rings around buildings."""
    overlay = orig_image.copy().astype(np.float32)

    building_border = cv2.dilate(building_mask, np.ones((3, 3), np.uint8)) - building_mask
    if building_border.sum() == 0:
        building_border = building_mask
    dist_m = ndimage.distance_transform_edt(1 - building_border, sampling=(gsd_y, gsd_x))

    for zone_name in ["zone_3", "zone_2", "zone_1b", "zone_1a"]:
        zone_def = ZONES[zone_name]
        zone_ring = (dist_m >= zone_def["min_m"]) & (dist_m < zone_def["max_m"])
        zone_ring = zone_ring & (~(building_mask > 0))
        if zone_ring.any():
            color = np.array(zone_def["color"], dtype=np.float32)
            overlay[zone_ring] = overlay[zone_ring] * 0.7 + color * 0.3

    veg_mask = woodland_mask > 0
    overlay[veg_mask] = overlay[veg_mask] * 0.6 + np.array([0, 180, 0], dtype=np.float32) * 0.4

    bld_mask = building_mask > 0
    overlay[bld_mask] = overlay[bld_mask] * 0.5 + np.array([0, 80, 255], dtype=np.float32) * 0.5

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

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


def create_parcel_overlay(orig_crop, building_mask, woodland_mask, parcel_mask,
                          buildings, gsd_x, gsd_y, parcel_boundary_pts):
    """Create overlay for a single parcel."""
    overlay = orig_crop.copy().astype(np.float32)

    building_border = cv2.dilate(building_mask, np.ones((3, 3), np.uint8)) - building_mask
    if building_border.sum() == 0:
        building_border = building_mask
    dist_m = ndimage.distance_transform_edt(1 - building_border, sampling=(gsd_y, gsd_x))

    for zone_name in ["zone_3", "zone_2", "zone_1b", "zone_1a"]:
        zone_def = ZONES[zone_name]
        zone_ring = (dist_m >= zone_def["min_m"]) & (dist_m < zone_def["max_m"])
        zone_ring = zone_ring & (~(building_mask > 0))
        if zone_ring.any():
            color = np.array(zone_def["color"], dtype=np.float32)
            overlay[zone_ring] = overlay[zone_ring] * 0.7 + color * 0.3

    veg_mask = woodland_mask > 0
    overlay[veg_mask] = overlay[veg_mask] * 0.6 + np.array([0, 180, 0], dtype=np.float32) * 0.4

    bld_mask = building_mask > 0
    overlay[bld_mask] = overlay[bld_mask] * 0.5 + np.array([0, 80, 255], dtype=np.float32) * 0.5

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    if parcel_boundary_pts:
        pts = np.array(parcel_boundary_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=True, color=(255, 255, 0), thickness=2)

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


def create_legend(height=400, width=250):
    """Create legend image."""
    legend = np.ones((height, width, 3), dtype=np.uint8) * 255
    y = 30
    cv2.putText(legend, "FireSmart Risk Map", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    y += 40
    for zone_name in ["zone_1a", "zone_1b", "zone_2", "zone_3"]:
        zone_def = ZONES[zone_name]
        color = tuple(zone_def["color"][::-1])
        cv2.rectangle(legend, (10, y - 12), (30, y + 4), color, -1)
        cv2.rectangle(legend, (10, y - 12), (30, y + 4), (0, 0, 0), 1)
        cv2.putText(legend, zone_def["label"], (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        y += 30

    y += 10
    cv2.rectangle(legend, (10, y - 12), (30, y + 4), (255, 80, 0), -1)
    cv2.putText(legend, "Building", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    y += 30
    cv2.rectangle(legend, (10, y - 12), (30, y + 4), (0, 180, 0), -1)
    cv2.putText(legend, "Vegetation", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    y += 40
    cv2.putText(legend, "Risk Score (0-10)", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    y += 25
    cv2.putText(legend, "7-10: High risk", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    y += 22
    cv2.putText(legend, "4-6.9: Moderate risk", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)
    y += 22
    cv2.putText(legend, "1-3.9: Low risk", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 0), 1)
    y += 22
    cv2.putText(legend, "0: No vegetation", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)

    return legend


# ============================================================
# TILE-LEVEL EVALUATION (without parcels)
# ============================================================

def run_tile_eval(region_name, config):
    """Run tile-level FireSmart evaluation for a region."""
    print(f"\n{'='*60}")
    print(f"TILE-LEVEL EVAL: {region_name.upper()}")
    print(f"{'='*60}")

    imagery_dir = BASE_DIR / config["imagery"]
    masks_dir = BASE_DIR / config["masks"]
    output_dir = BASE_DIR / "eval" / f"{region_name}_tiles"
    (output_dir / "overlay").mkdir(parents=True, exist_ok=True)

    tile_files = sorted(imagery_dir.glob("*.tif"))
    print(f"  Tiles: {len(tile_files)}")

    # Pre-load GSD from first valid tile as fallback
    fallback_gsd_x, fallback_gsd_y = None, None
    for tf in tile_files:
        m = load_geotiff_meta(tf)
        if m is not None:
            fallback_gsd_x, fallback_gsd_y = m["gsd_x"], m["gsd_y"]
            break

    # Save legend
    legend = create_legend()
    Image.fromarray(legend[:, :, ::-1]).save(output_dir / "legend.png")

    all_buildings = []
    tiles_processed = 0
    tiles_skipped_tags = 0

    for tile_path in tile_files:
        tile_name = tile_path.stem

        # Load GeoTIFF metadata for GSD
        meta = load_geotiff_meta(tile_path)
        if meta is not None:
            gsd_x = meta["gsd_x"]
            gsd_y = meta["gsd_y"]
        elif fallback_gsd_x is not None:
            gsd_x, gsd_y = fallback_gsd_x, fallback_gsd_y
            tiles_skipped_tags += 1
        else:
            continue

        # Load masks
        b_path = masks_dir / "building_masks" / f"{tile_name}.png"
        w_path = masks_dir / "woodland_masks" / f"{tile_name}.png"
        if not b_path.exists() or not w_path.exists():
            continue

        building_mask = load_mask(b_path)
        woodland_mask = load_mask(w_path)

        # Load original image
        orig = np.array(Image.open(tile_path).convert("RGB"))

        # Resize masks if needed
        if building_mask.shape != orig.shape[:2]:
            building_mask = cv2.resize(building_mask, (orig.shape[1], orig.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
            woodland_mask = cv2.resize(woodland_mask, (orig.shape[1], orig.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)

        # Extract buildings
        buildings = extract_buildings(building_mask)
        if not buildings:
            continue

        # Compute risk per building
        for bld in buildings:
            bld_mask = np.zeros_like(building_mask)
            cv2.drawContours(bld_mask, [bld["contour"]], -1, 1, -1)

            min_dist, _, _, zone_veg, risk_score = compute_building_risk(
                bld_mask, woodland_mask, gsd_x, gsd_y
            )
            bld["min_distance_m"] = min_dist
            bld["zone_vegetation"] = zone_veg
            bld["risk_score"] = risk_score
            bld["area_m2"] = bld["area_px"] * gsd_x * gsd_y

            all_buildings.append({
                "tile": tile_name,
                "id": bld["id"],
                "centroid": bld["centroid"],
                "area_m2": round(bld["area_m2"], 1),
                "min_veg_distance_m": round(min_dist, 2),
                "risk_score": risk_score,
                "zone_1a_veg_density": round(zone_veg["zone_1a"]["veg_density"], 3),
                "zone_1b_veg_density": round(zone_veg["zone_1b"]["veg_density"], 3),
                "zone_2_veg_density": round(zone_veg["zone_2"]["veg_density"], 3),
                "zone_3_veg_density": round(zone_veg["zone_3"]["veg_density"], 3),
                "overlay_image": f"overlay/{tile_name}.png",
            })

        # Generate overlay
        overlay = create_risk_overlay(orig, building_mask, woodland_mask, buildings, gsd_x, gsd_y)
        Image.fromarray(overlay).save(output_dir / "overlay" / f"{tile_name}.png")

        tiles_processed += 1

    # Summary
    total = len(all_buildings)
    print(f"  Tiles with buildings: {tiles_processed}")
    if tiles_skipped_tags > 0:
        print(f"  Tiles with missing GeoTIFF tags (used fallback GSD): {tiles_skipped_tags}")
    print(f"  Total buildings: {total}")

    if total == 0:
        print("  No buildings found!")
        return

    risk_scores = [b["risk_score"] for b in all_buildings]
    high = sum(1 for r in risk_scores if r >= 7)
    moderate = sum(1 for r in risk_scores if 4 <= r < 7)
    low = sum(1 for r in risk_scores if 0 < r < 4)
    none = sum(1 for r in risk_scores if r == 0)

    distances = [b["min_veg_distance_m"] for b in all_buildings if b["min_veg_distance_m"] < 999]
    zone1a_count = sum(1 for d in distances if d < 1.5) if distances else 0
    zone1b_count = sum(1 for d in distances if 1.5 <= d < 10) if distances else 0
    zone2_count = sum(1 for d in distances if 10 <= d < 30) if distances else 0
    zone3_count = sum(1 for d in distances if d >= 30) if distances else 0

    # Community-level stats
    avg_z1a = np.mean([b["zone_1a_veg_density"] for b in all_buildings])
    avg_z1b = np.mean([b["zone_1b_veg_density"] for b in all_buildings])
    avg_z2 = np.mean([b["zone_2_veg_density"] for b in all_buildings])
    avg_z3 = np.mean([b["zone_3_veg_density"] for b in all_buildings])
    community_score = (avg_z1a * 5.0 + avg_z1b * 3.0 + avg_z2 * 1.25 + avg_z3 * 0.75)

    print(f"\n  --- COMMUNITY STATISTICS ---")
    print(f"  Risk Distribution:")
    print(f"    High (7-10):     {high} ({high/total*100:.1f}%)")
    print(f"    Moderate (4-6.9): {moderate} ({moderate/total*100:.1f}%)")
    print(f"    Low (0.1-3.9):   {low} ({low/total*100:.1f}%)")
    print(f"    None (0):        {none} ({none/total*100:.1f}%)")
    print(f"  Per-Building Risk:")
    print(f"    Mean: {np.mean(risk_scores):.2f}")
    print(f"    Median: {np.median(risk_scores):.2f}")
    print(f"    Std Dev: {np.std(risk_scores):.2f}")
    print(f"    Min: {np.min(risk_scores):.2f}, Max: {np.max(risk_scores):.2f}")
    if distances:
        print(f"  Vegetation Proximity:")
        print(f"    Mean min distance: {np.mean(distances):.1f}m")
        print(f"    Median min distance: {np.median(distances):.1f}m")
        print(f"    Buildings with veg <1.5m (Zone 1a):  {zone1a_count} ({zone1a_count/len(distances)*100:.1f}%)")
        print(f"    Buildings with veg 1.5-10m (Zone 1b): {zone1b_count} ({zone1b_count/len(distances)*100:.1f}%)")
        print(f"    Buildings with veg 10-30m (Zone 2):   {zone2_count} ({zone2_count/len(distances)*100:.1f}%)")
        print(f"    Buildings with veg >30m (Zone 3):     {zone3_count} ({zone3_count/len(distances)*100:.1f}%)")
    print(f"  Community Average Zone Densities:")
    print(f"    Zone 1a: {avg_z1a:.3f}")
    print(f"    Zone 1b: {avg_z1b:.3f}")
    print(f"    Zone 2:  {avg_z2:.3f}")
    print(f"    Zone 3:  {avg_z3:.3f}")
    print(f"  Community Risk Score: {community_score:.2f}")
    print(f"    = ({avg_z1a:.3f} * 5.0) + ({avg_z1b:.3f} * 3.0) + ({avg_z2:.3f} * 1.25) + ({avg_z3:.3f} * 0.75)")

    # Save CSV
    csv_path = output_dir / "building_risk_scores.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "tile", "id", "centroid", "area_m2", "min_veg_distance_m",
            "risk_score", "zone_1a_veg_density", "zone_1b_veg_density",
            "zone_2_veg_density", "zone_3_veg_density", "overlay_image"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for bld in all_buildings:
            writer.writerow(bld)
    print(f"\n  CSV: {csv_path}")

    # Save JSON summary with community stats
    json_path = output_dir / "risk_summary.json"
    summary = {
        "region": region_name,
        "type": "tile_level",
        "scoring": "density_only",
        "formula": "(zone_1a_density * 5.0) + (zone_1b_density * 3.0) + (zone_2_density * 1.25) + (zone_3_density * 0.75)",
        "weights": DENSITY_WEIGHTS,
        "total_buildings": total,
        "tiles_processed": tiles_processed,
        "risk_distribution": {"high": high, "moderate": moderate, "low": low, "none": none},
        "per_building_stats": {
            "mean_risk_score": round(float(np.mean(risk_scores)), 2),
            "median_risk_score": round(float(np.median(risk_scores)), 2),
            "std_risk_score": round(float(np.std(risk_scores)), 2),
            "min_risk_score": round(float(np.min(risk_scores)), 2),
            "max_risk_score": round(float(np.max(risk_scores)), 2),
        },
        "vegetation_proximity": {
            "mean_min_distance_m": round(float(np.mean(distances)), 2) if distances else None,
            "median_min_distance_m": round(float(np.median(distances)), 2) if distances else None,
            "buildings_veg_zone_1a": zone1a_count,
            "buildings_veg_zone_1b": zone1b_count,
            "buildings_veg_zone_2": zone2_count,
            "buildings_veg_zone_3": zone3_count,
        },
        "community_stats": {
            "avg_zone_1a_density": round(float(avg_z1a), 4),
            "avg_zone_1b_density": round(float(avg_z1b), 4),
            "avg_zone_2_density": round(float(avg_z2), 4),
            "avg_zone_3_density": round(float(avg_z3), 4),
            "community_risk_score": round(float(community_score), 2),
        },
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON: {json_path}")
    print(f"  Overlays: {output_dir / 'overlay'}")


# ============================================================
# PARCEL-LEVEL EVALUATION (with parcels)
# ============================================================

def stitch_masks_lowres(tile_metas, masks_dir, mask_type="building_masks"):
    """
    Stitch per-tile masks into a single large raster using GeoTIFF metadata.
    Returns (stitched_mask, extent_meta).
    """
    if not tile_metas:
        return None, None

    # Use GSD from first tile (all tiles in a region share the same source)
    ref_gsd_x = tile_metas[0]["gsd_x"]
    ref_gsd_y = tile_metas[0]["gsd_y"]

    # Compute overall extent
    min_x = min(m["bbox"][0] for m in tile_metas)
    min_y = min(m["bbox"][1] for m in tile_metas)
    max_x = max(m["bbox"][2] for m in tile_metas)
    max_y = max(m["bbox"][3] for m in tile_metas)

    canvas_w = int(np.ceil((max_x - min_x) / ref_gsd_x))
    canvas_h = int(np.ceil((max_y - min_y) / ref_gsd_y))

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    for meta in tile_metas:
        mask_path = masks_dir / mask_type / f"{meta['name']}.png"
        if not mask_path.exists():
            continue
        tile_mask = load_mask(mask_path)

        # Compute pixel position on canvas
        px = int(round((meta["origin_x"] - min_x) / ref_gsd_x))
        py = int(round((max_y - meta["origin_y"]) / ref_gsd_y))

        th, tw = tile_mask.shape
        # Clip to canvas bounds
        src_y0 = max(0, -py)
        src_x0 = max(0, -px)
        dst_y0 = max(0, py)
        dst_x0 = max(0, px)
        copy_h = min(th - src_y0, canvas_h - dst_y0)
        copy_w = min(tw - src_x0, canvas_w - dst_x0)

        if copy_h > 0 and copy_w > 0:
            canvas[dst_y0:dst_y0+copy_h, dst_x0:dst_x0+copy_w] = \
                tile_mask[src_y0:src_y0+copy_h, src_x0:src_x0+copy_w]

    extent_meta = {
        "origin_x": min_x,
        "origin_y": max_y,
        "gsd_x": ref_gsd_x,
        "gsd_y": ref_gsd_y,
        "width": canvas_w,
        "height": canvas_h,
    }
    return canvas, extent_meta


def load_image_region_from_tiles(tile_metas, row_start, row_end, col_start, col_end, extent_meta):
    """Load a region of the stitched image from individual tiles."""
    out_h = row_end - row_start
    out_w = col_end - col_start
    result = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    for meta in tile_metas:
        # Tile position on canvas
        px = int(round((meta["origin_x"] - extent_meta["origin_x"]) / extent_meta["gsd_x"]))
        py = int(round((extent_meta["origin_y"] - meta["origin_y"]) / extent_meta["gsd_y"]))

        tile_x_end = px + meta["width"]
        tile_y_end = py + meta["height"]

        # Check overlap with requested region
        if px >= col_end or tile_x_end <= col_start or py >= row_end or tile_y_end <= row_start:
            continue

        # Load tile image
        img = np.array(Image.open(meta["path"]).convert("RGB"))

        # Compute overlap
        src_y0 = max(0, row_start - py)
        src_x0 = max(0, col_start - px)
        src_y1 = min(meta["height"], row_end - py)
        src_x1 = min(meta["width"], col_end - px)

        dst_y0 = max(0, py - row_start)
        dst_x0 = max(0, px - col_start)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x1 = dst_x0 + (src_x1 - src_x0)

        result[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]

    return result


def run_parcel_eval(region_name, config, target_pids=None):
    """Run parcel-level FireSmart evaluation for a region."""
    parcels_path = BASE_DIR / config["parcels"]
    if not parcels_path.exists():
        print(f"\n  SKIPPING parcel eval for {region_name}: {parcels_path} not found")
        print(f"  Run download script for {region_name} parcels first.")
        return

    print(f"\n{'='*60}")
    print(f"PARCEL-LEVEL EVAL: {region_name.upper()}")
    print(f"{'='*60}")

    import shapefile as shp

    imagery_dir = BASE_DIR / config["imagery"]
    masks_dir = BASE_DIR / config["masks"]
    output_dir = BASE_DIR / "eval" / f"{region_name}_parcels"
    (output_dir / "overlay").mkdir(parents=True, exist_ok=True)

    # Step 1: Load all tile metadata (skip tiles without GeoTIFF tags)
    print("  Loading tile metadata...")
    tile_metas = []
    skipped_meta = 0
    for tif_path in sorted(imagery_dir.glob("*.tif")):
        meta = load_geotiff_meta(tif_path)
        if meta is not None:
            tile_metas.append(meta)
        else:
            skipped_meta += 1
    if skipped_meta > 0:
        print(f"    Skipped {skipped_meta} tiles with missing GeoTIFF tags")
    print(f"    {len(tile_metas)} tiles")

    # Step 2: Stitch masks
    print("  Stitching building masks...", end=" ", flush=True)
    b_stitched, extent = stitch_masks_lowres(tile_metas, masks_dir, "building_masks")
    print(f"{b_stitched.shape[1]}x{b_stitched.shape[0]} px, {b_stitched.sum()} building px")

    print("  Stitching woodland masks...", end=" ", flush=True)
    w_stitched, _ = stitch_masks_lowres(tile_metas, masks_dir, "woodland_masks")
    print(f"{w_stitched.sum()} woodland px")

    gsd_x = extent["gsd_x"]
    gsd_y = extent["gsd_y"]

    # Step 3: Load parcels
    print("  Loading parcels...")
    sf = shp.Reader(str(parcels_path))
    fields = [f[0] for f in sf.fields[1:]]
    print(f"    {len(sf)} parcels, fields: {fields[:5]}...")

    # Step 4: Process each parcel
    print("  Processing parcels...")
    all_parcel_results = []
    parcels_processed = 0
    parcels_skipped = 0
    parcels_no_buildings = 0

    for idx in range(len(sf)):
        shape = sf.shape(idx)
        rec = sf.record(idx)
        rec_dict = dict(zip(fields, rec))

        pid = rec_dict.get("PID", str(idx))
        if target_pids and pid not in target_pids:
            continue
        pid_fmt = rec_dict.get("PID_FORMAT", pid)
        owner_type = rec_dict.get("OWNER_TYPE", "")
        parcel_area_m2 = rec_dict.get("FEATURE_AR", 0)

        # Parcel bounding box in UTM
        p_bbox = shape.bbox  # (min_x, min_y, max_x, max_y)

        # Add buffer for FireSmart zones
        buffered = (
            p_bbox[0] - BUFFER_M,
            p_bbox[1] - BUFFER_M,
            p_bbox[2] + BUFFER_M,
            p_bbox[3] + BUFFER_M,
        )

        # Convert to pixel coords in stitched raster
        px_min, py_min = utm_to_pixel(buffered[0], buffered[3], extent["origin_x"], extent["origin_y"], gsd_x, gsd_y)
        px_max, py_max = utm_to_pixel(buffered[2], buffered[1], extent["origin_x"], extent["origin_y"], gsd_x, gsd_y)

        px_min = max(0, int(px_min))
        py_min = max(0, int(py_min))
        px_max = min(extent["width"], int(px_max))
        py_max = min(extent["height"], int(py_max))

        crop_w = px_max - px_min
        crop_h = py_max - py_min
        if crop_w < 10 or crop_h < 10:
            parcels_skipped += 1
            continue

        # Crop masks
        b_crop = b_stitched[py_min:py_max, px_min:px_max].copy()
        w_crop = w_stitched[py_min:py_max, px_min:px_max].copy()

        # Create parcel boundary mask
        pixel_points = []
        for x, y in shape.points:
            px, py = utm_to_pixel(x, y, extent["origin_x"], extent["origin_y"], gsd_x, gsd_y)
            pixel_points.append((px, py))

        adjusted_pts = [(int(px - px_min), int(py - py_min)) for px, py in pixel_points]

        parcel_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        pts_arr = np.array(adjusted_pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(parcel_mask, [pts_arr], 1)

        # Find buildings overlapping this parcel
        buildings = extract_buildings(b_crop)
        parcel_buildings = []
        for bld in buildings:
            bld_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
            cv2.drawContours(bld_mask, [bld["contour"]], -1, 1, -1)
            if (bld_mask & parcel_mask).sum() > 0:
                parcel_buildings.append(bld)

        if not parcel_buildings:
            parcels_no_buildings += 1
            all_parcel_results.append({
                "pid": pid, "pid_format": pid_fmt, "owner_type": owner_type,
                "parcel_area_m2": round(parcel_area_m2, 1) if parcel_area_m2 else 0,
                "num_buildings": 0, "max_risk_score": None, "mean_risk_score": None,
                "min_veg_distance_m": None, "min_veg_dist_on_parcel": None,
                "min_veg_dist_off_parcel": None,
                "zone_1a_veg_density": None, "zone_1b_veg_density": None,
                "zone_2_veg_density": None, "zone_3_veg_density": None,
                "zone_1a_on_parcel": None, "zone_1a_off_parcel": None,
                "zone_1b_on_parcel": None, "zone_1b_off_parcel": None,
                "zone_2_on_parcel": None, "zone_2_off_parcel": None,
                "overlay_image": "",
            })
            continue

        # Compute risk for each building
        building_results = []
        for bld in parcel_buildings:
            bld_mask = np.zeros_like(b_crop)
            cv2.drawContours(bld_mask, [bld["contour"]], -1, 1, -1)

            min_dist, min_dist_on, min_dist_off, zone_veg, risk_score = \
                compute_building_risk(bld_mask, w_crop, gsd_x, gsd_y, parcel_mask)
            bld["min_distance_m"] = min_dist
            bld["min_dist_on_parcel"] = min_dist_on
            bld["min_dist_off_parcel"] = min_dist_off
            bld["zone_vegetation"] = zone_veg
            bld["risk_score"] = risk_score
            bld["area_m2"] = bld["area_px"] * gsd_x * gsd_y

            building_results.append({
                "risk_score": risk_score,
                "min_distance_m": min_dist,
                "min_dist_on_parcel": min_dist_on,
                "min_dist_off_parcel": min_dist_off,
                "zone_1a": zone_veg["zone_1a"]["veg_density"],
                "zone_1b": zone_veg["zone_1b"]["veg_density"],
                "zone_2": zone_veg["zone_2"]["veg_density"],
                "zone_3": zone_veg["zone_3"]["veg_density"],
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
        orig_crop = load_image_region_from_tiles(tile_metas, py_min, py_max, px_min, px_max, extent)
        overlay = create_parcel_overlay(
            orig_crop, b_crop, w_crop, parcel_mask,
            parcel_buildings, gsd_x, gsd_y, adjusted_pts
        )
        Image.fromarray(overlay).save(output_dir / "overlay" / overlay_name)

        all_parcel_results.append({
            "pid": pid,
            "pid_format": pid_fmt,
            "owner_type": owner_type,
            "parcel_area_m2": round(parcel_area_m2, 1) if parcel_area_m2 else 0,
            "num_buildings": len(parcel_buildings),
            "max_risk_score": max(risk_scores),
            "mean_risk_score": round(float(np.mean(risk_scores)), 1),
            "min_veg_distance_m": round(min(min_dists), 2) if min_dists else 999.0,
            "min_veg_dist_on_parcel": round(min(min_dists_on), 2) if min_dists_on else 999.0,
            "min_veg_dist_off_parcel": round(min(min_dists_off), 2) if min_dists_off else 999.0,
            "zone_1a_veg_density": round(float(np.mean([b["zone_1a"] for b in building_results])), 3),
            "zone_1b_veg_density": round(float(np.mean([b["zone_1b"] for b in building_results])), 3),
            "zone_2_veg_density": round(float(np.mean([b["zone_2"] for b in building_results])), 3),
            "zone_3_veg_density": round(float(np.mean([b["zone_3"] for b in building_results])), 3),
            "zone_1a_on_parcel": round(float(np.mean([b["zone_1a_on"] for b in building_results])), 3),
            "zone_1a_off_parcel": round(float(np.mean([b["zone_1a_off"] for b in building_results])), 3),
            "zone_1b_on_parcel": round(float(np.mean([b["zone_1b_on"] for b in building_results])), 3),
            "zone_1b_off_parcel": round(float(np.mean([b["zone_1b_off"] for b in building_results])), 3),
            "zone_2_on_parcel": round(float(np.mean([b["zone_2_on"] for b in building_results])), 3),
            "zone_2_off_parcel": round(float(np.mean([b["zone_2_off"] for b in building_results])), 3),
            "overlay_image": f"overlay/{overlay_name}",
        })
        parcels_processed += 1

        if parcels_processed % 50 == 0:
            print(f"    Processed {parcels_processed} parcels with buildings...")

    # Summary
    print(f"\n  Parcels with buildings: {parcels_processed}")
    print(f"  Parcels without buildings: {parcels_no_buildings}")
    print(f"  Parcels outside imagery: {parcels_skipped}")

    parcels_with_risk = [p for p in all_parcel_results if p["max_risk_score"] is not None]

    high = moderate = low = none = 0
    if parcels_with_risk:
        scores = [p["max_risk_score"] for p in parcels_with_risk]
        high = sum(1 for s in scores if s >= 7)
        moderate = sum(1 for s in scores if 4 <= s < 7)
        low = sum(1 for s in scores if 0 < s < 4)
        none = sum(1 for s in scores if s == 0)

        mean_scores = [p["mean_risk_score"] for p in parcels_with_risk if p["mean_risk_score"] is not None]
        min_dists_all = [p["min_veg_distance_m"] for p in parcels_with_risk
                         if p["min_veg_distance_m"] is not None and p["min_veg_distance_m"] < 999]

        # Community-level averages from parcels
        z1a_densities = [p["zone_1a_veg_density"] for p in parcels_with_risk if p["zone_1a_veg_density"] is not None]
        z1b_densities = [p["zone_1b_veg_density"] for p in parcels_with_risk if p["zone_1b_veg_density"] is not None]
        z2_densities = [p["zone_2_veg_density"] for p in parcels_with_risk if p["zone_2_veg_density"] is not None]
        z3_densities = [p["zone_3_veg_density"] for p in parcels_with_risk if p["zone_3_veg_density"] is not None]

        avg_z1a = np.mean(z1a_densities) if z1a_densities else 0
        avg_z1b = np.mean(z1b_densities) if z1b_densities else 0
        avg_z2 = np.mean(z2_densities) if z2_densities else 0
        avg_z3 = np.mean(z3_densities) if z3_densities else 0
        community_score = avg_z1a * 5.0 + avg_z1b * 3.0 + avg_z2 * 1.25 + avg_z3 * 0.75

        print(f"\n  --- COMMUNITY STATISTICS (PARCEL-LEVEL) ---")
        print(f"  Risk Distribution (max per property):")
        print(f"    High (7-10):     {high} ({high/len(parcels_with_risk)*100:.1f}%)")
        print(f"    Moderate (4-6.9): {moderate} ({moderate/len(parcels_with_risk)*100:.1f}%)")
        print(f"    Low (0.1-3.9):   {low} ({low/len(parcels_with_risk)*100:.1f}%)")
        print(f"    None (0):        {none} ({none/len(parcels_with_risk)*100:.1f}%)")
        print(f"  Per-Property Risk:")
        print(f"    Mean max risk: {np.mean(scores):.2f}")
        print(f"    Median max risk: {np.median(scores):.2f}")
        if mean_scores:
            print(f"    Mean avg risk: {np.mean(mean_scores):.2f}")
        if min_dists_all:
            print(f"  Vegetation Proximity:")
            print(f"    Mean min distance: {np.mean(min_dists_all):.1f}m")
            print(f"    Median min distance: {np.median(min_dists_all):.1f}m")
        print(f"  Community Average Zone Densities:")
        print(f"    Zone 1a: {avg_z1a:.3f}")
        print(f"    Zone 1b: {avg_z1b:.3f}")
        print(f"    Zone 2:  {avg_z2:.3f}")
        print(f"    Zone 3:  {avg_z3:.3f}")
        print(f"  Community Risk Score: {community_score:.2f}")
        print(f"    = ({avg_z1a:.3f} * 5.0) + ({avg_z1b:.3f} * 3.0) + ({avg_z2:.3f} * 1.25) + ({avg_z3:.3f} * 0.75)")

    # Save CSV
    csv_path = output_dir / "parcel_risk_scores.csv"
    fieldnames = [
        "pid", "pid_format", "owner_type", "parcel_area_m2", "num_buildings",
        "max_risk_score", "mean_risk_score", "min_veg_distance_m",
        "min_veg_dist_on_parcel", "min_veg_dist_off_parcel",
        "zone_1a_veg_density", "zone_1b_veg_density", "zone_2_veg_density", "zone_3_veg_density",
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
    print(f"\n  CSV: {csv_path}")

    # Save JSON summary with community stats
    json_path = output_dir / "parcel_risk_summary.json"
    summary = {
        "region": region_name,
        "type": "parcel_level",
        "scoring": "density_only",
        "formula": "(zone_1a_density * 5.0) + (zone_1b_density * 3.0) + (zone_2_density * 1.25) + (zone_3_density * 0.75)",
        "weights": DENSITY_WEIGHTS,
        "total_parcels": len(sf),
        "parcels_with_buildings": parcels_processed,
        "parcels_no_buildings": parcels_no_buildings,
        "parcels_outside_imagery": parcels_skipped,
        "risk_distribution": {"high": high, "moderate": moderate, "low": low, "none": none},
    }
    if parcels_with_risk:
        summary["per_property_stats"] = {
            "mean_max_risk": round(float(np.mean(scores)), 2),
            "median_max_risk": round(float(np.median(scores)), 2),
        }
        summary["community_stats"] = {
            "avg_zone_1a_density": round(float(avg_z1a), 4),
            "avg_zone_1b_density": round(float(avg_z1b), 4),
            "avg_zone_2_density": round(float(avg_z2), 4),
            "avg_zone_3_density": round(float(avg_z3), 4),
            "community_risk_score": round(float(community_score), 2),
        }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON: {json_path}")
    print(f"  Overlays: {output_dir / 'overlay'}")


# ============================================================
# COMBINED NEIGHBORHOOD EVALUATION
# ============================================================

def run_combined_eval(region_name, config):
    """
    Stitch all tiles into a full-community raster, compute risk for every
    building, generate a full neighborhood overlay, and produce community stats.
    Output: eval/{region}_combined/
    """
    print(f"\n{'='*60}")
    print(f"COMBINED NEIGHBORHOOD EVAL: {region_name.upper()}")
    print(f"{'='*60}")

    imagery_dir = BASE_DIR / config["imagery"]
    masks_dir = BASE_DIR / config["masks"]
    output_dir = BASE_DIR / "eval" / f"{region_name}_combined"
    (output_dir / "overlay").mkdir(parents=True, exist_ok=True)

    # Step 1: Load tile metadata
    print("  Loading tile metadata...")
    tile_metas = []
    for tif_path in sorted(imagery_dir.glob("*.tif")):
        meta = load_geotiff_meta(tif_path)
        if meta is not None:
            tile_metas.append(meta)
    print(f"    {len(tile_metas)} tiles with valid metadata")

    if not tile_metas:
        print("  No valid tiles found!")
        return

    # Step 2: Stitch masks
    print("  Stitching building masks...", end=" ", flush=True)
    b_stitched, extent = stitch_masks_lowres(tile_metas, masks_dir, "building_masks")
    print(f"{b_stitched.shape[1]}x{b_stitched.shape[0]} px")

    print("  Stitching woodland masks...", end=" ", flush=True)
    w_stitched, _ = stitch_masks_lowres(tile_metas, masks_dir, "woodland_masks")
    print("done")

    gsd_x = extent["gsd_x"]
    gsd_y = extent["gsd_y"]
    full_h, full_w = b_stitched.shape

    # Step 3: Extract all buildings from stitched mask
    print("  Extracting buildings from stitched mask...")
    all_buildings_data = extract_buildings(b_stitched, min_area=100)
    print(f"    Found {len(all_buildings_data)} buildings")

    if not all_buildings_data:
        print("  No buildings found!")
        return

    # Step 4: Compute risk per building
    print("  Computing risk per building...")
    all_building_results = []
    for i, bld in enumerate(all_buildings_data):
        # Create individual building mask using bounding box crop for efficiency
        x, y, bw, bh = cv2.boundingRect(bld["contour"])
        # Add buffer for zone computation (35m / gsd)
        buf_px_x = int(35.0 / gsd_x) + 5
        buf_px_y = int(35.0 / gsd_y) + 5

        crop_y0 = max(0, y - buf_px_y)
        crop_x0 = max(0, x - buf_px_x)
        crop_y1 = min(full_h, y + bh + buf_px_y)
        crop_x1 = min(full_w, x + bw + buf_px_x)

        b_crop = b_stitched[crop_y0:crop_y1, crop_x0:crop_x1]
        w_crop = w_stitched[crop_y0:crop_y1, crop_x0:crop_x1]

        # Shift contour to crop coordinates
        shifted_cnt = bld["contour"].copy()
        shifted_cnt[:, :, 0] -= crop_x0
        shifted_cnt[:, :, 1] -= crop_y0

        bld_mask = np.zeros_like(b_crop)
        cv2.drawContours(bld_mask, [shifted_cnt], -1, 1, -1)

        min_dist, _, _, zone_veg, risk_score = compute_building_risk(
            bld_mask, w_crop, gsd_x, gsd_y
        )

        bld["min_distance_m"] = min_dist
        bld["zone_vegetation"] = zone_veg
        bld["risk_score"] = risk_score
        bld["area_m2"] = bld["area_px"] * gsd_x * gsd_y

        all_building_results.append({
            "id": bld["id"],
            "centroid": bld["centroid"],
            "area_m2": round(bld["area_m2"], 1),
            "min_veg_distance_m": round(min_dist, 2),
            "risk_score": risk_score,
            "zone_1a_veg_density": round(zone_veg["zone_1a"]["veg_density"], 3),
            "zone_1b_veg_density": round(zone_veg["zone_1b"]["veg_density"], 3),
            "zone_2_veg_density": round(zone_veg["zone_2"]["veg_density"], 3),
            "zone_3_veg_density": round(zone_veg["zone_3"]["veg_density"], 3),
        })

        if (i + 1) % 200 == 0:
            print(f"    Processed {i + 1}/{len(all_buildings_data)} buildings...")

    print(f"    All {len(all_building_results)} buildings scored")

    # Step 5: Generate full neighborhood overlay (chunked to avoid OOM)
    print("  Generating neighborhood overlay...")

    # For very large regions, generate overlay in chunks
    CHUNK_H = 2048
    CHUNK_W = 2048
    n_chunks_y = (full_h + CHUNK_H - 1) // CHUNK_H
    n_chunks_x = (full_w + CHUNK_W - 1) // CHUNK_W

    # Create overlay output as chunked PNGs
    overlay_full = np.zeros((full_h, full_w, 3), dtype=np.uint8)

    # Stitch original imagery
    print("    Stitching original imagery for overlay...")
    for meta in tile_metas:
        px = int(round((meta["origin_x"] - extent["origin_x"]) / gsd_x))
        py = int(round((extent["origin_y"] - meta["origin_y"]) / gsd_y))
        img = np.array(Image.open(meta["path"]).convert("RGB"))
        th, tw = img.shape[:2]
        dst_y0, dst_x0 = max(0, py), max(0, px)
        src_y0, src_x0 = max(0, -py), max(0, -px)
        copy_h = min(th - src_y0, full_h - dst_y0)
        copy_w = min(tw - src_x0, full_w - dst_x0)
        if copy_h > 0 and copy_w > 0:
            overlay_full[dst_y0:dst_y0+copy_h, dst_x0:dst_x0+copy_w] = \
                img[src_y0:src_y0+copy_h, src_x0:src_x0+copy_w]

    # Apply zone overlay
    print("    Applying FireSmart zone overlay...")
    building_border = cv2.dilate(b_stitched, np.ones((3, 3), np.uint8)) - b_stitched
    if building_border.sum() == 0:
        building_border = b_stitched
    dist_m = ndimage.distance_transform_edt(1 - building_border, sampling=(gsd_y, gsd_x))

    overlay_f = overlay_full.astype(np.float32)

    for zone_name in ["zone_3", "zone_2", "zone_1b", "zone_1a"]:
        zone_def = ZONES[zone_name]
        zone_ring = (dist_m >= zone_def["min_m"]) & (dist_m < zone_def["max_m"])
        zone_ring = zone_ring & (~(b_stitched > 0))
        if zone_ring.any():
            color = np.array(zone_def["color"], dtype=np.float32)
            overlay_f[zone_ring] = overlay_f[zone_ring] * 0.7 + color * 0.3

    veg_mask = w_stitched > 0
    overlay_f[veg_mask] = overlay_f[veg_mask] * 0.6 + np.array([0, 180, 0], dtype=np.float32) * 0.4

    bld_mask = b_stitched > 0
    overlay_f[bld_mask] = overlay_f[bld_mask] * 0.5 + np.array([0, 80, 255], dtype=np.float32) * 0.5

    overlay_full = np.clip(overlay_f, 0, 255).astype(np.uint8)

    # Draw risk scores on buildings
    for bld in all_buildings_data:
        cx, cy = bld["centroid"]
        risk = bld.get("risk_score", 0)
        if risk >= 7:
            txt_color = (255, 0, 0)
        elif risk >= 4:
            txt_color = (255, 165, 0)
        else:
            txt_color = (0, 200, 0)
        cv2.putText(overlay_full, f"{risk}", (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(overlay_full, f"{risk}", (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1)

    # Save full neighborhood overlay
    print("    Saving full neighborhood overlay...")
    Image.fromarray(overlay_full).save(output_dir / "overlay" / f"{region_name}_neighborhood.png")

    # Also save a downscaled version for quick viewing
    scale = min(1.0, 4096.0 / max(full_h, full_w))
    if scale < 1.0:
        small_h, small_w = int(full_h * scale), int(full_w * scale)
        overlay_small = cv2.resize(overlay_full, (small_w, small_h), interpolation=cv2.INTER_AREA)
        Image.fromarray(overlay_small).save(output_dir / "overlay" / f"{region_name}_neighborhood_preview.png")
        print(f"    Preview: {small_w}x{small_h}")

    # Free large arrays
    del overlay_full, overlay_f, dist_m

    # Save legend
    legend = create_legend()
    Image.fromarray(legend[:, :, ::-1]).save(output_dir / "legend.png")

    # Step 6: Community statistics
    total = len(all_building_results)
    risk_scores = [b["risk_score"] for b in all_building_results]
    high = sum(1 for r in risk_scores if r >= 7)
    moderate = sum(1 for r in risk_scores if 4 <= r < 7)
    low = sum(1 for r in risk_scores if 0 < r < 4)
    none = sum(1 for r in risk_scores if r == 0)

    distances = [b["min_veg_distance_m"] for b in all_building_results if b["min_veg_distance_m"] < 999]
    zone1a_count = sum(1 for d in distances if d < 1.5) if distances else 0
    zone1b_count = sum(1 for d in distances if 1.5 <= d < 10) if distances else 0
    zone2_count = sum(1 for d in distances if 10 <= d < 30) if distances else 0
    zone3_count = sum(1 for d in distances if d >= 30) if distances else 0

    avg_z1a = float(np.mean([b["zone_1a_veg_density"] for b in all_building_results]))
    avg_z1b = float(np.mean([b["zone_1b_veg_density"] for b in all_building_results]))
    avg_z2 = float(np.mean([b["zone_2_veg_density"] for b in all_building_results]))
    avg_z3 = float(np.mean([b["zone_3_veg_density"] for b in all_building_results]))
    community_score = avg_z1a * 5.0 + avg_z1b * 3.0 + avg_z2 * 1.25 + avg_z3 * 0.75

    print(f"\n  --- COMBINED NEIGHBORHOOD STATISTICS ---")
    print(f"  Total buildings: {total}")
    print(f"  GSD: {gsd_x:.4f}m x {gsd_y:.4f}m")
    print(f"  Raster size: {full_w}x{full_h} px")
    print(f"\n  Risk Distribution:")
    print(f"    High (7-10):     {high} ({high/total*100:.1f}%)")
    print(f"    Moderate (4-6.9): {moderate} ({moderate/total*100:.1f}%)")
    print(f"    Low (0.1-3.9):   {low} ({low/total*100:.1f}%)")
    print(f"    None (0):        {none} ({none/total*100:.1f}%)")
    print(f"  Per-Building Risk:")
    print(f"    Mean: {np.mean(risk_scores):.2f}")
    print(f"    Median: {np.median(risk_scores):.2f}")
    print(f"    Std Dev: {np.std(risk_scores):.2f}")
    print(f"    Min: {np.min(risk_scores):.2f}, Max: {np.max(risk_scores):.2f}")
    if distances:
        print(f"  Vegetation Proximity:")
        print(f"    Mean: {np.mean(distances):.1f}m, Median: {np.median(distances):.1f}m")
        print(f"    Veg <1.5m (Zone 1a):  {zone1a_count} ({zone1a_count/len(distances)*100:.1f}%)")
        print(f"    Veg 1.5-10m (Zone 1b): {zone1b_count} ({zone1b_count/len(distances)*100:.1f}%)")
        print(f"    Veg 10-30m (Zone 2):   {zone2_count} ({zone2_count/len(distances)*100:.1f}%)")
        print(f"    Veg >30m (Zone 3):     {zone3_count} ({zone3_count/len(distances)*100:.1f}%)")
    print(f"  Community Average Zone Densities:")
    print(f"    Zone 1a: {avg_z1a:.3f}")
    print(f"    Zone 1b: {avg_z1b:.3f}")
    print(f"    Zone 2:  {avg_z2:.3f}")
    print(f"    Zone 3:  {avg_z3:.3f}")
    print(f"  Community Risk Score: {community_score:.2f}")
    print(f"    = ({avg_z1a:.3f} * 5.0) + ({avg_z1b:.3f} * 3.0) + ({avg_z2:.3f} * 1.25) + ({avg_z3:.3f} * 0.75)")

    # Save CSV
    csv_path = output_dir / "building_risk_scores.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "id", "centroid", "area_m2", "min_veg_distance_m",
            "risk_score", "zone_1a_veg_density", "zone_1b_veg_density",
            "zone_2_veg_density", "zone_3_veg_density"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for bld in all_building_results:
            writer.writerow(bld)
    print(f"\n  CSV: {csv_path}")

    # Save JSON summary
    json_path = output_dir / "risk_summary.json"
    summary = {
        "region": region_name,
        "type": "combined_neighborhood",
        "scoring": "density_only",
        "formula": "(zone_1a_density * 5.0) + (zone_1b_density * 3.0) + (zone_2_density * 1.25) + (zone_3_density * 0.75)",
        "weights": DENSITY_WEIGHTS,
        "raster_size": f"{full_w}x{full_h}",
        "gsd": {"x": round(gsd_x, 4), "y": round(gsd_y, 4)},
        "total_buildings": total,
        "risk_distribution": {"high": high, "moderate": moderate, "low": low, "none": none},
        "per_building_stats": {
            "mean_risk_score": round(float(np.mean(risk_scores)), 2),
            "median_risk_score": round(float(np.median(risk_scores)), 2),
            "std_risk_score": round(float(np.std(risk_scores)), 2),
            "min_risk_score": round(float(np.min(risk_scores)), 2),
            "max_risk_score": round(float(np.max(risk_scores)), 2),
        },
        "vegetation_proximity": {
            "mean_min_distance_m": round(float(np.mean(distances)), 2) if distances else None,
            "median_min_distance_m": round(float(np.median(distances)), 2) if distances else None,
            "buildings_veg_zone_1a": zone1a_count,
            "buildings_veg_zone_1b": zone1b_count,
            "buildings_veg_zone_2": zone2_count,
            "buildings_veg_zone_3": zone3_count,
        },
        "community_stats": {
            "avg_zone_1a_density": round(avg_z1a, 4),
            "avg_zone_1b_density": round(avg_z1b, 4),
            "avg_zone_2_density": round(avg_z2, 4),
            "avg_zone_3_density": round(avg_z3, 4),
            "community_risk_score": round(community_score, 2),
        },
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON: {json_path}")
    print(f"  Overlays: {output_dir / 'overlay'}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Low-Res FireSmart Evaluation")
    parser.add_argument("--regions", nargs="+", default=list(REGIONS.keys()),
                        help="Regions to evaluate (default: all)")
    parser.add_argument("--skip-parcels", action="store_true",
                        help="Skip parcel-level evaluation")
    parser.add_argument("--skip-tiles", action="store_true",
                        help="Skip tile-level evaluation")
    parser.add_argument("--skip-combined", action="store_true",
                        help="Skip combined neighborhood evaluation")
    parser.add_argument("--pids", type=str, default="",
                        help="Comma-separated PIDs to filter parcel eval (empty=all)")
    args = parser.parse_args()
    target_pids = set(args.pids.split(",")) if args.pids else None

    print("=" * 60)
    print("LOW-RESOLUTION FIRESMART EVALUATION")
    print("=" * 60)
    print(f"Regions: {args.regions}")
    print(f"Scoring: density-only (NO distance component)")
    print(f"Formula: (z1a_density * 5.0) + (z1b_density * 3.0) + (z2_density * 1.25) + (z3_density * 0.75)")
    print(f"Output: {BASE_DIR / 'eval'}")
    print(f"Modes: tiles={'ON' if not args.skip_tiles else 'OFF'}, "
          f"parcels={'ON' if not args.skip_parcels else 'OFF'}, "
          f"combined={'ON' if not args.skip_combined else 'OFF'}")

    for region_name in args.regions:
        if region_name not in REGIONS:
            print(f"\nUnknown region: {region_name}")
            continue

        config = REGIONS[region_name]

        if not args.skip_tiles:
            run_tile_eval(region_name, config)

        if not args.skip_parcels:
            run_parcel_eval(region_name, config, target_pids=target_pids)

        if not args.skip_combined:
            run_combined_eval(region_name, config)

    print(f"\n{'='*60}")
    print("ALL EVALUATIONS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
