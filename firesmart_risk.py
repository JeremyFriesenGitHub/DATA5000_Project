"""
FireSmart Risk Assessment Pipeline
====================================
Takes building + woodland segmentation masks and computes wildfire risk
based on vegetation proximity to structures.

FireSmart Zones (from building edge):
  Zone 1a:  0 - 1.5m   → Critical (immediate ignition zone)
  Zone 1b:  1.5 - 10m  → High (radiant heat / ember zone)
  Zone 2:   10 - 30m   → Moderate (fuel management zone)
  Zone 3:   30m+       → Low (extended planning zone)

Risk scoring per building:
  - Measures minimum distance from building edge to nearest vegetation
  - Computes vegetation density within each zone
  - Assigns overall risk score (1-10)

Usage:
    python firesmart_risk.py \
        --building_masks ./cumberland_final/building_masks \
        --woodland_masks ./cumberland_final/woodland_masks \
        --tiles_dir ~/COMP4900/filtered/has_structures \
        --output_dir ./firesmart_results \
        --gsd 0.06

    # Use combined masks instead:
    python firesmart_risk.py \
        --combined_masks ./cumberland_final/masks \
        --tiles_dir ~/COMP4900/filtered/has_structures \
        --output_dir ./firesmart_results \
        --gsd 0.06

Requirements:
    pip install numpy opencv-python Pillow scipy matplotlib
"""

import argparse
import csv
import json
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from scipy import ndimage


# ============================================================
# FIRESMART ZONE DEFINITIONS (meters from building edge)
# ============================================================

ZONES = {
    "zone_1a": {"min_m": 0.0,  "max_m": 1.5,  "color": [255, 0, 0],     "label": "Critical (0-1.5m)"},
    "zone_1b": {"min_m": 1.5,  "max_m": 10.0,  "color": [255, 128, 0],   "label": "High (1.5-10m)"},
    "zone_2":  {"min_m": 10.0, "max_m": 30.0,  "color": [255, 255, 0],   "label": "Moderate (10-30m)"},
    "zone_3":  {"min_m": 30.0, "max_m": 999.0, "color": [0, 200, 0],     "label": "Low (30m+)"},
}


def meters_to_pixels(meters, gsd):
    """Convert meters to pixels given ground sample distance."""
    return int(meters / gsd)


# ============================================================
# BUILDING EXTRACTION
# ============================================================

def extract_buildings(building_mask):
    """
    Find individual buildings as connected components.
    Returns list of dicts with contour, centroid, area, bounding box.
    """
    contours, _ = cv2.findContours(
        building_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    buildings = []
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < 100:  # Skip tiny fragments
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(cnt)

        buildings.append({
            "id": i,
            "contour": cnt,
            "centroid": (cx, cy),
            "area_px": area,
            "bbox": (x, y, w, h),
        })

    return buildings


# ============================================================
# RISK COMPUTATION
# ============================================================

def compute_building_risk(building_mask_single, woodland_mask, gsd):
    """
    Compute risk for a single building.

    Returns:
        min_distance_m: minimum distance from building edge to vegetation
        zone_vegetation: dict of vegetation pixel counts per zone
        risk_score: 1-10 risk score
    """
    h, w = building_mask_single.shape

    # Distance from every pixel to nearest building edge
    # Invert building mask: distance transform gives distance from non-building pixels to building
    building_border = cv2.dilate(building_mask_single, np.ones((3, 3), np.uint8)) - building_mask_single
    if building_border.sum() == 0:
        building_border = building_mask_single

    # Distance from each pixel to nearest building edge pixel
    dist_from_building = ndimage.distance_transform_edt(1 - building_border)

    # Find vegetation pixels
    veg_pixels = woodland_mask > 0

    # Minimum distance from building to any vegetation
    if veg_pixels.any():
        veg_distances = dist_from_building[veg_pixels]
        min_distance_px = veg_distances.min()
        min_distance_m = min_distance_px * gsd
    else:
        min_distance_m = 999.0

    # Count vegetation pixels in each zone
    zone_vegetation = {}
    total_zone_area = {}
    for zone_name, zone_def in ZONES.items():
        min_px = meters_to_pixels(zone_def["min_m"], gsd)
        max_px = meters_to_pixels(zone_def["max_m"], gsd)

        # Pixels in this zone (ring around building)
        zone_ring = (dist_from_building >= min_px) & (dist_from_building < max_px)
        # Exclude building interior
        zone_ring = zone_ring & (~(building_mask_single > 0))

        zone_total = zone_ring.sum()
        zone_veg = (zone_ring & veg_pixels).sum()

        zone_vegetation[zone_name] = {
            "veg_pixels": int(zone_veg),
            "total_pixels": int(zone_total),
            "veg_density": float(zone_veg / zone_total) if zone_total > 0 else 0.0,
        }

    # Risk score (1-10)
    risk_score = compute_risk_score(min_distance_m, zone_vegetation)

    return min_distance_m, zone_vegetation, risk_score


def compute_risk_score(min_distance_m, zone_vegetation):
    """
    Compute overall risk score (1-10) based on:
    - Minimum vegetation distance (most important)
    - Vegetation density in each zone (weighted by proximity)
    """
    # Distance component (0-5 points): closer vegetation = higher risk
    if min_distance_m < 1.5:
        dist_score = 5.0
    elif min_distance_m < 10.0:
        dist_score = 4.0 - (min_distance_m - 1.5) / 8.5 * 2.0  # 4.0 → 2.0
    elif min_distance_m < 30.0:
        dist_score = 2.0 - (min_distance_m - 10.0) / 20.0 * 1.5  # 2.0 → 0.5
    else:
        dist_score = 0.5

    # Density component (0-5 points): more vegetation nearby = higher risk
    density_score = 0.0
    weights = {"zone_1a": 2.5, "zone_1b": 1.5, "zone_2": 0.75, "zone_3": 0.25}
    for zone_name, weight in weights.items():
        if zone_name in zone_vegetation:
            density = zone_vegetation[zone_name]["veg_density"]
            density_score += weight * density

    risk_score = dist_score + density_score
    return round(min(max(risk_score, 1.0), 10.0), 1)


# ============================================================
# VISUALIZATION
# ============================================================

def create_risk_overlay(orig_image, building_mask, woodland_mask, buildings, gsd):
    """
    Create a colored overlay showing FireSmart zones around each building.
    """
    h, w = building_mask.shape
    overlay = orig_image.copy().astype(np.float32)

    # Distance from each pixel to nearest building pixel
    building_border = cv2.dilate(building_mask, np.ones((3, 3), np.uint8)) - building_mask
    if building_border.sum() == 0:
        building_border = building_mask
    dist_from_building = ndimage.distance_transform_edt(1 - building_border)

    # Draw zone rings (farthest first so closest zones are on top)
    zone_order = ["zone_3", "zone_2", "zone_1b", "zone_1a"]
    for zone_name in zone_order:
        zone_def = ZONES[zone_name]
        min_px = meters_to_pixels(zone_def["min_m"], gsd)
        max_px = meters_to_pixels(zone_def["max_m"], gsd)

        zone_ring = (dist_from_building >= min_px) & (dist_from_building < max_px)
        zone_ring = zone_ring & (~(building_mask > 0))  # Exclude buildings

        if zone_ring.any():
            color = np.array(zone_def["color"], dtype=np.float32)
            overlay[zone_ring] = overlay[zone_ring] * 0.7 + color * 0.3

    # Highlight vegetation in green
    veg_mask = woodland_mask > 0
    overlay[veg_mask] = overlay[veg_mask] * 0.6 + np.array([0, 180, 0], dtype=np.float32) * 0.4

    # Buildings in blue
    bld_mask = building_mask > 0
    overlay[bld_mask] = overlay[bld_mask] * 0.5 + np.array([0, 80, 255], dtype=np.float32) * 0.5

    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Draw building centroids and risk scores
    for bld in buildings:
        cx, cy = bld["centroid"]
        risk = bld.get("risk_score", 0)

        # Color by risk
        if risk >= 7:
            txt_color = (255, 0, 0)
        elif risk >= 4:
            txt_color = (255, 165, 0)
        else:
            txt_color = (0, 200, 0)

        # Draw risk score
        cv2.putText(overlay, f"{risk}", (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(overlay, f"{risk}", (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 2)

    return overlay


def create_legend(height=400, width=250):
    """Create a legend image for the risk map."""
    legend = np.ones((height, width, 3), dtype=np.uint8) * 255

    y = 30
    cv2.putText(legend, "FireSmart Risk Map", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    y += 40
    # Zone colors
    for zone_name in ["zone_1a", "zone_1b", "zone_2", "zone_3"]:
        zone_def = ZONES[zone_name]
        color = tuple(zone_def["color"][::-1])  # BGR for OpenCV
        cv2.rectangle(legend, (10, y - 12), (30, y + 4), color, -1)
        cv2.rectangle(legend, (10, y - 12), (30, y + 4), (0, 0, 0), 1)
        cv2.putText(legend, zone_def["label"], (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        y += 30

    # Other elements
    y += 10
    cv2.rectangle(legend, (10, y - 12), (30, y + 4), (255, 80, 0), -1)
    cv2.putText(legend, "Building", (40, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    y += 30
    cv2.rectangle(legend, (10, y - 12), (30, y + 4), (0, 180, 0), -1)
    cv2.putText(legend, "Vegetation", (40, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    y += 40
    cv2.putText(legend, "Risk Score (1-10)", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    y += 25
    cv2.putText(legend, "7-10: High risk", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    y += 22
    cv2.putText(legend, "4-6.9: Moderate risk", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)
    y += 22
    cv2.putText(legend, "1-3.9: Low risk", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 0), 1)

    return legend


# ============================================================
# MAIN PIPELINE
# ============================================================

def process_tile(tile_name, building_mask, woodland_mask, orig_image, gsd, output_dir):
    """Process a single tile: extract buildings, compute risk, generate overlay."""

    buildings = extract_buildings(building_mask)

    if not buildings:
        return None

    tile_results = {
        "tile": tile_name,
        "num_buildings": len(buildings),
        "buildings": [],
    }

    # Compute risk for each building
    for bld in buildings:
        # Create individual building mask
        bld_mask = np.zeros_like(building_mask)
        cv2.drawContours(bld_mask, [bld["contour"]], -1, 1, -1)

        min_dist, zone_veg, risk_score = compute_building_risk(
            bld_mask, woodland_mask, gsd
        )

        bld["min_distance_m"] = min_dist
        bld["zone_vegetation"] = zone_veg
        bld["risk_score"] = risk_score
        bld["area_m2"] = bld["area_px"] * gsd * gsd

        tile_results["buildings"].append({
            "id": bld["id"],
            "centroid": bld["centroid"],
            "area_m2": round(bld["area_m2"], 1),
            "min_veg_distance_m": round(min_dist, 2),
            "risk_score": risk_score,
            "zone_1a_veg_density": round(zone_veg["zone_1a"]["veg_density"], 3),
            "zone_1b_veg_density": round(zone_veg["zone_1b"]["veg_density"], 3),
            "zone_2_veg_density": round(zone_veg["zone_2"]["veg_density"], 3),
            "overlay_image": f"overlay/{tile_name}.png",
        })

    # Generate overlay
    overlay = create_risk_overlay(orig_image, building_mask, woodland_mask, buildings, gsd)
    Image.fromarray(overlay).save(output_dir / "overlay" / f"{tile_name}.png")

    return tile_results


def main():
    parser = argparse.ArgumentParser(description="FireSmart Risk Assessment")
    parser.add_argument("--building_masks", type=str, default=None)
    parser.add_argument("--woodland_masks", type=str, default=None)
    parser.add_argument("--combined_masks", type=str, default=None,
                        help="Combined masks (0=bg, 1=building, 2=woodland)")
    parser.add_argument("--tiles_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./firesmart_results")
    parser.add_argument("--gsd", type=float, default=0.06,
                        help="Ground sample distance in meters/pixel (default: 0.06 for Cumberland)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    (output_dir / "overlay").mkdir(parents=True, exist_ok=True)

    tiles_dir = Path(args.tiles_dir)
    tile_files = sorted(
        list(tiles_dir.glob("*.png")) + list(tiles_dir.glob("*.jpg"))
    )
    print(f"Found {len(tile_files)} tiles")
    print(f"GSD: {args.gsd} m/pixel")
    print(f"Zone 1a (0-1.5m): {meters_to_pixels(1.5, args.gsd)} px")
    print(f"Zone 1b (1.5-10m): {meters_to_pixels(10, args.gsd)} px")
    print(f"Zone 2 (10-30m): {meters_to_pixels(30, args.gsd)} px")

    # Save legend
    legend = create_legend()
    Image.fromarray(legend[:, :, ::-1]).save(output_dir / "legend.png")

    all_results = []
    all_buildings = []

    for tile_path in tile_files:
        tile_name = tile_path.stem

        # Load masks
        if args.combined_masks:
            mask_path = Path(args.combined_masks) / f"{tile_name}.png"
            if not mask_path.exists():
                continue
            combined = np.array(Image.open(mask_path))
            if combined.ndim == 3:
                combined = combined[:, :, 0]
            building_mask = (combined == 1).astype(np.uint8)
            woodland_mask = (combined == 2).astype(np.uint8)
        else:
            b_path = Path(args.building_masks) / f"{tile_name}.png"
            w_path = Path(args.woodland_masks) / f"{tile_name}.png"
            if not b_path.exists() or not w_path.exists():
                continue
            building_mask = np.array(Image.open(b_path))
            if building_mask.ndim == 3:
                building_mask = building_mask[:, :, 0]
            building_mask = (building_mask > 127).astype(np.uint8)

            woodland_mask = np.array(Image.open(w_path))
            if woodland_mask.ndim == 3:
                woodland_mask = woodland_mask[:, :, 0]
            woodland_mask = (woodland_mask > 127).astype(np.uint8)

        # Load original image
        orig = np.array(Image.open(tile_path).convert("RGB"))

        # Resize masks if needed
        if building_mask.shape != orig.shape[:2]:
            building_mask = cv2.resize(building_mask, (orig.shape[1], orig.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
            woodland_mask = cv2.resize(woodland_mask, (orig.shape[1], orig.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)

        # Process
        result = process_tile(tile_name, building_mask, woodland_mask, orig, args.gsd, output_dir)

        if result:
            all_results.append(result)
            for bld in result["buildings"]:
                bld["tile"] = tile_name
                all_buildings.append(bld)

    # ============================================================
    # SUMMARY STATISTICS
    # ============================================================

    print(f"\n{'=' * 60}")
    print("FIRESMART RISK ASSESSMENT SUMMARY")
    print(f"{'=' * 60}")

    total_buildings = len(all_buildings)
    print(f"Total buildings analyzed: {total_buildings}")

    if total_buildings == 0:
        print("No buildings found!")
        return

    # Risk distribution
    risk_scores = [b["risk_score"] for b in all_buildings]
    high_risk = sum(1 for r in risk_scores if r >= 7)
    moderate_risk = sum(1 for r in risk_scores if 4 <= r < 7)
    low_risk = sum(1 for r in risk_scores if r < 4)

    print(f"\nRisk Distribution:")
    print(f"  High risk (7-10):     {high_risk} buildings ({high_risk/total_buildings*100:.1f}%)")
    print(f"  Moderate risk (4-6.9): {moderate_risk} buildings ({moderate_risk/total_buildings*100:.1f}%)")
    print(f"  Low risk (1-3.9):     {low_risk} buildings ({low_risk/total_buildings*100:.1f}%)")
    print(f"\n  Mean risk score: {np.mean(risk_scores):.1f}")
    print(f"  Median risk score: {np.median(risk_scores):.1f}")

    # Distance statistics
    distances = [b["min_veg_distance_m"] for b in all_buildings if b["min_veg_distance_m"] < 999]
    if distances:
        print(f"\nVegetation Proximity:")
        print(f"  Mean min distance: {np.mean(distances):.1f}m")
        print(f"  Median min distance: {np.median(distances):.1f}m")
        zone1a_count = sum(1 for d in distances if d < 1.5)
        zone1b_count = sum(1 for d in distances if 1.5 <= d < 10)
        zone2_count = sum(1 for d in distances if 10 <= d < 30)
        zone3_count = sum(1 for d in distances if d >= 30)
        print(f"  Buildings with vegetation <1.5m:  {zone1a_count} ({zone1a_count/len(distances)*100:.1f}%)")
        print(f"  Buildings with vegetation 1.5-10m: {zone1b_count} ({zone1b_count/len(distances)*100:.1f}%)")
        print(f"  Buildings with vegetation 10-30m:  {zone2_count} ({zone2_count/len(distances)*100:.1f}%)")
        print(f"  Buildings with vegetation >30m:    {zone3_count} ({zone3_count/len(distances)*100:.1f}%)")

    # Save CSV
    csv_path = output_dir / "building_risk_scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "tile", "id", "centroid", "area_m2", "min_veg_distance_m",
            "risk_score", "zone_1a_veg_density", "zone_1b_veg_density", "zone_2_veg_density",
            "overlay_image"
        ])
        writer.writeheader()
        for bld in all_buildings:
            writer.writerow(bld)
    print(f"\nPer-building results: {csv_path}")

    # Save JSON
    json_path = output_dir / "risk_summary.json"
    summary = {
        "total_buildings": total_buildings,
        "gsd_m": args.gsd,
        "risk_distribution": {
            "high": high_risk,
            "moderate": moderate_risk,
            "low": low_risk,
        },
        "mean_risk_score": round(float(np.mean(risk_scores)), 2),
        "median_risk_score": round(float(np.median(risk_scores)), 2),
        "mean_min_distance_m": round(float(np.mean(distances)), 2) if distances else None,
        "tiles_processed": len(all_results),
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {json_path}")

    # Top 10 highest risk buildings
    all_buildings.sort(key=lambda x: x["risk_score"], reverse=True)
    print(f"\nTop 10 Highest Risk Buildings:")
    print(f"  {'Tile':<35} {'Risk':<6} {'Min Dist':<10} {'Area m²':<10}")
    print(f"  {'-'*61}")
    for bld in all_buildings[:10]:
        print(f"  {bld['tile']:<35} {bld['risk_score']:<6} "
              f"{bld['min_veg_distance_m']:<10.1f} {bld['area_m2']:<10.1f}")

    print(f"\nOverlays saved to: {output_dir / 'overlay'}")
    print(f"Legend saved to: {output_dir / 'legend.png'}")


if __name__ == "__main__":
    main()