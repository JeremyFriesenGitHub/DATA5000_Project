"""
Generate Neighbourhood Overview Maps
=====================================
Creates bird's-eye-view maps of entire GeoTIFF areas with parcels filled
by FireSmart risk color, risk score labels, vegetation overlay, building
outlines, and a legend. Inspired by FireSmart Status Inventory maps.

Usage:
    python generate_neighbourhood_maps.py \
        --tif_dir ./tif \
        --masks_dir ./cumberland_combined_v11 \
        --parcels ./ParcelsSubdivisonOnly/ParcelsSubdivisonOnly.shp \
        --risk_csv ./firesmart_property_results_v2/parcel_risk_scores.csv \
        --output_dir ./firesmart_property_results_v2/neighbourhood
"""

import argparse
import csv
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from scipy import ndimage
import shapefile

Image.MAX_IMAGE_PIXELS = 300000000

TILE_SIZE = 512

# Scale factor for output (full res is huge)
SCALE = 0.25

# Risk color palette (RGB) — matches FireSmart convention
RISK_COLORS = {
    "high":     (220, 40, 40),      # red — risk >= 7
    "mod_high": (240, 120, 20),     # orange — risk 5-7
    "moderate": (245, 210, 40),     # yellow — risk 3-5
    "low":      (80, 190, 60),      # green — risk < 3
    "no_data":  (180, 180, 180),    # grey — no risk data
}


def risk_to_color(risk):
    """Map risk score to RGB color."""
    if risk is None:
        return RISK_COLORS["no_data"]
    if risk >= 7.0:
        return RISK_COLORS["high"]
    elif risk >= 5.0:
        return RISK_COLORS["mod_high"]
    elif risk >= 3.0:
        return RISK_COLORS["moderate"]
    else:
        return RISK_COLORS["low"]


def risk_to_label(risk):
    """Short label for legend matching."""
    if risk is None:
        return "no_data"
    if risk >= 7.0:
        return "high"
    elif risk >= 5.0:
        return "mod_high"
    elif risk >= 3.0:
        return "moderate"
    else:
        return "low"


def load_geotiff_meta(tif_path):
    img = Image.open(tif_path)
    w, h = img.size
    tags = img.tag_v2
    pixel_scale = tags.get(33550, None)
    tiepoint = tags.get(33922, None)
    if pixel_scale is None or tiepoint is None:
        raise ValueError(f"Missing GeoTIFF tags in {tif_path}")
    img.close()
    return {
        "path": str(tif_path),
        "width": w,
        "height": h,
        "gsd_x": pixel_scale[0],
        "gsd_y": pixel_scale[1],
        "origin_x": tiepoint[3],
        "origin_y": tiepoint[4],
        "bbox": (
            tiepoint[3],
            tiepoint[4] - h * pixel_scale[1],
            tiepoint[3] + w * pixel_scale[0],
            tiepoint[4],
        ),
        "short_name": Path(tif_path).stem[:8],
    }


def utm_to_pixel(utm_x, utm_y, meta):
    px = (utm_x - meta["origin_x"]) / meta["gsd_x"]
    py = (meta["origin_y"] - utm_y) / meta["gsd_y"]
    return px, py


def stitch_masks(masks_dir, short_name, full_h, full_w, mask_type="building_masks"):
    mask = np.zeros((full_h, full_w), dtype=np.uint8)
    mask_path = Path(masks_dir) / mask_type
    for tile_file in mask_path.glob(f"{short_name}_*.png"):
        parts = tile_file.stem.split("_")
        row = int(parts[1])
        col = int(parts[2])
        tile = np.array(Image.open(tile_file))
        if tile.ndim == 3:
            tile = tile[:, :, 0]
        tile_bin = (tile > 127).astype(np.uint8)
        th, tw = tile_bin.shape
        end_row = min(row + th, full_h)
        end_col = min(col + tw, full_w)
        mask[row:end_row, col:end_col] = tile_bin[:end_row - row, :end_col - col]
    return mask


def draw_legend(overlay, x, y):
    """Draw a FireSmart risk legend box on the image."""
    legend_w = 220
    legend_h = 195
    pad = 10

    # Semi-transparent black background
    roi = overlay[y:y+legend_h, x:x+legend_w].astype(np.float32)
    roi = roi * 0.25 + np.array([0, 0, 0], dtype=np.float32) * 0.75
    overlay[y:y+legend_h, x:x+legend_w] = np.clip(roi, 0, 255).astype(np.uint8)

    # White border
    cv2.rectangle(overlay, (x, y), (x+legend_w, y+legend_h), (255, 255, 255), 2)

    # Title
    ty = y + 25
    cv2.putText(overlay, "FireSmart Risk", (x+pad, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Legend entries
    entries = [
        ("High (7-10)", RISK_COLORS["high"]),
        ("Mod-High (5-7)", RISK_COLORS["mod_high"]),
        ("Moderate (3-5)", RISK_COLORS["moderate"]),
        ("Low (1-3)", RISK_COLORS["low"]),
        ("Vegetation", (0, 180, 0)),
        ("Building", (100, 140, 220)),
    ]

    for i, (label, color) in enumerate(entries):
        ey = ty + 18 + i * 24
        # Color swatch
        cv2.rectangle(overlay, (x+pad, ey-10), (x+pad+16, ey+6), color, -1)
        cv2.rectangle(overlay, (x+pad, ey-10), (x+pad+16, ey+6), (255, 255, 255), 1)
        # Label
        cv2.putText(overlay, label, (x+pad+22, ey+3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    return overlay


def draw_text_with_outline(img, text, pos, scale, color, thickness=2, outline_color=(0, 0, 0)):
    """Draw text with a dark outline for readability."""
    x, y = pos
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, outline_color, thickness + 2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def main():
    parser = argparse.ArgumentParser(description="Generate neighbourhood overview maps")
    parser.add_argument("--tif_dir", type=str, required=True)
    parser.add_argument("--masks_dir", type=str, required=True)
    parser.add_argument("--parcels", type=str, required=True)
    parser.add_argument("--risk_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--scale", type=float, default=SCALE)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = args.scale

    # ── Load risk scores by PID ──
    print("Loading risk scores...")
    pid_data = {}
    with open(args.risk_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["pid"]
            risk = row["max_risk_score"]
            num_bld = row.get("num_buildings", "0")
            area = row.get("parcel_area_m2", "0")
            pid_data[pid] = {
                "risk": float(risk) if risk else None,
                "num_buildings": int(num_bld) if num_bld else 0,
                "area_m2": float(area) if area else 0,
            }

    # ── Load GeoTIFF metadata ──
    print("Loading GeoTIFF metadata...")
    tif_dir = Path(args.tif_dir)
    geotiff_metas = []
    for tif_path in sorted(tif_dir.glob("*.tif")):
        meta = load_geotiff_meta(tif_path)
        geotiff_metas.append(meta)
        print(f"  {meta['short_name']}: {meta['width']}x{meta['height']}")

    # ── Load parcels ──
    print("Loading parcels...")
    sf = shapefile.Reader(args.parcels)
    fields = [f[0] for f in sf.fields[1:]]

    # ── Process each GeoTIFF ──
    for meta in geotiff_metas:
        sn = meta["short_name"]
        gsd = (meta["gsd_x"] + meta["gsd_y"]) / 2.0
        full_h, full_w = meta["height"], meta["width"]
        out_h, out_w = int(full_h * scale), int(full_w * scale)

        print(f"\nProcessing {sn} ({full_w}x{full_h} -> {out_w}x{out_h})...")

        # Load and resize original image
        print(f"  Loading imagery...")
        orig = np.array(Image.open(meta["path"]).convert("RGB"))
        orig_small = cv2.resize(orig, (out_w, out_h), interpolation=cv2.INTER_AREA)
        del orig

        # Stitch and resize masks
        print(f"  Stitching masks...")
        b_mask = stitch_masks(args.masks_dir, sn, full_h, full_w, "building_masks")
        w_mask = stitch_masks(args.masks_dir, sn, full_h, full_w, "woodland_masks")
        b_mask_small = cv2.resize(b_mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        w_mask_small = cv2.resize(w_mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        del b_mask, w_mask

        # Start with slightly darkened base imagery
        overlay = (orig_small.astype(np.float32) * 0.85).astype(np.uint8)

        # ── Step 1: Subtle vegetation overlay on base ──
        veg_mask = w_mask_small > 0
        overlay[veg_mask] = np.clip(
            overlay[veg_mask].astype(np.float32) * 0.65 + np.array([0, 160, 0], dtype=np.float32) * 0.35,
            0, 255
        ).astype(np.uint8)

        # ── Step 2: Fill parcels with risk color ──
        print(f"  Filling parcels by risk...")
        parcels_drawn = 0
        parcel_label_positions = []  # collect (cx, cy, risk_score) for labeling

        for idx in range(len(sf)):
            shape = sf.shape(idx)
            rec = sf.record(idx)
            rec_dict = dict(zip(fields, rec))

            p_bbox = shape.bbox
            g_bbox = meta["bbox"]

            # Quick overlap check
            if p_bbox[2] < g_bbox[0] or p_bbox[0] > g_bbox[2]:
                continue
            if p_bbox[3] < g_bbox[1] or p_bbox[1] > g_bbox[3]:
                continue

            pid = rec_dict.get("PID", "")
            data = pid_data.get(pid, None)
            if data is None or data["risk"] is None or data["num_buildings"] == 0:
                continue

            risk = data["risk"]
            area_m2 = data.get("area_m2", 0)

            # Convert parcel polygon to scaled pixel coords
            pts = []
            for x, y in shape.points:
                px, py = utm_to_pixel(x, y, meta)
                pts.append((int(px * scale), int(py * scale)))

            pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

            # Only fill residential-sized parcels (< 10,000 m²)
            # Large parcels (farms, crown land, parks) just get a boundary
            if area_m2 < 10000:
                parcel_fill = np.zeros((out_h, out_w), dtype=np.uint8)
                cv2.fillPoly(parcel_fill, [pts_arr], 1)
                fill_mask = parcel_fill > 0

                color = np.array(risk_to_color(risk), dtype=np.float32)
                overlay[fill_mask] = np.clip(
                    overlay[fill_mask].astype(np.float32) * 0.4 + color * 0.6,
                    0, 255
                ).astype(np.uint8)

            # Draw parcel boundary (dark outline for contrast)
            cv2.polylines(overlay, [pts_arr], isClosed=True, color=(40, 40, 40), thickness=2)

            # Compute centroid for score label
            M = cv2.moments(pts_arr)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Only label if centroid is within image bounds
                if 0 <= cx < out_w and 0 <= cy < out_h:
                    parcel_label_positions.append((cx, cy, risk))

            parcels_drawn += 1

        # ── Step 3: Draw building outlines on top ──
        print(f"  Drawing building outlines...")
        # Find building contours at scaled resolution
        contours, _ = cv2.findContours(b_mask_small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:  # skip tiny fragments at scaled res
                continue
            # Fill building with a blue-grey tint
            bld_fill = np.zeros((out_h, out_w), dtype=np.uint8)
            cv2.drawContours(bld_fill, [cnt], -1, 1, -1)
            bld_pixels = bld_fill > 0
            overlay[bld_pixels] = np.clip(
                overlay[bld_pixels].astype(np.float32) * 0.4 + np.array([100, 140, 220], dtype=np.float32) * 0.6,
                0, 255
            ).astype(np.uint8)
            # Building outline
            cv2.drawContours(overlay, [cnt], -1, (50, 60, 80), 1)

        # ── Step 4: Draw risk score labels ──
        print(f"  Labeling {len(parcel_label_positions)} parcels...")
        for cx, cy, risk in parcel_label_positions:
            score_str = f"{risk:.1f}"

            # White text with dark outline — readable on any background
            font_scale = 0.45
            draw_text_with_outline(overlay, score_str, (cx - 14, cy + 5),
                                   font_scale, (255, 255, 255), thickness=1)

        print(f"  {parcels_drawn} parcels drawn")

        # ── Step 5: Draw legend ──
        legend_x = 10
        legend_y = 10
        draw_legend(overlay, legend_x, legend_y)

        # Save
        out_path = output_dir / f"neighbourhood_{sn}.png"
        Image.fromarray(overlay).save(out_path, optimize=True)
        print(f"  Saved: {out_path} ({out_w}x{out_h})")

    print("\nDone!")


if __name__ == "__main__":
    main()
