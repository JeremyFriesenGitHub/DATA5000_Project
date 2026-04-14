#!/usr/bin/env python3
"""
Comparison evaluation:
1. 4 Cumberland parcels × (high-res, low-res) × (with/without parcel boundary)
2. Community-level scores for Cumberland (drone+sat), Logan Lake (with/without), West Kelowna (without)
"""
import csv, json, sys, os
import numpy as np
from pathlib import Path
from scipy import ndimage
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # Allow large GeoTIFFs
sys.stdout.reconfigure(line_buffering=True)  # Force line-buffered output
BASE_DIR = Path("/home/jeremy/COMP4900_Project_backup_files")

ZONES = {
    "zone_1a": {"min_m": 0.0,  "max_m": 1.5},
    "zone_1b": {"min_m": 1.5,  "max_m": 10.0},
    "zone_2":  {"min_m": 10.0, "max_m": 30.0},
    "zone_3":  {"min_m": 30.0, "max_m": 999.0},
}
WEIGHTS = {"zone_1a": 5.0, "zone_1b": 3.0, "zone_2": 1.25, "zone_3": 0.75}
TARGET_PARCELS = ["017028141", "027266834", "027266966", "001296507"]


def load_mask(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return (img > 127).astype(np.uint8) if img is not None else None


def extract_buildings(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [{"contour": c, "area": cv2.contourArea(c)} for c in contours if cv2.contourArea(c) >= 5]


def compute_risk_both_modes(bld_mask, woodland, gsd_x, gsd_y, parcel_mask):
    """Returns {without_parcel: {...}, with_parcel: {...}}"""
    border = cv2.dilate(bld_mask, np.ones((3,3), np.uint8)) - bld_mask
    if border.sum() == 0:
        border = bld_mask
    dist_m = ndimage.distance_transform_edt(1 - border, sampling=(gsd_y, gsd_x))
    veg = woodland > 0
    on_parcel = parcel_mask > 0

    result = {}
    for mode in ["without_parcel", "with_parcel"]:
        densities = {}
        for zn, zd in ZONES.items():
            ring = (dist_m >= zd["min_m"]) & (dist_m < zd["max_m"]) & (~(bld_mask > 0))
            if mode == "with_parcel":
                zt = int((ring & on_parcel).sum())
                zv = int((ring & veg & on_parcel).sum())
            else:
                zt = int(ring.sum())
                zv = int((ring & veg).sum())
            densities[zn] = float(zv / zt) if zt > 0 else 0.0

        score = sum(WEIGHTS[z] * densities[z] for z in ZONES)
        if mode == "with_parcel":
            veg_m = veg & on_parcel
            md = float(dist_m[veg_m].min()) if veg_m.any() else 999.0
        else:
            md = float(dist_m[veg].min()) if veg.any() else 999.0

        result[mode] = {
            "risk_score": round(score, 2),
            "zone_1a": round(densities["zone_1a"], 4),
            "zone_1b": round(densities["zone_1b"], 4),
            "zone_2": round(densities["zone_2"], 4),
            "zone_3": round(densities["zone_3"], 4),
            "min_veg_dist": round(md, 2),
        }
    return result


# ============================================================
# HIGH-RES: stitch 512x512 CNN inference tiles per GeoTIFF
# ============================================================

def stitch_cnn_masks(masks_dir, short_name, full_h, full_w, mask_type="building_masks"):
    mask = np.zeros((full_h, full_w), dtype=np.uint8)
    mask_path = Path(masks_dir) / mask_type
    for tile_file in mask_path.glob(f"{short_name}_*.png"):
        parts = tile_file.stem.split("_")
        row, col = int(parts[1]), int(parts[2])
        tile = np.array(Image.open(tile_file))
        if tile.ndim == 3:
            tile = tile[:, :, 0]
        tile_bin = (tile > 127).astype(np.uint8)
        th, tw = tile_bin.shape
        er, ec = min(row + th, full_h), min(col + tw, full_w)
        mask[row:er, col:ec] = tile_bin[:er-row, :ec-col]
    return mask


def load_geotiff_meta_pil(tif_path):
    img = Image.open(tif_path)
    w, h = img.size
    tags = img.tag_v2
    ps = tags.get(33550)
    tp = tags.get(33922)
    img.close()
    if ps is None or tp is None:
        return None
    return {
        "path": str(tif_path), "width": w, "height": h,
        "gsd_x": ps[0], "gsd_y": ps[1],
        "origin_x": tp[3], "origin_y": tp[4],
        "short_name": Path(tif_path).stem[:8],
        "bbox": (tp[3], tp[4] - h * ps[1], tp[3] + w * ps[0], tp[4]),
    }


def utm_to_px(x, y, ox, oy, gx, gy):
    return (x - ox) / gx, (oy - y) / gy


# ============================================================
# LOW-RES: stitch per-tile masks using GeoTIFF coordinates
# ============================================================

def load_geotiff_meta_raw(tif_path):
    from struct import unpack
    with open(tif_path, 'rb') as f:
        header = f.read(4)
        endian = '<' if header[:2] == b'II' else '>'
        ifd_offset = unpack(f'{endian}I', f.read(4))[0]
        f.seek(ifd_offset)
        n = unpack(f'{endian}H', f.read(2))[0]
        tags = {}
        for _ in range(n):
            tid, tt, cnt = unpack(f'{endian}HHI', f.read(8))
            vo = f.read(4)
            tags[tid] = (tt, cnt, vo, f.tell() - 4)
        w = unpack(f'{endian}I', tags[256][2])[0] if 256 in tags else None
        h = unpack(f'{endian}I', tags[257][2])[0] if 257 in tags else None
        if 33922 not in tags or 33550 not in tags:
            return None
        _, cnt1, _, _ = tags[33922]
        off1 = unpack(f'{endian}I', tags[33922][2])[0]
        f.seek(off1)
        tp = unpack(f'{endian}{"d"*cnt1}', f.read(cnt1*8))
        _, cnt2, _, _ = tags[33550]
        off2 = unpack(f'{endian}I', tags[33550][2])[0]
        f.seek(off2)
        ps = unpack(f'{endian}{"d"*cnt2}', f.read(cnt2*8))
    return {
        "name": tif_path.stem, "path": tif_path, "width": w, "height": h,
        "gsd_x": ps[0], "gsd_y": ps[1],
        "origin_x": tp[3], "origin_y": tp[4],
        "bbox": (tp[3], tp[4] - h * ps[1], tp[3] + w * ps[0], tp[4]),
    }


def stitch_lowres_masks(tile_metas, masks_dir, mask_type):
    gx = tile_metas[0]["gsd_x"]
    gy = tile_metas[0]["gsd_y"]
    min_x = min(m["bbox"][0] for m in tile_metas)
    min_y = min(m["bbox"][1] for m in tile_metas)
    max_x = max(m["bbox"][2] for m in tile_metas)
    max_y = max(m["bbox"][3] for m in tile_metas)
    cw = int(np.ceil((max_x - min_x) / gx))
    ch = int(np.ceil((max_y - min_y) / gy))
    canvas = np.zeros((ch, cw), dtype=np.uint8)
    for m in tile_metas:
        mp = masks_dir / mask_type / f"{m['name']}.png"
        if not mp.exists():
            continue
        t = load_mask(mp)
        if t is None:
            continue
        px = int(round((m["origin_x"] - min_x) / gx))
        py = int(round((max_y - m["origin_y"]) / gy))
        th, tw = t.shape
        sy, sx = max(0, -py), max(0, -px)
        dy, dx = max(0, py), max(0, px)
        ch2 = min(th - sy, ch - dy)
        cw2 = min(tw - sx, cw - dx)
        if ch2 > 0 and cw2 > 0:
            canvas[dy:dy+ch2, dx:dx+cw2] = t[sy:sy+ch2, sx:sx+cw2]
    return canvas, {"origin_x": min_x, "origin_y": max_y, "gsd_x": gx, "gsd_y": gy, "width": cw, "height": ch}


def process_parcels(b_stitched, w_stitched, extent, parcels_path, target_pids=None):
    """Process parcels against stitched masks. Returns per-parcel results."""
    import shapefile as shp
    sf = shp.Reader(str(parcels_path))
    fields = [f[0] for f in sf.fields[1:]]
    gx, gy = extent["gsd_x"], extent["gsd_y"]
    results = []

    for idx in range(len(sf)):
        rec = sf.record(idx)
        rd = dict(zip(fields, rec))
        pid = rd.get("PID", str(idx))
        if target_pids and pid not in target_pids:
            continue

        shape = sf.shape(idx)
        bb = shape.bbox
        BUF = 35.0
        buf = (bb[0]-BUF, bb[1]-BUF, bb[2]+BUF, bb[3]+BUF)
        pxmin, pymin = utm_to_px(buf[0], buf[3], extent["origin_x"], extent["origin_y"], gx, gy)
        pxmax, pymax = utm_to_px(buf[2], buf[1], extent["origin_x"], extent["origin_y"], gx, gy)
        pxmin, pymin = max(0, int(pxmin)), max(0, int(pymin))
        pxmax, pymax = min(extent["width"], int(pxmax)), min(extent["height"], int(pymax))
        cw, ch = pxmax - pxmin, pymax - pymin
        if cw < 10 or ch < 10:
            continue

        bc = b_stitched[pymin:pymax, pxmin:pxmax].copy()
        wc = w_stitched[pymin:pymax, pxmin:pxmax].copy()

        pts = [(int(utm_to_px(x, y, extent["origin_x"], extent["origin_y"], gx, gy)[0] - pxmin),
                int(utm_to_px(x, y, extent["origin_x"], extent["origin_y"], gx, gy)[1] - pymin))
               for x, y in shape.points]
        pm = np.zeros((ch, cw), dtype=np.uint8)
        cv2.fillPoly(pm, [np.array(pts, dtype=np.int32).reshape(-1,1,2)], 1)

        blds = extract_buildings(bc)
        parcel_blds = []
        for b in blds:
            bm = np.zeros((ch, cw), dtype=np.uint8)
            cv2.drawContours(bm, [b["contour"]], -1, 1, -1)
            if (bm & pm).sum() > 0:
                parcel_blds.append(b)

        if not parcel_blds:
            results.append({"pid": pid, "num_buildings": 0, "without_parcel": {"risk_score": None}, "with_parcel": {"risk_score": None}})
            continue

        all_r = []
        for b in parcel_blds:
            bm = np.zeros((ch, cw), dtype=np.uint8)
            cv2.drawContours(bm, [b["contour"]], -1, 1, -1)
            all_r.append(compute_risk_both_modes(bm, wc, gx, gy, pm))

        def agg(mode):
            items = [r[mode] for r in all_r]
            return {
                "risk_score": round(max(d["risk_score"] for d in items), 2),
                "zone_1a": round(np.mean([d["zone_1a"] for d in items]), 4),
                "zone_1b": round(np.mean([d["zone_1b"] for d in items]), 4),
                "zone_2": round(np.mean([d["zone_2"] for d in items]), 4),
                "zone_3": round(np.mean([d["zone_3"] for d in items]), 4),
                "min_veg_dist": round(min(d["min_veg_dist"] for d in items), 2),
            }
        results.append({"pid": pid, "num_buildings": len(parcel_blds),
                        "without_parcel": agg("without_parcel"), "with_parcel": agg("with_parcel")})
    return results


def community_scores(b_stitched, w_stitched, gsd_x, gsd_y, label):
    """Community risk: all buildings, no parcel scoping. Crops per-building for speed."""
    blds = extract_buildings(b_stitched)
    print(f"  {label}: {len(blds)} buildings", flush=True)
    BUF_PX = int(np.ceil(100.0 / min(gsd_x, gsd_y)))  # 100m buffer covers zones 1a-3
    H, W = b_stitched.shape
    scores = []
    for i, b in enumerate(blds):
        if i % 200 == 0 and i > 0:
            print(f"    ... {i}/{len(blds)}", flush=True)
        x, y, bw, bh = cv2.boundingRect(b["contour"])
        # Crop region around building with buffer
        y0 = max(0, y - BUF_PX)
        x0 = max(0, x - BUF_PX)
        y1 = min(H, y + bh + BUF_PX)
        x1 = min(W, x + bw + BUF_PX)
        bc = b_stitched[y0:y1, x0:x1]
        wc = w_stitched[y0:y1, x0:x1]
        # Shift contour to crop coordinates
        offset = np.array([[-x0, -y0]], dtype=np.int32)
        cont_shifted = b["contour"] + offset
        bm = np.zeros_like(bc)
        cv2.drawContours(bm, [cont_shifted], -1, 1, -1)
        border = cv2.dilate(bm, np.ones((3,3), np.uint8)) - bm
        if border.sum() == 0: border = bm
        dist_m = ndimage.distance_transform_edt(1 - border, sampling=(gsd_y, gsd_x))
        veg = wc > 0
        densities = {}
        for zn, zd in ZONES.items():
            ring = (dist_m >= zd["min_m"]) & (dist_m < zd["max_m"]) & (~(bm > 0))
            zt = int(ring.sum())
            zv = int((ring & veg).sum())
            densities[zn] = float(zv/zt) if zt > 0 else 0.0
        scores.append(round(sum(WEIGHTS[z]*densities[z] for z in ZONES), 2))
    nz = [s for s in scores if s > 0]
    return {
        "total": len(scores), "with_veg": len(nz),
        "mean_all": round(np.mean(scores), 3) if scores else 0,
        "mean_filtered": round(np.mean(nz), 3) if nz else 0,
        "median": round(float(np.median(scores)), 3) if scores else 0,
        "max": round(float(np.max(scores)), 2) if scores else 0,
    }


def community_scores_with_parcels(b_stitched, w_stitched, extent, parcels_path, label):
    """Community risk: parcel-scoped densities."""
    import shapefile as shp
    sf = shp.Reader(str(parcels_path))
    fields = [f[0] for f in sf.fields[1:]]
    gx, gy = extent["gsd_x"], extent["gsd_y"]
    n_parcels = len(sf)
    print(f"  {label}: processing {n_parcels} parcels...", flush=True)
    scores = []
    for idx in range(n_parcels):
        if idx % 200 == 0 and idx > 0:
            print(f"    ... parcel {idx}/{n_parcels}, {len(scores)} buildings so far", flush=True)
        shape = sf.shape(idx)
        bb = shape.bbox
        BUF = 35.0
        buf = (bb[0]-BUF, bb[1]-BUF, bb[2]+BUF, bb[3]+BUF)
        pxmin, pymin = utm_to_px(buf[0], buf[3], extent["origin_x"], extent["origin_y"], gx, gy)
        pxmax, pymax = utm_to_px(buf[2], buf[1], extent["origin_x"], extent["origin_y"], gx, gy)
        pxmin, pymin = max(0, int(pxmin)), max(0, int(pymin))
        pxmax, pymax = min(extent["width"], int(pxmax)), min(extent["height"], int(pymax))
        cw, ch = pxmax - pxmin, pymax - pymin
        if cw < 10 or ch < 10: continue
        bc = b_stitched[pymin:pymax, pxmin:pxmax].copy()
        wc = w_stitched[pymin:pymax, pxmin:pxmax].copy()
        pts = [(int(utm_to_px(x, y, extent["origin_x"], extent["origin_y"], gx, gy)[0] - pxmin),
                int(utm_to_px(x, y, extent["origin_x"], extent["origin_y"], gx, gy)[1] - pymin))
               for x, y in shape.points]
        pm = np.zeros((ch, cw), dtype=np.uint8)
        cv2.fillPoly(pm, [np.array(pts, dtype=np.int32).reshape(-1,1,2)], 1)
        blds = extract_buildings(bc)
        for b in blds:
            bm = np.zeros((ch, cw), dtype=np.uint8)
            cv2.drawContours(bm, [b["contour"]], -1, 1, -1)
            if (bm & pm).sum() > 0:
                r = compute_risk_both_modes(bm, wc, gx, gy, pm)
                scores.append(r["with_parcel"]["risk_score"])
    nz = [s for s in scores if s > 0]
    print(f"  {label}: {len(scores)} buildings ({len(nz)} with veg)", flush=True)
    return {
        "total": len(scores), "with_veg": len(nz),
        "mean_all": round(np.mean(scores), 3) if scores else 0,
        "mean_filtered": round(np.mean(nz), 3) if nz else 0,
        "median": round(float(np.median(scores)), 3) if scores else 0,
        "max": round(float(np.max(scores)), 2) if scores else 0,
    }


def main():
    import shapefile as shp
    parcels_path = BASE_DIR / "ParcelsSubdivisonOnly" / "ParcelsSubdivisonOnly.shp"
    all_results = {}

    # ================================================================
    # PART 1A: Cumberland HIGH-RES (drone CNN) — 4 parcels
    # ================================================================
    print("="*60)
    print("CUMBERLAND HIGH-RES (DRONE/CNN) — 4 PARCELS")
    print("="*60)
    tif_dir = BASE_DIR / "tif"
    masks_dir = BASE_DIR / "cumberland_combined_v11"

    # Load GeoTIFF metadata
    hr_metas = []
    for tp in sorted(tif_dir.glob("*.tif")):
        m = load_geotiff_meta_pil(tp)
        if m:
            hr_metas.append(m)
            print(f"  {m['short_name']}: {m['width']}x{m['height']}, GSD={m['gsd_x']:.4f}m")

    # For high-res, we stitch CNN tiles per GeoTIFF then compose them
    # Compute overall extent from all GeoTIFFs
    min_x = min(m["bbox"][0] for m in hr_metas)
    min_y = min(m["bbox"][1] for m in hr_metas)
    max_x = max(m["bbox"][2] for m in hr_metas)
    max_y = max(m["bbox"][3] for m in hr_metas)
    ref_gx = hr_metas[0]["gsd_x"]
    ref_gy = hr_metas[0]["gsd_y"]
    total_w = int(np.ceil((max_x - min_x) / ref_gx))
    total_h = int(np.ceil((max_y - min_y) / ref_gy))
    print(f"\n  Overall canvas: {total_w}x{total_h} px, GSD={ref_gx:.4f}m")

    b_hr = np.zeros((total_h, total_w), dtype=np.uint8)
    w_hr = np.zeros((total_h, total_w), dtype=np.uint8)
    for m in hr_metas:
        sn = m["short_name"]
        print(f"  Stitching {sn}...", end=" ", flush=True)
        bm = stitch_cnn_masks(masks_dir, sn, m["height"], m["width"], "building_masks")
        wm = stitch_cnn_masks(masks_dir, sn, m["height"], m["width"], "woodland_masks")
        # Place on global canvas
        px = int(round((m["origin_x"] - min_x) / ref_gx))
        py = int(round((max_y - m["origin_y"]) / ref_gy))
        h, w = bm.shape
        ey = min(py + h, total_h)
        ex = min(px + w, total_w)
        ch = ey - py
        cw = ex - px
        if ch > 0 and cw > 0:
            b_hr[py:ey, px:ex] = np.maximum(b_hr[py:ey, px:ex], bm[:ch, :cw])
            w_hr[py:ey, px:ex] = np.maximum(w_hr[py:ey, px:ex], wm[:ch, :cw])
        print(f"bld={bm.sum()}, wld={wm.sum()}")

    hr_extent = {"origin_x": min_x, "origin_y": max_y, "gsd_x": ref_gx, "gsd_y": ref_gy,
                 "width": total_w, "height": total_h}
    print(f"  Total: bld={b_hr.sum()}, wld={w_hr.sum()}")

    hr_parcels = process_parcels(b_hr, w_hr, hr_extent, parcels_path, TARGET_PARCELS)
    all_results["highres_parcels"] = hr_parcels

    # ================================================================
    # PART 1B: Cumberland LOW-RES (satellite MS+Meta) — 4 parcels
    # ================================================================
    print("\n" + "="*60)
    print("CUMBERLAND LOW-RES (SATELLITE/MS+META) — 4 PARCELS")
    print("="*60)
    lr_imagery = BASE_DIR / "cumberland_imagery"
    lr_masks = BASE_DIR / "cumberland_meta_masks"
    lr_metas = [m for m in [load_geotiff_meta_raw(p) for p in sorted(lr_imagery.glob("*.tif"))] if m]
    print(f"  {len(lr_metas)} tiles")

    b_lr, lr_extent = stitch_lowres_masks(lr_metas, lr_masks, "building_masks")
    w_lr, _ = stitch_lowres_masks(lr_metas, lr_masks, "woodland_masks")
    print(f"  Canvas: {lr_extent['width']}x{lr_extent['height']} px, bld={b_lr.sum()}, wld={w_lr.sum()}")

    lr_parcels = process_parcels(b_lr, w_lr, lr_extent, parcels_path, TARGET_PARCELS)
    all_results["lowres_parcels"] = lr_parcels

    # ================================================================
    # PART 2: Community scores
    # ================================================================

    # Cumberland — drone (high-res), without parcels
    print("\n" + "="*60)
    print("COMMUNITY SCORES")
    print("="*60)
    c_hr_no = community_scores(b_hr, w_hr, ref_gx, ref_gy, "Cumberland drone (no parcel)")
    all_results["cumberland_drone_no_parcel"] = c_hr_no

    # Cumberland — drone (high-res), with parcels
    c_hr_wp = community_scores_with_parcels(b_hr, w_hr, hr_extent, parcels_path, "Cumberland drone (with parcel)")
    all_results["cumberland_drone_with_parcel"] = c_hr_wp

    # Cumberland — satellite (low-res), without parcels
    c_lr_no = community_scores(b_lr, w_lr, lr_extent["gsd_x"], lr_extent["gsd_y"], "Cumberland sat (no parcel)")
    all_results["cumberland_sat_no_parcel"] = c_lr_no

    # Cumberland — satellite (low-res), with parcels
    c_lr_wp = community_scores_with_parcels(b_lr, w_lr, lr_extent, parcels_path, "Cumberland sat (with parcel)")
    all_results["cumberland_sat_with_parcel"] = c_lr_wp

    # Free Cumberland masks
    del b_hr, w_hr, b_lr, w_lr

    # Logan Lake
    print()
    ll_img = BASE_DIR / "logan_lake_imagery"
    ll_masks = BASE_DIR / "logan_lake_meta_masks"
    ll_parcels = BASE_DIR / "logan_lake_parcels" / "logan_lake_parcels.shp"
    ll_metas = [m for m in [load_geotiff_meta_raw(p) for p in sorted(ll_img.glob("*.tif"))] if m]
    print(f"  Logan Lake: {len(ll_metas)} tiles")
    b_ll, ll_ext = stitch_lowres_masks(ll_metas, ll_masks, "building_masks")
    w_ll, _ = stitch_lowres_masks(ll_metas, ll_masks, "woodland_masks")
    print(f"  Canvas: {ll_ext['width']}x{ll_ext['height']}, bld={b_ll.sum()}, wld={w_ll.sum()}")

    ll_no = community_scores(b_ll, w_ll, ll_ext["gsd_x"], ll_ext["gsd_y"], "Logan Lake (no parcel)")
    all_results["logan_lake_no_parcel"] = ll_no

    ll_wp = community_scores_with_parcels(b_ll, w_ll, ll_ext, ll_parcels, "Logan Lake (with parcel)")
    all_results["logan_lake_with_parcel"] = ll_wp

    del b_ll, w_ll

    # West Kelowna — without parcels only
    print()
    wk_img = BASE_DIR / "west_kelowna_imagery"
    wk_masks = BASE_DIR / "west_kelowna_meta_masks"
    wk_metas = [m for m in [load_geotiff_meta_raw(p) for p in sorted(wk_img.glob("*.tif"))] if m]
    print(f"  West Kelowna: {len(wk_metas)} tiles")
    b_wk, wk_ext = stitch_lowres_masks(wk_metas, wk_masks, "building_masks")
    w_wk, _ = stitch_lowres_masks(wk_metas, wk_masks, "woodland_masks")
    print(f"  Canvas: {wk_ext['width']}x{wk_ext['height']}, bld={b_wk.sum()}, wld={w_wk.sum()}")

    wk_no = community_scores(b_wk, w_wk, wk_ext["gsd_x"], wk_ext["gsd_y"], "West Kelowna (no parcel)")
    all_results["west_kelowna_no_parcel"] = wk_no

    del b_wk, w_wk

    # ================================================================
    # SAVE + PRINT
    # ================================================================
    out_path = BASE_DIR / "eval_final" / "comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print parcel tables
    print("\n" + "="*80)
    print("4 CUMBERLAND PARCELS — HIGH-RES (DRONE/CNN) vs LOW-RES (SATELLITE/MS+META)")
    print("="*80)
    print(f"{'PID':<14} {'Pipeline':<12} {'Boundary':<14} {'Risk':>6} {'Z1a':>7} {'Z1b':>7} {'Z2':>7} {'Z3':>7} {'MinDist':>8}")
    print("-" * 85)
    for pipeline, label, data in [("High-Res", "highres_parcels", hr_parcels), ("Low-Res", "lowres_parcels", lr_parcels)]:
        for r in sorted(data, key=lambda x: x["pid"]):
            for ml, mk in [("No Parcel", "without_parcel"), ("With Parcel", "with_parcel")]:
                d = r[mk]
                if d["risk_score"] is not None:
                    print(f"{r['pid']:<14} {pipeline:<12} {ml:<14} {d['risk_score']:>6.2f} {d['zone_1a']:>7.4f} {d['zone_1b']:>7.4f} {d['zone_2']:>7.4f} {d['zone_3']:>7.4f} {d['min_veg_dist']:>8.2f}")
        print()

    # Community table
    print("="*90)
    print("COMMUNITY RISK SCORES")
    print("="*90)
    print(f"{'Community':<30} {'Buildings':>9} {'WithVeg':>8} {'Mean':>8} {'Filt.Mean':>10} {'Median':>8} {'Max':>6}")
    print("-" * 85)
    for label, key in [
        ("Cumberland Drone (no parcel)", "cumberland_drone_no_parcel"),
        ("Cumberland Drone (w/ parcel)", "cumberland_drone_with_parcel"),
        ("Cumberland Sat (no parcel)", "cumberland_sat_no_parcel"),
        ("Cumberland Sat (w/ parcel)", "cumberland_sat_with_parcel"),
        ("Logan Lake (no parcel)", "logan_lake_no_parcel"),
        ("Logan Lake (w/ parcel)", "logan_lake_with_parcel"),
        ("West Kelowna (no parcel)", "west_kelowna_no_parcel"),
    ]:
        d = all_results[key]
        print(f"{label:<30} {d['total']:>9} {d['with_veg']:>8} {d['mean_all']:>8.3f} {d['mean_filtered']:>10.3f} {d['median']:>8.3f} {d['max']:>6.2f}")


if __name__ == "__main__":
    main()
