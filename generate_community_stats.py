"""
Generate Community-Level Statistics from Building Risk Scores
==============================================================
Works with building_risk_scores.csv (tile-level) output from firesmart_risk.py.
Produces comprehensive community statistics and a human-readable report.

Usage:
    python generate_community_stats.py --input ./logan_lake_firesmart/building_risk_scores.csv --name "Logan Lake"
    python generate_community_stats.py --input ./west_kelowna_firesmart/building_risk_scores.csv --name "West Kelowna"
"""

import argparse
import csv
import numpy as np
from pathlib import Path
from collections import Counter


def load_data(csv_path):
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ["area_m2", "min_veg_distance_m", "risk_score",
                        "zone_1a_veg_density", "zone_1b_veg_density",
                        "zone_2_veg_density", "zone_3_veg_density"]:
                val = row.get(key, "")
                row[key] = float(val) if val not in ("", None) else None
            # Normalize: combined CSVs use "id" instead of "tile"
            if "tile" not in row and "id" in row:
                row["tile"] = row["id"]
            rows.append(row)
    return rows


def compute_stats(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "std": None,
                "min": None, "max": None, "q25": None, "q75": None}
    arr = np.array(values)
    return {
        "count": len(arr),
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "q25": round(float(np.percentile(arr, 25)), 4),
        "q75": round(float(np.percentile(arr, 75)), 4),
    }


def classify_risk(score):
    if score is None:
        return "no_data"
    if score >= 7.0:
        return "high"
    elif score >= 4.0:
        return "moderate"
    else:
        return "low"


def classify_compliance(row):
    min_dist = row["min_veg_distance_m"]
    z1a = row["zone_1a_veg_density"]
    if min_dist is None:
        return "no_data"
    if min_dist >= 999.0:
        return "compliant"
    if z1a is not None and z1a > 0.0:
        return "non_compliant"
    elif min_dist < 1.5:
        return "non_compliant"
    elif min_dist < 10.0:
        return "partially_compliant"
    else:
        return "compliant"


def generate_stats(rows, output_dir, community_name):
    risk_scores = [r["risk_score"] for r in rows if r["risk_score"] is not None]
    min_dists = [r["min_veg_distance_m"] for r in rows if r["min_veg_distance_m"] is not None]
    areas = [r["area_m2"] for r in rows if r["area_m2"] is not None]
    z1a = [r["zone_1a_veg_density"] for r in rows if r["zone_1a_veg_density"] is not None]
    z1b = [r["zone_1b_veg_density"] for r in rows if r["zone_1b_veg_density"] is not None]
    z2 = [r["zone_2_veg_density"] for r in rows if r["zone_2_veg_density"] is not None]
    z3 = [r["zone_3_veg_density"] for r in rows if r.get("zone_3_veg_density") is not None]

    # Tiles with buildings
    tiles_with_buildings = len(set(r["tile"] for r in rows))

    # Risk categories
    risk_cats = Counter(classify_risk(r["risk_score"]) for r in rows)

    # Compliance
    compliance = Counter(classify_compliance(r) for r in rows)

    # Score distribution
    score_bins = {"1-2": 0, "2-3": 0, "3-4": 0, "4-5": 0, "5-6": 0,
                  "6-7": 0, "7-8": 0, "8-9": 0, "9-10": 0}
    for s in risk_scores:
        if s <= 2: score_bins["1-2"] += 1
        elif s <= 3: score_bins["2-3"] += 1
        elif s <= 4: score_bins["3-4"] += 1
        elif s <= 5: score_bins["4-5"] += 1
        elif s <= 6: score_bins["5-6"] += 1
        elif s <= 7: score_bins["6-7"] += 1
        elif s <= 8: score_bins["7-8"] += 1
        elif s <= 9: score_bins["8-9"] += 1
        else: score_bins["9-10"] += 1

    # Distance distribution
    dist_bins = {
        "0-1.5m (Zone 1a)": 0,
        "1.5-10m (Zone 1b)": 0,
        "10-30m (Zone 2)": 0,
        "30m+ (Zone 3)": 0,
        "No vegetation": 0,
    }
    for d in min_dists:
        if d >= 999.0: dist_bins["No vegetation"] += 1
        elif d < 1.5: dist_bins["0-1.5m (Zone 1a)"] += 1
        elif d < 10.0: dist_bins["1.5-10m (Zone 1b)"] += 1
        elif d < 30.0: dist_bins["10-30m (Zone 2)"] += 1
        else: dist_bins["30m+ (Zone 3)"] += 1

    # Building size distribution
    size_bins = {
        "Small (<50 m²)": 0,
        "Medium (50-200 m²)": 0,
        "Large (200-500 m²)": 0,
        "Very large (>500 m²)": 0,
    }
    for a in areas:
        if a < 50: size_bins["Small (<50 m²)"] += 1
        elif a < 200: size_bins["Medium (50-200 m²)"] += 1
        elif a < 500: size_bins["Large (200-500 m²)"] += 1
        else: size_bins["Very large (>500 m²)"] += 1

    # Zone density non-zero counts
    z1a_nonzero = [v for v in z1a if v > 0]
    z1b_nonzero = [v for v in z1b if v > 0]
    z2_nonzero = [v for v in z2 if v > 0]
    z3_nonzero = [v for v in z3 if v > 0]

    # Stats objects
    rs = compute_stats(risk_scores)
    ds = compute_stats(min_dists)
    as_ = compute_stats(areas)
    z1a_s = compute_stats(z1a)
    z1b_s = compute_stats(z1b)
    z2_s = compute_stats(z2)
    z3_s = compute_stats(z3)

    # Compliance totals
    compliant = compliance.get("compliant", 0)
    partial = compliance.get("partially_compliant", 0)
    non_compliant = compliance.get("non_compliant", 0)
    total_compliance = compliant + partial + non_compliant

    # ── Write CSV ──
    csv_path = output_dir / "community_statistics.csv"
    stats_rows = []

    def add(cat, metric, value):
        stats_rows.append({"category": cat, "metric": metric, "value": value})

    add("overview", "community_name", community_name)
    add("overview", "total_buildings", len(rows))
    add("overview", "tiles_with_buildings", tiles_with_buildings)
    add("overview", "total_building_area_m2", round(sum(areas), 1))
    add("overview", "avg_building_area_m2", round(np.mean(areas), 1) if areas else "")

    for k, v in rs.items():
        add("risk_score", k, v if v is not None else "")
    for cat in ["high", "moderate", "low"]:
        c = risk_cats.get(cat, 0)
        pct = round(c / len(rows) * 100, 1) if rows else 0
        add("risk_category", f"{cat}_count", c)
        add("risk_category", f"{cat}_percent", pct)
    for b, c in score_bins.items():
        add("risk_score_distribution", f"score_{b}", c)

    add("compliance", "compliant_count", compliant)
    add("compliance", "compliant_percent", round(compliant / total_compliance * 100, 1) if total_compliance else 0)
    add("compliance", "partially_compliant_count", partial)
    add("compliance", "partially_compliant_percent", round(partial / total_compliance * 100, 1) if total_compliance else 0)
    add("compliance", "non_compliant_count", non_compliant)
    add("compliance", "non_compliant_percent", round(non_compliant / total_compliance * 100, 1) if total_compliance else 0)

    for k, v in ds.items():
        add("vegetation_distance", k, v if v is not None else "")
    for b, c in dist_bins.items():
        add("vegetation_distance_distribution", b, c)

    for zname, zstats, znonzero, ztotal in [
        ("zone_1a", z1a_s, z1a_nonzero, z1a),
        ("zone_1b", z1b_s, z1b_nonzero, z1b),
        ("zone_2", z2_s, z2_nonzero, z2),
        ("zone_3", z3_s, z3_nonzero, z3),
    ]:
        for k, v in zstats.items():
            add(f"{zname}_density", k, v if v is not None else "")
        add(f"{zname}_density", "buildings_with_vegetation", len(znonzero))
        add(f"{zname}_density", "pct_with_vegetation",
            round(len(znonzero) / len(ztotal) * 100, 1) if ztotal else 0)

    for b, c in size_bins.items():
        add("building_size_distribution", b, c)
    for k, v in as_.items():
        add("building_area", k, v if v is not None else "")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "metric", "value"])
        writer.writeheader()
        writer.writerows(stats_rows)

    # ── Write Report ──
    report_path = output_dir / "community_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"FIRESMART COMMUNITY RISK ASSESSMENT — {community_name.upper()}\n")
        f.write("=" * 70 + "\n\n")

        f.write("OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Community:                {community_name}\n")
        f.write(f"  Total buildings analysed: {len(rows):>8}\n")
        f.write(f"  Tiles with buildings:     {tiles_with_buildings:>8}\n")
        f.write(f"  Total building area:      {sum(areas):>10.1f} m²\n")
        f.write(f"  Avg building footprint:   {np.mean(areas):>10.1f} m²\n" if areas else "")
        f.write("\n")

        f.write("RISK SCORE SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Mean:     {rs['mean']:>8}\n")
        f.write(f"  Median:   {rs['median']:>8}\n")
        f.write(f"  Std Dev:  {rs['std']:>8}\n")
        f.write(f"  Min:      {rs['min']:>8}    Max: {rs['max']}\n")
        f.write(f"  Q25:      {rs['q25']:>8}    Q75: {rs['q75']}\n")
        f.write("\n")

        f.write("RISK CATEGORIES\n")
        f.write("-" * 40 + "\n")
        for cat in ["high", "moderate", "low"]:
            c = risk_cats.get(cat, 0)
            pct = round(c / len(rows) * 100, 1) if rows else 0
            label = {"high": "High (7-10)", "moderate": "Moderate (4-7)", "low": "Low (1-4)"}[cat]
            f.write(f"  {label:<20s} {c:>6} ({pct:>5.1f}%)\n")
        f.write("\n")

        f.write("RISK SCORE DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        max_bin = max(score_bins.values()) if score_bins.values() else 1
        for b, c in score_bins.items():
            bar = "#" * int(c / max(max_bin, 1) * 30)
            f.write(f"  Score {b:<6s} {c:>6}  {bar}\n")
        f.write("\n")

        f.write("FIRESMART COMPLIANCE\n")
        f.write("-" * 40 + "\n")
        if total_compliance > 0:
            f.write(f"  Compliant (veg >10m):          {compliant:>6} ({compliant/total_compliance*100:>5.1f}%)\n")
            f.write(f"  Partially compliant (1.5-10m): {partial:>6} ({partial/total_compliance*100:>5.1f}%)\n")
            f.write(f"  Non-compliant (veg <1.5m):     {non_compliant:>6} ({non_compliant/total_compliance*100:>5.1f}%)\n")
        f.write("\n")

        f.write("VEGETATION DISTANCE FROM BUILDINGS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Mean distance:    {ds['mean']:>8} m\n")
        f.write(f"  Median distance:  {ds['median']:>8} m\n")
        f.write(f"  Min distance:     {ds['min']:>8} m\n")
        f.write(f"  Max distance:     {ds['max']:>8} m\n")
        f.write("\n")
        f.write(f"  Distance distribution:\n")
        for b, c in dist_bins.items():
            pct = round(c / len(min_dists) * 100, 1) if min_dists else 0
            f.write(f"    {b:<25s} {c:>6} ({pct:>5.1f}%)\n")
        f.write("\n")

        f.write("VEGETATION DENSITY BY ZONE\n")
        f.write("-" * 40 + "\n")
        for zlabel, zstats, znz, ztotal in [
            ("Zone 1a (0-1.5m)", z1a_s, z1a_nonzero, z1a),
            ("Zone 1b (1.5-10m)", z1b_s, z1b_nonzero, z1b),
            ("Zone 2  (10-30m)", z2_s, z2_nonzero, z2),
            ("Zone 3  (30m+)", z3_s, z3_nonzero, z3),
        ]:
            pct_veg = round(len(znz) / len(ztotal) * 100, 1) if ztotal else 0
            f.write(f"  {zlabel}:\n")
            f.write(f"    Mean density:          {zstats['mean']:>8}\n")
            f.write(f"    Median density:        {zstats['median']:>8}\n")
            f.write(f"    Buildings with veg:    {len(znz):>6} ({pct_veg}%)\n")
            f.write(f"    Buildings without veg: {len(ztotal) - len(znz):>6}\n")
        f.write("\n")

        f.write("BUILDING SIZE DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        for b, c in size_bins.items():
            pct = round(c / len(areas) * 100, 1) if areas else 0
            f.write(f"  {b:<25s} {c:>6} ({pct:>5.1f}%)\n")
        f.write(f"\n  Mean footprint:   {as_['mean']:>8} m²\n")
        f.write(f"  Median footprint: {as_['median']:>8} m²\n")
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    print(f"\nStatistics CSV:  {csv_path}")
    print(f"Summary report:  {report_path}\n")

    with open(report_path) as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--name", type=str, default="Community")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return

    rows = load_data(input_path)
    print(f"Loaded {len(rows)} buildings from {input_path}")
    generate_stats(rows, input_path.parent, args.name)


if __name__ == "__main__":
    main()
