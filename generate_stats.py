"""
Generate Statistical Summary for FireSmart Property Risk Assessment
====================================================================
Reads parcel_risk_scores.csv and produces a comprehensive statistical
summary CSV and human-readable report.

Usage:
    python generate_stats.py --input ./firesmart_property_results/parcel_risk_scores.csv
    python generate_stats.py --input ./west_kelowna_results/parcel_risk_scores.csv

Output (in same directory as input):
    community_statistics.csv   — machine-readable stats
    community_report.txt       — human-readable summary report
"""

import argparse
import csv
import json
import numpy as np
from pathlib import Path
from collections import Counter


def load_data(csv_path):
    """Load parcel risk scores CSV."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in ["parcel_area_m2", "num_buildings", "max_risk_score",
                        "mean_risk_score", "min_veg_distance_m",
                        "min_veg_dist_on_parcel", "min_veg_dist_off_parcel",
                        "zone_1a_veg_density", "zone_1b_veg_density",
                        "zone_2_veg_density",
                        "zone_1a_on_parcel", "zone_1a_off_parcel",
                        "zone_1b_on_parcel", "zone_1b_off_parcel",
                        "zone_2_on_parcel", "zone_2_off_parcel"]:
                val = row.get(key, "")
                if val == "" or val is None:
                    row[key] = None
                else:
                    row[key] = float(val)
            if row["num_buildings"] is not None:
                row["num_buildings"] = int(row["num_buildings"])
            rows.append(row)
    return rows


def compute_stats(values, label=""):
    """Compute summary statistics for a list of numeric values."""
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
    """Classify risk score into category."""
    if score is None:
        return "no_data"
    if score >= 7.0:
        return "high"
    elif score >= 4.0:
        return "moderate"
    else:
        return "low"


def classify_compliance(row):
    """
    Classify FireSmart compliance based on vegetation proximity.
    A property is 'non-compliant' if vegetation is within Zone 1a (<1.5m)
    of any building. 'Partially compliant' if vegetation is in Zone 1b
    but not 1a. 'Compliant' if no vegetation within 10m.
    """
    if row["num_buildings"] is None or row["num_buildings"] == 0:
        return "no_buildings"

    min_dist = row["min_veg_distance_m"]
    z1a = row["zone_1a_veg_density"]

    if min_dist is None:
        return "no_data"

    if min_dist == 999.0:
        return "compliant"

    if z1a is not None and z1a > 0.0:
        return "non_compliant"
    elif min_dist < 1.5:
        return "non_compliant"
    elif min_dist < 10.0:
        return "partially_compliant"
    else:
        return "compliant"


def classify_compliance_on_parcel(row):
    """Compliance based only on vegetation within the property boundary."""
    if row["num_buildings"] is None or row["num_buildings"] == 0:
        return "no_buildings"
    min_dist = row.get("min_veg_dist_on_parcel")
    z1a = row.get("zone_1a_on_parcel")
    if min_dist is None:
        return "no_data"
    if min_dist == 999.0:
        return "compliant"
    if z1a is not None and z1a > 0.0:
        return "non_compliant"
    elif min_dist < 1.5:
        return "non_compliant"
    elif min_dist < 10.0:
        return "partially_compliant"
    else:
        return "compliant"


def classify_compliance_off_parcel(row):
    """Compliance based only on vegetation outside the property boundary."""
    if row["num_buildings"] is None or row["num_buildings"] == 0:
        return "no_buildings"
    min_dist = row.get("min_veg_dist_off_parcel")
    z1a = row.get("zone_1a_off_parcel")
    if min_dist is None:
        return "no_data"
    if min_dist == 999.0:
        return "compliant"
    if z1a is not None and z1a > 0.0:
        return "non_compliant"
    elif min_dist < 1.5:
        return "non_compliant"
    elif min_dist < 10.0:
        return "partially_compliant"
    else:
        return "compliant"


def generate_stats(rows, output_dir):
    """Generate all statistics and save to files."""

    # ── Partition rows ──
    all_parcels = rows
    with_buildings = [r for r in rows if r["num_buildings"] and r["num_buildings"] > 0]
    no_buildings = [r for r in rows if r["num_buildings"] is not None and r["num_buildings"] == 0]

    risk_scores = [r["max_risk_score"] for r in with_buildings if r["max_risk_score"] is not None]
    mean_scores = [r["mean_risk_score"] for r in with_buildings if r["mean_risk_score"] is not None]
    min_dists = [r["min_veg_distance_m"] for r in with_buildings if r["min_veg_distance_m"] is not None]
    z1a_densities = [r["zone_1a_veg_density"] for r in with_buildings if r["zone_1a_veg_density"] is not None]
    z1b_densities = [r["zone_1b_veg_density"] for r in with_buildings if r["zone_1b_veg_density"] is not None]
    z2_densities = [r["zone_2_veg_density"] for r in with_buildings if r["zone_2_veg_density"] is not None]
    building_counts = [r["num_buildings"] for r in with_buildings]
    parcel_areas = [r["parcel_area_m2"] for r in all_parcels if r["parcel_area_m2"] is not None]

    # ── Risk distribution ──
    risk_categories = Counter(classify_risk(r["max_risk_score"]) for r in with_buildings)

    # ── Compliance ──
    compliance = Counter(classify_compliance(r) for r in all_parcels)

    # ── Owner type breakdown ──
    owner_types = Counter(r.get("owner_type", "Unknown") for r in all_parcels)
    owner_risk = {}
    for otype in owner_types:
        scores = [r["max_risk_score"] for r in with_buildings
                  if r.get("owner_type") == otype and r["max_risk_score"] is not None]
        if scores:
            owner_risk[otype] = compute_stats(scores)

    # ── On/off parcel data ──
    has_on_off = any(r.get("min_veg_dist_on_parcel") is not None for r in with_buildings)

    if has_on_off:
        min_dists_on = [r["min_veg_dist_on_parcel"] for r in with_buildings
                        if r.get("min_veg_dist_on_parcel") is not None]
        min_dists_off = [r["min_veg_dist_off_parcel"] for r in with_buildings
                         if r.get("min_veg_dist_off_parcel") is not None]
        z1a_on = [r["zone_1a_on_parcel"] for r in with_buildings if r.get("zone_1a_on_parcel") is not None]
        z1a_off = [r["zone_1a_off_parcel"] for r in with_buildings if r.get("zone_1a_off_parcel") is not None]
        z1b_on = [r["zone_1b_on_parcel"] for r in with_buildings if r.get("zone_1b_on_parcel") is not None]
        z1b_off = [r["zone_1b_off_parcel"] for r in with_buildings if r.get("zone_1b_off_parcel") is not None]
        z2_on = [r["zone_2_on_parcel"] for r in with_buildings if r.get("zone_2_on_parcel") is not None]
        z2_off = [r["zone_2_off_parcel"] for r in with_buildings if r.get("zone_2_off_parcel") is not None]

        compliance_on = Counter(classify_compliance_on_parcel(r) for r in all_parcels)
        compliance_off = Counter(classify_compliance_off_parcel(r) for r in all_parcels)

    # ── Zone density stats ──
    # Properties with vegetation in each zone
    z1a_nonzero = [v for v in z1a_densities if v > 0]
    z1b_nonzero = [v for v in z1b_densities if v > 0]
    z2_nonzero = [v for v in z2_densities if v > 0]

    # ── Risk score distribution (histogram bins) ──
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

    # ── Distance distribution ──
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

    # ══════════════════════════════════════════════════════════
    # WRITE STATISTICS CSV
    # ══════════════════════════════════════════════════════════
    stats_rows = []

    def add_stat(category, metric, value):
        stats_rows.append({"category": category, "metric": metric, "value": value})

    # Overview
    add_stat("overview", "total_parcels_in_csv", len(all_parcels))
    add_stat("overview", "parcels_with_buildings", len(with_buildings))
    add_stat("overview", "parcels_no_buildings", len(no_buildings))
    add_stat("overview", "total_buildings_detected", sum(building_counts))
    add_stat("overview", "avg_buildings_per_property", round(np.mean(building_counts), 2) if building_counts else "")
    add_stat("overview", "max_buildings_on_single_property", max(building_counts) if building_counts else "")
    add_stat("overview", "properties_with_1_building", sum(1 for b in building_counts if b == 1))
    add_stat("overview", "properties_with_2plus_buildings", sum(1 for b in building_counts if b >= 2))

    # Risk score stats
    rs = compute_stats(risk_scores)
    for k, v in rs.items():
        add_stat("risk_score_max", k, v if v is not None else "")
    ms = compute_stats(mean_scores)
    for k, v in ms.items():
        add_stat("risk_score_mean", k, v if v is not None else "")

    # Risk categories
    for cat in ["high", "moderate", "low"]:
        count = risk_categories.get(cat, 0)
        pct = round(count / len(with_buildings) * 100, 1) if with_buildings else 0
        add_stat("risk_category", f"{cat}_count", count)
        add_stat("risk_category", f"{cat}_percent", pct)

    # Score distribution
    for bin_label, count in score_bins.items():
        add_stat("risk_score_distribution", f"score_{bin_label}", count)

    # Compliance
    for status in ["compliant", "partially_compliant", "non_compliant", "no_buildings"]:
        count = compliance.get(status, 0)
        add_stat("compliance", f"{status}_count", count)
        add_stat("compliance", f"{status}_percent",
                 round(count / len(all_parcels) * 100, 1) if all_parcels else 0)

    # Compliance among properties WITH buildings only
    compliant_bld = compliance.get("compliant", 0)
    partial_bld = compliance.get("partially_compliant", 0)
    non_compliant_bld = compliance.get("non_compliant", 0)
    total_bld_compliance = compliant_bld + partial_bld + non_compliant_bld
    if total_bld_compliance > 0:
        add_stat("compliance_buildings_only", "compliant_count", compliant_bld)
        add_stat("compliance_buildings_only", "compliant_percent", round(compliant_bld / total_bld_compliance * 100, 1))
        add_stat("compliance_buildings_only", "partially_compliant_count", partial_bld)
        add_stat("compliance_buildings_only", "partially_compliant_percent", round(partial_bld / total_bld_compliance * 100, 1))
        add_stat("compliance_buildings_only", "non_compliant_count", non_compliant_bld)
        add_stat("compliance_buildings_only", "non_compliant_percent", round(non_compliant_bld / total_bld_compliance * 100, 1))

    # Vegetation distance stats
    ds = compute_stats(min_dists)
    for k, v in ds.items():
        add_stat("vegetation_distance", k, v if v is not None else "")

    # Distance distribution
    for bin_label, count in dist_bins.items():
        add_stat("vegetation_distance_distribution", bin_label, count)

    # Zone density stats
    for zone_name, densities in [("zone_1a", z1a_densities), ("zone_1b", z1b_densities), ("zone_2", z2_densities)]:
        zs = compute_stats(densities)
        for k, v in zs.items():
            add_stat(f"{zone_name}_density", k, v if v is not None else "")
        nonzero = [d for d in densities if d > 0]
        add_stat(f"{zone_name}_density", "properties_with_vegetation", len(nonzero))
        add_stat(f"{zone_name}_density", "properties_without_vegetation", len(densities) - len(nonzero))
        add_stat(f"{zone_name}_density", "pct_with_vegetation",
                 round(len(nonzero) / len(densities) * 100, 1) if densities else 0)

    # On/off parcel stats
    if has_on_off:
        # On-parcel compliance
        for status in ["compliant", "partially_compliant", "non_compliant"]:
            count_on = compliance_on.get(status, 0)
            count_off = compliance_off.get(status, 0)
            add_stat("compliance_on_parcel", f"{status}_count", count_on)
            add_stat("compliance_on_parcel", f"{status}_percent",
                     round(count_on / total_bld_compliance * 100, 1) if total_bld_compliance > 0 else 0)
            add_stat("compliance_off_parcel", f"{status}_count", count_off)
            add_stat("compliance_off_parcel", f"{status}_percent",
                     round(count_off / total_bld_compliance * 100, 1) if total_bld_compliance > 0 else 0)

        # On/off parcel distance stats
        ds_on = compute_stats([d for d in min_dists_on if d < 999])
        ds_off = compute_stats([d for d in min_dists_off if d < 999])
        for k, v in ds_on.items():
            add_stat("veg_distance_on_parcel", k, v if v is not None else "")
        for k, v in ds_off.items():
            add_stat("veg_distance_off_parcel", k, v if v is not None else "")

        # On/off parcel zone density
        for zone_name, on_list, off_list in [("zone_1a", z1a_on, z1a_off),
                                              ("zone_1b", z1b_on, z1b_off),
                                              ("zone_2", z2_on, z2_off)]:
            zs_on = compute_stats(on_list)
            zs_off = compute_stats(off_list)
            for k, v in zs_on.items():
                add_stat(f"{zone_name}_on_parcel", k, v if v is not None else "")
            for k, v in zs_off.items():
                add_stat(f"{zone_name}_off_parcel", k, v if v is not None else "")

    # Parcel area stats
    pa = compute_stats(parcel_areas)
    for k, v in pa.items():
        add_stat("parcel_area_m2", k, v if v is not None else "")

    # Owner type breakdown
    for otype, count in sorted(owner_types.items(), key=lambda x: -x[1]):
        add_stat("owner_type", f"{otype}_count", count)
        add_stat("owner_type", f"{otype}_percent",
                 round(count / len(all_parcels) * 100, 1))
        if otype in owner_risk:
            add_stat("owner_type", f"{otype}_mean_risk", owner_risk[otype]["mean"])
            add_stat("owner_type", f"{otype}_median_risk", owner_risk[otype]["median"])

    # Write CSV
    csv_path = output_dir / "community_statistics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "metric", "value"])
        writer.writeheader()
        writer.writerows(stats_rows)

    # ══════════════════════════════════════════════════════════
    # WRITE HUMAN-READABLE REPORT
    # ══════════════════════════════════════════════════════════
    report_path = output_dir / "community_report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("FIRESMART PROPERTY RISK ASSESSMENT — STATISTICAL SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("OVERVIEW\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total parcels analysed:       {len(all_parcels):>8}\n")
        f.write(f"  Parcels with buildings:        {len(with_buildings):>8}\n")
        f.write(f"  Parcels without buildings:     {len(no_buildings):>8}\n")
        f.write(f"  Total buildings detected:      {sum(building_counts):>8}\n")
        f.write(f"  Avg buildings per property:    {np.mean(building_counts):>8.2f}\n" if building_counts else "")
        f.write(f"  Max buildings (single parcel): {max(building_counts):>8}\n" if building_counts else "")
        f.write(f"  Properties with 1 building:    {sum(1 for b in building_counts if b == 1):>8}\n")
        f.write(f"  Properties with 2+ buildings:  {sum(1 for b in building_counts if b >= 2):>8}\n")
        f.write("\n")

        f.write("RISK SCORE SUMMARY (properties with buildings)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Max risk score (per-property):\n")
        f.write(f"    Mean:     {rs['mean']:>8}\n")
        f.write(f"    Median:   {rs['median']:>8}\n")
        f.write(f"    Std Dev:  {rs['std']:>8}\n")
        f.write(f"    Min:      {rs['min']:>8}\n")
        f.write(f"    Max:      {rs['max']:>8}\n")
        f.write(f"    Q25:      {rs['q25']:>8}\n")
        f.write(f"    Q75:      {rs['q75']:>8}\n")
        f.write("\n")

        f.write("RISK CATEGORIES\n")
        f.write("-" * 40 + "\n")
        for cat in ["high", "moderate", "low"]:
            count = risk_categories.get(cat, 0)
            pct = round(count / len(with_buildings) * 100, 1) if with_buildings else 0
            label = {"high": "High (7-10)", "moderate": "Moderate (4-7)", "low": "Low (1-4)"}[cat]
            f.write(f"  {label:<20s} {count:>6} ({pct:>5.1f}%)\n")
        f.write("\n")

        f.write("RISK SCORE DISTRIBUTION\n")
        f.write("-" * 40 + "\n")
        for bin_label, count in score_bins.items():
            bar = "#" * int(count / max(max(score_bins.values()), 1) * 30)
            f.write(f"  Score {bin_label:<6s} {count:>6}  {bar}\n")
        f.write("\n")

        f.write("FIRESMART COMPLIANCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"  (Among {total_bld_compliance} properties with buildings)\n")
        if total_bld_compliance > 0:
            f.write(f"  Compliant (veg >10m):          {compliant_bld:>6} ({compliant_bld/total_bld_compliance*100:>5.1f}%)\n")
            f.write(f"  Partially compliant (1.5-10m): {partial_bld:>6} ({partial_bld/total_bld_compliance*100:>5.1f}%)\n")
            f.write(f"  Non-compliant (veg <1.5m):     {non_compliant_bld:>6} ({non_compliant_bld/total_bld_compliance*100:>5.1f}%)\n")
        f.write("\n")

        if has_on_off:
            f.write("FIRESMART COMPLIANCE — ON-PARCEL vs OFF-PARCEL\n")
            f.write("-" * 40 + "\n")
            f.write("  (Vegetation the property owner controls vs neighbouring/public land)\n\n")

            comp_on_total = (compliance_on.get("compliant", 0) +
                             compliance_on.get("partially_compliant", 0) +
                             compliance_on.get("non_compliant", 0))
            comp_off_total = (compliance_off.get("compliant", 0) +
                              compliance_off.get("partially_compliant", 0) +
                              compliance_off.get("non_compliant", 0))

            f.write(f"  ON-PARCEL (owner's property):\n")
            if comp_on_total > 0:
                for status, label in [("compliant", "Compliant (veg >10m)"),
                                      ("partially_compliant", "Partially compliant (1.5-10m)"),
                                      ("non_compliant", "Non-compliant (veg <1.5m)")]:
                    c = compliance_on.get(status, 0)
                    f.write(f"    {label:<35s} {c:>6} ({c/comp_on_total*100:>5.1f}%)\n")

            f.write(f"\n  OFF-PARCEL (neighbouring/public land):\n")
            if comp_off_total > 0:
                for status, label in [("compliant", "Compliant (veg >10m)"),
                                      ("partially_compliant", "Partially compliant (1.5-10m)"),
                                      ("non_compliant", "Non-compliant (veg <1.5m)")]:
                    c = compliance_off.get(status, 0)
                    f.write(f"    {label:<35s} {c:>6} ({c/comp_off_total*100:>5.1f}%)\n")

            # On vs off parcel distance summary
            dists_on_filtered = [d for d in min_dists_on if d < 999]
            dists_off_filtered = [d for d in min_dists_off if d < 999]
            f.write(f"\n  Nearest vegetation distance:\n")
            f.write(f"    {'':30s} {'On-parcel':>12s} {'Off-parcel':>12s}\n")
            ds_on = compute_stats(dists_on_filtered)
            ds_off = compute_stats(dists_off_filtered)
            for metric in ["mean", "median", "min"]:
                v_on = f"{ds_on[metric]:.2f}m" if ds_on[metric] is not None else "N/A"
                v_off = f"{ds_off[metric]:.2f}m" if ds_off[metric] is not None else "N/A"
                f.write(f"    {metric.capitalize():<30s} {v_on:>12s} {v_off:>12s}\n")

            # No on-parcel veg vs no off-parcel veg
            no_on = sum(1 for d in min_dists_on if d >= 999)
            no_off = sum(1 for d in min_dists_off if d >= 999)
            f.write(f"    {'No vegetation nearby':<30s} {no_on:>12d} {no_off:>12d}\n")
            f.write("\n")

        f.write("VEGETATION DISTANCE FROM BUILDINGS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Mean distance:    {ds['mean']:>8} m\n")
        f.write(f"  Median distance:  {ds['median']:>8} m\n")
        f.write(f"  Min distance:     {ds['min']:>8} m\n")
        f.write(f"  Max distance:     {ds['max']:>8} m\n")
        f.write("\n")
        f.write(f"  Distance distribution:\n")
        for bin_label, count in dist_bins.items():
            pct = round(count / len(min_dists) * 100, 1) if min_dists else 0
            f.write(f"    {bin_label:<25s} {count:>6} ({pct:>5.1f}%)\n")
        f.write("\n")

        f.write("VEGETATION DENSITY BY ZONE\n")
        f.write("-" * 40 + "\n")
        for zone_label, densities, nonzero_list in [
            ("Zone 1a (0-1.5m)", z1a_densities, z1a_nonzero),
            ("Zone 1b (1.5-10m)", z1b_densities, z1b_nonzero),
            ("Zone 2  (10-30m)", z2_densities, z2_nonzero),
        ]:
            zs = compute_stats(densities)
            nz_pct = round(len(nonzero_list) / len(densities) * 100, 1) if densities else 0
            f.write(f"  {zone_label}:\n")
            f.write(f"    Mean density:            {zs['mean']:>8}\n")
            f.write(f"    Median density:          {zs['median']:>8}\n")
            f.write(f"    Properties with veg:     {len(nonzero_list):>6} ({nz_pct}%)\n")
            f.write(f"    Properties without veg:  {len(densities) - len(nonzero_list):>6}\n")
        f.write("\n")

        f.write("PARCEL AREA\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Mean area:    {pa['mean']:>10} m²\n")
        f.write(f"  Median area:  {pa['median']:>10} m²\n")
        f.write(f"  Min area:     {pa['min']:>10} m²\n")
        f.write(f"  Max area:     {pa['max']:>10} m²\n")
        f.write("\n")

        f.write("OWNER TYPE BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        for otype, count in sorted(owner_types.items(), key=lambda x: -x[1]):
            pct = round(count / len(all_parcels) * 100, 1)
            risk_str = ""
            if otype in owner_risk:
                risk_str = f"  mean risk={owner_risk[otype]['mean']}"
            f.write(f"  {otype:<25s} {count:>6} ({pct:>5.1f}%){risk_str}\n")
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    print(f"Statistics CSV:  {csv_path}")
    print(f"Summary report:  {report_path}")

    # Also print the report to stdout
    with open(report_path) as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser(description="Generate FireSmart statistical summary")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to parcel_risk_scores.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return

    output_dir = input_path.parent
    rows = load_data(input_path)
    print(f"Loaded {len(rows)} parcels from {input_path}\n")

    generate_stats(rows, output_dir)


if __name__ == "__main__":
    main()
