# Data sheet for building_risk_scores.csv

## Definitions

| CSV Field | Label | Type | Description |
|-----------|-------|------|-------------|
| `tile` | Tile ID | String | Filename identifier for the 640×640 px image tile from the Cumberland orthoimagery. Encodes the pixel row and column origin of the tile within the source GeoTIFF (e.g., `tile_000000_000576` starts at row 0, column 576). |
| `id` | Building ID | Integer | Zero-indexed identifier for each building detected within a given tile. Assigned by connected-component labeling on the binary building segmentation mask. IDs are unique within a tile only. |
| `centroid` | Centroid (px) | String (x, y) | Pixel coordinates of the building's geometric centroid within the 640×640 tile, formatted as (column, row). |
| `area_m2` | Building Area (m²) | Float | Building footprint area in square meters. Minimum threshold: 100 pixels (0.36 m²). |
| `min_veg_distance_m` | Min. Vegetation Distance (m) | Float | Shortest Euclidean distance in meters from the building edge to the nearest pixel classified as woodland/vegetation.|
| `risk_score` | Risk Score | Float (1–10) | Wildfire risk score combining a distance component (0–5 pts) and a density component (0–5 pts). Higher scores indicate greater risk. See Risk Scoring Methodology below. |
| `zone_1a_veg_density` | Zone 1a Vegetation Density | Float (0–1) | Fraction of pixels within FireSmart Zone 1a (0–1.5 m from building edge) classified as vegetation. |
| `zone_1b_veg_density` | Zone 1b Vegetation Density | Float (0–1) | Fraction of pixels within FireSmart Zone 1b (1.5–10 m from building edge) classified as vegetation. |
| `zone_2_veg_density` | Zone 2 Vegetation Density | Float (0–1) | Fraction of pixels within FireSmart Zone 2 (10–30 m from building edge) classified as vegetation. |
| `overlay_image` | Overlay Image | String (path) | Relative path to the risk overlay image for this tile. Shows FireSmart zones, vegetation, buildings, and risk scores. |

---

## FireSmart Zone Definitions

Zones are circular rings measured outward from each building's edge, following FireSmart Canada guidelines. Pixels are computed at 6 cm ground sample distance.


## Risk Scoring Methodology

Each building receives a score from 1 to 10, computed as the sum of a distance component and a density component, clamped to [1, 10].

### Distance Component (0–5 points)

Based on the shortest Euclidean distance from the building edge to the nearest vegetation pixel:

| Min. Vegetation Distance | Distance Points (0–5) |
|--------------------------|-----------------------|
| < 1.5 m (Zone 1a) | 5.0 |
| 1.5 – 10 m (Zone 1b) | 4.0 → 2.0 (linear) |
| 10 – 30 m (Zone 2) | 2.0 → 0.5 (linear) |
| > 30 m (Zone 3) | 0.5 |


### Density Component (0–5 points)

Weighted sum of vegetation density across all zones:

| Zone | Distance | Radius (px) | Description | Density Weight |
|------|----------|-------------|-------------|----------------|
| Zone 1a | 0 – 1.5 m | 25 px | Critical ignition zone. Direct flame contact with structure. | 2.5× |
| Zone 1b | 1.5 – 10 m | 166 px | Radiant heat and ember accumulation zone. | 1.5× |
| Zone 2 | 10 – 30 m | 500 px | Fuel management zone. Slows fire spread. | 0.75× |
| Zone 3 | > 30 m | — | Extended planning zone. | 0.25× |

### Risk score formula
```md
**Total Risk score = (Zone 1a density × 2.5) + (Zone 1b density × 1.5) + (Zone 2 density × 0.75) + (Zone 3 density × 0.25) + f(min. vegetation distance)**
```


---

## Special Values/Cases

**`min_veg_distance_m = 999.0`** — No vegetation detected within the tile boundary. Building receives minimum risk score of 1.0.

**All zone densities = 0.0** — No vegetation present in any zone ring around this building.

---

## Detection

Buildings were detected using a Feature Pyramid Network (FPN) with an EfficientNet-B3 encoder, trained on the combined LandCover.ai (Poland, 25–50 cm) and INRIA Aerial Image Labeling (North America, 30 cm) datasets. The model achieves 0.74 IoU on the held-out test set. Vegetation was detected using a separate FPN + EfficientNet-B3 model trained on the LandCover.ai woodland class, achieving 0.76 IoU. Post-processing includes erosion, shape-based filtering (aspect ratio and solidity), dilation, and closing operations to filter false positives (such as roads or small objects).