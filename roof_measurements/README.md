# roof-measurements

Extract roof geometry from LiDAR LAS/LAZ point clouds — facet pitch, azimuth, area, eave height, ridge height, and more. Supports single-building files and city-scale tiles with automatic OSM footprint clipping.

---

## What It Does

Given a LAS/LAZ file, the pipeline:

1. Loads and classifies building points (ASPRS class 6, height-based fallback, or CSF ground filter)
2. Segments roof planes using iterative RANSAC with a region-growing fallback for complex roofs
3. Extracts per-facet geometry (pitch, azimuth, area, eave elevation) and building-level metrics (ridge height, eave height)
4. For city tiles: fetches OSM building footprints, clips the point cloud per building, and exports GeoJSON + CSV

---

## Installation

```bash
pip install -e ".[dev]"
```

**Core dependencies:** `laspy[lazrs]`, `pyransac3d`, `numpy`, `scipy`, `scikit-learn`, `shapely`, `pydantic`, `click`

**Optional (city tiles):** `geopandas`, `osmnx`, `pyproj`

**Optional (unclassified point clouds):** `CSF`

---

## Quick Start

### Lat/lon → automatic download + measurement

```bash
roof-measure query 30.2672 -97.7431          # Austin, TX
roof-measure query 40.7128 -74.0060          # New York, NY
roof-measure query 37.7749 -122.4194 -v      # San Francisco, CA
```

Downloads the USGS 3DEP LiDAR tile, fetches the OSM footprint, clips to the
building, and runs the full pipeline in one step.  Tiles are cached in
`~/.cache/roof_measurements/lidar/` so repeat queries are instant.

### Single building file → JSON

```bash
roof-measure process building.laz --output results.json
roof-measure process scan.las --min-facet-area 2.0 -v
```

### City tile → GeoJSON + CSV

```bash
roof-measure tile austin.laz --epsg 32614
roof-measure tile nyc_sandy.laz --epsg 32618 --out ./results -v
```

### Batch script (programmatic)

```bash
python scripts/batch_process.py data/austin_buildings_sample.laz --epsg 32614
python scripts/batch_process.py data/nyc_sandy.laz --epsg 32618 --max-buildings 10 -v
```

---

## Architecture

```
roof_measurements/
├── constants.py      — ASPRS classification codes (class 2 ground, class 6 building)
├── io.py             — LAS/LAZ loading + building point extraction
├── segmentation.py   — RANSAC plane segmentation + region-growing fallback
├── features.py       — Per-facet geometry extraction (pitch, azimuth, area, eave)
├── pipeline.py       — Top-level orchestration (process_file, process_building)
├── footprints.py     — OSM footprint fetching + per-building LiDAR clipping
├── datasources.py    — USGS 3DEP tile search, download, CRS extraction
├── export.py         — GeoDataFrame, GeoJSON, and CSV export
├── models.py         — Pydantic output models (FacetResult, BuildingResult)
└── cli.py            — Click CLI (process / tile / query subcommands)

scripts/
└── batch_process.py  — Standalone batch runner with progress table + multi-format export
```

### Data flow

```
LAS/LAZ file
    │
    ▼
io.load_building_points()
    │  ├─ Class 6 (building) → direct use
    │  ├─ Class 2 (ground) only → height-based filter (≥ 2.5m above ground)
    │  └─ No classification → CSF cloth simulation filter
    │
    ▼
segmentation.segment_planes()
    │  ├─ Iterative RANSAC (pyransac3d) — dominant plane removal
    │  └─ Region growing fallback — if RANSAC finds only 1 facet on complex roof
    │
    ▼
features.compute_facet()  ← per facet
    │  ├─ fit_plane_normal()    — PCA SVD
    │  ├─ pitch_from_normal()   — degrees from horizontal
    │  ├─ azimuth_from_normal() — compass bearing of downslope direction
    │  ├─ projected_area()      — 2D convex hull on facet plane
    │  └─ compute_eave_elevation() — 2nd-percentile Z
    │
features.compute_height()  ← building level
    │  └─ 99th-percentile Z minus ground_z
    │
    ▼
BuildingResult (Pydantic)
    │
    ├─ JSON (single building)
    └─ GeoDataFrame → GeoJSON + CSV (city tile)
```

---

## Output Schema

### `BuildingResult`

| Field | Type | Description |
|---|---|---|
| `building_id` | str | OSM id or filename stem |
| `num_facets` | int | Detected roof planes |
| `height_m` | float | Ridge elevation minus ground (m) |
| `eave_height_m` | float | Lowest eave minus ground (m) |
| `ground_elevation_m` | float | Median ground Z beneath building (m) |
| `ridge_elevation_m` | float | Absolute 99th-percentile Z of roof points (m) |
| `point_density_m2` | float | LiDAR pts/m² (< 4 may reduce accuracy) |
| `segmentation_method` | str | `ransac` or `region_growing` |
| `facets` | list[FacetResult] | Per-facet details |

### `FacetResult`

| Field | Type | Description |
|---|---|---|
| `facet_id` | int | 1-indexed |
| `pitch_deg` | float | Slope angle (0 = flat, 90 = vertical) |
| `azimuth_deg` | float | Downslope compass bearing (0 = North, 90 = East) |
| `area_m2` | float | Projected planar area |
| `eave_elevation_m` | float | 2nd-percentile Z of facet points (absolute) |
| `normal_vector` | [nx, ny, nz] | Unit normal |
| `num_points` | int | Points assigned to facet |
| `is_flat` | bool | True if pitch < 5° |
| `plane_rms_m` | float | RMS of point-to-plane residuals (m). Lower = better fit |
| `confidence` | float | Planarity confidence 0–1. 1.0 = perfect plane, 0.0 = RMS ≥ 0.30 m |

---

## LiDAR Data Handling

The pipeline handles varying data quality through layered fallbacks:

### Classification fallback (`io.py`)

| Input data | Strategy |
|---|---|
| ASPRS class 6 (building) present | Use directly |
| Class 2 (ground) only | Height-based filter — keep non-noise points ≥ 2.5m above median ground |
| No classification | CSF cloth simulation filter (requires `pip install CSF`) |

### Point density

- **Warning** logged when density < 4 pts/m² — processing continues
- **Subsampling** to 100,000 points when input exceeds that (random, fixed seed)

### Terrain variation (city tiles)

- Each building's ground elevation is computed from class-2 points clipped to its OSM footprint polygon
- Falls back to tile-wide median when fewer than 3 local ground points are found
- Prevents negative eave heights on hilly terrain

### Noise filtering

- Facets below `--min-facet-area` (default 1.0 m²) are discarded post-segmentation

---

## CLI Reference

```
roof-measure [--verbose/-v] COMMAND

Commands:
  process   Single-building LAS/LAZ → JSON
  tile      City tile LAS/LAZ → GeoJSON + CSV (uses OSM footprints)
```

### `process`

```
roof-measure process INPUT_FILE [OPTIONS]

Options:
  -o, --output PATH           Output JSON (default: <stem>_roof.json)
  --building-id TEXT          Override building ID
  --min-facet-area FLOAT      Min facet area in m² [default: 1.0]
  --distance-threshold FLOAT  RANSAC inlier tolerance in m [default: 0.15]
  --max-planes INT            Max roof planes to extract [default: 20]
```

### `tile`

```
roof-measure tile INPUT_FILE --epsg INT [OPTIONS]

Options:
  --epsg INT                  EPSG code of LAS file's CRS (required)
  -o, --out PATH              Output directory
  --min-points INT            Skip buildings with fewer points [default: 50]
  --geojson/--no-geojson      Write GeoJSON output [default: yes]
  --csv/--no-csv              Write CSV output [default: yes]
  --min-facet-area FLOAT      [default: 1.0]
  --distance-threshold FLOAT  [default: 0.15]
  --max-planes INT            [default: 20]
```

---

## Python API

```python
from roof_measurements.pipeline import process_file, process_building
from roof_measurements.footprints import iter_building_point_clouds
from roof_measurements.export import results_to_geodataframe, to_geojson, to_csv

# Single file
result = process_file("building.laz")
print(result.model_dump_json(indent=2))

# Pre-loaded point array (e.g. from your own clipping)
result = process_building("bldg_001", xyz_array, ground_z=12.4)

# City tile — iterate OSM buildings
results = []
for bldg_id, xyz, ground_z in iter_building_point_clouds("tile.laz", epsg=32614):
    results.append(process_building(bldg_id, xyz, ground_z))

# Export
gdf = results_to_geodataframe(results, footprints)
to_geojson(gdf, "roofs.geojson")
to_csv(gdf, "roofs.csv", include_wkt=True)
```

---

## Test Data

| File | Description | CRS |
|---|---|---|
| `data/simple.laz` | Synthetic gable roof | — |
| `data/autzen_trim.laz` | Autzen Stadium area (Oregon) | — |
| `data/nyc_sandy.laz` | NYC post-Sandy LiDAR tile | EPSG:32618 |
| `data/sf_sample.laz` | San Francisco sample | EPSG:32610 |
| `data/tx_central.laz` | Central Texas sample | — |
| `data/austin_buildings_sample.laz` | Austin urban buildings | EPSG:32614 |

---

## Known Limitations

- **RANSAC on large tiles**: Dense city tiles (> 100k building points) are subsampled. Accuracy on complex roofs may be reduced at very low density (< 4 pts/m²).
- **OSM coverage**: `tile` command depends on OSM building footprint completeness in the area. Rural or unmapped areas may yield no results.
- **Vertical surfaces**: Walls and dormers may occasionally be detected as facets — the `--min-facet-area` filter helps but does not eliminate them entirely.
- **CSF dependency**: Unclassified point clouds require `pip install CSF`, which is not automatically installed (binary build availability varies by platform).
