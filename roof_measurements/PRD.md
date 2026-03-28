# PRD: Roof Measurement Extraction from LiDAR Data

## Overview

A Python-based tool that processes LiDAR point cloud data to automatically extract key roof measurements: number of facets (planes), pitch (slope angle), and height. The system segments roof surfaces from raw point cloud data and computes geometric properties for each detected facet.

---

## Problem Statement

Manual roof measurement from field surveys or 2D imagery is time-consuming, error-prone, and expensive. LiDAR data provides dense 3D point clouds that can be programmatically analyzed to extract accurate roof geometry — enabling faster quoting for roofing, solar, and insurance applications.

---

## Goals

1. **Accurately segment roof facets** from LiDAR point clouds (`.las`/`.laz` files)
2. **Compute pitch** (slope angle in degrees) for each facet
3. **Compute roof height** (ridge height above ground level)
4. **Count the number of distinct facets** (planar roof surfaces)
5. Provide results as structured output (JSON) and optional visualization

---

## Non-Goals (v1)

- Real-time processing / streaming LiDAR ingestion
- Full building footprint extraction or wall segmentation
- Integration with GIS platforms (ArcGIS, QGIS plugins)
- Web UI or API server — this is a CLI/library tool first
- Handling non-building structures (towers, bridges, vegetation)

---

## Target Users

- **Roofing contractors** — estimating material quantities from roof geometry
- **Solar installers** — assessing roof suitability (pitch, area, orientation)
- **Insurance adjusters** — measuring roof characteristics for underwriting
- **Researchers** — urban morphology and building stock analysis

---

## Input

| Field | Description |
|-------|-------------|
| **LiDAR point cloud** | `.las` or `.laz` file (ASPRS LAS format) covering one or more buildings |
| **Classification** | Points should ideally be pre-classified (class 6 = building per ASPRS). If unclassified, the tool will attempt ground/building separation. |
| **CRS** | Any projected coordinate system with metric units (e.g., UTM). Heights must be orthometric or ellipsoidal. |
| **Optional: building footprint** | GeoJSON/Shapefile polygon to clip points to a specific building. If not provided, the tool clusters buildings automatically. |

---

## Output

### Structured Output (JSON per building)

```json
{
  "building_id": "bldg_001",
  "num_facets": 4,
  "height_m": 8.7,
  "facets": [
    {
      "facet_id": 1,
      "pitch_deg": 32.5,
      "azimuth_deg": 180.0,
      "area_m2": 45.2,
      "normal_vector": [0.0, -0.537, 0.843],
      "num_points": 1245
    },
    {
      "facet_id": 2,
      "pitch_deg": 33.1,
      "azimuth_deg": 0.0,
      "area_m2": 44.8,
      "normal_vector": [0.0, 0.537, 0.843],
      "num_points": 1198
    }
  ],
  "ground_elevation_m": 102.3,
  "ridge_elevation_m": 111.0
}
```

### Optional Visualization

- 3D point cloud colored by facet assignment
- 2D top-down view with facet boundaries and pitch labels

---

## Technical Approach

### Pipeline Stages

```
Raw LiDAR (.las/.laz)
  │
  ▼
1. Load & Filter
   - Read with laspy
   - Filter to building class (6) or apply ground/non-ground classification
   - Optional: clip to building footprint polygon
  │
  ▼
2. Ground Separation (if no classification)
   - CSF (Cloth Simulation Filter) or progressive morphological filter
   - Establish ground reference elevation per building
  │
  ▼
3. Building Clustering
   - DBSCAN or connected-component clustering on XY plane
   - Separate individual buildings from multi-building scenes
  │
  ▼
4. Roof Plane Segmentation
   - RANSAC-based multi-plane detection
     OR
   - Region-growing on normal vectors (compute normals via PCA on k-NN)
   - Merge coplanar segments within angle/distance thresholds
  │
  ▼
5. Facet Property Extraction
   - Pitch: arccos(normal_z) → degrees from horizontal
   - Azimuth: atan2(normal_y, normal_x) → compass bearing of downslope
   - Area: 2D alpha-shape or convex hull of facet points, projected to plane
   - Height: max(Z_roof) - median(Z_ground)
  │
  ▼
6. Output
   - Write JSON results
   - Optional: generate matplotlib/open3d visualization
```

### Key Algorithms

| Step | Primary Method | Fallback |
|------|---------------|----------|
| Ground filtering | CSF (`cloth_simulation_filter`) | Progressive morphological (PDAL) |
| Building clustering | DBSCAN (scikit-learn) | HDBSCAN for variable density |
| Plane segmentation | RANSAC (Open3D) | Region growing on normals |
| Normal estimation | PCA on k-NN (k=20-30) | Open3D `estimate_normals()` |
| Area computation | Alpha shape (alphashape lib) | Convex hull (scipy) |

---

## Tech Stack

| Component | Library | Why |
|-----------|---------|-----|
| Point cloud I/O | `laspy` (v2+) | Native LAS/LAZ read/write, numpy integration |
| 3D geometry | `open3d` | RANSAC, normals, visualization, well-maintained |
| Clustering | `scikit-learn` | DBSCAN, proven and fast |
| Ground filtering | `CSF` (`cloth_simulation_filter`) | Simple, effective, pip-installable |
| Geospatial | `shapely`, `geopandas` | Footprint clipping, area computation |
| Numerics | `numpy`, `scipy` | Core math, spatial KDTree |
| Visualization | `matplotlib`, `open3d` | 2D/3D plots |
| CLI | `click` or `argparse` | Command-line interface |
| Output | `pydantic` | Structured, validated JSON output models |

---

## Accuracy Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Facet count | ±0 for simple roofs (gable, hip), ±1 for complex | Validated against manual annotation |
| Pitch | ±2° | Compared to known roof plans or field measurements |
| Height | ±0.3m | Dependent on point density and ground classification quality |
| Min point density | 4 pts/m² | Below this, accuracy degrades significantly |

---

## Project Structure

```
roof_measurements/
├── PRD.md
├── pyproject.toml
├── README.md
├── src/
│   └── roof_measurements/
│       ├── __init__.py
│       ├── cli.py              # CLI entry point
│       ├── io.py               # LAS/LAZ loading, footprint clipping
│       ├── ground.py           # Ground filtering / classification
│       ├── clustering.py       # Building-level clustering
│       ├── segmentation.py     # Roof plane segmentation (RANSAC / region growing)
│       ├── features.py         # Pitch, azimuth, area, height extraction
│       ├── models.py           # Pydantic output models
│       └── viz.py              # Visualization utilities
├── tests/
│   ├── test_segmentation.py
│   ├── test_features.py
│   └── fixtures/               # Small sample .las files
└── examples/
    └── example_usage.py
```

---

## Milestones

### M1: Core Pipeline (MVP)
- Load LAS/LAZ, filter to building points (pre-classified)
- RANSAC plane segmentation
- Extract facet count, pitch, height
- JSON output
- Unit tests with synthetic point clouds

### M2: Robustness
- Ground filtering for unclassified data
- Building clustering (multi-building scenes)
- Region-growing segmentation as alternative
- Footprint-based clipping

### M3: Visualization & Polish
- 3D colored facet visualization
- 2D annotated top-down view
- CLI with configurable parameters
- Documentation and examples

### M4: Validation
- Test against known roof geometries (manual measurements)
- Benchmark on public LiDAR datasets (USGS 3DEP, AHN)
- Performance profiling for large scenes

---

## Open Questions

1. **Segmentation method**: RANSAC is simpler but struggles with noisy edges between facets. Region growing on normals is more robust but slower. Start with RANSAC, add region growing in M2?
2. **Flat roof handling**: Flat roofs (pitch < 5°) — should they count as a single facet or be subdivided by slight slopes for drainage?
3. **Dormers & complex features**: Dormers, chimneys, and skylights create small planar segments. Minimum facet area threshold to filter noise?
4. **Point density requirements**: What's the minimum viable density? 4 pts/m² as baseline, but should we warn or refuse below a threshold?
5. **Coordinate systems**: Require projected CRS or handle geographic (lat/lon) with auto-reprojection?
