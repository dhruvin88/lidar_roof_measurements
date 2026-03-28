"""Fetch open-source building footprints and clip LiDAR to individual buildings.

Source: OpenStreetMap via osmnx.

Typical usage
-------------
    from roof_measurements.footprints import iter_building_point_clouds
    from roof_measurements.pipeline import process_file_from_points

    for bldg_id, xyz, ground_z in iter_building_point_clouds("tile.laz", epsg=32618):
        result = process_building(bldg_id, xyz, ground_z)
        print(result.model_dump_json(indent=2))

Requires
--------
    pip install osmnx geopandas pyproj
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

import numpy as np

from roof_measurements.constants import ASPRS_BUILDING, ASPRS_GROUND, NON_BUILDING_CLASSES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CRS / bounding box helpers
# ---------------------------------------------------------------------------

def las_wgs84_bbox(path: str | Path, epsg: int) -> tuple[float, float, float, float]:
    """Return *(west, south, east, north)* in WGS84 for a LAS/LAZ file.

    Parameters
    ----------
    path :
        LAS/LAZ file path.
    epsg :
        EPSG code of the file's projected CRS (e.g. 32618 for UTM Zone 18N).

    Returns
    -------
    (west, south, east, north) in decimal degrees (EPSG:4326).
    """
    import laspy
    import pyproj

    with laspy.open(path) as f:
        las = f.read()
    hdr = las.header
    transformer = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

    corners_xy = [
        (hdr.x_min, hdr.y_min),
        (hdr.x_max, hdr.y_min),
        (hdr.x_min, hdr.y_max),
        (hdr.x_max, hdr.y_max),
    ]
    lons, lats = zip(*[transformer.transform(x, y) for x, y in corners_xy])
    return float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats))


# ---------------------------------------------------------------------------
# OSM footprint fetching
# ---------------------------------------------------------------------------

def fetch_osm_buildings(
    west: float,
    south: float,
    east: float,
    north: float,
) -> "geopandas.GeoDataFrame":
    """Fetch OSM building footprints for a bounding box.

    Parameters
    ----------
    west, south, east, north :
        Bounding box in WGS84 decimal degrees.

    Returns
    -------
    geopandas.GeoDataFrame
        Polygon/MultiPolygon footprints in EPSG:4326.
        Columns include ``geometry`` and any OSM tags present (``name``,
        ``building``, ``addr:*``, etc.).
    """
    import osmnx as ox
    from shapely.geometry import Polygon, MultiPolygon

    logger.info(
        "Fetching OSM buildings for bbox W=%.5f S=%.5f E=%.5f N=%.5f",
        west, south, east, north,
    )
    gdf = ox.features_from_bbox(
        bbox=(west, south, east, north),
        tags={"building": True},
    )
    # Keep only polygon geometries (drop point/line nodes)
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    # reset_index() (without drop=True) promotes 'element'/'id' from the
    # MultiIndex into regular columns so they survive downstream joins.
    gdf = gdf.reset_index().reset_index(drop=True)
    logger.info("Found %d building footprints", len(gdf))
    return gdf


# ---------------------------------------------------------------------------
# Single-point footprint lookup
# ---------------------------------------------------------------------------

def footprint_at_point(
    lat: float,
    lon: float,
    buffer_m: float = 100.0,
) -> tuple[str, "shapely.geometry.Polygon", "geopandas.GeoDataFrame"]:
    """Return the OSM building footprint that contains *(lat, lon)*.

    Parameters
    ----------
    lat, lon :
        WGS84 decimal degrees of the target point.
    buffer_m :
        Search radius in metres around the point.  Larger values find buildings
        whose footprint may not perfectly contain the click point.

    Returns
    -------
    osm_id : str
    polygon : shapely.geometry.Polygon
        The building footprint in WGS84.
    row : geopandas.GeoDataFrame
        Single-row GeoDataFrame with all OSM tags for the matched building.

    Raises
    ------
    ValueError
        No OSM buildings found in the search area, or the point does not fall
        inside any building footprint.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    point = Point(lon, lat)

    # Convert buffer_m to approximate degrees for the OSM query
    buffer_deg = buffer_m / 111_320.0
    west  = lon - buffer_deg
    east  = lon + buffer_deg
    south = lat - buffer_deg
    north = lat + buffer_deg

    gdf = fetch_osm_buildings(west, south, east, north)
    if gdf.empty:
        raise ValueError(
            f"No OSM building footprints found within {buffer_m:.0f} m of "
            f"({lat:.6f}, {lon:.6f})."
        )

    # Prefer buildings that strictly contain the point; fall back to nearest
    containing = gdf[gdf.geometry.contains(point)]
    if not containing.empty:
        row = containing.iloc[[0]]
    else:
        logger.warning(
            "Point not strictly inside any footprint — using nearest building"
        )
        gdf = gdf.copy()
        gdf["_dist"] = gdf.geometry.distance(point)
        row = gdf.nsmallest(1, "_dist").drop(columns=["_dist"])

    osm_id = str(row.iloc[0]["id"]) if "id" in row.columns else str(row.index[0])
    polygon = row.iloc[0].geometry
    logger.info("Matched building OSM id=%s", osm_id)
    return osm_id, polygon, row


# ---------------------------------------------------------------------------
# Point-cloud clipping
# ---------------------------------------------------------------------------

def clip_xyz_to_polygon(
    xyz: np.ndarray,
    polygon,
    transformer: "pyproj.Transformer",
) -> np.ndarray:
    """Return the subset of *xyz* whose XY falls inside *polygon*.

    Parameters
    ----------
    xyz :
        (N, 3) array in the file's projected CRS.
    polygon :
        Shapely Polygon/MultiPolygon in WGS84 (EPSG:4326).
    transformer :
        pyproj.Transformer from the projected CRS to WGS84, ``always_xy=True``.

    Returns
    -------
    np.ndarray, shape (M, 3)
    """
    from shapely import contains_xy

    lons, lats = transformer.transform(xyz[:, 0], xyz[:, 1])
    mask = contains_xy(polygon, lons, lats)
    return xyz[mask]


# ---------------------------------------------------------------------------
# High-level iterator
# ---------------------------------------------------------------------------

def iter_building_point_clouds(
    las_path: str | Path,
    epsg: int,
    min_points: int = 50,
) -> Generator[tuple[str, np.ndarray, float], None, None]:
    """Yield *(building_id, xyz, ground_z)* for every OSM building in the tile.

    The function:
    1. Loads the full LAS/LAZ file.
    2. Fetches OSM building footprints covering the tile's bounding box.
    3. For each footprint, clips the point cloud and yields the result.

    Building points are taken from ASPRS class 6 when available; otherwise
    all points excluding ground/noise/water are used.

    Parameters
    ----------
    las_path :
        LAS/LAZ file path.
    epsg :
        EPSG code of the file's projected CRS.
    min_points :
        Buildings with fewer than this many LiDAR points are skipped.

    Yields
    ------
    building_id : str
        OSM ``osmid`` value (or a sequential index fallback).
    xyz : np.ndarray, shape (N, 3)
        LiDAR points within the building footprint, in the file's native CRS.
    ground_z : float
        Median ground elevation beneath the tile (metres).
    """
    import laspy
    import pyproj

    las_path = Path(las_path)

    # ── Load all points ──────────────────────────────────────────────────────
    logger.info("Loading %s", las_path)
    with laspy.open(las_path) as f:
        las = f.read()

    xyz = np.column_stack([
        np.array(las.x, dtype=np.float64),
        np.array(las.y, dtype=np.float64),
        np.array(las.z, dtype=np.float64),
    ])
    cls = np.array(las.classification)

    has_class6 = np.any(cls == ASPRS_BUILDING)
    has_class2 = np.any(cls == ASPRS_GROUND)

    if has_class6:
        candidate_pts = xyz[cls == ASPRS_BUILDING]
        logger.info("Using %d class-6 building points", len(candidate_pts))
    else:
        non_noise_mask = ~np.isin(cls, list(NON_BUILDING_CLASSES))
        candidate_pts = xyz[non_noise_mask]
        logger.info(
            "No class 6 — using %d non-noise points as candidates", len(candidate_pts)
        )

    # Tile-wide ground fallback (used when a building has no local ground points)
    tile_ground_z = (
        float(np.median(xyz[cls == ASPRS_GROUND, 2]))
        if has_class2
        else float(np.min(xyz[:, 2]))
    )
    ground_pts = xyz[cls == ASPRS_GROUND] if has_class2 else np.empty((0, 3))

    # ── Fetch footprints ─────────────────────────────────────────────────────
    west, south, east, north = las_wgs84_bbox(las_path, epsg)
    footprints = fetch_osm_buildings(west, south, east, north)

    if len(footprints) == 0:
        logger.warning("No OSM building footprints found in bbox — nothing to yield")
        return

    transformer = pyproj.Transformer.from_crs(
        f"EPSG:{epsg}", "EPSG:4326", always_xy=True
    )

    # Pre-transform all candidate and ground points to WGS84 once — avoids
    # repeating the coordinate transform for every building footprint.
    from shapely import contains_xy

    cand_lons, cand_lats = transformer.transform(candidate_pts[:, 0], candidate_pts[:, 1])
    if len(ground_pts) > 0:
        gnd_lons, gnd_lats = transformer.transform(ground_pts[:, 0], ground_pts[:, 1])
    else:
        gnd_lons = gnd_lats = np.empty(0)

    # ── Clip per building ────────────────────────────────────────────────────
    for i, row in footprints.iterrows():
        polygon = row.geometry

        mask = contains_xy(polygon, cand_lons, cand_lats)
        clipped = candidate_pts[mask]

        if len(clipped) < min_points:
            logger.debug("Building %s: only %d pts — skipped", i, len(clipped))
            continue

        # Per-building ground elevation from local class-2 points.
        # Fall back to tile-wide median when fewer than 3 are found.
        if len(gnd_lons) > 0:
            gnd_mask = contains_xy(polygon, gnd_lons, gnd_lats)
            if gnd_mask.sum() >= 3:
                local_ground_z = float(np.median(ground_pts[gnd_mask, 2]))
            else:
                local_ground_z = tile_ground_z
        else:
            local_ground_z = tile_ground_z

        osm_id = str(row["id"]) if "id" in row.index else str(i)
        logger.info(
            "Building %s: %d points, ground_z=%.2fm", osm_id, len(clipped), local_ground_z
        )
        yield osm_id, clipped, local_ground_z


# ---------------------------------------------------------------------------
# Convenience: process all buildings in a tile
# ---------------------------------------------------------------------------

def process_tile(
    las_path: str | Path,
    epsg: int,
    **pipeline_kwargs,
) -> list:
    """Run the full pipeline on every OSM building in a LAS/LAZ tile.

    Parameters
    ----------
    las_path :
        LAS/LAZ file path.
    epsg :
        EPSG code of the file's projected CRS.
    **pipeline_kwargs :
        Forwarded to :func:`roof_measurements.pipeline.process_building`.

    Returns
    -------
    list of BuildingResult
    """
    from roof_measurements.pipeline import process_building

    results = []
    for bldg_id, xyz, ground_z in iter_building_point_clouds(las_path, epsg):
        try:
            result = process_building(bldg_id, xyz, ground_z, **pipeline_kwargs)
            results.append(result)
        except Exception as exc:
            logger.warning("Building %s failed: %s", bldg_id, exc)
    return results
