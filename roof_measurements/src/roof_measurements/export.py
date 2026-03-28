"""Export BuildingResult lists to GeoJSON and CSV.

Typical usage
-------------
    from roof_measurements.footprints import fetch_osm_buildings, iter_building_point_clouds
    from roof_measurements.pipeline import process_building
    from roof_measurements.export import results_to_geodataframe, to_geojson, to_csv

    results = [...]           # list[BuildingResult]
    footprints = fetch_osm_buildings(west, south, east, north)

    gdf = results_to_geodataframe(results, footprints)
    to_geojson(gdf, "output.geojson")
    to_csv(gdf, "output.csv")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import geopandas as gpd

from roof_measurements.models import BuildingResult

logger = logging.getLogger(__name__)


def build_single_building_geojson(
    result: BuildingResult,
    facet_point_lists: list,
    polygon_coords: list,
    epsg: int,
) -> str:
    """Build a GeoJSON FeatureCollection for one building.

    Includes the building footprint polygon plus one polygon per roof facet
    (convex hull of its LiDAR points, projected back to WGS84).

    Parameters
    ----------
    result :
        Processed BuildingResult.
    facet_point_lists :
        List of (N, 3) numpy arrays — one per facet, in the projected CRS.
    polygon_coords :
        Exterior ring of the footprint in WGS84 as (lon, lat) pairs.
    epsg :
        EPSG code of the projected CRS that ``facet_point_lists`` are in.

    Returns
    -------
    str
        GeoJSON FeatureCollection as an indented JSON string.
    """
    import numpy as np
    import pyproj
    from scipy.spatial import ConvexHull

    to_wgs84 = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    features = []

    # ── Per-facet polygons ────────────────────────────────────────────────────
    for f, pts in zip(result.facets, facet_point_lists):
        pts = np.asarray(pts)
        xy = pts[:, :2]
        try:
            verts = xy[ConvexHull(xy).vertices] if len(xy) >= 3 else xy
        except Exception:
            verts = xy
        # Close the ring and transform to WGS84
        ring = np.vstack([verts, verts[:1]])
        lons, lats = to_wgs84.transform(ring[:, 0], ring[:, 1])
        coords = [[round(lo, 7), round(la, 7)] for lo, la in zip(lons, lats)]

        features.append({
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "feature_type": "facet",
                "facet_id": f.facet_id,
                "pitch_deg": f.pitch_deg,
                "azimuth_deg": f.azimuth_deg,
                "area_m2": f.area_m2,
                "confidence": f.confidence,
                "solar_kwh_m2_yr": f.solar_kwh_m2_yr,
                "solar_suitability": f.solar_suitability,
                "is_flat": f.is_flat,
                "eave_elevation_m": f.eave_elevation_m,
                "plane_rms_m": f.plane_rms_m,
            },
        })

    # ── Building footprint ────────────────────────────────────────────────────
    fp_coords = [[round(lo, 7), round(la, 7)] for lo, la in polygon_coords]
    features.append({
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [fp_coords]},
        "properties": {
            "feature_type": "building_footprint",
            "building_id": result.building_id,
            "roof_type": result.roof_type,
            "num_facets": result.num_facets,
            "height_m": result.height_m,
            "eave_height_m": result.eave_height_m,
            "total_roof_area_m2": result.total_roof_area_m2,
            "total_solar_kwh_yr": result.total_solar_kwh_yr,
            "ground_elevation_m": result.ground_elevation_m,
        },
    })

    return json.dumps({"type": "FeatureCollection", "features": features}, indent=2)


def results_to_geodataframe(
    results: list[BuildingResult],
    footprints: "gpd.GeoDataFrame",
) -> "gpd.GeoDataFrame":
    """Join a list of BuildingResult to the OSM footprints GeoDataFrame.

    Matching is by the sequential integer row-position of *footprints* — the
    same index used by :func:`~roof_measurements.footprints.iter_building_point_clouds`.
    Buildings whose ``building_id`` cannot be matched to a footprint row are
    included without geometry (``NaN``).

    Parameters
    ----------
    results :
        Output of :func:`~roof_measurements.pipeline.process_building` calls.
    footprints :
        GeoDataFrame returned by
        :func:`~roof_measurements.footprints.fetch_osm_buildings`.
        Must have a sequential integer RangeIndex (0, 1, 2, …).

    Returns
    -------
    geopandas.GeoDataFrame
        One row per BuildingResult.  Columns:

        - **geometry** — building footprint polygon (WGS84 / EPSG:4326)
        - **osm_id** — OSM element id (from ``footprints["id"]`` if present)
        - **building_type** — OSM ``building`` tag value
        - **name** — OSM ``name`` tag (when present)
        - **building_id** — value from BuildingResult
        - **num_facets**
        - **height_m** — ridge height above ground
        - **eave_height_m** — lowest eave above ground
        - **mean_pitch_deg** — mean facet pitch
        - **max_pitch_deg**
        - **min_pitch_deg**
        - **total_roof_area_m2** — sum of all facet areas
        - **ground_elevation_m**
        - **ridge_elevation_m**
        - **point_density_m2**
        - **segmentation_method**
    """
    import geopandas as gpd
    import pandas as pd

    rows = []
    for r in results:
        pitches = [f.pitch_deg for f in r.facets]
        if pitches:
            min_p, max_p, sum_p = min(pitches), max(pitches), sum(pitches)
            mean_pitch = round(sum_p / len(pitches), 2)
            max_pitch  = round(max_p, 2)
            min_pitch  = round(min_p, 2)
        else:
            mean_pitch = max_pitch = min_pitch = None
        row: dict = {
            "building_id": r.building_id,
            "num_facets": r.num_facets,
            "height_m": r.height_m,
            "eave_height_m": r.eave_height_m,
            "mean_pitch_deg": mean_pitch,
            "max_pitch_deg": max_pitch,
            "min_pitch_deg": min_pitch,
            "total_roof_area_m2": round(sum(f.area_m2 for f in r.facets), 2),
            "ground_elevation_m": r.ground_elevation_m,
            "ridge_elevation_m": r.ridge_elevation_m,
            "point_density_m2": r.point_density_m2,
            "segmentation_method": r.segmentation_method,
        }
        rows.append(row)

    result_df = pd.DataFrame(rows)

    # ── Join to footprint geometry ────────────────────────────────────────────
    has_id_col = "id" in footprints.columns
    fp_key_col = footprints["id"].astype(str) if has_id_col else footprints.index.astype(str)
    fp_indexed = footprints.set_index(fp_key_col)

    def _lookup(bldg_id: str):
        if bldg_id in fp_indexed.index:
            return fp_indexed.loc[bldg_id]
        logger.warning("building_id %r not found in footprints", bldg_id)
        return None

    fp_rows = [_lookup(bid) for bid in result_df["building_id"]]
    bids = result_df["building_id"]
    result_df["osm_id"]        = [str(r["id"]) if (r is not None and has_id_col) else bid
                                   for r, bid in zip(fp_rows, bids)]
    result_df["building_type"] = [r.get("building") if r is not None else None for r in fp_rows]
    result_df["name"]          = [r.get("name")     if r is not None else None for r in fp_rows]
    # Reorder: put OSM metadata columns first
    meta_cols = ["osm_id", "building_type", "name"]
    result_df = result_df[meta_cols + [c for c in result_df.columns if c not in meta_cols]]
    geoms = [r.geometry if r is not None else None for r in fp_rows]

    gdf = gpd.GeoDataFrame(result_df, geometry=geoms, crs="EPSG:4326")
    logger.info("Built GeoDataFrame with %d rows", len(gdf))
    return gdf


def to_geojson(
    gdf: "gpd.GeoDataFrame",
    path: str | Path,
) -> Path:
    """Write *gdf* to a GeoJSON file.

    Parameters
    ----------
    gdf :
        GeoDataFrame from :func:`results_to_geodataframe`.
    path :
        Output file path (will be created or overwritten).

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(path, driver="GeoJSON")
    logger.info("Wrote GeoJSON → %s (%d features)", path, len(gdf))
    return path


def to_csv(
    gdf: "gpd.GeoDataFrame",
    path: str | Path,
    include_wkt: bool = False,
) -> Path:
    """Write results to CSV (geometry column omitted by default).

    Parameters
    ----------
    gdf :
        GeoDataFrame from :func:`results_to_geodataframe`.
    path :
        Output file path (will be created or overwritten).
    include_wkt :
        If *True*, include a ``geometry_wkt`` column with the footprint as
        Well-Known Text so the CSV can be re-imported into GIS tools.

    Returns
    -------
    Path
        Resolved path of the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = gdf.drop(columns=["geometry"])
    if include_wkt:
        df = df.copy()
        df.insert(3, "geometry_wkt", [g.wkt if g is not None else None for g in gdf.geometry])

    df.to_csv(path, index=False)
    logger.info("Wrote CSV → %s (%d rows)", path, len(df))
    return path
