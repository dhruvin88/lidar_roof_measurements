"""CLI entry point: roof-measure."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from roof_measurements.pipeline import process_file


# ---------------------------------------------------------------------------
# Shared options factory
# ---------------------------------------------------------------------------

def _segmentation_options(fn):
    """Decorator that attaches shared RANSAC / segmentation options."""
    fn = click.option(
        "--min-facet-area", default=1.0, show_default=True,
        help="Minimum facet area in m² — smaller facets are discarded as noise.",
    )(fn)
    fn = click.option(
        "--min-facet-points", default=10, show_default=True,
        help="Minimum number of LiDAR points a facet must have to be kept.",
    )(fn)
    fn = click.option(
        "--distance-threshold", default=0.15, show_default=True,
        help="RANSAC inlier distance tolerance in metres.",
    )(fn)
    fn = click.option(
        "--max-planes", default=20, show_default=True,
        help="Maximum number of roof planes to extract.",
    )(fn)
    return fn


# ---------------------------------------------------------------------------
# Top-level group
# ---------------------------------------------------------------------------

@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """roof-measure — extract roof geometry from LiDAR LAS/LAZ files.

    \b
    Commands:
      process   Single-building LAS/LAZ → JSON
      tile      City tile LAS/LAZ → GeoJSON + CSV (uses OSM footprints)
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# ---------------------------------------------------------------------------
# `process` — single-file command (original behaviour)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), default=None,
    help="Output JSON file. Defaults to <stem>_roof.json beside the input.",
)
@click.option("--building-id", default=None,
              help="Override the building ID. Defaults to the filename stem.")
@_segmentation_options
def process(
    input_file: Path,
    output: Path | None,
    building_id: str | None,
    min_facet_area: float,
    min_facet_points: int,
    distance_threshold: float,
    max_planes: int,
) -> None:
    """Extract roof measurements from a single LAS/LAZ file → JSON.

    \b
    Example:
        roof-measure process building.laz --output results.json
        roof-measure process scan.las --min-facet-area 2.0 -v
    """
    if output is None:
        output = input_file.parent / f"{input_file.stem}_roof.json"

    click.echo(f"Processing: {input_file}")

    try:
        result = process_file(
            path=input_file,
            building_id=building_id,
            distance_threshold=distance_threshold,
            min_facet_area_m2=min_facet_area,
            min_facet_points=min_facet_points,
            max_planes=max_planes,
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2))

    click.echo(f"Written → {output}")
    click.echo(f"  Facets:      {result.num_facets}")
    click.echo(f"  Height:      {result.height_m:.2f} m")
    click.echo(f"  Eave height: {result.eave_height_m:.2f} m")
    click.echo(f"  Method:      {result.segmentation_method}")
    for f in result.facets:
        flag = " [FLAT]" if f.is_flat else ""
        click.echo(
            f"  Facet {f.facet_id}: pitch={f.pitch_deg:.1f}°  "
            f"azimuth={f.azimuth_deg:.0f}°  area={f.area_m2:.1f} m²  "
            f"eave={f.eave_elevation_m:.2f} m  "
            f"conf={f.confidence:.2f}  rms={f.plane_rms_m:.3f} m{flag}"
        )


# ---------------------------------------------------------------------------
# `tile` — city-tile command using OSM footprints
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--epsg", required=True, type=int,
    help="EPSG code of the LAS file's projected CRS (e.g. 32614 for UTM 14N).",
)
@click.option(
    "--out", "-o", "output_dir", type=click.Path(path_type=Path), default=None,
    help="Output directory for GeoJSON + CSV. Defaults to a folder beside the input.",
)
@click.option(
    "--min-points", default=50, show_default=True,
    help="Skip buildings with fewer than this many LiDAR points.",
)
@click.option(
    "--geojson/--no-geojson", default=True, show_default=True,
    help="Write GeoJSON output.",
)
@click.option(
    "--csv/--no-csv", "write_csv", default=True, show_default=True,
    help="Write CSV output.",
)
@_segmentation_options
def tile(
    input_file: Path,
    epsg: int,
    output_dir: Path | None,
    min_points: int,
    geojson: bool,
    write_csv: bool,
    min_facet_area: float,
    min_facet_points: int,
    distance_threshold: float,
    max_planes: int,
) -> None:
    """Process a city LiDAR tile: fetch OSM footprints, clip per building, export.

    \b
    Example:
        roof-measure tile austin.laz --epsg 32614
        roof-measure tile nyc_sandy.laz --epsg 32618 --out ./results -v
    """
    from roof_measurements.export import results_to_geodataframe, to_geojson, to_csv
    from roof_measurements.footprints import (
        fetch_osm_buildings, iter_building_point_clouds, las_wgs84_bbox,
    )
    from roof_measurements.pipeline import process_building

    if output_dir is None:
        output_dir = input_file.parent / input_file.stem

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Fetch footprints ─────────────────────────────────────────────────────
    click.echo(f"Tile:  {input_file}")
    click.echo(f"EPSG:  {epsg}")

    west, south, east, north = las_wgs84_bbox(input_file, epsg)
    click.echo(f"Bbox:  W={west:.5f}  S={south:.5f}  E={east:.5f}  N={north:.5f}")

    footprints = fetch_osm_buildings(west, south, east, north)
    click.echo(f"OSM:   {len(footprints)} building footprint(s)")

    if len(footprints) == 0:
        click.echo("No buildings found — nothing to process.", err=True)
        sys.exit(1)

    # ── Process each building ────────────────────────────────────────────────
    results, skipped = [], 0
    for bldg_id, xyz, ground_z in iter_building_point_clouds(
        input_file, epsg=epsg, min_points=min_points
    ):
        try:
            r = process_building(
                bldg_id, xyz, ground_z,
                distance_threshold=distance_threshold,
                min_facet_area_m2=min_facet_area,
                min_facet_points=min_facet_points,
                max_planes=max_planes,
            )
            results.append(r)
            mean_pitch = sum(f.pitch_deg for f in r.facets) / r.num_facets
            click.echo(
                f"  {bldg_id:14s}  {r.num_facets:2d} facet(s)  "
                f"h={r.height_m:.1f}m  eave={r.eave_height_m:.1f}m  "
                f"pitch={mean_pitch:.1f}°  [{r.segmentation_method}]"
            )
        except Exception as exc:
            click.echo(f"  {bldg_id:14s}  SKIPPED ({exc})", err=True)
            skipped += 1

    click.echo(f"\nProcessed {len(results)} building(s) ({skipped} skipped).")

    if not results:
        click.echo("No results to export.", err=True)
        sys.exit(1)

    # ── Export ───────────────────────────────────────────────────────────────
    gdf = results_to_geodataframe(results, footprints)

    if geojson:
        path = to_geojson(gdf, output_dir / f"{input_file.stem}_roofs.geojson")
        click.echo(f"GeoJSON → {path}")

    if write_csv:
        path = to_csv(gdf, output_dir / f"{input_file.stem}_roofs.csv", include_wkt=True)
        click.echo(f"CSV     → {path}")


# ---------------------------------------------------------------------------
# `query` — lat/lon → download LiDAR + OSM footprint → JSON
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--lat", required=True, type=float, help="Latitude in decimal degrees (WGS84).")
@click.option("--lon", required=True, type=float, help="Longitude in decimal degrees (WGS84). Negative values accepted.")
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), default=None,
    help="Output JSON file. Defaults to <osm_id>_roof.json in the current directory.",
)
@click.option(
    "--epsg", type=int, default=None,
    help="Override EPSG code. Auto-detected from the LAS header when omitted.",
)
@click.option(
    "--cache-dir", type=click.Path(path_type=Path), default=None,
    help="Directory for cached LiDAR tiles. Defaults to ~/.cache/roof_measurements/lidar.",
)
@click.option(
    "--buffer", default=100.0, show_default=True,
    help="OSM building search radius in metres around the point.",
)
@_segmentation_options
def query(
    lat: float,
    lon: float,
    output: Path | None,
    epsg: int | None,
    cache_dir: Path | None,
    buffer: float,
    min_facet_area: float,
    min_facet_points: int,
    distance_threshold: float,
    max_planes: int,
) -> None:
    """Fetch LiDAR + OSM footprint for a coordinate and extract roof geometry.

    Downloads the USGS 3DEP tile that covers the point, clips it to the OSM
    building footprint, and runs the full measurement pipeline.

    \\b
    Example:
        roof-measure query --lat 30.2672 --lon -97.7431       # Austin, TX
        roof-measure query --lat 40.7128 --lon -74.0060       # New York, NY
        roof-measure -v query --lat 37.7749 --lon -122.4194   # San Francisco, CA
    """
    from roof_measurements.datasources import fetch_lidar_for_point
    from roof_measurements.footprints import footprint_at_point
    from roof_measurements.pipeline import process_building
    from roof_measurements.io import load_building_points
    import numpy as np

    click.echo(f"Location: {lat:.6f}, {lon:.6f}")

    # ── 1. Find OSM footprint ─────────────────────────────────────────────────
    click.echo("Fetching OSM building footprint...")
    try:
        osm_id, polygon, _ = footprint_at_point(lat, lon, buffer_m=buffer)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    click.echo(f"  OSM id: {osm_id}  |  footprint vertices: {len(polygon.exterior.coords)}")

    # ── 2. Download LiDAR tiles ───────────────────────────────────────────────
    click.echo("Fetching USGS 3DEP LiDAR tile(s)...")
    try:
        kwargs = {} if cache_dir is None else {"cache_dir": cache_dir}
        laz_paths, detected_epsg = fetch_lidar_for_point(lat, lon, **kwargs)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    resolved_epsg = epsg or detected_epsg
    for p in laz_paths:
        click.echo(f"  File:  {p.name}  ({p.stat().st_size >> 20} MB)")
    click.echo(f"  EPSG:  {resolved_epsg}")

    # ── 3. Clip LiDAR to building footprint ──────────────────────────────────
    click.echo("Clipping point cloud to building footprint...")
    import laspy
    import pyproj
    from shapely import contains_xy
    from roof_measurements.constants import ASPRS_BUILDING, ASPRS_GROUND, NON_BUILDING_CLASSES

    transformer = pyproj.Transformer.from_crs(
        f"EPSG:{resolved_epsg}", "EPSG:4326", always_xy=True
    )

    # Merge points from all tiles (handles buildings on tile boundaries)
    all_candidates, all_ground = [], []
    for laz_path in laz_paths:
        with laspy.open(laz_path) as f:
            las = f.read()
        xyz = np.column_stack([
            np.array(las.x, dtype=np.float64),
            np.array(las.y, dtype=np.float64),
            np.array(las.z, dtype=np.float64),
        ])
        cls = np.array(las.classification)
        has_class6 = np.any(cls == ASPRS_BUILDING)
        has_class2 = np.any(cls == ASPRS_GROUND)
        cands = xyz[cls == ASPRS_BUILDING] if has_class6 else xyz[~np.isin(cls, list(NON_BUILDING_CLASSES))]
        all_candidates.append(cands)
        if has_class2:
            all_ground.append(xyz[cls == ASPRS_GROUND])

    candidate_pts = np.vstack(all_candidates) if all_candidates else np.empty((0, 3))
    ground_pts    = np.vstack(all_ground)     if all_ground     else np.empty((0, 3))
    tile_ground_z = float(np.median(ground_pts[:, 2])) if len(ground_pts) else float(np.min(candidate_pts[:, 2]))

    cand_lons, cand_lats = transformer.transform(candidate_pts[:, 0], candidate_pts[:, 1])
    clipped = candidate_pts[contains_xy(polygon, cand_lons, cand_lats)]

    click.echo(f"  {len(clipped)} building points from {len(candidate_pts)} candidates across {len(laz_paths)} tile(s)")

    if len(clipped) < 10:
        click.echo(
            "Error: fewer than 10 points inside the building footprint. "
            "The LiDAR data may not cover this building.",
            err=True,
        )
        sys.exit(1)

    # Per-building ground elevation from local class-2 points
    if len(ground_pts) > 0:
        gnd_lons, gnd_lats = transformer.transform(ground_pts[:, 0], ground_pts[:, 1])
        gnd_mask = contains_xy(polygon, gnd_lons, gnd_lats)
        ground_z = float(np.median(ground_pts[gnd_mask, 2])) if gnd_mask.sum() >= 3 else tile_ground_z
    else:
        ground_z = tile_ground_z

    # ── 4. Run pipeline ───────────────────────────────────────────────────────
    click.echo("Running roof measurement pipeline...")
    try:
        result = process_building(
            osm_id, clipped, ground_z,
            distance_threshold=distance_threshold,
            min_facet_area_m2=min_facet_area,
            min_facet_points=min_facet_points,
            max_planes=max_planes,
        )
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # ── 5. Save + print ───────────────────────────────────────────────────────
    if output is None:
        output = Path(f"{osm_id}_roof.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.model_dump_json(indent=2))

    click.echo(f"\nWritten → {output}")
    click.echo(f"  Facets:      {result.num_facets}")
    click.echo(f"  Height:      {result.height_m:.2f} m")
    click.echo(f"  Eave height: {result.eave_height_m:.2f} m")
    click.echo(f"  Method:      {result.segmentation_method}")
    for f in result.facets:
        flag = " [FLAT]" if f.is_flat else ""
        click.echo(
            f"  Facet {f.facet_id}: pitch={f.pitch_deg:.1f}°  "
            f"azimuth={f.azimuth_deg:.0f}°  area={f.area_m2:.1f} m²  "
            f"conf={f.confidence:.2f}{flag}"
        )


# ---------------------------------------------------------------------------
# Entry point (kept for backwards-compat: `roof-measure <file>` still works
# via the legacy alias below)
# ---------------------------------------------------------------------------

# Backwards-compatible alias: `roof-measure building.laz` still invokes `process`.
# Register a standalone `main` that wraps the group so pyproject entry points work.
def main() -> None:
    cli()
