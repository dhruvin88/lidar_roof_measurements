"""Batch-process a LiDAR tile: fetch OSM footprints, run per-building pipeline,
save individual JSON results + combined GeoJSON and CSV.

Usage
-----
    python scripts/batch_process.py data/austin_buildings_sample.laz --epsg 32614
    python scripts/batch_process.py data/nyc_sandy.laz --epsg 32618 --max-buildings 5
    python scripts/batch_process.py data/sf_sample.laz --epsg 32610 --out results/sf
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from roof_measurements.export import results_to_geodataframe, to_csv, to_geojson
from roof_measurements.footprints import (
    fetch_osm_buildings,
    iter_building_point_clouds,
    las_wgs84_bbox,
)
from roof_measurements.pipeline import process_building


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("las_file", type=Path, help="Input LAS/LAZ file")
    p.add_argument("--epsg", type=int, required=True, help="EPSG code of the file's projected CRS")
    p.add_argument("--out", type=Path, default=None, help="Output directory (default: results/<stem>)")
    p.add_argument("--max-buildings", type=int, default=None, help="Stop after N buildings (useful for quick tests)")
    p.add_argument("--min-points", type=int, default=50, help="Skip buildings with fewer LiDAR points")
    p.add_argument("--min-facet-area", type=float, default=1.0, help="Min facet area in m²")
    p.add_argument("--distance-threshold", type=float, default=0.15, help="RANSAC inlier tolerance (m)")
    p.add_argument("--max-planes", type=int, default=20, help="Max roof planes per building")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    las_path: Path = args.las_file.resolve()
    if not las_path.exists():
        print(f"ERROR: File not found: {las_path}", file=sys.stderr)
        sys.exit(1)

    out_dir: Path = args.out or (Path("results") / las_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Fetch footprints ──────────────────────────────────────────────────────
    print(f"File : {las_path}")
    print(f"EPSG : {args.epsg}")
    print(f"Out  : {out_dir}")

    west, south, east, north = las_wgs84_bbox(las_path, args.epsg)
    print(f"Bbox : W={west:.5f}  S={south:.5f}  E={east:.5f}  N={north:.5f}")

    footprints = fetch_osm_buildings(west, south, east, north)
    print(f"OSM  : {len(footprints)} footprint(s) found")

    if len(footprints) == 0:
        print("No buildings found — nothing to process.", file=sys.stderr)
        sys.exit(1)

    # ── Per-building loop ─────────────────────────────────────────────────────
    results, skipped = [], 0
    print()
    print(f"{'Building':16s}  {'Facets':>6}  {'H (m)':>7}  {'Eave (m)':>8}  {'Pitch°':>7}  {'Conf':>5}  Method")
    print("-" * 78)

    for bldg_id, xyz, ground_z in iter_building_point_clouds(
        las_path, epsg=args.epsg, min_points=args.min_points
    ):
        try:
            r = process_building(
                bldg_id, xyz, ground_z,
                distance_threshold=args.distance_threshold,
                min_facet_area_m2=args.min_facet_area,
                max_planes=args.max_planes,
            )
        except Exception as exc:
            print(f"  {bldg_id:14s}  SKIPPED: {exc}", file=sys.stderr)
            skipped += 1
            continue

        results.append(r)

        mean_pitch = sum(f.pitch_deg for f in r.facets) / r.num_facets
        mean_conf  = sum(f.confidence  for f in r.facets) / r.num_facets
        print(
            f"  {bldg_id:14s}  {r.num_facets:6d}  {r.height_m:7.2f}  "
            f"{r.eave_height_m:8.2f}  {mean_pitch:7.1f}  {mean_conf:5.2f}  {r.segmentation_method}"
        )

        # Save individual JSON
        json_path = out_dir / f"{bldg_id}.json"
        json_path.write_text(r.model_dump_json(indent=2))

        if args.max_buildings and len(results) >= args.max_buildings:
            print(f"\n[--max-buildings {args.max_buildings} reached, stopping early]")
            break

    print("-" * 78)
    print(f"\nProcessed {len(results)} building(s), {skipped} skipped.")

    if not results:
        print("No results to export.", file=sys.stderr)
        sys.exit(1)

    # ── Export combined outputs ───────────────────────────────────────────────
    gdf = results_to_geodataframe(results, footprints)

    geojson_path = to_geojson(gdf, out_dir / f"{las_path.stem}_roofs.geojson")
    csv_path     = to_csv(gdf, out_dir / f"{las_path.stem}_roofs.csv", include_wkt=True)

    print(f"\nGeoJSON → {geojson_path}")
    print(f"CSV     → {csv_path}")
    print(f"JSON    → {out_dir}/<building_id>.json  ({len(results)} files)")


if __name__ == "__main__":
    main()
