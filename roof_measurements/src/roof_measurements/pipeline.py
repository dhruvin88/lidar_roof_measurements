"""Top-level pipeline: load → segment → extract → return BuildingResult."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from roof_measurements.features import (
    classify_roof_type,
    compute_continuity,
    compute_facet,
    compute_height,
    estimate_point_density,
    facet_solar_irradiance,
    filter_below_eave,
    filter_isolated_facets,
    filter_radius_outliers,
    filter_subground_points,
    merge_coplanar_facets,
)
from roof_measurements.io import load_building_points
from roof_measurements.models import BuildingResult
from roof_measurements.segmentation import segment_planes

logger = logging.getLogger(__name__)

_LOW_DENSITY_THRESHOLD = 4.0  # pts/m²


def assemble_result(
    building_id: str,
    facet_point_lists: list[np.ndarray],
    all_building_points: np.ndarray,
    ground_z: float,
    density: float,
    method: str,
    min_confidence: float = 0.0,
    max_pitch_deg: float = 70.0,
    lat: float | None = None,
) -> tuple[BuildingResult, list[np.ndarray]]:
    """Merge, filter, and assemble facets into a BuildingResult.

    Returns (result, final_facet_point_lists) — the second element is needed
    by callers that do 3D visualisation or obstacle detection.
    """
    # Merge RANSAC fragments that belong to the same physical plane
    facet_point_lists = merge_coplanar_facets(facet_point_lists)

    facets = [compute_facet(i + 1, pts) for i, pts in enumerate(facet_point_lists)]

    n_before = len(facets)
    kept = [
        (f, pts) for f, pts in zip(facets, facet_point_lists)
        if f.confidence >= min_confidence and f.pitch_deg <= max_pitch_deg
    ]
    if not kept:
        raise ValueError(
            f"Building {building_id!r}: no facets survived quality filters "
            f"(min_confidence={min_confidence:.2f}, max_pitch={max_pitch_deg:.0f}°) "
            f"— try relaxing thresholds."
        )
    if len(kept) < n_before:
        facets = [f.model_copy(update={"facet_id": i + 1}) for i, (f, _) in enumerate(kept)]
        facet_point_lists = [pts for _, pts in kept]
        logger.info("Quality filter: kept %d / %d facets", len(facets), n_before)

    # Remove facets not connected to the main roof structure
    filtered_point_lists = filter_isolated_facets(facet_point_lists)
    if len(filtered_point_lists) < len(facets):
        # Keep only facets whose points survived filtering
        kept_pts_set = {id(pts) for pts in filtered_point_lists}
        paired = [(f, pts) for f, pts in zip(facets, facet_point_lists) if id(pts) in kept_pts_set]
        facets = [f.model_copy(update={"facet_id": i + 1}) for i, (f, _) in enumerate(paired)]
        facet_point_lists = [pts for _, pts in paired]
    else:
        facet_point_lists = filtered_point_lists

    # Solar potential (requires latitude; skipped when lat is None)
    if lat is not None:
        updated = []
        for f in facets:
            kwh, suit = facet_solar_irradiance(f.pitch_deg, f.azimuth_deg, lat)
            updated.append(f.model_copy(update={"solar_kwh_m2_yr": kwh, "solar_suitability": suit}))
        facets = updated
        total_solar_kwh_yr: float | None = round(
            sum(f.solar_kwh_m2_yr * f.area_m2 for f in facets), 1  # type: ignore[operator]
        )
    else:
        total_solar_kwh_yr = None

    all_roof_points = np.vstack(facet_point_lists)
    height_m, ridge_elev = compute_height(all_roof_points, ground_z)
    eave_height_m = max(0.0, round(min(f.eave_elevation_m for f in facets) - float(ground_z), 3))
    total_roof_area_m2 = round(sum(f.area_m2 for f in facets), 2)
    roof_type = classify_roof_type(facets)
    continuity = compute_continuity(facet_point_lists, all_building_points)
    result = BuildingResult(
        building_id=building_id,
        num_facets=len(facets),
        height_m=height_m,
        eave_height_m=eave_height_m,
        ground_elevation_m=round(float(ground_z), 3),
        ridge_elevation_m=ridge_elev,
        facets=facets,
        point_density_m2=density,
        segmentation_method=method,
        unassigned_point_fraction=continuity["unassigned_point_fraction"],
        is_facets_connected=continuity["is_connected"],
        num_facet_components=continuity["num_components"],
        isolated_facet_ids=continuity["isolated_facet_ids"],
        total_roof_area_m2=total_roof_area_m2,
        roof_type=roof_type,
        total_solar_kwh_yr=total_solar_kwh_yr,
    )
    return result, facet_point_lists


def _preprocess_and_segment(
    building_id: str,
    points: np.ndarray,
    ground_z: float,
    distance_threshold: float,
    min_facet_area_m2: float,
    min_facet_points: int,
    max_planes: int,
) -> tuple[np.ndarray, list[np.ndarray], str, float]:
    """Shared preprocessing: filter → density → segment.

    Returns (filtered_points, facet_point_lists, method, density).
    """
    if len(points) == 0:
        raise ValueError(f"Building {building_id!r}: empty point array")

    points = filter_subground_points(points, ground_z)
    if len(points) == 0:
        raise ValueError(f"Building {building_id!r}: no points above ground elevation")

    points = filter_radius_outliers(points)
    if len(points) == 0:
        raise ValueError(f"Building {building_id!r}: no points remain after outlier filtering")

    points = filter_below_eave(points, ground_z)

    density = estimate_point_density(points)
    if density < _LOW_DENSITY_THRESHOLD:
        logger.warning(
            "Building %s: low point density (%.1f pts/m²) — accuracy may be reduced.",
            building_id, density,
        )

    facet_point_lists, method = segment_planes(
        points,
        distance_threshold=distance_threshold,
        min_facet_area_m2=min_facet_area_m2,
        min_facet_points=min_facet_points,
        max_planes=max_planes,
    )

    if not facet_point_lists:
        raise ValueError(
            f"Building {building_id!r}: no facets detected — "
            "try lowering min_facet_area_m2 or min_facet_points."
        )

    return points, facet_point_lists, method, density


def process_building(
    building_id: str,
    points: np.ndarray,
    ground_z: float,
    distance_threshold: float = 0.15,
    min_facet_area_m2: float = 1.0,
    min_facet_points: int = 10,
    max_planes: int = 20,
    min_confidence: float = 0.0,
    max_pitch_deg: float = 70.0,
    lat: float | None = None,
) -> BuildingResult:
    """Run segment → feature extraction on a pre-loaded point array.

    Use this when you already have a clipped point cloud (e.g. from
    :func:`roof_measurements.footprints.iter_building_point_clouds`).

    Parameters
    ----------
    building_id :
        Identifier to embed in the result.
    points :
        (N, 3) XYZ array of roof/building points.
    ground_z :
        Ground elevation reference (metres).
    distance_threshold, min_facet_area_m2, max_planes :
        Forwarded to :func:`segment_planes`.
    """
    points, facet_point_lists, method, density = _preprocess_and_segment(
        building_id, points, ground_z, distance_threshold,
        min_facet_area_m2, min_facet_points, max_planes,
    )
    result, _ = assemble_result(building_id, facet_point_lists, points, ground_z, density, method, min_confidence, max_pitch_deg, lat)
    return result


def process_file(
    path: str | Path,
    building_id: str | None = None,
    distance_threshold: float = 0.15,
    min_facet_area_m2: float = 1.0,
    min_facet_points: int = 10,
    max_planes: int = 20,
    min_confidence: float = 0.0,
    max_pitch_deg: float = 70.0,
    lat: float | None = None,
) -> BuildingResult:
    """Full pipeline: LAS/LAZ → BuildingResult.

    Parameters
    ----------
    path : str or Path
        Input LAS or LAZ file.
    building_id : str, optional
        Identifier to embed in results. Defaults to the filename stem.
    distance_threshold : float
        RANSAC inlier distance tolerance (metres).
    min_facet_area_m2 : float
        Minimum facet area to keep (m²). Default 1.0.
    max_planes : int
        Maximum number of planes to extract.
    """
    path = Path(path)
    building_id = building_id or path.stem

    points, ground_z = load_building_points(path)

    points, facet_point_lists, method, density = _preprocess_and_segment(
        building_id, points, ground_z, distance_threshold,
        min_facet_area_m2, min_facet_points, max_planes,
    )
    result, _ = assemble_result(building_id, facet_point_lists, points, ground_z, density, method, min_confidence, max_pitch_deg, lat)
    return result
