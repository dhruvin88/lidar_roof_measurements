"""LiDAR point cloud loading and building point extraction."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from roof_measurements.constants import ASPRS_BUILDING, ASPRS_GROUND, NON_BUILDING_CLASSES

logger = logging.getLogger(__name__)


def load_building_points(
    path: str | Path,
    max_points: int = 100_000,
) -> tuple[np.ndarray, float]:
    """Load a LAS/LAZ file and return (building_points_xyz, ground_elevation).

    Auto-detects classification:
    - If class 6 (building) points exist, uses them directly.
    - Elif class 2 (ground) exists, derives candidates by height above ground.
    - Otherwise, runs CSF ground/non-ground separation.

    If the resulting point set exceeds *max_points*, a random subsample is
    returned to keep processing time manageable.

    Returns
    -------
    points : np.ndarray, shape (N, 3)
        XYZ coordinates of building / roof points in the file's native CRS.
    ground_elevation : float
        Median Z of ground points (metres), used as height reference.
    """
    import laspy

    path = Path(path)
    logger.info("Loading %s", path)

    with laspy.open(path) as f:
        las = f.read()

    xyz = np.column_stack([
        las.x.scaled_array(),
        las.y.scaled_array(),
        las.z.scaled_array(),
    ])

    classification = np.array(las.classification)
    has_building_class = np.any(classification == ASPRS_BUILDING)
    has_ground_class = np.any(classification == ASPRS_GROUND)

    if has_building_class:
        logger.info("Using ASPRS class 6 building points")
        building_mask = classification == ASPRS_BUILDING

        if has_ground_class:
            ground_z = float(np.median(xyz[classification == ASPRS_GROUND, 2]))
        else:
            ground_z = float(np.min(xyz[:, 2]))

        building_points = xyz[building_mask]
    elif has_ground_class:
        logger.info("No class 6 found — deriving candidates from height above ground (class 2)")
        building_points, ground_z = _height_based_separate(xyz, classification)
    else:
        logger.info("No classification found — running CSF ground filter")
        building_points, ground_z = _csf_separate(xyz)

    if len(building_points) > max_points:
        original_count = len(building_points)
        rng = np.random.default_rng(0)
        idx = rng.choice(original_count, max_points, replace=False)
        building_points = building_points[idx]
        logger.warning(
            "Subsampled to %d points (original: %d) for tractable processing.",
            max_points, original_count,
        )

    logger.info(
        "Extracted %d building points, ground elev=%.2fm",
        len(building_points),
        ground_z,
    )
    return building_points, ground_z


def _height_based_separate(
    xyz: np.ndarray,
    classification: np.ndarray,
    min_height_m: float = 2.5,
) -> tuple[np.ndarray, float]:
    """Derive above-ground candidates using class 2 (ground) as reference.

    Returns points that are at least *min_height_m* above the median ground
    elevation, excluding known non-building classes (ground, noise, water).
    """
    ground_z = float(np.median(xyz[classification == ASPRS_GROUND, 2]))
    height_above = xyz[:, 2] - ground_z
    exclude = np.array(list(NON_BUILDING_CLASSES))
    candidate_mask = ~np.isin(classification, exclude) & (height_above >= min_height_m)
    candidates = xyz[candidate_mask]
    if len(candidates) == 0:
        candidate_mask = ~np.isin(classification, exclude) & (height_above > 0)
        candidates = xyz[candidate_mask]
    logger.info(
        "Height-based filter: %d candidates (>= %.1fm above ground @ %.2fm)",
        len(candidates), min_height_m, ground_z,
    )
    return candidates, ground_z


def _csf_separate(xyz: np.ndarray) -> tuple[np.ndarray, float]:
    """Use Cloth Simulation Filter to separate ground from non-ground."""
    try:
        import CSF
    except ImportError as e:
        raise ImportError(
            "CSF package is required for unclassified point clouds. "
            "Install it with: pip install CSF"
        ) from e

    csf = CSF.CSF()
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = 0.5
    csf.params.rigidness = 3
    csf.params.time_step = 0.65
    csf.params.class_threshold = 0.5
    csf.params.iterations = 500

    csf.setPointCloud(xyz.tolist())

    ground_indices = CSF.VecInt()
    non_ground_indices = CSF.VecInt()
    csf.do_filtering(ground_indices, non_ground_indices)

    ground_idx = np.array(ground_indices)
    non_ground_idx = np.array(non_ground_indices)

    ground_z = float(np.median(xyz[ground_idx, 2])) if len(ground_idx) > 0 else float(np.min(xyz[:, 2]))
    building_points = xyz[non_ground_idx]

    return building_points, ground_z
