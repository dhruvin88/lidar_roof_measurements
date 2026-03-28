"""Roof plane segmentation: RANSAC with region-growing fallback.

Uses pyransac3d for iterative plane extraction (no open3d dependency).
Region growing uses PCA-based normal estimation via numpy + scikit-learn KDTree.
"""

from __future__ import annotations

import collections
import logging

import numpy as np

logger = logging.getLogger(__name__)

# If RANSAC yields only 1 facet but the point cloud spans more than this
# height range (metres), suspect a complex roof and trigger region-growing.
_FLAT_ROOF_HEIGHT_RANGE_M = 0.5


def segment_planes(
    points: np.ndarray,
    distance_threshold: float = 0.15,
    min_facet_area_m2: float = 1.0,
    min_facet_points: int = 10,
    max_planes: int = 20,
    ransac_n: int = 3,
    num_iterations: int = 2000,
) -> tuple[list[np.ndarray], str]:
    """Segment roof points into planar facets.

    Strategy:
    1. Run iterative RANSAC (dominant plane removal).
    2. If result has only 1 facet on a non-flat roof (height range > threshold),
       fall back to region growing on normals.

    Returns
    -------
    facet_point_lists : list of np.ndarray  (each shape (M, 3))
    method : str  —  'ransac' or 'region_growing'
    """
    facets = _ransac_segment(
        points,
        distance_threshold=distance_threshold,
        min_facet_area_m2=min_facet_area_m2,
        min_facet_points=min_facet_points,
        max_planes=max_planes,
        num_iterations=num_iterations,
    )

    height_range = float(np.max(points[:, 2]) - np.min(points[:, 2]))
    is_non_flat = height_range > _FLAT_ROOF_HEIGHT_RANGE_M

    if len(facets) <= 1 and is_non_flat:
        logger.info(
            "RANSAC found %d facet(s) but height range=%.2fm suggests complex roof. "
            "Falling back to region growing.",
            len(facets),
            height_range,
        )
        rg_facets = _region_growing_segment(
            points,
            min_facet_area_m2=min_facet_area_m2,
            min_facet_points=min_facet_points,
        )
        if len(rg_facets) > len(facets):
            return rg_facets, "region_growing"

    return facets, "ransac"


def _ransac_segment(
    points: np.ndarray,
    distance_threshold: float,
    min_facet_area_m2: float,
    min_facet_points: int,
    max_planes: int,
    num_iterations: int,
) -> list[np.ndarray]:
    """Iteratively extract dominant planes via RANSAC using pyransac3d."""
    import pyransac3d as pyrsc

    remaining = points.copy()
    facets: list[np.ndarray] = []

    for _ in range(max_planes):
        if len(remaining) < 3:
            break

        plane = pyrsc.Plane()
        _, inlier_idx = plane.fit(remaining, thresh=distance_threshold, maxIteration=num_iterations)
        inlier_idx = np.asarray(inlier_idx, dtype=int)

        inlier_pts = remaining[inlier_idx]
        area = _xy_area(inlier_pts)

        if len(inlier_pts) >= min_facet_points and area >= min_facet_area_m2:
            facets.append(inlier_pts)
            logger.debug(
                "RANSAC plane %d: %d pts, area=%.1f m²",
                len(facets), len(inlier_pts), area,
            )

        outlier_mask = np.ones(len(remaining), dtype=bool)
        outlier_mask[inlier_idx] = False
        remaining = remaining[outlier_mask]

        if len(remaining) < max(10, len(points) * 0.05):
            break

    logger.info("RANSAC: found %d facets", len(facets))
    return facets


def _estimate_normals_pca(points: np.ndarray, nn_idx: np.ndarray) -> np.ndarray:
    """Estimate per-point normals via PCA on k nearest neighbours.

    Uses batched 3x3 covariance matrices + eigh instead of per-point SVD
    to avoid Python-loop overhead on large point clouds.
    """
    # (N, k, 3) neighbourhoods, centred per-point
    nbr_pts = points[nn_idx]                          # (N, k, 3)
    centred = nbr_pts - nbr_pts.mean(axis=1, keepdims=True)  # (N, k, 3)

    # Batched 3×3 covariance: C_i = centred_i^T @ centred_i
    cov = np.einsum("nki,nkj->nij", centred, centred)  # (N, 3, 3)

    # Smallest eigenvector of each covariance = plane normal
    eigenvalues, eigenvectors = np.linalg.eigh(cov)     # sorted ascending
    normals = eigenvectors[:, :, 0]                     # (N, 3) — smallest eigenvalue

    # Orient upward (z > 0)
    flip = normals[:, 2] < 0
    normals[flip] *= -1

    # Normalise
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    normals /= norms

    return normals


def _region_growing_segment(
    points: np.ndarray,
    min_facet_area_m2: float,
    min_facet_points: int = 10,
    k_neighbors: int = 20,
    angle_threshold_deg: float = 15.0,
    curvature_threshold: float = 0.1,
) -> list[np.ndarray]:
    """Region growing segmentation based on normal vector similarity."""
    from sklearn.neighbors import KDTree

    tree = KDTree(points)
    _, nn_idx = tree.query(points, k=k_neighbors)

    normals = _estimate_normals_pca(points, nn_idx)
    angle_thresh_rad = np.deg2rad(angle_threshold_deg)

    n_pts = len(points)
    labels = np.full(n_pts, -1, dtype=int)
    current_label = 0

    # Seed from flattest points first (lowest curvature)
    curvatures = _estimate_curvature(normals, nn_idx)
    seed_order = np.argsort(curvatures)

    for seed in seed_order:
        if labels[seed] != -1:
            continue

        queue = collections.deque([seed])
        labels[seed] = current_label

        while queue:
            pt = queue.popleft()
            for nb in nn_idx[pt]:
                if labels[nb] != -1:
                    continue
                angle = np.arccos(np.clip(np.abs(np.dot(normals[pt], normals[nb])), 0.0, 1.0))
                if angle < angle_thresh_rad and curvatures[nb] < curvature_threshold:
                    labels[nb] = current_label
                    queue.append(nb)

        current_label += 1

    facets: list[np.ndarray] = []
    for label in range(current_label):
        mask = labels == label
        facet_pts = points[mask]
        if len(facet_pts) >= min_facet_points and _xy_area(facet_pts) >= min_facet_area_m2:
            facets.append(facet_pts)

    logger.info("Region growing: found %d facets", len(facets))
    return facets


def _estimate_curvature(normals: np.ndarray, nn_idx: np.ndarray) -> np.ndarray:
    """Approximate per-point curvature as mean angular deviation from neighbours."""
    diffs = normals[nn_idx] - normals[:, None, :]  # (N, k, 3)
    return np.mean(np.linalg.norm(diffs, axis=2), axis=1)


def _xy_area(points: np.ndarray) -> float:
    """Approximate XY-projected area via convex hull (m²)."""
    from scipy.spatial import ConvexHull

    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points[:, :2])
        return float(hull.volume)  # 'volume' = area for 2D hull
    except Exception:
        return 0.0
