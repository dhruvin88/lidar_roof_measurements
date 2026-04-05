"""Extract geometric features from segmented roof facets."""

from __future__ import annotations

import logging

import numpy as np

from roof_measurements.models import FacetResult, ObstacleResult

logger = logging.getLogger(__name__)

FLAT_PITCH_THRESHOLD_DEG = 5.0
_ALPHA_CIRCUMRADIUS_MULTIPLIER = 4.0   # keep triangles with R < this × median_NN
_EAVE_PERCENTILE = 2    # low enough to catch eave edge, robust to noise
_RIDGE_PERCENTILE = 99  # high enough for ridge, robust to outlier spikes
# RMS distance at which confidence drops to 0.  Equal to 2× the default
# RANSAC inlier threshold (0.15 m) — planes worse than this are unreliable.
_CONF_RMS_CAP_M = 0.30


def fit_plane_normal(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit a plane to points via PCA.

    Returns
    -------
    normal : np.ndarray
        Unit normal, flipped so Z is always positive (pointing up).
    plane_rms_m : float
        RMS of signed point-to-plane distances (metres).  Lower = better fit.
    """
    centroid = points.mean(axis=0)
    centred = points - centroid
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    normal = vt[-1]  # eigenvector with smallest singular value = plane normal
    if normal[2] < 0:
        normal = -normal
    normal = normal / np.linalg.norm(normal)

    residuals = centred @ normal          # signed distance of each point from plane
    plane_rms_m = float(np.sqrt(np.mean(residuals ** 2)))
    return normal, plane_rms_m


def planarity_confidence(plane_rms_m: float) -> float:
    """Convert RMS residual to a 0–1 confidence score.

    Linearly maps [0, _CONF_RMS_CAP_M] → [1.0, 0.0] and clips outside.
    """
    return round(float(np.clip(1.0 - plane_rms_m / _CONF_RMS_CAP_M, 0.0, 1.0)), 3)


def pitch_from_normal(normal: np.ndarray) -> float:
    """Return pitch angle in degrees (0=flat, 90=vertical)."""
    nz = float(np.clip(np.abs(normal[2]), 0.0, 1.0))
    return float(np.degrees(np.arccos(nz)))


def azimuth_from_normal(normal: np.ndarray) -> float:
    """Return downslope azimuth in degrees (0=North/+Y, 90=East/+X, clockwise)."""
    nx, ny = float(normal[0]), float(normal[1])
    # Downslope direction: opposite to upward-projected normal in XY
    azimuth = float(np.degrees(np.arctan2(-nx, -ny))) % 360.0
    return azimuth


def _project_to_plane(
    points: np.ndarray, normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project 3D *points* onto the plane defined by *normal*.

    Returns ``(proj_2d, u_basis, v_basis, centroid_3d)``.
    Inverse: ``3D = centroid + proj[:, 0:1] * u + proj[:, 1:2] * v``
    """
    centroid = points.mean(axis=0)
    centred  = points - centroid
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref); u /= np.linalg.norm(u)
    v = np.cross(normal, u);   v /= np.linalg.norm(v)
    return np.column_stack([centred @ u, centred @ v]), u, v, centroid


def _delaunay_alpha_kept(
    proj: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Delaunay triangulation of 2-D *proj* filtered by alpha-shape circumradius.

    Keeps triangles whose circumradius < ``_ALPHA_CIRCUMRADIUS_MULTIPLIER`` ×
    median nearest-neighbour distance.

    Returns ``(kept_simplices, triangle_areas)`` or ``(None, None)`` on failure.
    """
    from scipy.spatial import Delaunay, cKDTree

    if len(proj) < 3:
        return None, None

    tree = cKDTree(proj)
    nn_d, _ = tree.query(proj, k=2)
    median_nn = float(np.median(nn_d[:, 1]))
    if median_nn < 1e-8:
        return None, None

    r_threshold = _ALPHA_CIRCUMRADIUS_MULTIPLIER * median_nn

    try:
        tri = Delaunay(proj)
    except Exception:
        return None, None

    t = proj[tri.simplices]
    a = np.linalg.norm(t[:, 1] - t[:, 0], axis=1)
    b = np.linalg.norm(t[:, 2] - t[:, 1], axis=1)
    c = np.linalg.norm(t[:, 0] - t[:, 2], axis=1)
    cross = (
        (t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
        - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0])
    )
    areas = 0.5 * np.abs(cross)
    valid = areas > 1e-12
    R = np.full(len(t), np.inf)
    R[valid] = (a[valid] * b[valid] * c[valid]) / (4.0 * areas[valid])
    mask = R < r_threshold
    return tri.simplices[mask], areas[mask]


def projected_area(points: np.ndarray, normal: np.ndarray) -> float:
    """Facet area via alpha-shape of projected points (m²).

    More accurate than convex hull for L-shaped or irregular facets.
    Falls back to convex hull for small point sets or degenerate input.
    """
    from scipy.spatial import ConvexHull

    if len(points) < 3:
        return 0.0

    proj, *_ = _project_to_plane(points, normal)

    if len(proj) < 10:
        try:
            return float(ConvexHull(proj).volume)
        except Exception:
            return 0.0

    kept_simplices, areas = _delaunay_alpha_kept(proj)
    if kept_simplices is None:
        try:
            return float(ConvexHull(proj).volume)
        except Exception:
            return 0.0

    total = float(areas.sum())
    if total < 1e-6:
        try:
            return float(ConvexHull(proj).volume)
        except Exception:
            return 0.0
    return total


def compute_eave_elevation(points: np.ndarray) -> float:
    """Return the eave elevation as the low-percentile Z of facet points.

    Using a low percentile rather than the minimum avoids noise spikes while
    still capturing the true bottom edge of the facet.
    """
    return round(float(np.percentile(points[:, 2], _EAVE_PERCENTILE)), 3)


def compute_facet(facet_id: int, points: np.ndarray) -> FacetResult:
    """Compute all features for a single facet."""
    normal, plane_rms_m = fit_plane_normal(points)
    pitch = pitch_from_normal(normal)
    azimuth = azimuth_from_normal(normal)
    area = projected_area(points, normal)
    is_flat = pitch < FLAT_PITCH_THRESHOLD_DEG
    eave_elev = compute_eave_elevation(points)

    return FacetResult(
        facet_id=facet_id,
        pitch_deg=round(pitch, 2),
        azimuth_deg=round(azimuth, 1),
        area_m2=round(area, 2),
        normal_vector=[round(float(n), 4) for n in normal],
        num_points=len(points),
        is_flat=is_flat,
        eave_elevation_m=eave_elev,
        plane_rms_m=round(plane_rms_m, 4),
        confidence=planarity_confidence(plane_rms_m),
    )


def compute_height(all_roof_points: np.ndarray, ground_elevation_m: float) -> tuple[float, float]:
    """Return (height_m, ridge_elevation_m).

    height_m = 99th-percentile Z of roof points minus ground_elevation_m
    (using 99th percentile avoids noise spikes from outlier points).
    """
    ridge_elev = float(np.percentile(all_roof_points[:, 2], _RIDGE_PERCENTILE))
    height = ridge_elev - ground_elevation_m
    return round(height, 3), round(ridge_elev, 3)


_GABLE_OPPOSITION_MIN_DEG = 150   # min azimuth separation to count as "opposite"
_MANSARD_PITCH_GAP_DEG   = 20    # bimodal pitch gap that signals mansard


def merge_coplanar_facets(
    facet_point_lists: list[np.ndarray],
    angle_threshold_deg: float = 8.0,
    adjacency_gap_m: float = 1.0,
    z_tolerance_m: float = 2.0,
) -> list[np.ndarray]:
    """Merge adjacent facets whose normals and heights are similar.

    RANSAC often splits a single physical roof plane into 2–3 fragments.
    Merging them gives the correct facet count, better area, and better
    pitch/azimuth (fitted to all points of the plane rather than a fragment).

    The *z_tolerance_m* check prevents merging flat facets at different
    building levels (e.g. stepped commercial roofs).
    """
    from collections import deque

    n = len(facet_point_lists)
    if n <= 1:
        return facet_point_lists

    # Fit a unit normal to each facet
    normals: list[np.ndarray] = []
    median_z: list[float] = []
    for pts in facet_point_lists:
        centred = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        normal = vt[-1]
        if normal[2] < 0:
            normal = -normal
        normals.append(normal / np.linalg.norm(normal))
        median_z.append(float(np.median(pts[:, 2])))

    adjacency = _xy_hull_adjacency(facet_point_lists, adjacency_gap_m)
    angle_thresh_rad = np.deg2rad(angle_threshold_deg)

    merge_graph: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in adjacency[i]:
            if j <= i:
                continue
            # Normal similarity check
            cos_angle = float(np.clip(np.abs(np.dot(normals[i], normals[j])), 0.0, 1.0))
            if np.arccos(cos_angle) > angle_thresh_rad:
                continue
            # Height similarity check (prevents merging across building levels)
            if abs(median_z[i] - median_z[j]) > z_tolerance_m:
                continue
            merge_graph[i].append(j)
            merge_graph[j].append(i)

    visited = [False] * n
    merged: list[np.ndarray] = []
    for start in range(n):
        if visited[start]:
            continue
        group: list[int] = []
        q: deque[int] = deque([start])
        visited[start] = True
        while q:
            node = q.popleft()
            group.append(node)
            for nb in merge_graph[node]:
                if not visited[nb]:
                    visited[nb] = True
                    q.append(nb)
        merged.append(np.vstack([facet_point_lists[i] for i in group]))

    logger.debug("merge_coplanar_facets: %d → %d facets", n, len(merged))
    return merged


def classify_roof_type(facets: list[FacetResult]) -> str:
    """Classify roof type from the filtered facet list.

    Returns one of: flat / shed / gable / hip / mansard / complex
    """
    if not facets:
        return "unknown"

    total_area = sum(f.area_m2 for f in facets)
    if total_area == 0:
        return "unknown"

    non_flat = [f for f in facets if not f.is_flat]

    # Flat roof (or nearly flat with small parapet/equipment facets)
    flat_area = total_area - sum(f.area_m2 for f in non_flat)
    if not non_flat or flat_area / total_area > 0.60:
        return "flat"

    n = len(non_flat)
    pitches  = np.array([f.pitch_deg  for f in non_flat])
    azimuths = np.array([f.azimuth_deg for f in non_flat])

    # Mansard: clear bimodal pitch distribution (gap > threshold between groups)
    if n >= 4:
        sorted_p = np.sort(pitches)
        if float(np.diff(sorted_p).max()) > _MANSARD_PITCH_GAP_DEG:
            return "mansard"

    if n == 1:
        return "shed"

    # Count opposite-facing pairs (gable/hip signature)
    opposite_pairs = 0
    perp_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(azimuths[i] - azimuths[j])
            diff = min(diff, 360.0 - diff)
            if diff >= _GABLE_OPPOSITION_MIN_DEG:
                opposite_pairs += 1
            elif 70.0 <= diff <= 110.0:
                perp_pairs += 1

    if n == 2:
        if opposite_pairs == 1 and abs(pitches[0] - pitches[1]) < 15:
            return "gable"
        return "shed"

    # Hip: opposite pairs + perpendicular pairs (triangular end facets)
    if opposite_pairs >= 1 and perp_pairs >= 1:
        return "hip"

    # Gable with dormers / additions
    if opposite_pairs >= 1:
        return "gable"

    return "complex"


def filter_subground_points(points: np.ndarray, ground_z: float) -> np.ndarray:
    """Remove points at or below ground elevation (misclassified noise)."""
    return points[points[:, 2] >= ground_z]


def filter_below_eave(
    points: np.ndarray,
    ground_z: float,
    eave_fraction: float = 0.65,
) -> np.ndarray:
    """Remove wall points below the estimated eave line.

    LiDAR building-class points include walls.  RANSAC picks these up as
    near-vertical facets — they are not roof surfaces.  This filter estimates
    the eave height as *eave_fraction* of the way from ground to ridge and
    discards everything below it.

    A fraction of 0.65 is conservative: for a typical residential building
    (3 m wall, 2 m roof rise → 5 m total) it places the cut at 3.25 m,
    cleanly above the wall while preserving even the lowest eave points.
    """
    if len(points) == 0:
        return points
    ridge_z = float(np.percentile(points[:, 2], 98))
    height = ridge_z - ground_z
    if height <= 0:
        return points
    cutoff_z = ground_z + eave_fraction * height
    kept = points[points[:, 2] >= cutoff_z]
    if len(kept) < 10:
        return points  # safety: never discard everything
    logger.debug(
        "filter_below_eave: ground=%.1f ridge=%.1f cutoff=%.1f → kept %d / %d pts",
        ground_z, ridge_z, cutoff_z, len(kept), len(points),
    )
    return kept


def filter_radius_outliers(
    points: np.ndarray,
    radius_m: float = 1.5,
    min_neighbors: int = 4,
) -> np.ndarray:
    """Remove points that have fewer than *min_neighbors* within *radius_m*.

    This eliminates isolated point clusters from nearby trees, vehicles, or
    misclassified objects that survived the footprint clip and ASPRS filter.
    Roof surface points always form dense clusters and are unaffected.

    Parameters
    ----------
    points :
        (N, 3) XYZ array.
    radius_m :
        Search radius in metres (3-D Euclidean distance).
    min_neighbors :
        Minimum number of *other* points required within the radius.

    Returns
    -------
    np.ndarray
        Filtered subset of *points*.
    """
    from scipy.spatial import cKDTree

    if len(points) <= min_neighbors:
        return points

    tree = cKDTree(points)
    # count_neighbors returns number of points within radius INCLUDING self
    counts = tree.query_ball_point(points, r=radius_m, return_length=True)
    mask = counts > min_neighbors          # strictly more than self + threshold
    kept = points[mask]
    if len(kept) < len(points):
        logger.debug(
            "filter_radius_outliers: removed %d / %d outlier points",
            len(points) - len(kept), len(points),
        )
    return kept


def _xy_hull_adjacency(
    facet_point_lists: list[np.ndarray],
    adjacency_gap_m: float,
) -> list[list[int]]:
    """Build an adjacency list from XY point-cloud proximity.

    For each pair of facets, the minimum 2-D distance between any point in
    one facet and any point in the other is computed via KD-tree query.
    Two facets are adjacent when that minimum distance is within
    *adjacency_gap_m* metres.

    This is more accurate than comparing convex-hull vertices: on large roofs
    the shared boundary between two RANSAC segments often runs between hull
    vertices rather than through them, causing hull-vertex distance to be
    several metres even when the facets physically touch.
    """
    from scipy.spatial import cKDTree

    n = len(facet_point_lists)
    trees = [cKDTree(pts[:, :2]) for pts in facet_point_lists]

    adjacency: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        xy_i = facet_point_lists[i][:, :2]
        for j in range(i + 1, n):
            dists, _ = trees[j].query(xy_i, k=1)
            if float(dists.min()) <= adjacency_gap_m:
                adjacency[i].append(j)
                adjacency[j].append(i)
    return adjacency


def filter_isolated_facets(
    facet_point_lists: list[np.ndarray],
    adjacency_gap_m: float = 1.0,
    min_component_fraction: float = 0.10,
) -> list[np.ndarray]:
    """Return facets from all significant spatially connected groups.

    Removes only tiny stray components (ground clutter, trees, detached
    artefacts) while keeping every roof level of a multi-level building.

    A component is kept when its total point count is at least
    *min_component_fraction* × the largest component's point count.
    The default (10 %) drops genuine strays (which are tiny) but preserves
    lower-level flat roofs that may be 20–80 % the size of the top level.
    """
    from collections import deque

    n = len(facet_point_lists)
    if n <= 1:
        return facet_point_lists

    adjacency = _xy_hull_adjacency(facet_point_lists, adjacency_gap_m)

    # BFS — collect every connected component
    visited = [False] * n
    components: list[list[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        component: list[int] = []
        q: deque[int] = deque([start])
        visited[start] = True
        while q:
            node = q.popleft()
            component.append(node)
            for nb in adjacency[node]:
                if not visited[nb]:
                    visited[nb] = True
                    q.append(nb)
        components.append(component)

    if len(components) == 1:
        return facet_point_lists

    # Size = total points across all facets in the component
    def _component_pts(comp: list[int]) -> int:
        return sum(len(facet_point_lists[i]) for i in comp)

    max_pts = max(_component_pts(c) for c in components)
    threshold = min_component_fraction * max_pts

    kept_indices: list[int] = []
    for comp in components:
        if _component_pts(comp) >= threshold:
            kept_indices.extend(comp)

    n_removed = n - len(kept_indices)
    if n_removed:
        logger.info(
            "filter_isolated_facets: removed %d / %d facet(s) "
            "(< %.0f%% of largest component)",
            n_removed, n, min_component_fraction * 100,
        )
    return [facet_point_lists[i] for i in sorted(kept_indices)]


def compute_continuity(
    facet_point_lists: list[np.ndarray],
    all_building_points: np.ndarray,
    adjacency_gap_m: float = 1.0,
) -> dict:
    """Measure how well the detected facets cover the building point cloud.

    Returns
    -------
    dict with keys:
        unassigned_point_fraction : float
            Fraction of building points not assigned to any facet (0 = fully covered).
        is_connected : bool
            True if all facets form a single connected component.
        isolated_facet_ids : list[int]
            1-indexed facet IDs that have no adjacent neighbours.
    """
    from collections import deque

    n = len(facet_point_lists)

    # ── Coverage ──────────────────────────────────────────────────────────────
    assigned_count = sum(len(pts) for pts in facet_point_lists)
    total = len(all_building_points)
    unassigned_fraction = round(1.0 - assigned_count / total, 4) if total > 0 else 0.0

    if n == 0:
        return {"unassigned_point_fraction": unassigned_fraction, "is_connected": True, "isolated_facet_ids": [], "num_components": 0}

    adjacency = _xy_hull_adjacency(facet_point_lists, adjacency_gap_m)

    # ── Connected-component count via BFS ─────────────────────────────────────
    visited = [False] * n
    num_components = 0
    for start in range(n):
        if visited[start]:
            continue
        num_components += 1
        q: deque[int] = deque([start])
        visited[start] = True
        while q:
            node = q.popleft()
            for nb in adjacency[node]:
                if not visited[nb]:
                    visited[nb] = True
                    q.append(nb)

    isolated_facet_ids = [i + 1 for i in range(n) if not adjacency[i]]

    return {
        "unassigned_point_fraction": unassigned_fraction,
        "is_connected": num_components == 1,
        "num_components": num_components,
        "isolated_facet_ids": isolated_facet_ids,
    }


def estimate_ground_elevation(
    ground_pts: np.ndarray,
    centroid_xy: np.ndarray,
) -> float:
    """Fit a plane to ground points and evaluate at the building centroid.

    More accurate than a simple median on sloped terrain — the plane
    interpolates to the point directly beneath the building rather than
    averaging over the whole footprint.

    Falls back to median if fewer than 3 points or if the fit is degenerate.
    """
    if len(ground_pts) < 3:
        return float(np.median(ground_pts[:, 2])) if len(ground_pts) > 0 else 0.0

    centroid = ground_pts.mean(axis=0)
    centred = ground_pts - centroid
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    normal = vt[-1]

    if abs(normal[2]) < 1e-4:  # near-vertical plane — physically impossible for ground
        return float(np.median(ground_pts[:, 2]))

    cx, cy = float(centroid_xy[0]), float(centroid_xy[1])
    z = centroid[2] - (normal[0] * (cx - centroid[0]) + normal[1] * (cy - centroid[1])) / normal[2]
    return float(z)


# ── Solar irradiance model (ASHRAE clear-sky) ─────────────────────────────────

# ASHRAE monthly constants (Threlkeld & Jordan, 1958; HOF 2009 Table 1 Ch.14)
# A: apparent extraterrestrial irradiance (W/m²), B: extinction coeff, C: diffuse factor
_ASHRAE_A = np.array([1202,1187,1164,1130,1106,1092,1093,1107,1136,1166,1190,1204], dtype=float)
_ASHRAE_B = np.array([0.141,0.142,0.149,0.164,0.177,0.185,0.186,0.182,0.165,0.152,0.144,0.141], dtype=float)
_ASHRAE_C = np.array([0.103,0.104,0.109,0.120,0.130,0.137,0.138,0.134,0.121,0.111,0.106,0.103], dtype=float)
_MONTH_START_DOY = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])

_HOURS = np.arange(0.5, 24.0, 1.0)                        # 24 mid-hour solar times
_HA_RAD = np.radians(15.0 * (_HOURS - 12.0))              # hour angles (rad)


def _annual_poa_wh(pitch_deg: float, azimuth_deg: float, lat_deg: float) -> float:
    """Integrate annual plane-of-array clear-sky irradiance (Wh/m²).

    Uses ASHRAE clear-sky model + isotropic diffuse sky + ground reflectance 0.2.
    """
    lat_rad = np.radians(lat_deg)
    tilt_rad = np.radians(pitch_deg)
    az_facet_rad = np.radians(azimuth_deg)  # compass radians (0=N, π/2=E, π=S, 3π/2=W)
    cos_tilt = np.cos(tilt_rad)
    sin_tilt = np.sin(tilt_rad)

    total_wh = 0.0
    for day in range(1, 366):
        m = int(np.searchsorted(_MONTH_START_DOY, day, side="right")) - 1
        A, B, C = _ASHRAE_A[m], _ASHRAE_B[m], _ASHRAE_C[m]
        decl_rad = np.radians(23.45 * np.sin(np.radians(360 / 365 * (284 + day))))

        sin_alt = np.clip(
            np.sin(lat_rad) * np.sin(decl_rad) + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(_HA_RAD),
            -1.0, 1.0,
        )
        mask = sin_alt > 0.01  # daylight only; >0.01 avoids grazing-angle instability
        if not np.any(mask):
            continue
        s = sin_alt[mask]
        ha = _HA_RAD[mask]

        I_DN = A * np.exp(-B / s)          # direct normal irradiance (W/m²)
        I_dH = C * I_DN                    # diffuse horizontal (W/m²)
        GHI  = I_DN * s + I_dH            # global horizontal

        # Solar azimuth (compass, radians)
        cos_alt = np.sqrt(np.clip(1.0 - s ** 2, 0.0, 1.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_az_s = np.where(
                cos_alt > 1e-6,
                np.clip((s * np.sin(lat_rad) - np.sin(decl_rad)) / (cos_alt * np.cos(lat_rad)), -1.0, 1.0),
                0.0,
            )
        az_from_south = np.where(ha > 0, np.arccos(cos_az_s), -np.arccos(cos_az_s))
        az_sun = np.pi + az_from_south  # convert to compass (south=π)

        # Angle of incidence on tilted surface
        cos_aoi = cos_alt * np.cos(az_sun - az_facet_rad) * sin_tilt + s * cos_tilt

        I_beam = I_DN * np.maximum(0.0, cos_aoi)
        I_diff = I_dH * (1.0 + cos_tilt) / 2.0
        I_refl = GHI * 0.2  * (1.0 - cos_tilt) / 2.0  # ground reflectance 0.2

        total_wh += float(np.sum(I_beam + I_diff + I_refl))  # ×1 hr each

    return total_wh


def facet_solar_irradiance(
    pitch_deg: float,
    azimuth_deg: float,
    lat_deg: float,
) -> tuple[float, float]:
    """Annual clear-sky solar irradiance for a tilted roof facet.

    Parameters
    ----------
    pitch_deg : float   Facet tilt from horizontal (0=flat, 90=vertical).
    azimuth_deg : float Downslope compass bearing (0=N, 90=E, 180=S, 270=W).
    lat_deg : float     Building latitude (positive=NH, negative=SH).

    Returns
    -------
    solar_kwh_m2_yr : float  Annual POA irradiance in kWh/m²/yr (clear-sky).
    solar_suitability : float  0–1 efficiency vs optimal tilt/azimuth at this latitude.
    """
    facet_wh = _annual_poa_wh(pitch_deg, azimuth_deg, lat_deg)

    # Optimal: tilt = |lat|, south-facing in NH / north-facing in SH
    optimal_az = 180.0 if lat_deg >= 0 else 0.0
    optimal_wh = _annual_poa_wh(abs(lat_deg), optimal_az, lat_deg)

    kwh_m2_yr   = round(facet_wh / 1000.0, 1)
    suitability = round(float(np.clip(facet_wh / optimal_wh, 0.0, 1.0)) if optimal_wh > 0 else 0.0, 3)
    return kwh_m2_yr, suitability


def detect_roof_obstacles(
    unassigned_pts: np.ndarray,
    facet_point_lists: list[np.ndarray],
    eps_m: float = 0.8,
    min_samples: int = 5,
) -> list:
    """Detect obstacles on the roof from unassigned LiDAR points.

    Clusters the unassigned points spatially (DBSCAN in XY), then for each
    cluster measures its footprint, vertical extent, and height above the
    local roof surface — yielding chimney / vent-HVAC / unknown labels.

    Parameters
    ----------
    unassigned_pts :
        (N, 3) points that were not assigned to any facet.
    facet_point_lists :
        All surviving facet point arrays (used to determine local roof Z).
    eps_m :
        DBSCAN neighbourhood radius (metres, XY only).
    min_samples :
        Minimum points to form a cluster.

    Returns
    -------
    list of ObstacleResult
    """
    from scipy.spatial import ConvexHull, cKDTree
    from sklearn.cluster import DBSCAN

    if len(unassigned_pts) < min_samples or not facet_point_lists:
        return []

    labels = DBSCAN(eps=eps_m, min_samples=min_samples).fit_predict(unassigned_pts[:, :2])

    all_facet_pts = np.vstack(facet_point_lists)
    xy_tree = cKDTree(all_facet_pts[:, :2])

    obstacles = []
    for label in sorted(set(labels)):
        if label == -1:          # DBSCAN noise
            continue

        cluster = unassigned_pts[labels == label]
        xy  = cluster[:, :2]
        z   = cluster[:, 2]

        centroid_xy  = xy.mean(axis=0)
        z_min, z_max = float(z.min()), float(z.max())
        vertical_ext = round(z_max - z_min, 3)

        # Local roof Z: median Z of the nearest facet points in XY
        k = min(30, len(all_facet_pts))
        _, idx = xy_tree.query(centroid_xy, k=k)
        local_roof_z = float(np.median(all_facet_pts[idx, 2]))
        height_above = round(max(0.0, z_max - local_roof_z), 3)

        try:
            fp_area = round(float(ConvexHull(xy).volume), 2) if len(xy) >= 3 else 0.0
        except Exception:
            fp_area = 0.0

        # Classification via aspect ratio: height_above / characteristic_width.
        # Chimneys are tall-and-narrow (high ratio); HVAC units are wide-and-flat
        # (low ratio).  Using sqrt(fp_area) as the characteristic width avoids
        # hard-coding absolute size thresholds that break for different building types.
        char_width = fp_area ** 0.5
        aspect = height_above / (char_width + 0.01)   # avoid div-by-zero

        if height_above < 0.15:
            obs_type = "unknown"      # barely protrudes — can't classify reliably
        elif aspect >= 0.9:
            obs_type = "chimney"      # tall relative to its footprint width
        elif fp_area >= 0.8 or aspect < 0.5:
            obs_type = "vent_hvac"    # wide or very flat
        else:
            obs_type = "unknown"      # ambiguous border zone

        # Confidence: requires both enough points AND a meaningful protrusion.
        # Saturates at 30 pts and 0.4 m height; weak on either axis → low score.
        point_conf  = min(1.0, len(cluster) / 30.0)
        height_conf = min(1.0, height_above / 0.4)
        confidence  = round(point_conf * height_conf, 3)

        obstacles.append(ObstacleResult(
            obstacle_id=len(obstacles) + 1,
            obstacle_type=obs_type,
            centroid_xy=[round(float(centroid_xy[0]), 3), round(float(centroid_xy[1]), 3)],
            footprint_area_m2=fp_area,
            vertical_extent_m=vertical_ext,
            height_above_roof_m=height_above,
            num_points=len(cluster),
            confidence=confidence,
        ))

    logger.info("detect_roof_obstacles: found %d obstacle(s)", len(obstacles))
    return obstacles



def _local_normals_pca(points: np.ndarray, k: int = 15) -> np.ndarray:
    """Estimate per-point surface normals via batched PCA on k nearest neighbours.

    Returns (N, 3) unit normals, oriented so Z ≥ 0.
    Uses the same vectorised approach as the region-growing segmenter to avoid
    per-point Python loops.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    _, nn_idx = tree.query(points, k=k + 1)
    nn_idx = nn_idx[:, 1:]  # exclude self

    nbr_pts = points[nn_idx]                               # (N, k, 3)
    centred = nbr_pts - nbr_pts.mean(axis=1, keepdims=True)

    cov = np.einsum("nki,nkj->nij", centred, centred)     # (N, 3, 3)
    _, eigenvectors = np.linalg.eigh(cov)                  # eigenvalues ascending
    normals = eigenvectors[:, :, 0]                        # smallest eigval → normal

    flip = normals[:, 2] < 0
    normals[flip] *= -1
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norms, 1e-12)
    return normals


def filter_above_surface_outliers(
    points: np.ndarray,
    k_neighbors: int = 15,
    z_sigma_thresh: float = 3.0,
) -> np.ndarray:
    """Remove multipath / ghost returns sitting anomalously high above the roof.

    LiDAR multipath returns arise when a pulse reflects off glass, standing
    water, or solar panels and arrives at the scanner later than expected,
    producing phantom points 0.3–2 m above the real surface.  These inflate
    ridge elevation estimates and can cause RANSAC to fit a spurious upper plane.

    Detection: for each point, compare its Z to the median Z of its *k*
    nearest XY neighbours.  Points more than *z_sigma_thresh* local standard
    deviations above that neighbourhood median are flagged and removed.

    Parameters
    ----------
    k_neighbors :
        Number of XY neighbours used to estimate the local surface Z.
    z_sigma_thresh :
        Outlier threshold in units of local Z standard deviation.  The default
        3.0 removes clear spikes while leaving valid steep-edge points.
    """
    from scipy.spatial import cKDTree

    if len(points) < k_neighbors + 1:
        return points

    tree = cKDTree(points[:, :2])                          # 2-D search only
    _, idx = tree.query(points[:, :2], k=k_neighbors + 1)
    idx = idx[:, 1:]                                       # exclude self

    nbr_z = points[idx, 2]                                # (N, k)
    local_median = np.median(nbr_z, axis=1)
    local_std = np.std(nbr_z, axis=1)
    local_std = np.maximum(local_std, 0.02)               # 2 cm floor — avoids /0 on flat roofs

    z_score = (points[:, 2] - local_median) / local_std
    mask = z_score <= z_sigma_thresh

    kept = points[mask]
    n_removed = len(points) - len(kept)
    if n_removed:
        logger.debug(
            "filter_above_surface_outliers: removed %d / %d phantom-return outlier(s)",
            n_removed, len(points),
        )
    return kept


def filter_near_vertical_points(
    points: np.ndarray,
    k_neighbors: int = 15,
    max_pitch_deg: float = 75.0,
) -> np.ndarray:
    """Remove points whose local neighbourhood is near-vertical (walls, parapets).

    Wall returns that survive :func:`filter_below_eave` — high dormer walls,
    chimney sides, parapet faces — have estimated surface normals close to
    horizontal (the surface itself is near-vertical).  When these reach RANSAC
    they consume plane slots and produce spurious high-pitch facets that the
    downstream quality filter must discard at the cost of max_planes budget.

    Removing them pre-RANSAC keeps the plane budget for actual roof surfaces.

    Parameters
    ----------
    max_pitch_deg :
        Maximum estimated surface pitch to keep (degrees from horizontal).
        Default 75° preserves steep-but-valid mansard lower slopes (≈ 60°)
        while removing obvious walls and parapet faces (≈ 85–90°).
    """
    if len(points) < k_neighbors + 1:
        return points

    normals = _local_normals_pca(points, k=k_neighbors)
    nz = np.clip(np.abs(normals[:, 2]), 0.0, 1.0)
    pitch = np.degrees(np.arccos(nz))

    mask = pitch <= max_pitch_deg
    kept = points[mask]
    n_removed = len(points) - len(kept)
    if n_removed:
        logger.debug(
            "filter_near_vertical_points: removed %d / %d near-vertical points (pitch > %.0f°)",
            n_removed, len(points), max_pitch_deg,
        )
    # Safety: never discard everything (would leave nothing for RANSAC)
    if len(kept) < 10:
        logger.warning(
            "filter_near_vertical_points: safety fallback — returning all %d points "
            "(near-vertical filter would have left < 10 points)",
            len(points),
        )
        return points
    return kept


def estimate_point_density(points: np.ndarray) -> float:
    """Estimate pts/m² using total XY bounding box area as denominator."""
    if len(points) < 3:
        return 0.0
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points[:, :2])
        area = hull.volume  # 2D area
        return round(len(points) / area, 1) if area > 0 else 0.0
    except Exception as exc:
        logger.debug("estimate_point_density: ConvexHull failed (%s) — returning 0", exc)
        return 0.0
