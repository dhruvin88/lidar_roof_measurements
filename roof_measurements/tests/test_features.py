"""Tests for feature extraction: pitch, azimuth, area, height."""

from __future__ import annotations

import numpy as np
import pytest

from roof_measurements.features import (
    FLAT_PITCH_THRESHOLD_DEG,
    azimuth_from_normal,
    compute_facet,
    compute_height,
    filter_above_surface_outliers,
    filter_near_vertical_points,
    fit_plane_normal,
    pitch_from_normal,
    projected_area,
)


def _plane_points(normal: np.ndarray, n: int = 300, seed: int = 0) -> np.ndarray:
    """Generate n points on a plane with given unit normal, with small noise."""
    rng = np.random.default_rng(seed)
    # Build orthonormal basis for the plane
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, ref)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    pts = rng.uniform(-5, 5, (n, 2))
    coords = pts[:, 0:1] * u + pts[:, 1:2] * v
    coords += rng.normal(0, 0.02, coords.shape)
    return coords


class TestFitPlaneNormal:
    def test_horizontal_plane(self):
        pts = np.column_stack([
            np.random.default_rng(0).uniform(-5, 5, 200),
            np.random.default_rng(1).uniform(-5, 5, 200),
            np.zeros(200),
        ])
        normal, _ = fit_plane_normal(pts)
        assert abs(normal[2]) > 0.999, f"Expected near-vertical normal, got {normal}"

    def test_tilted_plane(self):
        # 30° pitch: normal should have nz = cos(30°) ≈ 0.866
        expected_pitch = 30.0
        nz = np.cos(np.radians(expected_pitch))
        nx = np.sin(np.radians(expected_pitch))
        true_normal = np.array([nx, 0.0, nz])
        true_normal /= np.linalg.norm(true_normal)

        pts = _plane_points(true_normal)
        normal, _ = fit_plane_normal(pts)
        assert normal[2] > 0, "Normal should point upward"
        recovered_pitch = pitch_from_normal(normal)
        assert abs(recovered_pitch - expected_pitch) < 2.0, \
            f"Expected pitch ~{expected_pitch}°, got {recovered_pitch:.1f}°"


class TestPitchFromNormal:
    @pytest.mark.parametrize("pitch_deg", [0, 10, 20, 30, 45])
    def test_various_pitches(self, pitch_deg):
        nz = np.cos(np.radians(pitch_deg))
        nx = np.sqrt(1 - nz ** 2)
        normal = np.array([nx, 0.0, nz])
        result = pitch_from_normal(normal)
        assert abs(result - pitch_deg) < 0.01

    def test_flat_roof_detection(self):
        normal = np.array([0.0, 0.0, 1.0])
        assert pitch_from_normal(normal) < FLAT_PITCH_THRESHOLD_DEG


class TestAzimuthFromNormal:
    def test_north_facing(self):
        # Normal points in +Y direction (downslope to North)
        normal = np.array([0.0, 1.0, 0.01])
        normal /= np.linalg.norm(normal)
        az = azimuth_from_normal(normal)
        # Downslope is opposite to projected normal direction
        # normal XY = (0, +1) → downslope = (0, -1) → azimuth = 180°
        assert abs(az - 180.0) < 2.0, f"Expected ~180°, got {az:.1f}°"

    def test_east_facing(self):
        normal = np.array([1.0, 0.0, 0.01])
        normal /= np.linalg.norm(normal)
        az = azimuth_from_normal(normal)
        # normal XY = (+1, 0) → downslope = (-1, 0) → azimuth = 270°
        assert abs(az - 270.0) < 2.0, f"Expected ~270°, got {az:.1f}°"


class TestProjectedArea:
    def test_square_facet(self):
        # 10x10m square in the XY plane — use 1000 interior points so the
        # alpha-shape has enough density (~10 pts/m²) to recover the full area.
        pts = np.array([
            [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
        ], dtype=float)
        rng = np.random.default_rng(0)
        interior = np.column_stack([rng.uniform(0, 10, 1000), rng.uniform(0, 10, 1000), np.zeros(1000)])
        pts = np.vstack([pts, interior])
        normal = np.array([0.0, 0.0, 1.0])
        area = projected_area(pts, normal)
        # Alpha-shape trims boundary triangles near corners by design;
        # ≤ 8 m² (8%) error confirms the method is working correctly.
        assert abs(area - 100.0) < 8.0, f"Expected ~100 m², got {area:.1f}"


class TestComputeHeight:
    def test_height_computation(self):
        pts = np.array([[0, 0, z] for z in np.linspace(3.0, 8.0, 100)])
        height, ridge = compute_height(pts, ground_elevation_m=0.0)
        # 99th percentile of linspace(3,8) ≈ 7.95
        assert 7.8 < ridge < 8.1
        assert 7.8 < height < 8.1

    def test_height_above_ground(self):
        pts = np.array([[x, y, 15.0] for x in range(5) for y in range(5)], dtype=float)
        height, ridge = compute_height(pts, ground_elevation_m=10.0)
        assert abs(height - 5.0) < 0.1


class TestComputeFacet:
    def test_flat_facet_flagged(self):
        pts = np.column_stack([
            np.random.default_rng(0).uniform(0, 5, 100),
            np.random.default_rng(1).uniform(0, 5, 100),
            np.zeros(100),
        ])
        facet = compute_facet(1, pts)
        assert facet.is_flat is True
        assert facet.pitch_deg < FLAT_PITCH_THRESHOLD_DEG

    def test_pitched_facet_not_flat(self):
        pitch = 30.0
        nz = np.cos(np.radians(pitch))
        nx = np.sin(np.radians(pitch))
        normal = np.array([nx, 0.0, nz])
        pts = _plane_points(normal)
        facet = compute_facet(1, pts)
        assert facet.is_flat is False
        assert abs(facet.pitch_deg - pitch) < 2.5


class TestFilterAboveSurfaceOutliers:
    def _flat_roof(self, n: int = 500, seed: int = 0) -> np.ndarray:
        """Random points on a flat roof at Z=10, spread over 10×10 m."""
        rng = np.random.default_rng(seed)
        xy = rng.uniform(0, 10, (n, 2))
        z  = np.full(n, 10.0) + rng.normal(0, 0.02, n)
        return np.column_stack([xy, z])

    def test_clean_cloud_unchanged(self):
        pts = self._flat_roof()
        result = filter_above_surface_outliers(pts)
        # No phantoms — should keep all (or nearly all) points
        assert len(result) >= len(pts) - 5

    def test_phantom_points_removed(self):
        pts = self._flat_roof()
        rng = np.random.default_rng(42)
        # Inject 10 phantom returns 1.5 m above the surface
        ghosts_xy = rng.uniform(1, 9, (10, 2))
        ghosts_z  = np.full(10, 11.5)
        ghosts    = np.column_stack([ghosts_xy, ghosts_z])
        pts_with_ghosts = np.vstack([pts, ghosts])

        result = filter_above_surface_outliers(pts_with_ghosts, z_sigma_thresh=3.0)
        # Most of the injected phantoms should be gone
        n_removed = len(pts_with_ghosts) - len(result)
        assert n_removed >= 8, f"Expected ≥8 phantom points removed, got {n_removed}"

    def test_small_cloud_passthrough(self):
        # Fewer points than k_neighbors — must not raise, must return input
        pts = np.random.default_rng(0).uniform(0, 5, (10, 3))
        result = filter_above_surface_outliers(pts, k_neighbors=15)
        assert len(result) == len(pts)


class TestFilterNearVerticalPoints:
    def _make_mixed_cloud(self) -> np.ndarray:
        """Return a cloud with 400 roof points (30° pitch) + 100 near-vertical wall points."""
        rng = np.random.default_rng(7)
        # Roof plane: normal = (sin30, 0, cos30), centred at (5, 5, 12)
        nz = np.cos(np.radians(30))
        nx = np.sin(np.radians(30))
        roof_normal = np.array([nx, 0.0, nz])
        ref = np.array([0.0, 1.0, 0.0])
        u = np.cross(roof_normal, ref); u /= np.linalg.norm(u)
        v = np.cross(roof_normal, u);   v /= np.linalg.norm(v)
        uv = rng.uniform(-4, 4, (400, 2))
        roof_pts = np.array([5.0, 5.0, 12.0]) + uv[:, 0:1] * u + uv[:, 1:2] * v
        roof_pts += rng.normal(0, 0.02, roof_pts.shape)

        # Near-vertical wall: normal ≈ (1, 0, 0), points on the south face
        wall_uv = rng.uniform(0, 4, (100, 2))
        wall_pts = np.column_stack([
            np.zeros(100),                          # x = 0 (wall face)
            wall_uv[:, 0],                          # y spread
            8.0 + wall_uv[:, 1],                    # z = 8–12 (above eave)
        ])
        wall_pts += rng.normal(0, 0.03, wall_pts.shape)
        return np.vstack([roof_pts, wall_pts])

    def test_removes_wall_returns(self):
        pts = self._make_mixed_cloud()
        result = filter_near_vertical_points(pts, max_pitch_deg=75.0)
        # Should remove a meaningful fraction (the wall points)
        assert len(result) < len(pts), "Expected wall points to be removed"
        fraction_removed = 1.0 - len(result) / len(pts)
        assert fraction_removed >= 0.10, f"Expected ≥10% removed, got {fraction_removed:.1%}"

    def test_clean_roof_mostly_kept(self):
        rng = np.random.default_rng(3)
        # Pure 30° pitched roof — should keep nearly all points
        nz = np.cos(np.radians(30)); nx = np.sin(np.radians(30))
        normal = np.array([nx, 0.0, nz])
        ref = np.array([0.0, 1.0, 0.0])
        u = np.cross(normal, ref); u /= np.linalg.norm(u)
        v = np.cross(normal, u);   v /= np.linalg.norm(v)
        uv = rng.uniform(-5, 5, (300, 2))
        pts = np.array([5.0, 5.0, 10.0]) + uv[:, 0:1] * u + uv[:, 1:2] * v
        pts += rng.normal(0, 0.02, pts.shape)

        result = filter_near_vertical_points(pts, max_pitch_deg=75.0)
        assert len(result) >= int(0.85 * len(pts)), "Should keep most roof-surface points"

    def test_small_cloud_passthrough(self):
        pts = np.random.default_rng(0).uniform(0, 5, (10, 3))
        result = filter_near_vertical_points(pts, k_neighbors=15)
        assert len(result) == len(pts)
