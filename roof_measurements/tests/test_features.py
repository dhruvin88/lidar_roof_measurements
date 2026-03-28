"""Tests for feature extraction: pitch, azimuth, area, height."""

from __future__ import annotations

import numpy as np
import pytest

from roof_measurements.features import (
    FLAT_PITCH_THRESHOLD_DEG,
    azimuth_from_normal,
    compute_facet,
    compute_height,
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
        normal = fit_plane_normal(pts)
        assert abs(normal[2]) > 0.999, f"Expected near-vertical normal, got {normal}"

    def test_tilted_plane(self):
        # 30° pitch: normal should have nz = cos(30°) ≈ 0.866
        expected_pitch = 30.0
        nz = np.cos(np.radians(expected_pitch))
        nx = np.sin(np.radians(expected_pitch))
        true_normal = np.array([nx, 0.0, nz])
        true_normal /= np.linalg.norm(true_normal)

        pts = _plane_points(true_normal)
        normal = fit_plane_normal(pts)
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
        # 10x10m square in the XY plane
        pts = np.array([
            [0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0],
        ], dtype=float)
        # Fill with interior points
        rng = np.random.default_rng(0)
        interior = np.column_stack([rng.uniform(0, 10, 200), rng.uniform(0, 10, 200), np.zeros(200)])
        pts = np.vstack([pts, interior])
        normal = np.array([0.0, 0.0, 1.0])
        area = projected_area(pts, normal)
        assert abs(area - 100.0) < 2.0, f"Expected ~100 m², got {area:.1f}"


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
