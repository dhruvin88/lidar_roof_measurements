"""Synthetic roof point cloud fixtures."""

from __future__ import annotations

import numpy as np
import pytest


def _add_noise(pts: np.ndarray, sigma: float = 0.02) -> np.ndarray:
    return pts + np.random.default_rng(42).normal(0, sigma, pts.shape)


@pytest.fixture
def gable_roof_points() -> np.ndarray:
    """Simple gable roof: two planar faces meeting at a ridge.

    Ridge runs along Y axis at X=0, Z_ridge=5.
    Each face has pitch ~30° (tan(30°) = 0.577, so for X span of 5m, Z drop = 2.887m).
    Ground at Z=0.
    """
    rng = np.random.default_rng(0)
    pts = []

    # Left face: X from -5 to 0, Z decreases from ridge to eave
    for _ in range(800):
        x = rng.uniform(-5, 0)
        y = rng.uniform(0, 10)
        z = 5.0 + x * np.tan(np.radians(30))  # x is negative, so z < 5
        pts.append([x, y, z])

    # Right face: X from 0 to 5
    for _ in range(800):
        x = rng.uniform(0, 5)
        y = rng.uniform(0, 10)
        z = 5.0 - x * np.tan(np.radians(30))
        pts.append([x, y, z])

    return _add_noise(np.array(pts))


@pytest.fixture
def hip_roof_points() -> np.ndarray:
    """Hip roof: four planar faces (front, back, left, right).

    Building footprint 10m x 8m. Ridge at Z=4, ground at Z=0.
    """
    rng = np.random.default_rng(1)
    pts = []

    def hip_z(x, y, z_ridge=4.0, lx=5.0, ly=4.0):
        fx = 1.0 - abs(x) / lx
        fy = 1.0 - abs(y) / ly
        return z_ridge * min(fx, fy)

    for _ in range(2000):
        x = rng.uniform(-5, 5)
        y = rng.uniform(-4, 4)
        pts.append([x, y, hip_z(x, y)])

    return _add_noise(np.array(pts))


@pytest.fixture
def flat_roof_points() -> np.ndarray:
    """Flat roof: single plane at Z=3, slight 2° drainage slope."""
    rng = np.random.default_rng(2)
    pts = []
    for _ in range(500):
        x = rng.uniform(0, 8)
        y = rng.uniform(0, 6)
        z = 3.0 + x * np.tan(np.radians(2))
        pts.append([x, y, z])
    return _add_noise(np.array(pts), sigma=0.01)
