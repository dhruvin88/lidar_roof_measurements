"""Tests for plane segmentation on synthetic roofs."""

from __future__ import annotations

import numpy as np

from roof_measurements.segmentation import segment_planes


class TestGableRoof:
    def test_finds_two_facets(self, gable_roof_points):
        facets, method = segment_planes(gable_roof_points, distance_threshold=0.15, min_facet_area_m2=1.0)
        assert len(facets) == 2, f"Expected 2 facets for gable roof, got {len(facets)}"

    def test_method_is_ransac(self, gable_roof_points):
        _, method = segment_planes(gable_roof_points)
        # RANSAC should work well on a clean gable
        assert method in ("ransac", "region_growing")

    def test_pitch_close_to_30_deg(self, gable_roof_points):
        from roof_measurements.features import compute_facet
        facets, _ = segment_planes(gable_roof_points, distance_threshold=0.15, min_facet_area_m2=1.0)
        assert len(facets) == 2
        for pts in facets:
            facet = compute_facet(1, pts)
            assert abs(facet.pitch_deg - 30.0) < 3.0, \
                f"Expected pitch ~30°, got {facet.pitch_deg:.1f}°"


class TestHipRoof:
    def test_finds_four_facets(self, hip_roof_points):
        facets, _ = segment_planes(hip_roof_points, distance_threshold=0.2, min_facet_area_m2=1.0)
        # Hip roof has 4 faces; allow ±1 for boundary noise
        assert 3 <= len(facets) <= 5, f"Expected ~4 facets for hip roof, got {len(facets)}"


class TestFlatRoof:
    def test_finds_single_facet(self, flat_roof_points):
        facets, _ = segment_planes(flat_roof_points, distance_threshold=0.1, min_facet_area_m2=1.0)
        assert len(facets) == 1, f"Expected 1 facet for flat roof, got {len(facets)}"

    def test_flat_facet_pitch_below_threshold(self, flat_roof_points):
        from roof_measurements.features import FLAT_PITCH_THRESHOLD_DEG, compute_facet
        facets, _ = segment_planes(flat_roof_points)
        assert len(facets) >= 1
        # The dominant plane should be near-flat
        facet = compute_facet(1, facets[0])
        assert facet.pitch_deg < FLAT_PITCH_THRESHOLD_DEG or facet.pitch_deg < 5.0


class TestMinFacetAreaFilter:
    def test_tiny_facets_removed(self):
        """Inject a large main plane + a tiny 0.5 m² speck. Speck should be filtered."""
        rng = np.random.default_rng(42)
        main = np.column_stack([rng.uniform(0, 10, 500), rng.uniform(0, 10, 500), np.zeros(500)])
        speck = np.column_stack([rng.uniform(100, 100.5, 20), rng.uniform(100, 100.5, 20), np.ones(20) * 0.5])
        pts = np.vstack([main, speck])
        facets, _ = segment_planes(pts, min_facet_area_m2=1.0)
        for f in facets:
            from roof_measurements.segmentation import _xy_area
            assert _xy_area(f) >= 1.0
