"""Integration test: pipeline end-to-end on synthetic data (no real LAS files)."""

from __future__ import annotations

import numpy as np
import pytest

from roof_measurements.features import compute_facet, compute_height
from roof_measurements.models import BuildingResult
from roof_measurements.pipeline import process_file
from roof_measurements.segmentation import segment_planes


def test_gable_pipeline_integration(gable_roof_points, tmp_path):
    """Run the full pipeline on a synthetic gable roof written as a LAS file."""
    import laspy

    las = laspy.LasData(laspy.LasHeader(version="1.4", point_format=6))
    las.x = gable_roof_points[:, 0]
    las.y = gable_roof_points[:, 1]
    las.z = gable_roof_points[:, 2]
    # Mark all points as building (class 6)
    las.classification = np.full(len(gable_roof_points), 6, dtype=np.uint8)

    output_las = tmp_path / "gable.las"
    las.write(str(output_las))

    result = process_file(output_las, building_id="test_gable")

    assert isinstance(result, BuildingResult)
    assert result.building_id == "test_gable"
    assert result.num_facets == 2
    assert result.height_m > 0
    assert len(result.facets) == 2

    for facet in result.facets:
        assert abs(facet.pitch_deg - 30.0) < 3.5
        assert facet.area_m2 > 1.0
        assert facet.num_points > 0


def test_height_accuracy(gable_roof_points):
    """Height should be ~5m (ridge Z=5, ground Z=0)."""
    facets, _ = segment_planes(gable_roof_points, distance_threshold=0.15, min_facet_area_m2=1.0)
    all_pts = np.vstack(facets)
    height, ridge = compute_height(all_pts, ground_elevation_m=0.0)
    assert abs(height - 5.0) < 0.5, f"Expected height ~5m, got {height:.2f}m"
    assert abs(ridge - 5.0) < 0.5, f"Expected ridge ~5m, got {ridge:.2f}m"
