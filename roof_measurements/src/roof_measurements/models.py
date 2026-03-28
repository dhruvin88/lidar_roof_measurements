"""Pydantic output models for roof measurement results."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class FacetResult(BaseModel):
    facet_id: int = Field(description="1-indexed facet identifier")
    pitch_deg: float = Field(description="Slope angle in degrees from horizontal (0=flat, 90=vertical)")
    azimuth_deg: float = Field(description="Compass bearing of downslope direction (0=North, 90=East)")
    area_m2: float = Field(description="Projected planar area of facet in square metres")
    normal_vector: list[float] = Field(description="Unit normal vector [nx, ny, nz]")
    num_points: int = Field(description="Number of LiDAR points assigned to this facet")
    is_flat: bool = Field(default=False, description="True if pitch < 5° (flat/low-slope roof)")
    eave_elevation_m: float = Field(description="Eave elevation: 2nd-percentile Z of facet points (metres)")
    plane_rms_m: float = Field(description="RMS of point-to-plane residuals in metres. Lower = better fit.")
    confidence: float = Field(
        description="Planarity confidence 0–1. 1.0 = perfect plane, 0.0 = RMS ≥ 0.30 m."
    )
    solar_kwh_m2_yr: Optional[float] = Field(
        default=None,
        description="Annual clear-sky solar irradiance on this surface (kWh/m²/yr). None if latitude not provided.",
    )
    solar_suitability: Optional[float] = Field(
        default=None,
        description="Solar suitability 0–1 vs optimal tilt/orientation for the building's latitude.",
    )


class ObstacleResult(BaseModel):
    obstacle_id: int = Field(description="1-indexed obstacle identifier")
    obstacle_type: str = Field(description="Classified type: chimney / vent_hvac / unknown")
    centroid_xy: list[float] = Field(description="[x, y] centroid in the projected CRS (metres)")
    footprint_area_m2: float = Field(description="XY convex-hull area of the obstacle cluster (m²)")
    vertical_extent_m: float = Field(description="Z range of the obstacle cluster (m)")
    height_above_roof_m: float = Field(description="Max Z of cluster minus local roof surface Z (m)")
    num_points: int = Field(description="Number of LiDAR points in the cluster")
    confidence: float = Field(
        description=(
            "Detection confidence 0–1. Driven by point count (≥30 pts) and "
            "protrusion height (≥0.4 m). Both factors must be strong for high confidence."
        )
    )


class BuildingResult(BaseModel):
    building_id: str = Field(description="Identifier for the building (auto-generated or from input)")
    num_facets: int = Field(description="Number of detected roof facets")
    height_m: float = Field(description="Roof height: ridge elevation minus ground elevation")
    ground_elevation_m: float = Field(description="Median ground elevation beneath building (metres)")
    ridge_elevation_m: float = Field(description="Maximum Z of roof points (metres)")
    facets: list[FacetResult] = Field(default_factory=list)
    eave_height_m: float = Field(description="Eave height: lowest eave elevation minus ground elevation (metres)")
    point_density_m2: Optional[float] = Field(
        default=None,
        description="Estimated point density (pts/m²) of roof points. <4 may reduce accuracy."
    )
    segmentation_method: str = Field(
        default="ransac",
        description="Method used for plane segmentation: 'ransac' or 'region_growing'"
    )
    unassigned_point_fraction: float = Field(
        default=0.0,
        description="Fraction of building points not assigned to any facet (0=fully covered, 1=none covered)."
    )
    is_facets_connected: bool = Field(
        default=True,
        description="True if all facets form a single spatially connected set."
    )
    num_facet_components: int = Field(
        default=1,
        description="Number of spatially disconnected facet groups."
    )
    isolated_facet_ids: list[int] = Field(
        default_factory=list,
        description="1-indexed IDs of facets with no adjacent neighbours."
    )
    total_roof_area_m2: float = Field(
        default=0.0,
        description="Sum of all facet areas (m²). Accounts for pitch — larger than footprint area."
    )
    roof_type: str = Field(
        default="unknown",
        description="Classified roof type: flat / shed / gable / hip / mansard / complex."
    )
    total_solar_kwh_yr: Optional[float] = Field(
        default=None,
        description="Total annual clear-sky solar potential of all facets (kWh/yr). None if latitude not provided.",
    )
    obstacles: list[ObstacleResult] = Field(
        default_factory=list,
        description="Detected roof obstacles (chimneys, vents, HVAC units).",
    )
