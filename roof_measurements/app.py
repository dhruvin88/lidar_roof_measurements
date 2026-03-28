"""Streamlit app — enter a lat/lon, get roof measurements + visualizations."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from roof_measurements.constants import ASPRS_BUILDING, ASPRS_GROUND, NON_BUILDING_CLASSES
from roof_measurements.datasources import fetch_lidar_for_point
from roof_measurements.export import build_single_building_geojson
from roof_measurements.features import (
    _delaunay_alpha_kept,
    _project_to_plane,
    detect_roof_obstacles,
    estimate_ground_elevation,
    estimate_point_density,
    filter_below_eave,
    filter_radius_outliers,
    filter_subground_points,
)
from roof_measurements.footprints import footprint_at_point
from roof_measurements.models import BuildingResult, FacetResult
from roof_measurements.pipeline import assemble_result
from roof_measurements.segmentation import segment_planes

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Roof Measurements",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏠 Roof Measurement Tool")
st.caption("Enter a coordinate to fetch LiDAR data, detect roof facets, and visualize the results.")

# ── Geocoding ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600)
def geocode_address(address: str) -> tuple[float, float, str]:
    """Convert a free-text address to (lat, lon, display_name) via Nominatim."""
    import requests
    resp = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={"q": address, "format": "json", "limit": 1},
        headers={"User-Agent": "RoofMeasurementTool/1.0"},
        timeout=10,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise ValueError(f"No location found for '{address}'")
    r = results[0]
    return float(r["lat"]), float(r["lon"]), r["display_name"]


# ── Sidebar — inputs + options ────────────────────────────────────────────────

with st.sidebar:
    st.header("Location")

    # Apply pending coordinates BEFORE widgets are instantiated (city buttons + geocoder both use this)
    if "pending_lat" in st.session_state:
        st.session_state["lat_input"] = st.session_state.pop("pending_lat")
        st.session_state["lon_input"] = st.session_state.pop("pending_lon")

    # Address lookup
    addr_col, btn_col = st.columns([3, 1])
    address = addr_col.text_input("Address", placeholder="123 Main St, Oxford, MS", label_visibility="collapsed")
    lookup = btn_col.button("🔍", help="Look up address", use_container_width=True)

    if lookup:
        if address.strip():
            with st.spinner("Looking up…"):
                try:
                    g_lat, g_lon, g_name = geocode_address(address.strip())
                    st.session_state["pending_lat"] = g_lat
                    st.session_state["pending_lon"] = g_lon
                    st.session_state["geocode_display"] = g_name
                    st.session_state.pop("geocode_error", None)
                except Exception as exc:
                    st.session_state["geocode_error"] = str(exc)
                    st.session_state.pop("geocode_display", None)
            st.rerun()
        else:
            st.warning("Enter an address first.")

    if "geocode_display" in st.session_state:
        st.caption(f"📍 {st.session_state['geocode_display']}")
    if "geocode_error" in st.session_state:
        st.error(st.session_state.pop("geocode_error"))

    if "lat_input" not in st.session_state:
        st.session_state["lat_input"] = 34.333569
    if "lon_input" not in st.session_state:
        st.session_state["lon_input"] = -89.522289

    col_a, col_b = st.columns(2)
    lat = col_a.number_input("Latitude",  key="lat_input", format="%.6f", step=0.0001)
    lon = col_b.number_input("Longitude", key="lon_input", format="%.6f", step=0.0001)

    st.markdown("**Examples**")
    examples = {
        "Oxford, MS":       (34.3336, -89.5223),
        "Austin, TX":       (30.2672, -97.7431),
        "New York, NY":     (40.7128, -74.0060),
        "San Francisco, CA":(37.7749, -122.4194),
    }
    for label, (elat, elon) in examples.items():
        if st.button(label, use_container_width=True):
            st.session_state["pending_lat"] = elat
            st.session_state["pending_lon"] = elon
            st.session_state.pop("geocode_display", None)
            st.rerun()

    st.divider()
    st.header("Options")
    min_facet_area      = st.slider("Min facet area (m²)",    0.5, 5.0, 1.0, 0.5)
    min_facet_points    = st.slider("Min facet points",       5, 100, 10, 5)
    min_confidence      = st.slider("Min facet confidence",   0.0, 1.0, 0.1, 0.05)
    max_pitch_deg       = st.slider("Max facet pitch (°)",    30, 89, 70, 1)
    distance_threshold  = st.slider("RANSAC threshold (m)",   0.05, 0.40, 0.15, 0.05)
    max_planes          = st.slider("Max roof planes",        4, 30, 20)
    buffer_m            = st.slider("Footprint search (m)",   50, 300, 100, 50)
    footprint_buffer_m  = st.slider("Footprint edge buffer (m)", 0.0, 3.0, 0.5, 0.25)

    st.divider()
    run = st.button("▶  Analyze", type="primary", use_container_width=True)

# ── Pipeline (cached) ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_pipeline(
    lat: float, lon: float,
    min_facet_area: float, min_facet_points: int,
    min_confidence: float, max_pitch_deg: float,
    distance_threshold: float, max_planes: int,
    buffer_m: float, footprint_buffer_m: float,
) -> dict:
    """Full pipeline — cached so re-runs with same params skip downloads."""
    import laspy
    import pyproj
    from shapely import contains_xy
    from shapely.ops import transform as shapely_transform

    # 1. OSM footprint
    osm_id, polygon, _ = footprint_at_point(lat, lon, buffer_m=buffer_m)

    # 2. LiDAR tiles
    laz_paths, epsg = fetch_lidar_for_point(lat, lon)

    # 3. Load + merge + clip
    transformer = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

    # Expand the footprint polygon to capture eave overhangs and OSM inaccuracies.
    # Buffer in the projected CRS (metres), then convert back to WGS84.
    if footprint_buffer_m > 0:
        to_proj = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        polygon = shapely_transform(
            transformer.transform,
            shapely_transform(to_proj.transform, polygon).buffer(footprint_buffer_m),
        )

    all_candidates, all_ground = [], []
    for path in laz_paths:
        with laspy.open(path) as f:
            las = f.read()
        xyz = np.column_stack([
            np.array(las.x, dtype=np.float64),
            np.array(las.y, dtype=np.float64),
            np.array(las.z, dtype=np.float64),
        ])
        cls = np.array(las.classification)
        has6 = np.any(cls == ASPRS_BUILDING)
        has2 = np.any(cls == ASPRS_GROUND)
        cands = xyz[cls == ASPRS_BUILDING] if has6 else xyz[~np.isin(cls, list(NON_BUILDING_CLASSES))]
        all_candidates.append(cands)
        if has2:
            all_ground.append(xyz[cls == ASPRS_GROUND])

    candidate_pts = np.vstack(all_candidates)
    ground_pts    = np.vstack(all_ground) if all_ground else np.empty((0, 3))
    tile_ground_z = float(np.median(ground_pts[:, 2])) if len(ground_pts) else float(np.min(candidate_pts[:, 2]))

    cand_lons, cand_lats = transformer.transform(candidate_pts[:, 0], candidate_pts[:, 1])
    clipped = candidate_pts[contains_xy(polygon, cand_lons, cand_lats)]

    if len(clipped) < 10:
        raise ValueError(f"Only {len(clipped)} points inside footprint — too few to process.")

    if len(ground_pts) > 0:
        gnd_lons, gnd_lats = transformer.transform(ground_pts[:, 0], ground_pts[:, 1])
        gnd_mask = contains_xy(polygon, gnd_lons, gnd_lats)
        if gnd_mask.sum() >= 3:
            centroid_xy = clipped[:, :2].mean(axis=0) if len(clipped) > 0 else ground_pts[gnd_mask, :2].mean(axis=0)
            ground_z = estimate_ground_elevation(ground_pts[gnd_mask], centroid_xy)
        else:
            ground_z = tile_ground_z
    else:
        ground_z = tile_ground_z

    clipped = filter_subground_points(clipped, ground_z)
    if len(clipped) < 10:
        raise ValueError("Too few points remain after filtering sub-ground points.")

    # Remove spatially isolated point clusters (trees, vehicles, misclassified
    # objects) that survived footprint clipping and ASPRS classification.
    clipped = filter_radius_outliers(clipped)
    if len(clipped) < 10:
        raise ValueError("Too few points remain after outlier filtering.")

    # Remove wall points below the estimated eave line
    clipped = filter_below_eave(clipped, ground_z)

    # 4. Segment + features (keep facet_point_lists for 3D viz)
    facet_point_lists, method = segment_planes(
        clipped,
        distance_threshold=distance_threshold,
        min_facet_area_m2=min_facet_area,
        min_facet_points=min_facet_points,
        max_planes=max_planes,
    )
    if not facet_point_lists:
        # Retry once with relaxed thresholds before giving up
        facet_point_lists, method = segment_planes(
            clipped,
            distance_threshold=min(distance_threshold * 2.0, 0.40),
            min_facet_area_m2=max(0.5, min_facet_area * 0.5),
            min_facet_points=max(3, min_facet_points // 2),
            max_planes=max_planes,
        )
        if facet_point_lists:
            method += "_relaxed"
    if not facet_point_lists:
        raise ValueError(
            "No roof facets detected even with relaxed thresholds. "
            "Check the address or adjust the options."
        )

    density = estimate_point_density(clipped)
    result, facet_point_lists = assemble_result(
        osm_id, facet_point_lists, clipped, ground_z, density, method,
        min_confidence, max_pitch_deg, lat,
    )

    # Points not assigned to any facet — used for obstacle detection and viz
    from scipy.spatial import cKDTree
    all_roof = np.vstack(facet_point_lists)
    _tree = cKDTree(all_roof)
    _dists, _ = _tree.query(clipped, k=1)
    unassigned_pts = clipped[_dists > 1e-4]

    # Obstacle detection from unassigned point clusters
    obstacles = detect_roof_obstacles(unassigned_pts, facet_point_lists)
    if obstacles:
        result = result.model_copy(update={"obstacles": obstacles})

    return {
        "result": result,
        "facet_points": facet_point_lists,
        "clipped": clipped,
        "unassigned_points": unassigned_pts,
        "polygon_coords": list(polygon.exterior.coords),
        "polygon_bounds": polygon.bounds,
        "osm_id": osm_id,
        "epsg": epsg,
        "ground_z": ground_z,
        "tile_names": [p.name for p in laz_paths],
    }


# ── Visualisation helpers ─────────────────────────────────────────────────────

FACET_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3",
]


_SATELLITE_TILE = (
    "https://server.arcgisonline.com/ArcGIS/rest/services"
    "/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
_SATELLITE_ATTR = (
    "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics"
)


def make_picker_map(lat: float, lon: float) -> "folium.Map":
    """Interactive map: click to set lat/lon, satellite layer toggle."""
    import folium

    m = folium.Map(location=[lat, lon], zoom_start=18, tiles="OpenStreetMap")
    folium.TileLayer(
        tiles=_SATELLITE_TILE, attr=_SATELLITE_ATTR,
        name="Satellite", overlay=False, control=True,
    ).add_to(m)
    folium.Marker(
        [lat, lon],
        tooltip="Click map to move pin",
        icon=folium.Icon(color="red", icon="home"),
    ).add_to(m)
    folium.LayerControl(position="topright").add_to(m)
    return m


def make_footprint_map(polygon_coords, lat: float, lon: float) -> "folium.Map":
    """Footprint map shown after analysis — includes satellite layer + polygon."""
    import folium

    m = folium.Map(location=[lat, lon], zoom_start=18, tiles="OpenStreetMap")
    folium.TileLayer(
        tiles=_SATELLITE_TILE, attr=_SATELLITE_ATTR,
        name="Satellite", overlay=False, control=True,
    ).add_to(m)

    # Building footprint polygon
    poly_latlng = [(c[1], c[0]) for c in polygon_coords]
    folium.Polygon(
        locations=poly_latlng,
        color="#3182CE", weight=2,
        fill=True, fill_color="#63B3ED", fill_opacity=0.3,
        tooltip="Building footprint",
    ).add_to(m)

    # Query point
    folium.Marker(
        [lat, lon],
        tooltip=f"Query: {lat:.6f}, {lon:.6f}",
        icon=folium.Icon(color="red", icon="home"),
    ).add_to(m)

    folium.LayerControl(position="topright").add_to(m)
    return m


def _facet_boundary_3d(points: np.ndarray, normal: np.ndarray):
    """Return a closed (M+1, 3) convex-hull boundary polygon for a facet in 3D.

    Uses the convex hull of the projected points — always a valid polygon.
    The hull vertices are already in counter-clockwise order so no sorting needed.
    """
    from scipy.spatial import ConvexHull

    if len(points) < 6:
        return None

    proj, u, v, centroid = _project_to_plane(points, normal)
    try:
        hull = ConvexHull(proj)
    except Exception:
        return None

    hull_pts = proj[hull.vertices]
    ring_2d  = np.vstack([hull_pts, hull_pts[:1]])   # close the ring
    return centroid + ring_2d[:, 0:1] * u + ring_2d[:, 1:2] * v


def make_3d(facet_points, ground_z, unassigned_points=None, obstacles=None, facet_normals=None):
    import plotly.graph_objects as go

    # Use a single global centre so facets retain their spatial relationship
    all_pts = np.vstack(facet_points)
    cx, cy = all_pts[:, 0].mean(), all_pts[:, 1].mean()

    fig = go.Figure()

    # Unassigned points (grey) — rendered first so they sit behind facets
    if unassigned_points is not None and len(unassigned_points) > 0:
        u = unassigned_points
        fig.add_trace(go.Scatter3d(
            x=u[:, 0] - cx, y=u[:, 1] - cy, z=u[:, 2],
            mode="markers",
            marker=dict(size=1.5, color="#888888", opacity=0.35),
            name=f"Unassigned ({len(u)})",
        ))

    for i, pts in enumerate(facet_points):
        color = FACET_COLORS[i % len(FACET_COLORS)]
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0] - cx, y=pts[:, 1] - cy, z=pts[:, 2],
            mode="markers",
            marker=dict(size=2, color=color, opacity=0.85),
            name=f"Facet {i + 1}",
            legendgroup=f"facet{i}",
        ))
        # Facet boundary wireframe — use cached normal to avoid redundant SVD
        normal = np.array(facet_normals[i]) if facet_normals else None
        bnd = _facet_boundary_3d(pts, normal) if normal is not None else None
        if bnd is not None:
            fig.add_trace(go.Scatter3d(
                x=bnd[:, 0] - cx, y=bnd[:, 1] - cy, z=bnd[:, 2],
                mode="lines",
                line=dict(color=color, width=3),
                name=f"Facet {i + 1} boundary",
                legendgroup=f"facet{i}",
                showlegend=False,
            ))

    # Obstacle centroids
    if obstacles:
        _OBS_COLORS = {"chimney": "#FF4500", "vent_hvac": "#FF8C00", "unknown": "#FFD700"}
        _obs_shown: set[str] = set()
        for obs in obstacles:
            kind = obs.obstacle_type
            ox, oy = obs.centroid_xy[0] - cx, obs.centroid_xy[1] - cy
            # Approximate Z: local roof Z + half the vertical extent
            oz = all_pts[:, 2].mean() + obs.height_above_roof_m * 0.5
            show = kind not in _obs_shown
            _obs_shown.add(kind)
            fig.add_trace(go.Scatter3d(
                x=[ox], y=[oy], z=[oz],
                mode="markers+text",
                marker=dict(size=10, color=_OBS_COLORS.get(kind, "#FFD700"),
                            symbol="diamond", opacity=0.9),
                text=[f"#{obs.obstacle_id}"],
                textposition="top center",
                name=kind.replace("_", " ").title(),
                legendgroup=f"obs_{kind}",
                showlegend=show,
            ))

    # Ground plane reference
    span = max(all_pts[:, 0].max() - all_pts[:, 0].min(),
               all_pts[:, 1].max() - all_pts[:, 1].min()) * 0.6
    gx = [-span, span, span, -span, -span]
    gy = [-span, -span, span, span, -span]
    fig.add_trace(go.Scatter3d(
        x=gx, y=gy, z=[ground_z] * 5,
        mode="lines",
        line=dict(color="rgba(100,100,100,0.4)", width=2),
        name="Ground",
        showlegend=False,
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            camera=dict(eye=dict(x=1.4, y=1.4, z=0.8)),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=480,
        legend=dict(
            orientation="v", x=1.01, y=1,
            font=dict(size=11),
        ),
    )
    return fig


def make_facet_charts(facets: list[FacetResult]):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    ids     = [f"F{f.facet_id}" for f in facets]
    pitches = [f.pitch_deg   for f in facets]
    areas   = [f.area_m2     for f in facets]
    confs   = [f.confidence  for f in facets]
    suits   = [f.solar_suitability or 0.0 for f in facets]
    colors  = [FACET_COLORS[i % len(FACET_COLORS)] for i in range(len(facets))]

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=("Pitch (°)", "Area (m²)", "Confidence", "Solar suitability"),
        horizontal_spacing=0.07,
    )
    fig.add_trace(go.Bar(x=ids, y=pitches, marker_color=colors, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=ids, y=areas,   marker_color=colors, showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=ids, y=confs,   marker_color=colors, showlegend=False), row=1, col=3)
    fig.add_trace(go.Bar(x=ids, y=suits,   marker_color=colors, showlegend=False), row=1, col=4)

    fig.update_yaxes(range=[0, 90],  row=1, col=1)
    fig.update_yaxes(range=[0, 1.0], row=1, col=3)
    fig.update_yaxes(range=[0, 1.0], row=1, col=4)
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=30, b=0))
    return fig


# ── Render helpers ────────────────────────────────────────────────────────────

def _render_single_result(data: dict, q_lat: float, q_lon: float) -> None:
    result: BuildingResult = data["result"]

    st.subheader(f"Building  `{data['osm_id']}`")
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Roof type",       result.roof_type.capitalize())
    m2.metric("Total area",      f"{result.total_roof_area_m2:.1f} m²")
    m3.metric("Solar potential", f"{result.total_solar_kwh_yr:,.0f} kWh/yr" if result.total_solar_kwh_yr else "—")
    m4.metric("Facets",          result.num_facets)
    m5.metric("Height",          f"{result.height_m:.2f} m")
    m6.metric("Eave height",     f"{result.eave_height_m:.2f} m")
    m7.metric("Point density",   f"{result.point_density_m2:.1f} pts/m²" if result.point_density_m2 else "—")

    if result.segmentation_method.endswith("_relaxed"):
        st.warning(
            "⚠️ No facets found with the selected thresholds — results were obtained "
            "with **relaxed parameters**. Consider lowering Min facet area / Min facet points."
        )
    if not result.is_facets_connected:
        st.warning(
            f"⚠️ Facets are **not fully connected** — {result.num_facet_components} disconnected "
            f"group(s) detected. This may indicate a complex roof or segmentation gaps."
        )
    if result.isolated_facet_ids:
        ids_str = ", ".join(f"F{i}" for i in result.isolated_facet_ids)
        st.warning(f"⚠️ Isolated facets (no adjacent neighbours): **{ids_str}**")

    st.divider()

    col_map, col_3d = st.columns([1, 1])
    with col_map:
        from streamlit_folium import st_folium as _st_folium
        st.subheader("Building Footprint")
        st.caption(f"EPSG:{data['epsg']} · Tiles: {', '.join(data['tile_names'])}")
        _st_folium(
            make_footprint_map(data["polygon_coords"], q_lat, q_lon),
            height=340,
            returned_objects=[],
            use_container_width=True,
        )
    with col_3d:
        st.subheader("3D Roof Facets")
        _up = data.get("unassigned_points")
        n_unassigned = len(_up) if _up is not None else 0
        st.caption(
            f"{len(data['clipped'])} LiDAR points · {result.num_facets} facets · "
            f"{n_unassigned} unassigned pts"
        )
        facet_normals = [f.normal_vector for f in result.facets]
        st.plotly_chart(
            make_3d(
                data["facet_points"], data["ground_z"],
                data.get("unassigned_points"),
                result.obstacles or None,
                facet_normals=facet_normals,
            ),
            use_container_width=True,
        )

    st.subheader("Facet Summary")
    st.plotly_chart(make_facet_charts(result.facets), use_container_width=True)

    rows = []
    for f in result.facets:
        rows.append({
            "Facet":             f.facet_id,
            "Pitch (°)":         round(f.pitch_deg, 1),
            "Azimuth (°)":       round(f.azimuth_deg, 0),
            "Area (m²)":         round(f.area_m2, 1),
            "Eave elev (m)":     round(f.eave_elevation_m, 2),
            "Points":            f.num_points,
            "Confidence":        f.confidence,
            "Solar (kWh/m²/yr)": round(f.solar_kwh_m2_yr, 0) if f.solar_kwh_m2_yr is not None else "—",
            "Solar suit.":       f.solar_suitability if f.solar_suitability is not None else "—",
            "RMS (m)":           f.plane_rms_m,
            "Flat":              "✓" if f.is_flat else "",
            "Isolated":          "⚠" if f.facet_id in result.isolated_facet_ids else "",
        })
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Confidence":  st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
            "Solar suit.": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
            "Pitch (°)":   st.column_config.NumberColumn(format="%.1f °"),
        },
    )

    if result.obstacles:
        st.subheader(f"Detected Obstacles ({len(result.obstacles)})")
        obs_rows = [
            {
                "ID":                    o.obstacle_id,
                "Type":                  o.obstacle_type.replace("_", " ").title(),
                "Footprint (m²)":        o.footprint_area_m2,
                "Vert. extent (m)":      o.vertical_extent_m,
                "Height above roof (m)": o.height_above_roof_m,
                "Points":                o.num_points,
                "Confidence":            o.confidence,
            }
            for o in result.obstacles
        ]
        st.dataframe(
            pd.DataFrame(obs_rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    min_value=0, max_value=1, format="%.2f"
                ),
            },
        )

    dl_a, dl_b = st.columns(2)
    dl_a.download_button(
        label="⬇  Download JSON",
        data=result.model_dump_json(indent=2),
        file_name=f"{data['osm_id']}_roof.json",
        mime="application/json",
    )
    dl_b.download_button(
        label="⬇  Download GeoJSON",
        data=build_single_building_geojson(
            result, data["facet_points"], data["polygon_coords"], data["epsg"]
        ),
        file_name=f"{data['osm_id']}_roof.geojson",
        mime="application/geo+json",
    )


def _render_batch_tab(opts: dict) -> None:
    st.markdown("Process multiple addresses in one run. Uses the same options as the sidebar.")
    addresses_raw = st.text_area(
        "Addresses (one per line)",
        placeholder="123 Main St, Oxford, MS\n456 Oak Ave, Austin, TX\n789 Elm St, Memphis, TN",
        height=150,
    )
    run_batch = st.button("▶  Run Batch", type="primary")

    if not run_batch:
        return

    addresses = [a.strip() for a in (addresses_raw or "").splitlines() if a.strip()]
    if not addresses:
        st.warning("Enter at least one address.")
        return

    rows = []
    progress = st.progress(0.0, text="Starting…")
    for idx, addr in enumerate(addresses):
        progress.progress(idx / len(addresses), text=f"{idx + 1}/{len(addresses)}: {addr[:60]}")
        row: dict = {
            "address": addr, "lat": None, "lon": None,
            "building_id": None, "roof_type": None, "num_facets": None,
            "height_m": None, "eave_height_m": None,
            "total_roof_area_m2": None, "total_solar_kwh_yr": None,
            "point_density_m2": None, "unassigned_point_fraction": None,
            "error": None,
        }
        try:
            g_lat, g_lon, _ = geocode_address(addr)
            row["lat"], row["lon"] = round(g_lat, 6), round(g_lon, 6)
            data = run_pipeline(
                g_lat, g_lon,
                opts["min_facet_area"], opts["min_facet_points"],
                opts["min_confidence"], opts["max_pitch_deg"],
                opts["distance_threshold"], opts["max_planes"],
                opts["buffer_m"], opts["footprint_buffer_m"],
            )
            r: BuildingResult = data["result"]
            row.update({
                "building_id":            r.building_id,
                "roof_type":              r.roof_type,
                "num_facets":             r.num_facets,
                "height_m":               r.height_m,
                "eave_height_m":          r.eave_height_m,
                "total_roof_area_m2":     r.total_roof_area_m2,
                "total_solar_kwh_yr":     r.total_solar_kwh_yr,
                "point_density_m2":       r.point_density_m2,
                "unassigned_point_fraction": r.unassigned_point_fraction,
            })
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)

    progress.progress(1.0, text="Done.")
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        label="⬇  Download CSV",
        data=df.to_csv(index=False),
        file_name="batch_roof_results.csv",
        mime="text/csv",
    )


# ── Main UI ───────────────────────────────────────────────────────────────────

opts = dict(
    min_facet_area=min_facet_area, min_facet_points=min_facet_points,
    min_confidence=min_confidence, max_pitch_deg=max_pitch_deg,
    distance_threshold=distance_threshold, max_planes=max_planes,
    buffer_m=buffer_m, footprint_buffer_m=footprint_buffer_m,
)

tab_single, tab_batch = st.tabs(["📍 Single Building", "📋 Batch"])

with tab_single:
    from streamlit_folium import st_folium

    # Interactive location picker — always visible, click to set lat/lon
    st.subheader("📍 Pick Location")
    st.caption("Click anywhere on the map to set the coordinates, then click **▶ Analyze** in the sidebar.")
    map_data = st_folium(
        make_picker_map(lat, lon),
        height=360,
        returned_objects=["last_clicked"],
        key=f"picker_{lat}_{lon}",
        use_container_width=True,
    )
    if map_data and map_data.get("last_clicked"):
        clicked = map_data["last_clicked"]
        # Only rerun if the click is meaningfully different from current coords
        if abs(clicked["lat"] - lat) > 1e-6 or abs(clicked["lng"] - lon) > 1e-6:
            st.session_state["pending_lat"] = clicked["lat"]
            st.session_state["pending_lon"] = clicked["lng"]
            st.rerun()

    st.divider()

    if not run:
        st.info("Click the map to set a location, or enter coordinates in the sidebar, then click **▶ Analyze**.")
    else:
        try:
            with st.status("Running analysis…", expanded=True) as status:
                st.write("📍 Fetching OSM building footprint…")
                st.write("🛰️  Downloading USGS 3DEP LiDAR tile(s)…")
                st.write("✂️  Clipping point cloud to footprint…")
                st.write("📐 Segmenting roof planes…")
                data = run_pipeline(lat, lon, min_facet_area, min_facet_points, min_confidence, max_pitch_deg, distance_threshold, max_planes, buffer_m, footprint_buffer_m)
                status.update(label="Analysis complete", state="complete", expanded=False)
        except Exception as e:
            st.error(f"**Error:** {e}")
        else:
            _render_single_result(data, lat, lon)

with tab_batch:
    _render_batch_tab(opts)
