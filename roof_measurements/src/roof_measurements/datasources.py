"""Fetch LiDAR point cloud data from USGS 3DEP via Microsoft Planetary Computer.

Coverage: contiguous United States + parts of Alaska, Hawaii, and territories.
Data: USGS 3DEP Cloud-Optimized Point Cloud (COPC) tiles, ~1 km² each, ~10-100 MB.
Check coverage: https://apps.nationalmap.gov/lidar-explorer/

Typical usage
-------------
    from roof_measurements.datasources import fetch_lidar_for_point

    laz_path, epsg = fetch_lidar_for_point(lat=30.2672, lon=-97.7431)
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PC_STAC   = "https://planetarycomputer.microsoft.com/api/stac/v1"
_PC_SIGN   = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"
_COLLECTION = "3dep-lidar-copc"
_DEFAULT_CACHE = Path.home() / ".cache" / "roof_measurements" / "lidar"


# ---------------------------------------------------------------------------
# Tile search via Planetary Computer STAC
# ---------------------------------------------------------------------------

def search_3dep_tiles(lat: float, lon: float, max_results: int = 5) -> list[dict]:
    """Search USGS 3DEP COPC tiles that contain *(lat, lon)*.

    Uses the Microsoft Planetary Computer STAC API (no auth required for search).

    Returns
    -------
    List of STAC feature dicts, empty if none found.
    """
    import requests

    body = {
        "collections": [_COLLECTION],
        "intersects": {"type": "Point", "coordinates": [lon, lat]},
        "limit": max_results,
    }
    logger.debug("STAC search: %s", body)
    resp = requests.post(f"{_PC_STAC}/search", json=body, timeout=30)
    resp.raise_for_status()
    features = resp.json().get("features", [])
    logger.info("3DEP STAC: %d tile(s) found for (%.5f, %.5f)", len(features), lat, lon)
    return features


# ---------------------------------------------------------------------------
# URL signing (Planetary Computer SAS token)
# ---------------------------------------------------------------------------

def _sign_url(href: str) -> str:
    """Return a time-limited signed URL for a Planetary Computer blob asset."""
    import requests

    resp = requests.get(_PC_SIGN, params={"href": href}, timeout=15)
    resp.raise_for_status()
    return resp.json()["href"]


# ---------------------------------------------------------------------------
# Download with caching + progress
# ---------------------------------------------------------------------------

def download_tile(url: str, cache_dir: Path = _DEFAULT_CACHE) -> Path:
    """Download a LAZ/COPC tile to *cache_dir*, skipping if already present.

    Handles Planetary Computer SAS-token signing automatically when the URL
    is a blob.core.windows.net asset.

    Returns
    -------
    Path to the local file.
    """
    import requests

    cache_dir.mkdir(parents=True, exist_ok=True)

    basename = url.split("?")[0].split("/")[-1]
    local = cache_dir / basename

    if local.exists():
        logger.info("Cache hit: %s (%d MB)", local.name, local.stat().st_size >> 20)
        return local

    # Sign if it's a Planetary Computer blob asset
    download_url = _sign_url(url) if "blob.core.windows.net" in url else url

    logger.info("Downloading %s", basename)
    with requests.get(download_url, stream=True, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(local, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(
                        f"\r  {basename}  {100 * downloaded / total:.0f}%"
                        f"  ({downloaded >> 20}/{total >> 20} MB)",
                        end="", flush=True,
                    )
    print()

    logger.info("Saved %d MB → %s", local.stat().st_size >> 20, local)
    return local


# ---------------------------------------------------------------------------
# CRS extraction
# ---------------------------------------------------------------------------

def epsg_from_las(path: Path) -> int | None:
    """Extract the horizontal EPSG code embedded in a LAS/LAZ file header.

    Handles:
    - laspy-typed WktCoordinateSystemVlr (COPC and LAS 1.4 files)
    - Raw WKT VLR (record_id 2112) for generic LAZ files
    - Compound CRS (horizontal + vertical) — returns the horizontal EPSG
    - GeoKeyDirectory VLR (record_id 34735, key 3072) for older LAS 1.2 files
    """
    import struct
    import laspy
    import pyproj

    with laspy.open(path) as f:
        las = f.read()

    def _horizontal_epsg(crs: "pyproj.CRS") -> int | None:
        """Return EPSG of the horizontal component, handling compound CRS."""
        # Direct match first
        epsg = crs.to_epsg()
        if epsg:
            return epsg
        # Compound CRS: extract the projected/geographic sub-CRS
        try:
            sub = crs.sub_crs_list
            if sub:
                for s in sub:
                    if s.is_projected or s.is_geographic:
                        e = s.to_epsg()
                        if e:
                            return e
        except Exception:
            pass
        return None

    for vlr in las.vlrs:
        # laspy-typed WKT VLR (COPC / LAS 1.4)
        if vlr.record_id == 2112 and hasattr(vlr, "parse_crs"):
            try:
                crs = vlr.parse_crs()
                if crs:
                    epsg = _horizontal_epsg(crs)
                    if epsg:
                        logger.debug("EPSG %d from typed WKT VLR", epsg)
                        return epsg
            except Exception:
                pass

        # Raw WKT VLR (generic LAZ)
        if vlr.record_id == 2112 and hasattr(vlr, "record_data_bytes"):
            try:
                wkt = bytes(vlr.record_data_bytes()).decode("utf-8", errors="ignore").rstrip("\x00")
                crs = pyproj.CRS.from_wkt(wkt)
                epsg = _horizontal_epsg(crs)
                if epsg:
                    logger.debug("EPSG %d from raw WKT VLR", epsg)
                    return epsg
            except Exception:
                pass

        # GeoKeyDirectory (older LAS 1.2 files)
        if getattr(vlr, "user_id", "") == "LASF_Projection" and vlr.record_id == 34735:
            try:
                data = bytes(vlr.record_data_bytes())
                n_keys = struct.unpack_from("<H", data, 6)[0]
                for i in range(n_keys):
                    off = 8 + i * 8
                    key_id = struct.unpack_from("<H", data, off)[0]
                    val    = struct.unpack_from("<H", data, off + 6)[0]
                    if key_id == 3072 and val > 0:
                        logger.debug("EPSG %d from GeoKey 3072", val)
                        return int(val)
            except Exception:
                pass

    logger.warning("Could not determine EPSG from %s", path.name)
    return None


# ---------------------------------------------------------------------------
# High-level: point → (laz_path, epsg)
# ---------------------------------------------------------------------------

def fetch_lidar_for_point(
    lat: float,
    lon: float,
    cache_dir: Path = _DEFAULT_CACHE,
) -> tuple[list[Path], int]:
    """Find, download, and return *(laz_paths, epsg)* for a lat/lon coordinate.

    Returns all STAC tiles that overlap the point (typically 1, occasionally 2
    when a building sits on a tile boundary).  Callers should merge points from
    all returned paths before clipping to the building footprint.

    Parameters
    ----------
    lat, lon :
        WGS84 decimal degrees.
    cache_dir :
        Where to store downloaded tiles.

    Returns
    -------
    laz_paths : list[Path]
        Local paths to downloaded COPC LAZ files (≥ 1).
    epsg : int
        EPSG code of the projected CRS (read from the first tile).

    Raises
    ------
    ValueError
        No 3DEP tiles found, or CRS unresolvable.
    """
    tiles = search_3dep_tiles(lat, lon)
    if not tiles:
        raise ValueError(
            f"No USGS 3DEP LiDAR tiles found for ({lat:.5f}, {lon:.5f}).\n"
            "3DEP covers the contiguous US + parts of AK/HI/territories.\n"
            "Check coverage: https://apps.nationalmap.gov/lidar-explorer/"
        )

    laz_paths: list[Path] = []
    epsg: int | None = None

    for tile in tiles:
        logger.info("Fetching tile: %s", tile["id"])
        href = tile["assets"]["data"]["href"]
        laz_path = download_tile(href, cache_dir)
        laz_paths.append(laz_path)

        if epsg is None:
            epsg = epsg_from_las(laz_path)

    if epsg is None:
        raise ValueError(
            f"Could not read CRS from any downloaded tile. "
            "Specify the EPSG manually with --epsg."
        )

    return laz_paths, epsg
