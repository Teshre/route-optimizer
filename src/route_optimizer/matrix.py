"""
Distance / time matrix construction for Route Optimizer V2.

Two providers are supported:

* ``"haversine"`` — great-circle distances (km) computed locally; travel time is
  estimated as ``km / SPEED_KMH * 60`` minutes. Always available, no network.
* ``"osrm"`` — real road distances and durations from a public OSRM ``/table``
  endpoint. On **any** failure (network error, timeout, bad payload, HTTP
  error) it falls back to Haversine and records that in the returned flag, so a
  network hiccup can never crash the solver.

The public entry point is :func:`build_matrices`, which returns
``(dist_km, time_min, provider_used)`` where ``dist_km`` / ``time_min`` are
``n x n`` lists of floats and ``coords[0]`` is always the depot.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

import requests

from .config import (
    PROVIDER_HAVERSINE,
    PROVIDER_OSRM,
    SPEED_KMH,
)

# Public OSRM demo server. Coordinates are sent as lon,lat (OSRM convention).
OSRM_BASE_URL = "https://router.project-osrm.org"
OSRM_TIMEOUT_S = 8.0

EARTH_RADIUS_KM = 6371.0088

Coord = tuple[float, float]  # (lat, lon)


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------
def haversine_km(a: Coord, b: Coord) -> float:
    """Great-circle distance in kilometres between two ``(lat, lon)`` points."""
    lat1, lon1 = a
    lat2, lon2 = b
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    h = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def _haversine_matrices(
    coords: Sequence[Coord], speed_kmh: float = SPEED_KMH
) -> tuple[list[list[float]], list[list[float]]]:
    """Build symmetric distance (km) and time (min) matrices via Haversine."""
    n = len(coords)
    dist = [[0.0] * n for _ in range(n)]
    time = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(coords[i], coords[j])
            t = (d / speed_kmh) * 60.0 if speed_kmh > 0 else 0.0
            dist[i][j] = dist[j][i] = d
            time[i][j] = time[j][i] = t
    return dist, time


# ---------------------------------------------------------------------------
# OSRM
# ---------------------------------------------------------------------------
def _osrm_matrices(
    coords: Sequence[Coord],
    base_url: str = OSRM_BASE_URL,
    timeout: float = OSRM_TIMEOUT_S,
) -> tuple[list[list[float]], list[list[float]]]:
    """Query an OSRM ``/table`` endpoint for road distances and durations.

    Raises on any failure so the caller can fall back to Haversine.
    """
    # OSRM wants "lon,lat;lon,lat;..." — note the flip from our (lat, lon).
    coord_str = ";".join(f"{lon:.6f},{lat:.6f}" for (lat, lon) in coords)
    url = f"{base_url}/table/v1/driving/{coord_str}"
    params = {"annotations": "distance,duration"}

    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    if payload.get("code") != "Ok":
        raise ValueError(f"OSRM returned code={payload.get('code')!r}")

    durations = payload.get("durations")  # seconds
    distances = payload.get("distances")  # metres
    if not durations or not distances:
        raise ValueError("OSRM response missing durations/distances")

    n = len(coords)
    dist_km = [[0.0] * n for _ in range(n)]
    time_min = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            d = distances[i][j]
            t = durations[i][j]
            # OSRM can return null for unreachable pairs; substitute Haversine.
            if d is None or t is None:
                d = haversine_km(coords[i], coords[j]) * 1000.0
                t = (d / 1000.0 / SPEED_KMH) * 3600.0 if SPEED_KMH > 0 else 0.0
            dist_km[i][j] = float(d) / 1000.0
            time_min[i][j] = float(t) / 60.0
    return dist_km, time_min


# ---------------------------------------------------------------------------
# Caching by rounded-coords key
# ---------------------------------------------------------------------------
def _round_key(coords: Sequence[Coord]) -> tuple:
    """Hashable cache key: coordinates rounded to ~1 m precision."""
    return tuple((round(lat, 5), round(lon, 5)) for (lat, lon) in coords)


@lru_cache(maxsize=32)
def _cached_build(
    key: tuple, provider: str
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...], str]:
    """Cached core builder keyed on rounded coords + provider.

    Stores matrices as tuples-of-tuples so the ``lru_cache`` value is immutable
    and safe to share; :func:`build_matrices` copies them back into lists.
    """
    coords = [(lat, lon) for (lat, lon) in key]
    provider_used = provider

    if provider == PROVIDER_OSRM:
        try:
            dist, time = _osrm_matrices(coords)
            provider_used = PROVIDER_OSRM
        except Exception:  # noqa: BLE001 — any failure => graceful fallback
            dist, time = _haversine_matrices(coords)
            provider_used = PROVIDER_HAVERSINE
    else:
        dist, time = _haversine_matrices(coords)
        provider_used = PROVIDER_HAVERSINE

    dist_t = tuple(tuple(row) for row in dist)
    time_t = tuple(tuple(row) for row in time)
    return dist_t, time_t, provider_used


def build_matrices(
    coords: Sequence[Coord], provider: str = PROVIDER_HAVERSINE
) -> tuple[list[list[float]], list[list[float]], str]:
    """Build distance (km) and time (min) matrices for ``coords``.

    Parameters
    ----------
    coords:
        Sequence of ``(lat, lon)`` points. ``coords[0]`` is the depot.
    provider:
        :data:`~route_optimizer.config.PROVIDER_HAVERSINE` (default, offline) or
        :data:`~route_optimizer.config.PROVIDER_OSRM` (real road network).

    Returns
    -------
    tuple[list[list[float]], list[list[float]], str]
        ``(dist_km, time_min, provider_used)``. ``provider_used`` is
        ``"osrm"`` only if the OSRM call actually succeeded; otherwise
        ``"haversine"`` (including when ``provider="osrm"`` but the request
        failed / timed out / returned an error).
    """
    if provider not in (PROVIDER_HAVERSINE, PROVIDER_OSRM):
        provider = PROVIDER_HAVERSINE

    coords = list(coords)
    if len(coords) == 0:
        return [], [], PROVIDER_HAVERSINE

    key = _round_key(coords)
    dist_t, time_t, provider_used = _cached_build(key, provider)

    dist = [list(row) for row in dist_t]
    time = [list(row) for row in time_t]
    return dist, time, provider_used


def clear_cache() -> None:
    """Clear the matrix cache (useful in tests)."""
    _cached_build.cache_clear()
