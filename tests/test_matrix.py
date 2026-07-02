"""Tests for the distance/time matrix builder (Haversine + OSRM fallback)."""

from __future__ import annotations

from unittest import mock

import pytest

from route_optimizer.config import DEPOT, PROVIDER_HAVERSINE, PROVIDER_OSRM
from route_optimizer.matrix import (
    build_matrices,
    clear_cache,
    haversine_km,
)


@pytest.fixture(autouse=True)
def _clear():
    clear_cache()
    yield
    clear_cache()


def _coords():
    return [
        DEPOT,
        (19.40, -99.60),
        (19.35, -99.55),
        (19.42, -99.50),
    ]


def test_haversine_zero_diagonal_and_symmetric():
    dist, time, prov = build_matrices(_coords(), PROVIDER_HAVERSINE)
    n = len(_coords())
    assert prov == PROVIDER_HAVERSINE
    for i in range(n):
        assert dist[i][i] == 0.0
        assert time[i][i] == 0.0
        for j in range(n):
            assert dist[i][j] == pytest.approx(dist[j][i], abs=1e-9)
            assert time[i][j] == pytest.approx(time[j][i], abs=1e-9)


def test_haversine_time_is_distance_over_speed():
    from route_optimizer.config import SPEED_KMH

    dist, time, _ = build_matrices(_coords(), PROVIDER_HAVERSINE)
    for i in range(len(_coords())):
        for j in range(len(_coords())):
            expected = dist[i][j] / SPEED_KMH * 60.0
            assert time[i][j] == pytest.approx(expected, abs=1e-9)


def test_haversine_known_distance():
    # ~1 degree of latitude ~= 111 km.
    d = haversine_km((0.0, 0.0), (1.0, 0.0))
    assert d == pytest.approx(111.19, abs=1.0)


def test_empty_coords():
    dist, time, prov = build_matrices([], PROVIDER_HAVERSINE)
    assert dist == [] and time == [] and prov == PROVIDER_HAVERSINE


def test_unknown_provider_falls_back_to_haversine():
    dist, _, prov = build_matrices(_coords(), "not-a-provider")
    assert prov == PROVIDER_HAVERSINE
    assert dist[0][0] == 0.0


def test_osrm_network_failure_falls_back_cleanly():
    """A raised exception inside the OSRM path must NOT crash the build."""
    import route_optimizer.matrix as m

    with mock.patch.object(
        m.requests, "get", side_effect=Exception("network down")
    ):
        dist, time, prov = build_matrices(_coords(), PROVIDER_OSRM)

    # Fell back to haversine and produced a valid, symmetric matrix.
    assert prov == PROVIDER_HAVERSINE
    n = len(_coords())
    for i in range(n):
        assert dist[i][i] == 0.0
        for j in range(n):
            assert dist[i][j] == pytest.approx(dist[j][i], abs=1e-9)


def test_osrm_success_reports_osrm_provider():
    """When OSRM returns a well-formed table, provider_used == 'osrm'."""
    import route_optimizer.matrix as m

    n = len(_coords())
    # metres / seconds payload
    distances = [[abs(i - j) * 1000.0 for j in range(n)] for i in range(n)]
    durations = [[abs(i - j) * 60.0 for j in range(n)] for i in range(n)]

    fake_resp = mock.Mock()
    fake_resp.raise_for_status = mock.Mock()
    fake_resp.json = mock.Mock(
        return_value={
            "code": "Ok",
            "distances": distances,
            "durations": durations,
        }
    )

    with mock.patch.object(m.requests, "get", return_value=fake_resp):
        dist, time, prov = build_matrices(_coords(), PROVIDER_OSRM)

    assert prov == PROVIDER_OSRM
    # 1000 m -> 1 km, 60 s -> 1 min
    assert dist[0][1] == pytest.approx(1.0)
    assert time[0][1] == pytest.approx(1.0)


def test_osrm_timeout_falls_back():
    import route_optimizer.matrix as m
    import requests as real_requests

    with mock.patch.object(
        m.requests, "get", side_effect=real_requests.Timeout("timed out")
    ):
        _, _, prov = build_matrices(_coords(), PROVIDER_OSRM)
    assert prov == PROVIDER_HAVERSINE
