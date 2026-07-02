"""Tests for the OR-Tools CVRPTW solver on small hand-built instances."""

from __future__ import annotations

import pandas as pd
import pytest

from route_optimizer.config import (
    COL_ADDRESS,
    COL_LAT,
    COL_LON,
    COL_NAME,
    COL_VOLUME,
    COL_WINDOW,
    PROVIDER_HAVERSINE,
)
from route_optimizer.solver import solve_cvrptw


def _make_df(rows):
    """rows: list of (name, lat, lon, volume, window)."""
    return pd.DataFrame(
        [
            {
                COL_NAME: name,
                COL_ADDRESS: "addr",
                COL_LAT: lat,
                COL_LON: lon,
                COL_VOLUME: vol,
                COL_WINDOW: win,
            }
            for (name, lat, lon, vol, win) in rows
        ]
    )


def test_small_instance_serves_all_and_respects_capacity():
    # 4 clients near the depot; capacity comfortably covers all demand.
    df = _make_df(
        [
            ("A", 19.40, -99.60, 1000, "08:00-18:00"),
            ("B", 19.35, -99.55, 1000, "08:00-18:00"),
            ("C", 19.42, -99.50, 1000, "08:00-18:00"),
            ("D", 19.30, -99.62, 1000, "08:00-18:00"),
        ]
    )
    sol = solve_cvrptw(
        df,
        provider=PROVIDER_HAVERSINE,
        num_vehicles=2,
        capacity=5000,
        service_time=10,
        shift_minutes=630,
        time_limit_s=3,
    )
    # Every client served, none dropped.
    assert sol.dropped == []
    assert sol.num_served == 4
    # Capacity respected on every route.
    for pv in sol.per_vehicle:
        assert pv["load"] <= 5000 + 1e-6
        assert pv["cap_ok"] is True
    assert sol.feasible is True


def test_capacity_forces_split_across_vehicles():
    # Two clients each demanding 8000; one truck (cap 10000) cannot take both.
    df = _make_df(
        [
            ("A", 19.40, -99.60, 8000, "08:00-18:00"),
            ("B", 19.35, -99.55, 8000, "08:00-18:00"),
        ]
    )
    sol = solve_cvrptw(
        df,
        num_vehicles=2,
        capacity=10000,
        service_time=10,
        shift_minutes=630,
        time_limit_s=3,
    )
    assert sol.dropped == []
    # No single route may exceed capacity -> they must be on different trucks.
    for pv in sol.per_vehicle:
        assert pv["load"] <= 10000 + 1e-6
    loads = sorted(pv["load"] for pv in sol.per_vehicle)
    assert loads == [8000.0, 8000.0]


def test_capacity_forces_drop_when_infeasible():
    # Total demand 20000 but a single truck cap 10000 -> at least one dropped.
    df = _make_df(
        [
            ("A", 19.40, -99.60, 8000, "08:00-18:00"),
            ("B", 19.35, -99.55, 8000, "08:00-18:00"),
            ("C", 19.42, -99.50, 8000, "08:00-18:00"),
        ]
    )
    sol = solve_cvrptw(
        df,
        num_vehicles=1,
        capacity=10000,
        service_time=10,
        shift_minutes=630,
        time_limit_s=3,
    )
    # Only one 8000-client fits; the other two must be dropped.
    assert len(sol.dropped) == 2
    assert sol.num_served == 1
    for pv in sol.per_vehicle:
        assert pv["load"] <= 10000 + 1e-6


def test_time_windows_are_respected_by_arrivals():
    # Client B has a tight late window; served arrivals must fall within window.
    df = _make_df(
        [
            ("A", 19.40, -99.60, 500, "08:00-09:00"),
            ("B", 19.35, -99.55, 500, "10:00-11:00"),
            ("C", 19.42, -99.50, 500, "12:00-13:00"),
        ]
    )
    sol = solve_cvrptw(
        df,
        num_vehicles=1,
        capacity=5000,
        service_time=10,
        shift_minutes=630,
        time_limit_s=3,
    )
    # No window violations reported, and all served within their bounds.
    from route_optimizer.config import parse_window

    windows = [parse_window(w) for w in df[COL_WINDOW]]
    for v, route in enumerate(sol.routes):
        arrivals = sol.per_vehicle[v]["arrivals"]
        for seq, row in enumerate(route):
            earliest, latest = windows[row]
            arr = arrivals[seq]
            assert earliest - 1e-6 <= arr <= latest + 1e-6
        assert sol.per_vehicle[v]["window_violations"] == 0


def test_empty_dataframe_returns_feasible_empty_solution():
    df = _make_df([])
    sol = solve_cvrptw(df, num_vehicles=2, capacity=5000, time_limit_s=1)
    assert sol.num_served == 0
    assert sol.dropped == []
    assert sol.feasible is True


def test_shared_matrix_is_used():
    # Supplying matrices directly should bypass build_matrices and use them.
    df = _make_df(
        [
            ("A", 19.40, -99.60, 500, "08:00-18:00"),
            ("B", 19.35, -99.55, 500, "08:00-18:00"),
        ]
    )
    # 3 nodes (depot + 2). Symmetric, cheap distances.
    dist = [
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ]
    time = [
        [0.0, 2.0, 2.0],
        [2.0, 0.0, 2.0],
        [2.0, 2.0, 0.0],
    ]
    sol = solve_cvrptw(
        df,
        num_vehicles=1,
        capacity=5000,
        time_limit_s=2,
        distance_matrix=dist,
        time_matrix=time,
        provider_used="haversine",
    )
    assert sol.dropped == []
    assert sol.num_served == 2
    assert sol.provider_used == "haversine"
