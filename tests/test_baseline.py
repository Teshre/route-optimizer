"""Tests for the K-Means + nearest-neighbour baseline."""

from __future__ import annotations

import pandas as pd

from route_optimizer.baseline import solve_baseline
from route_optimizer.config import (
    COL_ADDRESS,
    COL_LAT,
    COL_LON,
    COL_NAME,
    COL_VOLUME,
    COL_WINDOW,
)


def _make_df(rows):
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


def _df6():
    return _make_df(
        [
            ("A", 19.40, -99.60, 1000, "08:00-12:00"),
            ("B", 19.41, -99.61, 1000, "08:00-12:00"),
            ("C", 19.30, -99.50, 1000, "14:00-18:00"),
            ("D", 19.31, -99.51, 1000, "14:00-18:00"),
            ("E", 19.45, -99.45, 1000, "09:00-13:00"),
            ("F", 19.46, -99.46, 1000, "09:00-13:00"),
        ]
    )


def test_baseline_serves_every_client_never_drops():
    df = _df6()
    sol = solve_baseline(df, num_vehicles=3, capacity=12000)
    assert sol.dropped == []
    assert sol.num_served == len(df)


def test_baseline_partition_is_complete_and_disjoint():
    df = _df6()
    sol = solve_baseline(df, num_vehicles=3, capacity=12000)
    all_rows = sorted(r for route in sol.routes for r in route)
    assert all_rows == list(range(len(df)))  # every row exactly once


def test_baseline_totals_match_per_route_sum():
    df = _df6()
    sol = solve_baseline(df, num_vehicles=3, capacity=12000)
    assert sol.total_distance_km == sum(
        pv["distance_km"] for pv in sol.per_vehicle
    )
    assert sol.total_time_min == sum(pv["time_min"] for pv in sol.per_vehicle)


def test_baseline_flags_capacity_violation():
    # One truck, everyone in one cluster, demand exceeds capacity.
    df = _df6()  # total demand 6000
    sol = solve_baseline(df, num_vehicles=1, capacity=3000)
    # Single route carries all 6000 > 3000 -> cap_ok False, feasible False.
    served_route = [pv for pv in sol.per_vehicle if pv["stops"] > 0][0]
    assert served_route["load"] == 6000.0
    assert served_route["cap_ok"] is False
    assert sol.feasible is False


def test_baseline_empty_df():
    df = _make_df([])
    sol = solve_baseline(df, num_vehicles=2, capacity=5000)
    assert sol.num_served == 0
    assert sol.total_distance_km == 0.0
