"""Tests for KPI computation and the baseline-vs-optimised benchmark math."""

from __future__ import annotations

import pytest

from route_optimizer.kpis import benchmark, compute_kpis
from route_optimizer.solution import Solution


def _sol(routes, per_vehicle, total_km, total_time, dropped=None, feasible=True):
    return Solution(
        routes=routes,
        per_vehicle=per_vehicle,
        total_distance_km=total_km,
        total_time_min=total_time,
        dropped=dropped or [],
        provider_used="haversine",
        feasible=feasible,
    )


def test_compute_kpis_fill_and_l_per_km():
    per_vehicle = [
        {
            "stops": 2,
            "load": 6000.0,
            "distance_km": 100.0,
            "time_min": 200.0,
            "cap_ok": True,
            "time_ok": True,
            "window_violations": 0,
        }
    ]
    sol = _sol([[0, 1]], per_vehicle, 100.0, 200.0)
    kpis = compute_kpis(sol, capacity=12000)
    r = kpis["per_route"][0]
    assert r["fill_pct"] == pytest.approx(50.0)  # 6000/12000
    assert r["l_per_km"] == pytest.approx(60.0)  # 6000/100
    g = kpis["global"]
    assert g["clients_served"] == 2
    assert g["total_distance_km"] == pytest.approx(100.0)
    assert g["vehicles_used"] == 1


def test_compute_kpis_counts_violations():
    per_vehicle = [
        {
            "stops": 1,
            "load": 15000.0,
            "distance_km": 50.0,
            "time_min": 700.0,
            "cap_ok": False,
            "time_ok": False,
            "window_violations": 3,
        },
        {
            "stops": 1,
            "load": 1000.0,
            "distance_km": 10.0,
            "time_min": 60.0,
            "cap_ok": True,
            "time_ok": True,
            "window_violations": 0,
        },
    ]
    sol = _sol([[0], [1]], per_vehicle, 60.0, 760.0)
    g = compute_kpis(sol, capacity=12000)["global"]
    assert g["capacity_violations"] == 1
    assert g["time_violations"] == 1
    assert g["window_violations"] == 3


def test_compute_kpis_handles_zero_distance():
    per_vehicle = [
        {
            "stops": 0,
            "load": 0.0,
            "distance_km": 0.0,
            "time_min": 0.0,
            "cap_ok": True,
            "time_ok": True,
            "window_violations": 0,
        }
    ]
    sol = _sol([[]], per_vehicle, 0.0, 0.0)
    r = compute_kpis(sol, capacity=12000)["per_route"][0]
    assert r["l_per_km"] == 0.0  # no division by zero


def _baseline_sol():
    pv = [
        {"stops": 2, "load": 5000.0, "distance_km": 120.0, "time_min": 300.0,
         "cap_ok": True, "time_ok": True, "window_violations": 4},
    ]
    return _sol([[0, 1]], pv, 120.0, 300.0)


def _optimized_sol():
    pv = [
        {"stops": 2, "load": 5000.0, "distance_km": 90.0, "time_min": 250.0,
         "cap_ok": True, "time_ok": True, "window_violations": 1},
    ]
    return _sol([[0, 1]], pv, 90.0, 250.0)


def test_benchmark_km_saved_and_pct():
    b = benchmark(_baseline_sol(), _optimized_sol())
    assert b["baseline_km"] == pytest.approx(120.0)
    assert b["optimized_km"] == pytest.approx(90.0)
    assert b["km_saved"] == pytest.approx(30.0)
    assert b["km_saved_pct"] == pytest.approx(25.0)  # 30/120


def test_benchmark_window_violation_reduction():
    b = benchmark(_baseline_sol(), _optimized_sol())
    assert b["baseline_window_violations"] == 4
    assert b["optimized_window_violations"] == 1
    assert b["window_violations_reduced"] == 3


def test_benchmark_zero_baseline_km_no_crash():
    empty = _sol([[]], [], 0.0, 0.0)
    b = benchmark(empty, empty)
    assert b["km_saved_pct"] == 0.0  # guarded division


def test_benchmark_served_and_dropped_counts():
    base = _baseline_sol()
    opt = _optimized_sol()
    opt.dropped = [5, 6]
    b = benchmark(base, opt)
    assert b["baseline_clients_served"] == 2
    assert b["optimized_clients_served"] == 2
    assert b["optimized_dropped"] == 2
