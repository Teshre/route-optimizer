"""
Route Optimizer V2 — Capacitated VRP with Time Windows (CVRPTW).

Public API:

* :func:`~route_optimizer.config.load_clients`,
  :func:`~route_optimizer.config.parse_window` — data loading / window parsing.
* :func:`~route_optimizer.matrix.build_matrices` — distance/time matrices
  (Haversine or OSRM with graceful fallback).
* :func:`~route_optimizer.solver.solve_cvrptw` — the OR-Tools CVRPTW solver.
* :func:`~route_optimizer.baseline.solve_baseline` — K-Means + nearest-neighbour
  baseline.
* :func:`~route_optimizer.kpis.compute_kpis`,
  :func:`~route_optimizer.kpis.benchmark` — KPIs and the baseline-vs-optimised
  comparison.
* :class:`~route_optimizer.solution.Solution` — the shared result container.
"""

from __future__ import annotations

from .baseline import solve_baseline
from .config import (
    DEFAULT_CSV_PATH,
    DEPOT,
    NUM_VEHICLES,
    PROVIDER_HAVERSINE,
    PROVIDER_OSRM,
    SERVICE_TIME_MIN,
    SHIFT_MINUTES,
    VEHICLE_CAPACITY,
    load_clients,
    parse_window,
)
from .kpis import benchmark, compute_kpis
from .matrix import build_matrices, haversine_km
from .solution import Solution
from .solver import solve_cvrptw

__all__ = [
    "Solution",
    "load_clients",
    "parse_window",
    "build_matrices",
    "haversine_km",
    "solve_cvrptw",
    "solve_baseline",
    "compute_kpis",
    "benchmark",
    "DEPOT",
    "NUM_VEHICLES",
    "VEHICLE_CAPACITY",
    "SERVICE_TIME_MIN",
    "SHIFT_MINUTES",
    "PROVIDER_HAVERSINE",
    "PROVIDER_OSRM",
    "DEFAULT_CSV_PATH",
]

__version__ = "2.0.0"
