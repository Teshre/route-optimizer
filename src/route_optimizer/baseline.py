"""
The naive **baseline** heuristic — the "V1" approach — kept so V2 can quantify
how much the OR-Tools CVRPTW actually buys us.

Algorithm
---------
1. **K-Means** on client ``(lat, lon)`` into ``num_vehicles`` clusters — one
   cluster per truck. This is geographic only; it ignores capacity and time
   windows entirely (that is the point — it is the naive strawman).
2. **Nearest-neighbour** ordering within each cluster, starting from the depot:
   repeatedly hop to the closest not-yet-visited client.

The result is packed into the same :class:`~route_optimizer.solution.Solution`
shape as the OR-Tools solver, so :mod:`route_optimizer.kpis` can benchmark them
head-to-head. Because the baseline honours neither capacity nor windows, its
per-route ``cap_ok`` / ``time_ok`` / ``window_violations`` fields will typically
flag the constraint breaches that the real solver avoids.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .config import (
    COL_VOLUME,
    COL_WINDOW,
    DEPOT,
    NUM_VEHICLES,
    PROVIDER_HAVERSINE,
    SERVICE_TIME_MIN,
    SHIFT_MINUTES,
    VEHICLE_CAPACITY,
    parse_window,
)
from .matrix import build_matrices
from .solution import Solution


def _nearest_neighbour_order(
    node_rows: list[int], distance_matrix: list[list[float]]
) -> list[int]:
    """Order ``node_rows`` (DataFrame row-indices) by nearest-neighbour.

    Distances use matrix node indices (row ``r`` -> node ``r + 1``); the tour
    starts from the depot (node ``0``).
    """
    remaining = list(node_rows)
    order: list[int] = []
    current_node = 0  # depot
    while remaining:
        nxt = min(remaining, key=lambda r: distance_matrix[current_node][r + 1])
        order.append(nxt)
        remaining.remove(nxt)
        current_node = nxt + 1
    return order


def solve_baseline(
    df: pd.DataFrame,
    provider: str = PROVIDER_HAVERSINE,
    num_vehicles: int = NUM_VEHICLES,
    capacity: int = VEHICLE_CAPACITY,
    service_time: int = SERVICE_TIME_MIN,
    shift_minutes: int = SHIFT_MINUTES,
    distance_matrix: Optional[list[list[float]]] = None,
    time_matrix: Optional[list[list[float]]] = None,
    provider_used: Optional[str] = None,
    random_state: int = 42,
) -> Solution:
    """K-Means + nearest-neighbour baseline, ignoring capacity & time windows.

    Signature mirrors :func:`route_optimizer.solver.solve_cvrptw` so callers can
    swap them freely. ``distance_matrix`` / ``time_matrix`` may be supplied to
    share the exact same matrices with the optimised solver for a fair compare.
    """
    n_clients = len(df)

    if n_clients == 0:
        return Solution(
            routes=[[] for _ in range(num_vehicles)],
            per_vehicle=[],
            total_distance_km=0.0,
            total_time_min=0.0,
            dropped=[],
            provider_used=provider_used or PROVIDER_HAVERSINE,
            feasible=True,
        )

    # --- Matrices ----------------------------------------------------------
    if distance_matrix is None or time_matrix is None:
        coords = [DEPOT] + list(
            zip(df["Latitud"].tolist(), df["Longitud"].tolist())
        )
        distance_matrix, time_matrix, provider_used = build_matrices(
            coords, provider
        )
    elif provider_used is None:
        provider_used = provider

    demands = df[COL_VOLUME].tolist()
    windows = [parse_window(w) for w in df[COL_WINDOW].tolist()]

    # --- K-Means clustering ------------------------------------------------
    latlon = df[["Latitud", "Longitud"]].to_numpy(dtype=float)
    k = min(num_vehicles, n_clients)  # cannot ask for more clusters than points
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(latlon)

    clusters: list[list[int]] = [[] for _ in range(num_vehicles)]
    for row, lab in enumerate(labels):
        clusters[int(lab)].append(row)

    # --- Nearest-neighbour ordering + per-route metrics --------------------
    routes: list[list[int]] = []
    per_vehicle: list[dict] = []
    total_distance_km = 0.0
    total_time_min = 0.0

    for cluster_rows in clusters:
        if not cluster_rows:
            routes.append([])
            per_vehicle.append(
                {
                    "stops": 0,
                    "load": 0.0,
                    "distance_km": 0.0,
                    "time_min": 0.0,
                    "cap_ok": True,
                    "time_ok": True,
                    "window_violations": 0,
                    "arrivals": [],
                }
            )
            continue

        order = _nearest_neighbour_order(cluster_rows, distance_matrix)
        routes.append(order)

        load = float(sum(demands[r] for r in order))
        route_km = 0.0
        clock = 0.0  # minutes from DEPART; baseline departs the depot at t=0
        arrivals: list[float] = []
        window_violations = 0
        prev_node = 0  # depot

        for r in order:
            node = r + 1
            route_km += distance_matrix[prev_node][node]
            clock += time_matrix[prev_node][node]  # travel to client
            arrivals.append(float(clock))
            earliest, latest = windows[r]
            # Baseline ignores windows -> just record whether it breached.
            if clock > latest or clock < earliest:
                window_violations += 1
            clock += service_time  # service the client
            prev_node = node

        # Return to depot to close the tour time/distance.
        route_km += distance_matrix[prev_node][0]
        clock += time_matrix[prev_node][0]

        cap_ok = load <= capacity + 1e-6
        time_ok = clock <= shift_minutes + 1e-6

        per_vehicle.append(
            {
                "stops": len(order),
                "load": load,
                "distance_km": float(route_km),
                "time_min": float(clock),
                "cap_ok": bool(cap_ok),
                "time_ok": bool(time_ok),
                "window_violations": int(window_violations),
                "arrivals": arrivals,
            }
        )
        total_distance_km += route_km
        total_time_min += clock

    # The baseline serves every client (it just may violate constraints).
    feasible = all(pv["cap_ok"] and pv["time_ok"] for pv in per_vehicle)

    return Solution(
        routes=routes,
        per_vehicle=per_vehicle,
        total_distance_km=float(total_distance_km),
        total_time_min=float(total_time_min),
        dropped=[],
        provider_used=provider_used or provider,
        feasible=bool(feasible),
    )
