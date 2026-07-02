"""
The centrepiece of Route Optimizer V2: a Capacitated Vehicle Routing Problem
with Time Windows (**CVRPTW**) solved with Google OR-Tools.

Model
-----
* **Nodes** — node ``0`` is the depot; node ``i + 1`` is the client at
  DataFrame row ``i`` (matching the convention in
  :func:`route_optimizer.config.load_clients`).
* **Objective** — minimise total travel **distance** (arc cost = distance in
  metres, kept integer for the CP solver).
* **Capacity dimension** — each client demands
  ``"Volumen estimado en litros"`` litres; every vehicle has ``capacity``.
* **Time dimension** — arc transit = travel time + the origin's service time;
  each client carries a hard ``[earliest, latest]`` window (minutes from
  ``DEPART``, via :func:`route_optimizer.config.parse_window`). Waiting is
  allowed (slack), so a truck may arrive early and idle until the window opens.
* **Dropping** — every client node gets an optional-visit disjunction with a
  large penalty, so the model is *always* feasible and simply drops the nodes
  it cannot serve within capacity / time rather than failing.
* **Search** — first solution = ``PATH_CHEAPEST_ARC``, then
  ``GUIDED_LOCAL_SEARCH`` metaheuristic under a wall-clock time limit
  (default ~5 s).

The single public entry point is :func:`solve_cvrptw`.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

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

# Distances are minimised in integer metres; times tracked in integer minutes.
_DIST_SCALE = 1000  # km -> m
# Penalty for dropping a node, in the same (metre) units as arc cost. Must be
# large enough that serving a node is always preferred when feasible, but finite
# so the solver can still drop nodes it genuinely cannot fit.
_DROP_PENALTY_M = 5_000_000


def _build_windows(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Per-client ``(earliest, latest)`` windows in minutes from DEPART."""
    return [parse_window(w) for w in df[COL_WINDOW].tolist()]


def solve_cvrptw(
    df: pd.DataFrame,
    provider: str = PROVIDER_HAVERSINE,
    num_vehicles: int = NUM_VEHICLES,
    capacity: int = VEHICLE_CAPACITY,
    service_time: int = SERVICE_TIME_MIN,
    shift_minutes: int = SHIFT_MINUTES,
    time_limit_s: int = 5,
    distance_matrix: Optional[list[list[float]]] = None,
    time_matrix: Optional[list[list[float]]] = None,
    provider_used: Optional[str] = None,
) -> Solution:
    """Solve the CVRPTW for ``df`` and return a :class:`Solution`.

    Parameters
    ----------
    df:
        Cleaned client DataFrame (see :func:`config.load_clients`). Row ``i``
        maps to node ``i + 1``.
    provider:
        Distance provider for the matrix build (``"haversine"`` / ``"osrm"``).
        Ignored when ``distance_matrix``/``time_matrix`` are supplied directly.
    num_vehicles, capacity, service_time, shift_minutes:
        Fleet / shift parameters (default to the ``config`` values).
    time_limit_s:
        Wall-clock limit for the guided-local-search metaheuristic.
    distance_matrix, time_matrix, provider_used:
        Optional pre-built matrices (e.g. supplied by the baseline so both
        solvers share the exact same distances). When omitted they are built
        from the depot + client coordinates via :func:`matrix.build_matrices`.

    Returns
    -------
    Solution
        Always returns a solution (never raises on infeasibility): unservable
        clients are reported in :attr:`Solution.dropped`.
    """
    n_clients = len(df)

    # Degenerate: no clients -> empty (but feasible) solution.
    if n_clients == 0:
        return Solution(
            routes=[[] for _ in range(num_vehicles)],
            per_vehicle=[
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
                for _ in range(num_vehicles)
            ],
            total_distance_km=0.0,
            total_time_min=0.0,
            dropped=[],
            provider_used=provider_used or PROVIDER_HAVERSINE,
            feasible=True,
        )

    # --- Build (or reuse) the distance / time matrices --------------------
    if distance_matrix is None or time_matrix is None:
        coords = [DEPOT] + list(
            zip(df["Latitud"].tolist(), df["Longitud"].tolist())
        )
        distance_matrix, time_matrix, provider_used = build_matrices(
            coords, provider
        )
    elif provider_used is None:
        provider_used = provider

    n_nodes = n_clients + 1  # + depot

    demands = [0] + [int(round(v)) for v in df[COL_VOLUME].tolist()]
    windows = _build_windows(df)

    # --- OR-Tools model ----------------------------------------------------
    manager = pywrapcp.RoutingIndexManager(n_nodes, num_vehicles, 0)  # depot=0
    routing = pywrapcp.RoutingModel(manager)

    # Arc cost = integer metres.
    def distance_cb(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(round(distance_matrix[i][j] * _DIST_SCALE))

    dist_cb_index = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_cb_index)

    # --- Capacity dimension ------------------------------------------------
    def demand_cb(from_index: int) -> int:
        return demands[manager.IndexToNode(from_index)]

    demand_cb_index = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_index,
        0,  # null slack
        [int(capacity)] * num_vehicles,
        True,  # start cumul at zero
        "Capacity",
    )

    # --- Time dimension (travel + service, with waiting slack) -------------
    def time_cb(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        travel = time_matrix[i][j]
        serve = service_time if i != 0 else 0  # service the origin (not depot)
        return int(round(travel + serve))

    time_cb_index = routing.RegisterTransitCallback(time_cb)
    routing.AddDimension(
        time_cb_index,
        shift_minutes,   # allow waiting (slack) up to the whole shift
        shift_minutes,   # max cumulative time per vehicle = shift length
        False,           # do NOT force start cumul to zero (depot can leave >0)
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # Depot start/end: whole shift available.
    for v in range(num_vehicles):
        start = routing.Start(v)
        end = routing.End(v)
        time_dim.CumulVar(start).SetRange(0, shift_minutes)
        time_dim.CumulVar(end).SetRange(0, shift_minutes)

    # Client windows (node i+1 for DataFrame row i).
    for row in range(n_clients):
        node = row + 1
        earliest, latest = windows[row]
        earliest = max(0, int(earliest))
        latest = min(int(latest), shift_minutes)
        if latest < earliest:
            latest = shift_minutes
        index = manager.NodeToIndex(node)
        time_dim.CumulVar(index).SetRange(earliest, latest)

    # --- Allow dropping nodes (optional visits with a penalty) -------------
    for row in range(n_clients):
        node = row + 1
        routing.AddDisjunction([manager.NodeToIndex(node)], _DROP_PENALTY_M)

    # --- Search parameters -------------------------------------------------
    search = pywrapcp.DefaultRoutingSearchParameters()
    search.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search.time_limit.FromSeconds(int(max(1, time_limit_s)))

    assignment = routing.SolveWithParameters(search)

    if assignment is None:
        # Should not happen (dropping guarantees feasibility), but be safe:
        return Solution(
            routes=[[] for _ in range(num_vehicles)],
            per_vehicle=[],
            total_distance_km=0.0,
            total_time_min=0.0,
            dropped=list(range(n_clients)),
            provider_used=provider_used or provider,
            feasible=False,
        )

    return _extract_solution(
        manager,
        routing,
        assignment,
        df,
        distance_matrix,
        time_matrix,
        demands,
        windows,
        num_vehicles,
        capacity,
        service_time,
        shift_minutes,
        provider_used or provider,
    )


def _extract_solution(
    manager,
    routing,
    assignment,
    df: pd.DataFrame,
    distance_matrix: list[list[float]],
    time_matrix: list[list[float]],
    demands: list[int],
    windows: list[tuple[int, int]],
    num_vehicles: int,
    capacity: int,
    service_time: int,
    shift_minutes: int,
    provider_used: str,
) -> Solution:
    """Walk the OR-Tools assignment into our shared :class:`Solution`."""
    time_dim = routing.GetDimensionOrDie("Time")

    routes: list[list[int]] = []
    per_vehicle: list[dict] = []
    total_distance_km = 0.0
    total_time_min = 0.0
    served_nodes: set[int] = set()

    for v in range(num_vehicles):
        index = routing.Start(v)
        route_rows: list[int] = []
        load = 0.0
        route_km = 0.0
        arrivals: list[float] = []
        window_violations = 0

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node != 0:  # skip depot
                row = node - 1
                route_rows.append(row)
                served_nodes.add(node)
                load += demands[node]
                arr = assignment.Value(time_dim.CumulVar(index))
                arrivals.append(float(arr))
                earliest, latest = windows[row]
                # latest is clamped to shift in the model; count as a violation
                # if arrival exceeds the client's own latest bound.
                if arr > min(int(latest), shift_minutes):
                    window_violations += 1

            next_index = assignment.Value(routing.NextVar(index))
            from_node = manager.IndexToNode(index)
            to_node = manager.IndexToNode(next_index)
            route_km += distance_matrix[from_node][to_node]
            index = next_index

        # End node arrival = total route time.
        end_time = float(assignment.Value(time_dim.CumulVar(index)))

        cap_ok = load <= capacity + 1e-6
        time_ok = end_time <= shift_minutes + 1e-6

        routes.append(route_rows)
        per_vehicle.append(
            {
                "stops": len(route_rows),
                "load": float(load),
                "distance_km": float(route_km),
                "time_min": float(end_time),
                "cap_ok": bool(cap_ok),
                "time_ok": bool(time_ok),
                "window_violations": int(window_violations),
                "arrivals": arrivals,
            }
        )
        total_distance_km += route_km
        total_time_min += end_time

    dropped = [
        node - 1
        for node in range(1, len(df) + 1)
        if node not in served_nodes
    ]

    feasible = all(pv["cap_ok"] and pv["time_ok"] for pv in per_vehicle)

    return Solution(
        routes=routes,
        per_vehicle=per_vehicle,
        total_distance_km=float(total_distance_km),
        total_time_min=float(total_time_min),
        dropped=dropped,
        provider_used=provider_used,
        feasible=bool(feasible),
    )
