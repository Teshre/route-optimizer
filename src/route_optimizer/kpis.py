"""
KPI computation and the baseline-vs-optimised **benchmark**.

Two public functions:

* :func:`compute_kpis` — turn a :class:`~route_optimizer.solution.Solution`
  into a nested dict of per-route and global fleet metrics (truck fill %,
  km, minutes, litres/km, and capacity / time / window compliance counts).
* :func:`benchmark` — compare a baseline solution against an optimised one and
  quantify the improvement (km saved, % saved, window-violation reduction,
  clients served / dropped, feasibility).
"""

from __future__ import annotations

from .config import VEHICLE_CAPACITY
from .solution import Solution


def compute_kpis(
    solution: Solution, df=None, capacity: float = VEHICLE_CAPACITY
) -> dict:
    """Compute per-route and global KPIs for ``solution``.

    Parameters
    ----------
    solution:
        The routing solution to summarise.
    df:
        Optional client DataFrame (unused for the core maths but accepted so the
        call site can pass it; kept for forward-compatibility / labelling).
    capacity:
        Vehicle capacity in litres, used for the truck-fill percentage.

    Returns
    -------
    dict
        ``{"per_route": [...], "global": {...}}``.
    """
    per_route = []
    total_km = 0.0
    total_time = 0.0
    total_load = 0.0
    total_stops = 0
    total_window_violations = 0
    cap_violations = 0
    time_violations = 0
    vehicles_used = 0

    for v, pv in enumerate(solution.per_vehicle):
        stops = pv.get("stops", 0)
        load = pv.get("load", 0.0)
        km = pv.get("distance_km", 0.0)
        tmin = pv.get("time_min", 0.0)
        cap_ok = pv.get("cap_ok", True)
        time_ok = pv.get("time_ok", True)
        wv = pv.get("window_violations", 0)

        fill_pct = (load / capacity * 100.0) if capacity else 0.0
        l_per_km = (load / km) if km > 0 else 0.0

        per_route.append(
            {
                "vehicle": v,
                "stops": stops,
                "load": float(load),
                "fill_pct": float(fill_pct),
                "distance_km": float(km),
                "time_min": float(tmin),
                "l_per_km": float(l_per_km),
                "cap_ok": bool(cap_ok),
                "time_ok": bool(time_ok),
                "window_violations": int(wv),
            }
        )

        total_km += km
        total_time += tmin
        total_load += load
        total_stops += stops
        total_window_violations += wv
        if not cap_ok:
            cap_violations += 1
        if not time_ok:
            time_violations += 1
        if stops > 0:
            vehicles_used += 1

    num_vehicles = len(solution.per_vehicle)
    total_capacity = capacity * num_vehicles if num_vehicles else 0.0
    global_fill = (total_load / total_capacity * 100.0) if total_capacity else 0.0
    global_l_per_km = (total_load / total_km) if total_km > 0 else 0.0

    global_kpis = {
        "vehicles_used": vehicles_used,
        "num_vehicles": num_vehicles,
        "clients_served": total_stops,
        "clients_dropped": len(solution.dropped),
        "total_distance_km": float(total_km),
        "total_time_min": float(total_time),
        "total_load": float(total_load),
        "global_fill_pct": float(global_fill),
        # Alias consumed by the Streamlit dashboard's "Fleet fill" metric card.
        "fleet_fill_pct": float(global_fill),
        "global_l_per_km": float(global_l_per_km),
        "capacity_violations": cap_violations,
        "time_violations": time_violations,
        "window_violations": int(total_window_violations),
        "provider_used": solution.provider_used,
        "feasible": bool(solution.feasible),
    }

    return {"per_route": per_route, "global": global_kpis}


def _window_violations(solution: Solution) -> int:
    return sum(pv.get("window_violations", 0) for pv in solution.per_vehicle)


def benchmark(baseline_sol: Solution, optimized_sol: Solution) -> dict:
    """Quantify the improvement of ``optimized_sol`` over ``baseline_sol``.

    Returns
    -------
    dict
        Keys: ``baseline_km``, ``optimized_km``, ``km_saved``,
        ``km_saved_pct``, ``baseline_window_violations``,
        ``optimized_window_violations``, ``window_violations_reduced``,
        ``baseline_time_min``, ``optimized_time_min``,
        ``baseline_clients_served``, ``optimized_clients_served``,
        ``optimized_dropped``, ``baseline_feasible``, ``optimized_feasible``.
    """
    baseline_km = float(baseline_sol.total_distance_km)
    optimized_km = float(optimized_sol.total_distance_km)
    km_saved = baseline_km - optimized_km
    km_saved_pct = (km_saved / baseline_km * 100.0) if baseline_km > 0 else 0.0

    base_wv = _window_violations(baseline_sol)
    opt_wv = _window_violations(optimized_sol)

    base_served = sum(len(r) for r in baseline_sol.routes)
    opt_served = sum(len(r) for r in optimized_sol.routes)

    return {
        "baseline_km": baseline_km,
        "optimized_km": optimized_km,
        "km_saved": float(km_saved),
        "km_saved_pct": float(km_saved_pct),
        "baseline_window_violations": int(base_wv),
        "optimized_window_violations": int(opt_wv),
        "window_violations_reduced": int(base_wv - opt_wv),
        "baseline_time_min": float(baseline_sol.total_time_min),
        "optimized_time_min": float(optimized_sol.total_time_min),
        "baseline_clients_served": int(base_served),
        "optimized_clients_served": int(opt_served),
        "optimized_dropped": len(optimized_sol.dropped),
        "baseline_feasible": bool(baseline_sol.feasible),
        "optimized_feasible": bool(optimized_sol.feasible),
    }
