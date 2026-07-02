"""
Command-line entry point for Route Optimizer V2.

``python -m route_optimizer.cli`` (or the ``route-optimizer`` console script)
loads ``data/clientes.csv``, builds a **single shared** distance/time matrix,
runs the naive **baseline** and the OR-Tools **CVRPTW** on it, prints a
side-by-side benchmark plus per-route KPIs, and writes result CSVs to ``out/``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .baseline import solve_baseline
from .config import (
    DEFAULT_CSV_PATH,
    DEPOT,
    NUM_VEHICLES,
    PROVIDER_HAVERSINE,
    PROVIDER_OSRM,
    PROJECT_ROOT,
    SERVICE_TIME_MIN,
    SHIFT_MINUTES,
    VEHICLE_CAPACITY,
    load_clients,
)
from .kpis import benchmark, compute_kpis
from .matrix import build_matrices
from .solver import solve_cvrptw
from .solution import Solution


def _print_kpi_table(title: str, sol: Solution, capacity: int) -> None:
    kpis = compute_kpis(sol, capacity=capacity)
    g = kpis["global"]
    print(f"\n=== {title} ===")
    print(
        f"  provider={g['provider_used']}  feasible={g['feasible']}  "
        f"served={g['clients_served']}  dropped={g['clients_dropped']}"
    )
    print(
        f"  total_km={g['total_distance_km']:.1f}  "
        f"total_time_min={g['total_time_min']:.0f}  "
        f"fill={g['global_fill_pct']:.1f}%  "
        f"L/km={g['global_l_per_km']:.1f}"
    )
    print(
        f"  cap_violations={g['capacity_violations']}  "
        f"time_violations={g['time_violations']}  "
        f"window_violations={g['window_violations']}"
    )
    print("  per-route:")
    for r in kpis["per_route"]:
        if r["stops"] == 0:
            continue
        print(
            f"    veh {r['vehicle']}: stops={r['stops']:>3}  "
            f"load={r['load']:>7.0f}  fill={r['fill_pct']:>5.1f}%  "
            f"km={r['distance_km']:>6.1f}  time={r['time_min']:>5.0f}m  "
            f"cap_ok={r['cap_ok']}  time_ok={r['time_ok']}  "
            f"win_viol={r['window_violations']}"
        )


def _write_routes_csv(path: Path, sol: Solution, df, capacity: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "vehicle",
                "seq",
                "row_index",
                "client",
                "volume_l",
                "window",
                "arrival_min_from_depart",
            ]
        )
        for v, route in enumerate(sol.routes):
            arrivals = sol.per_vehicle[v].get("arrivals", []) if v < len(
                sol.per_vehicle
            ) else []
            for seq, row in enumerate(route):
                arr = arrivals[seq] if seq < len(arrivals) else ""
                w.writerow(
                    [
                        v,
                        seq,
                        row,
                        df.iloc[row]["NombreCliente"],
                        df.iloc[row]["Volumen estimado en litros"],
                        df.iloc[row]["VentanaServicio"],
                        f"{arr:.0f}" if arr != "" else "",
                    ]
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Route Optimizer V2 — baseline vs OR-Tools CVRPTW benchmark."
    )
    parser.add_argument("--csv", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--provider", default=PROVIDER_HAVERSINE,
                        choices=[PROVIDER_HAVERSINE, PROVIDER_OSRM])
    parser.add_argument("--vehicles", type=int, default=NUM_VEHICLES)
    parser.add_argument("--capacity", type=int, default=VEHICLE_CAPACITY)
    parser.add_argument("--service-time", type=int, default=SERVICE_TIME_MIN)
    parser.add_argument("--shift-minutes", type=int, default=SHIFT_MINUTES)
    parser.add_argument("--time-limit", type=int, default=5)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "out"))
    args = parser.parse_args(argv)

    df = load_clients(args.csv)
    print(f"Loaded {len(df)} clients from {args.csv}")

    # Build ONE matrix and share it, so baseline and optimizer are compared on
    # identical distances/times (fair benchmark, single OSRM round-trip).
    coords = [DEPOT] + list(zip(df["Latitud"].tolist(), df["Longitud"].tolist()))
    dist_km, time_min, provider_used = build_matrices(coords, args.provider)
    print(f"Matrix built with provider_used={provider_used} "
          f"({len(coords)} nodes)")

    baseline_sol = solve_baseline(
        df,
        num_vehicles=args.vehicles,
        capacity=args.capacity,
        service_time=args.service_time,
        shift_minutes=args.shift_minutes,
        distance_matrix=dist_km,
        time_matrix=time_min,
        provider_used=provider_used,
    )

    optimized_sol = solve_cvrptw(
        df,
        num_vehicles=args.vehicles,
        capacity=args.capacity,
        service_time=args.service_time,
        shift_minutes=args.shift_minutes,
        time_limit_s=args.time_limit,
        distance_matrix=dist_km,
        time_matrix=time_min,
        provider_used=provider_used,
    )

    _print_kpi_table("BASELINE (K-Means + nearest-neighbour)", baseline_sol,
                     args.capacity)
    _print_kpi_table("OPTIMIZED (OR-Tools CVRPTW)", optimized_sol, args.capacity)

    bench = benchmark(baseline_sol, optimized_sol)
    print("\n=== BENCHMARK: baseline vs optimized ===")
    print(f"  baseline_km            : {bench['baseline_km']:.1f}")
    print(f"  optimized_km           : {bench['optimized_km']:.1f}")
    print(f"  km_saved               : {bench['km_saved']:.1f}")
    print(f"  km_saved_pct           : {bench['km_saved_pct']:.1f}%")
    print(f"  baseline_window_viol   : {bench['baseline_window_violations']}")
    print(f"  optimized_window_viol  : {bench['optimized_window_violations']}")
    print(f"  window_viol_reduced    : {bench['window_violations_reduced']}")
    print(f"  baseline_served        : {bench['baseline_clients_served']}")
    print(f"  optimized_served       : {bench['optimized_clients_served']}")
    print(f"  optimized_dropped      : {bench['optimized_dropped']}")
    print(f"  baseline_feasible      : {bench['baseline_feasible']}")
    print(f"  optimized_feasible     : {bench['optimized_feasible']}")

    out_dir = Path(args.out)
    _write_routes_csv(out_dir / "baseline_routes.csv", baseline_sol, df,
                      args.capacity)
    _write_routes_csv(out_dir / "optimized_routes.csv", optimized_sol, df,
                      args.capacity)
    print(f"\nWrote route CSVs to {out_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
