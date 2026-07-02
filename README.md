# 🚚 Route Optimizer V2 — Capacitated VRP with Time Windows

Plan last-mile delivery routes for a distribution operation: decide **which clients go on which truck, in what order, and at what time** — while respecting each truck's **volume capacity**, each client's **service window**, and the **shift length**. V2 replaces the original clustering heuristic with a real combinatorial optimizer (**Google OR-Tools**) solving the **Capacitated Vehicle Routing Problem with Time Windows (CVRPTW)**, uses **real road distances** (OSRM) with a Haversine fallback, and quantifies the improvement against a naive baseline.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://route-optimizer-demo.streamlit.app/) [![CI](https://github.com/Teshre/route-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/Teshre/route-optimizer/actions/workflows/ci.yml)

**▶ Live demo: [route-optimizer-demo.streamlit.app](https://route-optimizer-demo.streamlit.app/)**

---

## The problem: CVRPTW

Given a single **depot** (distribution center near Toluca, MX), a fleet of identical **trucks**, and a set of **clients** — each with a location, a delivery **volume** (litres), and a **service window** (`08:00-12:00`, …) — find a set of routes (one per truck, each starting and ending at the depot) that serves every client while:

- **Capacity constraint** — the total volume loaded on any truck never exceeds its capacity.
- **Time-window constraint** — each client is served **within** its `[earliest, latest]` window. A truck that arrives early **waits**; arriving after `latest` is not allowed.
- **Shift constraint** — every route fits inside the working day (depart `08:00`, shift length configurable, default 10.5 h → 18:30).
- **Objective** — **minimize total distance driven** across the whole fleet.

This is the classic CVRPTW. It is NP-hard, so we do not brute-force it — we model it exactly and let a mature solver search the space under a time budget.

## The model (Google OR-Tools)

The solver (`src/route_optimizer/solver.py`) uses OR-Tools' `RoutingIndexManager` / `RoutingModel`:

- **Nodes** — node `0` is the depot; client at DataFrame row `i` is node `i + 1`.
- **Arc cost** — the driving distance between nodes (from the distance matrix, see below). Minimizing the summed arc cost *is* the objective.
- **Capacity dimension** — each client node carries a demand equal to its `Volumen estimado en litros`; every vehicle has the same capacity ceiling. OR-Tools enforces that the cumulative load along a route never exceeds capacity.
- **Time dimension** — accumulates **travel time + service time** along each route. Each client node gets its window `[earliest_min, latest_min]` (measured from the depart time) as a hard constraint on the time dimension; slack lets a truck **wait** when it arrives before the window opens. The shift length bounds the dimension so no route runs past the end of the day.
- **Drop penalty** — every client node is given a large "disjunction" penalty so it *can* be dropped if serving it is infeasible (too little capacity / too tight a window). This guarantees the solver **always returns a solution** instead of failing; any dropped clients are reported explicitly (and shown in grey on the map) so the trade-off is visible rather than hidden.
- **Search strategy** — a cheap initial solution (`PATH_CHEAPEST_ARC`) is then improved with **Guided Local Search** metaheuristic under a **~5 s time limit**. GLS escapes the local optima that a plain nearest-neighbor / 2-opt pass gets stuck in.

## Distances: OSRM road distances vs Haversine

The distance/time matrix (`src/route_optimizer/matrix.py`) has two providers, selectable in the UI:

| Provider | What it measures | Cost | When it wins |
| --- | --- | --- | --- |
| **Haversine** | Great-circle (straight-line) km; time = km / speed | Instant, offline, deterministic | Fast iteration, no network, reproducible CI |
| **OSRM** | **Real driving** distance & duration from the public OSRM `/table` service | Network round-trip, rate-limited | Realistic routes that respect the actual road network |

Straight-line distance systematically **under**-estimates real travel (it ignores one-way streets, rivers, highways, detours), which can make a plan look better than it is and can mis-order stops. OSRM fixes that by querying a routing engine for the true driving matrix. Because a public endpoint can be slow, rate-limited, or offline, the OSRM path is wrapped so that **any failure falls back to Haversine** automatically, the matrix is **cached** by rounded-coordinate key, and the dashboard surfaces a small note when a fallback happens. This keeps the app deploy-safe on Streamlit Community Cloud while still showcasing road-accurate routing when the network cooperates.

## Baseline vs optimized benchmark

To show the optimizer earns its keep, V2 also runs the **old heuristic** as a baseline (`src/route_optimizer/baseline.py`): **K-Means** clusters clients into one group per truck, then a **nearest-neighbor** pass orders each cluster — **ignoring time windows entirely**. Both approaches return the same `Solution` shape, so `kpis.benchmark()` can compare them head-to-head:

- **total km** — baseline vs optimized, and the **% distance saved**;
- **time-window satisfaction** — the baseline racks up window **violations** (it never looked at windows); the CVRPTW solver drives them to **zero** by construction;
- per-truck **fill %**, **km**, **time**, and **L/km** efficiency.

> **Benchmark (default params — 50 clients, 4 trucks × 12,000 L, 10.5 h shift, Haversine):**
> _Distance saved:_ `__%` · _Baseline window violations:_ `__` → _Optimized:_ `0` · _Baseline km:_ `__` → _Optimized km:_ `__`.
> _(Numbers are reproducible via `make cli`; fill in from your run.)_

## Architecture

```
src/route_optimizer/
  config.py     constants + helpers: parse_window(), load_clients()
  matrix.py     build_matrices(coords, provider) -> (dist_km, time_min); Haversine | OSRM + cache
  solver.py     solve_cvrptw(...) -> Solution   (OR-Tools CVRPTW: capacity + time-window dimensions)
  baseline.py   solve_baseline(...) -> Solution (K-Means + nearest-neighbor, windows ignored)
  kpis.py       compute_kpis(solution, df) -> dict; benchmark(baseline, optimized) -> dict
  cli.py        `python -m route_optimizer.cli` — runs both, prints the benchmark, writes out/ CSVs
app.py          Streamlit dashboard (baseline-vs-optimized, Folium map, Plotly charts, tables)
tests/          pytest suite
data/           generate_data.py + clientes.csv (synthetic)
```

The shared **`Solution`** dataclass: `routes` (client row-indices per vehicle, depot excluded), `per_vehicle` (stops / load / distance_km / time_min / cap_ok / time_ok / window_violations / arrivals), `total_distance_km`, `total_time_min`, `dropped`, `provider_used`, `feasible`.

## Tech stack

Python · **OR-Tools** (CVRPTW) · **OSRM** (road distances) · pandas · NumPy · scikit-learn (K-Means baseline) · Folium · Streamlit · Plotly · Faker · **pytest**

---

## Quickstart

```bash
make setup    # create a virtualenv and install dependencies
make data     # generate the synthetic dataset -> data/clientes.csv
make app      # launch the interactive Streamlit dashboard
```

Run the command-line benchmark instead (prints the baseline-vs-optimized comparison + KPIs and writes CSVs to `out/`):

```bash
make cli
```

Run the tests, or list every target:

```bash
make test
make help
```

> The dashboard auto-generates `data/clientes.csv` on first run if it is missing, so it also works on a fresh Streamlit Community Cloud deploy with nothing uploaded.

> **Deploy note:** `runtime.txt` pins **Python 3.12** so Streamlit Community Cloud provisions an interpreter with prebuilt OR-Tools wheels (OR-Tools ships manylinux wheels for CPython 3.11–3.13). The code itself runs on any Python `>=3.11`.

---

## 🌐 Live demo

**[route-optimizer-demo.streamlit.app](https://route-optimizer-demo.streamlit.app/)** — the hosted dashboard. Pick a client set and a distance provider, press **Optimize routes**, and see the baseline-vs-optimized benchmark, the optimized routes on an interactive map (with per-stop arrival times), the per-truck charts, and the route tables. It generates the synthetic sample on load, so nothing needs to be uploaded.

---

## 📊 About the data

The data in this repository is **100% synthetic**. Client names, addresses, and coordinates are randomly generated (via Faker) around a distribution hub near Toluca, Mexico — they do **not** correspond to any real business, person, or location. Nothing real or private is exposed. Dataset schema (`data/clientes.csv`):

| Column | Description |
| --- | --- |
| `NombreCliente` | Fake client / business name |
| `Direccion` | Fake street address |
| `Latitud` | Latitude (~19.2–19.5 °N) |
| `Longitud` | Longitude (~-99.4 to -99.8 °W) |
| `Volumen estimado en litros` | Estimated delivery volume in litres (50–1500) |
| `VentanaServicio` | Service window, e.g. `08:00-12:00` |

Regenerate it anytime with `make data`.

---

## About this project

This is a personal **portfolio project**. It originated from a **logistics technical assessment**, and this published version has been fully **de-branded** and rebuilt on **synthetic data** so it can be shared openly. V2 upgrades the original K-Means + nearest-neighbor + 2-opt heuristic into a proper **CVRPTW** solved with **Google OR-Tools**, adds **OSRM road distances**, a **baseline-vs-optimized benchmark**, a packaged `route_optimizer` library with a **pytest** suite, and an upgraded Streamlit dashboard.

**Author:** Eduardo Perry Rangel — [github.com/Teshre](https://github.com/Teshre)
