# 🚚 Route Optimizer

Optimize last-mile delivery routes for a distribution operation: decide **which clients go on which truck, and in what order**, while respecting each truck's volume capacity and the shift's time limit. The goal is fewer kilometers driven, balanced truck loads, and delivery schedules that actually fit the working day.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://route-optimizer-demo.streamlit.app/) [![CI](https://github.com/Teshre/route-optimizer/actions/workflows/ci.yml/badge.svg)](https://github.com/Teshre/route-optimizer/actions/workflows/ci.yml)

**▶ Live demo: [route-optimizer-demo.streamlit.app](https://route-optimizer-demo.streamlit.app/)**

---

## What it does

Given a set of clients (location + delivery volume + service window) and a distribution center, the system builds and evaluates a full delivery plan:

- **Clustering into truck routes** — K-Means groups clients geographically into one route per available truck.
- **Route sequencing** — a **Nearest-Neighbor** heuristic builds each route's visit order, then **2-opt** local search improves it, both over a **Haversine** distance matrix (great-circle km between coordinates).
- **Constraint validation + load balancing** — routes are checked against truck **capacity** (litres) and the **shift-time limit**; overloaded clusters are rebalanced by moving the nearest client into a route that still has room.
- **Operations KPIs** — per route and overall: **truck fill %**, **km per route**, **time per route** (travel + service + break), and **L/km efficiency**, plus capacity/time compliance flags.
- **Interactive dashboard** — a Streamlit app with a **Folium** route map (colored routes, per-stop popups, estimated arrival times), **Plotly** charts (fill %, km, time, client distribution), route detail tables, and **CSV export** of KPIs and route plans.

## Tech stack

Python · pandas · NumPy · scikit-learn (K-Means) · Folium · Streamlit · Plotly · matplotlib · seaborn · Faker

---

## Quickstart

One command per step — create the environment, generate the dataset, launch the dashboard:

```bash
make setup    # create a virtualenv and install dependencies
make data     # generate the synthetic dataset -> data/clientes.csv
make app      # launch the interactive Streamlit dashboard
```

Prefer the CLI report (KPIs, static charts, and an interactive map saved to disk) instead of the dashboard:

```bash
make analysis # run route_optimizer.py
```

Run `make help` to see all available targets.

> The dashboard and the analysis script both auto-generate `data/clientes.csv` on first run if it's missing, so `make setup && make app` works on its own too.

---

## 🌐 Live demo

**[route-optimizer-demo.streamlit.app](https://route-optimizer-demo.streamlit.app/)** — the hosted dashboard: pick a random client set and see the optimized routes on an interactive map, the per-truck KPIs, and the charts. It generates the synthetic sample on load, so nothing needs to be uploaded.

---

## 📊 About the data

The data in this repository is **100% synthetic**. Client names, addresses, and coordinates are randomly generated (via Faker) around a distribution hub near Toluca, Mexico — they do **not** correspond to any real business, person, or location. Nothing real or private is exposed. The dataset schema (`data/clientes.csv`):

| Column                       | Description                                                        |
| ---------------------------- | ------------------------------------------------------------------ |
| `NombreCliente`              | Fake client / business name                                        |
| `Direccion`                  | Fake street address                                                |
| `Latitud`                    | Latitude (~19.2–19.5 °N)                                           |
| `Longitud`                   | Longitude (~-99.4 to -99.8 °W)                                     |
| `Volumen estimado en litros` | Estimated delivery volume in litres (50–1500)                      |
| `VentanaServicio`            | Service window, e.g. `08:00-12:00`                                 |

Regenerate it anytime with `make data`.

---

## About this project

This is a personal **portfolio project**. It originated from a logistics technical assessment, and this published version has been fully **de-branded** and rebuilt on **synthetic data** so it can be shared openly.

**Author:** Eduardo Perry Rangel — [github.com/Teshre](https://github.com/Teshre)
