"""
Route Optimizer V2 — Streamlit dashboard.

Interactive front-end for the CVRPTW route optimizer. It runs BOTH a naive
baseline (K-Means + nearest-neighbor, time windows ignored) and the OR-Tools
CVRPTW solver, then shows a baseline-vs-optimized benchmark, the optimized
routes on a Folium map, Plotly charts, and per-route tables.

The heavy lifting lives in the ``route_optimizer`` package (``src/``); this
file is presentation only. It is intentionally defensive so a fresh Streamlit
Community Cloud deploy works: it generates the dataset on first run and shows a
friendly message if an optional dependency is missing.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import streamlit as st

# ---------------------------------------------------------------------------
# Paths & package import (src-layout)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
DATA_PATH = os.path.join(BASE_DIR, "data", "clientes.csv")
GENERATOR_PATH = os.path.join(BASE_DIR, "data", "generate_data.py")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

st.set_page_config(page_title="Route Optimizer V2 — CVRPTW", page_icon="🚚", layout="wide")

# Per-vehicle colours (extended so larger fleets still get distinct colours).
COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12",
    "#9b59b6", "#1abc9c", "#e67e22", "#34495e",
    "#c0392b", "#2980b9",
]


def _color(i: int) -> str:
    return COLORS[i % len(COLORS)]


# ---------------------------------------------------------------------------
# Guarded heavy imports — a missing optional dep shows a friendly message
# instead of a raw traceback on the hosted demo.
# ---------------------------------------------------------------------------
def _friendly_import_error(exc: Exception) -> None:
    st.error(
        "⚠️ A required dependency could not be imported, so the optimizer cannot "
        "run in this environment.\n\n"
        f"**Details:** `{exc}`\n\n"
        "Install everything with `pip install -r requirements.txt` "
        "(the OR-Tools solver in particular needs the `ortools` wheel)."
    )
    st.stop()


try:
    import pandas as pd
    import numpy as np
    import folium
    from streamlit_folium import st_folium
    import plotly.graph_objects as go

    # Solver / KPI entry points are re-exported from the package root by
    # __init__.py. Depot + depart constants and provider names live in config
    # (guaranteed to exist), so import them from there directly to keep this
    # file working even if the package root re-exports change.
    from route_optimizer import (
        solve_baseline,
        solve_cvrptw,
        compute_kpis,
        benchmark,
    )
    from route_optimizer.config import (
        DEPOT_LAT,
        DEPOT_LON,
        DEPART,
        PROVIDER_HAVERSINE,
        PROVIDER_OSRM,
        load_clients,
    )
except Exception as exc:  # noqa: BLE001 — surface any import failure kindly
    _friendly_import_error(exc)


# ---------------------------------------------------------------------------
# Data bootstrap
# ---------------------------------------------------------------------------
def _ensure_data() -> None:
    """Generate ``data/clientes.csv`` on first run if it is missing.

    Imports the standalone generator by file path, registering it in
    ``sys.modules`` first so anything it references resolves cleanly.
    """
    if os.path.exists(DATA_PATH):
        return
    spec = importlib.util.spec_from_file_location("generate_data", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_data"] = module
    spec.loader.exec_module(module)
    module.generate()


@st.cache_data(show_spinner=False)
def _load_full_dataset() -> "pd.DataFrame":
    _ensure_data()
    return load_clients(DATA_PATH)


# Solving is the expensive step — cache on the exact parameter set + client set.
@st.cache_data(show_spinner=False)
def _solve_both(
    signature: tuple,
    _df: "pd.DataFrame",
    provider: str,
    num_vehicles: int,
    capacity: int,
    service_time: int,
    shift_minutes: int,
):
    """Run baseline + CVRPTW and compute KPIs. ``signature`` keys the cache;
    ``_df`` is passed with a leading underscore so Streamlit does not try to
    hash the whole DataFrame."""
    baseline_sol = solve_baseline(
        _df,
        provider=provider,
        num_vehicles=num_vehicles,
        capacity=capacity,
        service_time=service_time,
        shift_minutes=shift_minutes,
    )
    optimized_sol = solve_cvrptw(
        _df,
        provider=provider,
        num_vehicles=num_vehicles,
        capacity=capacity,
        service_time=service_time,
        shift_minutes=shift_minutes,
    )
    baseline_kpis = compute_kpis(baseline_sol, _df)
    optimized_kpis = compute_kpis(optimized_sol, _df)
    bench = benchmark(baseline_sol, optimized_sol)
    return baseline_sol, optimized_sol, baseline_kpis, optimized_kpis, bench


# ---------------------------------------------------------------------------
# Small helpers to read the Solution / KPI shapes defensively
# ---------------------------------------------------------------------------
def _fmt_hhmm(minutes_from_depart) -> str:
    """Render minutes-from-DEPART as a wall-clock ``HH:MM`` string."""
    try:
        m = int(round(float(minutes_from_depart)))
    except (TypeError, ValueError):
        return "—"
    dh, dm = int(DEPART.split(":")[0]), int(DEPART.split(":")[1])
    total = dh * 60 + dm + m
    total %= 24 * 60
    return f"{total // 60:02d}:{total % 60:02d}"


def _minutes_to_hrs(minutes) -> float:
    try:
        return float(minutes) / 60.0
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Sidebar — interactive parameters
# ---------------------------------------------------------------------------
df_full = _load_full_dataset()
n_available = len(df_full)

with st.sidebar:
    st.header("⚙️ Parameters")

    st.markdown("**Fleet**")
    num_vehicles = st.slider("🚚 Number of trucks", min_value=1, max_value=8, value=4, step=1)
    capacity = st.slider(
        "📦 Truck capacity (L)",
        min_value=4000,
        max_value=24000,
        value=12000,
        step=1000,
    )

    st.markdown("**Shift / SLA**")
    sla_hours = st.slider(
        "🕐 Shift length / SLA (hours)",
        min_value=6.0,
        max_value=14.0,
        value=10.5,
        step=0.5,
        help="Total working window from the depart time. Time windows are enforced within this shift.",
    )
    shift_minutes = int(round(sla_hours * 60))
    service_time = st.slider("⏱️ Service time per client (min)", min_value=0, max_value=30, value=10, step=1)

    st.markdown("**Distance provider**")
    provider_label = st.radio(
        "How to measure distances",
        options=["Haversine (straight-line)", "OSRM (road distances)"],
        index=0,
        help=(
            "Haversine is instant and offline (great-circle km). OSRM queries a "
            "public routing server for real driving distances/durations and "
            "falls back to Haversine automatically if it is unavailable."
        ),
    )
    provider = PROVIDER_OSRM if provider_label.startswith("OSRM") else PROVIDER_HAVERSINE

    st.markdown("**Client set**")
    max_clients = min(n_available, 120)
    n_clients = st.slider(
        "👥 Clients to route",
        min_value=5,
        max_value=max_clients,
        value=min(50, max_clients),
        step=5,
    )
    if "seed" not in st.session_state:
        st.session_state["seed"] = 42
    if st.button("🎲 New random client set", use_container_width=True):
        st.session_state["seed"] = int(np.random.randint(1, 10_000))
    st.caption(f"Client sample #{st.session_state['seed']} · {n_available} synthetic clients available")

    st.markdown("---")
    run = st.button("▶️ Optimize routes", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption(
        f"Depot near Toluca · depart {DEPART}\n\n"
        "CVRPTW solved with Google OR-Tools (guided local search). "
        "Baseline = K-Means + nearest-neighbor, time windows ignored."
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🚚 Route Optimizer V2 — CVRPTW")
st.markdown(
    "Capacitated Vehicle Routing **with Time Windows**, solved with Google OR-Tools and "
    "benchmarked against a naive K-Means baseline. Tune the parameters in the sidebar and "
    "press **Optimize routes**."
)

# Sample the client set (deterministic on the seed so reruns are stable).
clients = (
    df_full.sample(n=n_clients, random_state=st.session_state["seed"])
    .reset_index(drop=True)
)

if not run and "last_result" not in st.session_state:
    st.info(
        "👈 Set your fleet, shift/SLA, and distance provider in the sidebar, then press "
        "**Optimize routes** to run the baseline and the OR-Tools CVRPTW solver."
    )
    st.stop()

if run:
    signature = (
        st.session_state["seed"],
        n_clients,
        provider,
        num_vehicles,
        capacity,
        service_time,
        shift_minutes,
    )
    with st.spinner("Running baseline and OR-Tools CVRPTW solver…"):
        try:
            result = _solve_both(
                signature,
                clients,
                provider,
                num_vehicles,
                capacity,
                service_time,
                shift_minutes,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"The optimizer raised an error:\n\n`{exc}`")
            st.stop()
    st.session_state["last_result"] = result
    st.session_state["last_clients"] = clients

result = st.session_state["last_result"]
clients = st.session_state["last_clients"]
baseline_sol, optimized_sol, baseline_kpis, optimized_kpis, bench = result

provider_used = getattr(optimized_sol, "provider_used", provider)

# OSRM fallback note.
if provider == PROVIDER_OSRM and provider_used != PROVIDER_OSRM:
    st.warning(
        "🌐 OSRM road distances were requested but the routing server was "
        "unavailable or rate-limited, so distances fell back to **Haversine** "
        "(straight-line). Results below use the Haversine matrix."
    )
else:
    provider_pretty = "OSRM road distances" if provider_used == PROVIDER_OSRM else "Haversine (straight-line)"
    st.caption(f"Distance provider in use: **{provider_pretty}**")


# ===========================================================================
# 1) BASELINE vs OPTIMIZED benchmark
# ===========================================================================
st.subheader("📊 Baseline vs Optimized")

baseline_km = bench.get("baseline_km", getattr(baseline_sol, "total_distance_km", 0.0))
optimized_km = bench.get("optimized_km", getattr(optimized_sol, "total_distance_km", 0.0))
km_saved_pct = bench.get("km_saved_pct")
if km_saved_pct is None:
    km_saved_pct = ((baseline_km - optimized_km) / baseline_km * 100.0) if baseline_km else 0.0

base_viol = bench.get(
    "baseline_window_violations",
    baseline_kpis.get("window_violations", baseline_kpis.get("global", {}).get("window_violations", 0)),
)
opt_viol = bench.get(
    "optimized_window_violations",
    optimized_kpis.get("window_violations", optimized_kpis.get("global", {}).get("window_violations", 0)),
)

baseline_time = getattr(baseline_sol, "total_time_min", 0.0)
optimized_time = getattr(optimized_sol, "total_time_min", 0.0)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(
        "🛣️ Total distance",
        f"{optimized_km:,.1f} km",
        delta=f"-{km_saved_pct:.1f}% vs baseline",
        delta_color="inverse",
        help=f"Baseline drives {baseline_km:,.1f} km; the optimizer drives {optimized_km:,.1f} km.",
    )
with c2:
    st.metric("📉 Baseline distance", f"{baseline_km:,.1f} km", help="Naive K-Means + nearest-neighbor, time windows ignored.")
with c3:
    time_delta = optimized_time - baseline_time
    st.metric(
        "⏱️ Total route time",
        f"{_minutes_to_hrs(optimized_time):.1f} h",
        delta=f"{time_delta/60:+.1f} h vs baseline",
        delta_color="inverse",
    )
with c4:
    st.metric(
        "⏰ Window violations",
        f"{int(opt_viol)}",
        delta=f"{int(opt_viol) - int(base_viol):+d} vs baseline ({int(base_viol)})",
        delta_color="inverse",
        help="Deliveries that arrive outside the client's service window. The CVRPTW solver enforces windows as hard constraints.",
    )

dropped = getattr(optimized_sol, "dropped", []) or []
if dropped:
    st.info(
        f"ℹ️ The solver dropped **{len(dropped)}** client(s) it could not serve within capacity "
        "and time windows (they carry a penalty rather than making the whole plan infeasible). "
        "Add a truck, raise capacity, or extend the shift to serve them."
    )

st.markdown("---")


# ===========================================================================
# 2) OPTIMIZED KPIs — global + per truck
# ===========================================================================
def _global_kpis(k: dict) -> dict:
    """Return the global KPI block whether it is nested under 'global' or flat."""
    if isinstance(k, dict) and isinstance(k.get("global"), dict):
        return k["global"]
    return k


gk = _global_kpis(optimized_kpis)
routes = getattr(optimized_sol, "routes", []) or []
per_vehicle = getattr(optimized_sol, "per_vehicle", []) or []

st.subheader("📈 Optimized plan — global KPIs")
g1, g2, g3, g4, g5 = st.columns(5)
total_vol = float(clients["Volumen estimado en litros"].sum())
active_trucks = sum(1 for r in routes if r)
with g1:
    st.metric("👥 Clients served", f"{sum(len(r) for r in routes)}")
with g2:
    st.metric("🚚 Trucks used", f"{active_trucks} / {num_vehicles}")
with g3:
    st.metric("📦 Volume routed", f"{total_vol:,.0f} L")
with g4:
    fleet_fill = gk.get("fleet_fill_pct")
    if fleet_fill is None:
        fleet_fill = (total_vol / (capacity * max(active_trucks, 1))) * 100.0
    st.metric("📊 Fleet fill", f"{float(fleet_fill):.1f}%")
with g5:
    feasible = getattr(optimized_sol, "feasible", True)
    st.metric("✅ Feasible", "Yes" if feasible else "No")

st.markdown("#### 🚚 Per-truck indicators")
if active_trucks == 0:
    st.warning("No routes were produced for the current parameters.")
else:
    cols = st.columns(min(active_trucks, 4) or 1)
    shown = 0
    for v, route in enumerate(routes):
        if not route:
            continue
        pv = per_vehicle[v] if v < len(per_vehicle) else {}
        col = cols[shown % len(cols)]
        shown += 1
        with col:
            st.markdown(f"**Truck {v + 1}**")
            load = pv.get("load", float(clients.iloc[route]["Volumen estimado en litros"].sum()))
            dist_km = pv.get("distance_km", 0.0)
            time_min = pv.get("time_min", 0.0)
            fill = (load / capacity * 100.0) if capacity else 0.0
            eff = (load / dist_km) if dist_km else 0.0
            st.metric("Stops", pv.get("stops", len(route)))
            st.metric("Load", f"{load:,.0f} L")
            st.metric("Truck fill", f"{fill:.1f}%")
            st.metric("Distance", f"{dist_km:,.1f} km")
            st.metric("Time", f"{_minutes_to_hrs(time_min):.1f} h")
            st.metric("Efficiency", f"{eff:.1f} L/km")
            cap_ok = pv.get("cap_ok", load <= capacity)
            time_ok = pv.get("time_ok", time_min <= shift_minutes)
            win_v = int(pv.get("window_violations", 0))
            st.caption(
                f"Capacity {'✅' if cap_ok else '❌'} · "
                f"Time {'✅' if time_ok else '❌'} · "
                f"Windows {'✅' if win_v == 0 else f'❌ {win_v}'}"
            )

st.markdown("---")


# ===========================================================================
# 3) Charts — per-truck fill %, km, time (baseline vs optimized where useful)
# ===========================================================================
st.subheader("📊 Charts")


def _series(sol, key: str):
    vals = []
    for v, route in enumerate(getattr(sol, "routes", []) or []):
        pv_list = getattr(sol, "per_vehicle", []) or []
        pv = pv_list[v] if v < len(pv_list) else {}
        vals.append(pv.get(key, 0.0))
    return vals


labels = [f"Truck {i + 1}" for i in range(len(routes))]

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    fills = []
    for v, route in enumerate(routes):
        pv = per_vehicle[v] if v < len(per_vehicle) else {}
        load = pv.get("load", 0.0)
        fills.append((load / capacity * 100.0) if capacity else 0.0)
    fig_fill = go.Figure()
    fig_fill.add_trace(go.Bar(x=labels, y=fills, marker_color=[_color(i) for i in range(len(labels))]))
    fig_fill.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Capacity")
    fig_fill.update_layout(title="Truck fill (%)", yaxis_title="%", showlegend=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig_fill, use_container_width=True)

with chart_col2:
    kms = _series(optimized_sol, "distance_km")
    fig_km = go.Figure()
    fig_km.add_trace(go.Bar(x=labels, y=kms, marker_color=[_color(i) for i in range(len(labels))]))
    fig_km.update_layout(title="Distance per truck (km)", yaxis_title="km", showlegend=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig_km, use_container_width=True)

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    times = [_minutes_to_hrs(t) for t in _series(optimized_sol, "time_min")]
    fig_time = go.Figure()
    fig_time.add_trace(go.Bar(x=labels, y=times, marker_color=[_color(i) for i in range(len(labels))]))
    fig_time.add_hline(y=sla_hours, line_dash="dash", line_color="red", annotation_text="Shift limit")
    fig_time.update_layout(title="Time per truck (hours)", yaxis_title="Hours", showlegend=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig_time, use_container_width=True)

with chart_col4:
    # Baseline vs optimized total distance, side by side.
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(x=["Baseline", "Optimized"], y=[baseline_km, optimized_km],
                             marker_color=["#95a5a6", "#2ecc71"],
                             text=[f"{baseline_km:,.0f}", f"{optimized_km:,.0f}"], textposition="auto"))
    fig_cmp.update_layout(title=f"Total distance — {km_saved_pct:.1f}% saved", yaxis_title="km",
                          showlegend=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig_cmp, use_container_width=True)

st.markdown("---")


# ===========================================================================
# 4) Map — optimized routes
# ===========================================================================
st.subheader("🗺️ Optimized routes")


def _build_map(sol, df):
    m = folium.Map(location=[DEPOT_LAT, DEPOT_LON], zoom_start=11, tiles="CartoDB positron")
    folium.Marker(
        [DEPOT_LAT, DEPOT_LON],
        popup="Distribution center (depot)",
        tooltip="Depot",
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(m)

    routes_ = getattr(sol, "routes", []) or []
    pv_list = getattr(sol, "per_vehicle", []) or []

    for v, route in enumerate(routes_):
        if not route:
            continue
        color = _color(v)
        pv = pv_list[v] if v < len(pv_list) else {}
        arrivals = pv.get("arrivals", []) or []

        path = [[DEPOT_LAT, DEPOT_LON]]
        for order, row_idx in enumerate(route):
            client = df.iloc[row_idx]
            lat, lon = float(client["Latitud"]), float(client["Longitud"])
            path.append([lat, lon])
            arr = arrivals[order] if order < len(arrivals) else None
            arr_str = _fmt_hhmm(arr) if arr is not None else "—"
            popup_html = (
                f'<div style="width:260px;font-family:Arial;">'
                f'<div style="background:{color};color:#fff;padding:6px 8px;'
                f'border-radius:4px 4px 0 0;"><b>🚚 Truck {v + 1} · Stop {order + 1}</b></div>'
                f'<table style="width:100%;font-size:12px;margin-top:6px;">'
                f'<tr><td><b>Client</b></td><td>{client["NombreCliente"]}</td></tr>'
                f'<tr><td><b>Volume</b></td><td>{client["Volumen estimado en litros"]} L</td></tr>'
                f'<tr><td><b>Window</b></td><td>{client["VentanaServicio"]}</td></tr>'
                f'<tr><td><b>Arrival</b></td><td><b style="color:{color}">{arr_str}</b></td></tr>'
                f"</table></div>"
            )
            folium.CircleMarker(
                [lat, lon],
                radius=7,
                color=color,
                fill=True,
                fill_opacity=0.85,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"Truck {v + 1} · Stop {order + 1} · {client['NombreCliente']} · {arr_str}",
            ).add_to(m)
        path.append([DEPOT_LAT, DEPOT_LON])
        folium.PolyLine(path, color=color, weight=3, opacity=0.8, tooltip=f"Truck {v + 1}").add_to(m)

    # Mark any dropped clients in grey so they are visible on the map.
    for row_idx in getattr(sol, "dropped", []) or []:
        client = df.iloc[row_idx]
        folium.CircleMarker(
            [float(client["Latitud"]), float(client["Longitud"])],
            radius=6,
            color="#7f8c8d",
            fill=True,
            fill_opacity=0.6,
            tooltip=f"DROPPED · {client['NombreCliente']}",
        ).add_to(m)
    return m


st_folium(_build_map(optimized_sol, clients), width=None, height=520, returned_objects=[])

st.markdown("---")


# ===========================================================================
# 5) Route tables + export
# ===========================================================================
st.subheader("📋 Route detail")

active = [v for v, r in enumerate(routes) if r]
if not active:
    st.warning("No routes to display.")
else:
    tabs = st.tabs([f"🚚 Truck {v + 1}" for v in active])
    export_frames = []
    for tab, v in zip(tabs, active):
        route = routes[v]
        pv = per_vehicle[v] if v < len(per_vehicle) else {}
        arrivals = pv.get("arrivals", []) or []
        with tab:
            rows = []
            for order, row_idx in enumerate(route):
                client = clients.iloc[row_idx]
                arr = arrivals[order] if order < len(arrivals) else None
                rows.append(
                    {
                        "#": order + 1,
                        "Client": client["NombreCliente"],
                        "Address": str(client["Direccion"])[:60],
                        "Volume (L)": client["Volumen estimado en litros"],
                        "Window": client["VentanaServicio"],
                        "Arrival": _fmt_hhmm(arr) if arr is not None else "—",
                    }
                )
            table = pd.DataFrame(rows)
            st.dataframe(table, hide_index=True, use_container_width=True, height=360)
            frame = table.copy()
            frame.insert(0, "Truck", v + 1)
            export_frames.append(frame)

    if export_frames:
        all_routes = pd.concat(export_frames, ignore_index=True)
        csv_bytes = all_routes.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download optimized routes (CSV)",
            csv_bytes,
            file_name=f"optimized_routes_seed{st.session_state['seed']}.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption(
    "100% synthetic data · CVRPTW solved with Google OR-Tools · "
    "distances via Haversine or OSRM · Eduardo Perry Rangel (github.com/Teshre)"
)
