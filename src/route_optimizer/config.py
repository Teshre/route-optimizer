"""
Shared configuration for Route Optimizer V2.

This module holds the single source of truth for every default parameter and
the two small helpers that the rest of the package relies on:

* :func:`parse_window` — turn a ``"HH:MM-HH:MM"`` service window into a
  ``(earliest_min, latest_min)`` pair measured in minutes from :data:`DEPART`.
* :func:`load_clients` — read ``data/clientes.csv`` with the exact expected
  columns and coerce the coordinates into a canonical sign convention.

No solver / matrix logic lives here; those belong to the other modules
(``matrix.py``, ``solver.py``, ``baseline.py``, ``kpis.py``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Depot (distribution centre near Toluca, MX)
# ---------------------------------------------------------------------------
# coords[0] in every distance/time matrix is ALWAYS this depot.
DEPOT_LAT: float = 19.37709580527042
DEPOT_LON: float = -99.58287448741568
# Convenience tuple in (lat, lon) order — matches the coords[] convention used
# by matrix.build_matrices (coords[0] is the depot).
DEPOT: tuple[float, float] = (DEPOT_LAT, DEPOT_LON)

# ---------------------------------------------------------------------------
# Default fleet / shift parameters (overridable via the Streamlit UI or CLI)
# ---------------------------------------------------------------------------
NUM_VEHICLES: int = 4
VEHICLE_CAPACITY: int = 12000          # litres per truck
SERVICE_TIME_MIN: int = 10             # minutes spent servicing each client
SHIFT_MINUTES: int = 630               # 08:00 -> 18:30 working window
SPEED_KMH: float = 50.0                # only used to estimate Haversine time
DEPART: str = "08:00"                  # shift start; time windows are measured from here

# ---------------------------------------------------------------------------
# Distance provider names (canonical strings used across the package)
# ---------------------------------------------------------------------------
PROVIDER_HAVERSINE: str = "haversine"
PROVIDER_OSRM: str = "osrm"

# ---------------------------------------------------------------------------
# Data / column contract
# ---------------------------------------------------------------------------
# Repo root = three parents up from this file: src/route_optimizer/config.py
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH: Path = PROJECT_ROOT / "data" / "clientes.csv"

# Exact CSV columns (do not reorder / rename — the generator writes these).
COL_NAME: str = "NombreCliente"
COL_ADDRESS: str = "Direccion"
COL_LAT: str = "Latitud"
COL_LON: str = "Longitud"
COL_VOLUME: str = "Volumen estimado en litros"
COL_WINDOW: str = "VentanaServicio"

EXPECTED_COLUMNS: list[str] = [
    COL_NAME,
    COL_ADDRESS,
    COL_LAT,
    COL_LON,
    COL_VOLUME,
    COL_WINDOW,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _to_minutes(hhmm: str) -> int:
    """Parse a ``"HH:MM"`` clock string into minutes since midnight."""
    hh, mm = hhmm.strip().split(":")
    return int(hh) * 60 + int(mm)


# Minutes-since-midnight for the shift start; time windows are relative to this.
DEPART_MIN: int = _to_minutes(DEPART)


def parse_window(s: str) -> tuple[int, int]:
    """Convert a service window string into ``(earliest_min, latest_min)``.

    The returned pair is measured in **minutes from** :data:`DEPART` (the shift
    start), so a client whose window is ``"08:00-12:00"`` with ``DEPART="08:00"``
    yields ``(0, 240)``.

    The parser is deliberately permissive:

    * accepts ``"HH:MM-HH:MM"`` (the canonical format) as well as ``"HH.MM"``
      or plain ``"HH"`` hour components and en-dash / whitespace separators;
    * clamps negative offsets (windows that open before the depot departs) to
      ``0`` for the ``earliest`` bound;
    * on any parse failure, falls back to the full shift ``(0, SHIFT_MINUTES)``
      so the solver still receives a usable (unconstrained) window.

    Parameters
    ----------
    s:
        A window string such as ``"09:00-13:00"``.

    Returns
    -------
    tuple[int, int]
        ``(earliest_min, latest_min)`` measured from :data:`DEPART`.
    """
    default = (0, SHIFT_MINUTES)
    if s is None:
        return default
    text = str(s).strip()
    if not text:
        return default

    # Normalise common separators to a plain hyphen.
    for sep in ("–", "—", " to ", "~"):
        text = text.replace(sep, "-")
    text = text.replace(" ", "")

    if "-" not in text:
        return default

    start_raw, _, end_raw = text.partition("-")

    def _one(part: str) -> int | None:
        part = part.strip().replace(".", ":")
        if not part:
            return None
        try:
            if ":" in part:
                hh, mm = part.split(":", 1)
                return int(hh) * 60 + int(mm or 0)
            return int(part) * 60
        except (ValueError, TypeError):
            return None

    start_abs = _one(start_raw)
    end_abs = _one(end_raw)
    if start_abs is None or end_abs is None:
        return default

    earliest = start_abs - DEPART_MIN
    latest = end_abs - DEPART_MIN

    # Clamp / sanity-fix so downstream code always sees 0 <= earliest <= latest.
    earliest = max(0, earliest)
    if latest < earliest:
        latest = SHIFT_MINUTES
    return (earliest, latest)


def load_clients(csv_path: str | Path = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """Load the synthetic client CSV into a cleaned DataFrame.

    The CSV is expected to have exactly :data:`EXPECTED_COLUMNS`. Coordinates
    are coerced to numeric and normalised to the canonical sign convention for
    the Toluca region (northern hemisphere, western hemisphere):

    * ``Latitud``  -> ``abs(lat)``   (always positive, ~19 N)
    * ``Longitud`` -> ``-abs(lon)``  (always negative, ~ -99 W)

    Rows with unparseable coordinates are dropped and the index is reset so
    that row positions line up with matrix indices used elsewhere (client at
    DataFrame row ``i`` maps to matrix node ``i + 1`` because node ``0`` is the
    depot).

    Parameters
    ----------
    csv_path:
        Path to ``clientes.csv``. Defaults to :data:`DEFAULT_CSV_PATH`.

    Returns
    -------
    pandas.DataFrame
        Cleaned clients with a fresh ``RangeIndex``.
    """
    path = Path(csv_path)
    df = pd.read_csv(path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {missing}. "
            f"Expected exactly: {EXPECTED_COLUMNS}"
        )

    # Coerce coordinates to numeric; unparseable values become NaN.
    df[COL_LAT] = pd.to_numeric(df[COL_LAT], errors="coerce")
    df[COL_LON] = pd.to_numeric(df[COL_LON], errors="coerce")

    # Sign-fix for the Toluca region: positive latitude, negative longitude.
    df[COL_LAT] = df[COL_LAT].abs()
    df[COL_LON] = -df[COL_LON].abs()

    # Coerce demand to numeric as well (defensive against stray strings).
    df[COL_VOLUME] = pd.to_numeric(df[COL_VOLUME], errors="coerce")

    # Drop rows we cannot place or serve, then reset the index so positions are
    # contiguous (matrix nodes depend on this).
    df = df.dropna(subset=[COL_LAT, COL_LON, COL_VOLUME]).reset_index(drop=True)

    return df
