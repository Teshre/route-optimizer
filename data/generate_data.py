#!/usr/bin/env python3
"""
Synthetic data generator for the Route Optimizer portfolio project.

Produces a fully synthetic client dataset for a last-mile distribution
operation around Toluca, Mexico. No real personal data is used.

Usage:
    python3 data/generate_data.py

Writes ``clientes.csv`` next to this script (i.e. ``data/clientes.csv``
relative to the repository root) with these EXACT columns:

    NombreCliente, Direccion, Latitud, Longitud,
    Volumen estimado en litros, VentanaServicio
"""

import csv
import os
import random

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
NUM_ROWS = 200

# Distribution center (matches CENTRO_DIST used by the app / analysis script).
CENTRO_LAT = 19.377
CENTRO_LON = -99.583

# Coordinate bounds around Toluca, MX.
LAT_MIN, LAT_MAX = 19.20, 19.50          # positive latitudes (~19 N)
LON_MIN, LON_MAX = -99.80, -99.40        # negative longitudes (~ -99 W)

VOLUMEN_MIN, VOLUMEN_MAX = 50, 1500      # integer liters

VENTANAS = [
    "08:00-12:00",
    "09:00-13:00",
    "10:00-14:00",
    "12:00-16:00",
    "14:00-18:00",
]

COLUMNS = [
    "NombreCliente",
    "Direccion",
    "Latitud",
    "Longitud",
    "Volumen estimado en litros",
    "VentanaServicio",
]

# Output path: data/clientes.csv (next to this script).
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clientes.csv")


# ---------------------------------------------------------------------------
# Faker setup (try es_MX locale, fall back to default, then to stdlib)
# ---------------------------------------------------------------------------
def _build_fake():
    """Return a callable-based fake generator: (name_fn, address_fn)."""
    try:
        from faker import Faker  # type: ignore

        try:
            fake = Faker("es_MX")
        except Exception:
            fake = Faker()
        Faker.seed(SEED)

        def name_fn():
            return fake.company()

        def address_fn():
            # Single-line address (commas stripped so the CSV stays clean).
            return fake.address().replace("\n", ", ").replace('"', "'")

        return name_fn, address_fn
    except Exception:
        # Pure-stdlib fallback if Faker is unavailable.
        prefijos = [
            "Abarrotes", "Comercial", "Distribuidora", "Super", "Tienda",
            "Minisuper", "Mercado", "Autoservicio", "Bodega", "Central",
        ]
        sufijos = [
            "del Valle", "La Merced", "San Jose", "Santa Maria", "El Sol",
            "Reforma", "Juarez", "Morelos", "Hidalgo", "Independencia",
            "Toluca", "Metepec", "Lerma", "Zinacantepec", "Ocoyoacac",
        ]
        calles = [
            "Av. Independencia", "Calle Morelos", "Av. Juarez", "Calle Hidalgo",
            "Av. Las Torres", "Blvd. Aeropuerto", "Calle 5 de Mayo",
            "Av. Tecnologico", "Paseo Tollocan", "Av. Solidaridad",
        ]
        colonias = [
            "Centro", "San Bernardino", "Universidad", "La Providencia",
            "Seminario", "Del Parque", "Reforma", "Sanchez", "Vertiz",
        ]

        def name_fn():
            return f"{random.choice(prefijos)} {random.choice(sufijos)}"

        def address_fn():
            num = random.randint(1, 999)
            cp = random.randint(50000, 52999)
            return (
                f"{random.choice(calles)} {num}, "
                f"Col. {random.choice(colonias)}, Toluca, Mex. C.P. {cp}"
            )

        return name_fn, address_fn


def generate():
    random.seed(SEED)
    name_fn, address_fn = _build_fake()

    rows = []
    for _ in range(NUM_ROWS):
        lat = round(random.uniform(LAT_MIN, LAT_MAX), 6)
        lon = round(random.uniform(LON_MIN, LON_MAX), 6)
        volumen = random.randint(VOLUMEN_MIN, VOLUMEN_MAX)
        ventana = random.choice(VENTANAS)
        rows.append(
            {
                "NombreCliente": name_fn(),
                "Direccion": address_fn(),
                "Latitud": lat,
                "Longitud": lon,
                "Volumen estimado en litros": volumen,
                "VentanaServicio": ventana,
            }
        )

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} synthetic rows to {OUTPUT_PATH}")
    print(f"Columns: {', '.join(COLUMNS)}")


if __name__ == "__main__":
    generate()
