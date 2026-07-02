#!/usr/bin/env python3
"""
Análisis de Optimización de Rutas de Distribución
Proyecto de portafolio (originado en un assessment técnico de logística)

Este script realiza:
1. Carga y exploración de datos de clientes
2. Selección aleatoria de 50 clientes
3. Clustering geográfico (K-Means)
4. Optimización de rutas (Nearest Neighbor)
5. Cálculo de KPIs
6. Generación de visualizaciones y mapas
"""

import os
import sys
import importlib.util

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Repo-relative paths (works regardless of the current working directory).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'clientes.csv')
GENERATOR_PATH = os.path.join(BASE_DIR, 'data', 'generate_data.py')


def _ensure_data():
    """Ensure data/clientes.csv exists; generate it on first run if missing."""
    if os.path.exists(DATA_PATH):
        return
    spec = importlib.util.spec_from_file_location("generate_data", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_data"] = module
    spec.loader.exec_module(module)
    module.generate()

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURACIÓN Y CONSTANTES
# ============================================================================

# Centro de distribución (origen y destino)
CENTRO_DIST = {
    'lat': 19.37709580527042,
    'lon': -99.58287448741568,
    'nombre': 'Centro de Distribución'
}

# Restricciones operativas
CAPACIDAD_CAMION = 12000  # litros (capacidad por unidad de reparto a granel)
NUM_CAMIONES = 4
TIEMPO_SERVICIO = 10  # minutos por cliente (parada)
JORNADA_MAXIMA = 630  # minutos (10.5 horas: 8:00 a 18:30)
DESCANSO = 40  # minutos de descanso obligatorio
TIEMPO_OPERATIVO = JORNADA_MAXIMA - DESCANSO  # 590 minutos efectivos
VELOCIDAD_MAXIMA = 50  # km/h máximo permitido

# Horarios de jornada
HORA_INICIO = "08:00"  # Inicio de jornada
HORA_FIN_MAX = "18:30"  # Hora máxima de retorno

# Semilla para reproducibilidad
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia en kilómetros entre dos puntos geográficos
    usando la fórmula de Haversine.
    """
    R = 6371  # Radio de la Tierra en km
    
    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)
    
    a = sin(delta_phi/2)**2 + cos(phi1) * cos(phi2) * sin(delta_lambda/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def calcular_matriz_distancias(clientes_df, centro):
    """
    Calcula la matriz de distancias entre todos los clientes y el centro.
    """
    n = len(clientes_df)
    # +1 para incluir el centro de distribución
    matriz = np.zeros((n + 1, n + 1))
    
    coords = clientes_df[['Latitud', 'Longitud']].values
    
    # Agregar centro como primer punto
    all_points = np.vstack([[centro['lat'], centro['lon']], coords])
    
    for i in range(len(all_points)):
        for j in range(i + 1, len(all_points)):
            dist = haversine(
                all_points[i][0], all_points[i][1],
                all_points[j][0], all_points[j][1]
            )
            matriz[i][j] = dist
            matriz[j][i] = dist
    
    return matriz


def nearest_neighbor(matriz_dist, indices_clientes):
    """
    Implementa el algoritmo Nearest Neighbor para TSP.
    Retorna el orden óptimo de visitas comenzando y terminando en el centro (índice 0).
    """
    if len(indices_clientes) == 0:
        return [], 0
    
    # Convertir a índices de la matriz (+1 porque el centro es el índice 0)
    nodos = [i + 1 for i in indices_clientes]
    
    ruta = [0]  # Empezar en el centro
    disponibles = set(nodos)
    distancia_total = 0
    
    while disponibles:
        actual = ruta[-1]
        mejor_dist = float('inf')
        mejor_nodo = None
        
        for nodo in disponibles:
            if matriz_dist[actual][nodo] < mejor_dist:
                mejor_dist = matriz_dist[actual][nodo]
                mejor_nodo = nodo
        
        ruta.append(mejor_nodo)
        distancia_total += mejor_dist
        disponibles.remove(mejor_nodo)
    
    # Retornar al centro
    distancia_total += matriz_dist[ruta[-1]][0]
    ruta.append(0)
    
    return ruta, distancia_total


def mejorar_ruta_2opt(matriz_dist, ruta):
    """
    Aplica el algoritmo 2-opt para mejorar la ruta.
    """
    mejora = True
    mejor_distancia = calcular_distancia_ruta(matriz_dist, ruta)
    
    while mejora:
        mejora = False
        for i in range(1, len(ruta) - 2):
            for j in range(i + 1, len(ruta) - 1):
                nueva_ruta = ruta[:i] + ruta[i:j+1][::-1] + ruta[j+1:]
                nueva_distancia = calcular_distancia_ruta(matriz_dist, nueva_ruta)
                
                if nueva_distancia < mejor_distancia:
                    ruta = nueva_ruta
                    mejor_distancia = nueva_distancia
                    mejora = True
                    break
            if mejora:
                break
    
    return ruta, mejor_distancia


def calcular_distancia_ruta(matriz_dist, ruta):
    """Calcula la distancia total de una ruta."""
    return sum(matriz_dist[ruta[i]][ruta[i+1]] for i in range(len(ruta) - 1))


# ============================================================================
# CARGA DE DATOS
# ============================================================================

print("=" * 60)
print("ANÁLISIS DE OPTIMIZACIÓN DE RUTAS DE DISTRIBUCIÓN")
print("=" * 60)
print()

# Asegurar que exista el dataset sintético y cargarlo (UTF-8)
_ensure_data()
os.makedirs('resultados/graficos', exist_ok=True)
df = pd.read_csv(DATA_PATH, encoding='utf-8')

# Convertir columnas de coordenadas a numérico (por si vienen como string)
df['Latitud'] = pd.to_numeric(df['Latitud'], errors='coerce')
df['Longitud'] = pd.to_numeric(df['Longitud'], errors='coerce')

# Corregir signos de coordenadas - Toluca está en latitud positiva (~19) y longitud negativa (~-99)
# Algunos datos tienen signos incorrectos
df['Latitud'] = df['Latitud'].abs()  # Latitud debe ser positiva para México
df['Longitud'] = -df['Longitud'].abs()  # Longitud debe ser negativa para México (oeste)

print("📊 EXPLORACIÓN DE DATOS")
print("-" * 40)
print(f"Total de clientes: {len(df)}")
print(f"\nColumnas disponibles:")
for col in df.columns:
    print(f"  • {col}")

print(f"\nEstadísticas del volumen (litros):")
print(f"  - Mínimo: {df['Volumen estimado en litros'].min()}")
print(f"  - Máximo: {df['Volumen estimado en litros'].max()}")
print(f"  - Promedio: {df['Volumen estimado en litros'].mean():.1f}")
print(f"  - Total: {df['Volumen estimado en litros'].sum():,}")

print(f"\nRango de coordenadas (corregidas):")
print(f"  - Latitud: {df['Latitud'].min():.4f} a {df['Latitud'].max():.4f}")
print(f"  - Longitud: {df['Longitud'].min():.4f} a {df['Longitud'].max():.4f}")

# ============================================================================
# SELECCIÓN DE 50 CLIENTES ALEATORIOS
# ============================================================================

print("\n" + "=" * 60)
print("CASO 1: OPTIMIZACIÓN DE RUTAS")
print("=" * 60)

# Seleccionar 50 clientes aleatorios (usando sample de pandas)
# La semilla garantiza reproducibilidad
clientes_seleccionados = df.sample(n=50, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"\n🎲 SELECCIÓN ALEATORIA (seed={RANDOM_SEED})")
print(f"   Clientes seleccionados: {len(clientes_seleccionados)}")
print(f"   IDs: {sorted(clientes_seleccionados['NombreCliente'].tolist())[:10]}... (primeros 10)")
print(f"   Volumen total a entregar: {clientes_seleccionados['Volumen estimado en litros'].sum():,} litros")

# ============================================================================
# CLUSTERING GEOGRÁFICO
# ============================================================================

print("\n🗺️  AGRUPACIÓN GEOGRÁFICA (K-Means)")
print("-" * 40)

# Preparar coordenadas para clustering
coords = clientes_seleccionados[['Latitud', 'Longitud']].values

# Aplicar K-Means con 4 clusters
kmeans = KMeans(n_clusters=NUM_CAMIONES, random_state=RANDOM_SEED, n_init=10)
clientes_seleccionados['Cluster'] = kmeans.fit_predict(coords)

# Verificar distribución de clusters
print("\nDistribución inicial por cluster:")
for c in range(NUM_CAMIONES):
    mask = clientes_seleccionados['Cluster'] == c
    n_clientes = mask.sum()
    volumen = clientes_seleccionados.loc[mask, 'Volumen estimado en litros'].sum()
    print(f"  Ruta {c+1}: {n_clientes} clientes, {volumen:,} litros")

# ============================================================================
# BALANCEO DE CARGA (si excede capacidad)
# ============================================================================

def balancear_clusters(df_clientes, capacidad_max):
    """
    Rebalancea clusters si alguno excede la capacidad máxima.
    Mueve clientes del cluster más cargado al más cercano geográficamente
    que tenga espacio disponible.
    """
    df = df_clientes.copy()
    iteraciones = 0
    max_iter = 100
    
    while iteraciones < max_iter:
        # Calcular volumen por cluster
        volumenes = df.groupby('Cluster')['Volumen estimado en litros'].sum()
        
        # Verificar si hay exceso
        cluster_excedido = volumenes[volumenes > capacidad_max]
        if len(cluster_excedido) == 0:
            break
        
        # Tomar el cluster con más exceso
        cluster_origen = cluster_excedido.idxmax()
        
        # Encontrar clusters con espacio disponible
        espacio_disponible = capacidad_max - volumenes
        clusters_destino = espacio_disponible[espacio_disponible > 0].index.tolist()
        
        if not clusters_destino:
            print("⚠️  No hay espacio disponible para rebalancear")
            break
        
        # Clientes del cluster excedido
        clientes_origen = df[df['Cluster'] == cluster_origen]
        
        # Encontrar el cliente más cercano a otro cluster
        centros_destino = df[df['Cluster'].isin(clusters_destino)].groupby('Cluster')[['Latitud', 'Longitud']].mean()
        
        mejor_cliente = None
        mejor_distancia = float('inf')
        mejor_destino = None
        
        for idx, cliente in clientes_origen.iterrows():
            vol_cliente = cliente['Volumen estimado en litros']
            for dest in clusters_destino:
                if espacio_disponible[dest] >= vol_cliente:
                    dist = haversine(
                        cliente['Latitud'], cliente['Longitud'],
                        centros_destino.loc[dest, 'Latitud'],
                        centros_destino.loc[dest, 'Longitud']
                    )
                    if dist < mejor_distancia:
                        mejor_distancia = dist
                        mejor_cliente = idx
                        mejor_destino = dest
        
        if mejor_cliente is not None:
            df.loc[mejor_cliente, 'Cluster'] = mejor_destino
        else:
            break
        
        iteraciones += 1
    
    return df

# Aplicar balanceo
clientes_seleccionados = balancear_clusters(clientes_seleccionados, CAPACIDAD_CAMION)

print("\n✅ Distribución después del balanceo:")
for c in range(NUM_CAMIONES):
    mask = clientes_seleccionados['Cluster'] == c
    n_clientes = mask.sum()
    volumen = clientes_seleccionados.loc[mask, 'Volumen estimado en litros'].sum()
    status = "✓" if volumen <= CAPACIDAD_CAMION else "⚠️ EXCEDE"
    print(f"  Ruta {c+1}: {n_clientes} clientes, {volumen:,} litros {status}")

# ============================================================================
# OPTIMIZACIÓN DE RUTAS
# ============================================================================

print("\n🚚 OPTIMIZACIÓN DE SECUENCIAS (Nearest Neighbor + 2-opt)")
print("-" * 40)

# Calcular matriz de distancias
matriz_dist = calcular_matriz_distancias(clientes_seleccionados, CENTRO_DIST)

rutas_optimizadas = {}

for c in range(NUM_CAMIONES):
    mask = clientes_seleccionados['Cluster'] == c
    indices = clientes_seleccionados[mask].index.tolist()
    
    # Aplicar Nearest Neighbor
    ruta_inicial, dist_inicial = nearest_neighbor(matriz_dist, indices)
    
    # Mejorar con 2-opt
    ruta_mejorada, dist_mejorada = mejorar_ruta_2opt(matriz_dist, ruta_inicial)
    
    # Convertir índices de matriz a índices de DataFrame
    clientes_ruta = [i - 1 for i in ruta_mejorada if i != 0]
    
    rutas_optimizadas[c] = {
        'indices': clientes_ruta,
        'distancia_km': dist_mejorada,
        'secuencia_matriz': ruta_mejorada
    }
    
    mejora = ((dist_inicial - dist_mejorada) / dist_inicial) * 100 if dist_inicial > 0 else 0
    print(f"  Ruta {c+1}: {dist_mejorada:.2f} km (mejora 2-opt: {mejora:.1f}%)")

# ============================================================================
# CÁLCULO DE KPIs
# ============================================================================

print("\n" + "=" * 60)
print("CASO 2: ANÁLISIS DE INDICADORES (KPIs)")
print("=" * 60)

resultados_rutas = []

for c in range(NUM_CAMIONES):
    indices = rutas_optimizadas[c]['indices']
    clientes_ruta = clientes_seleccionados.iloc[indices]
    
    # Métricas
    num_clientes = len(clientes_ruta)
    volumen_total = clientes_ruta['Volumen estimado en litros'].sum()
    distancia_km = rutas_optimizadas[c]['distancia_km']
    
    # Truck Fill
    truck_fill = (volumen_total / CAPACIDAD_CAMION) * 100
    
    # Tiempo de servicio (10 min por cliente)
    tiempo_servicio = num_clientes * TIEMPO_SERVICIO
    
    # Tiempo de traslado (distancia / velocidad * 60 para convertir a minutos)
    tiempo_traslado = (distancia_km / VELOCIDAD_MAXIMA) * 60
    
    # Tiempo total
    tiempo_total = tiempo_servicio + tiempo_traslado + DESCANSO
    
    # Validación de restricciones
    cumple_capacidad = volumen_total <= CAPACIDAD_CAMION
    cumple_tiempo = tiempo_total <= JORNADA_MAXIMA
    
    resultados_rutas.append({
        'Ruta': c + 1,
        'Clientes': num_clientes,
        'Volumen (L)': volumen_total,
        'Truck Fill (%)': truck_fill,
        'Distancia (km)': distancia_km,
        'Tiempo Servicio (min)': tiempo_servicio,
        'Tiempo Traslado (min)': tiempo_traslado,
        'Tiempo Total (min)': tiempo_total,
        'Tiempo Total (horas)': tiempo_total / 60,
        'Cumple Capacidad': cumple_capacidad,
        'Cumple Tiempo': cumple_tiempo
    })

# Crear DataFrame de resultados
df_kpis = pd.DataFrame(resultados_rutas)

print("\n📊 RESUMEN DE KPIs POR RUTA")
print("-" * 40)
print(df_kpis[['Ruta', 'Clientes', 'Volumen (L)', 'Truck Fill (%)', 
               'Distancia (km)', 'Tiempo Total (min)']].to_string(index=False))

# Totales y promedios
print("\n📈 MÉTRICAS GLOBALES")
print("-" * 40)
print(f"  Total de clientes atendidos: {df_kpis['Clientes'].sum()}")
print(f"  Volumen total entregado: {df_kpis['Volumen (L)'].sum():,} litros")
print(f"  Kilómetros totales: {df_kpis['Distancia (km)'].sum():.2f} km")
print(f"  Truck Fill promedio: {df_kpis['Truck Fill (%)'].mean():.1f}%")
print(f"  Tiempo promedio por ruta: {df_kpis['Tiempo Total (min)'].mean():.1f} min ({df_kpis['Tiempo Total (min)'].mean()/60:.2f} hrs)")

# Identificar ruta más eficiente
eficiencia = df_kpis['Volumen (L)'] / df_kpis['Distancia (km)']
ruta_mas_eficiente = eficiencia.idxmax() + 1
print(f"\n🏆 Ruta más eficiente (L/km): Ruta {ruta_mas_eficiente}")

# Verificación de restricciones
print("\n✅ VALIDACIÓN DE RESTRICCIONES")
print("-" * 40)
for _, row in df_kpis.iterrows():
    cap_status = "✓" if row['Cumple Capacidad'] else "✗"
    tiempo_status = "✓" if row['Cumple Tiempo'] else "✗"
    print(f"  Ruta {int(row['Ruta'])}: Capacidad {cap_status} | Tiempo {tiempo_status}")

# ============================================================================
# EXPORTAR RESULTADOS
# ============================================================================

print("\n💾 EXPORTANDO RESULTADOS...")
print("-" * 40)

# Guardar KPIs
df_kpis.to_csv('resultados/kpis_resumen.csv', index=False, encoding='utf-8')
print("  ✓ resultados/kpis_resumen.csv")

# Guardar rutas detalladas
rutas_detalle = []
for c in range(NUM_CAMIONES):
    indices = rutas_optimizadas[c]['indices']
    orden = 1
    for idx in indices:
        cliente = clientes_seleccionados.iloc[idx]
        rutas_detalle.append({
            'Ruta': c + 1,
            'Orden_Visita': orden,
            'Cliente': cliente['NombreCliente'],
            'Direccion': cliente['Direccion'],
            'Latitud': cliente['Latitud'],
            'Longitud': cliente['Longitud'],
            'Volumen (L)': cliente['Volumen estimado en litros'],
            'Ventana_Servicio': cliente['VentanaServicio']
        })
        orden += 1

df_rutas = pd.DataFrame(rutas_detalle)
df_rutas.to_csv('resultados/rutas_optimizadas.csv', index=False, encoding='utf-8')
print("  ✓ resultados/rutas_optimizadas.csv")

# ============================================================================
# VISUALIZACIONES
# ============================================================================

print("\n📊 GENERANDO VISUALIZACIONES...")
print("-" * 40)

# Configurar colores para las rutas
colores = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

# 1. Gráfico de Truck Fill por Ruta
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(df_kpis['Ruta'], df_kpis['Truck Fill (%)'], color=colores, edgecolor='white', linewidth=2)
ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Capacidad máxima')
ax.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Objetivo 80%')
ax.set_xlabel('Ruta', fontsize=12, fontweight='bold')
ax.set_ylabel('Truck Fill (%)', fontsize=12, fontweight='bold')
ax.set_title('Utilización de Capacidad por Ruta', fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, 110)
ax.legend()

# Añadir valores sobre las barras
for bar, val in zip(bars, df_kpis['Truck Fill (%)']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('resultados/graficos/truck_fill_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ graficos/truck_fill_comparison.png")

# 2. Gráfico de Kilómetros por Ruta
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(df_kpis['Ruta'], df_kpis['Distancia (km)'], color=colores, edgecolor='white', linewidth=2)
ax.set_xlabel('Ruta', fontsize=12, fontweight='bold')
ax.set_ylabel('Distancia (km)', fontsize=12, fontweight='bold')
ax.set_title('Kilómetros Recorridos por Ruta', fontsize=14, fontweight='bold', pad=15)

for bar, val in zip(bars, df_kpis['Distancia (km)']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('resultados/graficos/km_por_ruta.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ graficos/km_por_ruta.png")

# 3. Gráfico de Tiempo por Ruta (stacked bar)
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.6
x = df_kpis['Ruta']

# Barras apiladas
p1 = ax.bar(x, df_kpis['Tiempo Servicio (min)'], width, label='Servicio', color='#3498db')
p2 = ax.bar(x, df_kpis['Tiempo Traslado (min)'], width, bottom=df_kpis['Tiempo Servicio (min)'], 
            label='Traslado', color='#2ecc71')
p3 = ax.bar(x, [DESCANSO]*4, width, 
            bottom=df_kpis['Tiempo Servicio (min)'] + df_kpis['Tiempo Traslado (min)'],
            label='Descanso', color='#f39c12')

ax.axhline(y=JORNADA_MAXIMA, color='red', linestyle='--', alpha=0.7, label='Jornada máxima (630 min)')
ax.set_xlabel('Ruta', fontsize=12, fontweight='bold')
ax.set_ylabel('Tiempo (minutos)', fontsize=12, fontweight='bold')
ax.set_title('Distribución del Tiempo por Ruta', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper right')

# Añadir totales
for i, total in enumerate(df_kpis['Tiempo Total (min)']):
    ax.text(i + 1, total + 10, f'{total:.0f} min', ha='center', va='bottom', 
            fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('resultados/graficos/tiempo_por_ruta.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ graficos/tiempo_por_ruta.png")

# 4. Mapa de distribución de clientes
fig, ax = plt.subplots(figsize=(12, 10))

for c in range(NUM_CAMIONES):
    mask = clientes_seleccionados['Cluster'] == c
    clientes_cluster = clientes_seleccionados[mask]
    ax.scatter(clientes_cluster['Longitud'], clientes_cluster['Latitud'], 
               c=colores[c], s=100, alpha=0.7, label=f'Ruta {c+1}', edgecolors='white', linewidth=1)

# Centro de distribución
ax.scatter(CENTRO_DIST['lon'], CENTRO_DIST['lat'], 
           c='black', s=300, marker='*', label='Centro de Distribución', zorder=5)

ax.set_xlabel('Longitud', fontsize=12, fontweight='bold')
ax.set_ylabel('Latitud', fontsize=12, fontweight='bold')
ax.set_title('Distribución Geográfica de Clientes por Ruta', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('resultados/graficos/distribucion_clientes.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ graficos/distribucion_clientes.png")

# 5. Dashboard resumen
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Pie chart de volumen por ruta
ax1 = axes[0, 0]
wedges, texts, autotexts = ax1.pie(df_kpis['Volumen (L)'], labels=[f'Ruta {i}' for i in df_kpis['Ruta']], 
                                    colors=colores, autopct='%1.1f%%', startangle=90,
                                    textprops={'fontweight': 'bold'})
ax1.set_title('Distribución del Volumen por Ruta', fontsize=12, fontweight='bold')

# Subplot 2: Barras de eficiencia (L/km)
ax2 = axes[0, 1]
eficiencia = df_kpis['Volumen (L)'] / df_kpis['Distancia (km)']
bars = ax2.bar(df_kpis['Ruta'], eficiencia, color=colores, edgecolor='white', linewidth=2)
ax2.set_xlabel('Ruta', fontsize=10, fontweight='bold')
ax2.set_ylabel('Eficiencia (L/km)', fontsize=10, fontweight='bold')
ax2.set_title('Eficiencia por Ruta', fontsize=12, fontweight='bold')
for bar, val in zip(bars, eficiencia):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Subplot 3: Comparación clientes por ruta
ax3 = axes[1, 0]
bars = ax3.bar(df_kpis['Ruta'], df_kpis['Clientes'], color=colores, edgecolor='white', linewidth=2)
ax3.set_xlabel('Ruta', fontsize=10, fontweight='bold')
ax3.set_ylabel('Número de Clientes', fontsize=10, fontweight='bold')
ax3.set_title('Clientes por Ruta', fontsize=12, fontweight='bold')
for bar, val in zip(bars, df_kpis['Clientes']):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
             f'{int(val)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Subplot 4: Tabla resumen
ax4 = axes[1, 1]
ax4.axis('off')
tabla_data = [
    ['Métrica', 'Valor'],
    ['Total Clientes', f"{df_kpis['Clientes'].sum()}"],
    ['Volumen Total', f"{df_kpis['Volumen (L)'].sum():,} L"],
    ['Km Totales', f"{df_kpis['Distancia (km)'].sum():.2f} km"],
    ['Truck Fill Prom.', f"{df_kpis['Truck Fill (%)'].mean():.1f}%"],
    ['Tiempo Prom.', f"{df_kpis['Tiempo Total (min)'].mean():.0f} min"],
    ['Ruta más Eficiente', f"Ruta {ruta_mas_eficiente}"]
]
tabla = ax4.table(cellText=tabla_data, loc='center', cellLoc='center',
                  colWidths=[0.4, 0.4])
tabla.auto_set_font_size(False)
tabla.set_fontsize(12)
tabla.scale(1.2, 1.8)
# Estilo de encabezado
for key, cell in tabla.get_celld().items():
    if key[0] == 0:
        cell.set_text_props(fontweight='bold')
        cell.set_facecolor('#3498db')
        cell.set_text_props(color='white', fontweight='bold')
    else:
        cell.set_facecolor('#f8f9fa')

ax4.set_title('Resumen Global', fontsize=12, fontweight='bold', y=0.95)

plt.suptitle('Dashboard de Optimización de Rutas de Distribución', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('resultados/graficos/dashboard_resumen.png', dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ graficos/dashboard_resumen.png")

# ============================================================================
# MAPA INTERACTIVO (Folium)
# ============================================================================

try:
    import folium
    from folium.plugins import MarkerCluster
    
    print("\n🗺️  GENERANDO MAPA INTERACTIVO...")
    print("-" * 40)
    
    # Crear mapa centrado en Toluca
    mapa = folium.Map(
        location=[CENTRO_DIST['lat'], CENTRO_DIST['lon']],
        zoom_start=12,
        tiles='cartodbpositron'
    )
    
    # Marcador del centro de distribución
    folium.Marker(
        [CENTRO_DIST['lat'], CENTRO_DIST['lon']],
        popup='<b>Centro de Distribución</b>',
        icon=folium.Icon(color='black', icon='warehouse', prefix='fa'),
        tooltip='Centro de Distribución'
    ).add_to(mapa)
    
    # Colores para folium
    colores_folium = ['red', 'blue', 'green', 'orange']
    
    # Añadir marcadores y líneas de ruta
    for c in range(NUM_CAMIONES):
        indices = rutas_optimizadas[c]['indices']
        
        # Crear grupo de capa para esta ruta
        grupo_ruta = folium.FeatureGroup(name=f'Ruta {c+1}')
        
        # Coordenadas de la ruta (incluyendo centro al inicio y fin)
        coords_ruta = [[CENTRO_DIST['lat'], CENTRO_DIST['lon']]]
        
        # Calcular hora de llegada para cada parada
        from datetime import datetime, timedelta
        hora_actual = datetime.strptime(HORA_INICIO, "%H:%M")
        lat_anterior, lon_anterior = CENTRO_DIST['lat'], CENTRO_DIST['lon']
        
        orden = 1
        for idx in indices:
            cliente = clientes_seleccionados.iloc[idx]
            lat, lon = cliente['Latitud'], cliente['Longitud']
            coords_ruta.append([lat, lon])
            
            # Calcular distancia y tiempo hasta este cliente
            dist_cliente = haversine(lat_anterior, lon_anterior, lat, lon)
            tiempo_viaje = (dist_cliente / VELOCIDAD_MAXIMA) * 60  # minutos
            hora_actual += timedelta(minutes=tiempo_viaje)
            
            hora_llegada_str = hora_actual.strftime("%H:%M")
            
            # Popup con información completa del cliente
            popup_html = f"""
            <div style="width:280px; font-family: Arial, sans-serif;">
                <div style="background: {colores[c]}; color: white; padding: 8px; margin: -10px -10px 10px -10px; border-radius: 5px 5px 0 0;">
                    <b>🚚 Ruta {c+1} - Parada {orden}</b>
                </div>
                <table style="width:100%; font-size: 12px;">
                    <tr><td><b>📍 Cliente:</b></td><td>{cliente['NombreCliente']}</td></tr>
                    <tr><td><b>📫 Dirección:</b></td><td>{cliente['Direccion'][:40]}...</td></tr>
                    <tr><td><b>🌐 Coordenadas:</b></td><td>{lat:.6f}, {lon:.6f}</td></tr>
                    <tr><td><b>📦 Volumen:</b></td><td>{cliente['Volumen estimado en litros']} L</td></tr>
                    <tr><td><b>⏰ Ventana:</b></td><td>{cliente['VentanaServicio']}</td></tr>
                    <tr><td><b>🕐 Llegada Est.:</b></td><td><b style="color: {colores[c]}">{hora_llegada_str}</b></td></tr>
                </table>
            </div>
            """
            
            folium.CircleMarker(
                [lat, lon],
                radius=8,
                popup=folium.Popup(popup_html, max_width=300),
                color=colores_folium[c],
                fill=True,
                fill_color=colores_folium[c],
                fill_opacity=0.7,
                tooltip=f"🚚 Ruta {c+1} | Cliente {cliente['NombreCliente']} | {hora_llegada_str}"
            ).add_to(grupo_ruta)
            
            # Número de orden
            folium.Marker(
                [lat, lon],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:10px;font-weight:bold;color:white;background:{colores[c]};border-radius:50%;width:20px;height:20px;text-align:center;line-height:20px;">{orden}</div>'
                )
            ).add_to(grupo_ruta)
            
            # Actualizar para siguiente iteración
            lat_anterior, lon_anterior = lat, lon
            hora_actual += timedelta(minutes=TIEMPO_SERVICIO)  # Agregar tiempo de servicio
            orden += 1
        
        # Cerrar la ruta volviendo al centro
        coords_ruta.append([CENTRO_DIST['lat'], CENTRO_DIST['lon']])
        
        # Dibujar línea de ruta
        folium.PolyLine(
            coords_ruta,
            color=colores[c],
            weight=3,
            opacity=0.8,
            tooltip=f'Ruta {c+1} - {rutas_optimizadas[c]["distancia_km"]:.2f} km'
        ).add_to(grupo_ruta)
        
        grupo_ruta.add_to(mapa)
    
    # Añadir control de capas
    folium.LayerControl().add_to(mapa)
    
    # Añadir leyenda
    leyenda_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
                padding: 10px; border: 2px solid grey; border-radius: 5px; font-family: Arial;">
        <b>Leyenda</b><br>
        <i style="background:#e74c3c;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Ruta 1<br>
        <i style="background:#3498db;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Ruta 2<br>
        <i style="background:#2ecc71;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Ruta 3<br>
        <i style="background:#f39c12;width:12px;height:12px;display:inline-block;border-radius:50%;"></i> Ruta 4<br>
        <i style="color:black;font-size:16px;">★</i> Centro de Distribución
    </div>
    '''
    mapa.get_root().html.add_child(folium.Element(leyenda_html))
    
    # Guardar mapa
    mapa.save('resultados/mapa_rutas.html')
    print("  ✓ resultados/mapa_rutas.html")
    
except ImportError:
    print("  ⚠️  Folium no instalado. Instala con: pip install folium")
    print("     El mapa interactivo no se generó.")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "=" * 60)
print("✅ ANÁLISIS COMPLETADO")
print("=" * 60)

print("""
📁 ARCHIVOS GENERADOS:
   • resultados/kpis_resumen.csv
   • resultados/rutas_optimizadas.csv
   • resultados/mapa_rutas.html
   • resultados/graficos/truck_fill_comparison.png
   • resultados/graficos/km_por_ruta.png
   • resultados/graficos/tiempo_por_ruta.png
   • resultados/graficos/distribucion_clientes.png
   • resultados/graficos/dashboard_resumen.png

📊 MÉTRICAS CLAVE:
""")
print(f"   • Total Clientes: {df_kpis['Clientes'].sum()}")
print(f"   • Volumen Total: {df_kpis['Volumen (L)'].sum():,} L")
print(f"   • Km Totales: {df_kpis['Distancia (km)'].sum():.2f} km")
print(f"   • Truck Fill Promedio: {df_kpis['Truck Fill (%)'].mean():.1f}%")
print(f"   • Tiempo Promedio: {df_kpis['Tiempo Total (min)'].mean():.1f} min")
_todas_ok = bool(df_kpis['Cumple Capacidad'].all() and df_kpis['Cumple Tiempo'].all())
print(f"   • Todas las rutas cumplen restricciones: {'✓' if _todas_ok else '✗'}")
print()
