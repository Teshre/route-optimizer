"""
Dashboard Interactivo - Optimización de Rutas de Distribución
"""
import os
import sys
import importlib.util

import streamlit as st
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go

# Repo-relative paths (works regardless of the current working directory).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'clientes.csv')
GENERATOR_PATH = os.path.join(BASE_DIR, 'data', 'generate_data.py')


def _ensure_data():
    """Ensure data/clientes.csv exists; generate it on first run if missing.

    This keeps the demo working on a fresh clone / Streamlit Community Cloud
    deploy where only the committed sample may be absent.
    """
    if os.path.exists(DATA_PATH):
        return
    spec = importlib.util.spec_from_file_location("generate_data", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so any dataclasses / pickling inside resolve cleanly.
    sys.modules["generate_data"] = module
    spec.loader.exec_module(module)
    module.generate()

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
st.set_page_config(
    page_title="Route Optimizer",
    page_icon="🚚",
    layout="wide"
)

# Constantes
CENTRO_DIST = {'lat': 19.37709580527042, 'lon': -99.58287448741568}
CAPACIDAD_CAMION = 12000
NUM_CAMIONES = 4
TIEMPO_SERVICIO = 10
VELOCIDAD_MAXIMA = 50
HORA_INICIO = "08:00"
JORNADA_MAXIMA = 630  # minutos

# Colores para rutas
COLORES = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

# ============================================================================
# FUNCIONES
# ============================================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = radians(lat1), radians(lat2)
    delta_phi = radians(lat2 - lat1)
    delta_lambda = radians(lon2 - lon1)
    a = sin(delta_phi/2)**2 + cos(phi1) * cos(phi2) * sin(delta_lambda/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def nearest_neighbor(matriz_dist, indices):
    n = len(indices)
    if n == 0:
        return [], 0
    visitados = [False] * n
    ruta = [0]
    visitados_clientes = []
    
    actual = 0
    distancia_total = 0
    
    while len(visitados_clientes) < n:
        mejor_dist = float('inf')
        mejor_idx = -1
        
        for i, idx in enumerate(indices):
            if not visitados[i]:
                d = matriz_dist[actual][i + 1]
                if d < mejor_dist:
                    mejor_dist = d
                    mejor_idx = i
        
        if mejor_idx != -1:
            visitados[mejor_idx] = True
            visitados_clientes.append(mejor_idx)
            distancia_total += mejor_dist
            actual = mejor_idx + 1
    
    distancia_total += matriz_dist[actual][0]
    return visitados_clientes, distancia_total

@st.cache_data
def cargar_datos():
    _ensure_data()
    df = pd.read_csv(DATA_PATH, encoding='utf-8')
    original_count = len(df)
    
    # Convertir coordenadas
    df['Latitud'] = pd.to_numeric(df['Latitud'], errors='coerce')
    df['Longitud'] = pd.to_numeric(df['Longitud'], errors='coerce')
    
    # Contar registros con errores
    nan_count = df[['Latitud', 'Longitud']].isna().any(axis=1).sum()
    
    # Corregir signos (Toluca: lat positiva ~19, lon negativa ~-99)
    df['Latitud'] = df['Latitud'].abs()
    df['Longitud'] = -df['Longitud'].abs()
    
    # Eliminar registros con NaN
    df = df.dropna(subset=['Latitud', 'Longitud'])
    
    return df, original_count, nan_count

def optimizar_rutas(df_clientes):
    coords = df_clientes[['Latitud', 'Longitud']].values
    kmeans = KMeans(n_clusters=NUM_CAMIONES, random_state=42, n_init=10)
    df_clientes = df_clientes.copy()
    df_clientes['Cluster'] = kmeans.fit_predict(coords)
    
    rutas = {}
    for c in range(NUM_CAMIONES):
        mask = df_clientes['Cluster'] == c
        clientes_cluster = df_clientes[mask]
        indices = clientes_cluster.index.tolist()
        
        if len(indices) == 0:
            rutas[c] = {'clientes': pd.DataFrame(), 'distancia': 0, 'tiempo': 0}
            continue
        
        n = len(indices) + 1
        matriz = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == 0:
                    lat1, lon1 = CENTRO_DIST['lat'], CENTRO_DIST['lon']
                else:
                    lat1 = clientes_cluster.iloc[i-1]['Latitud']
                    lon1 = clientes_cluster.iloc[i-1]['Longitud']
                
                if j == 0:
                    lat2, lon2 = CENTRO_DIST['lat'], CENTRO_DIST['lon']
                else:
                    lat2 = clientes_cluster.iloc[j-1]['Latitud']
                    lon2 = clientes_cluster.iloc[j-1]['Longitud']
                
                matriz[i][j] = haversine(lat1, lon1, lat2, lon2)
        
        orden, distancia = nearest_neighbor(matriz, list(range(len(indices))))
        clientes_ordenados = clientes_cluster.iloc[orden].reset_index(drop=True)
        
        hora = datetime.strptime(HORA_INICIO, "%H:%M")
        lat_ant, lon_ant = CENTRO_DIST['lat'], CENTRO_DIST['lon']
        horarios = []
        
        for _, cliente in clientes_ordenados.iterrows():
            dist = haversine(lat_ant, lon_ant, cliente['Latitud'], cliente['Longitud'])
            tiempo_viaje = (dist / VELOCIDAD_MAXIMA) * 60
            hora += timedelta(minutes=tiempo_viaje)
            horarios.append(hora.strftime("%H:%M"))
            hora += timedelta(minutes=TIEMPO_SERVICIO)
            lat_ant, lon_ant = cliente['Latitud'], cliente['Longitud']
        
        clientes_ordenados['Hora Llegada'] = horarios
        clientes_ordenados['Orden'] = range(1, len(clientes_ordenados) + 1)
        
        tiempo_total = len(indices) * TIEMPO_SERVICIO + (distancia / VELOCIDAD_MAXIMA) * 60 + 40
        
        rutas[c] = {
            'clientes': clientes_ordenados,
            'distancia': distancia,
            'tiempo': tiempo_total
        }
    
    return df_clientes, rutas

def crear_mapa(df_clientes, rutas):
    m = folium.Map(location=[19.30, -99.65], zoom_start=12, tiles='CartoDB positron')
    
    folium.Marker(
        [CENTRO_DIST['lat'], CENTRO_DIST['lon']],
        popup="Centro de Distribución",
        icon=folium.Icon(color='black', icon='home')
    ).add_to(m)
    
    for c in range(NUM_CAMIONES):
        if rutas[c]['clientes'].empty:
            continue
        
        coords = [[CENTRO_DIST['lat'], CENTRO_DIST['lon']]]
        
        for _, cliente in rutas[c]['clientes'].iterrows():
            lat, lon = cliente['Latitud'], cliente['Longitud']
            coords.append([lat, lon])
            
            popup_html = f"""
            <div style="width:280px; font-family: Arial;">
                <div style="background:{COLORES[c]};color:white;padding:8px;margin:-10px -10px 10px -10px;border-radius:5px 5px 0 0;">
                    <b>🚚 Ruta {c+1} - Parada {cliente['Orden']}</b>
                </div>
                <table style="width:100%; font-size:12px;">
                    <tr><td><b>📍 Cliente:</b></td><td>{cliente['NombreCliente']}</td></tr>
                    <tr><td><b>📫 Dirección:</b></td><td>{str(cliente['Direccion'])[:40]}...</td></tr>
                    <tr><td><b>🌐 Coordenadas:</b></td><td>{lat:.6f}, {lon:.6f}</td></tr>
                    <tr><td><b>📦 Volumen:</b></td><td>{cliente['Volumen estimado en litros']} L</td></tr>
                    <tr><td><b>⏰ Ventana:</b></td><td>{cliente['VentanaServicio']}</td></tr>
                    <tr><td><b>🕐 Llegada:</b></td><td><b style="color:{COLORES[c]}">{cliente['Hora Llegada']}</b></td></tr>
                </table>
            </div>
            """
            
            folium.CircleMarker(
                [lat, lon],
                radius=8,
                popup=folium.Popup(popup_html, max_width=300),
                color=COLORES[c],
                fill=True,
                fill_opacity=0.7,
                tooltip=f"🚚 Ruta {c+1} | Cliente {cliente['NombreCliente']} | {cliente['Hora Llegada']}"
            ).add_to(m)
        
        coords.append([CENTRO_DIST['lat'], CENTRO_DIST['lon']])
        folium.PolyLine(coords, color=COLORES[c], weight=3, opacity=0.8).add_to(m)
    
    return m

# ============================================================================
# DASHBOARD
# ============================================================================
st.title("🚚 Optimización de Rutas de Distribución")

# Cargar datos
df, original_count, nan_count = cargar_datos()

# Alerta de limpieza de datos
st.warning(f"""⚠️ **Limpieza de datos realizada:**
- **Registros originales:** {original_count}
- **Problema detectado:** Algunas coordenadas tenían signos incorrectos (latitudes negativas, longitudes positivas)
- **Impacto:** Sin corrección, los puntos aparecían en el **hemisferio sur** (Sudamérica) en lugar de Toluca, México
- **Corrección aplicada:** Latitudes → positivas (~19°N), Longitudes → negativas (~99°W)
- **Registros con datos faltantes:** {nan_count}
""")

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Selección de Clientes")
    
    if st.button("🎲 Nueva Selección Aleatoria", type="primary", use_container_width=True):
        st.session_state['seed'] = np.random.randint(1, 10000)
    
    if 'seed' not in st.session_state:
        st.session_state['seed'] = 42
    
    st.caption(f"Grupo de clientes #{st.session_state['seed']}")
    
    st.markdown("---")
    st.markdown("### 📊 Parámetros Operativos")
    st.markdown(f"- 🚚 **Unidades:** {NUM_CAMIONES}")
    st.markdown(f"- 📦 **Capacidad:** {CAPACIDAD_CAMION:,} L/unidad")
    st.markdown(f"- ⚡ **Velocidad:** {VELOCIDAD_MAXIMA} km/h")
    st.markdown(f"- ⏱️ **Servicio:** {TIEMPO_SERVICIO} min/cliente")
    st.markdown(f"- 🕐 **Horario:** 8:00 - 18:30")

# Selección y optimización
clientes_seleccionados = df.sample(n=50, random_state=st.session_state['seed']).reset_index(drop=True)
df_optimizado, rutas = optimizar_rutas(clientes_seleccionados)

# Métricas principales
st.subheader("📈 Métricas Globales")
col1, col2, col3, col4, col5 = st.columns(5)

total_km = sum(r['distancia'] for r in rutas.values())
total_vol = clientes_seleccionados['Volumen estimado en litros'].sum()
tiempo_prom = sum(r['tiempo'] for r in rutas.values()) / NUM_CAMIONES
truck_fill_total = (total_vol / (CAPACIDAD_CAMION * NUM_CAMIONES)) * 100

with col1:
    st.metric("👥 Clientes", "50")
with col2:
    st.metric("📦 Volumen Total", f"{total_vol:,} L")
with col3:
    st.metric("🛣️ Km Totales", f"{total_km:.1f} km")
with col4:
    st.metric("⏱️ Tiempo Prom.", f"{tiempo_prom/60:.1f} hrs")
with col5:
    st.metric("📊 Fill Total", f"{truck_fill_total:.1f}%")

st.markdown("---")

# Métricas por Unidad
st.subheader("🚚 Indicadores por Unidad")
cols = st.columns(NUM_CAMIONES)

for c, col in enumerate(cols):
    with col:
        vol = rutas[c]['clientes']['Volumen estimado en litros'].sum() if not rutas[c]['clientes'].empty else 0
        clientes = len(rutas[c]['clientes'])
        fill = (vol / CAPACIDAD_CAMION) * 100
        eficiencia = vol / rutas[c]['distancia'] if rutas[c]['distancia'] > 0 else 0
        
        st.markdown(f"### Unidad {c+1}")
        st.metric("Clientes", clientes)
        st.metric("Volumen", f"{vol} L")
        st.metric("Truck Fill", f"{fill:.1f}%")
        st.metric("Eficiencia", f"{eficiencia:.1f} L/km")
        st.metric("Distancia", f"{rutas[c]['distancia']:.1f} km")
        st.metric("Tiempo", f"{rutas[c]['tiempo']/60:.1f} hrs")
        
        # Validación
        cumple_cap = "✅" if vol <= CAPACIDAD_CAMION else "❌"
        cumple_tiempo = "✅" if rutas[c]['tiempo'] <= JORNADA_MAXIMA else "❌"
        st.caption(f"Capacidad: {cumple_cap} | Tiempo: {cumple_tiempo}")

st.markdown("---")

# Gráficos
st.subheader("📊 Análisis Visual")
col_g1, col_g2 = st.columns(2)

with col_g1:
    # Gráfico de Truck Fill por unidad
    fig_fill = go.Figure()
    for c in range(NUM_CAMIONES):
        vol = rutas[c]['clientes']['Volumen estimado en litros'].sum() if not rutas[c]['clientes'].empty else 0
        fill = (vol / CAPACIDAD_CAMION) * 100
        fig_fill.add_trace(go.Bar(
            x=[f"Unidad {c+1}"],
            y=[fill],
            marker_color=COLORES[c],
            name=f"Unidad {c+1}"
        ))
    fig_fill.update_layout(title="Truck Fill por Unidad (%)", yaxis_title="%", showlegend=False)
    fig_fill.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Capacidad máxima")
    st.plotly_chart(fig_fill, use_container_width=True)

with col_g2:
    # Gráfico de distribución de clientes
    clientes_por_ruta = [len(rutas[c]['clientes']) for c in range(NUM_CAMIONES)]
    fig_clientes = go.Figure(data=[go.Pie(
        labels=[f"Unidad {c+1}" for c in range(NUM_CAMIONES)],
        values=clientes_por_ruta,
        marker_colors=COLORES,
        hole=0.4
    )])
    fig_clientes.update_layout(title="Distribución de Clientes por Unidad")
    st.plotly_chart(fig_clientes, use_container_width=True)

col_g3, col_g4 = st.columns(2)

with col_g3:
    # Gráfico de km por unidad
    km_por_ruta = [rutas[c]['distancia'] for c in range(NUM_CAMIONES)]
    fig_km = go.Figure()
    for c in range(NUM_CAMIONES):
        fig_km.add_trace(go.Bar(
            x=[f"Unidad {c+1}"],
            y=[km_por_ruta[c]],
            marker_color=COLORES[c],
            name=f"Unidad {c+1}"
        ))
    fig_km.update_layout(title="Kilómetros por Unidad", yaxis_title="km", showlegend=False)
    st.plotly_chart(fig_km, use_container_width=True)

with col_g4:
    # Gráfico de tiempo por unidad
    tiempo_por_ruta = [rutas[c]['tiempo']/60 for c in range(NUM_CAMIONES)]
    fig_tiempo = go.Figure()
    for c in range(NUM_CAMIONES):
        fig_tiempo.add_trace(go.Bar(
            x=[f"Unidad {c+1}"],
            y=[tiempo_por_ruta[c]],
            marker_color=COLORES[c],
            name=f"Unidad {c+1}"
        ))
    fig_tiempo.update_layout(title="Tiempo por Unidad (hrs)", yaxis_title="Horas", showlegend=False)
    fig_tiempo.add_hline(y=10.5, line_dash="dash", line_color="red", annotation_text="Jornada máxima")
    st.plotly_chart(fig_tiempo, use_container_width=True)

st.markdown("---")

# Mapa
st.subheader("🗺️ Mapa de Rutas")
mapa = crear_mapa(df_optimizado, rutas)
st_folium(mapa, width=None, height=500)

st.markdown("---")

# Detalle de rutas
st.subheader("📋 Detalle de Entregas por Unidad")

tabs = st.tabs([f"🚚 Unidad {i+1}" for i in range(NUM_CAMIONES)])

for c, tab in enumerate(tabs):
    with tab:
        if rutas[c]['clientes'].empty:
            st.warning("Sin clientes asignados")
        else:
            df_ruta = rutas[c]['clientes'][['Orden', 'NombreCliente', 'Direccion', 'Volumen estimado en litros', 'VentanaServicio', 'Hora Llegada']].copy()
            df_ruta.columns = ['#', 'Cliente', 'Dirección', 'Volumen (L)', 'Ventana Servicio', 'Hora Llegada']
            
            st.dataframe(df_ruta, hide_index=True, use_container_width=True, height=400)

st.markdown("---")

# Exportar reporte
st.subheader("📥 Exportar Reporte")
col_exp1, col_exp2 = st.columns(2)

# Generar resumen para exportar
resumen_data = []
for c in range(NUM_CAMIONES):
    vol = rutas[c]['clientes']['Volumen estimado en litros'].sum() if not rutas[c]['clientes'].empty else 0
    resumen_data.append({
        'Unidad': c + 1,
        'Clientes': len(rutas[c]['clientes']),
        'Volumen (L)': vol,
        'Truck Fill (%)': round((vol / CAPACIDAD_CAMION) * 100, 1),
        'Distancia (km)': round(rutas[c]['distancia'], 2),
        'Tiempo (hrs)': round(rutas[c]['tiempo'] / 60, 2),
        'Cumple Capacidad': 'Sí' if vol <= CAPACIDAD_CAMION else 'No',
        'Cumple Tiempo': 'Sí' if rutas[c]['tiempo'] <= JORNADA_MAXIMA else 'No'
    })

df_resumen = pd.DataFrame(resumen_data)

with col_exp1:
    csv_resumen = df_resumen.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📊 Descargar KPIs (CSV)",
        csv_resumen,
        f"kpis_rutas_{st.session_state['seed']}.csv",
        "text/csv",
        use_container_width=True
    )

with col_exp2:
    # Combinar todas las rutas
    all_rutas = []
    for c in range(NUM_CAMIONES):
        if not rutas[c]['clientes'].empty:
            df_temp = rutas[c]['clientes'][['Orden', 'NombreCliente', 'Direccion', 'Volumen estimado en litros', 'VentanaServicio', 'Hora Llegada']].copy()
            df_temp['Unidad'] = c + 1
            all_rutas.append(df_temp)
    
    if all_rutas:
        df_todas_rutas = pd.concat(all_rutas, ignore_index=True)
        csv_rutas = df_todas_rutas.to_csv(index=False).encode('utf-8')
        st.download_button(
            "🚚 Descargar Rutas (CSV)",
            csv_rutas,
            f"rutas_distribucion_{st.session_state['seed']}.csv",
            "text/csv",
            use_container_width=True
        )

st.info("💡 **Tip:** Para exportar a PDF, usa Ctrl+P (o Cmd+P en Mac) en el navegador y selecciona 'Guardar como PDF'")

st.markdown("---")
st.caption(f"🎲 Grupo #{st.session_state['seed']} | Generado: {datetime.now().strftime('%H:%M:%S')} | Clientes válidos: {len(df)}")
