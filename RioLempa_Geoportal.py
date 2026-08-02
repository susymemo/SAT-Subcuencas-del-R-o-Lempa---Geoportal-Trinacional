# -*- coding: utf-8 -*-
import streamlit as st
import ee
import geemap.foliumap as geemap
from streamlit_folium import st_folium
import pandas as pd
from google.oauth2 import service_account

# 1. CONFIGURACIÓN VISUAL INSTITUCIONAL
st.set_page_config(page_title="SAT Río Lempa | Geoportal Hidroclimático", page_icon="🚨", layout="wide")

st.markdown('''
    <style>
        .banner-semaforo {
            padding: 18px; border-radius: 8px; margin-bottom: 20px;
            font-family: 'Sitka Text', 'Sitka', Georgia, serif;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06);
        }
    </style>
''', unsafe_allow_html=True)

# 2. AUTENTICACIÓN GEE SEGURA (Evita alertas de GitHub Secret Scanning)
@st.cache_resource
def init_gee():
    try:
        # Se leen las credenciales desde el gestor de secretos de Streamlit
        # En local, esto requiere un archivo .streamlit/secrets.toml
        key_dict = dict(st.secrets["gcp_service_account"])
        credentials = service_account.Credentials.from_service_account_info(key_dict)
        scoped_credentials = credentials.with_scopes(['https://www.googleapis.com/auth/earthengine'])
        ee.Initialize(scoped_credentials, project=key_dict.get("project_id", "gis-srl-2026"))
    except Exception as e:
        st.error(f"Error GEE: {e}. Asegúrate de configurar los secretos correctamente en .streamlit/secrets.toml.")
        st.stop()

init_gee()

# 3. CARGA DE DATOS ESPACIALES
@st.cache_data
def load_spatial_data():
    cuencas_global = ee.FeatureCollection("WWF/HydroSHEDS/v1/Basins/hybas_12")
    punto_lempa = ee.Geometry.Point([-89.03, 14.02])
    cuenca_muestra = cuencas_global.filterBounds(punto_lempa).first()
    LEMPA_MAIN_BAS = cuenca_muestra.get('MAIN_BAS')
    subcuencas_lempa = cuencas_global.filter(ee.Filter.eq('MAIN_BAS', LEMPA_MAIN_BAS))
    cuenca_estudio = subcuencas_lempa.union(100)
    return subcuencas_lempa, cuenca_estudio

subcuencas_lempa, cuenca_estudio = load_spatial_data()

# 4. BARRA LATERAL: CONTROLADOR METEOROLÓGICO
with st.sidebar:
    st.title("🎛️ Panel Operativo SAT")
    st.markdown("---")
    st.subheader("🌧️ Simulador de Tormentas")
    lluvia_evaluar = st.slider("Pronóstico de Lluvia (mm/día):", 0.0, 150.0, 45.0, 5.0)
    
    modulo_sat = st.radio("Módulo de Evaluación:", [
        "🌊 Inundaciones (Escorrentía Corto Plazo)",
        "☀️ Sequías (Índice SPI Mediano Plazo)"
    ])
    st.markdown("---")
    st.caption("Estructura operativa centrada en el **Plan de Manejo de la Cuenca** | Programa *Somos Río Lempa*")

# 5. ENCABEZADO PRINCIPAL
st.title("🚨 Sistema de Alerta Temprana (SAT) | Cuenca Río Lempa")
st.markdown("**Clasificación Espacial de Riesgos Hidroclimáticos y Directrices Operativas para el Plan de Manejo**")

# Umbrales base (pueden ser dinámicos post-selección)
P90_REF, P95_REF, P99_REF = 35.0, 55.0, 90.0

col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
col_kpi1.metric("Unidades de Drenaje", "139 Subcuencas", "HydroSHEDS Nivel 12")
col_kpi2.metric("Umbral Preventivo (P90)", f"{P90_REF} mm/día", "Monitoreo cauces")
col_kpi3.metric("Umbral Crítico (P95)", f"{P95_REF} mm/día", "Restricción agrícola")
col_kpi4.metric("Umbral Emergencia (P99)", f"{P99_REF} mm/día", "Evacuación obligatoria", delta_color="inverse")
st.markdown("---")

# 6. MAPA INTERACTIVO PARA SELECCIÓN
st.subheader("📍 Selección Territorial Directa")
st.write("Haz clic en una subcuenca del mapa para evaluar su riesgo y visualizar las directrices del plan de manejo correspondientes (e.g. Río Tamulasco u otras zonas críticas).")

m = geemap.Map()
m.centerObject(cuenca_estudio.geometry(), 9)
estilo_subcuencas = {'color': '#555555', 'fillColor': '00000000', 'width': 1.5}
m.addLayer(subcuencas_lempa.style(**estilo_subcuencas), {}, 'Subcuencas Nivel 12')

# Renderizar mapa y capturar clic
map_data = st_folium(m, width=1200, height=500, returned_objects=["last_clicked"])

subcuenca_id = "Ninguna (Selecciona en el mapa)"

if map_data and map_data.get("last_clicked"):
    lat = map_data["last_clicked"]["lat"]
    lon = map_data["last_clicked"]["lng"]
    punto_clic = ee.Geometry.Point([lon, lat])
    
    # Filtrar subcuenca cliqueada en GEE
    cuenca_seleccionada = subcuencas_lempa.filterBounds(punto_clic)
    
    if cuenca_seleccionada.size().getInfo() > 0:
        subcuenca_id = str(cuenca_seleccionada.first().get('HYBAS_ID').getInfo())
        
        # 7. LÓGICA DINÁMICA DEL SEMÁFORO Y PLAN DE MANEJO
        if modulo_sat == "🌊 Inundaciones (Escorrentía Corto Plazo)":
            if lluvia_evaluar >= P99_REF:
                alerta_nivel = "🔴 ALERTA ROJA - EMERGENCIA POR ESCORRENTÍA"
                bg_color, bord_color, txt_color = "#fef2f2", "#e74c3c", "#991b1b"
                accion_sat = "EVACUACIÓN OBLIGATORIA en zonas vulnerables detectadas en el DEM."
            elif lluvia_evaluar >= P95_REF:
                alerta_nivel = "🟠 ALERTA NARANJA - ESTADO CRÍTICO"
                bg_color, bord_color, txt_color = "#fff7ed", "#f18e21", "#9a3412"
                accion_sat = "RESTRICCIÓN TEMPORAL de actividades agrícolas en riberas y llanuras aluviales."
            elif lluvia_evaluar >= P90_REF:
                alerta_nivel = "🟡 ALERTA AMARILLA - FASE PREVENTIVA"
                bg_color, bord_color, txt_color = "#fefce8", "#f1c40f", "#854d0e"
                accion_sat = "ACTIVACIÓN DE COMITÉS DE CUENCA y monitoreo intensivo de niveles en cauces."
            else:
                alerta_nivel = "🟢 ALERTA VERDE - SISTEMA ESTABLE"
                bg_color, bord_color, txt_color = "#f0fdf4", "#089e49", "#166534"
                accion_sat = "Precipitación dentro de la capacidad hidrológica. Continuar monitoreo estándar."
        else:
            alerta_nivel = "☀️ MONITOREO DE SEQUÍA ESTACIONAL (ÍNDICE SPI)"
            bg_color, bord_color, txt_color = "#fefce8", "#f39c12", "#854d0e"
            accion_sat = "Evaluación mensual: Para SPI < -1.0 activar protocolos de conservación de humedad."

        st.markdown(f'''
            <div class='banner-semaforo' style='background-color: {bg_color}; border-left: 8px solid {bord_color}; border: 1px solid {bord_color};'>
                <div style='font-size: 14px; color: #475569;'>📍 <b>SUBCUENCA EVALUADA: {subcuenca_id}</b> &nbsp;|&nbsp; Lluvia: <b style='color: #0f172a;'>{lluvia_evaluar} mm/día</b></div>
                <div style='font-size: 22px; font-weight: bold; color: {txt_color}; margin: 8px 0;'>{alerta_nivel}</div>
                <div style='font-size: 15px; color: #1e293b;'>📋 <b style='color: {txt_color};'>Directriz del Plan de Manejo:</b> {accion_sat}</div>
            </div>
        ''', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Clic fuera del área de estudio. Selecciona un polígono dentro de la cuenca.")
