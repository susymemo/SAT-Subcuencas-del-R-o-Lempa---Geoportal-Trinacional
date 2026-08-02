# -*- coding: utf-8 -*-
"""Geoportal SAT de la cuenca trinacional del río Lempa.

Aplicación Streamlit + Google Earth Engine + Folium.
No utiliza geemap, geopandas, xyzservices ni python-box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import ee
import folium
import streamlit as st
from folium.plugins import Fullscreen, MousePosition
from google.oauth2 import service_account
from streamlit_folium import st_folium


# -----------------------------------------------------------------------------
# Configuración general
# -----------------------------------------------------------------------------

APP_TITLE: Final = "SAT Río Lempa | Geoportal Hidroclimático"
HYBAS_ASSET: Final = "WWF/HydroSHEDS/v1/Basins/hybas_12"
LEMPA_REFERENCE_POINT: Final = (-89.03, 14.02)  # longitud, latitud
DEFAULT_CENTER: Final = [14.25, -89.15]
DEFAULT_ZOOM: Final = 8
EE_SCOPE: Final = "https://www.googleapis.com/auth/earthengine"

P90_REF: Final = 35.0
P95_REF: Final = 55.0
P99_REF: Final = 90.0

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Modelos simples
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AlertResult:
    title: str
    background: str
    border: str
    text: str
    action: str
    detail: str


# -----------------------------------------------------------------------------
# Estilos de la interfaz
# -----------------------------------------------------------------------------


def apply_styles() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.3rem;
                padding-bottom: 2rem;
                max-width: 1500px;
            }

            [data-testid="stSidebar"] {
                border-right: 1px solid #e2e8f0;
            }

            .sat-subtitle {
                color: #475569;
                font-size: 1.02rem;
                margin-top: -0.55rem;
                margin-bottom: 0.7rem;
            }

            .sat-note {
                padding: 0.85rem 1rem;
                border-radius: 0.65rem;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                color: #334155;
                font-size: 0.94rem;
            }

            .alert-card {
                border-radius: 0.75rem;
                padding: 1rem 1.15rem;
                margin-top: 0.8rem;
                margin-bottom: 0.6rem;
                box-shadow: 0 3px 10px rgba(15, 23, 42, 0.06);
            }

            .alert-location {
                color: #475569;
                font-size: 0.88rem;
            }

            .alert-title {
                font-size: 1.28rem;
                font-weight: 750;
                margin: 0.35rem 0;
            }

            .alert-action {
                color: #1e293b;
                font-size: 0.98rem;
            }

            .alert-detail {
                color: #64748b;
                font-size: 0.86rem;
                margin-top: 0.45rem;
            }

            div[data-testid="stMetric"] {
                border: 1px solid #e2e8f0;
                border-radius: 0.7rem;
                padding: 0.75rem 0.85rem;
                background: #ffffff;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Autenticación de Google Earth Engine
# -----------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def initialize_earth_engine() -> str:
    """Inicializa Earth Engine con una cuenta de servicio guardada en st.secrets."""

    try:
        secret_section = st.secrets["gcp_service_account"]
    except (FileNotFoundError, KeyError):
        st.error("No se encontraron las credenciales de Google Earth Engine.")
        st.info(
            "Agrega la cuenta de servicio en **Settings → Secrets** de Streamlit "
            "usando la sección `[gcp_service_account]`."
        )
        st.stop()

    key_dict = dict(secret_section)

    required_fields = {
        "type",
        "project_id",
        "private_key_id",
        "private_key",
        "client_email",
        "client_id",
        "auth_uri",
        "token_uri",
        "auth_provider_x509_cert_url",
        "client_x509_cert_url",
    }
    missing = sorted(required_fields.difference(key_dict))
    if missing:
        st.error(
            "La sección `[gcp_service_account]` está incompleta. "
            f"Faltan: {', '.join(missing)}"
        )
        st.stop()

    # Permite pegar en Streamlit una llave que contenga "\\n" literales.
    private_key = str(key_dict["private_key"])
    if "\\n" in private_key:
        key_dict["private_key"] = private_key.replace("\\n", "\n")

    try:
        credentials = service_account.Credentials.from_service_account_info(
            key_dict,
            scopes=[EE_SCOPE],
        )
        project_id = str(key_dict["project_id"])
        ee.Initialize(credentials=credentials, project=project_id)
        return project_id
    except Exception as exc:
        st.error("No fue posible iniciar Google Earth Engine.")
        st.code(str(exc), language="text")
        st.warning(
            "Verifica que el proyecto esté registrado para Earth Engine, que la API "
            "esté habilitada y que la cuenta de servicio tenga acceso al proyecto."
        )
        st.stop()


# -----------------------------------------------------------------------------
# Datos espaciales
# -----------------------------------------------------------------------------


def _lempa_collection() -> ee.FeatureCollection:
    """Devuelve las subcuencas HydroSHEDS nivel 12 asociadas al río Lempa."""

    basins = ee.FeatureCollection(HYBAS_ASSET)
    reference_point = ee.Geometry.Point(list(LEMPA_REFERENCE_POINT))
    sample = ee.Feature(basins.filterBounds(reference_point).first())
    main_basin_id = sample.get("MAIN_BAS")
    return basins.filter(ee.Filter.eq("MAIN_BAS", main_basin_id))


@st.cache_data(ttl=86_400, show_spinner=False)
def load_lempa_geojson(project_id: str) -> dict[str, Any]:
    """Descarga una versión simplificada de las subcuencas para dibujarla en Folium."""

    del project_id  # Solo se usa como parte de la llave de caché.

    collection = _lempa_collection().select(
        ["HYBAS_ID", "MAIN_BAS", "SUB_AREA", "UP_AREA"]
    )

    def simplify_feature(feature: ee.Feature) -> ee.Feature:
        feature = ee.Feature(feature)
        simplified = feature.geometry().simplify(maxError=120)
        return feature.setGeometry(simplified)

    geojson = collection.map(simplify_feature).getInfo()

    features = geojson.get("features", []) if isinstance(geojson, dict) else []
    if not features:
        raise RuntimeError("Earth Engine no devolvió subcuencas para el río Lempa.")

    return geojson


@st.cache_data(ttl=3_600, show_spinner=False)
def find_subbasin_at(
    longitude: float,
    latitude: float,
    project_id: str,
) -> dict[str, Any] | None:
    """Consulta la subcuenca que contiene el punto seleccionado."""

    del project_id  # Solo se usa como parte de la llave de caché.

    point = ee.Geometry.Point([longitude, latitude])
    result = (
        _lempa_collection()
        .filterBounds(point)
        .select(["HYBAS_ID", "MAIN_BAS", "SUB_AREA", "UP_AREA"])
        .limit(1)
        .getInfo()
    )

    features = result.get("features", []) if isinstance(result, dict) else []
    if not features:
        return None

    properties = dict(features[0].get("properties", {}))
    properties["longitude"] = longitude
    properties["latitude"] = latitude
    return properties


# -----------------------------------------------------------------------------
# Utilidades cartográficas
# -----------------------------------------------------------------------------


def _iter_coordinate_pairs(value: Any):
    """Recorre recursivamente pares [longitud, latitud] de un GeoJSON."""

    if (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        yield float(value[0]), float(value[1])
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_coordinate_pairs(item)


def geojson_bounds(geojson: dict[str, Any]) -> list[list[float]]:
    """Calcula límites Leaflet [[sur,oeste],[norte,este]] desde un GeoJSON."""

    pairs: list[tuple[float, float]] = []
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        pairs.extend(_iter_coordinate_pairs(geometry.get("coordinates", [])))

    if not pairs:
        return [[13.0, -90.5], [15.2, -87.5]]

    longitudes = [pair[0] for pair in pairs]
    latitudes = [pair[1] for pair in pairs]
    return [
        [min(latitudes), min(longitudes)],
        [max(latitudes), max(longitudes)],
    ]


def build_map(
    geojson: dict[str, Any],
    selected_id: str | None,
) -> folium.Map:
    """Construye el mapa interactivo con Folium puro."""

    map_object = folium.Map(
        location=DEFAULT_CENTER,
        zoom_start=DEFAULT_ZOOM,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Mapa base",
        control=True,
        show=True,
    ).add_to(map_object)

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr=(
            "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, "
            "and the GIS User Community"
        ),
        name="Imagen satelital",
        control=True,
        show=False,
    ).add_to(map_object)

    def style_function(feature: dict[str, Any]) -> dict[str, Any]:
        feature_id = str(feature.get("properties", {}).get("HYBAS_ID", ""))
        is_selected = selected_id is not None and feature_id == selected_id
        return {
            "color": "#b91c1c" if is_selected else "#334155",
            "weight": 4 if is_selected else 1.4,
            "fillColor": "#ef4444" if is_selected else "#38bdf8",
            "fillOpacity": 0.32 if is_selected else 0.08,
        }

    def highlight_function(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "color": "#f59e0b",
            "weight": 3,
            "fillColor": "#fde68a",
            "fillOpacity": 0.28,
        }

    folium.GeoJson(
        data=geojson,
        name="Subcuencas HydroSHEDS nivel 12",
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["HYBAS_ID", "SUB_AREA", "UP_AREA"],
            aliases=["ID de subcuenca:", "Área local (km²):", "Área aguas arriba (km²):"],
            localize=True,
            sticky=False,
            labels=True,
            style=(
                "background-color: white; color: #0f172a; "
                "font-family: Arial; font-size: 12px; padding: 8px;"
            ),
        ),
        popup=folium.GeoJsonPopup(
            fields=["HYBAS_ID", "MAIN_BAS", "SUB_AREA", "UP_AREA"],
            aliases=[
                "HYBAS_ID:",
                "Cuenca principal:",
                "Área local (km²):",
                "Área aguas arriba (km²):",
            ],
            localize=True,
            labels=True,
        ),
        zoom_on_click=False,
        smooth_factor=1.2,
    ).add_to(map_object)

    map_object.fit_bounds(geojson_bounds(geojson), padding=(12, 12))

    Fullscreen(
        position="topright",
        title="Pantalla completa",
        title_cancel="Salir de pantalla completa",
        force_separate_button=True,
    ).add_to(map_object)

    MousePosition(
        position="bottomright",
        separator=" | ",
        prefix="Coordenadas",
        num_digits=5,
    ).add_to(map_object)

    folium.LayerControl(collapsed=True, position="topright").add_to(map_object)
    return map_object


# -----------------------------------------------------------------------------
# Lógica de alertas
# -----------------------------------------------------------------------------


def evaluate_flood_alert(rainfall: float) -> AlertResult:
    if rainfall >= P99_REF:
        return AlertResult(
            title="🔴 ALERTA ROJA — EMERGENCIA POR ESCORRENTÍA",
            background="#fef2f2",
            border="#dc2626",
            text="#991b1b",
            action=(
                "Activar evacuación en zonas vulnerables, coordinar albergues y "
                "mantener vigilancia continua de cauces y pasos críticos."
            ),
            detail=f"La lluvia simulada iguala o supera el umbral P99 ({P99_REF:.0f} mm/día).",
        )
    if rainfall >= P95_REF:
        return AlertResult(
            title="🟠 ALERTA NARANJA — ESTADO CRÍTICO",
            background="#fff7ed",
            border="#ea580c",
            text="#9a3412",
            action=(
                "Restringir temporalmente actividades en riberas y llanuras de "
                "inundación; preparar evacuaciones preventivas."
            ),
            detail=f"La lluvia simulada iguala o supera el umbral P95 ({P95_REF:.0f} mm/día).",
        )
    if rainfall >= P90_REF:
        return AlertResult(
            title="🟡 ALERTA AMARILLA — FASE PREVENTIVA",
            background="#fefce8",
            border="#ca8a04",
            text="#854d0e",
            action=(
                "Activar comités de cuenca, verificar rutas de evacuación e "
                "intensificar el monitoreo de niveles en cauces."
            ),
            detail=f"La lluvia simulada iguala o supera el umbral P90 ({P90_REF:.0f} mm/día).",
        )
    return AlertResult(
        title="🟢 ALERTA VERDE — SISTEMA ESTABLE",
        background="#f0fdf4",
        border="#16a34a",
        text="#166534",
        action=(
            "Mantener monitoreo ordinario y comunicar cualquier cambio observado "
            "en niveles de ríos, drenajes o laderas."
        ),
        detail=f"La lluvia simulada permanece por debajo de P90 ({P90_REF:.0f} mm/día).",
    )


def evaluate_drought_alert(spi: float) -> AlertResult:
    if spi <= -2.0:
        return AlertResult(
            title="🔴 SEQUÍA EXTREMA — RESPUESTA PRIORITARIA",
            background="#fef2f2",
            border="#dc2626",
            text="#991b1b",
            action=(
                "Priorizar abastecimiento humano, activar planes de emergencia "
                "hídrica y evaluar pérdidas agropecuarias."
            ),
            detail=f"SPI simulado: {spi:.1f} (sequía extrema).",
        )
    if spi <= -1.5:
        return AlertResult(
            title="🟠 SEQUÍA SEVERA — ESTADO CRÍTICO",
            background="#fff7ed",
            border="#ea580c",
            text="#9a3412",
            action=(
                "Aplicar restricciones de uso no prioritario, reforzar reservorios "
                "y activar asistencia técnica agropecuaria."
            ),
            detail=f"SPI simulado: {spi:.1f} (sequía severa).",
        )
    if spi <= -1.0:
        return AlertResult(
            title="🟡 SEQUÍA MODERADA — FASE PREVENTIVA",
            background="#fefce8",
            border="#ca8a04",
            text="#854d0e",
            action=(
                "Promover conservación de humedad, revisar disponibilidad de agua "
                "y preparar medidas de apoyo a cultivos sensibles."
            ),
            detail=f"SPI simulado: {spi:.1f} (sequía moderada).",
        )
    return AlertResult(
        title="🟢 CONDICIÓN HÍDRICA SIN ALERTA DE SEQUÍA",
        background="#f0fdf4",
        border="#16a34a",
        text="#166534",
        action=(
            "Continuar el seguimiento mensual y mantener medidas ordinarias de "
            "uso eficiente y conservación del agua."
        ),
        detail=f"SPI simulado: {spi:.1f}.",
    )


def render_alert_card(
    result: AlertResult,
    selected: dict[str, Any] | None,
    scenario_text: str,
) -> None:
    selected_id = (
        str(selected.get("HYBAS_ID"))
        if selected and selected.get("HYBAS_ID") is not None
        else "Sin selección"
    )

    st.markdown(
        f"""
        <div class="alert-card"
             style="background:{result.background}; border:1px solid {result.border};
                    border-left:8px solid {result.border};">
            <div class="alert-location">
                📍 <b>SUBCUENCA EVALUADA:</b> {selected_id}
                &nbsp;|&nbsp; <b>ESCENARIO:</b> {scenario_text}
            </div>
            <div class="alert-title" style="color:{result.text};">
                {result.title}
            </div>
            <div class="alert-action">
                📋 <b style="color:{result.text};">Directriz operativa:</b>
                {result.action}
            </div>
            <div class="alert-detail">{result.detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Aplicación
# -----------------------------------------------------------------------------


def main() -> None:
    apply_styles()

    project_id = initialize_earth_engine()

    try:
        with st.spinner("Cargando límites de subcuencas desde Earth Engine..."):
            geojson = load_lempa_geojson(project_id)
    except Exception as exc:
        st.error("No fue posible cargar las subcuencas HydroSHEDS.")
        st.code(str(exc), language="text")
        st.stop()

    if "selected_subbasin" not in st.session_state:
        st.session_state.selected_subbasin = None

    feature_count = len(geojson.get("features", []))

    with st.sidebar:
        st.title("🎛️ Panel operativo SAT")
        st.caption(f"Earth Engine conectado: `{project_id}`")
        st.divider()

        module = st.radio(
            "Módulo de evaluación",
            options=["🌊 Inundaciones", "☀️ Sequías"],
            index=0,
        )

        if module == "🌊 Inundaciones":
            rainfall = st.slider(
                "Lluvia simulada (mm/día)",
                min_value=0.0,
                max_value=150.0,
                value=45.0,
                step=5.0,
            )
            alert = evaluate_flood_alert(rainfall)
            scenario_text = f"{rainfall:.0f} mm/día"
        else:
            spi = st.slider(
                "Índice SPI simulado",
                min_value=-3.0,
                max_value=3.0,
                value=-0.5,
                step=0.1,
                help=(
                    "Valores negativos indican déficit. Como referencia operativa: "
                    "SPI ≤ -1 sequía moderada, ≤ -1.5 severa y ≤ -2 extrema."
                ),
            )
            alert = evaluate_drought_alert(spi)
            scenario_text = f"SPI {spi:.1f}"

        st.divider()
        st.markdown("**Selección territorial**")
        st.caption("Haz clic dentro de una subcuenca en el mapa.")

        if st.button("Limpiar selección", use_container_width=True):
            st.session_state.selected_subbasin = None
            st.rerun()

        st.divider()
        st.caption(
            "Herramienta demostrativa para apoyar la planificación de la cuenca. "
            "Los umbrales deben validarse con datos y protocolos institucionales."
        )

    st.title("🚨 Sistema de Alerta Temprana | Cuenca del Río Lempa")
    st.markdown(
        "<div class='sat-subtitle'>Geoportal trinacional para la evaluación espacial "
        "de escenarios hidroclimáticos y directrices operativas.</div>",
        unsafe_allow_html=True,
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Unidades de drenaje", f"{feature_count}", "HydroSHEDS nivel 12")
    kpi2.metric("Umbral preventivo", f"{P90_REF:.0f} mm/día", "P90 de referencia")
    kpi3.metric("Umbral crítico", f"{P95_REF:.0f} mm/día", "P95 de referencia")
    kpi4.metric("Emergencia", f"{P99_REF:.0f} mm/día", "P99 de referencia")

    st.divider()
    st.subheader("📍 Selección territorial directa")
    st.markdown(
        "<div class='sat-note'>Haz clic dentro de una subcuenca. El sistema consultará "
        "Earth Engine, resaltará el polígono seleccionado y aplicará el escenario "
        "definido en el panel lateral.</div>",
        unsafe_allow_html=True,
    )

    selected = st.session_state.selected_subbasin
    selected_id = (
        str(selected.get("HYBAS_ID"))
        if selected and selected.get("HYBAS_ID") is not None
        else None
    )

    map_object = build_map(geojson, selected_id)
    map_data = st_folium(
        map_object,
        use_container_width=True,
        height=560,
        returned_objects=["last_clicked"],
        key="rio_lempa_map",
    )

    clicked = map_data.get("last_clicked") if map_data else None
    if clicked:
        latitude = round(float(clicked["lat"]), 6)
        longitude = round(float(clicked["lng"]), 6)

        with st.spinner("Identificando subcuenca seleccionada..."):
            clicked_subbasin = find_subbasin_at(
                longitude=longitude,
                latitude=latitude,
                project_id=project_id,
            )

        previous_id = (
            str(selected.get("HYBAS_ID"))
            if selected and selected.get("HYBAS_ID") is not None
            else None
        )
        clicked_id = (
            str(clicked_subbasin.get("HYBAS_ID"))
            if clicked_subbasin and clicked_subbasin.get("HYBAS_ID") is not None
            else None
        )

        if clicked_id != previous_id:
            st.session_state.selected_subbasin = clicked_subbasin
            st.rerun()

        if clicked_subbasin is None:
            st.warning(
                "El punto seleccionado está fuera de las subcuencas del área de estudio."
            )

    selected = st.session_state.selected_subbasin

    if selected is None:
        st.info("Selecciona una subcuenca para generar una evaluación territorial.")
    else:
        render_alert_card(alert, selected, scenario_text)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("HYBAS_ID", str(selected.get("HYBAS_ID", "N/D")))
        c2.metric("Cuenca principal", str(selected.get("MAIN_BAS", "N/D")))
        c3.metric(
            "Área local",
            f"{float(selected.get('SUB_AREA', 0) or 0):,.1f} km²",
        )
        c4.metric(
            "Área aguas arriba",
            f"{float(selected.get('UP_AREA', 0) or 0):,.1f} km²",
        )

        st.caption(
            "Punto consultado: "
            f"{float(selected.get('latitude', 0)):.5f}, "
            f"{float(selected.get('longitude', 0)):.5f}"
        )

    with st.expander("ℹ️ Alcance y uso responsable"):
        st.markdown(
            """
            - La delimitación procede de **HydroSHEDS nivel 12** en Google Earth Engine.
            - Los valores P90, P95 y P99 incluidos son **umbrales operativos de referencia**,
              no percentiles calculados automáticamente para cada subcuenca.
            - El módulo SPI es un simulador de clasificación; todavía no descarga ni calcula
              series de precipitación.
            - Antes de usar el resultado para decisiones oficiales, valida umbrales,
              exposición, vulnerabilidad y protocolos con las instituciones responsables.
            """
        )


if __name__ == "__main__":
    main()
