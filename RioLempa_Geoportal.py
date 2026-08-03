# -*- coding: utf-8 -*-
"""SAT Río Lempa — Geoportal hidroclimático.

Aplicación Streamlit + Google Earth Engine + Folium.

Funciones principales
---------------------
- Selección de subcuencas HydroSHEDS nivel 12 por mapa o lista.
- Precipitación diaria y mensual CHIRPS.
- Umbrales P90, P95 y P99 calculados para cada subcuenca y mes.
- Alertas por lluvia observada o por escenario manual.
- SPI de 1, 3, 6 o 12 meses calculado con distribución gamma.
- Gráficos interactivos y descargas CSV/GeoJSON/ZIP.
- Interfaz forzada a tema claro, incluso si el dispositivo usa modo oscuro.
"""

from __future__ import annotations

import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Final, Iterable

import ee
import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import Fullscreen, MousePosition
from google.oauth2 import service_account
from scipy.stats import gamma, norm
from streamlit_folium import st_folium


# =============================================================================
# Configuración
# =============================================================================

APP_TITLE: Final = "SAT - Cuenca del Río Lempa | Geoportal Hidroclimático"
HYBAS_ASSET: Final = "WWF/HydroSHEDS/v1/Basins/hybas_12"
CHIRPS_ASSET: Final = "UCSB-CHG/CHIRPS/DAILY"
CHIRPS_BAND: Final = "precipitation"
CHIRPS_SCALE_M: Final = 5_566
LEMPA_REFERENCE_POINT: Final = (-89.03, 14.02)  # longitud, latitud
DEFAULT_CENTER: Final = [14.25, -89.15]
DEFAULT_ZOOM: Final = 8
EE_SCOPE: Final = "https://www.googleapis.com/auth/earthengine"
REFERENCE_START_YEAR: Final = 1991
REFERENCE_END_YEAR: Final = 2020
WET_DAY_THRESHOLD_MM: Final = 1.0
MAX_DAILY_WINDOW_DAYS: Final = 730

MONTH_NAMES: Final = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre",
}

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Modelos
# =============================================================================


@dataclass(frozen=True)
class Thresholds:
    p90: float
    p95: float
    p99: float
    sample_all: int
    sample_wet: int
    month: int
    reference_start: int = REFERENCE_START_YEAR
    reference_end: int = REFERENCE_END_YEAR
    wet_day_mm: float = WET_DAY_THRESHOLD_MM


@dataclass(frozen=True)
class AlertResult:
    level: str
    title: str
    background: str
    border: str
    text: str
    action: str
    detail: str


# =============================================================================
# Apariencia: forzar tema claro
# =============================================================================


def apply_styles() -> None:
    st.markdown(
        """
        <style>
            :root { color-scheme: light !important; }
            html, body, [class*="css"] { color-scheme: light !important; }

            .stApp,
            [data-testid="stAppViewContainer"],
            [data-testid="stMain"],
            [data-testid="stMainBlockContainer"],
            [data-testid="stHeader"],
            [data-testid="stToolbar"],
            [data-testid="stSidebar"],
            [data-testid="stSidebarContent"] {
                background-color: #ffffff !important;
                color: #0f172a !important;
            }

            [data-testid="stSidebar"] {
                border-right: 1px solid #e2e8f0 !important;
            }

            .block-container {
                padding-top: 1.25rem;
                padding-bottom: 2.5rem;
                max-width: 1550px;
            }

            h1, h2, h3, h4, h5, h6, p, label, span,
            [data-testid="stMarkdownContainer"] {
                color: #0f172a;
            }

            input, textarea,
            [data-baseweb="select"] > div,
            [data-baseweb="input"] > div,
            [data-baseweb="popover"],
            [data-baseweb="menu"] {
                background-color: #ffffff !important;
                color: #0f172a !important;
            }

            [data-baseweb="select"] *,
            [data-baseweb="input"] *,
            [data-baseweb="menu"] * {
                color: #0f172a !important;
            }

            [data-testid="stMetric"] {
                border: 1px solid #dbe4ee;
                border-radius: 0.75rem;
                padding: 0.78rem 0.9rem;
                background: #ffffff !important;
                box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
            }

            [data-testid="stDataFrame"],
            [data-testid="stTable"] {
                background: #ffffff !important;
            }

            .sat-subtitle {
                color: #475569 !important;
                font-size: 1.02rem;
                margin-top: -0.5rem;
                margin-bottom: 0.85rem;
            }

            .sat-note {
                padding: 0.85rem 1rem;
                border-radius: 0.65rem;
                background: #f8fafc;
                border: 1px solid #dbe4ee;
                color: #334155 !important;
                font-size: 0.94rem;
            }

            .method-card {
                padding: 0.95rem 1rem;
                border-radius: 0.7rem;
                border: 1px solid #dbe4ee;
                background: #ffffff;
                color: #334155 !important;
            }

            .alert-card {
                border-radius: 0.78rem;
                padding: 1rem 1.15rem;
                margin: 0.7rem 0 0.9rem 0;
                box-shadow: 0 3px 10px rgba(15, 23, 42, 0.06);
            }

            .alert-location {
                color: #475569 !important;
                font-size: 0.88rem;
            }

            .alert-title {
                font-size: 1.24rem;
                font-weight: 750;
                margin: 0.35rem 0;
            }

            .alert-action {
                color: #1e293b !important;
                font-size: 0.97rem;
            }

            .alert-detail {
                color: #64748b !important;
                font-size: 0.86rem;
                margin-top: 0.45rem;
            }

            button[kind="primary"],
            [data-testid="stDownloadButton"] button {
                border-radius: 0.55rem !important;
            }

            /* Iframe del mapa y gráficos siempre sobre fondo blanco */
            iframe { background: #ffffff !important; }

            /* Ocultar el botón de cambio de tema si la versión lo muestra */
            [data-testid="stBaseButton-headerNoPadding"] { color: #334155 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Autenticación de Earth Engine
# =============================================================================


@st.cache_resource(show_spinner=False)
def initialize_earth_engine() -> str:
    """Inicializa Earth Engine con la cuenta de servicio en st.secrets."""

    try:
        secret_section = st.secrets["gcp_service_account"]
    except (FileNotFoundError, KeyError):
        st.error("No se encontraron las credenciales de Google Earth Engine.")
        st.info(
            "Agrega la cuenta de servicio en **Manage app → Settings → Secrets** "
            "bajo la sección `[gcp_service_account]`."
        )
        st.stop()

    key_dict = dict(secret_section)
    required = {
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
    missing = sorted(required.difference(key_dict))
    if missing:
        st.error(
            "La sección `[gcp_service_account]` está incompleta. "
            f"Faltan: {', '.join(missing)}"
        )
        st.stop()

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
            "esté habilitada y que la cuenta de servicio tenga acceso."
        )
        st.stop()


# =============================================================================
# Colecciones y geometrías de Earth Engine
# =============================================================================


def _lempa_collection() -> ee.FeatureCollection:
    basins = ee.FeatureCollection(HYBAS_ASSET)
    point = ee.Geometry.Point(list(LEMPA_REFERENCE_POINT))
    sample = ee.Feature(basins.filterBounds(point).first())
    main_basin_id = sample.get("MAIN_BAS")
    return basins.filter(ee.Filter.eq("MAIN_BAS", main_basin_id))


def _subbasin_feature(hybas_id: str | int) -> ee.Feature:
    numeric_id = int(str(hybas_id).replace(",", ""))
    feature = _lempa_collection().filter(ee.Filter.eq("HYBAS_ID", numeric_id)).first()
    return ee.Feature(feature)


def _subbasin_geometry(hybas_id: str | int) -> ee.Geometry:
    return _subbasin_feature(hybas_id).geometry()


def _chirps() -> ee.ImageCollection:
    return ee.ImageCollection(CHIRPS_ASSET).select(CHIRPS_BAND)


# =============================================================================
# Límites espaciales y selección
# =============================================================================


@st.cache_data(ttl=86_400, show_spinner=False)
def load_lempa_geojson(project_id: str) -> dict[str, Any]:
    del project_id  # forma parte de la llave de caché

    collection = _lempa_collection().select(
        ["HYBAS_ID", "MAIN_BAS", "SUB_AREA", "UP_AREA"]
    )

    def simplify_feature(feature: ee.Feature) -> ee.Feature:
        feature = ee.Feature(feature)
        return feature.setGeometry(feature.geometry().simplify(maxError=120))

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
) -> str | None:
    del project_id

    point = ee.Geometry.Point([longitude, latitude])
    result = (
        _lempa_collection()
        .filterBounds(point)
        .select(["HYBAS_ID"])
        .limit(1)
        .getInfo()
    )
    features = result.get("features", []) if isinstance(result, dict) else []
    if not features:
        return None
    value = features[0].get("properties", {}).get("HYBAS_ID")
    return normalize_hybas_id(value)


def normalize_hybas_id(value: Any) -> str | None:
    if value is None:
        return None
    text = re.sub(r"[^0-9]", "", str(value))
    return text or None


def feature_index(geojson: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for feature in geojson.get("features", []):
        hybas_id = normalize_hybas_id(feature.get("properties", {}).get("HYBAS_ID"))
        if hybas_id:
            index[hybas_id] = feature
    return index


def extract_hybas_id_from_map_event(
    map_data: dict[str, Any] | None,
    valid_ids: set[str],
) -> str | None:
    """Extrae HYBAS_ID desde tooltip/popup devuelto por streamlit-folium."""

    if not map_data:
        return None

    for key in ("last_object_clicked_tooltip", "last_object_clicked_popup"):
        raw = map_data.get(key)
        if not raw:
            continue
        text = re.sub(r"<[^>]+>", " ", str(raw))
        match = re.search(r"HYBAS[_ ]?ID\s*:?\s*([0-9,\. ]+)", text, re.I)
        if match:
            candidate = normalize_hybas_id(match.group(1))
            if candidate in valid_ids:
                return candidate

        # Respaldo: buscar cualquiera de los identificadores válidos dentro del texto.
        compact = re.sub(r"[^0-9]", "", text)
        for candidate in valid_ids:
            if candidate in compact:
                return candidate

    return None


# =============================================================================
# Datos CHIRPS
# =============================================================================


@st.cache_data(ttl=21_600, show_spinner=False)
def get_latest_chirps_date(project_id: str) -> date:
    del project_id
    value = (
        ee.Image(_chirps().sort("system:time_start", False).first())
        .date()
        .format("YYYY-MM-dd")
        .getInfo()
    )
    return datetime.strptime(value, "%Y-%m-%d").date()


def _annotate_basin_mean(
    collection: ee.ImageCollection,
    geometry: ee.Geometry,
    date_format: str,
) -> ee.ImageCollection:
    """Añade a cada imagen las propiedades date y rain_mm."""

    def annotate(image: ee.Image) -> ee.Image:
        image = ee.Image(image)
        mean_value = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=CHIRPS_SCALE_M,
            bestEffort=True,
            maxPixels=1_000_000_000,
            tileScale=4,
        ).get(CHIRPS_BAND)
        return image.set(
            {
                "date": image.date().format(date_format),
                "rain_mm": mean_value,
            }
        )

    return collection.map(annotate).filter(ee.Filter.notNull(["rain_mm"]))


def _collection_properties_to_frame(
    collection: ee.ImageCollection,
    date_column: str = "date",
    value_column: str = "rain_mm",
) -> pd.DataFrame:
    payload = ee.Dictionary(
        {
            "dates": collection.aggregate_array(date_column),
            "values": collection.aggregate_array(value_column),
        }
    ).getInfo()

    dates = payload.get("dates", []) if isinstance(payload, dict) else []
    values = payload.get("values", []) if isinstance(payload, dict) else []
    frame = pd.DataFrame({"fecha": dates, "precipitacion_mm": values})
    if frame.empty:
        return frame

    frame["fecha"] = pd.to_datetime(frame["fecha"], errors="coerce")
    frame["precipitacion_mm"] = pd.to_numeric(
        frame["precipitacion_mm"], errors="coerce"
    )
    frame = frame.dropna().sort_values("fecha").reset_index(drop=True)
    return frame


@st.cache_data(ttl=86_400, show_spinner=False)
def get_daily_rainfall(
    hybas_id: str,
    start_iso: str,
    end_iso: str,
    project_id: str,
) -> pd.DataFrame:
    del project_id

    start = ee.Date(start_iso)
    end_exclusive = ee.Date(end_iso).advance(1, "day")
    geometry = _subbasin_geometry(hybas_id)
    collection = _chirps().filterDate(start, end_exclusive)
    annotated = _annotate_basin_mean(collection, geometry, "YYYY-MM-dd")
    return _collection_properties_to_frame(annotated)


@st.cache_data(ttl=86_400, show_spinner=False)
def get_month_reference_rainfall(
    hybas_id: str,
    month: int,
    project_id: str,
) -> pd.DataFrame:
    del project_id

    geometry = _subbasin_geometry(hybas_id)
    collection = (
        _chirps()
        .filterDate(
            f"{REFERENCE_START_YEAR}-01-01",
            f"{REFERENCE_END_YEAR + 1}-01-01",
        )
        .filter(ee.Filter.calendarRange(month, month, "month"))
    )
    annotated = _annotate_basin_mean(collection, geometry, "YYYY-MM-dd")
    return _collection_properties_to_frame(annotated)


def calculate_thresholds(reference_df: pd.DataFrame, month: int) -> Thresholds:
    if reference_df.empty:
        raise ValueError("No hay datos históricos para calcular los umbrales.")

    all_values = reference_df["precipitacion_mm"].dropna().to_numpy(dtype=float)
    wet_values = all_values[all_values >= WET_DAY_THRESHOLD_MM]

    if wet_values.size < 30:
        raise ValueError(
            "La muestra de días húmedos es insuficiente para calcular percentiles "
            "con estabilidad."
        )

    p90, p95, p99 = np.percentile(wet_values, [90, 95, 99])
    return Thresholds(
        p90=float(p90),
        p95=float(p95),
        p99=float(p99),
        sample_all=int(all_values.size),
        sample_wet=int(wet_values.size),
        month=int(month),
    )


def _month_periods(start_year: int, end_month_iso: str) -> list[pd.Period]:
    end_period = pd.Period(end_month_iso, freq="M")
    start_period = pd.Period(f"{start_year}-01", freq="M")
    return list(pd.period_range(start_period, end_period, freq="M"))


@st.cache_data(ttl=86_400, show_spinner=False)
def get_monthly_rainfall(
    hybas_id: str,
    end_month_iso: str,
    project_id: str,
) -> pd.DataFrame:
    """Obtiene totales mensuales medios de la subcuenca desde 1991."""

    del project_id
    geometry = _subbasin_geometry(hybas_id)
    daily = _chirps()
    monthly_images: list[ee.Image] = []

    for period in _month_periods(REFERENCE_START_YEAR, end_month_iso):
        start = ee.Date.fromYMD(period.year, period.month, 1)
        end = start.advance(1, "month")
        image = (
            daily.filterDate(start, end)
            .sum()
            .rename(CHIRPS_BAND)
            .set(
                {
                    "system:time_start": start.millis(),
                    "date": start.format("YYYY-MM-dd"),
                }
            )
        )
        monthly_images.append(image)

    monthly_collection = ee.ImageCollection.fromImages(monthly_images)
    annotated = _annotate_basin_mean(monthly_collection, geometry, "YYYY-MM-dd")
    frame = _collection_properties_to_frame(annotated)
    if not frame.empty:
        frame["anio"] = frame["fecha"].dt.year
        frame["mes"] = frame["fecha"].dt.month
        frame["mes_nombre"] = frame["mes"].map(MONTH_NAMES)
    return frame


# =============================================================================
# SPI
# =============================================================================


def calculate_spi(
    monthly_df: pd.DataFrame,
    accumulation_months: int,
) -> pd.DataFrame:
    """Calcula SPI con ajuste gamma y probabilidad de ceros por mes calendario."""

    if monthly_df.empty:
        return monthly_df.copy()

    work = monthly_df[["fecha", "precipitacion_mm"]].copy()
    work = work.sort_values("fecha").reset_index(drop=True)
    work["acumulado_mm"] = (
        work["precipitacion_mm"]
        .rolling(window=accumulation_months, min_periods=accumulation_months)
        .sum()
    )
    work["mes"] = work["fecha"].dt.month
    work["anio"] = work["fecha"].dt.year
    work["spi"] = np.nan

    reference_mask = work["anio"].between(
        REFERENCE_START_YEAR, REFERENCE_END_YEAR, inclusive="both"
    )

    for calendar_month in range(1, 13):
        ref_values = work.loc[
            reference_mask
            & (work["mes"] == calendar_month)
            & work["acumulado_mm"].notna(),
            "acumulado_mm",
        ].to_numpy(dtype=float)

        if ref_values.size < 20:
            continue

        zero_probability = float(np.mean(ref_values <= 0.0))
        positive_values = ref_values[ref_values > 0.0]
        if positive_values.size < 15 or np.nanstd(positive_values) < 1e-8:
            continue

        try:
            shape, _, scale = gamma.fit(positive_values, floc=0)
        except Exception:
            continue

        target_index = work.index[
            (work["mes"] == calendar_month) & work["acumulado_mm"].notna()
        ]
        for idx in target_index:
            value = float(work.at[idx, "acumulado_mm"])
            if value <= 0:
                probability = max(zero_probability / 2.0, 1e-8)
            else:
                probability = zero_probability + (
                    (1.0 - zero_probability)
                    * float(gamma.cdf(value, shape, loc=0, scale=scale))
                )
            probability = float(np.clip(probability, 1e-8, 1 - 1e-8))
            work.at[idx, "spi"] = float(norm.ppf(probability))

    work["escala_meses"] = accumulation_months
    return work


# =============================================================================
# Alertas
# =============================================================================


def evaluate_flood_alert(rainfall: float, thresholds: Thresholds) -> AlertResult:
    if rainfall >= thresholds.p99:
        return AlertResult(
            level="Roja",
            title="🔴 ALERTA ROJA — PRECIPITACIÓN EXTREMA",
            background="#fef2f2",
            border="#dc2626",
            text="#991b1b",
            action=(
                "Activar protocolos de emergencia, vigilancia continua de cauces, "
                "revisión de rutas de evacuación y coordinación con protección civil."
            ),
            detail=(
                f"La precipitación evaluada ({rainfall:.1f} mm/día) iguala o supera "
                f"el P99 local ({thresholds.p99:.1f} mm/día)."
            ),
        )
    if rainfall >= thresholds.p95:
        return AlertResult(
            level="Naranja",
            title="🟠 ALERTA NARANJA — CONDICIÓN MUY ALTA",
            background="#fff7ed",
            border="#ea580c",
            text="#9a3412",
            action=(
                "Intensificar el monitoreo, preparar evacuaciones preventivas y "
                "restringir actividades en riberas, vados y llanuras de inundación."
            ),
            detail=(
                f"La precipitación evaluada ({rainfall:.1f} mm/día) iguala o supera "
                f"el P95 local ({thresholds.p95:.1f} mm/día)."
            ),
        )
    if rainfall >= thresholds.p90:
        return AlertResult(
            level="Amarilla",
            title="🟡 ALERTA AMARILLA — FASE PREVENTIVA",
            background="#fefce8",
            border="#ca8a04",
            text="#854d0e",
            action=(
                "Activar vigilancia preventiva, verificar drenajes y puntos críticos, "
                "y mantener comunicación con los comités locales."
            ),
            detail=(
                f"La precipitación evaluada ({rainfall:.1f} mm/día) iguala o supera "
                f"el P90 local ({thresholds.p90:.1f} mm/día)."
            ),
        )
    return AlertResult(
        level="Verde",
        title="🟢 ALERTA VERDE — BAJO EL UMBRAL P90",
        background="#f0fdf4",
        border="#16a34a",
        text="#166534",
        action=(
            "Mantener monitoreo ordinario y reportar cambios observados en niveles de "
            "ríos, drenajes, vados o laderas."
        ),
        detail=(
            f"La precipitación evaluada ({rainfall:.1f} mm/día) está por debajo "
            f"del P90 local ({thresholds.p90:.1f} mm/día)."
        ),
    )


def evaluate_drought_alert(spi_value: float) -> AlertResult:
    if spi_value <= -2.0:
        return AlertResult(
            level="Sequía extrema",
            title="🔴 SEQUÍA EXTREMA — RESPUESTA PRIORITARIA",
            background="#fef2f2",
            border="#dc2626",
            text="#991b1b",
            action=(
                "Priorizar abastecimiento humano, activar planes de emergencia hídrica "
                "y evaluar afectaciones agropecuarias."
            ),
            detail=f"SPI observado: {spi_value:.2f}.",
        )
    if spi_value <= -1.5:
        return AlertResult(
            level="Sequía severa",
            title="🟠 SEQUÍA SEVERA — ESTADO CRÍTICO",
            background="#fff7ed",
            border="#ea580c",
            text="#9a3412",
            action=(
                "Reforzar reservorios, revisar disponibilidad de agua y activar "
                "asistencia técnica para actividades productivas vulnerables."
            ),
            detail=f"SPI observado: {spi_value:.2f}.",
        )
    if spi_value <= -1.0:
        return AlertResult(
            level="Sequía moderada",
            title="🟡 SEQUÍA MODERADA — FASE PREVENTIVA",
            background="#fefce8",
            border="#ca8a04",
            text="#854d0e",
            action=(
                "Promover conservación de humedad, uso eficiente del agua y seguimiento "
                "de cultivos y fuentes locales."
            ),
            detail=f"SPI observado: {spi_value:.2f}.",
        )
    if spi_value >= 2.0:
        title = "🔵 CONDICIÓN EXTREMADAMENTE HÚMEDA"
    elif spi_value >= 1.5:
        title = "🔵 CONDICIÓN MUY HÚMEDA"
    elif spi_value >= 1.0:
        title = "🔵 CONDICIÓN MODERADAMENTE HÚMEDA"
    else:
        title = "🟢 CONDICIÓN CERCANA A LO NORMAL"

    return AlertResult(
        level="Sin sequía",
        title=title,
        background="#eff6ff" if spi_value >= 1.0 else "#f0fdf4",
        border="#2563eb" if spi_value >= 1.0 else "#16a34a",
        text="#1e40af" if spi_value >= 1.0 else "#166534",
        action=(
            "Continuar el seguimiento mensual y mantener medidas ordinarias de "
            "conservación y uso eficiente del agua."
        ),
        detail=f"SPI observado: {spi_value:.2f}.",
    )


# =============================================================================
# Monitoreo general de todas las subcuencas
# =============================================================================


@st.cache_data(ttl=21_600, show_spinner=False)
def get_all_basin_daily_rainfall(
    selected_date_iso: str,
    project_id: str,
) -> pd.DataFrame:
    """Obtiene la precipitación media diaria de todas las subcuencas en una operación."""

    del project_id

    selected_ee_date = ee.Date(selected_date_iso)
    image_collection = _chirps().filterDate(
        selected_ee_date,
        selected_ee_date.advance(1, "day"),
    )

    if int(image_collection.size().getInfo()) == 0:
        return pd.DataFrame(
            columns=[
                "HYBAS_ID",
                "MAIN_BAS",
                "SUB_AREA_km2",
                "UP_AREA_km2",
                "fecha_lluvia",
                "lluvia_mm_dia",
            ]
        )

    image = ee.Image(image_collection.first())
    image_date = image.date().format("YYYY-MM-dd").getInfo()

    basins = _lempa_collection().select(
        ["HYBAS_ID", "MAIN_BAS", "SUB_AREA", "UP_AREA"]
    )

    reduced = image.reduceRegions(
        collection=basins,
        reducer=ee.Reducer.mean().setOutputs(["rain_mm"]),
        scale=CHIRPS_SCALE_M,
        tileScale=4,
    )

    payload = reduced.getInfo()
    records: list[dict[str, Any]] = []

    for feature in payload.get("features", []):
        properties = feature.get("properties", {})
        hybas_id = normalize_hybas_id(properties.get("HYBAS_ID"))
        rainfall = properties.get("rain_mm")

        if hybas_id is None or rainfall is None:
            continue

        records.append(
            {
                "HYBAS_ID": hybas_id,
                "MAIN_BAS": properties.get("MAIN_BAS"),
                "SUB_AREA_km2": properties.get("SUB_AREA"),
                "UP_AREA_km2": properties.get("UP_AREA"),
                "fecha_lluvia": image_date,
                "lluvia_mm_dia": float(rainfall),
            }
        )

    frame = pd.DataFrame.from_records(records)
    if not frame.empty:
        frame["fecha_lluvia"] = pd.to_datetime(frame["fecha_lluvia"])
        frame = frame.sort_values("HYBAS_ID", key=lambda column: column.astype("int64"))

    return frame.reset_index(drop=True)


@st.cache_data(ttl=2_592_000, show_spinner=False)
def get_all_basin_month_thresholds(
    month: int,
    project_id: str,
) -> pd.DataFrame:
    """Calcula P90, P95 y P99 para todas las subcuencas y un mes calendario."""

    del project_id

    basins = _lempa_collection().select(["HYBAS_ID"])
    collection = (
        _chirps()
        .filterDate(
            f"{REFERENCE_START_YEAR}-01-01",
            f"{REFERENCE_END_YEAR + 1}-01-01",
        )
        .filter(ee.Filter.calendarRange(int(month), int(month), "month"))
    )

    image_list = collection.toList(collection.size())

    def reduce_one_image(image_object: Any) -> ee.FeatureCollection:
        image = ee.Image(image_object)
        reduced = image.reduceRegions(
            collection=basins,
            reducer=ee.Reducer.mean().setOutputs(["rain_mm"]),
            scale=CHIRPS_SCALE_M,
            tileScale=4,
        )

        def keep_fields(feature: ee.Feature) -> ee.Feature:
            feature = ee.Feature(feature)
            return ee.Feature(
                None,
                {
                    "HYBAS_ID": feature.get("HYBAS_ID"),
                    "rain_mm": feature.get("rain_mm"),
                },
            )

        return reduced.map(keep_fields)

    samples = (
        ee.FeatureCollection(image_list.map(reduce_one_image))
        .flatten()
        .filter(ee.Filter.notNull(["HYBAS_ID", "rain_mm"]))
        .filter(ee.Filter.gte("rain_mm", WET_DAY_THRESHOLD_MM))
    )

    payload = ee.Dictionary(
        {
            "ids": samples.aggregate_array("HYBAS_ID"),
            "values": samples.aggregate_array("rain_mm"),
        }
    ).getInfo()

    ids = payload.get("ids", []) if isinstance(payload, dict) else []
    values = payload.get("values", []) if isinstance(payload, dict) else []

    sample_frame = pd.DataFrame(
        {
            "HYBAS_ID": [normalize_hybas_id(value) for value in ids],
            "rain_mm": pd.to_numeric(values, errors="coerce"),
        }
    ).dropna()

    if sample_frame.empty:
        return pd.DataFrame(
            columns=["HYBAS_ID", "P90_mm_dia", "P95_mm_dia", "P99_mm_dia", "dias_humedos"]
        )

    quantiles = (
        sample_frame.groupby("HYBAS_ID")["rain_mm"]
        .quantile([0.90, 0.95, 0.99])
        .unstack()
        .rename(
            columns={
                0.90: "P90_mm_dia",
                0.95: "P95_mm_dia",
                0.99: "P99_mm_dia",
            }
        )
        .reset_index()
    )

    counts = (
        sample_frame.groupby("HYBAS_ID")
        .size()
        .rename("dias_humedos")
        .reset_index()
    )

    return quantiles.merge(counts, on="HYBAS_ID", how="left")


@st.cache_data(ttl=2_592_000, show_spinner=False)
def get_all_basin_monthly_rainfall(
    end_month_iso: str,
    project_id: str,
) -> pd.DataFrame:
    """Obtiene la serie mensual 1991–fecha final para todas las subcuencas."""

    del project_id

    end_period = pd.Period(end_month_iso, freq="M")
    first_month = ee.Date(f"{REFERENCE_START_YEAR}-01-01")
    end_exclusive = ee.Date(
        f"{end_period.year}-{end_period.month:02d}-01"
    ).advance(1, "month")

    month_count = end_exclusive.difference(first_month, "month").round()
    offsets = ee.List.sequence(0, ee.Number(month_count).subtract(1))
    daily = _chirps()

    def make_month(offset: Any) -> ee.Image:
        offset = ee.Number(offset)
        start = first_month.advance(offset, "month")
        end = start.advance(1, "month")
        return (
            daily.filterDate(start, end)
            .sum()
            .rename(CHIRPS_BAND)
            .set(
                {
                    "system:time_start": start.millis(),
                    "date": start.format("YYYY-MM-dd"),
                }
            )
        )

    monthly_collection = ee.ImageCollection.fromImages(offsets.map(make_month))
    monthly_list = monthly_collection.toList(monthly_collection.size())
    basins = _lempa_collection().select(["HYBAS_ID"])

    def reduce_one_month(image_object: Any) -> ee.FeatureCollection:
        image = ee.Image(image_object)
        month_date = image.get("date")
        reduced = image.reduceRegions(
            collection=basins,
            reducer=ee.Reducer.mean().setOutputs(["rain_mm"]),
            scale=CHIRPS_SCALE_M,
            tileScale=4,
        )

        def keep_fields(feature: ee.Feature) -> ee.Feature:
            feature = ee.Feature(feature)
            return ee.Feature(
                None,
                {
                    "HYBAS_ID": feature.get("HYBAS_ID"),
                    "date": month_date,
                    "rain_mm": feature.get("rain_mm"),
                },
            )

        return reduced.map(keep_fields)

    samples = (
        ee.FeatureCollection(monthly_list.map(reduce_one_month))
        .flatten()
        .filter(ee.Filter.notNull(["HYBAS_ID", "date", "rain_mm"]))
    )

    payload = ee.Dictionary(
        {
            "ids": samples.aggregate_array("HYBAS_ID"),
            "dates": samples.aggregate_array("date"),
            "values": samples.aggregate_array("rain_mm"),
        }
    ).getInfo()

    ids = payload.get("ids", []) if isinstance(payload, dict) else []
    dates = payload.get("dates", []) if isinstance(payload, dict) else []
    values = payload.get("values", []) if isinstance(payload, dict) else []

    frame = pd.DataFrame(
        {
            "HYBAS_ID": [normalize_hybas_id(value) for value in ids],
            "fecha": pd.to_datetime(dates, errors="coerce"),
            "precipitacion_mm": pd.to_numeric(values, errors="coerce"),
        }
    ).dropna()

    return frame.sort_values(["HYBAS_ID", "fecha"]).reset_index(drop=True)


def calculate_latest_spi_all_basins(
    monthly_frame: pd.DataFrame,
    spi_scale: int,
    selected_date: date,
) -> pd.DataFrame:
    """Calcula el último SPI disponible de cada subcuenca."""

    if monthly_frame.empty:
        return pd.DataFrame(columns=["HYBAS_ID", "SPI_fecha", "SPI_valor"])

    target_date = pd.Timestamp(selected_date)
    records: list[dict[str, Any]] = []

    for hybas_id, group in monthly_frame.groupby("HYBAS_ID", sort=False):
        basin_series = group[["fecha", "precipitacion_mm"]].copy()
        spi_frame = calculate_spi(basin_series, int(spi_scale))
        available = spi_frame.loc[
            spi_frame["fecha"] <= target_date
        ].dropna(subset=["spi"])

        if available.empty:
            records.append(
                {
                    "HYBAS_ID": hybas_id,
                    "SPI_fecha": pd.NaT,
                    "SPI_valor": math.nan,
                }
            )
            continue

        latest = available.iloc[-1]
        records.append(
            {
                "HYBAS_ID": hybas_id,
                "SPI_fecha": latest["fecha"],
                "SPI_valor": float(latest["spi"]),
            }
        )

    return pd.DataFrame.from_records(records)


def build_general_monitoring_table(
    daily_frame: pd.DataFrame,
    threshold_frame: pd.DataFrame,
    spi_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    """Integra lluvia, percentiles y SPI y asigna prioridades de monitoreo."""

    result = daily_frame.merge(
        threshold_frame,
        on="HYBAS_ID",
        how="left",
    )

    if spi_frame is not None and not spi_frame.empty:
        result = result.merge(spi_frame, on="HYBAS_ID", how="left")
    else:
        result["SPI_fecha"] = pd.NaT
        result["SPI_valor"] = math.nan

    def rain_classification(row: pd.Series) -> tuple[str, int]:
        rainfall = row.get("lluvia_mm_dia")
        p90 = row.get("P90_mm_dia")
        p95 = row.get("P95_mm_dia")
        p99 = row.get("P99_mm_dia")

        if not all(np.isfinite(value) for value in [rainfall, p90, p95, p99]):
            return "⚪ Sin dato", -1
        if rainfall >= p99:
            return "🔴 Extrema", 4
        if rainfall >= p95:
            return "🟠 Muy alta", 3
        if rainfall >= p90:
            return "🟡 Preventiva", 2
        return "🟢 Normal", 0

    def spi_classification(value: Any) -> tuple[str, int, bool]:
        if value is None or not np.isfinite(value):
            return "⚪ Sin cálculo", -1, False
        value = float(value)
        if value <= -2.0:
            return "🔴 Sequía extrema", 4, True
        if value <= -1.5:
            return "🟠 Sequía severa", 3, True
        if value <= -1.0:
            return "🟡 Sequía moderada", 2, True
        if value >= 2.0:
            return "🔵 Extremadamente húmeda", 2, False
        if value >= 1.5:
            return "🔵 Muy húmeda", 1, False
        if value >= 1.0:
            return "🔵 Moderadamente húmeda", 0, False
        return "🟢 Cercana a lo normal", 0, False

    rain_values = result.apply(rain_classification, axis=1)
    result["estado_lluvia"] = [value[0] for value in rain_values]
    result["puntaje_lluvia"] = [value[1] for value in rain_values]

    spi_values = result["SPI_valor"].apply(spi_classification)
    result["estado_SPI"] = [value[0] for value in spi_values]
    result["puntaje_SPI"] = [value[1] for value in spi_values]
    result["es_sequia"] = [value[2] for value in spi_values]

    result["alerta_lluvia"] = result["puntaje_lluvia"] >= 2
    result["alerta_sequia"] = result["es_sequia"]
    result["humedad_persistente"] = result["SPI_valor"].ge(1.5).fillna(False)
    result["doble_senal"] = result["alerta_lluvia"] & result["alerta_sequia"]
    result["requiere_monitoreo"] = (
        result["alerta_lluvia"]
        | result["alerta_sequia"]
        | result["humedad_persistente"]
    )

    def priority_label(row: pd.Series) -> str:
        if bool(row["doble_senal"]):
            return "🟣 Atención combinada"
        highest = max(int(row["puntaje_lluvia"]), int(row["puntaje_SPI"]))
        if highest >= 4:
            return "🔴 Prioridad muy alta"
        if highest == 3:
            return "🟠 Prioridad alta"
        if highest == 2:
            return "🟡 Monitoreo preventivo"
        if bool(row["humedad_persistente"]):
            return "🔵 Humedad persistente"
        return "🟢 Seguimiento ordinario"

    def recommendation(row: pd.Series) -> str:
        if bool(row["doble_senal"]):
            return (
                "Vigilar cauces y drenajes, pero mantener medidas de recuperación "
                "hídrica: una lluvia intensa puede ocurrir tras un déficit prolongado."
            )
        if int(row["puntaje_lluvia"]) >= 4:
            return "Activar coordinación operativa y vigilancia continua de cauces."
        if int(row["puntaje_lluvia"]) == 3:
            return "Intensificar monitoreo de riberas, vados y zonas bajas."
        if int(row["puntaje_lluvia"]) == 2:
            return "Verificar drenajes y puntos críticos; mantener vigilancia preventiva."
        if bool(row["alerta_sequia"]):
            if float(row["SPI_valor"]) <= -2.0:
                return "Priorizar abastecimiento humano y activar respuesta por sequía."
            if float(row["SPI_valor"]) <= -1.5:
                return "Revisar fuentes de agua y afectaciones agropecuarias."
            return "Promover conservación de humedad y uso eficiente del agua."
        if bool(row["humedad_persistente"]):
            return "Observar saturación antecedente, laderas y respuesta de los cauces."
        return "Mantener seguimiento ordinario."

    result["prioridad"] = result.apply(priority_label, axis=1)
    result["recomendacion"] = result.apply(recommendation, axis=1)
    result["puntaje_prioridad"] = result[["puntaje_lluvia", "puntaje_SPI"]].max(axis=1)
    result.loc[result["doble_senal"], "puntaje_prioridad"] = 5

    return result.sort_values(
        ["requiere_monitoreo", "puntaje_prioridad", "lluvia_mm_dia"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def render_general_monitoring_tab(
    project_id: str,
    selected_date: date,
    latest_date: date,
    spi_scale: int,
    expected_basin_count: int,
) -> None:
    """Interfaz del monitoreo general, ejecutado solo cuando la persona lo solicita."""

    st.subheader("🚨 Monitoreo general de subcuencas")
    st.markdown(
        """
        <div class="sat-note">
            Esta sección revisa todas las subcuencas disponibles y señala cuáles requieren
            <b>monitoreo hidroclimático</b>. El cálculo
            se ejecuta únicamente al presionar el botón.
        </div>
        """,
        unsafe_allow_html=True,
    )

    controls_left, controls_middle, controls_right = st.columns([1.1, 1.1, 1.4])

    with controls_left:
        st.metric(
            "Fecha de lluvia",
            selected_date.strftime("%d/%m/%Y"),
            "Control del panel lateral",
        )

    with controls_middle:
        st.metric(
            "Escala de sequía",
            f"SPI-{spi_scale}",
            "Control del panel lateral",
        )

    with controls_right:
        include_spi = st.checkbox(
            "Incluir SPI para todas las subcuencas",
            value=True,
            key="general_monitor_include_spi",
            help=(
                "El SPI general requiere una consulta mensual histórica. Es la parte más "
                "lenta, pero queda almacenada en caché durante 30 días."
            ),
        )

    run_monitoring = st.button(
        "🔄 Actualizar monitoreo general",
        type="primary",
        width="stretch",
        key="run_general_monitoring",
    )

    if run_monitoring:
        try:
            with st.status(
                "Procesando monitoreo general...",
                expanded=True,
            ) as status:
                st.write("1/3 Consultando precipitación diaria de todas las subcuencas...")
                daily_all = get_all_basin_daily_rainfall(
                    selected_date.isoformat(),
                    project_id,
                )

                st.write(
                    "2/3 Calculando o recuperando P90, P95 y P99 del mes seleccionado..."
                )
                thresholds_all = get_all_basin_month_thresholds(
                    selected_date.month,
                    project_id,
                )

                spi_latest: pd.DataFrame | None = None
                if include_spi:
                    st.write(
                        f"3/3 Calculando o recuperando series mensuales y SPI-{spi_scale}..."
                    )
                    end_month = latest_complete_month(latest_date)
                    monthly_all = get_all_basin_monthly_rainfall(
                        str(end_month),
                        project_id,
                    )
                    spi_latest = calculate_latest_spi_all_basins(
                        monthly_all,
                        int(spi_scale),
                        selected_date,
                    )
                else:
                    st.write("3/3 SPI omitido por elección de la persona usuaria.")

                monitoring_frame = build_general_monitoring_table(
                    daily_all,
                    thresholds_all,
                    spi_latest,
                )

                st.session_state["general_monitoring_frame"] = monitoring_frame
                st.session_state["general_monitoring_metadata"] = {
                    "selected_date": selected_date.isoformat(),
                    "spi_scale": int(spi_scale),
                    "include_spi": bool(include_spi),
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                }

                status.update(
                    label="Monitoreo general completado.",
                    state="complete",
                    expanded=False,
                )
        except Exception as exc:
            st.error("No fue posible completar el monitoreo general.")
            st.code(str(exc), language="text")
            st.info(
                "Puedes intentar primero sin SPI. Los umbrales y series históricas quedan "
                "en caché para que las siguientes consultas sean más rápidas."
            )

    monitoring_frame = st.session_state.get("general_monitoring_frame")
    metadata = st.session_state.get("general_monitoring_metadata", {})

    if monitoring_frame is None or monitoring_frame.empty:
        st.info(
            "Presiona **Actualizar monitoreo general** para evaluar todas las subcuencas. "
            "La primera ejecución con SPI puede tardar más que las siguientes."
        )
        return

    result_date = metadata.get("selected_date")
    result_scale = metadata.get("spi_scale")
    result_has_spi = metadata.get("include_spi", False)

    if (
        result_date != selected_date.isoformat()
        or int(result_scale or 0) != int(spi_scale)
        or bool(result_has_spi) != bool(include_spi)
    ):
        st.warning(
            "Los resultados mostrados corresponden a otros parámetros. Presiona "
            "**Actualizar monitoreo general** para aplicar la fecha y escala actuales."
        )

    evaluated_count = len(monitoring_frame)
    rain_count = int(monitoring_frame["alerta_lluvia"].sum())
    drought_count = int(monitoring_frame["alerta_sequia"].sum())
    combined_count = int(monitoring_frame["doble_senal"].sum())
    wet_count = int(monitoring_frame["humedad_persistente"].sum())
    watch_count = int(monitoring_frame["requiere_monitoreo"].sum())

    metric1, metric2, metric3, metric4, metric5 = st.columns(5)
    metric1.metric(
        "Evaluadas",
        evaluated_count,
        f"de {expected_basin_count} disponibles",
    )
    metric2.metric("Alerta de lluvia", rain_count, "≥ P90")
    metric3.metric("Alerta de sequía", drought_count, "SPI ≤ -1")
    metric4.metric("Doble señal", combined_count, "Lluvia + sequía")
    metric5.metric("Humedad persistente", wet_count, "SPI ≥ 1.5")

    watch_frame = monitoring_frame.loc[
        monitoring_frame["requiere_monitoreo"]
    ].copy()

    if watch_frame.empty:
        st.success(
            f"✅ Ninguna de las {evaluated_count} subcuencas evaluadas presenta una señal "
            "que requiera monitoreo extraordinario con los criterios actuales."
        )
    else:
        priority_ids = watch_frame["HYBAS_ID"].head(12).tolist()
        priority_text = ", ".join(priority_ids)
        remaining = max(0, watch_count - len(priority_ids))
        suffix = f" y {remaining} más" if remaining else ""

        st.warning(
            f"⚠️ **{watch_count} de {evaluated_count} subcuencas requieren atención o "
            f"monitoreo.** Prioridad inicial: {priority_text}{suffix}."
        )

    show_all = st.checkbox(
        "Mostrar también subcuencas en seguimiento ordinario",
        value=False,
        key="show_all_general_monitoring",
    )
    table_frame = monitoring_frame.copy() if show_all else watch_frame.copy()

    display_columns = [
        "prioridad",
        "HYBAS_ID",
        "lluvia_mm_dia",
        "P90_mm_dia",
        "P95_mm_dia",
        "P99_mm_dia",
        "estado_lluvia",
        "SPI_valor",
        "estado_SPI",
        "recomendacion",
    ]
    display_columns = [
        column for column in display_columns if column in table_frame.columns
    ]

    st.dataframe(
        table_frame[display_columns].style.format(
            {
                "lluvia_mm_dia": "{:.2f}",
                "P90_mm_dia": "{:.2f}",
                "P95_mm_dia": "{:.2f}",
                "P99_mm_dia": "{:.2f}",
                "SPI_valor": "{:.2f}",
            },
            na_rep="N/D",
        ),
        use_container_width=True,
        hide_index=True,
        height=520,
    )

    st.download_button(
        "⬇️ Descargar monitoreo general CSV",
        data=dataframe_csv_bytes(monitoring_frame),
        file_name=(
            f"monitoreo_general_{selected_date.isoformat()}_SPI{spi_scale}.csv"
        ),
        mime="text/csv",
        width="stretch",
        on_click="ignore",
    )

    generated_at = metadata.get("generated_at", "N/D")
    st.caption(
        f"Resultados generados: {generated_at}. Fuente: CHIRPS Daily. "
        "Los resultados son señales de monitoreo y no sustituyen alertas oficiales."
    )

    with st.expander("Metodología y consumo de Earth Engine"):
        st.markdown(
            f"""
            - La lluvia diaria se obtiene con una sola operación `reduceRegions` para todas
              las subcuencas.
            - Los P90, P95 y P99 usan días húmedos de
              **{REFERENCE_START_YEAR}–{REFERENCE_END_YEAR}** para el mismo mes calendario.
            - El SPI se calcula por subcuenca con distribución gamma y la escala seleccionada.
            - Las consultas diarias se guardan en caché durante 6 horas; los umbrales y
              series mensuales, durante 30 días.
            - Se considera monitoreo por lluvia cuando la media diaria alcanza P90;
              por sequía cuando SPI ≤ -1; y por humedad persistente cuando SPI ≥ 1.5.
            - Esta es una **prealerta hidroclimática**. Para declarar riesgo deben integrarse
              caudales, niveles de río, humedad antecedente, exposición y vulnerabilidad.
            """
        )


def render_alert_card(
    result: AlertResult,
    hybas_id: str,
    scenario_text: str,
) -> None:
    st.markdown(
        f"""
        <div class="alert-card"
             style="background:{result.background}; border:1px solid {result.border};
                    border-left:8px solid {result.border};">
            <div class="alert-location">
                📍 <b>SUBCUENCA:</b> {hybas_id}
                &nbsp;|&nbsp; <b>EVALUACIÓN:</b> {scenario_text}
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


# =============================================================================
# Cartografía
# =============================================================================


def _iter_coordinate_pairs(value: Any) -> Iterable[tuple[float, float]]:
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
    pairs: list[tuple[float, float]] = []
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry") or {}
        pairs.extend(_iter_coordinate_pairs(geometry.get("coordinates", [])))
    if not pairs:
        return [[13.0, -90.5], [15.2, -87.5]]
    longitudes = [pair[0] for pair in pairs]
    latitudes = [pair[1] for pair in pairs]
    return [[min(latitudes), min(longitudes)], [max(latitudes), max(longitudes)]]


@st.cache_data(ttl=21_600, show_spinner=False)
def get_chirps_tile_url(
    hybas_id: str,
    selected_date_iso: str,
    project_id: str,
) -> str | None:
    del project_id

    start = ee.Date(selected_date_iso)
    image_collection = _chirps().filterDate(start, start.advance(1, "day"))
    if int(image_collection.size().getInfo()) == 0:
        return None

    image = ee.Image(image_collection.first()).clip(_subbasin_geometry(hybas_id))
    map_id = image.getMapId(
        {
            "min": 0,
            "max": 80,
            "palette": [
                "ffffff",
                "dbeafe",
                "93c5fd",
                "3b82f6",
                "facc15",
                "f97316",
                "dc2626",
            ],
        }
    )
    return map_id["tile_fetcher"].url_format


def build_map(
    geojson: dict[str, Any],
    selected_id: str | None,
    rain_tile_url: str | None = None,
) -> folium.Map:
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

    if rain_tile_url:
        folium.TileLayer(
            tiles=rain_tile_url,
            attr="CHIRPS / UCSB-CHG / Google Earth Engine",
            name="Precipitación CHIRPS del día",
            overlay=True,
            control=True,
            show=True,
            opacity=0.72,
        ).add_to(map_object)

    def style_function(feature: dict[str, Any]) -> dict[str, Any]:
        feature_id = normalize_hybas_id(feature.get("properties", {}).get("HYBAS_ID"))
        is_selected = selected_id is not None and feature_id == selected_id
        return {
            "color": "#b91c1c" if is_selected else "#334155",
            "weight": 4 if is_selected else 1.35,
            "fillColor": "#facc15" if is_selected else "#38bdf8",
            "fillOpacity": 0.42 if is_selected else 0.07,
        }

    def highlight_function(_: dict[str, Any]) -> dict[str, Any]:
        return {
            "color": "#f59e0b",
            "weight": 3,
            "fillColor": "#fde68a",
            "fillOpacity": 0.32,
        }

    folium.GeoJson(
        data=geojson,
        name="Subcuencas HydroSHEDS nivel 12",
        style_function=style_function,
        highlight_function=highlight_function,
        popup_keep_highlighted=True,
        tooltip=folium.GeoJsonTooltip(
            fields=["HYBAS_ID"],
            aliases=["HYBAS_ID:"],
            localize=False,
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
            localize=False,
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


# =============================================================================
# Gráficos
# =============================================================================


def _white_figure_layout(fig: go.Figure, title: str, y_title: str) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#0f172a"},
        margin={"l": 45, "r": 25, "t": 70, "b": 45},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hovermode="x unified",
    )
    fig.update_yaxes(title=y_title, gridcolor="#e2e8f0", zerolinecolor="#cbd5e1")
    fig.update_xaxes(gridcolor="#f1f5f9")
    return fig


def daily_rainfall_chart(
    daily_df: pd.DataFrame,
    thresholds: Thresholds,
    selected_date: date,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=daily_df["fecha"],
            y=daily_df["precipitacion_mm"],
            name="Precipitación",
            marker_color="#2563eb",
            hovertemplate="%{x|%d/%m/%Y}<br>%{y:.1f} mm<extra></extra>",
        )
    )
    lines = [
        (thresholds.p90, "P90", "#ca8a04"),
        (thresholds.p95, "P95", "#ea580c"),
        (thresholds.p99, "P99", "#dc2626"),
    ]
    for value, label, color in lines:
        fig.add_hline(
            y=value,
            line_dash="dash",
            line_color=color,
            annotation_text=f"{label}: {value:.1f}",
            annotation_position="top left",
        )
    fig.add_vline(
        x=pd.Timestamp(selected_date),
        line_dash="dot",
        line_color="#0f172a",
        annotation_text="Fecha evaluada",
        annotation_position="top right",
    )
    return _white_figure_layout(
        fig,
        "Precipitación diaria media de la subcuenca",
        "Precipitación (mm/día)",
    )


def reference_distribution_chart(
    reference_df: pd.DataFrame,
    thresholds: Thresholds,
) -> go.Figure:
    wet = reference_df.loc[
        reference_df["precipitacion_mm"] >= WET_DAY_THRESHOLD_MM,
        "precipitacion_mm",
    ]
    fig = go.Figure(
        go.Histogram(
            x=wet,
            nbinsx=35,
            name="Días húmedos",
            marker_color="#0ea5e9",
            opacity=0.82,
            hovertemplate="Rango: %{x}<br>Frecuencia: %{y}<extra></extra>",
        )
    )
    for value, label, color in [
        (thresholds.p90, "P90", "#ca8a04"),
        (thresholds.p95, "P95", "#ea580c"),
        (thresholds.p99, "P99", "#dc2626"),
    ]:
        fig.add_vline(
            x=value,
            line_dash="dash",
            line_color=color,
            annotation_text=f"{label}: {value:.1f}",
            annotation_position="top",
        )
    return _white_figure_layout(
        fig,
        f"Distribución histórica de días húmedos — {MONTH_NAMES[thresholds.month]}",
        "Frecuencia",
    )


def monthly_climatology_chart(monthly_df: pd.DataFrame) -> go.Figure:
    reference = monthly_df.loc[
        monthly_df["anio"].between(
            REFERENCE_START_YEAR, REFERENCE_END_YEAR, inclusive="both"
        )
    ]
    climatology = (
        reference.groupby("mes", as_index=False)["precipitacion_mm"]
        .agg(["mean", "median"])
        .reset_index()
    )
    climatology["mes_nombre"] = climatology["mes"].map(MONTH_NAMES)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=climatology["mes_nombre"],
            y=climatology["mean"],
            name="Promedio 1991–2020",
            marker_color="#0284c7",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=climatology["mes_nombre"],
            y=climatology["median"],
            name="Mediana 1991–2020",
            mode="lines+markers",
            line={"color": "#0f172a", "width": 2},
        )
    )
    return _white_figure_layout(
        fig,
        "Climatología mensual de precipitación",
        "Precipitación mensual (mm)",
    )


def monthly_series_chart(monthly_df: pd.DataFrame) -> go.Figure:
    recent = monthly_df.tail(120)
    fig = go.Figure(
        go.Bar(
            x=recent["fecha"],
            y=recent["precipitacion_mm"],
            name="Precipitación mensual",
            marker_color="#0284c7",
            hovertemplate="%{x|%b %Y}<br>%{y:.1f} mm<extra></extra>",
        )
    )
    return _white_figure_layout(
        fig,
        "Precipitación mensual — últimos 10 años disponibles",
        "Precipitación (mm/mes)",
    )


def spi_chart(spi_df: pd.DataFrame, scale_months: int) -> go.Figure:
    plot_df = spi_df.dropna(subset=["spi"]).tail(240)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["fecha"],
            y=plot_df["spi"],
            mode="lines",
            name=f"SPI-{scale_months}",
            line={"color": "#334155", "width": 2},
            fill="tozeroy",
            fillcolor="rgba(59, 130, 246, 0.12)",
            hovertemplate="%{x|%b %Y}<br>SPI: %{y:.2f}<extra></extra>",
        )
    )
    for value, color in [(-2.0, "#dc2626"), (-1.5, "#ea580c"), (-1.0, "#ca8a04")]:
        fig.add_hline(y=value, line_dash="dash", line_color=color)
    fig.add_hline(y=0, line_color="#64748b", line_width=1)
    return _white_figure_layout(
        fig,
        f"Índice estandarizado de precipitación SPI-{scale_months}",
        "SPI",
    )


# =============================================================================
# Descargas
# =============================================================================


def dataframe_csv_bytes(frame: pd.DataFrame) -> bytes:
    export = frame.copy()
    for column in export.columns:
        if pd.api.types.is_datetime64_any_dtype(export[column]):
            export[column] = export[column].dt.strftime("%Y-%m-%d")
    return export.to_csv(index=False).encode("utf-8-sig")


def selected_geojson_bytes(feature: dict[str, Any]) -> bytes:
    payload = {"type": "FeatureCollection", "features": [feature]}
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def create_download_zip(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, content in files.items():
            archive.writestr(filename, content)
    return buffer.getvalue()


# =============================================================================
# Utilidades de interfaz
# =============================================================================


def selected_observation(
    daily_df: pd.DataFrame,
    selected_date: date,
) -> tuple[date, float] | None:
    if daily_df.empty:
        return None
    target = pd.Timestamp(selected_date)
    available = daily_df.loc[daily_df["fecha"] <= target]
    if available.empty:
        return None
    row = available.iloc[-1]
    return row["fecha"].date(), float(row["precipitacion_mm"])


def sum_last_days(daily_df: pd.DataFrame, end_date: date, days: int) -> float:
    if daily_df.empty:
        return math.nan
    end = pd.Timestamp(end_date)
    start = end - pd.Timedelta(days=days - 1)
    values = daily_df.loc[
        daily_df["fecha"].between(start, end, inclusive="both"),
        "precipitacion_mm",
    ]
    return float(values.sum()) if not values.empty else math.nan


def latest_complete_month(latest_daily_date: date) -> pd.Period:
    current_period = pd.Period(latest_daily_date, freq="M")
    return current_period - 1


def safe_metric(value: float | None, suffix: str = "", decimals: int = 1) -> str:
    if value is None or not np.isfinite(value):
        return "N/D"
    return f"{value:,.{decimals}f}{suffix}"


def process_pending_selection() -> None:
    """Sincroniza una selección proveniente del mapa antes de crear el selectbox."""

    if "pending_selected_id" in st.session_state:
        pending = st.session_state.pop("pending_selected_id")
        st.session_state["selected_id"] = pending
        st.session_state["basin_dropdown"] = pending


# =============================================================================
# Aplicación
# =============================================================================


def main() -> None:
    apply_styles()
    process_pending_selection()

    project_id = initialize_earth_engine()

    try:
        with st.spinner("Cargando límites de subcuencas desde Earth Engine..."):
            geojson = load_lempa_geojson(project_id)
            latest_date = get_latest_chirps_date(project_id)
    except Exception as exc:
        st.error("No fue posible cargar los datos base del geoportal.")
        st.code(str(exc), language="text")
        st.stop()

    features_by_id = feature_index(geojson)
    basin_ids = sorted(features_by_id.keys(), key=int)
    valid_ids = set(basin_ids)

    if "selected_id" not in st.session_state:
        st.session_state.selected_id = None
    if "basin_dropdown" not in st.session_state:
        st.session_state.basin_dropdown = st.session_state.selected_id

    # -------------------------------------------------------------------------
    # Panel lateral
    # -------------------------------------------------------------------------
    with st.sidebar:
        st.title("🎛️ Panel operativo SAT")
        st.caption(f"Earth Engine conectado: `{project_id}`")
        st.caption(f"Último día CHIRPS disponible: **{latest_date:%d/%m/%Y}**")
        st.divider()

        selected_from_list = st.selectbox(
            "Seleccionar subcuenca",
            options=[None, *basin_ids],
            key="basin_dropdown",
            format_func=lambda value: (
                "— Haz clic en el mapa o selecciona un ID —"
                if value is None
                else f"HYBAS_ID {value}"
            ),
        )
        st.session_state.selected_id = selected_from_list

        selected_date = st.date_input(
            "Fecha de evaluación",
            value=latest_date,
            min_value=date(1981, 1, 1),
            max_value=latest_date,
            help="La fecha máxima corresponde al último dato disponible en CHIRPS.",
        )

        evaluation_mode = st.radio(
            "Tipo de evaluación de lluvia",
            options=["Observación CHIRPS", "Escenario manual"],
            index=0,
        )

        daily_window = st.select_slider(
            "Ventana del gráfico diario",
            options=[90, 180, 365, 730],
            value=180,
            format_func=lambda days: f"{days} días",
        )

        spi_scale = st.selectbox(
            "Escala para sequía",
            options=[1, 3, 6, 12],
            index=1,
            format_func=lambda months: f"SPI-{months}",
        )

        show_rain_layer = st.checkbox(
            "Mostrar lluvia del día sobre el mapa",
            value=False,
        )

        if st.button("Limpiar selección", width="stretch"):
            st.session_state.pending_selected_id = None
            st.rerun()

        st.divider()
        st.caption(
            "Los percentiles se calculan para la subcuenca seleccionada, el mismo "
            "mes calendario y el período de referencia 1991–2020, usando días con "
            f"precipitación ≥ {WET_DAY_THRESHOLD_MM:.0f} mm."
        )

    # -------------------------------------------------------------------------
    # Encabezado
    # -------------------------------------------------------------------------
    st.title("🚨 Sistema de Alerta Temprana | Cuenca del Río Lempa")
    
    st.markdown(
        """
        <div class="sat-subtitle">
            Geoportal trinacional con precipitación CHIRPS, umbrales
            hidroclimáticos locales, SPI y descargas de datos.
            <br>
            <span style="
                display: inline-block;
                margin-top: 6px;
                font-size: 0.88rem;
                color: #64748b;
            ">
                Elaborado por: <strong>Susana Melgar</strong> ·
                Especialista GIS · Programa Somos Río Lempa
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    selected_id = st.session_state.selected_id
    selected_feature = features_by_id.get(selected_id) if selected_id else None

    # Capa raster opcional
    rain_tile_url: str | None = None
    if selected_id and show_rain_layer:
        try:
            with st.spinner("Preparando capa de precipitación CHIRPS..."):
                rain_tile_url = get_chirps_tile_url(
                    selected_id,
                    selected_date.isoformat(),
                    project_id,
                )
        except Exception:
            rain_tile_url = None

    # KPIs generales. Los umbrales calculados aparecen después del mapa.
    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Subcuencas", f"{len(features_by_id)}", "HydroSHEDS nivel 12")
    top2.metric("Fuente de lluvia", "CHIRPS Daily", "Media espacial")
    top3.metric("Período de referencia", "1991–2020", "Percentiles y SPI")
    top4.metric("Último dato", latest_date.strftime("%d/%m/%Y"), "Disponibilidad CHIRPS")

    st.divider()
    st.subheader("📍 Selección territorial")
    st.markdown(
        "<div class='sat-note'><b>Haz clic dentro de una subcuenca.</b> También puedes "
        "seleccionarla por HYBAS_ID en el panel lateral. Después de la selección, "
        "el sistema calculará los indicadores y mostrará las pestañas de resultados "
        "debajo del mapa.</div>",
        unsafe_allow_html=True,
    )

    map_object = build_map(geojson, selected_id, rain_tile_url)
    map_data = st_folium(
        map_object,
        use_container_width=True,
        height=560,
        returned_objects=[
            "last_clicked",
            "last_object_clicked",
            "last_object_clicked_tooltip",
            "last_object_clicked_popup",
        ],
        key="rio_lempa_map_v2",
    )

    # Procesar únicamente eventos nuevos. streamlit-folium conserva el último clic
    # entre reruns; sin esta firma, una selección antigua podría sobrescribir el
    # selector lateral.
    event_payload = {
        key: map_data.get(key) if map_data else None
        for key in (
            "last_clicked",
            "last_object_clicked",
            "last_object_clicked_tooltip",
            "last_object_clicked_popup",
        )
    }
    event_signature = json.dumps(event_payload, sort_keys=True, default=str)
    previous_signature = st.session_state.get("last_map_event_signature")
    is_new_map_event = event_signature != previous_signature
    st.session_state.last_map_event_signature = event_signature

    clicked_id: str | None = None
    if is_new_map_event:
        # Primero intenta obtener el ID directamente del polígono.
        clicked_id = extract_hybas_id_from_map_event(map_data, valid_ids)

        # Respaldo: consulta espacial con las coordenadas del clic.
        if clicked_id is None and map_data:
            coordinate_event = map_data.get("last_clicked") or map_data.get(
                "last_object_clicked"
            )
            if isinstance(coordinate_event, dict):
                lat = coordinate_event.get("lat")
                lon = coordinate_event.get("lng")
                if lon is None:
                    lon = coordinate_event.get("lon")
                if lat is not None and lon is not None:
                    try:
                        clicked_id = find_subbasin_at(
                            round(float(lon), 6),
                            round(float(lat), 6),
                            project_id,
                        )
                    except Exception:
                        clicked_id = None

    if clicked_id and clicked_id != selected_id:
        st.session_state.pending_selected_id = clicked_id
        st.rerun()

    general_monitoring_tab, selected_analysis_tab = st.tabs(
        [
            "🚨 Monitoreo general",
            "📍 Análisis por subcuenca",
        ]
    )

    with general_monitoring_tab:
        render_general_monitoring_tab(
            project_id=project_id,
            selected_date=selected_date,
            latest_date=latest_date,
            spi_scale=int(spi_scale),
            expected_basin_count=len(features_by_id),
        )

    with selected_analysis_tab:
        if selected_id is None or selected_feature is None:
            st.info(
                "Selecciona una subcuenca. Cuando el ID quede seleccionado, aparecerán "
                "las pestañas **Resumen**, **Lluvia diaria**, **Climatología y SPI** y "
                "**Descargas**. La pestaña de monitoreo general funciona sin selección."
            )
            with st.expander("Metodología que aplicará el sistema"):
                st.markdown(
                    f"""
                    - Fuente de precipitación: **CHIRPS Daily**, banda `{CHIRPS_BAND}`.
                    - Resolución nominal: aproximadamente **5.6 km**.
                    - Umbrales P90, P95 y P99: percentiles de la precipitación diaria media
                      de la subcuenca durante **{REFERENCE_START_YEAR}–{REFERENCE_END_YEAR}**,
                      para el mismo mes calendario y considerando días húmedos
                      (≥ {WET_DAY_THRESHOLD_MM:.0f} mm).
                    - SPI: ajuste gamma por mes calendario y transformación a distribución
                      normal estándar.
                    """
                )
            return

        # -------------------------------------------------------------------------
        # Análisis dinámico de la subcuenca
        # -------------------------------------------------------------------------
        properties = selected_feature.get("properties", {})
        start_date = max(
            date(1981, 1, 1),
            selected_date - timedelta(days=int(daily_window) - 1),
        )

        try:
            with st.spinner(
                "Calculando precipitación, percentiles locales, climatología y SPI..."
            ):
                daily_df = get_daily_rainfall(
                    selected_id,
                    start_date.isoformat(),
                    selected_date.isoformat(),
                    project_id,
                )
                reference_df = get_month_reference_rainfall(
                    selected_id,
                    selected_date.month,
                    project_id,
                )
                thresholds = calculate_thresholds(reference_df, selected_date.month)

                end_month = latest_complete_month(latest_date)
                monthly_df = get_monthly_rainfall(
                    selected_id,
                    str(end_month),
                    project_id,
                )
                spi_df = calculate_spi(monthly_df, int(spi_scale))
        except Exception as exc:
            st.error("No fue posible completar el análisis hidroclimático.")
            st.code(str(exc), language="text")
            st.info(
                "Puedes limpiar la selección e intentar otra subcuenca. Si el error "
                "persiste, revisa los permisos o cuotas de Earth Engine."
            )
            st.stop()

        observation = selected_observation(daily_df, selected_date)
        if observation is None:
            observed_date = selected_date
            observed_rain = math.nan
        else:
            observed_date, observed_rain = observation

        if evaluation_mode == "Escenario manual":
            default_scenario = float(round(thresholds.p90, 1))
            maximum_scenario = float(max(200.0, math.ceil(thresholds.p99 * 1.8 / 10) * 10))
            manual_rain = st.sidebar.number_input(
                "Lluvia del escenario (mm/día)",
                min_value=0.0,
                max_value=maximum_scenario,
                value=min(default_scenario, maximum_scenario),
                step=1.0,
            )
            evaluated_rain = float(manual_rain)
            evaluation_label = f"Escenario manual: {evaluated_rain:.1f} mm/día"
        else:
            evaluated_rain = float(observed_rain)
            evaluation_label = (
                f"CHIRPS {observed_date:%d/%m/%Y}: {evaluated_rain:.1f} mm/día"
                if np.isfinite(evaluated_rain)
                else "Sin observación disponible"
            )

        # Sobrescribir KPIs con valores reales. Streamlit los muestra aquí como bloque final.
        st.markdown("#### Umbrales calculados para la selección")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("HYBAS_ID", selected_id, MONTH_NAMES[selected_date.month])
        k2.metric("P90 local", f"{thresholds.p90:.1f} mm/día", "Preventivo")
        k3.metric("P95 local", f"{thresholds.p95:.1f} mm/día", "Muy alto")
        k4.metric("P99 local", f"{thresholds.p99:.1f} mm/día", "Extremo")

        tab_summary, tab_daily, tab_climate, tab_downloads = st.tabs(
            [
                "📊 Resumen",
                "🌧️ Lluvia diaria y umbrales",
                "☀️ Climatología y SPI",
                "⬇️ Descargas",
            ]
        )

        # -------------------------------------------------------------------------
        # Resumen
        # -------------------------------------------------------------------------
        with tab_summary:
            st.subheader("Evaluación territorial")

            if np.isfinite(evaluated_rain):
                flood_alert = evaluate_flood_alert(evaluated_rain, thresholds)
                render_alert_card(flood_alert, selected_id, evaluation_label)
            else:
                flood_alert = None
                st.warning("No hay una observación CHIRPS disponible para la fecha evaluada.")

            area1, area2, acc3, acc7, acc30 = st.columns(5)
            area1.metric(
                "Área local",
                safe_metric(float(properties.get("SUB_AREA", math.nan)), " km²"),
            )
            area2.metric(
                "Área aguas arriba",
                safe_metric(float(properties.get("UP_AREA", math.nan)), " km²"),
            )
            acc3.metric(
                "Acumulado 3 días",
                safe_metric(sum_last_days(daily_df, observed_date, 3), " mm"),
            )
            acc7.metric(
                "Acumulado 7 días",
                safe_metric(sum_last_days(daily_df, observed_date, 7), " mm"),
            )
            acc30.metric(
                "Acumulado 30 días",
                safe_metric(sum_last_days(daily_df, observed_date, 30), " mm"),
            )

            latest_spi_rows = spi_df.loc[
                spi_df["fecha"] <= pd.Timestamp(selected_date),
            ].dropna(subset=["spi"])
            latest_spi = (
                float(latest_spi_rows.iloc[-1]["spi"])
                if not latest_spi_rows.empty
                else math.nan
            )
            latest_spi_date = (
                latest_spi_rows.iloc[-1]["fecha"].date()
                if not latest_spi_rows.empty
                else None
            )

            st.subheader(f"Condición de sequía — SPI-{spi_scale}")
            if np.isfinite(latest_spi):
                drought_alert = evaluate_drought_alert(latest_spi)
                render_alert_card(
                    drought_alert,
                    selected_id,
                    f"{latest_spi_date:%m/%Y} · SPI-{spi_scale} = {latest_spi:.2f}",
                )
            else:
                drought_alert = None
                st.info("No fue posible calcular SPI para el período seleccionado.")

            st.markdown(
                f"""
                <div class="method-card">
                    <b>Cómo se obtuvieron los umbrales:</b> se calculó la precipitación diaria
                    media espacial de la subcuenca para todos los días de
                    <b>{MONTH_NAMES[selected_date.month]}</b> entre
                    <b>{REFERENCE_START_YEAR} y {REFERENCE_END_YEAR}</b>. Después se conservaron
                    los días con lluvia ≥ {WET_DAY_THRESHOLD_MM:.0f} mm y se calcularon los
                    percentiles P90, P95 y P99. Muestra: <b>{thresholds.sample_wet}</b> días
                    húmedos de <b>{thresholds.sample_all}</b> días disponibles.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -------------------------------------------------------------------------
        # Lluvia diaria
        # -------------------------------------------------------------------------
        with tab_daily:
            st.plotly_chart(
                daily_rainfall_chart(daily_df, thresholds, observed_date),
                use_container_width=True,
                config={"displaylogo": False, "scrollZoom": False},
            )

            left_chart, right_chart = st.columns([1.6, 1])
            with left_chart:
                st.plotly_chart(
                    reference_distribution_chart(reference_df, thresholds),
                    use_container_width=True,
                    config={"displaylogo": False},
                )
            with right_chart:
                threshold_table = pd.DataFrame(
                    {
                        "Indicador": ["P90", "P95", "P99"],
                        "Valor_mm_dia": [
                            thresholds.p90,
                            thresholds.p95,
                            thresholds.p99,
                        ],
                        "Interpretación": [
                            "Preventivo",
                            "Muy alto",
                            "Extremo",
                        ],
                    }
                )
                st.markdown("#### Tabla de umbrales")
                st.dataframe(
                    threshold_table.style.format({"Valor_mm_dia": "{:.2f}"}),
                    use_container_width=True,
                    hide_index=True,
                )
                st.markdown("#### Datos diarios")
                table_daily = daily_df.copy()
                table_daily["fecha"] = table_daily["fecha"].dt.strftime("%Y-%m-%d")
                st.dataframe(
                    table_daily.tail(30),
                    use_container_width=True,
                    hide_index=True,
                    height=360,
                )

        # -------------------------------------------------------------------------
        # Climatología y SPI
        # -------------------------------------------------------------------------
        with tab_climate:
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.plotly_chart(
                    monthly_climatology_chart(monthly_df),
                    use_container_width=True,
                    config={"displaylogo": False},
                )
            with chart_col2:
                st.plotly_chart(
                    monthly_series_chart(monthly_df),
                    use_container_width=True,
                    config={"displaylogo": False},
                )

            st.plotly_chart(
                spi_chart(spi_df, int(spi_scale)),
                use_container_width=True,
                config={"displaylogo": False},
            )

            spi_table = spi_df.dropna(subset=["spi"]).tail(24).copy()
            spi_table["fecha"] = spi_table["fecha"].dt.strftime("%Y-%m")
            spi_table = spi_table[
                ["fecha", "precipitacion_mm", "acumulado_mm", "spi", "escala_meses"]
            ]
            st.dataframe(
                spi_table.style.format(
                    {
                        "precipitacion_mm": "{:.2f}",
                        "acumulado_mm": "{:.2f}",
                        "spi": "{:.2f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        # -------------------------------------------------------------------------
        # Descargas
        # -------------------------------------------------------------------------
        with tab_downloads:
            st.subheader("Descarga de resultados")

            thresholds_df = pd.DataFrame(
                [
                    {
                        "HYBAS_ID": selected_id,
                        "mes": selected_date.month,
                        "mes_nombre": MONTH_NAMES[selected_date.month],
                        "periodo_referencia": (
                            f"{REFERENCE_START_YEAR}-{REFERENCE_END_YEAR}"
                        ),
                        "criterio_dia_humedo_mm": WET_DAY_THRESHOLD_MM,
                        "muestra_total_dias": thresholds.sample_all,
                        "muestra_dias_humedos": thresholds.sample_wet,
                        "P90_mm_dia": thresholds.p90,
                        "P95_mm_dia": thresholds.p95,
                        "P99_mm_dia": thresholds.p99,
                    }
                ]
            )

            latest_spi_value = (
                latest_spi if np.isfinite(latest_spi) else None
            )
            summary_df = pd.DataFrame(
                [
                    {
                        "HYBAS_ID": selected_id,
                        "MAIN_BAS": properties.get("MAIN_BAS"),
                        "SUB_AREA_km2": properties.get("SUB_AREA"),
                        "UP_AREA_km2": properties.get("UP_AREA"),
                        "fecha_solicitada": selected_date.isoformat(),
                        "fecha_observada": observed_date.isoformat(),
                        "precipitacion_evaluada_mm_dia": (
                            evaluated_rain if np.isfinite(evaluated_rain) else None
                        ),
                        "modo_evaluacion": evaluation_mode,
                        "alerta_lluvia": flood_alert.level if flood_alert else None,
                        "SPI_escala_meses": spi_scale,
                        "SPI_fecha": (
                            latest_spi_date.isoformat() if latest_spi_date else None
                        ),
                        "SPI_valor": latest_spi_value,
                        "alerta_sequia": drought_alert.level if drought_alert else None,
                        "fuente": CHIRPS_ASSET,
                    }
                ]
            )

            export_daily = daily_df.copy()
            export_daily["HYBAS_ID"] = selected_id
            export_monthly = monthly_df.copy()
            export_monthly["HYBAS_ID"] = selected_id
            export_spi = spi_df.copy()
            export_spi["HYBAS_ID"] = selected_id

            files = {
                f"resumen_{selected_id}.csv": dataframe_csv_bytes(summary_df),
                f"umbrales_{selected_id}_{selected_date.month:02d}.csv": dataframe_csv_bytes(
                    thresholds_df
                ),
                f"precipitacion_diaria_{selected_id}.csv": dataframe_csv_bytes(
                    export_daily
                ),
                f"precipitacion_mensual_{selected_id}.csv": dataframe_csv_bytes(
                    export_monthly
                ),
                f"spi_{spi_scale}_{selected_id}.csv": dataframe_csv_bytes(export_spi),
                f"subcuenca_{selected_id}.geojson": selected_geojson_bytes(
                    selected_feature
                ),
            }

            st.download_button(
                "📦 Descargar paquete completo ZIP",
                data=create_download_zip(files),
                file_name=f"SAT_Rio_Lempa_{selected_id}.zip",
                mime="application/zip",
                type="primary",
                width="stretch",
                on_click="ignore",
            )

            download_columns = st.columns(3)
            with download_columns[0]:
                st.download_button(
                    "Resumen CSV",
                    dataframe_csv_bytes(summary_df),
                    file_name=f"resumen_{selected_id}.csv",
                    mime="text/csv",
                    width="stretch",
                    on_click="ignore",
                )
                st.download_button(
                    "Serie diaria CSV",
                    dataframe_csv_bytes(export_daily),
                    file_name=f"precipitacion_diaria_{selected_id}.csv",
                    mime="text/csv",
                    width="stretch",
                    on_click="ignore",
                )
            with download_columns[1]:
                st.download_button(
                    "Umbrales CSV",
                    dataframe_csv_bytes(thresholds_df),
                    file_name=f"umbrales_{selected_id}.csv",
                    mime="text/csv",
                    width="stretch",
                    on_click="ignore",
                )
                st.download_button(
                    "Serie mensual CSV",
                    dataframe_csv_bytes(export_monthly),
                    file_name=f"precipitacion_mensual_{selected_id}.csv",
                    mime="text/csv",
                    width="stretch",
                    on_click="ignore",
                )
            with download_columns[2]:
                st.download_button(
                    f"SPI-{spi_scale} CSV",
                    dataframe_csv_bytes(export_spi),
                    file_name=f"spi_{spi_scale}_{selected_id}.csv",
                    mime="text/csv",
                    width="stretch",
                    on_click="ignore",
                )
                st.download_button(
                    "Subcuenca GeoJSON",
                    selected_geojson_bytes(selected_feature),
                    file_name=f"subcuenca_{selected_id}.geojson",
                    mime="application/geo+json",
                    width="stretch",
                    on_click="ignore",
                )

            st.caption(
                "Los CSV se exportan con codificación UTF-8 con BOM para facilitar su "
                "apertura en Excel."
            )

        with st.expander("ℹ️ Alcance, datos y uso responsable"):
            st.markdown(
                f"""
                - **CHIRPS** es una estimación satelital combinada con estaciones; no sustituye
                  mediciones hidrometeorológicas locales ni pronósticos oficiales.
                - La precipitación mostrada es la **media espacial** de la subcuenca, no el
                  máximo puntual dentro de ella.
                - Los percentiles son estadísticos hidroclimáticos reales calculados con
                  CHIRPS, pero **no equivalen por sí solos a umbrales oficiales de inundación**.
                  Para operación institucional deben combinarse con niveles de río, humedad
                  antecedente, topografía, exposición, vulnerabilidad y protocolos nacionales.
                - El SPI se calcula con período de referencia
                  **{REFERENCE_START_YEAR}–{REFERENCE_END_YEAR}** y distribución gamma.
                - Delimitación: HydroSHEDS nivel 12. Fuente de lluvia: `{CHIRPS_ASSET}`.
                """
            )

if __name__ == "__main__":
    main()
