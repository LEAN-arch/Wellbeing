# config.py
import plotly.express as px
import pandas as pd # Needed for pd.NA comparisons

# --- General Configuration ---
APP_TITLE = "Laboral Vital Signs Dashboard"
APP_ICON = "📊" # Use an icon that's generally understandable

# --- Filter Defaults ---
# Use empty lists to show all options initially
DEFAULT_SITES = []
DEFAULT_REGIONS = []
DEFAULT_DEPARTMENTS = []
DEFAULT_FUNCTIONAL_CATEGORIES = []
DEFAULT_SHIFTS = []

# --- Color Palettes ---
# Recommended: Colorblind-safe and distinct palettes
COLOR_SCHEME_CATEGORICAL = px.colors.qualitative.Plotly # Or 'Safe', 'Vivid'
COLOR_SCHEME_SEQUENTIAL = px.colors.sequential.Viridis  # Good for intensity heatmaps if used later

COLOR_GREEN_SEMAFORO = "#2ECC71"  # Good/Low Risk/Positive
COLOR_YELLOW_SEMAFORO = "#F1C40F" # Warning/Moderate Risk
COLOR_RED_SEMAFORO = "#E74C3C"    # Critical/High Risk/Negative
COLOR_NEUTRAL_METRIC = "#3498DB"  # For neutral indicators or lines/bars without performance coloring
COLOR_GRAY_TEXT = "#7f8c8d"       # For secondary text, annotations, non-data ink
COLOR_TARGET_LINE = "#34495E"     # A distinct, stable color for targets (e.g., dark gray/blue)

# --- Thresholds (Adjust based on your organization's standards and the data scale) ---
# Make sure thresholds are numeric where used in calculations
# Laboral Stability - For Rotation Rate (higher is worse, e.g. %)
ROTATION_RATE_THRESHOLD_GOOD = 8.0
ROTATION_RATE_THRESHOLD_WARNING = 15.0
ROTATION_RATE_TARGET = 8.0 # Specific target line value

# Retention (higher is better, e.g. %)
RETENTION_THRESHOLD_GOOD = 90.0
RETENTION_THRESHOLD_WARNING = 75.0 # Warning if *below* this

# Safety Pulse (counts, higher is worse)
INCIDENTS_THRESHOLD_GOOD = 1.0    # Zero is ideal, but >=0 and <= this might be considered good
INCIDENTS_THRESHOLD_WARNING = 5.0 # Above GOOD up to WARNING

# Operational Stress (Scale 1-10 typical, higher is worse)
STRESS_LEVEL_THRESHOLD_LOW = 3.5    # Values <= this are "Low Stress" (Green)
STRESS_LEVEL_THRESHOLD_MEDIUM = 7.0 # Values > LOW and <= MEDIUM are "Moderate Stress" (Yellow)
STRESS_LEVEL_MAX_SCALE = 10.0      # Max value for stress scale display axis

# Engagement & Survey Scores (Higher is better, e.g., 1-5, % or scale specific)
ENPS_THRESHOLD_GOOD = 50.0
ENPS_THRESHOLD_WARNING = 10.0 # Warning if *below* this
CLIMATE_SCORE_THRESHOLD_GOOD = 80.0 # Example for a score out of 100
CLIMATE_SCORE_THRESHOLD_WARNING = 60.0 # Warning if *below* this
PARTICIPATION_THRESHOLD_GOOD = 80.0 # Example for survey participation %

# --- Placeholder Texts ---
PLACEHOLDER_TEXT_PLANT_MAP = """
### Interactive Facility Map
(Placeholder: This module will visualize data spatially, such as staff distribution or risk heatmaps. Future development requires integration with mapping libraries (e.g., Plotly Mapbox or Folium) and potentially real-time data feeds for actionable location-based insights. Ensuring accessibility for map interactions is crucial.)
"""

PLACEHOLDER_TEXT_AI_INSIGHTS = """
### Predictive AI Insights
(Placeholder: This module will provide data-driven forecasts on psychosocial risks, potentially based on aggregated Human Affect & Behavior Scores, predicting outcomes like burnout or turnover likelihood for areas/teams with confidence intervals. Implementation requires trained ML models (e.g., with scikit-learn or TensorFlow) and robust data pipelines. **Ensuring the fairness, transparency, and ethical implications of AI outputs is paramount.**)
"""

# --- Language Strings (Basic for ES/EN MVP - expand as needed) ---
LANG = "EN" # Default language code (e.g., "EN", "ES")

# Use full, clear labels and keys. Maintain consistency.
# Keys ending in '_label' or '_title' indicate text to be displayed in the UI or as plot titles.
TEXT_STRINGS = {
    "EN": {
        "dashboard_title": "Laboral Vital Signs Dashboard",
        "dashboard_subtitle": "A Human-Centered Intelligence System for Workplace Wellbeing and Organizational Performance.",
        "alignment_note": "Aligned with NOM-035-STPS-2018, ISO 45003, and DEI principles.",
        "filters_header": "Filters",
        "language_selector": "Language",
        "select_site": "Select Site(s):",
        "select_region": "Select Region(s):",
        "select_department": "Select Department(s):",
        "select_fc": "Select Functional Category (FC):",
        "select_shift": "Select Shift(s):",

        "stability_panel_title": "1. Laboral Stability Panel",
        "safety_pulse_title": "2. Safety Pulse Module",
        "engagement_title": "3. Employee Engagement & Commitment",
        "stress_title": "4. Operational Stress Dashboard",
        "plant_map_title": "5. Interactive Facility Map",
        "ai_insights_title": "6. Predictive AI Insights",

        # KPI & Metric Labels (used for metric cards and potentially gauge/radar titles)
        "rotation_rate_gauge": "Employee Rotation Rate (%)",
        "retention_6m_metric": "Retention (6 Months)",
        "retention_12m_metric": "Retention (12 Months)",
        "retention_18m_metric": "Retention (18 Months)",
        "days_without_accidents_metric": "Days Without Recordable Incidents",
        "active_safety_alerts_metric": "Active Safety Alerts",
        "labor_climate_score_metric": "Labor Climate Score", # E.g., 0-100 scale
        "enps_metric": "eNPS", # Employee Net Promoter Score
        "survey_participation_metric": "Survey Participation (%)",
        "recognitions_count_metric": "Recognitions Logged (Total)",
        "overtime_hours_metric": "Overtime Hours (Total)",
        "unfilled_shifts_metric": "Unfilled Shifts (Total)",
        "avg_stress_level_metric": "Average Stress Level", # E.g., 1-10 scale

        # Chart Titles
        "hires_vs_exits_chart_title": "Monthly Hires vs. Exits Trend",
        "monthly_incidents_chart_title": "Monthly Safety Incidents & Near Misses",
        "engagement_dimensions_radar_title": "Average Engagement Dimensions",
        "monthly_shift_load_chart_title": "Monthly Shift Load (Overtime & Unfilled)",
        "workload_vs_psych_chart_title": "Workload Perception vs. Psychological Signals",


        # General Terms / Labels
        "metrics_legend": "Metrics",
        "dimension_label": "Dimension", # Used in radar chart axis
        "score_label": "Score",         # Used in radar chart value axis
        "count_label": "Count",         # Used in bar chart value axis
        "hours_label": "Hours",
        "shifts_label": "Shifts",
        "month_axis_label": "Month",
        "date_time_axis_label": "Date/Time",
        "category_axis_label": "Category",
        "value_axis_label": "Value", # Generic value axis
        "people_count_label": "Number of People", # For hires/exits

        # KPI Performance Statuses / Levels
        "good_label": "Good",
        "warning_label": "Warning",
        "critical_label": "Critical",
        "low_label": "Low",       # Generic "Low", used for stress etc.
        "moderate_label": "Moderate", # Generic "Moderate"
        "high_label": "High",       # Generic "High", used for stress etc.
        "stress_na": "N/A",       # Not Applicable / Not Available for Stress Semáforo
        "days_label": "days",     # Plural
        "hours_label_singular": "hour", # Singular if needed
        "shift_label_singular": "shift",

        # Trend/Target Indicators
        "average_label": "Average",
        "target_label": "Target",
        "prev_period_label": "vs Prev", # For Metric Delta label prefix/suffix

        # Radar Dimensions (mapping internal keys to display labels)
        # Use underscores and match keys in ENGAGEMENT_RADAR_DATA_COLS
        "initiative_label": "Initiative",
        "punctuality_label": "Punctuality",
        "recognition_label": "Recognition",
        "feedback_label": "Feedback Culture", # Renamed from 'Feedback'

        # Stress Load Chart series names
        "overtime_label": "Overtime Hours",
        "unfilled_shifts_label": "Unfilled Shifts",

        # Stress Trend Chart series names (assuming mapped from col names)
        "workload_perception_label": "Workload Perception",
        "psychological_signals_label": "Psychological Signals",


        # Data Availability Messages
        "no_data_available": "No data available for the selected filters in this module.",
        "no_data_for_selection": "No data for current selection",
        "no_data_hires_exits": "Hires/Exits data or date column not found for trend chart.",
        "no_data_incidents_near_misses": "Incidents/Near Misses data or month column not found for chart.",
        "no_data_radar_columns": "Required columns for engagement radar chart are missing from the data.",
        "no_data_radar": "Data for radar chart dimensions is insufficient or unavailable for current selection.",
        "no_data_shift_load": "Shift load data or date column not found for chart.",
        "no_data_workload_psych": "Workload/Psychological signals data or date column not found for trend chart.",
        "error_loading_data": "Error loading data from file: {}",


        # Psychosocial Safety & Compliance Note
        "psych_safety_note": "Note: Data related to individual well-being and sensitive topics (stress, climate, sentiment) is processed and presented anonymously and aggregated where necessary to ensure employee psychological safety, privacy, and align with DEI principles and NOM-035 recommendations.",


        # Optional Modules & Future Vision
        "optional_modules_header": "Future Modules & Vision",
        "show_optional_modules": "Show Planned Modules (Placeholders)",
        "optional_modules_title": "Optional & Strategic Modules (Future Development)",
        "optional_modules_list": """
- **🌡️ Fatigue & Absenteeism Index:** Analyze attendance patterns and estimate fatigue risk based on working hours/shifts.
- **🎯 Goal Alignment & Purpose:** Visualize alignment between individual/team goals and organizational strategy.
- **🧭 Onboarding & Integration Success:** Track new hire sentiment, milestones, and early risk factors.
- **🔁 Organizational Trust & Psychological Safety:** Deeper metrics on open communication, trust levels, and reporting culture.
- **💬 Voice of Employee & Sentiment:** NLP analysis of feedback (survey comments, potentially other aggregated sources) for key themes and emotional trends.
- **🧠 Learning, Development & Growth:** Track training completion, skill progression, and career mobility.
- **🧯 Crisis & Trauma Response:** Monitor support utilization and post-incident recovery indicators (especially in high-risk environments).
- **🌐 Diversity, Equity & Inclusion (DEI):** Granular metrics on representation, inclusion sentiment, and fairness perceptions.
- **📡 Well-being Program Impact:** Evaluate the ROI and effectiveness of wellness initiatives by tracking utilization vs. health/performance outcomes.
        """,

    },
    "ES": {
        # Populate with Spanish translations for *ALL* English keys above.
        # Use _label, _metric, _title suffixes consistently.
        # Translate descriptions accurately.
        "dashboard_title": "Tablero de Signos Vitales Laborales",
        "dashboard_subtitle": "Un Sistema de Inteligencia Centrado en el Humano para el Bienestar Laboral y el Rendimiento Organizacional.",
        "alignment_note": "Alineado con NOM-035-STPS-2018, ISO 45003 y principios DEI.",
        "filters_header": "Filtros",
        "language_selector": "Idioma",
        "select_site": "Seleccionar Sitio(s):",
        "select_region": "Seleccionar Región(es):",
        "select_department": "Seleccionar Departamento(s):",
        "select_fc": "Seleccionar Categoría Funcional (CF):",
        "select_shift": "Seleccionar Turno(s):",

        "stability_panel_title": "1. Panel de Estabilidad Laboral",
        "safety_pulse_title": "2. Módulo de Pulso de Seguridad",
        "engagement_title": "3. Compromiso y Vinculación del Personal",
        "stress_title": "4. Tablero de Estrés Operacional",
        "plant_map_title": "5. Mapa Interactivo de Instalación",
        "ai_insights_title": "6. Perspectivas Predictivas con IA",

        "rotation_rate_gauge": "Tasa de Rotación de Personal (%)",
        "retention_6m_metric": "Retención (6 Meses)",
        "retention_12m_metric": "Retención (12 Meses)",
        "retention_18m_metric": "Retención (18 Meses)",
        "days_without_accidents_metric": "Días Sin Incidentes Registrables",
        "active_safety_alerts_metric": "Alertas de Seguridad Activas",
        "labor_climate_score_metric": "Clima Laboral (Puntuación)",
        "enps_metric": "eNPS (Net Promoter Score de Empleados)",
        "survey_participation_metric": "Participación en Encuestas (%)",
        "recognitions_count_metric": "Reconocimientos Registrados (Total)",
        "overtime_hours_metric": "Horas Extra (Total)",
        "unfilled_shifts_metric": "Turnos No Cubiertos (Total)",
        "avg_stress_level_metric": "Nivel Promedio de Estrés",

        "hires_vs_exits_chart_title": "Tendencia Mensual Contrataciones vs. Bajas",
        "monthly_incidents_chart_title": "Incidentes y Casi Incidentes Mensuales",
        "engagement_dimensions_radar_title": "Dimensiones Promedio del Compromiso",
        "monthly_shift_load_chart_title": "Carga de Turno Mensual (Horas Extra y Vacantes)",
        "workload_vs_psych_chart_title": "Percepción Carga Laboral vs. Señales Psicológicas",


        "metrics_legend": "Métricas",
        "dimension_label": "Dimensión",
        "score_label": "Puntuación",
        "count_label": "Cantidad",
        "hours_label": "Horas",
        "shifts_label": "Turnos",
        "month_axis_label": "Mes",
        "date_time_axis_label": "Fecha/Hora",
        "category_axis_label": "Categoría",
        "value_axis_label": "Valor",
        "people_count_label": "Número de Personas",

        "good_label": "Bueno",
        "warning_label": "Advertencia",
        "critical_label": "Crítico",
        "low_label": "Bajo",
        "moderate_label": "Moderado",
        "high_label": "Alto",
        "stress_na": "N/D",
        "days_label": "días",
        "hours_label_singular": "hora",
        "shift_label_singular": "turno",


        "average_label": "Promedio",
        "target_label": "Objetivo",
        "prev_period_label": "vs Periodo Ant.",

        "initiative_label": "Iniciativa",
        "punctuality_label": "Puntualidad",
        "recognition_label": "Reconocimiento",
        "feedback_label": "Cultura Retroalimentación",


        "overtime_label": "Horas Extra",
        "unfilled_shifts_label": "Turnos Vacantes",


        "workload_perception_label": "Percepción Carga Laboral",
        "psychological_signals_label": "Señales Psicológicas",


        "no_data_available": "No hay datos disponibles para los filtros seleccionados en este módulo.",
        "no_data_for_selection": "Sin datos para la selección actual",
        "no_data_hires_exits": "Datos de Contrataciones/Bajas o columna de fecha no encontrados para el gráfico de tendencia.",
        "no_data_incidents_near_misses": "Datos de Incidentes/Casi Incidentes o columna de mes no encontrados para el gráfico.",
        "no_data_radar_columns": "Faltan columnas requeridas para el gráfico radar de compromiso.",
        "no_data_radar": "Datos para las dimensiones del gráfico radar insuficientes o no disponibles para la selección actual.",
        "no_data_shift_load": "Datos de carga de turno o columna de fecha no encontrados para el gráfico.",
        "no_data_workload_psych": "Datos de carga laboral/señales psicológicas o columna de fecha no encontrados para el gráfico de tendencia.",
        "error_loading_data": "Error al cargar datos del archivo: {}",


        "psych_safety_note": "Nota: Los datos relacionados con el bienestar individual y temas sensibles (estrés, clima, sentimiento) se procesan y presentan de forma anónima y agregada cuando es necesario para garantizar la seguridad psicológica del personal, la privacidad y alinearse con los principios DEI y las recomendaciones de NOM-035.",


        "optional_modules_header": "Módulos Futuros y Visión",
        "show_optional_modules": "Mostrar Módulos Planeados (Marcadores de Posición)",
        "optional_modules_title": "Módulos Opcionales y Estratégicos (Desarrollo Futuro)",
        "optional_modules_list": """
- **🌡️ Índice de Fatiga y Ausentismo:** Analiza patrones de asistencia y estima el riesgo de fatiga basado en horas/turnos trabajados.
- **🎯 Alineación de Objetivos y Propósito:** Visualiza la alineación entre los objetivos individuales/de equipo y la estrategia organizacional.
- **🧭 Éxito en la Incorporación e Integración:** Seguimiento del sentimiento de nuevos empleados, hitos y factores de riesgo tempranos.
- **🔁 Confianza Organizacional y Seguridad Psicológica:** Métricas más profundas sobre comunicación abierta, niveles de confianza y cultura de reporte.
- **💬 Voz del Empleado y Análisis de Sentimiento:** Análisis de procesamiento de lenguaje natural (PLN) de comentarios abiertos para identificar temas clave y tendencias emocionales.
- **🧠 Aprendizaje, Desarrollo y Crecimiento:** Seguimiento de la finalización de capacitaciones, progresión de habilidades y movilidad profesional.
- **🧯 Respuesta a Crisis y Trauma:** Monitoreo de la utilización de apoyo y métricas de recuperación post-incidente (especialmente en entornos de alto riesgo).
- **🌐 Indicadores de Diversidad, Equidad e Inclusión (DEI):** Métricas granulares sobre representación, sentimiento de inclusión y percepción de equidad.
- **📡 Impacto del Programa de Bienestar:** Evaluación del ROI y efectividad de las iniciativas de bienestar mediante el seguimiento de la utilización vs. resultados de salud/rendimiento.
        """,
    }
}


# --- Data File Paths (Assumes CSVs are in the same directory as app.py) ---
STABILITY_DATA_FILE = "stability_data.csv"
SAFETY_DATA_FILE = "safety_data.csv"
ENGAGEMENT_DATA_FILE = "engagement_data.csv"
STRESS_DATA_FILE = "stress_data.csv"


# --- Column Name Mappings (Maps conceptual role to actual CSV column name) ---
# Using these mappings makes the code more readable and easier to adapt
# if CSV column names change later.

COLUMN_MAP = {
    "site": "site",
    "region": "region",
    "department": "department",
    "fc": "fc",
    "shift": "shift",
    "date": "date",     # Generic date column (used for time series trends)
    "month": "month",   # Generic month column (used for month-based summaries like incidents)

    # Stability
    "rotation_rate": "rotation_rate",
    "retention_6m": "retention_6m",
    "retention_12m": "retention_12m",
    "retention_18m": "retention_18m",
    "hires": "hires",
    "exits": "exits",

    # Safety
    "incidents": "incidents",
    "near_misses": "near_misses",
    "days_without_accidents": "days_without_accidents", # Likely a single org value or date calculation in reality
    "active_alerts": "active_alerts",             # Likely a single org value or from system count

    # Engagement
    # Raw data columns for radar dimensions
    "engagement_initiative_raw": "initiative", # Raw column names in the CSV
    "engagement_punctuality_raw": "punctuality",
    "engagement_recognition_raw": "recognition_data",
    "engagement_feedback_raw": "feedback_data",

    # Mapping raw radar data columns to conceptual/localization keys
    # Used by visualizations to know which raw columns map to which display labels
    "engagement_radar_dims": {
        "engagement_initiative_raw": "initiative_label",
        "engagement_punctuality_raw": "punctuality_label",
        "engagement_recognition_raw": "recognition_label",
        "engagement_feedback_raw": "feedback_label",
    },

    # Engagement Metrics
    "labor_climate_score": "labor_climate_score",
    "enps_score": "nps",
    "participation_rate": "participation", # e.g., survey participation rate
    "recognitions_count": "recognitions_count", # e.g., count of formal/informal recognitions

    # Operational Stress
    "overtime_hours": "overtime_hours",
    "unfilled_shifts": "unfilled_shifts",
    "stress_level_survey": "stress_level_survey", # E.g., average score from a survey
    "workload_perception": "workload_perception", # E.g., survey question score
    "psychological_signals": "psychological_signals", # E.g., a calculated score
}
