# config.py
import plotly.express as px
import pandas as pd # Needed for pd.NA comparisons

# --- General Configuration ---
APP_TITLE = "Laboral Vital Signs Dashboard"
APP_ICON = "📊"

# --- Filter Defaults ---
DEFAULT_SITES = []
DEFAULT_REGIONS = []
DEFAULT_DEPARTMENTS = []
DEFAULT_FUNCTIONAL_CATEGORIES = []
DEFAULT_SHIFTS = []

# --- Color Palettes ---
COLOR_SCHEME_CATEGORICAL = px.colors.qualitative.Plotly
COLOR_SCHEME_SEQUENTIAL = px.colors.sequential.Viridis

COLOR_GREEN_SEMAFORO = "#2ECC71"
COLOR_YELLOW_SEMAFORO = "#F1C40F"
COLOR_RED_SEMAFORO = "#E74C3C"
COLOR_NEUTRAL_METRIC = "#3498DB"
COLOR_GRAY_TEXT = "#7f8c8d"
COLOR_TARGET_LINE = "#34495E"

# --- Thresholds (Adjust based on your organization's standards) ---
# Laboral Stability - For Rotation Rate (higher is worse, e.g. %)
ROTATION_RATE_THRESHOLD_GOOD = 8.0
ROTATION_RATE_THRESHOLD_WARNING = 15.0
ROTATION_RATE_TARGET = 8.0

# Retention (higher is better, e.g. %)
RETENTION_THRESHOLD_GOOD = 90.0
RETENTION_THRESHOLD_WARNING = 75.0 # Warning if *below* this

# Safety Pulse (counts, higher is worse)
INCIDENTS_THRESHOLD_GOOD = 1.0
INCIDENTS_THRESHOLD_WARNING = 5.0

# Operational Stress (Scale 1-10 typical, higher is worse)
STRESS_LEVEL_THRESHOLD_LOW = 3.5
STRESS_LEVEL_THRESHOLD_MEDIUM = 7.0
STRESS_LEVEL_MAX_SCALE = 10.0

# Engagement & Survey Scores (Higher is better)
ENPS_THRESHOLD_GOOD = 50.0
ENPS_THRESHOLD_WARNING = 10.0
CLIMATE_SCORE_THRESHOLD_GOOD = 80.0
CLIMATE_SCORE_THRESHOLD_WARNING = 60.0
PARTICIPATION_THRESHOLD_GOOD = 80.0

# --- Placeholder Texts & UI Labels ---
PLACEHOLDER_TEXT_PLANT_MAP = """
### Interactive Facility Map
(Placeholder: This module will visualize data spatially, such as staff distribution or risk heatmaps. Future development requires integration with mapping libraries (e.g., Plotly Mapbox or Folium) and potentially real-time data feeds for actionable location-based insights. Ensuring accessibility for map interactions is crucial.)
"""

PLACEHOLDER_TEXT_AI_INSIGHTS = """
### Predictive AI Insights
(Placeholder: This module will provide data-driven forecasts on psychosocial risks, potentially based on aggregated Human Affect & Behavior Scores, predicting outcomes like burnout or turnover likelihood for areas/teams with confidence intervals. Implementation requires trained ML models (e.g., with scikit-learn or TensorFlow) and robust data pipelines. **Ensuring the fairness, transparency, and ethical implications of AI outputs is paramount.**)
"""

LANG = "EN" # Default language code

TEXT_STRINGS = {
    "EN": {
        "dashboard_title": "Laboral Vital Signs Dashboard",
        "dashboard_subtitle": "A Human-Centered Intelligence System for Workplace Wellbeing and Organizational Performance.",
        "alignment_note": "Aligned with NOM-035-STPS-2018, ISO 45003, and DEI principles.",
        "filters_header": "Filters",
        "language_selector": "Language", # This key will be used for the actual EN label
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
        "rotation_rate_gauge": "Employee Rotation Rate (%)", # Used as gauge title key
        "retention_6m_metric": "Retention (6 Months)", # Used as metric label key
        "retention_12m_metric": "Retention (12 Months)",
        "retention_18m_metric": "Retention (18 Months)",
        "days_without_accidents_metric": "Days Without Recordable Incidents",
        "active_safety_alerts_metric": "Active Safety Alerts",
        "labor_climate_score_metric": "Labor Climate Score",
        "enps_metric": "eNPS",
        "survey_participation_metric": "Survey Participation (%)",
        "recognitions_count_metric": "Recognitions Logged (Total)",
        "hires_vs_exits_chart_title": "Monthly Hires vs. Exits Trend",
        "monthly_incidents_chart_title": "Monthly Safety Incidents & Near Misses",
        "engagement_dimensions_radar_title": "Average Engagement Dimensions",
        "monthly_shift_load_chart_title": "Monthly Shift Load (Overtime & Unfilled)",
        "overall_stress_indicator_title": "Average Psychosocial Stress Level",
        "workload_vs_psych_chart_title": "Workload Perception vs. Psychological Signals",
        "metrics_legend": "Metrics",
        "dimension_label": "Dimension",
        "score_label": "Score",
        "count_label": "Count",
        "hours_label": "Hours",
        "shifts_label": "Shifts",
        "month_axis_label": "Month",
        "date_time_axis_label": "Date/Time",
        "category_axis_label": "Category",
        "value_axis_label": "Value",
        "people_count_label": "Number of People",
        "good_label": "Good",
        "warning_label": "Warning",
        "critical_label": "Critical",
        "low_label": "Low",
        "moderate_label": "Moderate",
        "high_label": "High",
        "stress_na": "N/A",
        "days_label": "days",
        "hours_label_singular": "hour",
        "shift_label_singular": "shift",
        "average_label": "Average",
        "target_label": "Target",
        "prev_period_label": "vs Prev",
        "initiative_label": "Initiative",
        "punctuality_label": "Punctuality",
        "recognition_label": "Recognition",
        "feedback_label": "Feedback Culture",
        "overtime_label": "Overtime Hours", # For bar chart labels
        "unfilled_shifts_label": "Unfilled Shifts", # For bar chart labels
        "hires_label": "Hires", # For trend chart labels
        "exits_label": "Exits", # For trend chart labels
        "incidents_label": "Incidents", # For bar chart labels
        "near_misses_label": "Near Misses", # For bar chart labels
        "workload_perception_label": "Workload Perception", # For trend chart
        "psychological_signals_label": "Psychological Signals", # For trend chart
        "no_data_available": "No data available for the selected filters in this module.",
        "no_data_for_selection": "No data for current selection",
        "no_data_hires_exits": "Hires/Exits data or date column not found for trend chart.",
        "no_data_incidents_near_misses": "Incidents/Near Misses data or month column not found for chart.",
        "no_data_radar_columns": "Required columns for engagement radar chart are missing from the data.",
        "no_data_radar": "Data for radar chart dimensions is insufficient or unavailable for current selection.",
        "no_data_shift_load": "Shift load data or date column not found for chart.",
        "no_data_workload_psych": "Workload/Psychological signals data or date column not found for trend chart.",
        "error_loading_data": "Error loading data from file: {}",
        "psych_safety_note": "Note: Individual well-being data is aggregated and presented anonymously to ensure psychological safety, privacy, and align with DEI principles and NOM-035 recommendations.",
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
        "enps_metric": "eNPS",
        "survey_participation_metric": "Participación en Encuestas (%)",
        "recognitions_count_metric": "Reconocimientos Registrados (Total)",
        "hires_vs_exits_chart_title": "Tendencia Mensual Contrataciones vs. Bajas",
        "monthly_incidents_chart_title": "Incidentes y Casi Incidentes Mensuales",
        "engagement_dimensions_radar_title": "Dimensiones Promedio del Compromiso",
        "monthly_shift_load_chart_title": "Carga de Turno Mensual (Horas Extra y Vacantes)",
        "overall_stress_indicator_title": "Nivel Promedio de Estrés Psicosocial",
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
        "prev_period_label": "vs Per. Ant.",
        "initiative_label": "Iniciativa",
        "punctuality_label": "Puntualidad",
        "recognition_label": "Reconocimiento",
        "feedback_label": "Cultura Retroalimentación",
        "overtime_label": "Horas Extra",
        "unfilled_shifts_label": "Turnos Vacantes",
        "hires_label": "Contrataciones",
        "exits_label": "Bajas",
        "incidents_label": "Incidentes",
        "near_misses_label": "Casi Incidentes",
        "workload_perception_label": "Percepción Carga Laboral",
        "psychological_signals_label": "Señales Psicológicas",
        "no_data_available": "No hay datos disponibles para los filtros seleccionados en este módulo.",
        "no_data_for_selection": "Sin datos para la selección actual.",
        "no_data_hires_exits": "Datos de Contrataciones/Bajas o columna de fecha no encontrados.",
        "no_data_incidents_near_misses": "Datos de Incidentes/Casi Incidentes o columna de mes no encontrados.",
        "no_data_radar_columns": "Faltan columnas requeridas para el gráfico radar de compromiso.",
        "no_data_radar": "Datos para las dimensiones del gráfico radar insuficientes o no disponibles.",
        "no_data_shift_load": "Datos de carga de turno o columna de fecha no encontrados.",
        "no_data_workload_psych": "Datos de carga laboral/señales psicológicas o columna de fecha no encontrados.",
        "error_loading_data": "Error al cargar datos del archivo: {}",
        "psych_safety_note": "Nota: Los datos relacionados con el bienestar individual y temas sensibles se procesan y presentan de forma anónima y agregada para garantizar la seguridad psicológica, la privacidad y alinearse con los principios DEI y NOM-035.",
        "optional_modules_header": "Módulos Futuros y Visión",
        "show_optional_modules": "Mostrar Módulos Planeados (Marcadores)",
        "optional_modules_title": "Módulos Opcionales y Estratégicos (Desarrollo Futuro)",
        "optional_modules_list": """
- **🌡️ Índice de Fatiga y Ausentismo:** Analizar patrones de asistencia y estimar el riesgo de fatiga.
- **🎯 Alineación de Objetivos y Propósito:** Visualizar la alineación de metas con la estrategia organizacional.
- **🧭 Éxito en la Incorporación e Integración:** Seguimiento del sentimiento de nuevos empleados y factores de riesgo.
- **🔁 Confianza Organizacional y Seguridad Psicológica:** Métricas de comunicación abierta y cultura de reporte.
- **💬 Voz del Empleado e Inteligencia de Sentimiento:** Análisis PLN de retroalimentación para temas y tendencias emocionales.
- **🧠 Aprendizaje, Desarrollo y Crecimiento:** Seguimiento de capacitaciones, progresión de habilidades y movilidad.
- **🧯 Respuesta a Crisis y Trauma:** Monitoreo del uso de apoyo y recuperación post-incidente.
- **🌐 Indicadores de Diversidad, Equidad e Inclusión (DEI):** Métricas de representación, sentimiento de inclusión y equidad.
- **📡 Impacto del Programa de Bienestar:** Evaluar ROI de iniciativas de bienestar vs. resultados de salud/rendimiento.
        """,
    }
}

# --- Data File Paths (CSVs in the same directory as app.py) ---
STABILITY_DATA_FILE = "stability_data.csv"
SAFETY_DATA_FILE = "safety_data.csv"
ENGAGEMENT_DATA_FILE = "engagement_data.csv"
STRESS_DATA_FILE = "stress_data.csv"

# --- Column Name Mappings ---
COLUMN_MAP = {
    # Standard Dimensions
    "site": "site", "region": "region", "department": "department", "fc": "fc", "shift": "shift",
    "date": "date", "month": "month",

    # Laboral Stability
    "rotation_rate": "rotation_rate", "retention_6m": "retention_6m",
    "retention_12m": "retention_12m", "retention_18m": "retention_18m",
    "hires": "hires", "exits": "exits",

    # Safety Pulse
    "incidents": "incidents", "near_misses": "near_misses",
    "days_without_accidents": "days_without_accidents", "active_alerts": "active_alerts",

    # Engagement & Commitment
    # These keys map to the actual column names in 'engagement_data.csv' for radar items
    "engagement_radar_dims_cols": {
        "initiative": "initiative", # "conceptual_metric_name": "csv_column_name"
        "punctuality": "punctuality",
        "recognition": "recognition_data",
        "feedback": "feedback_data",
    },
    # These keys map the conceptual metric names to their display label keys in TEXT_STRINGS
    "engagement_radar_dims_labels": {
        "initiative": "initiative_label",
        "punctuality": "punctuality_label",
        "recognition": "recognition_label",
        "feedback": "feedback_label",
    },
    # Other engagement metrics
    "labor_climate_score": "labor_climate_score",
    "enps_score": "nps", # Standardized key for Employee Net Promoter Score
    "participation_rate": "participation",
    "recognitions_count": "recognitions_count",

    # Operational Stress
    "overtime_hours": "overtime_hours", "unfilled_shifts": "unfilled_shifts",
    "stress_level_survey": "stress_level_survey",
    "workload_perception": "workload_perception",
    "psychological_signals": "psychological_signals",
}
