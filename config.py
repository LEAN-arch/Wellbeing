# config.py
import plotly.express as px

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

COLOR_GREEN_SEMAFORO = "#2ECC71"  # Good/Low Risk
COLOR_YELLOW_SEMAFORO = "#F1C40F" # Warning/Medium Risk
COLOR_RED_SEMAFORO = "#E74C3C"    # Critical/High Risk
COLOR_NEUTRAL_METRIC = "#3498DB"  # For neutral indicators or bars
COLOR_GRAY_TEXT = "#7f8c8d"       # For annotations, secondary text
COLOR_TARGET_LINE = "#E74C3C"     # Prominent color for target lines

# --- Thresholds (Adjust based on your organization's standards) ---
# Laboral Stability - For Rotation Rate (higher is worse)
ROTATION_RATE_THRESHOLD_GOOD = 8      # Values <= this are "Good" (Green)
ROTATION_RATE_THRESHOLD_WARNING = 15  # Values > GOOD and <= WARNING are "Warning" (Yellow)
                                      # Values > WARNING are "Critical" (Red)
# If only one critical threshold is needed for the gauge display, it could be:
ROTATION_RATE_TARGET = 8 # Often the "good" threshold is the target
ROTATION_RATE_CRITICAL = 20 # A specific critical point

# Retention (higher is better)
RETENTION_THRESHOLD_GOOD = 90     # Values >= this are "Good"
RETENTION_THRESHOLD_WARNING = 75  # Values < GOOD and >= WARNING are "Warning"
                                  # Values < WARNING are "Critical"

# Safety Pulse
INCIDENTS_THRESHOLD_GOOD = 1       # <= 1 is good
INCIDENTS_THRESHOLD_WARNING = 5    # > 1 and <= 5 is warning, > 5 is critical

# Operational Stress (Scale 1-10 typically, higher is worse)
STRESS_LEVEL_THRESHOLD_LOW = 3.5    # <= LOW is "Low Stress" (Green)
STRESS_LEVEL_THRESHOLD_MEDIUM = 7.0 # > LOW and <= MEDIUM is "Moderate Stress" (Yellow)
                                   # > MEDIUM is "High Stress" (Red)
STRESS_LEVEL_MAX_SCALE = 10.0      # Max value for stress scale display

# Engagement (Higher is better for these examples)
ENPS_THRESHOLD_GOOD = 50
ENPS_THRESHOLD_WARNING = 10
CLIMATE_SCORE_THRESHOLD_GOOD = 80
CLIMATE_SCORE_THRESHOLD_WARNING = 60
PARTICIPATION_THRESHOLD_GOOD = 80 # e.g. Survey Participation

# --- Placeholder Texts & UI Labels ---
# ... (TEXT_STRINGS remain the same as in the previous corrected version, including new keys)
# --- Language Strings ---
LANG = "EN" # Default language

TEXT_STRINGS = {
    "EN": {
        "dashboard_title": "Laboral Vital Signs Dashboard",
        "dashboard_subtitle": "A Human-Centered Intelligence System for Workplace Wellbeing and Organizational Performance.",
        "alignment_note": "Aligned with NOM-035, ISO 45003, and DEI principles.",
        "filters_header": "Filters",
        "language_selector": "Language / Idioma:",
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
        "rotation_rate_gauge": "Employee Rotation (%)",
        "retention_6m": "Retention (6 Months)",
        "retention_12m": "Retention (12 Months)",
        "retention_18m": "Retention (18 Months)",
        "hires_vs_exits_chart": "Monthly Hires vs. Exits Trend",
        "monthly_incidents_chart": "Monthly Incidents & Near Misses",
        "days_without_accidents_metric": "Days Without Recordable Incidents",
        "active_safety_alerts_metric": "Active Safety Alerts",
        "engagement_dimensions_radar": "Key Engagement Dimensions",
        "labor_climate_score_metric": "Labor Climate Score",
        "enps_metric": "eNPS (Employee Net Promoter Score)",
        "survey_participation_metric": "Survey Participation (%)",
        "recognitions_count_metric": "Recognitions Logged",
        "monthly_shift_load_chart": "Monthly Shift Load (Overtime & Unfilled)",
        "overall_stress_indicator_title": "Average Psychosocial Stress Level", # Title for the semaforo visual
        "stress_low": "Low",
        "stress_medium": "Moderate", # Changed from Medium
        "stress_high": "High",
        "stress_na": "N/A",
        "workload_vs_psych_chart": "Workload Perception vs. Psychological Signals",
        "no_data_available": "No data available for the selected filters in this module.",
        "no_data_for_selection": "No data for current selection", # Generic for empty charts
        "no_data_hires_exits": "Hires/Exits data or date column not found for trend chart.",
        "no_data_incidents_near_misses": "Incidents/Near Misses data or month column not found for chart.",
        "no_data_radar_columns": "Required columns for engagement radar chart are missing from the data.",
        "no_data_radar": "Data for radar chart dimensions is insufficient or unavailable for current selection.",
        "no_data_shift_load": "Shift load data or date column not found for chart.",
        "no_data_workload_psych": "Workload/Psychological signals data or date column not found for trend chart.",
        "psych_safety_note": "Note: Individual well-being data is aggregated and presented anonymously to ensure psychological safety and privacy, aligning with DEI principles.",
        "metrics_legend": "Metrics",
        "month_axis": "Month",
        "date_time_axis": "Date/Time",
        "category_axis": "Category",
        "value_axis": "Value",
        "count_axis": "Count",
        "days_label": "days",
        "hours_or_shifts_label": "Hours / Count",
        "average_score_label": "Average Score",
        "optional_modules_header": "Optional Modules (Future Vision)",
        "show_optional_modules": "Show Optional Modules (Placeholders)",
        "optional_modules_title": "Optional & Strategic Modules (Future Development)",
        "optional_modules_list": """
- Fatigue Index & Absenteeism Monitor
- Goal Alignment & Purpose Module
- Onboarding & Integration Success
- Organizational Trust & Psychological Safety
- Voice of Employee & Sentiment Intelligence
- Learning, Development & Growth
- Crisis & Trauma Response Module
- Diversity, Equity & Inclusion (DEI) Indicators
- Well-being Program Impact Tracker
        """,
        "initiative_label": "Initiative",
        "punctuality_label": "Punctuality",
        "recognition_label": "Recognition",
        "feedback_label": "Feedback Culture",
        "good_label": "Good",
        "warning_label": "Warning",
        "critical_label": "Critical",
        "average_label": "Avg.",
        "target_label": "Target",
        "thresholds_label": "Thresholds",
        "low_label": "Low",
        "high_label": "High", # Generic High
        "rotation_gauge_caption": "Lower rotation is generally better. Green: ≤{good}%, Yellow: ≤{warn}%, Red: >{warn}%. Target: {target}%.",
        "stress_semaforo_caption": "Scale 1-{max_scale}. Low: ≤{low:.1f}, Moderate: >{low:.1f} & ≤{medium:.1f}, High: >{medium:.1f}"
    },
    "ES": {
        # ... (Ensure all new keys from EN are translated here)
        "dashboard_title": "Tablero de Signos Vitales Laborales",
        "dashboard_subtitle": "Un Sistema de Inteligencia Centrado en el Humano para el Bienestar Laboral y el Rendimiento Organizacional.",
        "alignment_note": "Alineado con NOM-035, ISO 45003 y principios DEI.",
        "filters_header": "Filtros",
        "language_selector": "Idioma / Language:",
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
        "rotation_rate_gauge": "Rotación de Personal (%)",
        "retention_6m": "Retención (6 Meses)",
        "retention_12m": "Retención (12 Meses)",
        "retention_18m": "Retención (18 Meses)",
        "hires_vs_exits_chart": "Tendencia Mensual de Contrataciones vs. Bajas",
        "monthly_incidents_chart": "Incidentes y Casi Incidentes Mensuales",
        "days_without_accidents_metric": "Días Sin Incidentes Registrables",
        "active_safety_alerts_metric": "Alertas de Seguridad Activas",
        "engagement_dimensions_radar": "Dimensiones Clave del Compromiso",
        "labor_climate_score_metric": "Clima Laboral (Puntuación)",
        "enps_metric": "eNPS (Net Promoter Score de Empleados)",
        "survey_participation_metric": "Participación en Encuestas (%)",
        "recognitions_count_metric": "Reconocimientos Registrados",
        "monthly_shift_load_chart": "Carga de Turno Mensual (Horas Extra y Vacantes)",
        "overall_stress_indicator_title": "Nivel Promedio de Estrés Psicosocial",
        "stress_low": "Bajo",
        "stress_medium": "Moderado",
        "stress_high": "Alto",
        "stress_na": "N/D",
        "workload_vs_psych_chart": "Percepción de Carga Laboral vs. Señales Psicológicas",
        "no_data_available": "No hay datos disponibles para los filtros seleccionados en este módulo.",
        "no_data_for_selection": "Sin datos para la selección actual",
        "no_data_hires_exits": "Datos de Contrataciones/Bajas o columna de fecha no encontrados para el gráfico de tendencia.",
        "no_data_incidents_near_misses": "Datos de Incidentes/Casi Incidentes o columna de mes no encontrados para el gráfico.",
        "no_data_radar_columns": "Faltan columnas requeridas para el gráfico radar de compromiso.",
        "no_data_radar": "Datos para las dimensiones del gráfico radar insuficientes o no disponibles para la selección actual.",
        "no_data_shift_load": "Datos de carga de turno o columna de fecha no encontrados para el gráfico.",
        "no_data_workload_psych": "Datos de carga laboral/señales psicológicas o columna de fecha no encontrados para el gráfico de tendencia.",
        "psych_safety_note": "Nota: Los datos relacionados con el bienestar individual se presentan de forma agregada y anonimizada para garantizar la seguridad psicológica y la privacidad, en alineación con los principios DEI.",
        "metrics_legend": "Métricas",
        "month_axis": "Mes",
        "date_time_axis": "Fecha/Hora",
        "category_axis": "Categoría",
        "value_axis": "Valor",
        "count_axis": "Cantidad",
        "days_label": "días",
        "hours_or_shifts_label": "Horas / Cantidad",
        "average_score_label": "Puntuación Promedio",
        "optional_modules_header": "Módulos Opcionales (Visión Futura)",
        "show_optional_modules": "Mostrar Módulos Opcionales (Marcadores de Posición)",
        "optional_modules_title": "Módulos Opcionales y Estratégicos (Desarrollo Futuro)",
        "optional_modules_list": """
- Índice de Fatiga y Ausentismo
- Módulo de Alineación de Objetivos y Propósito
- Éxito en la Incorporación e Integración
- Confianza Organizacional y Seguridad Psicológica
- Voz del Empleado e Inteligencia de Sentimiento
- Aprendizaje, Desarrollo y Crecimiento
- Módulo de Respuesta a Crisis y Trauma
- Indicadores de Diversidad, Equidad e Inclusión (DEI)
- Seguimiento del Impacto del Programa de Bienestar
        """,
        "initiative_label": "Iniciativa",
        "punctuality_label": "Puntualidad",
        "recognition_label": "Reconocimiento",
        "feedback_label": "Cultura de Retroalimentación",
        "good_label": "Bueno",
        "warning_label": "Advertencia",
        "critical_label": "Crítico",
        "average_label": "Prom.",
        "target_label": "Objetivo",
        "thresholds_label": "Umbrales",
        "low_label": "Bajo",
        "high_label": "Alto",
        "rotation_gauge_caption": "Una rotación más baja es generalmente mejor. Verde: ≤{good}%, Amarillo: ≤{warn}%, Rojo: >{warn}%. Objetivo: {target}%.",
        "stress_semaforo_caption": "Escala 1-{max_scale}. Bajo: ≤{low:.1f}, Moderado: >{low:.1f} & ≤{medium:.1f}, Alto: >{medium:.1f}"
    }
}


# --- Data File Paths ---
STABILITY_DATA_FILE = "stability_data.csv"
SAFETY_DATA_FILE = "safety_data.csv"
ENGAGEMENT_DATA_FILE = "engagement_data.csv"
STRESS_DATA_FILE = "stress_data.csv"

# --- Column Names (Ensure these match your CSV headers) ---
COLUMN_SITE = "site"
COLUMN_REGION = "region"
COLUMN_DEPARTMENT = "department"
COLUMN_FC = "fc"
COLUMN_SHIFT = "shift"
COLUMN_DATE = "date"
COLUMN_MONTH = "month"

COLUMN_ROTATION_RATE = "rotation_rate"
COLUMN_RETENTION_6M = "retention_6m"
COLUMN_RETENTION_12M = "retention_12m"
COLUMN_RETENTION_18M = "retention_18m"
COLUMN_HIRES = "hires"
COLUMN_EXITS = "exits"

COLUMN_INCIDENTS = "incidents"
COLUMN_NEAR_MISSES = "near_misses"
COLUMN_DAYS_WITHOUT_ACCIDENTS = "days_without_accidents"
COLUMN_ACTIVE_ALERTS = "active_alerts"

ENGAGEMENT_RADAR_DATA_COLS = {
    "initiative": "initiative",
    "punctuality": "punctuality",
    "recognition": "recognition_data",
    "feedback": "feedback_data"
}
ENGAGEMENT_RADAR_LABELS_KEYS = {
    "initiative": "initiative_label",
    "punctuality": "punctuality_label",
    "recognition": "recognition_label",
    "feedback": "feedback_label"
}
COLUMN_LABOR_CLIMATE = "labor_climate_score"
COLUMN_ENPS = "nps"
COLUMN_PARTICIPATION = "participation"
COLUMN_RECOGNITIONS_COUNT = "recognitions_count"

COLUMN_OVERTIME_HOURS = "overtime_hours"
COLUMN_UNFILLED_SHIFTS = "unfilled_shifts"
COLUMN_STRESS_LEVEL_SURVEY = "stress_level_survey"
COLUMN_WORKLOAD_PERCEPTION = "workload_perception"
COLUMN_PSYCH_SIGNAL_SCORE = "psychological_signals"
COLUMN_STRESS_LEVEL_SURVEY = "stress_level_survey"
COLUMN_WORKLOAD_PERCEPTION = "workload_perception"
COLUMN_PSYCH_SIGNAL_SCORE = "psychological_signals"
