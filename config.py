# config.py
import plotly.express as px
import pandas as pd

# --- GENERAL APPLICATION SETTINGS ---
APP_VERSION = "v0.8.0 (UX/DX Enhanced)"
APP_TITLE_KEY = "dashboard_title" # Key for localized title
APP_ICON = "❤️‍🩹" # Icon reflecting wellbeing

# --- FILTER DEFAULTS ---
DEFAULT_SITES = []
DEFAULT_REGIONS = []
DEFAULT_DEPARTMENTS = []
DEFAULT_FUNCTIONAL_CATEGORIES = []
DEFAULT_SHIFTS = []

# --- VISUALIZATION & THEME ---
# (Keep color scheme choices, they are good for accessibility)
COLOR_SCHEME_CATEGORICAL = px.colors.qualitative.Plotly
COLOR_SCHEME_SEQUENTIAL = px.colors.sequential.Viridis

COLOR_STATUS_GOOD = "#2ECC71"      # Renamed from GREEN_SEMAFORO
COLOR_STATUS_WARNING = "#F1C40F" # Renamed from YELLOW_SEMAFORO
COLOR_STATUS_CRITICAL = "#E74C3C" # Renamed from RED_SEMAFORO
COLOR_NEUTRAL_INFO = "#3498DB"    # Renamed from NEUTRAL_METRIC
COLOR_TEXT_SECONDARY = "#7f8c8d"  # Renamed from GRAY_TEXT
COLOR_TARGET_LINE = "#2c3e50"     # Darker, more professional target line

# --- KPI THRESHOLDS & TARGETS ---
# Group thresholds by module or KPI for clarity
# Laboral Stability
STABILITY_ROTATION_RATE = {
    "good": 8.0,    # Values <= good are "good"
    "warning": 15.0, # Values > good and <= warning are "warning"
    "target": 8.0   # Explicit target line for gauge
    # Critical is implied > warning
}
STABILITY_RETENTION = { # For 6, 12, 18 month metrics
    "good": 90.0,   # Values >= good are "good"
    "warning": 75.0 # Values < good and >= warning are "warning"
    # Critical is implied < warning
}

# Safety Pulse
SAFETY_INCIDENTS = { # For incidents count (lower is better)
    "good": 1.0,
    "warning": 5.0,
    "target": 0.0
}
SAFETY_DAYS_NO_INCIDENTS = { # Higher is better
    "good": 180,
    "warning": 90 # Warning if below this
}

# Operational Stress (Example scale 1-10, higher is worse)
STRESS_LEVEL_PSYCHOSOCIAL = {
    "low": 3.5,         # Equivalent to "good"
    "medium": 7.0,      # Equivalent to "warning"
    "max_scale": 10.0
}

# Engagement & Survey Scores
ENGAGEMENT_ENPS = { # Higher is better
    "good": 50.0,
    "warning": 10.0 # Warning if below this
}
ENGAGEMENT_CLIMATE_SCORE = { # Higher is better
    "good": 80.0, # Out of 100
    "warning": 60.0
}
ENGAGEMENT_PARTICIPATION = { # Survey participation %, higher is better
    "good": 80.0
    # No specific warning, just aim for high participation
}
ENGAGEMENT_RADAR_DIM_TARGET = 4.0 # Example generic target for 1-5 scale radar dimensions

# --- COLUMN MAPPING (Conceptual Keys to Actual CSV Column Names) ---
# This structure is good for DX. Ensure it's comprehensive.
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
    "engagement_radar_dims_cols": { # Maps conceptual dimension to actual CSV column
        "initiative": "initiative",
        "punctuality": "punctuality",
        "recognition": "recognition_data",
        "feedback": "feedback_data",
    },
    "engagement_radar_dims_labels": { # Maps conceptual dimension to its TEXT_STRINGS label key
        "initiative": "initiative_label",
        "punctuality": "punctuality_label",
        "recognition": "recognition_label",
        "feedback": "feedback_label",
    },
    "labor_climate_score": "labor_climate_score",
    "enps_score": "nps",
    "participation_rate": "participation",
    "recognitions_count": "recognitions_count",

    # Operational Stress
    "overtime_hours": "overtime_hours", "unfilled_shifts": "unfilled_shifts",
    "stress_level_survey": "stress_level_survey", # Main value for the semaforo
    "workload_perception": "workload_perception",
    "psychological_signals": "psychological_signals",
}

# --- TEXT STRINGS FOR LOCALIZATION (i18n) ---
DEFAULT_LANG = "EN" # Default language if selected one fails

# UX: Use more descriptive keys. Group by module/context.
TEXT_STRINGS = {
    "EN": {
        # General App
        "app_title": APP_TITLE_KEY, # Allows title to be separate if needed
        "dashboard_title": "Laboral Vital Signs Dashboard",
        "dashboard_subtitle": "A Human-Centered Intelligence System for Workplace Wellbeing and Organizational Performance.",
        "alignment_note": "Aligned with NOM-035, ISO 45003, and DEI principles.",
        "psych_safety_note": "Note: Data concerning individual wellbeing is aggregated and presented anonymously to uphold psychological safety, privacy, and align with DEI principles and regulatory standards like NOM-035.",
        "error_loading_data": "Error loading data from file: {}",
        "filters_header": "Dashboard Filters",
        "language_selector": "Language",
        "optional_modules_header": "Future Vision: Extended Modules",
        "show_optional_modules": "Show Planned Features",
        "optional_modules_title": "Planned Strategic & Operational Modules",
        "no_data_available": "No data to display for the current filter selection in this module.",
        "no_data_for_selection": "No data available for current selection", # Generic for charts

        # Filter Labels
        "select_site": "Site(s):", "select_region": "Region(s):",
        "select_department": "Department(s):", "select_fc": "Functional Category (FC):",
        "select_shift": "Shift(s):",

        # Common Chart/Metric Elements
        "metrics_legend": "Legend",
        "average_label": "Avg.", "target_label": "Target",
        "prev_period_label_short": "vs Prev.", # For compact delta

        # Axis Labels
        "month_axis_label": "Month", "date_time_axis_label": "Date / Time",
        "category_axis_label": "Category", "value_axis_label": "Value",
        "count_label": "Count", "people_count_label": "Number of Employees",
        "hours_label": "Hours", "shifts_label": "Shifts",
        "percentage_label": "Percentage (%)", "score_label": "Score",
        "average_score_label": "Average Score",
        "hours_or_shifts_label": "Hours / Count",

        # Status/Threshold Labels
        "good_label": "Good", "warning_label": "Warning", "critical_label": "Critical",
        "low_label": "Low", "moderate_label": "Moderate", "high_label": "High",
        "status_na_label": "N/A", "days_unit": "days",

        # Laboral Stability
        "stability_panel_title": "1. Laboral Stability",
        "rotation_rate_gauge": "Employee Rotation Rate", # Unit is added by gauge function
        "rotation_rate_metric_help": "Percentage of employees who left the organization during a defined period.",
        "retention_6m_metric": "6-Month Retention",
        "retention_12m_metric": "12-Month Retention",
        "retention_18m_metric": "18-Month Retention",
        "retention_metric_help": "Percentage of new hires remaining after the specified period.",
        "hires_vs_exits_chart_title": "Monthly Hiring and Attrition Trends",
        "hires_label": "New Hires", "exits_label": "Departures", # For legend/labels

        # Safety Pulse
        "safety_pulse_title": "2. Safety Pulse",
        "monthly_incidents_chart_title": "Monthly Safety Events",
        "incidents_label": "Incidents", "near_misses_label": "Near Misses",
        "days_without_accidents_metric": "Days Since Last Recordable Incident",
        "days_no_incident_help": "Continuously tracks days without a major safety incident. Resets on occurrence.",
        "active_safety_alerts_metric": "Open Safety Alerts",

        # Employee Engagement
        "engagement_title": "3. Employee Engagement & Commitment",
        "engagement_dimensions_radar_title": "Key Engagement Dimensions",
        "initiative_label": "Initiative", "punctuality_label": "Punctuality",
        "recognition_label": "Recognition Culture", "feedback_label": "Feedback Environment",
        "labor_climate_score_metric": "Work Climate Index",
        "enps_metric": "eNPS (Employee Net Promoter Score)",
        "enps_metric_help": "Likelihood of employees to recommend the organization as a great place to work.",
        "survey_participation_metric": "Survey Response Rate",
        "recognitions_count_metric": "Total Recognitions",

        # Operational Stress
        "stress_title": "4. Operational Stress & Workload",
        "overall_stress_indicator_title": "Current Psychosocial Stress Index",
        "stress_indicator_help": "Aggregated measure of workplace stress based on recent survey data or other indicators (scale 0-10).",
        "monthly_shift_load_chart_title": "Monthly Operational Load",
        "overtime_label": "Overtime Hours", "unfilled_shifts_label": "Unfilled Shifts",
        "workload_vs_psych_chart_title": "Workload Perception vs. Psychological Signals Trend",
        "workload_perception_label": "Perceived Workload",
        "psychological_signals_label": "Wellbeing Signals", # E.g., inverse of negative sentiment or stress symptoms

        # Placeholders
        "plant_map_title": "5. Interactive Facility Overview",
        "ai_insights_title": "6. Predictive Risk Insights",
        
        # Placeholder texts can remain similar but ensure markdown compatibility for lists etc.
        "optional_modules_list": """
- **🌡️ Fatigue & Absenteeism Index:** Identify patterns leading to burnout.
- **🎯 Goal Alignment & Purpose:** Gauge connection to organizational mission.
- **🧭 Onboarding Experience:** Monitor early integration and satisfaction.
- **🔁 Trust & Psychological Safety:** Measure open communication climate.
- **💬 Voice of Employee (VoE) & Sentiment:** Analyze feedback trends.
- **🧠 Learning & Growth Opportunities:** Track skill development engagement.
- **🧯 Crisis Response Readiness:** Assess post-incident support effectiveness.
- **🌐 DEI Dashboard:** Monitor representation, inclusion, and equity perception.
- **📡 Wellbeing Program ROI:** Measure impact of wellness initiatives.
"""
        # ... many other specific labels as needed ...
    },
    "ES": {
        # General App
        "dashboard_title": "Tablero de Signos Vitales Laborales",
        "dashboard_subtitle": "Un Sistema de Inteligencia Centrado en el Ser Humano para el Bienestar y Rendimiento Organizacional.",
        "alignment_note": "Alineado con NOM-035-STPS-2018, ISO 45003 y principios DEI.",
        "psych_safety_note": "Nota: Los datos sobre el bienestar individual y temas sensibles se procesan y presentan de forma anónima y agregada para garantizar la seguridad psicológica del personal, la privacidad y cumplir con los principios DEI y normativas como NOM-035.",
        "error_loading_data": "Error al cargar datos del archivo: {}",
        "filters_header": "Filtros del Tablero",
        "language_selector": "Idioma",
        "optional_modules_header": "Visión Futura: Módulos Extendidos",
        "show_optional_modules": "Mostrar Funcionalidades Planeadas",
        "optional_modules_title": "Módulos Estratégicos y Operativos Planeados",
        "no_data_available": "No hay datos disponibles para la selección de filtros actual en este módulo.",
        "no_data_for_selection": "Sin datos para la selección actual",

        # Filter Labels
        "select_site": "Sitio(s):", "select_region": "Región(es):",
        "select_department": "Departamento(s):", "select_fc": "Categoría Funcional (CF):",
        "select_shift": "Turno(s):",

        # Common Chart/Metric Elements
        "metrics_legend": "Leyenda",
        "average_label": "Prom.", "target_label": "Objetivo",
        "prev_period_label_short": "vs Per. Ant.",

        # Axis Labels
        "month_axis_label": "Mes", "date_time_axis_label": "Fecha / Hora",
        "category_axis_label": "Categoría", "value_axis_label": "Valor",
        "count_label": "Cantidad", "people_count_label": "Número de Empleados",
        "hours_label": "Horas", "shifts_label": "Turnos",
        "percentage_label": "Porcentaje (%)", "score_label": "Puntuación",
        "average_score_label": "Puntuación Promedio",
        "hours_or_shifts_label": "Horas / Cantidad",

        # Status/Threshold Labels
        "good_label": "Bueno", "warning_label": "Advertencia", "critical_label": "Crítico",
        "low_label": "Bajo", "moderate_label": "Moderado", "high_label": "Alto",
        "status_na_label": "N/D", "days_unit": "días",

        # Laboral Stability
        "stability_panel_title": "1. Estabilidad Laboral",
        "rotation_rate_gauge": "Tasa de Rotación de Personal",
        "rotation_rate_metric_help": "Porcentaje de empleados que dejaron la organización durante un período definido.",
        "retention_6m_metric": "Retención a 6 Meses",
        "retention_12m_metric": "Retención a 12 Meses",
        "retention_18m_metric": "Retención a 18 Meses",
        "retention_metric_help": "Porcentaje de nuevas contrataciones que permanecen después del período especificado.",
        "hires_vs_exits_chart_title": "Tendencias Mensuales de Contratación y Desvinculación",
        "hires_label": "Nuevas Contrataciones", "exits_label": "Bajas",

        # Safety Pulse
        "safety_pulse_title": "2. Pulso de Seguridad",
        "monthly_incidents_chart_title": "Eventos de Seguridad Mensuales",
        "incidents_label": "Incidentes", "near_misses_label": "Casi Incidentes",
        "days_without_accidents_metric": "Días Desde Último Incidente Registrable",
        "days_no_incident_help": "Realiza un seguimiento continuo de los días sin incidentes de seguridad graves. Se reinicia al ocurrir uno.",
        "active_safety_alerts_metric": "Alertas de Seguridad Abiertas",

        # Employee Engagement
        "engagement_title": "3. Compromiso y Vinculación del Personal",
        "engagement_dimensions_radar_title": "Dimensiones Clave de Compromiso",
        "initiative_label": "Iniciativa", "punctuality_label": "Puntualidad",
        "recognition_label": "Cultura de Reconocimiento", "feedback_label": "Entorno de Retroalimentación",
        "labor_climate_score_metric": "Índice de Clima Laboral",
        "enps_metric": "eNPS (Net Promoter Score de Empleados)",
        "enps_metric_help": "Probabilidad de que los empleados recomienden la organización como un excelente lugar para trabajar.",
        "survey_participation_metric": "Tasa de Respuesta a Encuestas",
        "recognitions_count_metric": "Total de Reconocimientos",

        # Operational Stress
        "stress_title": "4. Estrés Operacional y Carga Laboral",
        "overall_stress_indicator_title": "Índice Actual de Estrés Psicosocial",
        "stress_indicator_help": "Medida agregada del estrés laboral basada en datos recientes de encuestas u otros indicadores (escala 0-10).",
        "monthly_shift_load_chart_title": "Carga Operacional Mensual",
        "overtime_label": "Horas Extra", "unfilled_shifts_label": "Turnos Sin Cubrir",
        "workload_vs_psych_chart_title": "Tendencia: Percepción de Carga vs. Señales de Bienestar",
        "workload_perception_label": "Percepción de Carga Laboral",
        "psychological_signals_label": "Señales de Bienestar",

        # Placeholders
        "plant_map_title": "5. Vista Interactiva de Instalaciones",
        "ai_insights_title": "6. Perspectivas de Riesgo Predictivas",
        "optional_modules_list": """
- **🌡️ Índice de Fatiga y Ausentismo:** Identificar patrones que conducen al agotamiento.
- **🎯 Alineación de Objetivos y Propósito:** Medir la conexión con la misión organizacional.
- **🧭 Experiencia de Incorporación:** Monitorear la integración temprana y satisfacción.
- **🔁 Confianza y Seguridad Psicológica:** Medir el clima de comunicación abierta.
- **💬 Voz del Empleado (VoE) y Sentimiento:** Analizar tendencias en la retroalimentación.
- **🧠 Oportunidades de Aprendizaje y Crecimiento:** Seguimiento del desarrollo de habilidades.
- **🧯 Preparación para Respuesta a Crisis:** Evaluar efectividad del apoyo post-incidente.
- **🌐 Panel DEI:** Monitorear representación, inclusión y percepción de equidad.
- **📡 ROI del Programa de Bienestar:** Medir impacto de iniciativas de bienestar.
"""
        # ... Add all other necessary Spanish translations ...
    }
}

# --- File Paths ---
STABILITY_DATA_FILE = "stability_data.csv"
SAFETY_DATA_FILE = "safety_data.csv"
ENGAGEMENT_DATA_FILE = "engagement_data.csv"
STRESS_DATA_FILE = "stress_data.csv"
