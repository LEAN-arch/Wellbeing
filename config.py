# config.py
import plotly.express as px # <--- ENSURE THIS IMPORT IS PRESENT AT THE TOP
import pandas as pd

# --- GENERAL APPLICATION SETTINGS ---
APP_VERSION = "v1.0.4 (PlotlyPathFix)" # Incrementing for this fix
APP_TITLE_KEY = "app_title"
APP_ICON = "🦺"

# --- FILTER DEFAULTS ---
DEFAULT_SITES = []
DEFAULT_REGIONS = []
DEFAULT_DEPARTMENTS = []
DEFAULT_FUNCTIONAL_CATEGORIES = []
DEFAULT_SHIFTS = []

# --- VISUALIZATION & THEME ---
# These now correctly use 'px' which is imported at the top of this file
COLOR_SCHEME_CATEGORICAL = px.colors.qualitative.Plotly
COLOR_SCHEME_CATEGORICAL_SET2 = px.colors.qualitative.Set2
COLOR_SCHEME_CATEGORICAL_ACCENT = px.colors.qualitative.Accent # Line that caused the error
COLOR_SCHEME_CATEGORICAL_DARK2 = px.colors.qualitative.Dark2
COLOR_SCHEME_CATEGORICAL_QUALITATIVE = px.colors.qualitative.Pastel
COLOR_SCHEME_PROFESSIONAL_LINES = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]
COLOR_SCHEME_SEQUENTIAL = px.colors.sequential.Viridis

COLOR_STATUS_GOOD = "#2ECC71"
COLOR_STATUS_WARNING = "#F1C40F"
COLOR_STATUS_CRITICAL = "#E74C3C"
COLOR_NEUTRAL_INFO = "#3498DB"
COLOR_TEXT_SECONDARY = "#566573"
COLOR_TARGET_LINE = "#2c3e50"

# --- VISUALIZATION STYLING CONSTANTS ---
FONT_FAMILY_DEFAULT = "Arial, sans-serif"
FONT_SIZE_TITLE_DEFAULT = 16
FONT_SIZE_BODY_DEFAULT = 11
FONT_SIZE_LEGEND = 10
FONT_SIZE_HOVER_LABEL = 11
FONT_SIZE_AXIS_TITLE = 13
FONT_SIZE_AXIS_TICKS = 10
FONT_SIZE_RANGESELECTOR_BUTTONS = 10
FONT_SIZE_ANNOTATION_SMALL = 9
FONT_SIZE_ANNOTATION_TARGET = 10
FONT_FAMILY_TARGET_ANNOTATION = "Arial Black"
FONT_SIZE_BAR_TEXT = 9
FONT_SIZE_SUBTITLE = 10

COLOR_PAPER_BACKGROUND = 'rgba(0,0,0,0)'
COLOR_PLOT_BACKGROUND = 'rgba(0,0,0,0)'
COLOR_TEXT_PRIMARY = "#222222"
COLOR_HOVER_LABEL_BACKGROUND = "white"
COLOR_LEGEND_BACKGROUND = "rgba(255,255,255,0.7)"
COLOR_LEGEND_BORDER = 'rgba(0,0,0,0.1)'
COLOR_GRID_PRIMARY = 'rgba(0,0,0,0.1)'
COLOR_GRID_SECONDARY = 'rgba(0,0,0,0.05)'
COLOR_AXIS_LINE = 'rgba(0,0,0,0.2)'
COLOR_SPIKE_LINE = 'rgba(0,0,0,0.3)'
COLOR_RANGESELECTOR_BACKGROUND = 'rgba(230,230,230,0.7)'
COLOR_RANGESELECTOR_BORDER = 'rgba(0,0,0,0.1)'
COLOR_BAR_TEXT_INSIDE = "#FFFFFF"
COLOR_BAR_TEXT_OUTSIDE = COLOR_TEXT_PRIMARY
COLOR_BAR_MARKER_BORDER = 'rgba(0,0,0,0.3)'
COLOR_ANNOTATION_BG = "rgba(255,255,255,0.8)"

FONT_SIZE_TITLE_GAUGE = 13
FONT_SIZE_GAUGE_NUMBER = 30
FONT_SIZE_GAUGE_DELTA = 12
FONT_SIZE_AXIS_TICKS_GAUGE = 9
COLOR_GAUGE_TICK = "rgba(0,0,0,0.2)"
COLOR_GAUGE_NEEDLE_BASE = "rgba(0,0,0,0.8)"
COLOR_GAUGE_NEEDLE_BORDER = "rgba(0,0,0,1)"
COLOR_GAUGE_BACKGROUND = "rgba(255,255,255,0.0)"
COLOR_GAUGE_BORDERCOLOR = "rgba(0,0,0,0.1)"

COLOR_SCHEME_RADAR_DEFAULT = COLOR_SCHEME_PROFESSIONAL_LINES
RADAR_FILL_OPACITY = 0.15
COLOR_RADAR_POLAR_BACKGROUND = 'rgba(255,255,255,0)'
COLOR_RADAR_AXIS_LINE = 'rgba(0,0,0,0.3)'
COLOR_RADAR_GRID_LINE = "rgba(0,0,0,0.15)"
COLOR_RADAR_ANGULAR_GRID_LINE = "rgba(0,0,0,0.1)"
FONT_SIZE_RADAR_TICK = 9
FONT_SIZE_RADAR_ANGULAR_TICK = 10
COLOR_RADAR_TICK_LABEL = COLOR_TEXT_PRIMARY

FONT_SIZE_STRESS_SEMAFORO_NUMBER = 20
FONT_SIZE_STRESS_SEMAFORO_TITLE = 12
FONT_SIZE_STRESS_SEMAFORO_AXIS_TICK = 8
COLOR_STRESS_BULLET_LOW = "rgba(46, 204, 113, 0.4)"
COLOR_STRESS_BULLET_MEDIUM = "rgba(241, 196, 15, 0.4)"
COLOR_STRESS_BULLET_HIGH = "rgba(231, 76, 60, 0.4)"
COLOR_STRESS_BULLET_BAR_BORDER = 'rgba(0,0,0,0.3)'
COLOR_STRESS_BULLET_BACKGROUND = "rgba(255,255,255,0)"
COLOR_STRESS_BULLET_BORDER = "rgba(0,0,0,0.1)"

DEFAULT_CHART_MARGINS = dict(l=50, r=30, t=70, b=50)
EPSILON = 1e-9

# --- KPI THRESHOLDS & TARGETS ---
STABILITY_ROTATION_RATE = { "good": 8.0, "warning": 15.0, "target": 8.0 }
STABILITY_RETENTION = { "good": 90.0, "warning": 75.0 }
SAFETY_INCIDENTS = { "good": 1.0, "warning": 5.0, "target": 0.0 }
SAFETY_DAYS_NO_INCIDENTS = { "good": 180, "warning": 90 }
STRESS_LEVEL_PSYCHOSOCIAL = { "low": 3.5, "medium": 7.0, "max_scale": 10.0 }
ENGAGEMENT_ENPS = { "good": 50.0, "warning": 10.0 }
ENGAGEMENT_CLIMATE_SCORE = { "good": 80.0, "warning": 60.0 }
ENGAGEMENT_PARTICIPATION = { "good": 75.0 }
ENGAGEMENT_RADAR_DIM_TARGET = 4.0
ENGAGEMENT_RADAR_DIM_SCALE_MAX = 5.0

# --- COLUMN MAPPING ---
COLUMN_MAP = {
    "site": "site", "region": "region", "department": "department", "fc": "fc", "shift": "shift",
    "date": "date", "month": "month",
    "rotation_rate": "rotation_rate", "retention_6m": "retention_6m",
    "retention_12m": "retention_12m", "retention_18m": "retention_18m",
    "hires": "hires", "exits": "exits",
    "incidents": "incidents", "near_misses": "near_misses",
    "days_without_accidents": "days_without_accidents", "active_alerts": "active_alerts",
    "engagement_radar_dims_cols": {
        "initiative": "initiative", "punctuality": "punctuality",
        "recognition": "recognition_data", "feedback": "feedback_data",
    },
    "engagement_radar_dims_labels": {
        "initiative": "initiative_label", "punctuality": "punctuality_label",
        "recognition": "recognition_label", "feedback": "feedback_label",
    },
    "labor_climate_score": "labor_climate_score", "enps_score": "nps",
    "participation_rate": "participation", "recognitions_count": "recognitions_count",
    "overtime_hours": "overtime_hours", "unfilled_shifts": "unfilled_shifts",
    "stress_level_survey": "stress_level_survey",
    "workload_perception": "workload_perception", "psychological_signals": "psychological_signals",
}

# --- UI PLACEHOLDER TEXTS ---
PLACEHOLDER_TEXT_PLANT_MAP = """
### Interactive Facility Map
(Placeholder: This module is envisioned to visualize data spatially, such as staff distribution heatmaps for stress, incidents, or absenteeism. 
Future development would involve integration with mapping libraries like Plotly Mapbox or Folium, and potentially real-time data feeds if ethically and technically feasible. 
Ensuring accessibility for map interactions will be a key consideration.)
"""
PLACEHOLDER_TEXT_AI_INSIGHTS = """
### Predictive AI Insights
(Placeholder: This module aims to provide data-driven forecasts on psychosocial risks (e.g., based on aggregated Human Affect & Behavior Scores). 
It could predict outcomes like burnout or turnover likelihood for specific areas/teams, ideally with confidence intervals. 
Implementation requires significant historical data, trained machine learning models (e.g., using scikit-learn or TensorFlow), and robust data pipelines. 
**Crucially, the fairness, transparency, and ethical implications of any AI-driven predictions must be paramount considerations during development and deployment.**)
"""

# --- TEXT STRINGS FOR LOCALIZATION ---
DEFAULT_LANG = "EN"
TEXT_STRINGS = {
    "EN": {
        "app_title": "Laboral Vital Signs",
        "dashboard_title": "Laboral Vital Signs Dashboard",
        "dashboard_subtitle": "A Human-Centered Intelligence System for Workplace Wellbeing and Organizational Performance.",
        "alignment_note": "Aligned with NOM-035-STPS-2018, ISO 45003, and DEI principles.",
        "psych_safety_note": "Note: Data concerning individual wellbeing is aggregated and presented anonymously to uphold psychological safety, privacy, and align with DEI principles and regulatory standards.",
        "error_loading_data": "Error loading data from file: {}. Ensure the file exists and is correctly formatted.",
        "check_file_path_instruction": "Please check the file path.",
        "exception_detail_prefix": "Exception",
        "chart_generation_error_label": "Chart Generation Error",

        "navigation_label": "Navigation", "dashboard_nav_label": "Dashboard", "glossary_nav_label": "Glossary",
        "filters_header": "Dashboard Filters", "language_selector": "Language",
        "select_site": "Site(s):", "select_region": "Region(s):", "select_department": "Department(s):",
        "select_fc": "Functional Category (FC):", "select_shift": "Shift(s):",

        "metrics_legend": "Legend", "average_label": "Avg.", "target_label": "Target",
        "prev_period_label_short": "vs Prev.", "month_axis_label": "Month", "date_time_axis_label": "Date / Time",
        "category_axis_label": "Category", "value_axis_label": "Value", "count_label": "Count",
        "people_count_label": "Number of Employees", "hours_label": "Hours", "shifts_label": "Shifts",
        "days_unit": "days", "percentage_label": "Percentage (%)", "score_label": "Score",
        "average_score_label": "Average Score", "hours_or_shifts_label": "Hours / Count",
        "dimension_label": "Dimension",
        "date_label": "Date", "1y_range_label": "1Y", "all_range_label": "All",

        "good_label": "Good", "warning_label": "Warning", "critical_label": "Critical",
        "low_label": "Low", "moderate_label": "Moderate", "high_label": "High", "status_na_label": "N/A",

        "stability_panel_title": "1. Laboral Stability",
        "rotation_rate_gauge": "Employee Rotation Rate",
        "rotation_rate_metric_help": "Percentage of employees leaving the organization. Lower is better. Target: < {target}%.",
        "retention_6m_metric": "6-Month Retention", "retention_12m_metric": "12-Month Retention", "retention_18m_metric": "18-Month Retention",
        "retention_metric_help": "Percentage of new hires remaining. Higher is better. Target: > {target}%.",
        "hires_vs_exits_chart_title": "Monthly Hiring and Attrition Trends",
        "hires_label": "New Hires", "exits_label": "Departures",

        "safety_pulse_title": "2. Safety Pulse",
        "monthly_incidents_chart_title": "Monthly Safety Events",
        "incidents_label": "Incidents", "near_misses_label": "Near Misses",
        "days_without_accidents_metric": "Days Since Last Recordable Incident",
        "days_no_incident_help": "Tracks days without a major safety incident. Target: Maximize.",
        "active_safety_alerts_metric": "Open Safety Alerts",

        "engagement_title": "3. Employee Engagement & Commitment",
        "engagement_dimensions_radar_title": "Key Engagement Dimensions (Avg. Score)",
        "initiative_label": "Initiative", "punctuality_label": "Punctuality",
        "recognition_label": "Recognition Culture", "feedback_label": "Feedback Environment",
        "labor_climate_score_metric": "Work Climate Index",
        "enps_metric": "eNPS",
        "enps_metric_help": "Employee Net Promoter Score. Target: > {target}.",
        "survey_participation_metric": "Survey Response Rate",
        "recognitions_count_metric": "Total Recognitions Given",

        "stress_title": "4. Operational Stress & Workload",
        "overall_stress_indicator_title": "Psychosocial Stress Index",
        "stress_indicator_help": "Aggregated measure of workplace stress (0-10 scale). Lower is better. Target: < {target}.",
        "monthly_shift_load_chart_title": "Monthly Operational Load",
        "overtime_label": "Overtime Hours", "unfilled_shifts_label": "Unfilled Shifts",
        "workload_vs_psych_chart_title": "Workload Perception vs. Wellbeing Signals",
        "workload_perception_label": "Perceived Workload", "psychological_signals_label": "Wellbeing Signals",

        "plant_map_title": "5. Interactive Facility Overview",
        "ai_insights_title": "6. Predictive Risk Insights",

        "actionable_insights_title": "Actionable Insights & Recommendations",
        "no_insights_generated": "No specific insights generated for the current selection. Data may be within normal ranges or insufficient for detailed analysis.",
        "no_insights_generated_stability": "Stability metrics appear within defined targets or data is insufficient for detailed trend analysis.",
        "no_insights_generated_safety": "Safety metrics indicate performance within acceptable ranges or data is insufficient for specific alerts.",
        "no_insights_generated_engagement": "Engagement indicators are generally positive or data is insufficient for specific recommendations.",
        "no_insights_generated_stress": "Stress indicators appear within acceptable ranges or data is insufficient for detailed trend analysis.",
        "no_data_for_engagement_insights": "Insufficient data to generate engagement insights.",
        "no_data_for_stress_insights": "Insufficient data to generate stress insights.",


        "no_data_available": "No data available for the selected filters in this module.",
        "no_data_for_selection": "No data for current selection.",
        "no_data_hires_exits": "Hires/Exits data not available for trend.",
        "no_data_incidents_near_misses": "Incident/Near Miss data not available for chart.",
        "no_data_radar_columns": "Required engagement dimension columns are missing.",
        "no_data_radar": "Insufficient data for engagement radar chart.",
        "no_data_shift_load": "Shift load data not available for chart.",
        "no_data_workload_psych": "Workload/Wellbeing signal data not available for trend.",

        "optional_modules_header": "Future Vision: Extended Modules",
        "show_optional_modules": "Show Planned Features",
        "optional_modules_title": "Planned Strategic & Operational Modules",
        "optional_modules_list": """
- **🌡️ Fatigue & Absenteeism Index:** Identify patterns leading to burnout.
- **🎯 Goal Alignment & Purpose:** Gauge connection to organizational mission.
- **🧭 Onboarding Experience:** Monitor early integration and satisfaction.
- **🔁 Trust & Psychological Safety:** Measure open communication climate.
- **💬 Voice of Employee (VoE) & Sentiment:** Analyze feedback trends.
- **🧠 Learning & Growth Opportunities:** Track skill development engagement.
- **🧯 Crisis Response Readiness:** Assess post-incident support effectiveness.
- **🌐 DEI Dashboard:** Monitor representation, inclusion, and equity perception.
- **📡 Wellbeing Program ROI:** Measure impact of wellness initiatives. """,

        "glossary_page_title": "Glossary of Terms",
        "glossary_intro": "Find definitions for key terms and metrics used in this dashboard.",
        "search_term_label": "Search Term:",
        "no_term_found": "No term found matching your search criteria.",
        "definition_label": "Definition",
        "glossary_empty_message": "The glossary is currently empty or could not be loaded.",
        "language_name_full_EN": "English",
        "language_name_full_ES": "Español",

        "rotation_high_alert": "High Rotation Alert",
        "rotation_high_insight_v2": "Average rotation ({rotation_val}%) significantly exceeds warning ({warn_thresh}%). Target is {target_thresh}%. Prioritize root cause analysis for high-turnover areas (filter dashboard) and review exit interview data. Implement targeted retention programs for critical roles.",
        "rotation_moderate_warn": "Rotation Warning",
        "rotation_moderate_insight_v2": "Rotation rate ({rotation_val}%) is above desired ({good_thresh}%). Monitor trends and focus on at-risk groups or early tenure attrition. Target: {target_thresh}%.",
        "rotation_good_status": "Good Rotation Rate",
        "rotation_good_insight_v2": "Rotation rate ({rotation_val}%) meets or is below good level ({good_thresh}%). Maintain strong engagement and onboarding practices.",
        "rotation_no_data": "Rotation data unavailable for detailed insights.",
        "retention_low_alert":"Low 12-Month Retention",
        "retention_low_insight_v2": "12-month retention ({retention_val}%) is critically low (below {warn_thresh}%). Target is {target_thresh}%. Investigate early attrition causes: onboarding effectiveness, manager support, role clarity, and growth opportunities within the first year.",
        "retention_needs_improvement":"12-Month Retention Needs Improvement",
        "retention_improvement_insight_v2":"12-month retention ({retention_val}%) is below target ({target_thresh}%). Identify and address factors impacting new hire experience and commitment.",
        "retention_good_status": "Good 12m Retention",
        "retention_good_insight_detail": "12-month retention ({retention_val}%) meets/exceeds target ({target_thresh}%). Current strategies appear effective. Continue monitoring.",
        "review_hires_exits_trend":"Review Hires vs. Exits Trend",
        "hires_exits_insight_detail": "Analyze patterns. Sustained net employee loss requires strategic workforce planning adjustments.",
        "net_loss_trend_warn": "Net Employee Loss Trend",
        "net_loss_insight":"Averaging a net loss of {avg_change} employees per period. This indicates potential challenges in workforce replenishment or retention. Review talent acquisition strategies and drivers of attrition.",
        "recent_loss_focus":"Focus on Recent Departures",
        "recent_loss_insight_detail":"Recent period shows a net loss of {recent_change} employees. Address immediate concerns contributing to departures.",
        "high_incidents_alert": "High Incident Count",
        "high_incidents_insight_v2": "{count} incidents this period (above warning {warn_thresh}). Critical review of high-risk tasks/areas needed. Implement immediate corrective actions and review safety training effectiveness.",
        "moderate_incidents_warn": "Moderate Incidents",
        "moderate_incidents_insight_v2": "{count} incidents this period (target is {target_thresh}). Focus on proactive measures: near-miss analysis, safety walks, and toolbox talks.",
        "low_incidents_status": "Low Incident Count",
        "low_incidents_insight_v2": "{count} incidents. Excellent! Reinforce positive safety behaviors and maintain high standards of hazard identification.",
        "dwa_low_warn":"Days Without Incidents Low",
        "dwa_low_insight_v2": "{days} days without incidents is below warning level of {warn_thresh}. Review recent incident patterns and refresh safety focus in relevant areas.",
        "dwa_needs_improvement":"Days Without Incidents: Monitor",
        "dwa_improvement_insight": "{days} days without incidents. Aim to increase this towards the target of {target_thresh}+ days. Celebrate milestones reached.",
        "dwa_good_status": "Good Days Without Incidents",
        "dwa_good_insight_detail":"{days} days without incidents. Keep up the great work!",
        "enps_low_alert": "Low eNPS Alert",
        "enps_low_insight": "eNPS ({enps_val}) is critically low, indicating a significant number of detractors. Urgently gather feedback on dissatisfaction drivers.",
        "enps_needs_focus": "eNPS Needs Focus",
        "enps_focus_insight": "eNPS ({enps_val}) suggests room for improving employee loyalty. Analyze feedback from passives and detractors to identify key improvement areas.",
        "enps_good_status":"Healthy eNPS",
        "enps_good_insight": "eNPS ({enps_val}) is healthy. Leverage promoters as change champions and gather insights on what makes their experience positive.",
        "climate_low_alert": "Low Work Climate Score",
        "climate_low_insight": "Work Climate Score ({climate_val}) is significantly low. This may reflect issues with leadership, culture, or work environment. A detailed diagnostic is recommended.",
        "climate_needs_focus": "Climate Score: Needs Attention",
        "climate_focus_insight": "Work Climate ({climate_val}) is below target ({target_thresh}). Address areas identified in survey feedback.",
        "participation_low_warn":"Low Survey Participation",
        "participation_low_insight":"Survey participation ({part_val}%) is below target ({target_thresh}%). Understand barriers to participation for more representative data.",
        "engagement_no_critical_insights":"Engagement metrics indicate no immediate critical concerns. Explore radar dimensions for specific strengths and opportunities.",
        "stress_high_alert":"High Stress Levels Detected",
        "stress_high_insight": "Average psychosocial stress ({stress_val}) is high. Risk of burnout, errors, and turnover. Identify and address key stressors (workload, control, support). Refer to NOM-035 guidelines for intervention strategies.",
        "stress_moderate_warn":"Moderate Stress Levels",
        "stress_moderate_insight": "Average psychosocial stress ({stress_val}) is moderate. Monitor closely, especially in high-demand roles/departments. Promote available wellbeing resources and ensure manageable workloads.",
        "stress_low_status":"Low Stress Levels",
        "stress_low_insight": "Average psychosocial stress ({stress_val}) is in a healthy range. Commendable! Continue to foster a supportive work environment and open communication channels.",
        "stress_trend_warn":"Concerning Stress Trend",
        "stress_trend_insight": "Recent trends suggest increasing workload perception or declining wellbeing signals. Proactively review team capacities and support mechanisms to mitigate burnout risk."
    },
    "ES": {
        "app_title": "Signos Vitales Laborales",
        "dashboard_title": "Tablero de Signos Vitales Laborales",
        "dashboard_subtitle": "Un Sistema de Inteligencia Centrado en el Ser Humano para el Bienestar y Rendimiento Organizacional.",
        "alignment_note": "Alineado con NOM-035-STPS-2018, ISO 45003 y principios DEI.",
        "psych_safety_note": "Nota: Los datos sobre el bienestar individual se presentan de forma anónima y agregada para garantizar la seguridad psicológica, la privacidad y cumplir con principios DEI y normativas.",
        "error_loading_data": "Error al cargar datos del archivo: {}. Asegúrese de que el archivo exista y esté formateado correctamente.",
        "check_file_path_instruction": "Por favor, verifique la ruta del archivo.",
        "exception_detail_prefix": "Excepción",
        "chart_generation_error_label": "Error al Generar Gráfico",
        "navigation_label": "Navegación", "dashboard_nav_label": "Tablero Principal", "glossary_nav_label": "Glosario",
        "filters_header": "Filtros del Tablero", "language_selector": "Idioma",
        "select_site": "Sitio(s):", "select_region": "Región(es):", "select_department": "Departamento(s):",
        "select_fc": "Categoría Funcional (CF):", "select_shift": "Turno(s):",
        "metrics_legend": "Leyenda", "average_label": "Prom.", "target_label": "Objetivo",
        "prev_period_label_short": "vs Per. Ant.", "month_axis_label": "Mes", "date_time_axis_label": "Fecha / Hora",
        "category_axis_label": "Categoría", "value_axis_label": "Valor", "count_label": "Cantidad",
        "people_count_label": "Número de Empleados", "hours_label": "Horas", "shifts_label": "Turnos",
        "days_unit": "días", "percentage_label": "Porcentaje (%)", "score_label": "Puntuación",
        "average_score_label": "Puntuación Promedio", "hours_or_shifts_label": "Horas / Cantidad",
        "dimension_label": "Dimensión",
        "date_label": "Fecha", "1y_range_label": "1A", "all_range_label": "Todo",
        "good_label": "Bueno", "warning_label": "Advertencia", "critical_label": "Crítico",
        "low_label": "Bajo", "moderate_label": "Moderado", "high_label": "Alto", "status_na_label": "N/D",
        "stability_panel_title": "1. Estabilidad Laboral",
        "rotation_rate_gauge": "Tasa de Rotación de Personal",
        "rotation_rate_metric_help": "Porcentaje de empleados que dejan la organización. Menor es mejor. Objetivo: < {target}%.",
        "retention_6m_metric": "Retención a 6 Meses", "retention_12m_metric": "Retención a 12 Meses", "retention_18m_metric": "Retención a 18 Meses",
        "retention_metric_help": "Porcentaje de nuevas contrataciones que permanecen. Mayor es mejor. Objetivo: > {target}%.",
        "hires_vs_exits_chart_title": "Tendencias Mensuales de Contratación y Desvinculación",
        "hires_label": "Contrataciones", "exits_label": "Bajas",
        "safety_pulse_title": "2. Pulso de Seguridad",
        "monthly_incidents_chart_title": "Eventos de Seguridad Mensuales",
        "incidents_label": "Incidentes", "near_misses_label": "Casi Incidentes",
        "days_without_accidents_metric": "Días Desde Último Incidente Registrable",
        "days_no_incident_help": "Días sin incidentes graves. Objetivo: Maximizar.",
        "active_safety_alerts_metric": "Alertas de Seguridad Abiertas",
        "engagement_title": "3. Compromiso y Vinculación del Personal",
        "engagement_dimensions_radar_title": "Dimensiones Clave de Compromiso (Promedio)",
        "initiative_label": "Iniciativa", "punctuality_label": "Puntualidad",
        "recognition_label": "Cultura de Reconocimiento", "feedback_label": "Entorno de Retroalimentación",
        "labor_climate_score_metric": "Índice de Clima Laboral",
        "enps_metric": "eNPS",
        "enps_metric_help": "Employee Net Promoter Score. Objetivo: > {target}.",
        "survey_participation_metric": "Tasa de Respuesta a Encuestas",
        "recognitions_count_metric": "Total de Reconocimientos Otorgados",
        "stress_title": "4. Estrés Operacional y Carga Laboral",
        "overall_stress_indicator_title": "Índice de Estrés Psicosocial",
        "stress_indicator_help": "Medida agregada del estrés laboral (escala 0-10). Menor es mejor. Objetivo: < {target}.",
        "monthly_shift_load_chart_title": "Carga Operacional Mensual",
        "overtime_label": "Horas Extra", "unfilled_shifts_label": "Turnos Sin Cubrir",
        "workload_vs_psych_chart_title": "Percepción de Carga vs. Señales de Bienestar",
        "workload_perception_label": "Percepción de Carga Laboral", "psychological_signals_label": "Señales de Bienestar",
        "plant_map_title": "5. Vista Interactiva de Instalaciones",
        "ai_insights_title": "6. Perspectivas de Riesgo Predictivas",
        "actionable_insights_title": "Perspectivas Accionables y Recomendaciones",
        "no_insights_generated": "No se generaron perspectivas específicas para la selección actual. Los datos podrían estar en rangos normales o ser insuficientes.",
        "no_insights_generated_stability": "Métricas de estabilidad parecen dentro de objetivos o datos insuficientes para análisis detallado.",
        "no_insights_generated_safety": "Métricas de seguridad indican desempeño aceptable o datos insuficientes para alertas específicas.",
        "no_insights_generated_engagement":"Indicadores de compromiso generalmente positivos o datos insuficientes para recomendaciones específicas.",
        "no_insights_generated_stress":"Indicadores de estrés parecen dentro de rangos aceptables o datos insuficientes para análisis detallado.",
        "no_data_for_engagement_insights":"Datos insuficientes para generar perspectivas de compromiso.",
        "no_data_for_stress_insights":"Datos insuficientes para generar perspectivas de estrés.",
        "no_data_available": "No hay datos disponibles para los filtros seleccionados.",
        "no_data_for_selection": "Sin datos para la selección actual.",
        "no_data_hires_exits": "Datos de contrataciones/bajas no disponibles para la tendencia.",
        "no_data_incidents_near_misses": "Datos de incidentes/casi incidentes no disponibles para el gráfico.",
        "no_data_radar_columns": "Faltan columnas para el radar de compromiso.",
        "no_data_radar": "Datos insuficientes para el radar de compromiso.",
        "no_data_shift_load": "Datos de carga de turno no disponibles para el gráfico.",
        "no_data_workload_psych": "Datos de carga/señales de bienestar no disponibles para tendencia.",
        "optional_modules_header": "Visión Futura: Módulos Extendidos",
        "show_optional_modules": "Mostrar Funcionalidades Planeadas",
        "optional_modules_title": "Módulos Estratégicos y Operativos Planeados",
        "optional_modules_list": """
- **🌡️ Índice de Fatiga y Ausentismo:** Identificar patrones de agotamiento.
- **🎯 Alineación de Objetivos y Propósito:** Medir conexión con la misión.
- **🧭 Experiencia de Incorporación:** Monitorear integración y satisfacción.
- **🔁 Confianza y Seguridad Psicológica:** Medir clima de comunicación.
- **💬 Voz del Empleado (VoE) y Sentimiento:** Analizar tendencias en retroalimentación.
- **🧠 Aprendizaje y Crecimiento:** Seguimiento de desarrollo de habilidades.
- **🧯 Preparación para Respuesta a Crisis:** Evaluar apoyo post-incidente.
- **🌐 Panel DEI:** Monitorear representación, inclusión, y equidad.
- **📡 ROI del Programa de Bienestar:** Medir impacto de iniciativas. """,
        "glossary_page_title": "Glosario de Términos",
        "glossary_intro": "Definiciones de términos y métricas clave de este tablero.",
        "search_term_label": "Buscar Término:",
        "no_term_found": "No se encontró ningún término con su búsqueda.",
        "definition_label": "Definición",
        "glossary_empty_message": "El glosario está vacío o no se pudo cargar.",
        "language_name_full_EN": "Inglés",
        "language_name_full_ES": "Español",
        "rotation_high_alert": "Alerta: Alta Rotación",
        "rotation_high_insight_v2": "Rotación promedio ({rotation_val}%) excede significativamente advertencia ({warn_thresh}%). Objetivo: {target_thresh}%. Priorizar análisis de causa raíz en áreas de alta rotación (filtrar tablero) y revisar entrevistas de salida. Implementar programas de retención específicos.",
        "rotation_moderate_warn": "Advertencia: Rotación",
        "rotation_moderate_insight_v2": "Tasa de rotación ({rotation_val}%) está sobre el nivel deseado ({good_thresh}%). Monitorear tendencias y enfocarse en grupos en riesgo o deserción temprana. Objetivo: {target_thresh}%.",
        "rotation_good_status": "Tasa de Rotación Adecuada",
        "rotation_good_insight_v2": "Tasa de rotación ({rotation_val}%) cumple o está por debajo del nivel bueno ({good_thresh}%). Mantener fuertes prácticas de compromiso e incorporación.",
        "rotation_no_data": "Datos de rotación no disponibles para perspectivas detalladas.",
        "retention_low_alert":"Alerta: Baja Retención a 12 Meses",
        "retention_low_insight_v2": "Retención a 12 meses ({retention_val}%) es críticamente baja (bajo {warn_thresh}%). Objetivo: {target_thresh}%. Investigar causas de deserción temprana: efectividad de onboarding, apoyo gerencial, claridad del rol y oportunidades de crecimiento.",
        "retention_needs_improvement":"Retención a 12 Meses Necesita Mejora",
        "retention_improvement_insight_v2":"Retención a 12 meses ({retention_val}%) está por debajo del objetivo ({target_thresh}%). Identificar y abordar factores que impactan la experiencia y compromiso de nuevos empleados.",
        "retention_good_status": "Buena Retención a 12m",
        "retention_good_insight_detail": "Retención a 12 meses ({retention_val}%) cumple/supera objetivo ({target_thresh}%). Estrategias actuales efectivas.",
        "review_hires_exits_trend":"Revisar Tendencia Contrataciones vs. Bajas",
        "hires_exits_insight_detail": "Analizar patrones. Pérdida neta constante de empleados requiere ajustes estratégicos de planificación de la fuerza laboral.",
        "net_loss_trend_warn":"Tendencia de Pérdida Neta de Empleados",
        "net_loss_insight":"Promedio de pérdida neta de {avg_change} empleados por período. Indica desafíos en reposición o retención. Revisar estrategias de adquisición de talento y causas de deserción.",
        "recent_loss_focus":"Enfocarse en Bajas Recientes",
        "recent_loss_insight_detail":"Período reciente muestra pérdida neta de {recent_change} empleados. Abordar preocupaciones inmediatas que contribuyen a las bajas.",
        "high_incidents_alert": "Alto Número de Incidentes",
        "high_incidents_insight_v2": "{count} incidentes este período (sobre advertencia {warn_thresh}). Revisión crítica de tareas/áreas de alto riesgo. Implementar acciones correctivas y revisar efectividad de capacitación en seguridad.",
        "moderate_incidents_warn": "Incidentes Moderados",
        "moderate_incidents_insight_v2": "{count} incidentes este período (objetivo es {target_thresh}). Enfocarse en medidas proactivas: análisis de casi-accidentes, rondas de seguridad y charlas informativas.",
        "low_incidents_status": "Bajo Número de Incidentes",
        "low_incidents_insight_v2": "{count} incidentes. ¡Excelente! Reforzar comportamientos seguros y mantener altos estándares de identificación de peligros.",
        "dwa_low_warn":"Pocos Días Sin Incidentes",
        "dwa_low_insight_v2": "{days} días sin incidentes está por debajo del nivel de advertencia de {warn_thresh}. Revisar patrones de incidentes recientes y refrescar el enfoque de seguridad.",
        "dwa_needs_improvement":"Días Sin Incidentes: Monitorear",
        "dwa_improvement_insight": "{days} días sin incidentes. Aspirar a aumentar hacia el objetivo de {target_thresh}+ días. Celebrar hitos.",
        "dwa_good_status": "Buenos Días Sin Incidentes",
        "dwa_good_insight_detail":"{days} días sin incidentes. ¡Mantener el buen trabajo!",
        "enps_low_alert": "Alerta: eNPS Bajo",
        "enps_low_insight": "eNPS ({enps_val}) críticamente bajo. Urgente obtener retroalimentación sobre motivos de insatisfacción.",
        "enps_needs_focus": "eNPS Necesita Atención",
        "enps_focus_insight": "eNPS ({enps_val}) sugiere margen para mejorar lealtad. Analizar retroalimentación de pasivos y detractores.",
        "enps_good_status":"eNPS Saludable",
        "enps_good_insight": "eNPS ({enps_val}) saludable. Usar promotores como campeones del cambio.",
        "climate_low_alert": "Puntuación Baja de Clima Laboral",
        "climate_low_insight": "Clima Laboral ({climate_val}) preocupante. Podría reflejar problemas de liderazgo, cultura o entorno. Se recomienda diagnóstico detallado.",
        "climate_needs_focus":"Clima Laboral: Requiere Atención",
        "climate_focus_insight":"Clima Laboral ({climate_val}) bajo el objetivo ({target_thresh}). Atender áreas identificadas en retroalimentación.",
        "participation_low_warn":"Baja Participación en Encuesta",
        "participation_low_insight":"Participación en encuesta ({part_val}%) bajo el objetivo ({target_thresh}%). Entender barreras para datos más representativos.",
        "engagement_no_critical_insights":"Métricas de compromiso sin alertas críticas. Explorar radar para detalles.",
        "stress_high_alert":"Alerta: Niveles Altos de Estrés",
        "stress_high_insight": "Estrés psicosocial promedio ({stress_val}) alto. Riesgo de burnout, errores y rotación. Identificar y abordar estresores. Ver NOM-035.",
        "stress_moderate_warn":"Niveles Moderados de Estrés",
        "stress_moderate_insight": "Estrés psicosocial promedio ({stress_val}) moderado. Monitorear de cerca, promover recursos de bienestar y cargas manejables.",
        "stress_low_status":"Niveles Bajos de Estrés",
        "stress_low_insight": "Estrés psicosocial promedio ({stress_val}) en rango saludable. ¡Felicitaciones! Continuar fomentando un ambiente de apoyo.",
        "stress_trend_warn":"Tendencia Preocupante de Estrés",
        "stress_trend_insight": "Tendencias recientes: percepción de carga al alza o señales de bienestar a la baja. Revisar capacidades y apoyo para mitigar riesgo de burnout."
    }
}

# --- File Paths ---
STABILITY_DATA_FILE = "stability_data.csv"
SAFETY_DATA_FILE = "safety_data.csv"
ENGAGEMENT_DATA_FILE = "engagement_data.csv"
STRESS_DATA_FILE = "stress_data.csv"
