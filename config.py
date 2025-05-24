# config.py
import plotly.express as px
import pandas as pd # Not strictly needed here but often imported alongside

# --- GENERAL APPLICATION SETTINGS ---
APP_VERSION = "v1.0.0 (MVP - Insights & Glossary)"
APP_TITLE_KEY = "app_title" # For browser tab, potentially shorter
APP_ICON = "❤️‍🩹" # Icon symbolizing wellbeing/health

# --- FILTER DEFAULTS ---
DEFAULT_SITES = []
DEFAULT_REGIONS = []
DEFAULT_DEPARTMENTS = []
DEFAULT_FUNCTIONAL_CATEGORIES = []
DEFAULT_SHIFTS = []

# --- VISUALIZATION & THEME ---
COLOR_SCHEME_CATEGORICAL = px.colors.qualitative.Plotly
COLOR_SCHEME_SEQUENTIAL = px.colors.sequential.Viridis

COLOR_STATUS_GOOD = "#2ECC71"
COLOR_STATUS_WARNING = "#F1C40F"
COLOR_STATUS_CRITICAL = "#E74C3C"
COLOR_NEUTRAL_INFO = "#3498DB"
COLOR_TEXT_SECONDARY = "#566573"
COLOR_TARGET_LINE = "#2c3e50" # Dark, professional

# --- KPI THRESHOLDS & TARGETS ---
# Values are examples, adjust to your organization's context and data scales
STABILITY_ROTATION_RATE = { # Percentage
    "good": 8.0,    # Value <= this is 'good'
    "warning": 15.0, # Value > 'good' and <= 'warning' is 'warning'. > 'warning' is 'critical'.
    "target": 8.0   # Explicit target line for the gauge
}
STABILITY_RETENTION = { # Percentage
    "good": 90.0,   # Value >= this is 'good'
    "warning": 75.0 # Value < 'good' and >= 'warning' is 'warning'. < 'warning' is 'critical'.
}

SAFETY_INCIDENTS = { # Count (lower is better)
    "good": 1.0,     # e.g., 0 or 1 incident is good
    "warning": 5.0,  # e.g., 2-5 incidents is warning
    "target": 0.0    # Ideal target is 0
}
SAFETY_DAYS_NO_INCIDENTS = { # Count (higher is better)
    "good": 180,
    "warning": 90 # Warning if it drops to this or below (and not yet 'good')
}

STRESS_LEVEL_PSYCHOSOCIAL = { # Example 0-10 scale (higher is worse)
    "low": 3.5,        # This is effectively 'good' (<= low)
    "medium": 7.0,     # This is effectively 'warning' (> low and <= medium)
    "max_scale": 10.0  # For the visual scale of the indicator
}

ENGAGEMENT_ENPS = { # Score, e.g., -100 to 100 (higher is better)
    "good": 50.0,
    "warning": 10.0 # Warning if it's this low or lower (but not yet critical if you define that)
}
ENGAGEMENT_CLIMATE_SCORE = { # Score, e.g., 0-100 (higher is better)
    "good": 80.0,
    "warning": 60.0
}
ENGAGEMENT_PARTICIPATION = { # Percentage
    "good": 75.0 # Target for good survey participation
}
ENGAGEMENT_RADAR_DIM_TARGET = 4.0 # Generic target for 1-5 scale radar items
ENGAGEMENT_RADAR_DIM_SCALE_MAX = 5.0 # Max value for radar dimension axes

# --- COLUMN MAPPING (Conceptual Keys to Actual CSV Column Names) ---
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

# --- TEXT STRINGS FOR LOCALIZATION (i18n) ---
DEFAULT_LANG = "EN"

TEXT_STRINGS = {
    "EN": {
        "app_title": "Laboral Vital Signs",
        "dashboard_title": "Laboral Vital Signs Dashboard",
        "dashboard_subtitle": "A Human-Centered Intelligence System for Workplace Wellbeing and Organizational Performance.",
        "alignment_note": "Aligned with NOM-035-STPS-2018, ISO 45003, and DEI principles.",
        "psych_safety_note": "Note: Data concerning individual wellbeing is aggregated and presented anonymously to uphold psychological safety, privacy, and align with DEI principles and regulatory standards.",
        "error_loading_data": "Error loading data from file: {}. Ensure the file exists and is correctly formatted.",
        
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
        
        "actionable_insights_title": "Actionable Insights & Recommendations", # New
        "no_insights_generated": "No specific insights generated for the current selection. Data may be within normal ranges or insufficient for detailed analysis.",


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

        # Insight specific strings
        "rotation_high_alert": "High Rotation Alert",
        "rotation_high_insight": "Average rotation ({rotation_val:.1f}%) exceeds warning ({warn_thresh}%). Investigate causes (e.g., exit interviews, filter by department/shift). Consider targeted retention strategies.",
        "rotation_moderate_warn": "Rotation Warning",
        "rotation_moderate_insight": "Rotation rate ({rotation_val:.1f}%) is above desired ({good_thresh}%). Monitor trends and focus on at-risk groups or early tenure attrition.",
        "rotation_good_status": "Good Rotation Rate",
        "rotation_good_insight": "Rotation ({rotation_val:.1f}%) is within/below target ({good_thresh}%). Maintain positive engagement and development programs.",
        "rotation_no_data": "Rotation data unavailable for detailed insights.",
        "retention_low_alert":"Low 12-Month Retention",
        "retention_low_insight": "12-month retention ({retention_val:.1f}%) is below warning level ({warn_thresh}%). Review onboarding, career pathing, and manager effectiveness for newer cohorts.",
        "retention_needs_improvement":"12-Month Retention Needs Improvement",
        "retention_improvement_insight":"12-month retention ({retention_val:.1f}%) is below target ({target_thresh}%). Focus on enhancing overall employee experience.",
        "review_hires_exits_trend":"Review Hires vs. Exits Trend",
        "hires_exits_insight_detail": "Analyze patterns. Sustained net employee loss requires strategic workforce planning adjustments.",
        
        "high_incidents_alert": "High Incident Count",
        "high_incidents_insight": "{count:.0f} incidents reported. Above warning level. Prioritize root cause analysis and implement corrective actions, focusing on high-frequency areas/types.",
        "moderate_incidents_warn": "Moderate Incidents",
        "moderate_incidents_insight": "{count:.0f} incidents reported. Review trends to prevent escalation. Strengthen near-miss reporting and proactive hazard identification.",
        "low_incidents_status": "Low Incident Count",
        "low_incidents_insight": "{count:.0f} incidents. Good! Maintain vigilance and continue promoting proactive safety culture.",
        "dwa_low_warn":"Days Without Incidents Low",
        "dwa_low_insight": "Currently {days:.0f} days without a recordable incident, below the target. Reinforce safety protocols and conduct targeted safety awareness campaigns.",

        "enps_low_alert": "Low eNPS Alert",
        "enps_low_insight": "eNPS ({enps_val:.0f}) is critically low, indicating a significant number of detractors. Urgently conduct pulse surveys or focus groups to understand dissatisfaction drivers.",
        "enps_needs_focus": "eNPS Needs Focus",
        "enps_focus_insight": "eNPS ({enps_val:.0f}) suggests room for improving employee loyalty. Analyze feedback from passives and detractors to identify key improvement areas.",
        "enps_good_status":"Healthy eNPS",
        "enps_good_insight": "eNPS ({enps_val:.0f}) is healthy. Leverage promoters as change champions and gather insights on what makes their experience positive.",
        "climate_low_alert": "Low Work Climate Score",
        "climate_low_insight": "Work Climate Score ({climate_val:.1f}) is concerning. This may reflect issues with leadership, team dynamics, workload, or organizational culture. A detailed diagnostic is recommended.",
        "engagement_no_critical_insights":"Engagement metrics indicate no immediate critical concerns. Explore radar dimensions for specific strengths and opportunities.",

        "stress_high_alert":"High Stress Levels Detected",
        "stress_high_insight": "Average psychosocial stress ({stress_val:.1f}) is high. Risk of burnout, errors, and turnover. Identify and address key stressors (workload, control, support). Refer to NOM-035 guidelines for intervention strategies.",
        "stress_moderate_warn":"Moderate Stress Levels",
        "stress_moderate_insight": "Average psychosocial stress ({stress_val:.1f}) is moderate. Monitor closely, especially in high-demand roles/departments. Promote available wellbeing resources and ensure manageable workloads.",
        "stress_low_status":"Low Stress Levels",
        "stress_low_insight": "Average psychosocial stress ({stress_val:.1f}) is in a healthy range. Commendable! Continue to foster a supportive work environment and open communication channels.",
        "stress_trend_warn":"Concerning Stress Trend",
        "stress_trend_insight": "Recent trends suggest increasing workload perception or declining wellbeing signals. Proactively review team capacities and support mechanisms to mitigate burnout risk."
    },
    "ES": {
        # --- ALL SPANISH TRANSLATIONS ---
        "app_title": "Signos Vitales Laborales",
        "dashboard_title": "Tablero de Signos Vitales Laborales",
        "dashboard_subtitle": "Un Sistema de Inteligencia Centrado en el Ser Humano para el Bienestar y Rendimiento Organizacional.",
        "alignment_note": "Alineado con NOM-035-STPS-2018, ISO 45003 y principios DEI.",
        "psych_safety_note": "Nota: Los datos sobre el bienestar individual se presentan de forma anónima y agregada para garantizar la seguridad psicológica, la privacidad y cumplir con principios DEI y normativas.",
        "error_loading_data": "Error al cargar datos del archivo: {}. Verifique la ruta y el formato.",
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
        "no_insights_generated": "No se generaron perspectivas específicas. Los datos podrían estar en rangos normales o ser insuficientes.",
        "no_data_available": "No hay datos disponibles para los filtros seleccionados.",
        "no_data_for_selection": "Sin datos para la selección actual.",
        "no_data_hires_exits": "Datos de contrataciones/bajas no disponibles.",
        "no_data_incidents_near_misses": "Datos de incidentes/casi incidentes no disponibles.",
        "no_data_radar_columns": "Faltan columnas para el radar de compromiso.",
        "no_data_radar": "Datos insuficientes para el radar de compromiso.",
        "no_data_shift_load": "Datos de carga de turno no disponibles.",
        "no_data_workload_psych": "Datos de carga/señales de bienestar no disponibles.",
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
        "rotation_high_insight": "Rotación promedio ({rotation_val:.1f}%) excede advertencia ({warn_thresh}%). Investigar causas y aplicar estrategias de retención.",
        "rotation_moderate_warn": "Advertencia: Rotación",
        "rotation_moderate_insight": "Tasa de rotación ({rotation_val:.1f}%) sobre nivel deseado ({good_thresh}%). Monitorear tendencias y grupos en riesgo.",
        "rotation_good_status": "Tasa de Rotación Adecuada",
        "rotation_good_insight": "Rotación ({rotation_val:.1f}%) dentro/bajo objetivo ({good_thresh}%). Mantener compromiso positivo.",
        "rotation_no_data": "Datos de rotación no disponibles para análisis detallado.",
        "retention_low_alert":"Alerta: Baja Retención a 12 Meses",
        "retention_low_insight": "Retención a 12 meses ({retention_val:.1f}%) bajo advertencia ({warn_thresh}%). Revisar incorporación y desarrollo.",
        "retention_needs_improvement":"Retención a 12 Meses: Área de Mejora",
        "retention_improvement_insight":"Retención a 12 meses ({retention_val:.1f}%) bajo objetivo ({target_thresh}%). Continuar mejorando experiencia.",
        "review_hires_exits_trend":"Revisar Tendencia Contrataciones vs. Bajas",
        "hires_exits_insight_detail": "Analizar patrones. Pérdida neta constante de empleados puede indicar problemas sistémicos.",
        
        "high_incidents_alert": "Alto Número de Incidentes",
        "high_incidents_insight": "{count:.0f} incidentes. Arriba de advertencia. Priorizar análisis de causa raíz e implementar acciones correctivas.",
        "moderate_incidents_warn": "Incidentes Moderados",
        "moderate_incidents_insight": "{count:.0f} incidentes. Revisar tendencias. Fortalecer reporte de casi incidentes.",
        "low_incidents_status": "Bajo Número de Incidentes",
        "low_incidents_insight": "{count:.0f} incidentes. ¡Bien! Mantener vigilancia y cultura proactiva de seguridad.",
        "dwa_low_warn":"Pocos Días Sin Incidentes",
        "dwa_low_insight": "{days:.0f} días sin incidentes, bajo el objetivo. Reforzar protocolos y campañas de seguridad.",

        "enps_low_alert": "Alerta: eNPS Bajo",
        "enps_low_insight": "eNPS ({enps_val:.0f}) críticamente bajo. Urgente recopilar feedback sobre causas de insatisfacción.",
        "enps_needs_focus": "eNPS Necesita Atención",
        "enps_focus_insight": "eNPS ({enps_val:.0f}) indica espacio para mejorar lealtad. Analizar feedback de pasivos/detractores.",
        "enps_good_status":"eNPS Saludable",
        "enps_good_insight": "eNPS ({enps_val:.0f}) saludable. Aprovechar promotores como embajadores.",
        "climate_low_alert": "Puntuación Baja de Clima Laboral",
        "climate_low_insight": "Clima Laboral ({climate_val:.1f}) preocupante. Puede indicar problemas de liderazgo, cultura o entorno. Se recomienda diagnóstico.",
        "engagement_no_critical_insights":"Métricas de compromiso sin alertas críticas. Explorar radar para fortalezas/oportunidades.",

        "stress_high_alert":"Alerta: Niveles Altos de Estrés",
        "stress_high_insight": "Estrés psicosocial promedio ({stress_val:.1f}) alto. Riesgo de burnout. Investigar carga laboral, apoyo, presiones. Ver NOM-035.",
        "stress_moderate_warn":"Niveles Moderados de Estrés",
        "stress_moderate_insight": "Estrés psicosocial promedio ({stress_val:.1f}) moderado. Monitorear de cerca, promover recursos de bienestar.",
        "stress_low_status":"Niveles Bajos de Estrés",
        "stress_low_insight": "Estrés psicosocial promedio ({stress_val:.1f}) en rango saludable. ¡Excelente! Continuar fomentando apoyo y comunicación.",
        "stress_trend_warn":"Tendencia Preocupante de Estrés",
        "stress_trend_insight": "Tendencias recientes: percepción de carga al alza, señales de bienestar a la baja. Revisar capacidades y apoyo."
    }
}

# --- File Paths (Ensure these files are in the same directory as app.py for MVP) ---
STABILITY_DATA_FILE = "stability_data.csv"
SAFETY_DATA_FILE = "safety_data.csv"
ENGAGEMENT_DATA_FILE = "engagement_data.csv"
STRESS_DATA_FILE = "stress_data.csv"
