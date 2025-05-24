import streamlit as st
import pandas as pd
# import plotly.express as px # Only if directly used, otherwise viz handles it
# import plotly.graph_objects as go # Only if directly used
import visualizations as viz
import config

# --- Page Configuration (Applied once at the top) ---
st.set_page_config(
    page_title=config.TEXT_STRINGS[config.LANG].get("dashboard_title", config.APP_TITLE),
    page_icon=config.APP_ICON,
    layout="wide"
)

# --- Language Selection ---
st.sidebar.markdown("---") # Separator
available_langs = list(config.TEXT_STRINGS.keys())
# Try to get index, default to 0 if LANG not in available_langs (safety)
try:
    default_lang_index = available_langs.index(config.LANG)
except ValueError:
    default_lang_index = 0

selected_lang_code = st.sidebar.selectbox(
    label=config.TEXT_STRINGS["EN"].get("language_selector", "Language / Idioma:") + " / " + config.TEXT_STRINGS["ES"].get("language_selector", "Idioma / Language:"), # Show both for clarity
    options=available_langs,
    index=default_lang_index,
    format_func=lambda x: "English" if x == "EN" else "Español" if x == "ES" else x # Nicer display names
)
current_lang_texts = config.TEXT_STRINGS[selected_lang_code]


# --- Data Loading Functions with Caching ---
@st.cache_data
def load_data(file_path, date_cols=None):
    try:
        df = pd.read_csv(file_path, parse_dates=date_cols if date_cols else False)
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].notna().any():
                 df[col] = df[col].astype(str).str.strip()
        return df
    except FileNotFoundError:
        st.error(f"Error: File {file_path} not found in 'data/' directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

df_stability = load_data(config.STABILITY_DATA_FILE, date_cols=[config.COLUMN_DATE])
df_safety = load_data(config.SAFETY_DATA_FILE)
df_engagement = load_data(config.ENGAGEMENT_DATA_FILE)
df_stress = load_data(config.STRESS_DATA_FILE, date_cols=[config.COLUMN_DATE])


# --- Sidebar Filters ---
st.sidebar.header(current_lang_texts.get("filters_header", "Filters"))

# Helper to get unique sorted values, or empty list if column doesn't exist or df is empty
def get_unique_options(df, column_name):
    if not df.empty and column_name in df.columns:
        return sorted(df[column_name].dropna().unique())
    return []

sites = get_unique_options(df_stability, config.COLUMN_SITE)
regions = get_unique_options(df_stability, config.COLUMN_REGION)
departments = get_unique_options(df_stability, config.COLUMN_DEPARTMENT)
fcs = get_unique_options(df_stability, config.COLUMN_FC)
shifts = get_unique_options(df_stability, config.COLUMN_SHIFT)

selected_sites = st.sidebar.multiselect(current_lang_texts.get("select_site"), options=sites, default=config.DEFAULT_SITES)
selected_regions = st.sidebar.multiselect(current_lang_texts.get("select_region"), options=regions, default=config.DEFAULT_REGIONS)
selected_departments = st.sidebar.multiselect(current_lang_texts.get("select_department"), options=departments, default=config.DEFAULT_DEPARTMENTS)
selected_fcs = st.sidebar.multiselect(current_lang_texts.get("select_fc"), options=fcs, default=config.DEFAULT_FUNCTIONAL_CATEGORIES)
selected_shifts = st.sidebar.multiselect(current_lang_texts.get("select_shift"), options=shifts, default=config.DEFAULT_SHIFTS)


# --- Filter DataFrames Utility ---
def filter_dataframe(df, selections_map):
    if df.empty:
        return df.copy() # Return a copy of the empty dataframe
    
    filtered_df = df.copy()
    for col_config_name, selected_values in selections_map.items():
        if selected_values and col_config_name in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[col_config_name].isin(selected_values)]
    return filtered_df

selections = {
    config.COLUMN_SITE: selected_sites,
    config.COLUMN_REGION: selected_regions,
    config.COLUMN_DEPARTMENT: selected_departments,
    config.COLUMN_FC: selected_fcs,
    config.COLUMN_SHIFT: selected_shifts
}

filtered_stability = filter_dataframe(df_stability, selections)
filtered_safety = filter_dataframe(df_safety, selections)
filtered_engagement = filter_dataframe(df_engagement, selections)
filtered_stress = filter_dataframe(df_stress, selections)


# --- Main Dashboard Title & Introduction ---
st.title(current_lang_texts.get("dashboard_title"))
st.markdown(current_lang_texts.get("dashboard_subtitle"))
st.caption(current_lang_texts.get("alignment_note"))
st.markdown("---")
st.info(current_lang_texts.get("psych_safety_note")) # General note on psychological safety
st.markdown("---")


# --- CORE MODULES ---

# 1. Laboral Stability Panel
st.header(current_lang_texts.get("stability_panel_title"))
if not filtered_stability.empty:
    col1, col2, col3 = st.columns(3)
    avg_rotation = filtered_stability[config.COLUMN_ROTATION_RATE].mean()
    with col1:
        st.plotly_chart(viz.create_kpi_gauge(avg_rotation, "rotation_rate_gauge", lang=selected_lang_code,
                                            low_threshold=config.ROTATION_RATE_LOW_THRESHOLD,
                                            high_threshold=config.ROTATION_RATE_HIGH_THRESHOLD),
                        use_container_width=True)
        st.caption(f"Thresholds: Low < {config.ROTATION_RATE_LOW_THRESHOLD}%, High > {config.ROTATION_RATE_HIGH_THRESHOLD}%")


    with col2:
        ret_6m = filtered_stability[config.COLUMN_RETENTION_6M].mean() if config.COLUMN_RETENTION_6M in filtered_stability.columns else float('nan')
        viz.display_metric_card(st, "retention_6m", ret_6m, lang=selected_lang_code)
        ret_12m = filtered_stability[config.COLUMN_RETENTION_12M].mean() if config.COLUMN_RETENTION_12M in filtered_stability.columns else float('nan')
        viz.display_metric_card(st, "retention_12m", ret_12m, lang=selected_lang_code)

    with col3:
        ret_18m = filtered_stability[config.COLUMN_RETENTION_18M].mean() if config.COLUMN_RETENTION_18M in filtered_stability.columns else float('nan')
        viz.display_metric_card(st, "retention_18m", ret_18m, lang=selected_lang_code)

    if config.COLUMN_DATE in filtered_stability.columns and config.COLUMN_HIRES in filtered_stability.columns and config.COLUMN_EXITS in filtered_stability.columns:
        # Ensure date is datetime for Grouper
        stability_trend_df = filtered_stability.copy()
        stability_trend_df[config.COLUMN_DATE] = pd.to_datetime(stability_trend_df[config.COLUMN_DATE])

        hires_exits_trend = stability_trend_df.groupby(pd.Grouper(key=config.COLUMN_DATE, freq='M')).agg(
            Hires=(config.COLUMN_HIRES, 'sum'),
            Exits=(config.COLUMN_EXITS, 'sum')
        ).reset_index()
        st.plotly_chart(viz.create_line_chart(hires_exits_trend, config.COLUMN_DATE, ['Hires', 'Exits'],
                                            "hires_vs_exits_chart", lang=selected_lang_code,
                                            x_axis_title_key="month_axis"), use_container_width=True)
    else:
        st.warning(current_lang_texts.get("no_data_hires_exits"))
else:
    st.info(current_lang_texts.get("no_data_available"))
st.markdown("---")


# 2. Safety Pulse Module
st.header(current_lang_texts.get("safety_pulse_title"))
if not filtered_safety.empty:
    col1, col2, col3 = st.columns(3)
    with col1:
        if config.COLUMN_MONTH in filtered_safety.columns and \
           config.COLUMN_INCIDENTS in filtered_safety.columns and \
           config.COLUMN_NEAR_MISSES in filtered_safety.columns:

            safety_summary = filtered_safety.groupby(config.COLUMN_MONTH, as_index=False).agg(
                Incidents=(config.COLUMN_INCIDENTS, 'sum'),
                Near_Misses=(config.COLUMN_NEAR_MISSES, 'sum')
            )
            # Attempt to sort months if they are string names
            try:
                month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                safety_summary[config.COLUMN_MONTH] = pd.Categorical(safety_summary[config.COLUMN_MONTH], categories=month_order, ordered=True)
                safety_summary = safety_summary.sort_values(config.COLUMN_MONTH)
            except: # Fallback if month names are not standard or numeric
                try:
                    safety_summary = safety_summary.sort_values(by=config.COLUMN_MONTH) # Sort if numeric months
                except:
                    pass # Keep original order
            
            st.plotly_chart(viz.create_bar_chart(safety_summary, config.COLUMN_MONTH, ['Incidents', 'Near_Misses'],
                                                "monthly_incidents_chart", lang=selected_lang_code,
                                                x_axis_title_key="month_axis"), use_container_width=True)
        else:
            st.warning(current_lang_texts.get("no_data_incidents_near_misses"))

    with col2:
        days_no_accidents = filtered_safety[config.COLUMN_DAYS_WITHOUT_ACCIDENTS].max() if config.COLUMN_DAYS_WITHOUT_ACCIDENTS in filtered_safety.columns else "N/A"
        viz.display_metric_card(st, "days_without_accidents_metric", days_no_accidents, lang=selected_lang_code)

    with col3:
        # Assuming active_alerts is a count per row that should be summed up for the filtered selection
        active_alerts_count = filtered_safety[config.COLUMN_ACTIVE_ALERTS].sum() if config.COLUMN_ACTIVE_ALERTS in filtered_safety.columns else "N/A"
        viz.display_metric_card(st, "active_safety_alerts_metric", active_alerts_count, lang=selected_lang_code)
else:
    st.info(current_lang_texts.get("no_data_available"))
st.markdown("---")


# 3. Employee Engagement & Commitment
st.header(current_lang_texts.get("engagement_title"))
if not filtered_engagement.empty:
    col1, col2 = st.columns([2,1]) # Radar chart needs more space

    with col1:
        # Prepare data for radar chart: average of each engagement dimension
        radar_data_list = []
        for col_key, label_key in config.ENGAGEMENT_RADAR_CATEGORIES_KEYS.items():
            if col_key in filtered_engagement.columns:
                avg_val = filtered_engagement[col_key].mean()
                radar_data_list.append({
                    "Dimension": current_lang_texts.get(label_key, label_key.replace('_label','').title()), # Get localized label
                    "Score": avg_val if pd.notna(avg_val) else 0
                })
        
        if radar_data_list:
            df_radar = pd.DataFrame(radar_data_list)
            st.plotly_chart(viz.create_radar_chart(df_radar, "Dimension", "Score",
                                                    "engagement_dimensions_radar", lang=selected_lang_code),
                            use_container_width=True)
        else:
            st.warning(current_lang_texts.get("no_data_radar_columns"))

    with col2:
        climate_score = filtered_engagement[config.COLUMN_LABOR_CLIMATE].mean() if config.COLUMN_LABOR_CLIMATE in filtered_engagement.columns else float('nan')
        viz.display_metric_card(st, "labor_climate_score_metric", climate_score, lang=selected_lang_code)

        nps = filtered_engagement[config.COLUMN_ENPS].mean() if config.COLUMN_ENPS in filtered_engagement.columns else float('nan')
        viz.display_metric_card(st, "enps_metric", nps, lang=selected_lang_code)

        participation = filtered_engagement[config.COLUMN_PARTICIPATION].mean() if config.COLUMN_PARTICIPATION in filtered_engagement.columns else float('nan')
        viz.display_metric_card(st, "survey_participation_metric", participation, lang=selected_lang_code)
        
        recognitions = filtered_engagement[config.COLUMN_RECOGNITIONS_COUNT].sum() if config.COLUMN_RECOGNITIONS_COUNT in filtered_engagement.columns else float('nan')
        viz.display_metric_card(st, "recognitions_count_metric", recognitions, lang=selected_lang_code)

else:
    st.info(current_lang_texts.get("no_data_available"))
st.markdown("---")


# 4. Operational Stress Dashboard
st.header(current_lang_texts.get("stress_title"))
if not filtered_stress.empty:
    col1, col2 = st.columns(2)
    with col1:
        if config.COLUMN_DATE in filtered_stress.columns and \
           config.COLUMN_OVERTIME_HOURS in filtered_stress.columns and \
           config.COLUMN_UNFILLED_SHIFTS in filtered_stress.columns:
            
            stress_trend_df = filtered_stress.copy()
            stress_trend_df[config.COLUMN_DATE] = pd.to_datetime(stress_trend_df[config.COLUMN_DATE])
            
            stress_summary_monthly = stress_trend_df.groupby(pd.Grouper(key=config.COLUMN_DATE, freq='M')).agg(
                Total_Overtime=(config.COLUMN_OVERTIME_HOURS, 'sum'),
                Total_Unfilled_Shifts=(config.COLUMN_UNFILLED_SHIFTS, 'sum')
            ).reset_index()
            st.plotly_chart(viz.create_bar_chart(stress_summary_monthly, config.COLUMN_DATE, ['Total_Overtime', 'Total_Unfilled_Shifts'],
                                               "monthly_shift_load_chart", lang=selected_lang_code,
                                               x_axis_title_key="month_axis", y_axis_title="Hours / Count"), use_container_width=True)
        else:
            st.warning(current_lang_texts.get("no_data_shift_load"))

    with col2:
        avg_stress_level = filtered_stress[config.COLUMN_STRESS_LEVEL_SURVEY].mean() if config.COLUMN_STRESS_LEVEL_SURVEY in filtered_stress.columns else float('nan')
        semaforo_display = viz.create_stress_semaforo(avg_stress_level, lang=selected_lang_code)
        st.subheader(current_lang_texts.get("overall_stress_indicator"))
        st.markdown(f"## {semaforo_display}") # Larger display for semaforo
        st.caption(
            f"{current_lang_texts.get('stress_low')}: <= {config.STRESS_LEVEL_LOW_THRESHOLD:.1f} | "
            f"{current_lang_texts.get('stress_medium')}: >{config.STRESS_LEVEL_LOW_THRESHOLD:.1f} & <= {config.STRESS_LEVEL_MEDIUM_THRESHOLD:.1f} | "
            f"{current_lang_texts.get('stress_high')}: > {config.STRESS_LEVEL_MEDIUM_THRESHOLD:.1f}"
        )

    # Workload trends vs. psychological signals
    if config.COLUMN_DATE in filtered_stress.columns and \
       config.COLUMN_WORKLOAD_PERCEPTION in filtered_stress.columns and \
       config.COLUMN_PSYCH_SIGNAL_SCORE in filtered_stress.columns:

        workload_psych_df = filtered_stress.copy()
        workload_psych_df[config.COLUMN_DATE] = pd.to_datetime(workload_psych_df[config.COLUMN_DATE])

        workload_psych_trend = workload_psych_df.groupby(pd.Grouper(key=config.COLUMN_DATE, freq='M')).agg(
            Workload_Perception=(config.COLUMN_WORKLOAD_PERCEPTION, 'mean'),
            Psych_Signals=(config.COLUMN_PSYCH_SIGNAL_SCORE, 'mean')
        ).reset_index()
        st.plotly_chart(viz.create_line_chart(workload_psych_trend, config.COLUMN_DATE, ['Workload_Perception', 'Psych_Signals'],
                                            "workload_vs_psych_chart", lang=selected_lang_code,
                                            x_axis_title_key="month_axis", y_axis_title="Average Score"), use_container_width=True)
    else:
        st.warning(current_lang_texts.get("no_data_workload_psych"))
else:
    st.info(current_lang_texts.get("no_data_available"))
st.markdown("---")


# 5. Interactive Plant Map (Placeholder)
st.header(current_lang_texts.get("plant_map_title"))
st.markdown(config.PLACEHOLDER_TEXT_PLANT_MAP)
st.warning("This feature requires significant additional development including mapping libraries (e.g., Plotly Mapbox, Folium) and potentially real-time data integration. Accessibility for map interactions will need careful design to ensure usability for all.")
st.markdown("---")

# 6. Predictive AI Insights (Placeholder)
st.header(current_lang_texts.get("ai_insights_title"))
st.markdown(config.PLACEHOLDER_TEXT_AI_INSIGHTS)
st.warning("This feature requires data science expertise, model development (e.g., scikit-learn, TensorFlow), and robust data pipelines. Ensuring fairness, transparency, and ethical considerations in AI outputs is critical.")
st.markdown("---")

# --- Optional & Strategic Modules (Placeholders in Sidebar) ---
st.sidebar.markdown("---") # Separator
st.sidebar.markdown(f"## {current_lang_texts.get('optional_modules_header')}")
if st.sidebar.checkbox(current_lang_texts.get('show_optional_modules')):
    with st.expander(current_lang_texts.get('optional_modules_title'), expanded=False):
        st.info(current_lang_texts.get('optional_modules_list'))

st.sidebar.markdown("---")
st.sidebar.caption(f"{current_lang_texts.get('dashboard_title')} v0.2.0 (MVP - Enhanced)")
st.sidebar.caption("Built with Streamlit, Plotly, and Pandas.")