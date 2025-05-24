import streamlit as st
import pandas as pd
import visualizations as viz
import config

# --- Page Configuration (Applied once at the top) ---
initial_lang_code = config.LANG
if 'selected_lang_code' in st.session_state:
    initial_lang_code = st.session_state.selected_lang_code

st.set_page_config(
    page_title=config.TEXT_STRINGS[initial_lang_code].get("dashboard_title", config.APP_TITLE),
    page_icon=config.APP_ICON,
    layout="wide"
)

# --- Language Selection ---
st.sidebar.markdown("---")
available_langs = list(config.TEXT_STRINGS.keys())
if 'selected_lang_code' not in st.session_state:
    st.session_state.selected_lang_code = config.LANG

def update_lang():
    st.session_state.selected_lang_code = st.session_state._lang_selector

selected_lang_code = st.sidebar.selectbox(
    label=config.TEXT_STRINGS["EN"].get("language_selector") + " / " + config.TEXT_STRINGS["ES"].get("language_selector"),
    options=available_langs,
    index=available_langs.index(st.session_state.selected_lang_code),
    format_func=lambda x: "English" if x == "EN" else "Español" if x == "ES" else x,
    key="_lang_selector",
    on_change=update_lang
)
current_lang_texts = config.TEXT_STRINGS[st.session_state.selected_lang_code]

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
        st.error(f"Error: File '{file_path}' not found. Please ensure it's in the same directory as app.py.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

df_stability_raw = load_data(config.STABILITY_DATA_FILE, date_cols=[config.COLUMN_DATE])
df_safety_raw = load_data(config.SAFETY_DATA_FILE)
df_engagement_raw = load_data(config.ENGAGEMENT_DATA_FILE)
df_stress_raw = load_data(config.STRESS_DATA_FILE, date_cols=[config.COLUMN_DATE])

# --- Sidebar Filters ---
st.sidebar.header(current_lang_texts.get("filters_header"))

def get_unique_options(df, column_name):
    if not df.empty and column_name in df.columns:
        return sorted(df[column_name].dropna().astype(str).unique())
    return []

# Combine all dataframes to get comprehensive filter options
combined_df_for_filters = pd.concat([df_stability_raw, df_safety_raw, df_engagement_raw, df_stress_raw], ignore_index=True)

sites = get_unique_options(combined_df_for_filters, config.COLUMN_SITE)
regions = get_unique_options(combined_df_for_filters, config.COLUMN_REGION)
departments = get_unique_options(combined_df_for_filters, config.COLUMN_DEPARTMENT)
fcs = get_unique_options(combined_df_for_filters, config.COLUMN_FC)
shifts = get_unique_options(combined_df_for_filters, config.COLUMN_SHIFT)

selected_sites = st.sidebar.multiselect(current_lang_texts.get("select_site"), options=sites, default=config.DEFAULT_SITES)
selected_regions = st.sidebar.multiselect(current_lang_texts.get("select_region"), options=regions, default=config.DEFAULT_REGIONS)
selected_departments = st.sidebar.multiselect(current_lang_texts.get("select_department"), options=departments, default=config.DEFAULT_DEPARTMENTS)
selected_fcs = st.sidebar.multiselect(current_lang_texts.get("select_fc"), options=fcs, default=config.DEFAULT_FUNCTIONAL_CATEGORIES)
selected_shifts = st.sidebar.multiselect(current_lang_texts.get("select_shift"), options=shifts, default=config.DEFAULT_SHIFTS)

# --- Filter DataFrames Utility ---
def filter_dataframe_by_selections(df, current_selections):
    if df.empty:
        return df.copy()
    
    filtered_df = df.copy()
    for col_config_key, selected_values in current_selections.items():
        actual_col_name = getattr(config, col_config_key, None) # Get actual column name from config
        if actual_col_name and selected_values and actual_col_name in filtered_df.columns:
            if filtered_df[actual_col_name].dtype == 'object' or pd.api.types.is_string_dtype(filtered_df[actual_col_name]):
                 selected_values_str = [str(v) for v in selected_values]
                 filtered_df = filtered_df[filtered_df[actual_col_name].isin(selected_values_str)]
            else: # For numerical or other types if exact match is needed
                 filtered_df = filtered_df[filtered_df[actual_col_name].isin(selected_values)]
    return filtered_df


filter_selections = {
    'COLUMN_SITE': selected_sites,
    'COLUMN_REGION': selected_regions,
    'COLUMN_DEPARTMENT': selected_departments,
    'COLUMN_FC': selected_fcs,
    'COLUMN_SHIFT': selected_shifts
}

df_stability = filter_dataframe_by_selections(df_stability_raw, filter_selections)
df_safety = filter_dataframe_by_selections(df_safety_raw, filter_selections)
df_engagement = filter_dataframe_by_selections(df_engagement_raw, filter_selections)
df_stress = filter_dataframe_by_selections(df_stress_raw, filter_selections)


# --- Main Dashboard Title & Introduction ---
st.title(current_lang_texts.get("dashboard_title"))
st.markdown(current_lang_texts.get("dashboard_subtitle"))
st.caption(current_lang_texts.get("alignment_note"))
st.markdown("---")
st.info(current_lang_texts.get("psych_safety_note"))
st.markdown("---")

# --- CORE MODULES ---

# 1. Laboral Stability Panel
st.header(current_lang_texts.get("stability_panel_title"))
if not df_stability.empty:
    col1, col2, col3, col4 = st.columns(4) # Added a column for 18m retention

    with col1:
        avg_rotation = df_stability[config.COLUMN_ROTATION_RATE].mean() if config.COLUMN_ROTATION_RATE in df_stability.columns else float('nan')
        st.plotly_chart(viz.create_kpi_gauge(
            value=avg_rotation,
            title_key="rotation_rate_gauge",
            lang=st.session_state.selected_lang_code,
            unit="%",
            higher_is_worse=True,
            low_threshold=config.ROTATION_RATE_LOW,
            medium_threshold=config.ROTATION_RATE_MEDIUM,
            high_threshold=config.ROTATION_RATE_HIGH, # Also used as critical
            target_threshold=config.ROTATION_RATE_LOW # Example: Target is the low threshold
        ), use_container_width=True)
        st.caption(current_lang_texts.get("rotation_gauge_caption", "Lower rotation is generally better. Target: {}%.").format(config.ROTATION_RATE_LOW))

    with col2:
        ret_6m = df_stability[config.COLUMN_RETENTION_6M].mean() if config.COLUMN_RETENTION_6M in df_stability.columns else float('nan')
        viz.display_metric_card(st, "retention_6m", ret_6m, lang=st.session_state.selected_lang_code, unit="%", higher_is_better=True, target_value=config.RETENTION_TARGET_HIGH)

    with col3:
        ret_12m = df_stability[config.COLUMN_RETENTION_12M].mean() if config.COLUMN_RETENTION_12M in df_stability.columns else float('nan')
        viz.display_metric_card(st, "retention_12m", ret_12m, lang=st.session_state.selected_lang_code, unit="%", higher_is_better=True, target_value=config.RETENTION_TARGET_HIGH)

    with col4:
        ret_18m = df_stability[config.COLUMN_RETENTION_18M].mean() if config.COLUMN_RETENTION_18M in df_stability.columns else float('nan')
        viz.display_metric_card(st, "retention_18m", ret_18m, lang=st.session_state.selected_lang_code, unit="%", higher_is_better=True, target_value=config.RETENTION_TARGET_HIGH)

    if config.COLUMN_DATE in df_stability.columns and \
       config.COLUMN_HIRES in df_stability.columns and \
       config.COLUMN_EXITS in df_stability.columns:
        stability_trend_df = df_stability.copy()
        if not pd.api.types.is_datetime64_any_dtype(stability_trend_df[config.COLUMN_DATE]):
            stability_trend_df[config.COLUMN_DATE] = pd.to_datetime(stability_trend_df[config.COLUMN_DATE], errors='coerce')
        stability_trend_df = stability_trend_df.dropna(subset=[config.COLUMN_DATE])

        if not stability_trend_df.empty:
            hires_exits_trend = stability_trend_df.groupby(pd.Grouper(key=config.COLUMN_DATE, freq='M')).agg(
                Hires=(config.COLUMN_HIRES, 'sum'),
                Exits=(config.COLUMN_EXITS, 'sum')
            ).reset_index()
            st.plotly_chart(viz.create_trend_chart(
                hires_exits_trend, config.COLUMN_DATE, ['Hires', 'Exits'],
                "hires_vs_exits_chart", lang=st.session_state.selected_lang_code,
                y_axis_title_key="count_axis", x_axis_title_key="month_axis",
                show_average_line=True
            ), use_container_width=True)
        else:
            st.warning(current_lang_texts.get("no_data_hires_exits"))
    else:
        st.warning(current_lang_texts.get("no_data_hires_exits"))
else:
    st.info(current_lang_texts.get("no_data_available"))
st.markdown("---")


# 2. Safety Pulse Module
st.header(current_lang_texts.get("safety_pulse_title"))
if not df_safety.empty:
    col1, col2, col3 = st.columns(3)

    with col1: # Chart takes more space
        if config.COLUMN_MONTH in df_safety.columns and \
           config.COLUMN_INCIDENTS in df_safety.columns and \
           config.COLUMN_NEAR_MISSES in df_safety.columns:

            safety_summary = df_safety.groupby(config.COLUMN_MONTH, as_index=False).agg(
                Incidents=(config.COLUMN_INCIDENTS, 'sum'),
                Near_Misses=(config.COLUMN_NEAR_MISSES, 'sum')
            )
            # Improved Month Sorting
            try:
                # Attempt to convert to datetime if possible for robust sorting, then format back
                temp_month = pd.to_datetime(safety_summary[config.COLUMN_MONTH], format='%b', errors='coerce') # Try MMM format
                if temp_month.notna().all(): # if all converted
                     safety_summary[config.COLUMN_MONTH] = temp_month.dt.strftime('%b')
                     safety_summary = safety_summary.sort_values(by=config.COLUMN_MONTH, key=lambda x: pd.to_datetime(x, format='%b'))
                else: # Fallback to simple categorical sorting
                    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    safety_summary[config.COLUMN_MONTH] = pd.Categorical(safety_summary[config.COLUMN_MONTH], categories=month_order, ordered=True)
                    safety_summary = safety_summary.sort_values(config.COLUMN_MONTH)
            except Exception:
                safety_summary = safety_summary.sort_values(by=config.COLUMN_MONTH, errors='ignore') # Basic sort if all else fails


            st.plotly_chart(viz.create_comparison_bar_chart(
                safety_summary, config.COLUMN_MONTH, ['Incidents', 'Near_Misses'],
                "monthly_incidents_chart", lang=st.session_state.selected_lang_code,
                x_axis_title_key="month_axis", y_axis_title_key="count_axis", barmode='stack' # Stacked for total safety events
            ), use_container_width=True)
        else:
            st.warning(current_lang_texts.get("no_data_incidents_near_misses"))

    with col2:
        days_no_accidents = df_safety[config.COLUMN_DAYS_WITHOUT_ACCIDENTS].max() if config.COLUMN_DAYS_WITHOUT_ACCIDENTS in df_safety.columns else "N/A"
        viz.display_metric_card(st, "days_without_accidents_metric", days_no_accidents, lang=st.session_state.selected_lang_code, unit=" " + current_lang_texts.get("days_label", "days"), higher_is_better=True)

    with col3:
        active_alerts_count = df_safety[config.COLUMN_ACTIVE_ALERTS].sum() if config.COLUMN_ACTIVE_ALERTS in df_safety.columns else "N/A"
        viz.display_metric_card(st, "active_safety_alerts_metric", active_alerts_count, lang=st.session_state.selected_lang_code, higher_is_better=False, target_value=0)
else:
    st.info(current_lang_texts.get("no_data_available"))
st.markdown("---")


# 3. Employee Engagement & Commitment
st.header(current_lang_texts.get("engagement_title"))
if not df_engagement.empty:
    col1, col2 = st.columns([3,2]) # Radar chart larger

    with col1:
        radar_data_list = []
        for internal_key, data_col_name in config.ENGAGEMENT_RADAR_DATA_COLS.items():
            if data_col_name in df_engagement.columns:
                avg_val = df_engagement[data_col_name].mean()
                label_key = config.ENGAGEMENT_RADAR_LABELS_KEYS.get(internal_key, data_col_name) # Get display label key
                radar_data_list.append({
                    "Dimension": current_lang_texts.get(label_key, label_key.replace('_label','').replace('_data','').title()),
                    "Score": avg_val if pd.notna(avg_val) else 0
                })
        
        if radar_data_list:
            df_radar = pd.DataFrame(radar_data_list)
            if not df_radar.empty:
                st.plotly_chart(viz.create_enhanced_radar_chart(
                    df_radar, "Dimension", "Score",
                    "engagement_dimensions_radar", lang=st.session_state.selected_lang_code,
                    range_max=5 # Assuming a 1-5 scale for these dimensions
                ), use_container_width=True)
            else:
                st.warning(current_lang_texts.get("no_data_radar"))
        else:
            st.warning(current_lang_texts.get("no_data_radar_columns"))

    with col2:
        climate_score = df_engagement[config.COLUMN_LABOR_CLIMATE].mean() if config.COLUMN_LABOR_CLIMATE in df_engagement.columns else float('nan')
        viz.display_metric_card(st, "labor_climate_score_metric", climate_score, lang=st.session_state.selected_lang_code, unit="", higher_is_better=True, target_value=config.CLIMATE_SCORE_TARGET_HIGH)

        nps = df_engagement[config.COLUMN_ENPS].mean() if config.COLUMN_ENPS in df_engagement.columns else float('nan')
        viz.display_metric_card(st, "enps_metric", nps, lang=st.session_state.selected_lang_code, unit="", higher_is_better=True, target_value=config.ENPS_TARGET_HIGH)

        participation = df_engagement[config.COLUMN_PARTICIPATION].mean() if config.COLUMN_PARTICIPATION in df_engagement.columns else float('nan')
        viz.display_metric_card(st, "survey_participation_metric", participation, lang=st.session_state.selected_lang_code, unit="%", higher_is_better=True, target_value=80) # Example target
        
        recognitions = df_engagement[config.COLUMN_RECOGNITIONS_COUNT].sum() if config.COLUMN_RECOGNITIONS_COUNT in df_engagement.columns else float('nan')
        viz.display_metric_card(st, "recognitions_count_metric", recognitions, lang=st.session_state.selected_lang_code, unit="", higher_is_better=True)
else:
    st.info(current_lang_texts.get("no_data_available"))
st.markdown("---")


# 4. Operational Stress Dashboard
st.header(current_lang_texts.get("stress_title"))
if not df_stress.empty:
    col1, col2 = st.columns([3,2]) # Chart first, then semaforo
    with col1:
        if config.COLUMN_DATE in df_stress.columns and \
           config.COLUMN_OVERTIME_HOURS in df_stress.columns and \
           config.COLUMN_UNFILLED_SHIFTS in df_stress.columns:
            
            stress_trend_df = df_stress.copy()
            if not pd.api.types.is_datetime64_any_dtype(stress_trend_df[config.COLUMN_DATE]):
                stress_trend_df[config.COLUMN_DATE] = pd.to_datetime(stress_trend_df[config.COLUMN_DATE], errors='coerce')
            stress_trend_df = stress_trend_df.dropna(subset=[config.COLUMN_DATE])
            
            if not stress_trend_df.empty:
                stress_summary_monthly = stress_trend_df.groupby(pd.Grouper(key=config.COLUMN_DATE, freq='M')).agg(
                    Total_Overtime=(config.COLUMN_OVERTIME_HOURS, 'sum'),
                    Total_Unfilled_Shifts=(config.COLUMN_UNFILLED_SHIFTS, 'sum')
                ).reset_index()
                st.plotly_chart(viz.create_comparison_bar_chart(
                    stress_summary_monthly, config.COLUMN_DATE, ['Total_Overtime', 'Total_Unfilled_Shifts'],
                    "monthly_shift_load_chart", lang=st.session_state.selected_lang_code,
                    x_axis_title_key="month_axis", y_axis_title_key="hours_or_shifts_label", barmode='group' # Grouped for comparison
                ), use_container_width=True)
            else:
                st.warning(current_lang_texts.get("no_data_shift_load"))
        else:
            st.warning(current_lang_texts.get("no_data_shift_load"))

    with col2:
        avg_stress_level = df_stress[config.COLUMN_STRESS_LEVEL_SURVEY].mean() if config.COLUMN_STRESS_LEVEL_SURVEY in df_stress.columns else float('nan')
        st.plotly_chart(viz.create_stress_semaforo_visual(avg_stress_level, lang=st.session_state.selected_lang_code), use_container_width=True)
        semaforo_caption_key = "stress_semaforo_caption"
        st.caption(current_lang_texts.get(semaforo_caption_key).format(
            config.STRESS_LEVEL_LOW_THRESHOLD,
            config.STRESS_LEVEL_LOW_THRESHOLD,
            config.STRESS_LEVEL_MEDIUM_THRESHOLD,
            config.STRESS_LEVEL_MEDIUM_THRESHOLD
        ))


    if config.COLUMN_DATE in df_stress.columns and \
       config.COLUMN_WORKLOAD_PERCEPTION in df_stress.columns and \
       config.COLUMN_PSYCH_SIGNAL_SCORE in df_stress.columns:

        workload_psych_df = df_stress.copy()
        if not pd.api.types.is_datetime64_any_dtype(workload_psych_df[config.COLUMN_DATE]):
            workload_psych_df[config.COLUMN_DATE] = pd.to_datetime(workload_psych_df[config.COLUMN_DATE], errors='coerce')
        workload_psych_df = workload_psych_df.dropna(subset=[config.COLUMN_DATE])

        if not workload_psych_df.empty:
            workload_psych_trend = workload_psych_df.groupby(pd.Grouper(key=config.COLUMN_DATE, freq='M')).agg(
                Workload_Perception=(config.COLUMN_WORKLOAD_PERCEPTION, 'mean'),
                Psychological_Signals=(config.COLUMN_PSYCH_SIGNAL_SCORE, 'mean') # Changed alias for clarity
            ).reset_index()
            st.plotly_chart(viz.create_trend_chart(
                workload_psych_trend, config.COLUMN_DATE, ['Workload_Perception', 'Psychological_Signals'],
                "workload_vs_psych_chart", lang=st.session_state.selected_lang_code,
                x_axis_title_key="month_axis", y_axis_title_key="average_score_label",
                show_average_line=True
            ), use_container_width=True)
        else:
            st.warning(current_lang_texts.get("no_data_workload_psych"))
    else:
        st.warning(current_lang_texts.get("no_data_workload_psych"))
else:
    st.info(current_lang_texts.get("no_data_available"))
st.markdown("---")


# 5. Interactive Plant Map (Placeholder)
st.header(current_lang_texts.get("plant_map_title"))
st.markdown(config.PLACEHOLDER_TEXT_PLANT_MAP, unsafe_allow_html=True)
st.warning("This feature requires significant additional development including mapping libraries (e.g., Plotly Mapbox, Folium) and potentially real-time data integration. Accessibility for map interactions will need careful design to ensure usability for all.")
st.markdown("---")

# 6. Predictive AI Insights (Placeholder)
st.header(current_lang_texts.get("ai_insights_title"))
st.markdown(config.PLACEHOLDER_TEXT_AI_INSIGHTS, unsafe_allow_html=True)
st.warning("This feature requires data science expertise, model development (e.g., scikit-learn, TensorFlow), and robust data pipelines. Ensuring fairness, transparency, and ethical considerations in AI outputs is critical.")
st.markdown("---")

# --- Optional & Strategic Modules ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"## {current_lang_texts.get('optional_modules_header')}")
if st.sidebar.checkbox(current_lang_texts.get('show_optional_modules')):
    with st.expander(current_lang_texts.get('optional_modules_title'), expanded=False):
        st.markdown(current_lang_texts.get('optional_modules_list'))

st.sidebar.markdown("---")
st.sidebar.caption(f"{current_lang_texts.get('dashboard_title')} v0.3.0 (MVP - Viz Enhanced)")
st.sidebar.caption("Built with Streamlit, Plotly, and Pandas.")
