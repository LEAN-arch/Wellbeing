import streamlit as st
import pandas as pd
import visualizations as viz
import config

# --- Page Configuration (Applied once at the top) ---
# Determine initial language for page config (before selector is available)
initial_lang_code = config.LANG
if 'selected_lang_code' in st.session_state: # If language has been selected already
    initial_lang_code = st.session_state.selected_lang_code

st.set_page_config(
    page_title=config.TEXT_STRINGS[initial_lang_code].get("dashboard_title", config.APP_TITLE),
    page_icon=config.APP_ICON,
    layout="wide"
)

# --- Language Selection ---
st.sidebar.markdown("---")
available_langs = list(config.TEXT_STRINGS.keys())

# Use session state to persist language choice
if 'selected_lang_code' not in st.session_state:
    st.session_state.selected_lang_code = config.LANG # Default from config

def update_lang():
    st.session_state.selected_lang_code = st.session_state._lang_selector # _ to avoid direct manipulation by widget
    # No need to rerun explicitly for selectbox, it triggers on change

selected_lang_code = st.sidebar.selectbox(
    label=config.TEXT_STRINGS["EN"].get("language_selector", "Language / Idioma:") + " / " + config.TEXT_STRINGS["ES"].get("language_selector", "Idioma / Language:"),
    options=available_langs,
    index=available_langs.index(st.session_state.selected_lang_code),
    format_func=lambda x: "English" if x == "EN" else "Español" if x == "ES" else x,
    key="_lang_selector", # Use a private key for session state tracking
    on_change=update_lang # Callback to update session state
)
current_lang_texts = config.TEXT_STRINGS[st.session_state.selected_lang_code]


# --- Data Loading Functions with Caching ---
@st.cache_data
def load_data(file_path, date_cols=None):
    try:
        # Assumes CSV files are in the same directory as app.py
        df = pd.read_csv(file_path, parse_dates=date_cols if date_cols else False)
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].notna().any():
                 df[col] = df[col].astype(str).str.strip()
        return df
    except FileNotFoundError:
        st.error(f"Error: File '{file_path}' not found. Please ensure it's in the same directory as the application (app.py).")
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

def get_unique_options(df, column_name):
    if not df.empty and column_name in df.columns:
        return sorted(df[column_name].dropna().astype(str).unique()) # Ensure options are strings for multiselect
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
        return df.copy()
    
    filtered_df = df.copy()
    for col_config_key, selected_values in selections_map.items():
        # Get actual column name from config based on a standardized key
        actual_col_name = getattr(config, col_config_key, None)
        if actual_col_name and selected_values and actual_col_name in filtered_df.columns:
            # Ensure selected_values are strings if the column is object/string type
            if filtered_df[actual_col_name].dtype == 'object':
                selected_values_str = [str(v) for v in selected_values]
                filtered_df = filtered_df[filtered_df[actual_col_name].isin(selected_values_str)]
            else:
                filtered_df = filtered_df[filtered_df[actual_col_name].isin(selected_values)]
    return filtered_df

# Map config constants (which hold column names) to the selected filter values
selections = {
    'COLUMN_SITE': selected_sites, # Use the string key that corresponds to the attribute name in config
    'COLUMN_REGION': selected_regions,
    'COLUMN_DEPARTMENT': selected_departments,
    'COLUMN_FC': selected_fcs,
    'COLUMN_SHIFT': selected_shifts
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
st.info(current_lang_texts.get("psych_safety_note"))
st.markdown("---")


# --- CORE MODULES ---

# 1. Laboral Stability Panel
st.header(current_lang_texts.get("stability_panel_title"))
if not filtered_stability.empty:
    col1, col2, col3 = st.columns(3)
    avg_rotation = filtered_stability[config.COLUMN_ROTATION_RATE].mean() if config.COLUMN_ROTATION_RATE in filtered_stability.columns else 0
    with col1:
        st.plotly_chart(viz.create_kpi_gauge(avg_rotation, "rotation_rate_gauge", lang=st.session_state.selected_lang_code,
                                            low_threshold=config.ROTATION_RATE_LOW_THRESHOLD,
                                            high_threshold=config.ROTATION_RATE_HIGH_THRESHOLD),
                        use_container_width=True)
        st.caption(f"{current_lang_texts.get('thresholds_label', 'Thresholds')}: {current_lang_texts.get('low_label', 'Low')} < {config.ROTATION_RATE_LOW_THRESHOLD}%, {current_lang_texts.get('high_label', 'High')} > {config.ROTATION_RATE_HIGH_THRESHOLD}%")

    with col2:
        ret_6m = filtered_stability[config.COLUMN_RETENTION_6M].mean() if config.COLUMN_RETENTION_6M in filtered_stability.columns else float('nan')
        viz.display_metric_card(st, "retention_6m", ret_6m, lang=st.session_state.selected_lang_code)
    with col3:
        ret_12m = filtered_stability[config.COLUMN_RETENTION_12M].mean() if config.COLUMN_RETENTION_12M in filtered_stability.columns else float('nan')
        viz.display_metric_card(st, "retention_12m", ret_12m, lang=st.session_state.selected_lang_code)

    # Display 18-month retention below, or adjust layout
    ret_18m = filtered_stability[config.COLUMN_RETENTION_18M].mean() if config.COLUMN_RETENTION_18M in filtered_stability.columns else float('nan')
    viz.display_metric_card(st, "retention_18m", ret_18m, lang=st.session_state.selected_lang_code)


    if config.COLUMN_DATE in filtered_stability.columns and \
       config.COLUMN_HIRES in filtered_stability.columns and \
       config.COLUMN_EXITS in filtered_stability.columns:
        stability_trend_df = filtered_stability.copy()
        if not pd.api.types.is_datetime64_any_dtype(stability_trend_df[config.COLUMN_DATE]):
            stability_trend_df[config.COLUMN_DATE] = pd.to_datetime(stability_trend_df[config.COLUMN_DATE], errors='coerce')
        stability_trend_df = stability_trend_df.dropna(subset=[config.COLUMN_DATE])


        if not stability_trend_df.empty:
            hires_exits_trend = stability_trend_df.groupby(pd.Grouper(key=config.COLUMN_DATE, freq='M')).agg(
                Hires=(config.COLUMN_HIRES, 'sum'),
                Exits=(config.COLUMN_EXITS, 'sum')
            ).reset_index()
            st.plotly_chart(viz.create_line_chart(hires_exits_trend, config.COLUMN_DATE, ['Hires', 'Exits'],
                                                "hires_vs_exits_chart", lang=st.session_state.selected_lang_code,
                                                x_axis_title_key="month_axis"), use_container_width=True)
        else:
            st.warning(current_lang_texts.get("no_data_hires_exits"))
    else:
        st.warning(current_lang_texts.get("no_data_hires_exits"))
else:
    st.info(current_lang_texts.get("no_data_available"))
st.markdown("---")


# 2. Safety Pulse Module
st.header(current_lang_texts.get("safety_pulse_title"))
if not filtered_safety.empty:
    cols = st.columns(3) # Use a list for easier access

    with cols[0]:
        if config.COLUMN_MONTH in filtered_safety.columns and \
           config.COLUMN_INCIDENTS in filtered_safety.columns and \
           config.COLUMN_NEAR_MISSES in filtered_safety.columns:

            safety_summary = filtered_safety.groupby(config.COLUMN_MONTH, as_index=False).agg(
                Incidents=(config.COLUMN_INCIDENTS, 'sum'),
                Near_Misses=(config.COLUMN_NEAR_MISSES, 'sum')
            )
            try:
                month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                # Create a dictionary mapping from textual month to numerical for sorting
                month_to_num = {month: i for i, month in enumerate(month_order)}
                # Apply only if all months in data are in our map
                if all(m in month_to_num for m in safety_summary[config.COLUMN_MONTH].unique()):
                    safety_summary['month_num'] = safety_summary[config.COLUMN_MONTH].map(month_to_num)
                    safety_summary = safety_summary.sort_values('month_num').drop(columns=['month_num'])
                else: # If not all months match, try a simple sort (might not be chronological for names)
                    safety_summary = safety_summary.sort_values(by=config.COLUMN_MONTH)
            except Exception:
                 safety_summary = safety_summary.sort_values(by=config.COLUMN_MONTH, errors='ignore')

            st.plotly_chart(viz.create_bar_chart(safety_summary, config.COLUMN_MONTH, ['Incidents', 'Near_Misses'],
                                                "monthly_incidents_chart", lang=st.session_state.selected_lang_code,
                                                x_axis_title_key="month_axis"), use_container_width=True)
        else:
            st.warning(current_lang_texts.get("no_data_incidents_near_misses"))

    with cols[1]:
        days_no_accidents = filtered_safety[config.COLUMN_DAYS_WITHOUT_ACCIDENTS].max() if config.COLUMN_DAYS_WITHOUT_ACCIDENTS in filtered_safety.columns else "N/A"
        viz.display_metric_card(st, "days_without_accidents_metric", days_no_accidents, lang=st.session_state.selected_lang_code)

    with cols[2]:
        active_alerts_count = filtered_safety[config.COLUMN_ACTIVE_ALERTS].sum() if config.COLUMN_ACTIVE_ALERTS in filtered_safety.columns else "N/A"
        viz.display_metric_card(st, "active_safety_alerts_metric", active_alerts_count, lang=st.session_state.selected_lang_code)
else:
    st.info(current_lang_texts.get("no_data_available"))
st.markdown("---")


# 3. Employee Engagement & Commitment
st.header(current_lang_texts.get("engagement_title"))
if not filtered_engagement.empty:
    col1, col2 = st.columns([2,1])

    with col1:
        radar_data_list = []
        radar_categories_actual_cols = [] # Store actual column names found in df
        
        # Use keys from ENGAGEMENT_RADAR_CATEGORIES_KEYS to find corresponding data columns
        # config.COLUMN_INITIATIVE, config.COLUMN_PUNCTUALITY, config.COLUMN_RECOGNITION, config.COLUMN_FEEDBACK
        # are now the actual column names. The ENGAGEMENT_RADAR_CATEGORIES_KEYS maps these to display labels.
        
        potential_radar_cols = {
            config.COLUMN_INITIATIVE: config.ENGAGEMENT_RADAR_CATEGORIES_KEYS.get("initiative", "initiative_label"),
            config.COLUMN_PUNCTUALITY: config.ENGAGEMENT_RADAR_CATEGORIES_KEYS.get("punctuality", "punctuality_label"),
            config.COLUMN_RECOGNITION: config.ENGAGEMENT_RADAR_CATEGORIES_KEYS.get("recognition_data", "recognition_label"),
            config.COLUMN_FEEDBACK: config.ENGAGEMENT_RADAR_CATEGORIES_KEYS.get("feedback_data", "feedback_label")
        }

        for data_col_name, label_key in potential_radar_cols.items():
            if data_col_name in filtered_engagement.columns:
                avg_val = filtered_engagement[data_col_name].mean()
                radar_data_list.append({
                    "Dimension": current_lang_texts.get(label_key, label_key.replace('_label','').title()),
                    "Score": avg_val if pd.notna(avg_val) else 0
                })
                radar_categories_actual_cols.append(data_col_name)
        
        if radar_data_list:
            df_radar = pd.DataFrame(radar_data_list)
            if not df_radar.empty:
                st.plotly_chart(viz.create_radar_chart(df_radar, "Dimension", "Score",
                                                        "engagement_dimensions_radar", lang=st.session_state.selected_lang_code),
                                use_container_width=True)
            else: # This case might not be reached if radar_data_list is non-empty
                st.warning(current_lang_texts.get("no_data_radar"))
        else:
            st.warning(current_lang_texts.get("no_data_radar_columns"))


    with col2:
        climate_score = filtered_engagement[config.COLUMN_LABOR_CLIMATE].mean() if config.COLUMN_LABOR_CLIMATE in filtered_engagement.columns else float('nan')
        viz.display_metric_card(st, "labor_climate_score_metric", climate_score, lang=st.session_state.selected_lang_code)

        nps = filtered_engagement[config.COLUMN_ENPS].mean() if config.COLUMN_ENPS in filtered_engagement.columns else float('nan')
        viz.display_metric_card(st, "enps_metric", nps, lang=st.session_state.selected_lang_code)

        participation = filtered_engagement[config.COLUMN_PARTICIPATION].mean() if config.COLUMN_PARTICIPATION in filtered_engagement.columns else float('nan')
        viz.display_metric_card(st, "survey_participation_metric", participation, lang=st.session_state.selected_lang_code)
        
        recognitions = filtered_engagement[config.COLUMN_RECOGNITIONS_COUNT].sum() if config.COLUMN_RECOGNITIONS_COUNT in filtered_engagement.columns else float('nan')
        viz.display_metric_card(st, "recognitions_count_metric", recognitions, lang=st.session_state.selected_lang_code)
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
            if not pd.api.types.is_datetime64_any_dtype(stress_trend_df[config.COLUMN_DATE]):
                stress_trend_df[config.COLUMN_DATE] = pd.to_datetime(stress_trend_df[config.COLUMN_DATE], errors='coerce')
            stress_trend_df = stress_trend_df.dropna(subset=[config.COLUMN_DATE])
            
            if not stress_trend_df.empty:
                stress_summary_monthly = stress_trend_df.groupby(pd.Grouper(key=config.COLUMN_DATE, freq='M')).agg(
                    Total_Overtime=(config.COLUMN_OVERTIME_HOURS, 'sum'),
                    Total_Unfilled_Shifts=(config.COLUMN_UNFILLED_SHIFTS, 'sum')
                ).reset_index()
                st.plotly_chart(viz.create_bar_chart(stress_summary_monthly, config.COLUMN_DATE, ['Total_Overtime', 'Total_Unfilled_Shifts'],
                                                   "monthly_shift_load_chart", lang=st.session_state.selected_lang_code,
                                                   x_axis_title_key="month_axis", y_axis_title="Hours / Count"), use_container_width=True)
            else:
                st.warning(current_lang_texts.get("no_data_shift_load"))

        else:
            st.warning(current_lang_texts.get("no_data_shift_load"))

    with col2:
        avg_stress_level = filtered_stress[config.COLUMN_STRESS_LEVEL_SURVEY].mean() if config.COLUMN_STRESS_LEVEL_SURVEY in filtered_stress.columns else float('nan')
        semaforo_display = viz.create_stress_semaforo(avg_stress_level, lang=st.session_state.selected_lang_code)
        st.subheader(current_lang_texts.get("overall_stress_indicator"))
        st.markdown(f"<div style='font-size: 2em; text-align: center;'>{semaforo_display}</div>", unsafe_allow_html=True)
        st.caption(
            f"{current_lang_texts.get('stress_low')}: ≤ {config.STRESS_LEVEL_LOW_THRESHOLD:.1f} | "
            f"{current_lang_texts.get('stress_medium')}: >{config.STRESS_LEVEL_LOW_THRESHOLD:.1f} & ≤ {config.STRESS_LEVEL_MEDIUM_THRESHOLD:.1f} | "
            f"{current_lang_texts.get('stress_high')}: > {config.STRESS_LEVEL_MEDIUM_THRESHOLD:.1f}"
        )

    if config.COLUMN_DATE in filtered_stress.columns and \
       config.COLUMN_WORKLOAD_PERCEPTION in filtered_stress.columns and \
       config.COLUMN_PSYCH_SIGNAL_SCORE in filtered_stress.columns:

        workload_psych_df = filtered_stress.copy()
        if not pd.api.types.is_datetime64_any_dtype(workload_psych_df[config.COLUMN_DATE]):
            workload_psych_df[config.COLUMN_DATE] = pd.to_datetime(workload_psych_df[config.COLUMN_DATE], errors='coerce')
        workload_psych_df = workload_psych_df.dropna(subset=[config.COLUMN_DATE])

        if not workload_psych_df.empty:
            workload_psych_trend = workload_psych_df.groupby(pd.Grouper(key=config.COLUMN_DATE, freq='M')).agg(
                Workload_Perception=(config.COLUMN_WORKLOAD_PERCEPTION, 'mean'),
                Psych_Signals=(config.COLUMN_PSYCH_SIGNAL_SCORE, 'mean')
            ).reset_index()
            st.plotly_chart(viz.create_line_chart(workload_psych_trend, config.COLUMN_DATE, ['Workload_Perception', 'Psych_Signals'],
                                                "workload_vs_psych_chart", lang=st.session_state.selected_lang_code,
                                                x_axis_title_key="month_axis", y_axis_title="Average Score"), use_container_width=True)
        else:
            st.warning(current_lang_texts.get("no_data_workload_psych"))
    else:
        st.warning(current_lang_texts.get("no_data_workload_psych"))
else:
    st.info(current_lang_texts.get("no_data_available"))
st.markdown("---")


# 5. Interactive Plant Map (Placeholder)
st.header(current_lang_texts.get("plant_map_title"))
st.markdown(config.PLACEHOLDER_TEXT_PLANT_MAP, unsafe_allow_html=True) # Allow basic HTML in placeholder if needed
st.warning("This feature requires significant additional development including mapping libraries (e.g., Plotly Mapbox, Folium) and potentially real-time data integration. Accessibility for map interactions will need careful design to ensure usability for all.")
st.markdown("---")

# 6. Predictive AI Insights (Placeholder)
st.header(current_lang_texts.get("ai_insights_title"))
st.markdown(config.PLACEHOLDER_TEXT_AI_INSIGHTS, unsafe_allow_html=True)
st.warning("This feature requires data science expertise, model development (e.g., scikit-learn, TensorFlow), and robust data pipelines. Ensuring fairness, transparency, and ethical considerations in AI outputs is critical.")
st.markdown("---")

# --- Optional & Strategic Modules (Placeholders in Sidebar) ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"## {current_lang_texts.get('optional_modules_header')}")
if st.sidebar.checkbox(current_lang_texts.get('show_optional_modules')):
    with st.expander(current_lang_texts.get('optional_modules_title'), expanded=False):
        # Using markdown for better formatting of list in expander
        st.markdown(current_lang_texts.get('optional_modules_list'))

st.sidebar.markdown("---")
st.sidebar.caption(f"{current_lang_texts.get('dashboard_title')} v0.2.1 (MVP - Path Simplified)")
st.sidebar.caption("Built with Streamlit, Plotly, and Pandas.")
