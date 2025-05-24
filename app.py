import streamlit as st
import pandas as pd
import visualizations as viz # Your custom visualization functions
import config # Your configuration file
import numpy as np # For isnan etc.

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

# Using HTML to put language text side-by-side within the label
lang_label_html = f"**{config.TEXT_STRINGS['EN'].get('language_selector', 'Language')} / {config.TEXT_STRINGS['ES'].get('language_selector', 'Idioma')}**"

selected_lang_code = st.sidebar.selectbox(
    label=lang_label_html, # Use HTML for label
    options=available_langs,
    index=available_langs.index(st.session_state.selected_lang_code),
    format_func=lambda x: "English" if x == "EN" else "Español" if x == "ES" else x, # Nicer display names
    key="_lang_selector",
    on_change=update_lang
)
# Get current language texts using session state value
current_lang_texts = config.TEXT_STRINGS.get(st.session_state.selected_lang_code, config.TEXT_STRINGS["EN"]) # Fallback to EN

# --- Data Loading Functions with Caching ---
# Pass coluumn map to load_data to rename columns during loading if needed, based on config
# But for this MVP, columns match config, so direct load is okay
@st.cache_data
def load_data(file_path, date_cols=None):
    try:
        # Assumes CSV files are in the same directory as app.py
        df = pd.read_csv(file_path, parse_dates=date_cols if date_cols else False)
        for col in df.select_dtypes(include=['object']).columns:
             if df[col].notna().any():
                 df[col] = df[col].astype(str).str.strip() # Clean strings
        return df
    except FileNotFoundError:
        st.error(current_lang_texts.get("error_loading_data", "Error loading data from file: {}").format(file_path) + f". Please ensure '{file_path}' is in the same directory as app.py.")
        return pd.DataFrame()
    except Exception as e:
        st.error(current_lang_texts.get("error_loading_data", "Error loading data from file: {}").format(file_path) + f" - {e}")
        return pd.DataFrame()


# Use COLUMN_MAP values to load data
df_stability_raw = load_data(config.STABILITY_DATA_FILE, date_cols=[config.COLUMN_MAP["date"]])
df_safety_raw = load_data(config.SAFETY_DATA_FILE, date_cols=[config.COLUMN_MAP["month"]]) # If month is like 'YYYY-MM'
df_engagement_raw = load_data(config.ENGAGEMENT_DATA_FILE)
df_stress_raw = load_data(config.STRESS_DATA_FILE, date_cols=[config.COLUMN_MAP["date"]])


# --- Sidebar Filters ---
st.sidebar.header(current_lang_texts.get("filters_header", "Filters"))

def get_unique_options(df, column_key):
    """Gets unique sorted string options based on a column key from COLUMN_MAP."""
    column_name = config.COLUMN_MAP.get(column_key)
    if not df.empty and column_name and column_name in df.columns:
        return sorted(df[column_name].dropna().astype(str).unique())
    return []

# Get filter options using column keys
sites = get_unique_options(df_stability_raw, "site")
regions = get_unique_options(df_stability_raw, "region")
departments = get_unique_options(df_stability_raw, "department")
fcs = get_unique_options(df_stability_raw, "fc")
shifts = get_unique_options(df_stability_raw, "shift")

selected_sites = st.sidebar.multiselect(current_lang_texts.get("select_site"), options=sites, default=config.DEFAULT_SITES)
selected_regions = st.sidebar.multiselect(current_lang_texts.get("select_region"), options=regions, default=config.DEFAULT_REGIONS)
selected_departments = st.sidebar.multiselect(current_lang_texts.get("select_department"), options=departments, default=config.DEFAULT_DEPARTMENTS)
selected_fcs = st.sidebar.multiselect(current_lang_texts.get("select_fc"), options=fcs, default=config.DEFAULT_FUNCTIONAL_CATEGORIES)
selected_shifts = st.sidebar.multiselect(current_lang_texts.get("select_shift"), options=shifts, default=config.DEFAULT_SHIFTS)

# --- Filter DataFrames Utility ---
def filter_dataframe_by_selections(df, column_map, current_selections):
    """Filters a DataFrame based on selections using COLUMN_MAP."""
    if df.empty:
        return df.copy()
    
    filtered_df = df.copy()
    for filter_col_key, selected_values in current_selections.items():
        actual_col_name = column_map.get(filter_col_key) # Get actual name from map
        if actual_col_name and selected_values and actual_col_name in filtered_df.columns:
             # Filter applies to the *actual* column name in the dataframe
             if filtered_df[actual_col_name].dtype == 'object' or pd.api.types.is_string_dtype(filtered_df[actual_col_name]):
                 selected_values_str = [str(v) for v in selected_values]
                 filtered_df = filtered_df[filtered_df[actual_col_name].astype(str).isin(selected_values_str)]
             else: # For numerical or other types if exact match is needed
                 filtered_df = filtered_df[filtered_df[actual_col_name].isin(selected_values)]

    return filtered_df

# Define the selection criteria based on column keys for filtering
filter_selections_map = {
    'site': selected_sites,
    'region': selected_regions,
    'department': selected_departments,
    'fc': selected_fcs,
    'shift': selected_shifts
}

df_stability = filter_dataframe_by_selections(df_stability_raw, config.COLUMN_MAP, filter_selections_map)
df_safety = filter_dataframe_by_selections(df_safety_raw, config.COLUMN_MAP, filter_selections_map)
df_engagement = filter_dataframe_by_selections(df_engagement_raw, config.COLUMN_MAP, filter_selections_map)
df_stress = filter_dataframe_by_selections(df_stress_raw, config.COLUMN_MAP, filter_selections_map)


# --- Helper to get "Previous Period" data (Simplified for MVP) ---
# In a real system, this logic would be much more sophisticated
# (e.g., compare current month to previous month, or current quarter to previous quarter)
# For sample data, we'll just grab some arbitrary values that could represent previous state.
def get_previous_value(df_raw, column_key, current_filters, comparison_date=None):
     # This needs complex logic to filter raw data for a _prior_ period matching
     # the dimensions of the _current_ filtered data.
     # For MVP, return None, or a fixed arbitrary value for demonstration.
     # Returning None means delta is not shown.
     return None # Simplest MVP implementation

# --- Main Dashboard Title & Introduction ---
st.title(current_lang_texts.get("dashboard_title", config.APP_TITLE))
st.markdown(current_lang_texts.get("dashboard_subtitle", "Human-Centered Intelligence System..."))
st.caption(current_lang_texts.get("alignment_note", "Aligned with NOM-035, ISO 45003..."))
st.markdown("---")
st.info(current_lang_texts.get("psych_safety_note", config.TEXT_STRINGS["EN"]["psych_safety_note"])) # Use config directly as fallback
st.markdown("---")


# --- CORE MODULES ---

# 1. Laboral Stability Panel
st.header(current_lang_texts.get("stability_panel_title"))
if not df_stability.empty:
    metric_cols = st.columns(4)

    # Rotation Rate Gauge & Metric Card
    rotation_col = config.COLUMN_MAP["rotation_rate"]
    avg_rotation = df_stability[rotation_col].mean() if rotation_col in df_stability.columns else float('nan')
    # Add dummy previous value for demo delta
    prev_avg_rotation = avg_rotation + np.random.uniform(-5, 5) if pd.notna(avg_rotation) else None

    with metric_cols[0]:
        # Gauge - showing status
        st.plotly_chart(viz.create_kpi_gauge(
            value=avg_rotation,
            title_key="rotation_rate_gauge",
            lang=st.session_state.selected_lang_code,
            unit="%",
            higher_is_worse=True,
            threshold_good=config.ROTATION_RATE_THRESHOLD_GOOD,
            threshold_warning=config.ROTATION_RATE_THRESHOLD_WARNING,
            target_line_value=config.ROTATION_RATE_TARGET,
            previous_value=prev_avg_rotation # Pass dummy previous value
        ), use_container_width=True)

        # Use metric card to also show delta and potential icon
        viz.display_metric_card(st, "rotation_rate_gauge", avg_rotation, lang=st.session_state.selected_lang_code,
                               unit="%", higher_is_better=False, target_value=config.ROTATION_RATE_TARGET, # For target line on card if desired
                               threshold_good=config.ROTATION_RATE_THRESHOLD_GOOD, threshold_warning=config.ROTATION_RATE_THRESHOLD_WARNING,
                               previous_value=prev_avg_rotation) # Use gauge title key for metric card label

    # Retention Metrics (Cards only for space)
    ret_6m_col = config.COLUMN_MAP["retention_6m"]
    ret_12m_col = config.COLUMN_MAP["retention_12m"]
    ret_18m_col = config.COLUMN_MAP["retention_18m"]
    
    ret_6m = df_stability[ret_6m_col].mean() if ret_6m_col in df_stability.columns else float('nan')
    ret_12m = df_stability[ret_12m_col].mean() if ret_12m_col in df_stability.columns else float('nan')
    ret_18m = df_stability[ret_18m_col].mean() if ret_18m_col in df_stability.columns else float('nan')

    # Add dummy previous retention values for demo
    prev_ret_6m = ret_6m + np.random.uniform(-2, 2) if pd.notna(ret_6m) else None
    prev_ret_12m = ret_12m + np.random.uniform(-3, 3) if pd.notna(ret_12m) else None
    prev_ret_18m = ret_18m + np.random.uniform(-4, 4) if pd.notna(ret_18m) else None


    with metric_cols[1]:
        viz.display_metric_card(st, "retention_6m_metric", ret_6m, lang=st.session_state.selected_lang_code, unit="%", higher_is_better=True,
                               target_value=config.RETENTION_THRESHOLD_GOOD, threshold_warning=config.RETENTION_THRESHOLD_WARNING,
                               previous_value=prev_ret_6m) # Pass dummy previous
    with metric_cols[2]:
        viz.display_metric_card(st, "retention_12m_metric", ret_12m, lang=st.session_state.selected_lang_code, unit="%", higher_is_better=True,
                               target_value=config.RETENTION_THRESHOLD_GOOD, threshold_warning=config.RETENTION_THRESHOLD_WARNING,
                               previous_value=prev_ret_12m) # Pass dummy previous
    with metric_cols[3]:
        viz.display_metric_card(st, "retention_18m_metric", ret_18m, lang=st.session_state.selected_lang_code, unit="%", higher_is_better=True,
                               target_value=config.RETENTION_THRESHOLD_GOOD, threshold_warning=config.RETENTION_THRESHOLD_WARNING,
                               previous_value=prev_ret_18m) # Pass dummy previous

    st.markdown("---")

    # Historical trend of hires vs. exits
    date_col = config.COLUMN_MAP["date"]
    hires_col = config.COLUMN_MAP["hires"]
    exits_col = config.COLUMN_MAP["exits"]

    if date_col in df_stability.columns and \
       hires_col in df_stability.columns and \
       exits_col in df_stability.columns:
        
        stability_trend_df = df_stability.copy()
        if not pd.api.types.is_datetime64_any_dtype(stability_trend_df[date_col]):
            stability_trend_df[date_col] = pd.to_datetime(stability_trend_df[date_col], errors='coerce')
        stability_trend_df = stability_trend_df.dropna(subset=[date_col])

        if not stability_trend_df.empty:
            # Group by month for the trend
            hires_exits_trend = stability_trend_df.groupby(pd.Grouper(key=date_col, freq='M')).agg(
                Hires=(hires_col, 'sum'), # Use aliases matching labels/series names
                Exits=(exits_col, 'sum')
            ).reset_index()

            # Add placeholder for target values map if targets for hires/exits exist
            # e.g., hires_targets = {'Hires': 15, 'Exits': 5} # You would define this
            hires_exits_targets = {} # Placeholder - No targets in config for these yet

            # Units for hovertemplate
            units = {'Hires': 'people', 'Exits': 'people'} # Define units for hover

            st.plotly_chart(viz.create_trend_chart(
                hires_exits_trend, date_col, ['Hires', 'Exits'], # Use alias names
                "hires_vs_exits_chart_title", lang=st.session_state.selected_lang_code,
                y_axis_title_key="people_count_label", x_axis_title_key="month_axis_label",
                show_average_line=True,
                target_value_map=hires_exits_targets,
                rolling_avg_window=3, # Example: show 3-month rolling average
                value_col_units=units # Pass units for hover
            ), use_container_width=True)
        else:
            st.warning(current_lang_texts.get("no_data_hires_exits"))
    else:
        st.warning(current_lang_texts.get("no_data_hires_exits"))
st.markdown("---")


# 2. Safety Pulse Module
st.header(current_lang_texts.get("safety_pulse_title"))
if not df_safety.empty:
    chart_col, metrics_col1, metrics_col2 = st.columns([2, 1, 1]) # Chart takes more space

    # Monthly bar graph of incidents & near misses
    month_col = config.COLUMN_MAP["month"]
    incidents_col = config.COLUMN_MAP["incidents"]
    near_misses_col = config.COLUMN_MAP["near_misses"]


    with chart_col: # Chart column
        if month_col in df_safety.columns and incidents_col in df_safety.columns and \
           (near_misses_col in df_safety.columns):

            y_cols_safety_present = [col for col in [incidents_col, near_misses_col] if col in df_safety.columns]
            
            safety_summary = df_safety.groupby(month_col, as_index=False).agg(
                {col: 'sum' for col in y_cols_safety_present}
            )
            # Improve Month Sorting
            try:
                safety_summary[month_col] = pd.Categorical(safety_summary[month_col], categories=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], ordered=True)
                safety_summary = safety_summary.sort_values(month_col)
            except Exception: # Fallback
                 safety_summary = safety_summary.sort_values(by=month_col, errors='ignore')

            st.plotly_chart(viz.create_comparison_bar_chart(
                safety_summary, month_col, y_cols_safety_present, # Use present columns
                "monthly_incidents_chart_title", lang=st.session_state.selected_lang_code,
                x_axis_title_key="month_axis_label", y_axis_title_key="count_label",
                barmode='stack', # Stack incidents and near misses
                show_total_for_stacked=True # Show total safety events per month
            ), use_container_width=True)
        else:
            st.warning(current_lang_texts.get("no_data_incidents_near_misses"))

    # KPI Cards: Days Without Accidents & Active Safety Alerts
    days_no_accidents_col = config.COLUMN_MAP["days_without_accidents"]
    active_alerts_col = config.COLUMN_MAP["active_alerts"]

    # Dummy previous values for metrics
    prev_days_no_accidents = 145 # Example fixed previous
    prev_active_alerts = 3 # Example fixed previous

    with metrics_col1:
        days_no_accidents = df_safety[days_no_accidents_col].max() if days_no_accidents_col in df_safety.columns else "N/A"
        viz.display_metric_card(st, "days_without_accidents_metric", days_no_accidents, lang=st.session_state.selected_lang_code,
                               unit=" " + current_lang_texts.get("days_label", "days"), higher_is_better=True,
                               threshold_good=200, threshold_warning=100, # Example thresholds
                               previous_value=prev_days_no_accidents) # Pass dummy previous

    with metrics_col2:
        active_alerts_count = df_safety[active_alerts_col].sum() if active_alerts_col in df_safety.columns else "N/A"
        viz.display_metric_card(st, "active_safety_alerts_metric", active_alerts_count, lang=st.session_state.selected_lang_code,
                               higher_is_better=False, target_value=0, # Target 0
                               threshold_good=0, threshold_warning=1, # Good is 0, warning if >0
                               previous_value=prev_active_alerts) # Pass dummy previous
else:
    st.info(current_lang_texts.get("no_data_available"))
st.markdown("---")

# 3. Employee Engagement & Commitment
st.header(current_lang_texts.get("engagement_title"))
if not df_engagement.empty:
    col1, col2 = st.columns([3,2]) # Radar chart larger

    with col1: # Radar chart
        radar_data_list = []
        radar_categories_present_localized = []

        # Map config radar data keys to the actual column names present in the dataframe
        actual_radar_cols = {k: v for k, v in config.COLUMN_MAP.items() if k in config.ENGAGEMENT_RADAR_DATA_COLS.keys() and v in df_engagement.columns}

        if actual_radar_cols:
             for internal_key, data_col_name in actual_radar_cols.items():
                avg_val = df_engagement[data_col_name].mean()
                label_key = config.ENGAGEMENT_RADAR_LABELS_KEYS.get(internal_key, data_col_name)
                display_label = current_lang_texts.get(label_key, data_col_name.replace('_data','').replace('_',' ').title())
                
                # Only add if value is meaningful
                if pd.notna(avg_val):
                    radar_data_list.append({
                        "Dimension": display_label,
                        "Score": avg_val
                    })
                    radar_categories_present_localized.append(display_label)

        if radar_data_list:
            df_radar = pd.DataFrame(radar_data_list)

            # Define radar target values for the localized dimension labels
            # You'll need to define these based on what a good score is for each dimension on its scale (assuming 1-5)
            engagement_targets = {
                 current_lang_texts.get(config.ENGAGEMENT_RADAR_LABELS_KEYS['initiative'], 'Initiative'): 4.0,
                 current_lang_texts.get(config.ENGAGEMENT_RADAR_LABELS_KEYS['punctuality'], 'Punctuality'): 4.5,
                 current_lang_texts.get(config.ENGAGEMENT_RADAR_LABELS_KEYS['recognition'], 'Recognition'): 4.0,
                 current_lang_texts.get(config.ENGAGEMENT_RADAR_LABELS_KEYS['feedback'], 'Feedback Culture'): 4.2,
            }
            # Filter targets to only include dimensions actually present in the filtered data
            filtered_radar_targets = {k:v for k,v in engagement_targets.items() if k in radar_categories_present_localized}

            st.plotly_chart(viz.create_enhanced_radar_chart(
                df_radar, "Dimension", "Score",
                "engagement_dimensions_radar_title", lang=st.session_state.selected_lang_code,
                range_max_override=5, # Assuming a 1-5 scale for all dimensions
                target_values_map=filtered_radar_targets # Pass the targets
            ), use_container_width=True)
        elif actual_radar_cols: # Columns were found in data but filtered/aggregated to be empty/NA
             st.warning(current_lang_texts.get("no_data_radar"))
        else: # Required columns for radar not even found in raw data
            st.warning(current_lang_texts.get("no_data_radar_columns"))


    with col2: # Metric Cards
        climate_col = config.COLUMN_MAP["labor_climate_score"]
        nps_col = config.COLUMN_MAP["enps_score"]
        participation_col = config.COLUMN_MAP["participation_rate"]
        recognitions_col = config.COLUMN_MAP["recognitions_count"]

        climate_score = df_engagement[climate_col].mean() if climate_col in df_engagement.columns else float('nan')
        nps = df_engagement[nps_col].mean() if nps_col in df_engagement.columns else float('nan')
        participation = df_engagement[participation_col].mean() if participation_col in df_engagement.columns else float('nan')
        recognitions = df_engagement[recognitions_col].sum() if recognitions_col in df_engagement.columns else float('nan')

        # Dummy previous values for metrics
        prev_climate_score = climate_score + np.random.uniform(-10, 10) if pd.notna(climate_score) else None
        prev_nps = nps + np.random.uniform(-10, 10) if pd.notna(nps) else None
        prev_participation = participation + np.random.uniform(-5, 5) if pd.notna(participation) else None
        prev_recognitions = recognitions + np.random.uniform(-5, 5) if pd.notna(recognitions) else None


        viz.display_metric_card(st, "labor_climate_score_metric", climate_score, lang=st.session_state.selected_lang_code,
                                unit="", higher_is_better=True, target_value=config.CLIMATE_SCORE_THRESHOLD_GOOD,
                                threshold_warning=config.CLIMATE_SCORE_THRESHOLD_WARNING, previous_value=prev_climate_score)

        viz.display_metric_card(st, "enps_metric", nps, lang=st.session_state.selected_lang_code,
                                unit="", higher_is_better=True, target_value=config.ENPS_THRESHOLD_GOOD,
                                threshold_warning=config.ENPS_THRESHOLD_WARNING, previous_value=prev_nps)

        viz.display_metric_card(st, "survey_participation_metric", participation, lang=st.session_state.selected_lang_code,
                                unit="%", higher_is_better=True, target_value=config.PARTICIPATION_THRESHOLD_GOOD,
                                previous_value=prev_participation)
        
        viz.display_metric_card(st, "recognitions_count_metric", recognitions, lang=st.session_state.selected_lang_code,
                                unit="", higher_is_better=True,
                                previous_value=prev_recognitions) # No specific thresholds for count, higher is always better
else:
    st.info(current_lang_texts.get("no_data_available"))
st.markdown("---")


# 4. Operational Stress Dashboard
st.header(current_lang_texts.get("stress_title"))
if not df_stress.empty:
    stress_gauge_col, shift_load_chart_col = st.columns([1, 2]) # Gauge then bar chart

    stress_level_col = config.COLUMN_MAP["stress_level_survey"]
    overtime_col = config.COLUMN_MAP["overtime_hours"]
    unfilled_shifts_col = config.COLUMN_MAP["unfilled_shifts"]
    date_col = config.COLUMN_MAP["date"]
    workload_col = config.COLUMN_MAP["workload_perception"]
    psych_col = config.COLUMN_MAP["psychological_signals"]


    with stress_gauge_col:
        st.subheader(current_lang_texts.get("overall_stress_indicator_title"))
        avg_stress_level = df_stress[stress_level_col].mean() if stress_level_col in df_stress.columns else float('nan')
        st.plotly_chart(viz.create_stress_semaforo_visual(
            avg_stress_level, lang=st.session_state.selected_lang_code
        ), use_container_width=True)
        stress_caption = current_lang_texts.get("stress_semaforo_caption").format(
                max_scale=config.STRESS_LEVEL_MAX_SCALE,
                low=config.STRESS_LEVEL_THRESHOLD_LOW,
                medium=config.STRESS_LEVEL_THRESHOLD_MEDIUM
        )
        st.caption(stress_caption)


    with shift_load_chart_col: # Shift Load Chart
        if date_col in df_stress.columns and overtime_col in df_stress.columns and unfilled_shifts_col in df_stress.columns:
            
            stress_trend_df = df_stress.copy()
            if not pd.api.types.is_datetime64_any_dtype(stress_trend_df[date_col]):
                stress_trend_df[date_col] = pd.to_datetime(stress_trend_df[date_col], errors='coerce')
            stress_trend_df = stress_trend_df.dropna(subset=[date_col])
            
            if not stress_trend_df.empty:
                # Group by month for the bar chart
                stress_summary_monthly = stress_trend_df.groupby(pd.Grouper(key=date_col, freq='M')).agg(
                    Overtime=(overtime_col, 'sum'), # Use alias names for series
                    Unfilled_Shifts=(unfilled_shifts_col, 'sum')
                ).reset_index()
                
                st.plotly_chart(viz.create_comparison_bar_chart(
                    stress_summary_monthly, date_col, ['Overtime', 'Unfilled_Shifts'],
                    "monthly_shift_load_chart_title", lang=st.session_state.selected_lang_code,
                    x_axis_title_key="month_axis_label", y_axis_title_key="hours_or_shifts_label",
                    barmode='group', # Grouped for direct comparison per month
                    text_auto_format='.0f' # Show as whole numbers
                ), use_container_width=True)
            else:
                st.warning(current_lang_texts.get("no_data_shift_load"))
        else:
            st.warning(current_lang_texts.get("no_data_shift_load"))

    st.markdown("---") # Separator below the two columns before the trend chart

    # Workload trends vs. psychological signals (Full width below)
    if date_col in df_stress.columns and workload_col in df_stress.columns and psych_col in df_stress.columns:

        workload_psych_df = df_stress.copy()
        if not pd.api.types.is_datetime64_any_dtype(workload_psych_df[date_col]):
            workload_psych_df[date_col] = pd.to_datetime(workload_psych_df[date_col], errors='coerce')
        workload_psych_df = workload_psych_df.dropna(subset=[date_col])

        if not workload_psych_df.empty:
            # Group by month/week for trend
            workload_psych_trend = workload_psych_df.groupby(pd.Grouper(key=date_col, freq='W')).agg( # Group by week for potentially more detail
                Workload_Perception=(workload_col, 'mean'),
                Psychological_Signals=(psych_col, 'mean')
            ).reset_index()

            # Define dummy targets for stress trends if needed
            # e.g. trend_targets = {'Workload_Perception': 6.0, 'Psychological_Signals': 5.5} # assuming scale is 1-10
            trend_targets = {} # Placeholder - No targets in config yet

            st.plotly_chart(viz.create_trend_chart(
                workload_psych_trend, date_col, ['Workload_Perception', 'Psychological_Signals'],
                "workload_vs_psych_chart_title", lang=st.session_state.selected_lang_code,
                x_axis_title_key="date_time_axis_label", y_axis_title_key="average_score_label", # Adjust y-axis label
                show_average_line=True,
                target_value_map=trend_targets,
                rolling_avg_window=4 # Example: 4-week rolling average
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
st.warning("This module is a placeholder for future development.") # Simpler warning for UI consistency
st.markdown("---")


# 6. Predictive AI Insights (Placeholder)
st.header(current_lang_texts.get("ai_insights_title"))
st.markdown(config.PLACEHOLDER_TEXT_AI_INSIGHTS, unsafe_allow_html=True)
st.warning("This module is a placeholder for future development.") # Simpler warning
st.markdown("---")


# --- Optional & Strategic Modules ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"## {current_lang_texts.get('optional_modules_header')}")
# Using a check box + expander to keep the sidebar clean by default
show_optional = st.sidebar.checkbox(current_lang_texts.get('show_optional_modules', "Show Planned Modules"))
if show_optional:
    st.header(current_lang_texts.get('optional_modules_title', "Planned Modules"))
    with st.expander(current_lang_texts.get('optional_modules_title'), expanded=show_optional): # Expanded based on checkbox state
        st.markdown(current_lang_texts.get('optional_modules_list', config.TEXT_STRINGS["EN"]["optional_modules_list"])) # Fallback to EN

st.sidebar.markdown("---")
# Version and build info
st.sidebar.caption(f"{current_lang_texts.get('dashboard_title')} v0.4.0 (MVP - Viz Finalized)")
st.sidebar.caption("Built with Streamlit, Plotly, and Pandas.")
st.sidebar.caption("Data Last Updated: (N/A for sample data)") # Add real timestamp later
