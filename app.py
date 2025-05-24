import streamlit as st
import pandas as pd
import numpy as np
import visualizations as viz
import config

# --- Page Configuration (Applied once at the top) ---
# Determine initial language for page config before full session state might be active
initial_lang_code_for_config = config.LANG # Default
if 'selected_lang_code' in st.session_state:
    initial_lang_code_for_config = st.session_state.selected_lang_code

st.set_page_config(
    page_title=config.TEXT_STRINGS[initial_lang_code_for_config].get("dashboard_title", config.APP_TITLE),
    page_icon=config.APP_ICON,
    layout="wide"
)

# --- Language Selection ---
st.sidebar.markdown("---") # Separator
available_langs = list(config.TEXT_STRINGS.keys())
if 'selected_lang_code' not in st.session_state: # Initialize session state for language if not present
    st.session_state.selected_lang_code = config.LANG

def update_lang_session_state(): # Callback function for language selector
    st.session_state.selected_lang_code = st.session_state._app_lang_selector_key_

# Use a unique key for the selectbox to avoid conflicts if key is reused elsewhere
lang_selector_key_unique = "_app_lang_selector_key_"

selected_lang_code = st.sidebar.selectbox(
    label=f"{config.TEXT_STRINGS['EN'].get('language_selector', 'Language')} / {config.TEXT_STRINGS['ES'].get('language_selector', 'Idioma')}", # Bilingual label for the selector itself
    options=available_langs,
    index=available_langs.index(st.session_state.selected_lang_code), # Get current index from session state
    format_func=lambda x: "English" if x == "EN" else "Español" if x == "ES" else x, # Nicer display names
    key=lang_selector_key_unique, # Assign unique key
    on_change=update_lang_session_state # Use callback to update session state
)
# Fetch the current language dictionary based on session state
current_lang_texts = config.TEXT_STRINGS[st.session_state.selected_lang_code]

# --- Helper to get localized text using the current session language ---
def _(text_key, default_text_override=None):
    """Shortcut for getting localized text. Falls back to the key or override."""
    return current_lang_texts.get(text_key, default_text_override if default_text_override is not None else text_key)

# --- Data Loading Functions with Caching ---
@st.cache_data # Using the modern decorator
def load_data_core(file_path: str, date_col_names_for_parse: list[str] | None = None):
    """
    Loads data from a CSV file.
    file_path: Path to the CSV file.
    date_col_names_for_parse: A list of actual column names (from CSV) to parse as dates.
    """
    try:
        df = pd.read_csv(file_path, parse_dates=date_col_names_for_parse if date_col_names_for_parse else False)
        
        # Clean string columns
        for col in df.columns: # Iterate over actual columns in the loaded DataFrame
            if df[col].dtype == 'object': # Check if column is of object type (often strings)
                # Ensure there are non-NA values to operate on
                if df[col].notna().any():
                    # Attempt to strip whitespace, but handle potential mixed types gracefully
                    try:
                        df[col] = df[col].str.strip()
                    except AttributeError:
                        # This column might contain mixed types or non-string objects that .str accessor doesn't work on.
                        # For MVP, we'll let it pass. In production, this might need logging or specific handling.
                        pass 
        return df
    except FileNotFoundError:
        # Use the _() helper for localized error messages
        st.error(_("error_loading_data", "Error loading data from file: {}").format(file_path) + f". Please ensure '{file_path}' is in the correct location.")
        return pd.DataFrame() # Return empty DataFrame to prevent downstream errors
    except Exception as e:
        st.error(_("error_loading_data", "Error loading data from file: {}").format(file_path) + f" - Exception: {e}")
        return pd.DataFrame()

# Prepare the list of actual date column names *before* calling load_data
# This ensures the argument to the cached function is simple and hashable
stability_date_cols_to_parse = [config.COLUMN_MAP["date"]] if config.COLUMN_MAP.get("date") else None
df_stability_raw = load_data_core(config.STABILITY_DATA_FILE, date_col_names_for_parse=stability_date_cols_to_parse)

# safety_data.csv 'month' column is likely text like 'Jan', 'Feb'. If it's a date, adjust here.
df_safety_raw = load_data_core(config.SAFETY_DATA_FILE, date_col_names_for_parse=None) 

df_engagement_raw = load_data_core(config.ENGAGEMENT_DATA_FILE, date_col_names_for_parse=None)

stress_date_cols_to_parse = [config.COLUMN_MAP["date"]] if config.COLUMN_MAP.get("date") else None
df_stress_raw = load_data_core(config.STRESS_DATA_FILE, date_col_names_for_parse=stress_date_cols_to_parse)

# --- Sidebar Filters ---
st.sidebar.header(_("filters_header"))

def get_unique_options_from_dfs(dfs_list, column_key_in_map):
    """Gets unique sorted string options from a list of DataFrames for a given column map key."""
    actual_column_name = config.COLUMN_MAP.get(column_key_in_map)
    if not actual_column_name: return [] # Should not happen if COLUMN_MAP is correct
    
    all_options = set()
    for df in dfs_list:
        if not df.empty and actual_column_name in df.columns:
            # Ensure options are strings and handle NaNs
            all_options.update(df[actual_column_name].dropna().astype(str).tolist())
    return sorted(list(all_options))

# Collect all valid (non-empty) raw dataframes for comprehensive filter options
all_valid_raw_dfs = [df for df in [df_stability_raw, df_safety_raw, df_engagement_raw, df_stress_raw] if not df.empty]

# Get unique options using the conceptual keys from config.COLUMN_MAP
sites = get_unique_options_from_dfs(all_valid_raw_dfs, "site")
regions = get_unique_options_from_dfs(all_valid_raw_dfs, "region")
departments = get_unique_options_from_dfs(all_valid_raw_dfs, "department")
fcs = get_unique_options_from_dfs(all_valid_raw_dfs, "fc")
shifts = get_unique_options_from_dfs(all_valid_raw_dfs, "shift")

selected_sites = st.sidebar.multiselect(_("select_site"), options=sites, default=config.DEFAULT_SITES)
selected_regions = st.sidebar.multiselect(_("select_region"), options=regions, default=config.DEFAULT_REGIONS)
selected_departments = st.sidebar.multiselect(_("select_department"), options=departments, default=config.DEFAULT_DEPARTMENTS)
selected_fcs = st.sidebar.multiselect(_("select_fc"), options=fcs, default=config.DEFAULT_FUNCTIONAL_CATEGORIES)
selected_shifts = st.sidebar.multiselect(_("select_shift"), options=shifts, default=config.DEFAULT_SHIFTS)

# --- Filter DataFrames Utility ---
def apply_filters_to_df(df_input, col_map_config, selections_by_concept_key):
    """Applies filters to a DataFrame based on selections.
    df_input: The DataFrame to filter.
    col_map_config: The config.COLUMN_MAP dictionary.
    selections_by_concept_key: Dict where keys are conceptual (e.g., 'site') and values are selected filter options.
    """
    if df_input.empty:
        return df_input.copy()
    
    df = df_input.copy() # Work on a copy

    for concept_key, selected_filter_values in selections_by_concept_key.items():
        actual_col_name = col_map_config.get(concept_key)
        if actual_col_name and selected_filter_values and actual_col_name in df.columns:
            # Filters are based on string selections from multiselect, so compare as string
            df = df[df[actual_col_name].astype(str).isin([str(v) for v in selected_filter_values])]
    return df

# Map conceptual filter keys to their selected values
active_filter_selections_conceptual = {
    'site': selected_sites, 'region': selected_regions, 'department': selected_departments,
    'fc': selected_fcs, 'shift': selected_shifts
}

# Apply filters to each raw DataFrame
df_stability = apply_filters_to_df(df_stability_raw, config.COLUMN_MAP, active_filter_selections_conceptual)
df_safety = apply_filters_to_df(df_safety_raw, config.COLUMN_MAP, active_filter_selections_conceptual)
df_engagement = apply_filters_to_df(df_engagement_raw, config.COLUMN_MAP, active_filter_selections_conceptual)
df_stress = apply_filters_to_df(df_stress_raw, config.COLUMN_MAP, active_filter_selections_conceptual)


# --- Main Dashboard Title & Introduction ---
st.title(_("dashboard_title"))
st.markdown(_("dashboard_subtitle"))
st.caption(_("alignment_note"))
st.markdown("---")
st.info(_("psych_safety_note")) # This provides important context for the whole dashboard
st.markdown("---")

# --- Helper for dummy previous values for metric cards (MVP DEMO ONLY) ---
def get_dummy_previous(current_val, variation_factor=0.1, is_percentage=False):
    if pd.isna(current_val) or not isinstance(current_val, (int,float, np.number)): return None
    change = float(current_val) * variation_factor * np.random.uniform(-1, 1)
    prev = float(current_val) - change
    if is_percentage:
        return max(0.0, min(100.0, prev)) # Keep percentages within 0-100
    return prev if not pd.isna(prev) else None

# --- 1. Laboral Stability Panel ---
st.header(_("stability_panel_title"))
if not df_stability.empty:
    metric_cols = st.columns(4) # For 1 Gauge + 3 Retention Metrics

    # Employee Rotation Rate
    rotation_col_name = config.COLUMN_MAP["rotation_rate"]
    avg_rotation = df_stability[rotation_col_name].mean() if rotation_col_name in df_stability.columns else float('nan')
    prev_avg_rotation = get_dummy_previous(avg_rotation, 0.05, is_percentage=True) # Smaller variation for rotation

    with metric_cols[0]: # Gauge takes one column
        st.plotly_chart(viz.create_kpi_gauge(
            value=avg_rotation, title_key="rotation_rate_gauge", lang=st.session_state.selected_lang_code,
            unit="%", higher_is_worse=True,
            threshold_good=config.ROTATION_RATE_THRESHOLD_GOOD,
            threshold_warning=config.ROTATION_RATE_THRESHOLD_WARNING,
            target_line_value=config.ROTATION_RATE_TARGET,
            previous_value=prev_avg_rotation
        ), use_container_width=True)

    # Retention Metrics
    retention_defs = [
        ("retention_6m", "retention_6m_metric"),
        ("retention_12m", "retention_12m_metric"),
        ("retention_18m", "retention_18m_metric")
    ]
    for i, (col_map_key, label_key) in enumerate(retention_defs):
        actual_col = config.COLUMN_MAP[col_map_key]
        current_value = df_stability[actual_col].mean() if actual_col in df_stability.columns else float('nan')
        previous_period_value = get_dummy_previous(current_value, 0.03, is_percentage=True) # Small variation
        with metric_cols[i+1]: # Use remaining columns for metrics
            viz.display_metric_card(st, label_key, current_value, st.session_state.selected_lang_code, unit="%", higher_is_better=True,
                                    target_value=config.RETENTION_THRESHOLD_GOOD,
                                    threshold_good=config.RETENTION_THRESHOLD_GOOD,
                                    threshold_warning=config.RETENTION_THRESHOLD_WARNING, # Note: lower than good is warning
                                    previous_value=previous_period_value)
    st.markdown("---")

    # Historical trend of hires vs. exits
    date_col_actual_stab = config.COLUMN_MAP["date"]
    hires_col_actual_stab = config.COLUMN_MAP["hires"]
    exits_col_actual_stab = config.COLUMN_MAP["exits"]

    if all(col in df_stability.columns for col in [date_col_actual_stab, hires_col_actual_stab, exits_col_actual_stab]):
        stability_trend_df = df_stability[[date_col_actual_stab, hires_col_actual_stab, exits_col_actual_stab]].copy()
        if not pd.api.types.is_datetime64_any_dtype(stability_trend_df[date_col_actual_stab]):
            stability_trend_df[date_col_actual_stab] = pd.to_datetime(stability_trend_df[date_col_actual_stab], errors='coerce')
        stability_trend_df = stability_trend_df.dropna(subset=[date_col_actual_stab]).sort_values(by=date_col_actual_stab)


        if not stability_trend_df.empty:
            hires_exits_trend_agg = stability_trend_df.groupby(pd.Grouper(key=date_col_actual_stab, freq='M')).agg(
                Hires_Total=(hires_col_actual_stab, 'sum'), # Use temporary aggregate names
                Exits_Total=(exits_col_actual_stab, 'sum')
            ).reset_index()
            
            # Map conceptual keys to the new aggregated column names for the chart function
            hires_exits_value_map = {"hires_label": "Hires_Total", "exits_label": "Exits_Total"}
            units_map_hires_exits = {"Hires_Total": "", "Exits_Total": ""} # No units for these counts, axis title covers it
            
            st.plotly_chart(viz.create_trend_chart(
                hires_exits_trend_agg, date_col_actual_stab, hires_exits_value_map,
                "hires_vs_exits_chart_title", st.session_state.selected_lang_code,
                y_axis_title_key="people_count_label", x_axis_title_key="month_axis_label",
                show_average_line=True, rolling_avg_window=3, value_col_units_map=units_map_hires_exits
            ), use_container_width=True)
        else: st.warning(_("no_data_hires_exits"))
    else: st.warning(_("no_data_hires_exits"))
else: st.info(_("no_data_available"))
st.markdown("---")


# --- 2. Safety Pulse Module ---
st.header(_("safety_pulse_title"))
if not df_safety.empty:
    chart_col, metrics_col1, metrics_col2 = st.columns([2, 1, 1])

    month_col_safety_actual = config.COLUMN_MAP["month"]
    incidents_col_safety_actual = config.COLUMN_MAP["incidents"]
    near_misses_col_safety_actual = config.COLUMN_MAP["near_misses"]
    days_no_acc_actual = config.COLUMN_MAP["days_without_accidents"]
    active_alerts_actual = config.COLUMN_MAP["active_alerts"]

    with chart_col:
        if all(col in df_safety.columns for col in [month_col_safety_actual, incidents_col_safety_actual, near_misses_col_safety_actual]):
            safety_summary = df_safety.groupby(month_col_safety_actual, as_index=False).agg(
                Incidents_Sum=(incidents_col_safety_actual, 'sum'), # Temp aggregate names
                Near_Misses_Sum=(near_misses_col_safety_actual, 'sum')
            )
            # Month sorting
            try:
                month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                safety_summary[month_col_safety_actual] = pd.Categorical(safety_summary[month_col_safety_actual].astype(str), categories=month_order, ordered=True)
                safety_summary = safety_summary.sort_values(month_col_safety_actual).dropna(subset=[month_col_safety_actual]) # Drop if month became NaN
            except Exception: safety_summary = safety_summary.sort_values(by=month_col_safety_actual, errors='ignore')
            
            if not safety_summary.empty:
                safety_bar_value_map = {"incidents_label": "Incidents_Sum", "near_misses_label": "Near_Misses_Sum"}
                st.plotly_chart(viz.create_comparison_bar_chart(
                    safety_summary, month_col_safety_actual, safety_bar_value_map,
                    "monthly_incidents_chart_title", st.session_state.selected_lang_code,
                    x_axis_title_key="month_axis_label", y_axis_title_key="count_label",
                    barmode='stack', show_total_for_stacked=True, data_label_format_str=".0f"
                ), use_container_width=True)
            else: st.warning(_("no_data_incidents_near_misses"))
        else: st.warning(_("no_data_incidents_near_misses"))

    with metrics_col1:
        days_val = df_safety[days_no_acc_actual].max() if days_no_acc_actual in df_safety.columns else float('nan')
        prev_days = get_dummy_previous(days_val, 0.05)
        viz.display_metric_card(st, "days_without_accidents_metric", days_val, st.session_state.selected_lang_code,
                               unit=" "+_("days_label"), higher_is_better=True, threshold_good=180, threshold_warning=90, previous_value=prev_days)
    with metrics_col2:
        alerts_val = df_safety[active_alerts_actual].sum() if active_alerts_actual in df_safety.columns else float('nan')
        prev_alerts = get_dummy_previous(alerts_val, variation_factor=0.3, is_percentage=False) # higher variation for small numbers
        prev_alerts = int(prev_alerts) if pd.notna(prev_alerts) else None
        viz.display_metric_card(st, "active_safety_alerts_metric", alerts_val, st.session_state.selected_lang_code,
                                unit="", higher_is_better=False, target_value=0, threshold_good=0, threshold_warning=1, previous_value=prev_alerts)
else: st.info(_("no_data_available"))
st.markdown("---")

# --- 3. Employee Engagement & Commitment ---
st.header(_("engagement_title"))
if not df_engagement.empty:
    col_radar, col_metrics = st.columns([2,1])

    with col_radar:
        radar_data_points = []
        radar_target_map_localized = {} # For target line, keys are localized dimension names
        radar_scale_max = 5 # Common scale for such dimensions

        # Iterate through the conceptual keys and map to actual CSV column names
        for conceptual_key, actual_col_name in config.COLUMN_MAP["engagement_radar_dims_cols"].items():
            if actual_col_name in df_engagement.columns:
                avg_score = df_engagement[actual_col_name].mean()
                # Get the label key (e.g., 'initiative_label') from the conceptual key
                label_key_for_display = config.COLUMN_MAP["engagement_radar_dims_labels"].get(conceptual_key, actual_col_name)
                display_name = _(label_key_for_display, actual_col_name.replace('_', ' ').title())

                if pd.notna(avg_score):
                    radar_data_points.append({"Dimension": display_name, "Score": avg_score})
                    # Set some example targets, this should come from config or a defined strategy
                    if "Initiative" in display_name or "Iniciativa" in display_name: radar_target_map_localized[display_name] = 4.0
                    elif "Punctuality" in display_name or "Puntualidad" in display_name: radar_target_map_localized[display_name] = 4.5
                    elif "Recognition" in display_name or "Reconocimiento" in display_name: radar_target_map_localized[display_name] = 3.8
                    elif "Feedback" in display_name or "Retroalimentación" in display_name: radar_target_map_localized[display_name] = 4.2
                    else: radar_target_map_localized[display_name] = radar_scale_max * 0.8 # Generic 80% target

        if radar_data_points:
            df_radar_viz = pd.DataFrame(radar_data_points)
            st.plotly_chart(viz.create_enhanced_radar_chart(
                df_radar_viz, "Dimension", "Score", "engagement_dimensions_radar_title", st.session_state.selected_lang_code,
                range_max_override=radar_scale_max, target_values_map=radar_target_map_localized, fill_opacity=0.45
            ), use_container_width=True)
        elif any(config.COLUMN_MAP["engagement_radar_dims_cols"].get(k) in df_engagement.columns for k in config.COLUMN_MAP["engagement_radar_dims_cols"]):
             st.warning(_("no_data_radar"))
        else:
            st.warning(_("no_data_radar_columns"))

    with col_metrics:
        engagement_kpis = [
            ("labor_climate_score", "labor_climate_score_metric", "", True, config.CLIMATE_SCORE_THRESHOLD_GOOD, config.CLIMATE_SCORE_THRESHOLD_WARNING),
            ("enps_score", "enps_metric", "", True, config.ENPS_THRESHOLD_GOOD, config.ENPS_THRESHOLD_WARNING),
            ("participation_rate", "survey_participation_metric", "%", True, config.PARTICIPATION_THRESHOLD_GOOD, None),
            ("recognitions_count", "recognitions_count_metric", "", True, None, None) # higher is better, no specific target shown here
        ]
        for col_key, label_key, unit, higher_better, th_good, th_warn in engagement_kpis:
            actual_col_name = config.COLUMN_MAP[col_key]
            # Aggregate appropriately: sum for counts, mean for scores/rates
            if "count" in col_key: # Simple heuristic
                val = df_engagement[actual_col_name].sum() if actual_col_name in df_engagement.columns else float('nan')
            else:
                val = df_engagement[actual_col_name].mean() if actual_col_name in df_engagement.columns else float('nan')
            
            prev_val = get_dummy_previous(val, 0.08, is_percentage=(unit=="%"))
            viz.display_metric_card(st, label_key, val, st.session_state.selected_lang_code, unit=unit, higher_is_better=higher_better,
                                    target_value=th_good, threshold_good=th_good, threshold_warning=th_warn, previous_value=prev_val)
else: st.info(_("no_data_available"))
st.markdown("---")

# --- 4. Operational Stress Dashboard ---
st.header(_("stress_title"))
if not df_stress.empty:
    stress_cols_layout = st.columns([1, 2]) # Stress Semafaro, then Shift Load Chart

    stress_level_actual_col = config.COLUMN_MAP["stress_level_survey"]
    overtime_actual_col = config.COLUMN_MAP["overtime_hours"]
    unfilled_shifts_actual_col = config.COLUMN_MAP["unfilled_shifts"]
    date_stress_actual_col = config.COLUMN_MAP["date"]
    workload_actual_col = config.COLUMN_MAP["workload_perception"]
    psych_signals_actual_col = config.COLUMN_MAP["psychological_signals"]

    with stress_cols_layout[0]: # Stress Semaforo
        st.subheader(_("overall_stress_indicator_title")) # More context for the single KPI
        avg_stress_level = df_stress[stress_level_actual_col].mean() if stress_level_actual_col in df_stress.columns else float('nan')
        st.plotly_chart(viz.create_stress_semaforo_visual(
            avg_stress_level, lang=st.session_state.selected_lang_code, scale_max=config.STRESS_LEVEL_MAX_SCALE
        ), use_container_width=True)

    with stress_cols_layout[1]: # Shift Load Chart
        if all(c in df_stress.columns for c in [date_stress_actual_col, overtime_actual_col, unfilled_shifts_actual_col]):
            df_sl_trend = df_stress[[date_stress_actual_col, overtime_actual_col, unfilled_shifts_actual_col]].copy()
            if not pd.api.types.is_datetime64_any_dtype(df_sl_trend[date_stress_actual_col]):
                df_sl_trend[date_stress_actual_col] = pd.to_datetime(df_sl_trend[date_stress_actual_col], errors='coerce')
            df_sl_trend = df_sl_trend.dropna(subset=[date_stress_actual_col]).sort_values(by=date_stress_actual_col)
            
            if not df_sl_trend.empty:
                sl_summary = df_sl_trend.groupby(pd.Grouper(key=date_stress_actual_col, freq='M')).agg(
                   Overtime=(overtime_actual_col, 'sum'), # Direct naming for Plotly Express if labels provided
                   Unfilled_Shifts=(unfilled_shifts_actual_col, 'sum')
                ).reset_index()
                
                shift_load_value_map = { # Conceptual key -> Aggregated column name for viz function
                    "overtime_label": "Overtime", # This refers to the "Overtime" in sl_summary
                    "unfilled_shifts_label": "Unfilled_Shifts"
                }
                st.plotly_chart(viz.create_comparison_bar_chart(
                    sl_summary, date_stress_actual_col, shift_load_value_map,
                    "monthly_shift_load_chart_title", st.session_state.selected_lang_code,
                    x_axis_title_key="month_axis_label", y_axis_title_key="hours_or_shifts_label",
                    barmode='group', data_label_format_str=".0f"
                ), use_container_width=True)
            else: st.warning(_("no_data_shift_load"))
        else: st.warning(_("no_data_shift_load"))
    st.markdown("---") # Separator before full-width chart

    # Workload trends vs. psychological signals (Full width)
    if all(c in df_stress.columns for c in [date_stress_actual_col, workload_actual_col, psych_signals_actual_col]):
        df_wp_trend = df_stress[[date_stress_actual_col, workload_actual_col, psych_signals_actual_col]].copy()
        if not pd.api.types.is_datetime64_any_dtype(df_wp_trend[date_stress_actual_col]):
            df_wp_trend[date_stress_actual_col] = pd.to_datetime(df_wp_trend[date_stress_actual_col], errors='coerce')
        df_wp_trend = df_wp_trend.dropna(subset=[date_stress_actual_col]).sort_values(by=date_stress_actual_col)

        if not df_wp_trend.empty:
            wp_summary = df_wp_trend.groupby(pd.Grouper(key=date_stress_actual_col, freq='M')).agg( # Changed to Monthly
                Workload_Agg=(workload_actual_col, 'mean'),
                Psych_Signals_Agg=(psych_signals_actual_col, 'mean')
            ).reset_index()

            workload_psych_value_map = { # Map label key to aggregated column name
                "workload_perception_label": "Workload_Agg",
                "psychological_signals_label": "Psych_Signals_Agg"
            }
            workload_psych_units = { "Workload_Agg": "", "Psych_Signals_Agg": ""} # No units, scores

            st.plotly_chart(viz.create_trend_chart(
                wp_summary, date_stress_actual_col, workload_psych_value_map,
                "workload_vs_psych_chart_title", st.session_state.selected_lang_code,
                y_axis_title_key="average_score_label", x_axis_title_key="month_axis_label", # X axis is monthly
                show_average_line=True, rolling_avg_window=3, value_col_units_map=workload_psych_units
            ), use_container_width=True)
        else: st.warning(_("no_data_workload_psych"))
    else: st.warning(_("no_data_workload_psych"))
else: st.info(_("no_data_available"))
st.markdown("---")

# --- 5. Interactive Plant Map (Placeholder) ---
st.header(_("plant_map_title"))
st.markdown(config.PLACEHOLDER_TEXT_PLANT_MAP, unsafe_allow_html=True)
st.warning(_("This module is a placeholder for future development.", "This module requires advanced setup.")) # More generic
st.markdown("---")

# --- 6. Predictive AI Insights (Placeholder) ---
st.header(_("ai_insights_title"))
st.markdown(config.PLACEHOLDER_TEXT_AI_INSIGHTS, unsafe_allow_html=True)
st.warning(_("This module is a placeholder for future development.", "This module requires advanced setup."))
st.markdown("---")

# --- Optional & Strategic Modules ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"## {_('optional_modules_header')}")
show_optional = st.sidebar.checkbox(_('show_optional_modules'), key="show_optional_modules_checkbox", value=False) # Give it a specific key
if show_optional:
    # Use st.expander in the main area if toggled, for better readability of longer text.
    with st.expander(_('optional_modules_title'), expanded=True):
        st.markdown(_('optional_modules_list', config.TEXT_STRINGS["EN"]["optional_modules_list"]), unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption(f"{_('dashboard_title')} v0.7.0 (SME Final)") # Increment version
st.sidebar.caption(_("Built with Streamlit, Plotly, and Pandas.", "Construido con Streamlit, Plotly y Pandas."))
st.sidebar.caption(_("Data Last Updated: (N/A for sample data)", "Última Actualización de Datos: (N/A para datos de muestra)"))
