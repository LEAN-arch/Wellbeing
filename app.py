import streamlit as st
import pandas as pd
import numpy as np # For dummy previous values and isnan
import visualizations as viz
import config

# --- Page Configuration (Applied once at the top) ---
initial_lang_code = st.session_state.get('selected_lang_code', config.LANG)
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
    st.session_state.selected_lang_code = st.session_state._lang_selector_key # Use the actual key

lang_selector_key = "_lang_selector_key"
selected_lang_code = st.sidebar.selectbox(
    label=f"{config.TEXT_STRINGS['EN'].get('language_selector')} / {config.TEXT_STRINGS['ES'].get('language_selector')}",
    options=available_langs,
    index=available_langs.index(st.session_state.selected_lang_code),
    format_func=lambda x: "English" if x == "EN" else "Español" if x == "ES" else x,
    key=lang_selector_key,
    on_change=update_lang
)
current_lang_texts = config.TEXT_STRINGS[st.session_state.selected_lang_code]

# --- Helper to get localized text using the current session language ---
def _(text_key, default_text_override=None):
    return current_lang_texts.get(text_key, default_text_override if default_text_override is not None else text_key)


# --- Data Loading Functions with Caching ---
@st.cache_data
def load_data(file_path: str, date_col_names_for_parsing: list[str] | None = None):
    """Loads data from a CSV file.
    file_path: Path to the CSV file.
    date_col_names_for_parsing: A list of actual column names to parse as dates.
    """
    try:
        df = pd.read_csv(file_path, parse_dates=date_col_names_for_parsing if date_col_names_for_parsing else False)
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if any non-NA values exist before trying to strip
                if df[col].notna().any():
                    # Attempt to strip, but handle cases where items might not be strings (though less likely after object check)
                    try:
                        df[col] = df[col].str.strip()
                    except AttributeError:
                        # This column might contain mixed types or non-string objects. Log or handle as needed.
                        # For MVP, we'll let it pass if not all are strippable strings.
                        pass 
        return df
    except FileNotFoundError:
        lang_code_for_error = st.session_state.get('selected_lang_code', config.LANG)
        error_msg_template = config.TEXT_STRINGS.get(lang_code_for_error, config.TEXT_STRINGS["EN"]).get("error_loading_data", "Error loading data from file: {}")
        st.error(error_msg_template.format(file_path) + f". Check path relative to app.py.")
        return pd.DataFrame()
    except Exception as e:
        lang_code_for_error = st.session_state.get('selected_lang_code', config.LANG)
        error_msg_template = config.TEXT_STRINGS.get(lang_code_for_error, config.TEXT_STRINGS["EN"]).get("error_loading_data", "Error loading data from file: {}")
        st.error(error_msg_template.format(file_path) + f" - Exception: {e}")
        return pd.DataFrame()

# --- Prepare date column names for parsing *before* calling load_data ---
# Ensure the keys exist in COLUMN_MAP
stability_date_cols_actual = [config.COLUMN_MAP["date"]] if "date" in config.COLUMN_MAP and config.COLUMN_MAP["date"] else None
df_stability_raw = load_data(config.STABILITY_DATA_FILE, date_col_names_for_parsing=stability_date_cols_actual)

# Assuming 'month' in safety_data.csv is 'Jan', 'Feb', etc. and not 'YYYY-MM-DD'
# If it were date-like, you'd pass [config.COLUMN_MAP["month"]]
df_safety_raw = load_data(config.SAFETY_DATA_FILE, date_col_names_for_parsing=None) 

df_engagement_raw = load_data(config.ENGAGEMENT_DATA_FILE, date_col_names_for_parsing=None)

stress_date_cols_actual = [config.COLUMN_MAP["date"]] if "date" in config.COLUMN_MAP and config.COLUMN_MAP["date"] else None
df_stress_raw = load_data(config.STRESS_DATA_FILE, date_col_names_for_parsing=stress_date_cols_actual)


# --- Sidebar Filters ---
st.sidebar.header(_("filters_header"))

def get_unique_options(df_list, column_key_in_map):
    column_name = config.COLUMN_MAP.get(column_key_in_map)
    if not column_name: return []
    all_options = set()
    for df in df_list:
        if not df.empty and column_name in df.columns:
            all_options.update(df[column_name].dropna().astype(str).tolist())
    return sorted(list(all_options))

all_raw_dfs_for_filters = [df for df in [df_stability_raw, df_safety_raw, df_engagement_raw, df_stress_raw] if not df.empty]

sites = get_unique_options(all_raw_dfs_for_filters, "site")
regions = get_unique_options(all_raw_dfs_for_filters, "region")
departments = get_unique_options(all_raw_dfs_for_filters, "department")
fcs = get_unique_options(all_raw_dfs_for_filters, "fc")
shifts = get_unique_options(all_raw_dfs_for_filters, "shift")

selected_sites = st.sidebar.multiselect(_("select_site"), options=sites, default=config.DEFAULT_SITES)
selected_regions = st.sidebar.multiselect(_("select_region"), options=regions, default=config.DEFAULT_REGIONS)
selected_departments = st.sidebar.multiselect(_("select_department"), options=departments, default=config.DEFAULT_DEPARTMENTS)
selected_fcs = st.sidebar.multiselect(_("select_fc"), options=fcs, default=config.DEFAULT_FUNCTIONAL_CATEGORIES)
selected_shifts = st.sidebar.multiselect(_("select_shift"), options=shifts, default=config.DEFAULT_SHIFTS)

# --- Filter DataFrames Utility ---
def filter_dataframe_by_selections(df, column_map_dict, current_selections_map):
    if df.empty: return df.copy()
    filtered_df = df.copy()
    for filter_col_key, selected_values in current_selections_map.items():
        actual_col_name = column_map_dict.get(filter_col_key)
        if actual_col_name and selected_values and actual_col_name in filtered_df.columns:
            # Ensure selected_values are strings for string columns (robustly handle numeric-like strings in options)
            if filtered_df[actual_col_name].dtype == 'object' or pd.api.types.is_string_dtype(filtered_df[actual_col_name]):
                 selected_values_str = [str(v) for v in selected_values]
                 filtered_df = filtered_df[filtered_df[actual_col_name].astype(str).isin(selected_values_str)]
            else: # For numerical columns, selected_values might need casting if they came from string multiselect
                try:
                    # Attempt to cast selected_values to the df column's dtype if numeric, bool etc.
                    # This is complex, safer to ensure filter options match data type or handle conversions upstream
                    # For simplicity here, assuming direct isin works if not string.
                    if pd.api.types.is_numeric_dtype(filtered_df[actual_col_name]):
                        # Try converting selected values to the column's numeric type, skip if error
                        try:
                            selected_values_casted = [pd.to_numeric(v) for v in selected_values]
                            filtered_df = filtered_df[filtered_df[actual_col_name].isin(selected_values_casted)]
                        except ValueError: # If selected_values cannot be cast to numeric, skip this filter for safety
                            pass 
                    else: # For boolean or other types, direct isin
                         filtered_df = filtered_df[filtered_df[actual_col_name].isin(selected_values)]

                except Exception: # Fallback if type casting / comparison issue
                    pass # Could log this issue
    return filtered_df


filter_selections_dict = {
    'site': selected_sites, 'region': selected_regions, 'department': selected_departments,
    'fc': selected_fcs, 'shift': selected_shifts
}

df_stability = filter_dataframe_by_selections(df_stability_raw.copy(), config.COLUMN_MAP, filter_selections_dict)
df_safety = filter_dataframe_by_selections(df_safety_raw.copy(), config.COLUMN_MAP, filter_selections_dict)
df_engagement = filter_dataframe_by_selections(df_engagement_raw.copy(), config.COLUMN_MAP, filter_selections_dict)
df_stress = filter_dataframe_by_selections(df_stress_raw.copy(), config.COLUMN_MAP, filter_selections_dict)


# --- Main Dashboard Title & Introduction ---
st.title(_("dashboard_title"))
st.markdown(_("dashboard_subtitle"))
st.caption(_("alignment_note"))
st.markdown("---")
st.info(_("psych_safety_note"))
st.markdown("---")

# --- CORE MODULES (Ensure all column access uses config.COLUMN_MAP) ---

# 1. Laboral Stability Panel
st.header(_("stability_panel_title"))
if not df_stability.empty:
    metric_cols = st.columns(4)

    rotation_col_actual = config.COLUMN_MAP["rotation_rate"]
    avg_rotation = df_stability[rotation_col_actual].mean() if rotation_col_actual in df_stability.columns else float('nan')
    prev_avg_rotation = (avg_rotation * (1 + np.random.uniform(-0.1, 0.1))) if pd.notna(avg_rotation) else None

    with metric_cols[0]:
        st.plotly_chart(viz.create_kpi_gauge(
            value=avg_rotation, title_key="rotation_rate_gauge", lang=st.session_state.selected_lang_code,
            unit="%", higher_is_worse=True,
            threshold_good=config.ROTATION_RATE_THRESHOLD_GOOD,
            threshold_warning=config.ROTATION_RATE_THRESHOLD_WARNING,
            target_line_value=config.ROTATION_RATE_TARGET,
            previous_value=prev_avg_rotation
        ), use_container_width=True)

    ret_6m_actual = config.COLUMN_MAP["retention_6m"]
    ret_12m_actual = config.COLUMN_MAP["retention_12m"]
    ret_18m_actual = config.COLUMN_MAP["retention_18m"]

    ret_6m_val = df_stability[ret_6m_actual].mean() if ret_6m_actual in df_stability.columns else float('nan')
    ret_12m_val = df_stability[ret_12m_actual].mean() if ret_12m_actual in df_stability.columns else float('nan')
    ret_18m_val = df_stability[ret_18m_actual].mean() if ret_18m_actual in df_stability.columns else float('nan')
    
    prev_ret_6m = ret_6m_val - np.random.uniform(1, 5) if pd.notna(ret_6m_val) else None
    prev_ret_12m = ret_12m_val - np.random.uniform(1, 5) if pd.notna(ret_12m_val) else None
    prev_ret_18m = ret_18m_val - np.random.uniform(1, 5) if pd.notna(ret_18m_val) else None


    with metric_cols[1]:
        viz.display_metric_card(st, "retention_6m_metric", ret_6m_val, lang=st.session_state.selected_lang_code, unit="%", higher_is_better=True,
                                target_value=config.RETENTION_THRESHOLD_GOOD, threshold_warning=config.RETENTION_THRESHOLD_WARNING, previous_value=prev_ret_6m)
    with metric_cols[2]:
        viz.display_metric_card(st, "retention_12m_metric", ret_12m_val, lang=st.session_state.selected_lang_code, unit="%", higher_is_better=True,
                                target_value=config.RETENTION_THRESHOLD_GOOD, threshold_warning=config.RETENTION_THRESHOLD_WARNING, previous_value=prev_ret_12m)
    with metric_cols[3]:
        viz.display_metric_card(st, "retention_18m_metric", ret_18m_val, lang=st.session_state.selected_lang_code, unit="%", higher_is_better=True,
                                target_value=config.RETENTION_THRESHOLD_GOOD, threshold_warning=config.RETENTION_THRESHOLD_WARNING, previous_value=prev_ret_18m)

    st.markdown("---")

    date_col_stability_actual = config.COLUMN_MAP["date"]
    hires_col_actual = config.COLUMN_MAP["hires"]
    exits_col_actual = config.COLUMN_MAP["exits"]

    if date_col_stability_actual in df_stability.columns and hires_col_actual in df_stability.columns and exits_col_actual in df_stability.columns:
        stability_trend_df = df_stability[[date_col_stability_actual, hires_col_actual, exits_col_actual]].copy()
        if not pd.api.types.is_datetime64_any_dtype(stability_trend_df[date_col_stability_actual]):
            stability_trend_df[date_col_stability_actual] = pd.to_datetime(stability_trend_df[date_col_stability_actual], errors='coerce')
        stability_trend_df = stability_trend_df.dropna(subset=[date_col_stability_actual])

        if not stability_trend_df.empty:
            hires_exits_trend = stability_trend_df.groupby(pd.Grouper(key=date_col_stability_actual, freq='M')).agg(
                Hires=(hires_col_actual, 'sum'),
                Exits=(exits_col_actual, 'sum')
            ).reset_index()
            st.plotly_chart(viz.create_trend_chart(
                hires_exits_trend, date_col_stability_actual, ['Hires', 'Exits'],
                "hires_vs_exits_chart_title", lang=st.session_state.selected_lang_code,
                y_axis_title_key="people_count_label", x_axis_title_key="month_axis_label",
                show_average_line=True, rolling_avg_window=3,
                value_col_units={'Hires': ' ' + _("people_count_label", "people").lower(), 'Exits': ' ' + _("people_count_label", "people").lower()}
            ), use_container_width=True)
        else: st.warning(_("no_data_hires_exits"))
    else: st.warning(_("no_data_hires_exits"))
else: st.info(_("no_data_available"))
st.markdown("---")


# 2. Safety Pulse Module
st.header(_("safety_pulse_title"))
if not df_safety.empty:
    chart_col, metrics_col1, metrics_col2 = st.columns([2, 1, 1])

    month_col_actual = config.COLUMN_MAP["month"]
    incidents_col_actual = config.COLUMN_MAP["incidents"]
    near_misses_col_actual = config.COLUMN_MAP["near_misses"]
    days_no_accidents_actual = config.COLUMN_MAP["days_without_accidents"]
    active_alerts_actual = config.COLUMN_MAP["active_alerts"]

    with chart_col:
        if month_col_actual in df_safety.columns and incidents_col_actual in df_safety.columns and near_misses_col_actual in df_safety.columns:
            safety_summary = df_safety.groupby(month_col_actual, as_index=False).agg(
                Incidents=(incidents_col_actual, 'sum'),
                Near_Misses=(near_misses_col_actual, 'sum')
            )
            try:
                month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                safety_summary[month_col_actual] = pd.Categorical(safety_summary[month_col_actual].astype(str), categories=month_order, ordered=True)
                safety_summary = safety_summary.sort_values(month_col_actual)
            except Exception:
                safety_summary = safety_summary.sort_values(by=month_col_actual, errors='ignore')
            
            st.plotly_chart(viz.create_comparison_bar_chart(
                safety_summary, month_col_actual, ['Incidents', 'Near_Misses'],
                "monthly_incidents_chart_title", lang=st.session_state.selected_lang_code,
                x_axis_title_key="month_axis_label", y_axis_title_key="count_label",
                barmode='stack', show_total_for_stacked=True
            ), use_container_width=True)
        else: st.warning(_("no_data_incidents_near_misses"))

    with metrics_col1:
        days_val = df_safety[days_no_accidents_actual].max() if days_no_accidents_actual in df_safety.columns else float('nan')
        prev_days_val = days_val - np.random.randint(5,15) if pd.notna(days_val) and days_val > 15 else (days_val / 2 if pd.notna(days_val) else None)
        viz.display_metric_card(st, "days_without_accidents_metric", days_val, lang=st.session_state.selected_lang_code,
                               unit=" " + _("days_label"), higher_is_better=True,
                               threshold_good=180, threshold_warning=90, previous_value=prev_days_val)

    with metrics_col2:
        alerts_val = df_safety[active_alerts_actual].sum() if active_alerts_actual in df_safety.columns else float('nan')
        prev_alerts_val = alerts_val + np.random.randint(0,1) if pd.notna(alerts_val) else None
        viz.display_metric_card(st, "active_safety_alerts_metric", alerts_val, lang=st.session_state.selected_lang_code,
                               higher_is_better=False, target_value=0,
                               threshold_good=0, threshold_warning=1, previous_value=prev_alerts_val)
else: st.info(_("no_data_available"))
st.markdown("---")

# 3. Employee Engagement & Commitment
st.header(_("engagement_title"))
if not df_engagement.empty:
    col1, col2 = st.columns([2,1])

    with col1: # Radar chart
        radar_data_list = []
        radar_target_values_map = {}
        radar_score_scale_max = 5

        for internal_key, label_key in config.COLUMN_MAP["engagement_radar_dims"].items():
            # `internal_key` is like "engagement_initiative_raw", `label_key` is "initiative_label"
            data_col_name_in_csv = config.COLUMN_MAP.get(internal_key) # Get actual CSV column name
            if data_col_name_in_csv and data_col_name_in_csv in df_engagement.columns:
                avg_val = df_engagement[data_col_name_in_csv].mean()
                display_label = _(label_key, data_col_name_in_csv.replace('_', ' ').title())
                
                if pd.notna(avg_val):
                    radar_data_list.append({"Dimension": display_label, "Score": avg_val})
                    # Example targets - these should ideally come from config or business logic
                    if "Initiative" in display_label or "Iniciativa" in display_label: radar_target_values_map[display_label] = 4.0
                    elif "Punctuality" in display_label or "Puntualidad" in display_label: radar_target_values_map[display_label] = 4.5
                    elif "Recognition" in display_label or "Reconocimiento" in display_label: radar_target_values_map[display_label] = 3.8
                    elif "Feedback" in display_label or "Retroalimentación" in display_label: radar_target_values_map[display_label] = 4.2
        
        if radar_data_list:
            df_radar = pd.DataFrame(radar_data_list)
            st.plotly_chart(viz.create_enhanced_radar_chart(
                df_radar, "Dimension", "Score",
                "engagement_dimensions_radar_title", lang=st.session_state.selected_lang_code,
                range_max_override=radar_score_scale_max,
                target_values_map=radar_target_values_map
            ), use_container_width=True)
        elif any(config.COLUMN_MAP.get(k) in df_engagement.columns for k in config.COLUMN_MAP["engagement_radar_dims"]):
             st.warning(_("no_data_radar"))
        else:
            st.warning(_("no_data_radar_columns"))

    with col2: # Metric Cards
        lc_actual = config.COLUMN_MAP["labor_climate_score"]
        nps_actual = config.COLUMN_MAP["enps_score"]
        part_actual = config.COLUMN_MAP["participation_rate"]
        rec_actual = config.COLUMN_MAP["recognitions_count"]

        climate_score = df_engagement[lc_actual].mean() if lc_actual in df_engagement.columns else float('nan')
        nps = df_engagement[nps_actual].mean() if nps_actual in df_engagement.columns else float('nan')
        participation = df_engagement[part_actual].mean() if part_actual in df_engagement.columns else float('nan')
        recognitions = df_engagement[rec_actual].sum() if rec_actual in df_engagement.columns else float('nan')
        
        prev_climate = climate_score * 0.92 if pd.notna(climate_score) else None
        prev_nps = nps - 3 if pd.notna(nps) else None
        prev_participation = participation * 0.98 if pd.notna(participation) else None
        prev_recognitions = recognitions * 0.85 if pd.notna(recognitions) else None

        viz.display_metric_card(st, "labor_climate_score_metric", climate_score, lang=st.session_state.selected_lang_code,
                                unit="", higher_is_better=True, target_value=config.CLIMATE_SCORE_THRESHOLD_GOOD,
                                threshold_good=config.CLIMATE_SCORE_THRESHOLD_GOOD, threshold_warning=config.CLIMATE_SCORE_THRESHOLD_WARNING, previous_value=prev_climate)
        viz.display_metric_card(st, "enps_metric", nps, lang=st.session_state.selected_lang_code,
                                unit="", higher_is_better=True, target_value=config.ENPS_THRESHOLD_GOOD,
                                threshold_good=config.ENPS_THRESHOLD_GOOD, threshold_warning=config.ENPS_THRESHOLD_WARNING, previous_value=prev_nps)
        viz.display_metric_card(st, "survey_participation_metric", participation, lang=st.session_state.selected_lang_code,
                                unit="%", higher_is_better=True, target_value=config.PARTICIPATION_THRESHOLD_GOOD, previous_value=prev_participation)
        viz.display_metric_card(st, "recognitions_count_metric", recognitions, lang=st.session_state.selected_lang_code,
                                unit="", higher_is_better=True, previous_value=prev_recognitions)
else: st.info(_("no_data_available"))
st.markdown("---")

# 4. Operational Stress Dashboard
st.header(_("stress_title"))
if not df_stress.empty:
    stress_viz_cols = st.columns([1, 2]) 

    stress_level_actual = config.COLUMN_MAP["stress_level_survey"]
    overtime_actual = config.COLUMN_MAP["overtime_hours"]
    unfilled_shifts_actual = config.COLUMN_MAP["unfilled_shifts"]
    date_stress_actual = config.COLUMN_MAP["date"] # Renamed for clarity
    workload_actual = config.COLUMN_MAP["workload_perception"]
    psych_actual = config.COLUMN_MAP["psychological_signals"]

    with stress_viz_cols[0]:
        st.subheader(_("overall_stress_indicator_title"))
        avg_stress_level = df_stress[stress_level_actual].mean() if stress_level_actual in df_stress.columns else float('nan')
        st.plotly_chart(viz.create_stress_semaforo_visual(
            avg_stress_level, lang=st.session_state.selected_lang_code, scale_max=config.STRESS_LEVEL_MAX_SCALE
        ), use_container_width=True)

    with stress_viz_cols[1]:
        if date_stress_actual in df_stress.columns and overtime_actual in df_stress.columns and unfilled_shifts_actual in df_stress.columns:
            stress_trend_df_sl = df_stress[[date_stress_actual, overtime_actual, unfilled_shifts_actual]].copy()
            if not pd.api.types.is_datetime64_any_dtype(stress_trend_df_sl[date_stress_actual]):
                stress_trend_df_sl[date_stress_actual] = pd.to_datetime(stress_trend_df_sl[date_stress_actual], errors='coerce')
            stress_trend_df_sl = stress_trend_df_sl.dropna(subset=[date_stress_actual])
            
            if not stress_trend_df_sl.empty:
                stress_summary_monthly = stress_trend_df_sl.groupby(pd.Grouper(key=date_stress_actual, freq='M')).agg(
                    Overtime=(overtime_actual, 'sum'),
                    Unfilled_Shifts=(unfilled_shifts_actual, 'sum')
                ).reset_index()
                st.plotly_chart(viz.create_comparison_bar_chart(
                    stress_summary_monthly, date_stress_actual, ['Overtime', 'Unfilled_Shifts'],
                    "monthly_shift_load_chart_title", lang=st.session_state.selected_lang_code,
                    x_axis_title_key="month_axis_label", y_axis_title_key="hours_or_shifts_label",
                    barmode='group', text_auto_format_str=".0f"
                ), use_container_width=True)
            else: st.warning(_("no_data_shift_load"))
        else: st.warning(_("no_data_shift_load"))

    st.markdown("---")

    if date_stress_actual in df_stress.columns and workload_actual in df_stress.columns and psych_actual in df_stress.columns:
        workload_psych_df = df_stress[[date_stress_actual, workload_actual, psych_actual]].copy()
        if not pd.api.types.is_datetime64_any_dtype(workload_psych_df[date_stress_actual]):
            workload_psych_df[date_stress_actual] = pd.to_datetime(workload_psych_df[date_stress_actual], errors='coerce')
        workload_psych_df = workload_psych_df.dropna(subset=[date_stress_actual])

        if not workload_psych_df.empty:
            workload_psych_trend = workload_psych_df.groupby(pd.Grouper(key=date_stress_actual, freq='W')).agg(
                Workload_Perception=(workload_actual, 'mean'),
                Psychological_Signals=(psych_actual, 'mean')
            ).reset_index()
            st.plotly_chart(viz.create_trend_chart(
                workload_psych_trend, date_stress_actual, ['Workload_Perception', 'Psychological_Signals'],
                "workload_vs_psych_chart_title", lang=st.session_state.selected_lang_code,
                y_axis_title_key="average_score_label", x_axis_title_key="date_time_axis_label",
                show_average_line=True, rolling_avg_window=4,
                value_col_units={'Workload_Perception': '', 'Psychological_Signals': ''}
            ), use_container_width=True)
        else: st.warning(_("no_data_workload_psych"))
    else: st.warning(_("no_data_workload_psych"))
else: st.info(_("no_data_available"))
st.markdown("---")

# 5. Interactive Plant Map (Placeholder)
st.header(_("plant_map_title"))
st.markdown(config.PLACEHOLDER_TEXT_PLANT_MAP, unsafe_allow_html=True)
st.warning(_("This module is a placeholder for future development."))
st.markdown("---")

# 6. Predictive AI Insights (Placeholder)
st.header(_("ai_insights_title"))
st.markdown(config.PLACEHOLDER_TEXT_AI_INSIGHTS, unsafe_allow_html=True)
st.warning(_("This module is a placeholder for future development."))
st.markdown("---")

# --- Optional & Strategic Modules ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"## {_('optional_modules_header')}")
show_optional = st.sidebar.checkbox(_('show_optional_modules'), key="show_optional_cb")
if show_optional:
    st.header(_('optional_modules_title'))
    with st.expander(_('optional_modules_title'), expanded=show_optional):
        st.markdown(_('optional_modules_list', config.TEXT_STRINGS["EN"]["optional_modules_list"]), unsafe_allow_html=True) # Allow HTML for markdown formatting

st.sidebar.markdown("---")
st.sidebar.caption(f"{_('dashboard_title')} v0.5.1 (SME Final Refinement)")
st.sidebar.caption(_("Built with Streamlit, Plotly, and Pandas."))
st.sidebar.caption(_("Data Last Updated: (N/A for sample data)"))
st.sidebar.caption(f"{_('dashboard_title')} v0.5.0 (SME Viz Refined)")
st.sidebar.caption(_("Built with Streamlit, Plotly, and Pandas.", "Built with Streamlit, Plotly, and Pandas."))
st.sidebar.caption(_("Data Last Updated: (N/A for sample data)", "Data Last Updated: (N/A for sample data)"))
