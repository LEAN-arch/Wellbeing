import streamlit as st
import pandas as pd
import numpy as np
import visualizations as viz # Custom visualization functions
import config                  # Main config (excluding glossary terms)
import insights                # New module for generating insights
from glossary_data import GLOSSARY_TERMS # Import glossary from its own file
from typing import List, Dict, Optional, Any, Union # For type hinting


# --- Page Configuration (Applied once at the top) ---
# Determine initial language for page config before full session state might be active
initial_lang_code_for_config = st.session_state.get('selected_lang_code', config.DEFAULT_LANG)
st.set_page_config(
    page_title=config.TEXT_STRINGS[initial_lang_code_for_config].get("app_title", "Vital Signs Dashboard"), # Fallback title
    page_icon=config.APP_ICON,
    layout="wide"
)

# --- Language Selection Logic ---
st.sidebar.markdown("---") # Visual separator in sidebar
available_langs_list = list(config.TEXT_STRINGS.keys())
if 'selected_lang_code' not in st.session_state: # Initialize session state for language if not present
    st.session_state.selected_lang_code = config.DEFAULT_LANG

def update_language_state_callback(): # Renamed for clarity
    st.session_state.selected_lang_code = st.session_state._app_lang_selector_key_widget # Access through private key

# Use a unique key for the selectbox
lang_selector_widget_key = "_app_lang_selector_key_widget" # Consistent key
st.sidebar.selectbox(
    label=f"{config.TEXT_STRINGS['EN'].get('language_selector', 'Language')} / {config.TEXT_STRINGS['ES'].get('language_selector', 'Idioma')}", # Bilingual label
    options=available_langs_list,
    index=available_langs_list.index(st.session_state.selected_lang_code), # Get current index
    format_func=lambda x: config.TEXT_STRINGS[x].get(f"language_name_full_{x.upper()}", x.upper()), # Get full lang name from config
    key=lang_selector_widget_key, # Assign unique key
    on_change=update_language_state_callback
)
# Fetch the current language dictionary based on session state
current_lang_texts = config.TEXT_STRINGS.get(st.session_state.selected_lang_code, config.TEXT_STRINGS[config.DEFAULT_LANG])

# --- Localization Helper Function ---
def _(text_key: str, default_text_override: Optional[str] = None) -> str:
    """Shortcut for getting localized text. Falls back to the key or override."""
    return current_lang_texts.get(text_key, default_text_override if default_text_override is not None else text_key)

# --- App Navigation (Main area control) ---
st.sidebar.markdown("---")
# Get localized labels for navigation options
dashboard_nav_label_loc = _("dashboard_nav_label", "Dashboard")
glossary_nav_label_loc = _("glossary_nav_label", "Glossary")

app_mode_selected = st.sidebar.radio(
    label=_("navigation_label", "Navigation"),
    options=[dashboard_nav_label_loc, glossary_nav_label_loc],
    key="app_navigation_mode_radio" # Unique key
)
st.sidebar.markdown("---")


# --- DASHBOARD MODE ---
if app_mode_selected == dashboard_nav_label_loc:
    # --- Data Loading ---
    @st.cache_data # Using the modern decorator
    def load_data_main(file_path_str: str, date_cols_actual_names: Optional[List[str]] = None):
        """Loads and minimally cleans data from a CSV file."""
        try:
            df = pd.read_csv(file_path_str, parse_dates=date_cols_actual_names if date_cols_actual_names else False)
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].notna().any():
                    try: df[col] = df[col].astype(str).str.strip() # Ensure string conversion before strip
                    except AttributeError: pass # In case some objects in column are not strippable
            return df
        except FileNotFoundError:
            st.error(_("error_loading_data").format(file_path_str) + f". " + _("check_file_path_instruction", "Please check the file path."))
            return pd.DataFrame()
        except Exception as e:
            st.error(_("error_loading_data").format(file_path_str) + f" - {_('exception_detail_prefix','Exception')}: {e}")
            return pd.DataFrame()

    # Load all raw dataframes
    # Ensure actual date column names are correctly fetched from COLUMN_MAP
    stability_date_actual_cols = [config.COLUMN_MAP.get("date")] if config.COLUMN_MAP.get("date") else None
    df_stability_raw = load_data_main(config.STABILITY_DATA_FILE, date_cols_actual_names=stability_date_cols)

    df_safety_raw = load_data_main(config.SAFETY_DATA_FILE) # 'month' column typically text ('Jan', 'Feb')

    df_engagement_raw = load_data_main(config.ENGAGEMENT_DATA_FILE)

    stress_date_actual_cols = [config.COLUMN_MAP.get("date")] if config.COLUMN_MAP.get("date") else None
    df_stress_raw = load_data_main(config.STRESS_DATA_FILE, date_cols_actual_names=stress_date_actual_cols)

    # --- Sidebar Filters for Dashboard ---
    st.sidebar.header(_("filters_header"))
    def get_unique_options_from_dfs_list(dfs_list_input: List[pd.DataFrame], column_conceptual_key: str) -> List[str]:
        actual_col_name = config.COLUMN_MAP.get(column_conceptual_key)
        if not actual_col_name: return []
        all_options = set()
        for df_item in dfs_list_input:
            if not df_item.empty and actual_col_name in df_item.columns:
                all_options.update(df_item[actual_col_name].dropna().astype(str).tolist())
        return sorted(list(all_options))

    all_raw_dataframes_for_filters = [df for df in [df_stability_raw, df_safety_raw, df_engagement_raw, df_stress_raw] if not df.empty]
    
    # Get unique options using conceptual keys and map to actual column names
    sites_options_list = get_unique_options_from_dfs_list(all_raw_dataframes_for_filters, "site")
    regions_options_list = get_unique_options_from_dfs_list(all_raw_dataframes_for_filters, "region")
    departments_options_list = get_unique_options_from_dfs_list(all_raw_dataframes_for_filters, "department")
    fcs_options_list = get_unique_options_from_dfs_list(all_raw_dataframes_for_filters, "fc")
    shifts_options_list = get_unique_options_from_dfs_list(all_raw_dataframes_for_filters, "shift")

    selected_sites = st.sidebar.multiselect(_("select_site"), options=sites_options_list, default=config.DEFAULT_SITES, key="sel_sites")
    selected_regions = st.sidebar.multiselect(_("select_region"), options=regions_options_list, default=config.DEFAULT_REGIONS, key="sel_regions")
    selected_departments = st.sidebar.multiselect(_("select_department"), options=departments_options_list, default=config.DEFAULT_DEPARTMENTS, key="sel_departments")
    selected_fcs = st.sidebar.multiselect(_("select_fc"), options=fcs_options_list, default=config.DEFAULT_FUNCTIONAL_CATEGORIES, key="sel_fcs")
    selected_shifts = st.sidebar.multiselect(_("select_shift"), options=shifts_options_list, default=config.DEFAULT_SHIFTS, key="sel_shifts")

    # --- Filter DataFrames Utility ---
    def apply_all_filters_to_df(df_to_filter: pd.DataFrame, conceptual_col_map: Dict[str, str], 
                              selections_by_conceptual_key: Dict[str, List[str]]) -> pd.DataFrame:
        if df_to_filter.empty: return df_to_filter.copy()
        df_filtered = df_to_filter.copy()
        for concept_key, selected_options_list in selections_by_conceptual_key.items():
            actual_col_in_df = conceptual_col_map.get(concept_key)
            if actual_col_in_df and selected_options_list and actual_col_in_df in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[actual_col_in_df].astype(str).isin([str(opt) for opt in selected_options_list])]
        return df_filtered

    filter_selections_active_map = {
        'site': selected_sites, 'region': selected_regions, 'department': selected_departments,
        'fc': selected_fcs, 'shift': selected_shifts
    }

    df_stability_filtered = apply_all_filters_to_df(df_stability_raw, config.COLUMN_MAP, filter_selections_active_map)
    df_safety_filtered = apply_all_filters_to_df(df_safety_raw, config.COLUMN_MAP, filter_selections_active_map)
    df_engagement_filtered = apply_all_filters_to_df(df_engagement_raw, config.COLUMN_MAP, filter_selections_active_map)
    df_stress_filtered = apply_all_filters_to_df(df_stress_raw, config.COLUMN_MAP, filter_selections_active_map)

    # --- Main Dashboard Area ---
    st.title(_("dashboard_title"))
    st.markdown(_("dashboard_subtitle"))
    st.caption(_("alignment_note"))
    st.markdown("---")
    st.info(_("psych_safety_note")) # Global note for the dashboard
    st.markdown("---")

    # --- Helper for dummy previous values for MVP demo ---
    def get_dummy_prev_val(curr_val: Optional[Union[int, float, np.number]], 
                           factor: float = 0.1, is_percent: bool = False, 
                           variation_abs: Optional[Union[int, float]] = None) -> Optional[float]:
        if pd.isna(curr_val) or not isinstance(curr_val, (int,float,np.number)): return None
        
        # Prefer absolute variation if value is small or variation_abs provided
        if variation_abs is not None or (abs(float(curr_val)) < 10 and not is_percent) :
            abs_var = variation_abs if variation_abs is not None else (1 if float(curr_val) >= 0 else -1) # default absolute change of 1
            change_amt = abs_var * np.random.uniform(-1, 1)
        else:
            change_amt = float(curr_val) * factor * np.random.uniform(-0.7, 0.7) # More subtle variation
        
        prev = float(curr_val) - change_amt
        if is_percent: return round(max(0.0, min(100.0, prev)),1)
        return round(prev, 1) if not pd.isna(prev) else None


    # --- 1. Laboral Stability Panel ---
    st.header(_("stability_panel_title"))
    if not df_stability_filtered.empty:
        cols_metrics_stab = st.columns(4)
        rot_rate_actual_col = config.COLUMN_MAP["rotation_rate"]
        avg_rotation_current = df_stability_filtered[rot_rate_actual_col].mean() if rot_rate_actual_col in df_stability_filtered.columns else float('nan')
        prev_avg_rotation_val = get_dummy_prev_val(avg_rotation_current, 0.05, True)

        with cols_metrics_stab[0]:
            viz.display_metric_card(st, "rotation_rate_gauge", avg_rotation_current, st.session_state.selected_lang_code, unit="%",
                                    higher_is_better=False, target_value=config.STABILITY_ROTATION_RATE["target"],
                                    threshold_good=config.STABILITY_ROTATION_RATE["good"],
                                    threshold_warning=config.STABILITY_ROTATION_RATE["warning"],
                                    previous_value=prev_avg_rotation_val, 
                                    help_text_key="rotation_rate_metric_help".format(target=config.STABILITY_ROTATION_RATE['target']))
            if pd.notna(avg_rotation_current): # Only show gauge if there's a value
                st.plotly_chart(viz.create_kpi_gauge(
                    avg_rotation_current, "rotation_rate_gauge", st.session_state.selected_lang_code, unit="%", higher_is_worse=True,
                    threshold_good=config.STABILITY_ROTATION_RATE["good"], threshold_warning=config.STABILITY_ROTATION_RATE["warning"],
                    target_line_value=config.STABILITY_ROTATION_RATE["target"], previous_value=prev_avg_rotation_val
                ), use_container_width=True)
            
        retention_metrics_def = [
            ("retention_6m", "retention_6m_metric"), ("retention_12m", "retention_12m_metric"), ("retention_18m", "retention_18m_metric")
        ]
        for i, (col_map_k_ret, label_k_ret) in enumerate(retention_metrics_def):
            actual_col_ret = config.COLUMN_MAP[col_map_k_ret]
            val_ret = df_stability_filtered[actual_col_ret].mean() if actual_col_ret in df_stability_filtered.columns else float('nan')
            prev_val_ret = get_dummy_prev_val(val_ret, 0.03, True)
            with cols_metrics_stab[i+1]:
                viz.display_metric_card(st, label_k_ret, val_ret, st.session_state.selected_lang_code, unit="%", higher_is_better=True,
                                        target_value=config.STABILITY_RETENTION["good"], 
                                        threshold_good=config.STABILITY_RETENTION["good"], threshold_warning=config.STABILITY_RETENTION["warning"],
                                        previous_value=prev_val_ret,
                                        help_text_key="retention_metric_help".format(target=config.STABILITY_RETENTION['good']))
        st.markdown("<br>", unsafe_allow_html=True) # Visual spacer

        # Hires vs. Exits Trend Chart
        date_actual_stability = config.COLUMN_MAP["date"]
        hires_actual_col = config.COLUMN_MAP["hires"]
        exits_actual_col = config.COLUMN_MAP["exits"]

        if all(col in df_stability_filtered.columns for col in [date_actual_stability, hires_actual_col, exits_actual_col]):
            trend_df_stability = df_stability_filtered[[date_actual_stability, hires_actual_col, exits_actual_col]].copy()
            if not pd.api.types.is_datetime64_any_dtype(trend_df_stability[date_actual_stability]):
                trend_df_stability[date_actual_stability] = pd.to_datetime(trend_df_stability[date_actual_stability], errors='coerce')
            trend_df_stability.dropna(subset=[date_actual_stability], inplace=True); trend_df_stability.sort_values(by=date_actual_stability, inplace=True)
            
            if not trend_df_stability.empty:
                agg_trend_stability = trend_df_stability.groupby(pd.Grouper(key=date_actual_stability, freq='M')).agg(
                    AggHires=(hires_actual_col, 'sum'), AggExits=(exits_actual_col, 'sum')).reset_index()
                
                map_trend_stability = {"hires_label": "AggHires", "exits_label": "AggExits"}
                unit_map_trend_stability = {"AggHires": "", "AggExits": ""} # Counts, no unit needed for hover suffix if axis has it
                
                st.plotly_chart(viz.create_trend_chart(
                    agg_trend_stability, date_actual_stability, map_trend_stability, "hires_vs_exits_chart_title", st.session_state.selected_lang_code,
                    y_axis_title_key="people_count_label", x_axis_title_key="month_axis_label",
                    show_average_line=True, rolling_avg_window=3, value_col_units_map=unit_map_trend_stability
                ), use_container_width=True)
            else: st.warning(_("no_data_hires_exits"))
        else: st.warning(_("no_data_hires_exits"))
        
        # Actionable Insights for Stability Panel
        action_insights_stability = insights.generate_stability_insights(df_stability_filtered, avg_rotation_current, agg_trend_stability if 'agg_trend_stability' in locals() else None, st.session_state.selected_lang_code)
        if action_insights_stability:
            st.markdown("---")
            st.subheader(_("actionable_insights_title"))
            for insight_item in action_insights_stability: st.markdown(f"💡 {insight_item}")
    else: st.info(_("no_data_available"))
    st.markdown("---")


    # --- 2. Safety Pulse Module ---
    st.header(_("safety_pulse_title"))
    if not df_safety_filtered.empty:
        cols_layout_safety = st.columns([2, 1, 1]) 
        month_col = config.COLUMN_MAP["month"]
        inc_col = config.COLUMN_MAP["incidents"]
        nm_col = config.COLUMN_MAP["near_misses"]
        dwa_col = config.COLUMN_MAP["days_without_accidents"]
        aa_col = config.COLUMN_MAP["active_alerts"]

        with cols_layout_safety[0]: # Bar Chart of Incidents & Near Misses
            if all(c in df_safety_filtered.columns for c in [month_col, inc_col, nm_col]):
                summary_safety_events = df_safety_filtered.groupby(month_col, as_index=False).agg(
                    IncidentsAgg=(inc_col, 'sum'), NearMissesAgg=(nm_col, 'sum'))
                try: # Sort months chronologically
                    month_cat_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    summary_safety_events[month_col] = pd.Categorical(summary_safety_events[month_col].astype(str), categories=month_cat_order, ordered=True)
                    summary_safety_events.sort_values(month_col, inplace=True); summary_safety_events.dropna(subset=[month_col], inplace=True)
                except Exception: summary_safety_events.sort_values(by=month_col, errors='ignore', inplace=True)
                
                if not summary_safety_events.empty:
                    map_safety_event_bars = {"incidents_label": "IncidentsAgg", "near_misses_label": "NearMissesAgg"}
                    st.plotly_chart(viz.create_comparison_bar_chart(
                        summary_safety_events, month_col, map_safety_event_bars, "monthly_incidents_chart_title", st.session_state.selected_lang_code,
                        x_axis_title_key="month_axis_label", y_axis_title_key="count_label",
                        barmode='stack', show_total_for_stacked=True, data_label_format_str=".0f"
                    ), use_container_width=True)
                else: st.warning(_("no_data_incidents_near_misses"))
            else: st.warning(_("no_data_incidents_near_misses"))
        
        total_inc_current_period = df_safety_filtered[inc_col].sum() if inc_col in df_safety_filtered.columns else float('nan')
        current_dwa_val = df_safety_filtered[dwa_col].max() if dwa_col in df_safety_filtered.columns else float('nan')
        with cols_layout_safety[1]: # Days Without Accidents Metric
            prev_dwa_val = get_dummy_prev_val(current_dwa_val, 0.1, variation_abs=5)
            viz.display_metric_card(st, "days_without_accidents_metric", current_dwa_val, st.session_state.selected_lang_code, unit=" "+_("days_unit"),
                                   higher_is_better=True, help_text_key="days_no_incident_help",
                                   threshold_good=config.SAFETY_DAYS_NO_INCIDENTS["good"], threshold_warning=config.SAFETY_DAYS_NO_INCIDENTS["warning"],
                                   previous_value=prev_dwa_val)
        with cols_layout_safety[2]: # Active Safety Alerts Metric
            active_alerts_val = df_safety_filtered[aa_col].sum() if aa_col in df_safety_filtered.columns else float('nan')
            prev_active_alerts_val = get_dummy_prev_val(active_alerts_val, 0.2, variation_abs=1)
            prev_active_alerts_val = int(prev_active_alerts_val) if pd.notna(prev_active_alerts_val) else None
            viz.display_metric_card(st, "active_safety_alerts_metric", active_alerts_val, st.session_state.selected_lang_code, unit="",
                                   higher_is_better=False, target_value=0, threshold_good=0, threshold_warning=1, # Target is 0 alerts
                                   previous_value=prev_active_alerts_val)

        # Actionable Insights for Safety Pulse
        action_insights_safety = insights.generate_safety_insights(df_safety_filtered, current_dwa_val, total_inc_current_period, st.session_state.selected_lang_code)
        if action_insights_safety:
            st.markdown("---")
            st.subheader(_("actionable_insights_title"))
            for insight_text in action_insights_safety: st.markdown(f"💡 {insight_text}")
    else: st.info(_("no_data_available"))
    st.markdown("---")

    # --- 3. Employee Engagement & Commitment ---
    st.header(_("engagement_title"))
    if not df_engagement_filtered.empty:
        cols_layout_engagement = st.columns([2,1]) # Radar Chart, Metrics
        with cols_layout_engagement[0]: # Radar Chart
            radar_data_points_eng = []
            radar_targets_localized_eng = {}
            
            for conceptual_key_eng, actual_col_eng_radar in config.COLUMN_MAP["engagement_radar_dims_cols"].items():
                if actual_col_eng_radar in df_engagement_filtered.columns:
                    avg_score_eng_radar = df_engagement_filtered[actual_col_eng_radar].mean()
                    label_key_for_display_eng = config.COLUMN_MAP["engagement_radar_dims_labels"].get(conceptual_key_eng, actual_col_eng_radar)
                    display_name_eng_radar = _(label_key_for_display_eng, actual_col_eng_radar.replace('_', ' ').title())
                    if pd.notna(avg_score_eng_radar):
                        radar_data_points_eng.append({"Dimension": display_name_eng_radar, "Score": avg_score_eng_radar})
                        # Map targets using the final display name
                        radar_targets_localized_eng[display_name_eng_radar] = config.ENGAGEMENT_RADAR_DIM_TARGET 
            
            if radar_data_points_eng:
                df_radar_visual = pd.DataFrame(radar_data_points_eng)
                st.plotly_chart(viz.create_enhanced_radar_chart(
                    df_radar_visual, "Dimension", "Score", "engagement_dimensions_radar_title", st.session_state.selected_lang_code,
                    range_max_override=config.ENGAGEMENT_RADAR_DIM_SCALE_MAX, target_values_map=radar_targets_localized_eng, fill_opacity=0.4
                ), use_container_width=True)
            elif any(config.COLUMN_MAP["engagement_radar_dims_cols"].get(k) in df_engagement_filtered.columns for k in config.COLUMN_MAP["engagement_radar_dims_cols"]):
                 st.warning(_("no_data_radar"))
            else: st.warning(_("no_data_radar_columns"))
        
        with cols_layout_engagement[1]: # Engagement Metric Cards
            kpis_engagement_list = [
                ("labor_climate_score", "labor_climate_score_metric", "", True, config.ENGAGEMENT_CLIMATE_SCORE, None),
                ("enps_score", "enps_metric", "", True, config.ENGAGEMENT_ENPS, "enps_metric_help"),
                ("participation_rate", "survey_participation_metric", "%", True, config.ENGAGEMENT_PARTICIPATION, None),
                ("recognitions_count", "recognitions_count_metric", "", True, None, None) # No thresholds defined for recognitions count, just track
            ]
            avg_enps_for_insight, avg_climate_for_insight, participation_for_insight = float('nan'), float('nan'), float('nan')

            for col_map_key_eng, label_key_eng, unit_str_eng, hib_eng, thresholds_dict_eng, help_key_eng in kpis_engagement_list:
                actual_col_name_eng = config.COLUMN_MAP[col_map_key_eng]
                is_count = "count" in col_map_key_eng # Heuristic for sum vs mean
                current_val_metric_eng = (df_engagement_filtered[actual_col_name_eng].sum() if is_count else df_engagement_filtered[actual_col_name_eng].mean()) \
                                         if actual_col_name_eng in df_engagement_filtered.columns else float('nan')
                prev_val_metric_eng = get_dummy_prev_val(current_val_metric_eng, 0.05, (unit_str_eng=="%"), variation_abs=5 if is_count else None)
                
                thresh_good_eng = thresholds_dict_eng.get("good") if thresholds_dict_eng else None
                thresh_warn_eng = thresholds_dict_eng.get("warning") if thresholds_dict_eng else None
                help_text_formatted_eng = _(help_key_eng, "").format(target=thresh_good_eng) if help_key_eng and thresh_good_eng and "{target}" in _(help_key_eng,"") else _(help_key_eng,"")


                viz.display_metric_card(st, label_key_eng, current_val_metric_eng, st.session_state.selected_lang_code, unit=unit_str_eng, 
                                        higher_is_better=hib_eng, target_value=thresh_good_eng, 
                                        threshold_good=thresh_good_eng, threshold_warning=thresh_warn_eng,
                                        previous_value=prev_val_metric_eng, help_text_key=help_text_formatted_eng) # Pass formatted help text if needed or key
                
                if col_map_key_eng == "enps_score": avg_enps_for_insight = current_val_metric_eng
                if col_map_key_eng == "labor_climate_score": avg_climate_for_insight = current_val_metric_eng
                if col_map_key_eng == "participation_rate": participation_for_insight = current_val_metric_eng

        # Actionable Insights for Engagement
        action_insights_engagement = insights.generate_engagement_insights(avg_enps_for_insight, avg_climate_for_insight, participation_for_insight, st.session_state.selected_lang_code)
        if action_insights_engagement:
            st.markdown("---")
            st.subheader(_("actionable_insights_title"))
            for insight_text in action_insights_engagement: st.markdown(f"💡 {insight_text}")
    else: st.info(_("no_data_available"))
    st.markdown("---")

    # --- 4. Operational Stress Dashboard ---
    st.header(_("stress_title"))
    if not df_stress_filtered.empty:
        cols_stress_layout = st.columns([1, 2]) # Semaforo first, then Bar Chart for Shift Load
        stress_level_actual = config.COLUMN_MAP["stress_level_survey"]
        overtime_actual = config.COLUMN_MAP["overtime_hours"]
        unfilled_actual = config.COLUMN_MAP["unfilled_shifts"]
        date_actual_stress = config.COLUMN_MAP["date"]
        workload_actual = config.COLUMN_MAP["workload_perception"]
        psych_actual = config.COLUMN_MAP["psychological_signals"]

        avg_stress_current = df_stress_filtered[stress_level_actual].mean() if stress_level_actual in df_stress_filtered.columns else float('nan')
        with cols_stress_layout[0]: # Stress Semaforo / Indicator
            st.subheader(_("overall_stress_indicator_title"))
            st.plotly_chart(viz.create_stress_semaforo_visual(
                avg_stress_current, st.session_state.selected_lang_code, scale_max=config.STRESS_LEVEL_PSYCHOSOCIAL["max_scale"]
            ), use_container_width=True)
            # Format target for help text
            stress_target_help = config.STRESS_LEVEL_PSYCHOSOCIAL['low']
            st.caption(_("stress_indicator_help").format(target=f"{stress_target_help:.1f}"))


        with cols_stress_layout[1]: # Shift Load Bar Chart
            if all(c in df_stress_filtered.columns for c in [date_actual_stress, overtime_actual, unfilled_actual]):
                df_sl_trend_stress = df_stress_filtered[[date_actual_stress, overtime_actual, unfilled_actual]].copy()
                if not pd.api.types.is_datetime64_any_dtype(df_sl_trend_stress[date_actual_stress]):
                    df_sl_trend_stress[date_actual_stress] = pd.to_datetime(df_sl_trend_stress[date_actual_stress], errors='coerce')
                df_sl_trend_stress.dropna(subset=[date_actual_stress], inplace=True); df_sl_trend_stress.sort_values(by=date_actual_stress, inplace=True)
                if not df_sl_trend_stress.empty:
                    sl_summary_stress = df_sl_trend_stress.groupby(pd.Grouper(key=date_actual_stress, freq='M')).agg(
                       OvertimeData=(overtime_actual, 'sum'), UnfilledData=(unfilled_actual, 'sum')).reset_index()
                    map_sl_stress_bars = {"overtime_label": "OvertimeData", "unfilled_shifts_label": "UnfilledData"}
                    st.plotly_chart(viz.create_comparison_bar_chart(
                        sl_summary_stress, date_actual_stress, map_sl_stress_bars, "monthly_shift_load_chart_title", st.session_state.selected_lang_code,
                        x_axis_title_key="month_axis_label", y_axis_title_key="hours_or_shifts_label",
                        barmode='group', data_label_format_str=".0f"
                    ), use_container_width=True)
                else: st.warning(_("no_data_shift_load"))
            else: st.warning(_("no_data_shift_load"))
        st.markdown("---") # Separator

        # Workload vs Psych Signals Trend (Full Width below)
        df_stress_trends_for_insight_func = pd.DataFrame() # Init empty for insights function
        if all(c in df_stress_filtered.columns for c in [date_actual_stress, workload_actual, psych_actual]):
            df_wp_ps_trend_stress = df_stress_filtered[[date_actual_stress, workload_actual, psych_actual]].copy()
            if not pd.api.types.is_datetime64_any_dtype(df_wp_ps_trend_stress[date_actual_stress]):
                df_wp_ps_trend_stress[date_actual_stress] = pd.to_datetime(df_wp_ps_trend_stress[date_actual_stress], errors='coerce')
            df_wp_ps_trend_stress.dropna(subset=[date_actual_stress], inplace=True); df_wp_ps_trend_stress.sort_values(by=date_actual_stress, inplace=True)
            
            if not df_wp_ps_trend_stress.empty:
                # Using specific aggregate column names for insights logic later
                df_stress_trends_for_insight_func = df_wp_ps_trend_stress.groupby(pd.Grouper(key=date_actual_stress, freq='M')).agg(
                    Workload_Avg=(workload_actual, 'mean'), Psych_Signals_Avg=(psych_actual, 'mean')).reset_index()
                
                map_wp_ps_stress_trend = {"workload_perception_label": "Workload_Avg", "psychological_signals_label": "Psych_Signals_Avg"}
                unit_map_wp_ps_stress = {"Workload_Avg": "", "Psych_Signals_Avg": ""}
                st.plotly_chart(viz.create_trend_chart(
                    df_stress_trends_for_insight_func, date_actual_stress, map_wp_ps_stress_trend, "workload_vs_psych_chart_title", st.session_state.selected_lang_code,
                    y_axis_title_key="average_score_label", x_axis_title_key="month_axis_label",
                    show_average_line=True, rolling_avg_window=3, value_col_units_map=unit_map_wp_ps_stress
                ), use_container_width=True)
            else: st.warning(_("no_data_workload_psych"))
        else: st.warning(_("no_data_workload_psych"))

        # Actionable Insights for Stress
        action_insights_stress = insights.generate_stress_insights(avg_stress_current, df_stress_trends_for_insight_func, st.session_state.selected_lang_code)
        if action_insights_stress:
            st.markdown("---")
            st.subheader(_("actionable_insights_title"))
            for insight_text in action_insights_stress: st.markdown(f"💡 {insight_text}")
    else: st.info(_("no_data_available"))
    st.markdown("---")

    # --- 5. Interactive Plant Map (Placeholder) ---
    st.header(_("plant_map_title"))
    st.markdown(config.PLACEHOLDER_TEXT_PLANT_MAP, unsafe_allow_html=True)
    st.warning(_("This module is a placeholder for future development.", "Module in development phase.")) # Generic placeholder message
    st.markdown("---")

    # --- 6. Predictive AI Insights (Placeholder) ---
    st.header(_("ai_insights_title"))
    st.markdown(config.PLACEHOLDER_TEXT_AI_INSIGHTS, unsafe_allow_html=True)
    st.warning(_("This module is a placeholder for future development.", "Module in development phase."))
    st.markdown("---")


# --- GLOSSARY PAGE ---
elif app_mode_selected == glossary_nav_label_loc:
    st.title(_("glossary_page_title"))
    st.markdown(_("glossary_intro"))
    st.markdown("---")

    search_term_glossary = st.text_input(_("search_term_label"), key="glossary_search_input_field") # Unique key
    
    sorted_glossary_data = dict(sorted(GLOSSARY_TERMS.items())) # GLOSSARY_TERMS from glossary_data.py

    num_terms_found = 0
    if sorted_glossary_data:
        for term_key_eng, definitions_dict in sorted_glossary_data.items():
            # Search logic: match in term key or any definition text
            term_header_display = term_key_eng # Consistent header (English)
            show_this_term = True
            if search_term_glossary:
                search_val_lower = search_term_glossary.lower()
                match_in_key = search_val_lower in term_header_display.lower()
                match_in_def_en = search_val_lower in definitions_dict.get("EN", "").lower() if definitions_dict.get("EN") else False
                match_in_def_es = search_val_lower in definitions_dict.get("ES", "").lower() if definitions_dict.get("ES") else False
                if not (match_in_key or match_in_def_en or match_in_def_es):
                    show_this_term = False
            
            if show_this_term:
                num_terms_found +=1
                with st.expander(term_header_display, expanded=(search_term_glossary != "")):
                    primary_lang_key = st.session_state.selected_lang_code.upper()
                    secondary_lang_key = "ES" if primary_lang_key == "EN" else "EN"

                    if primary_lang_key in definitions_dict and definitions_dict[primary_lang_key]:
                        st.markdown(f"**{_('definition_label')}:**")
                        st.markdown(definitions_dict[primary_lang_key])
                    
                    # Show secondary language definition if available and different from primary
                    if secondary_lang_key in definitions_dict and definitions_dict[secondary_lang_key] and \
                       definitions_dict.get(primary_lang_key) != definitions_dict[secondary_lang_key] :
                         if primary_lang_key in definitions_dict and definitions_dict[primary_lang_key]: # Add spacer
                             st.markdown("---")
                         st.caption(f"*{_(f'language_name_full_{secondary_lang_key.upper()}', secondary_lang_key)}:* {definitions_dict[secondary_lang_key]}")
                # st.markdown("---") # Optional: separator between expanders
        
        if search_term_glossary and num_terms_found == 0:
            st.info(_("no_term_found"))
    else: # Glossary is empty
        st.warning(_("glossary_empty_message"))


# --- Optional Modules Display (Always in Sidebar for toggling) ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"## {_('optional_modules_header')}")
show_optional_sb_expander = st.sidebar.checkbox(_('show_optional_modules'), key="sidebar_optional_mods_cb", value=False)
if show_optional_sb_expander :
    with st.sidebar.expander(_('optional_modules_title'), expanded=True): # Expanded if checkbox is ticked
        # Ensure the list content defaults gracefully if a language string is missing
        optional_list_content = _('optional_modules_list', default_text_override=config.TEXT_STRINGS[config.DEFAULT_LANG].get('optional_modules_list',""))
        st.markdown(optional_list_content, unsafe_allow_html=True)


# --- Footer / App Info (Always in Sidebar) ---
st.sidebar.markdown("---")
st.sidebar.caption(f"{_(config.APP_TITLE_KEY)} {config.APP_VERSION}") # Using localized app title key
st.sidebar.caption(_("Built with Streamlit, Plotly, and Pandas.", "Built with Streamlit, Plotly, and Pandas."))
st.sidebar.caption(_("Data Last Updated: (N/A for sample data)", "Última Actualización de Datos: (N/A para datos de muestra)"))
