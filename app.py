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
selected_lang_code_from_widget = st.sidebar.selectbox(
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
if app_mode_selected == dashboard_nav_label_loc: # Load data and display dashboard only if selected
    # --- Data Loading ---
    @st.cache_data # Using the modern decorator
    def load_data_main(file_path_str: str, date_cols_actual_names: Optional[List[str]] = None):
        """Loads and minimally cleans data from a CSV file."""
        try:
            df = pd.read_csv(file_path_str, parse_dates=date_cols_actual_names if date_cols_actual_names else False)
            for col in df.columns: # Iterate over actual columns in the loaded DataFrame
                if df[col].dtype == 'object' and df[col].notna().any(): # Check if column is of object type
                    try: df[col] = df[col].astype(str).str.strip() # Ensure string conversion before strip
                    except AttributeError: pass # Handles non-string objects if any slip through
            return df
        except FileNotFoundError:
            # Using _() for localized error message (ensure keys exist in TEXT_STRINGS)
            st.error(_("error_loading_data").format(file_path_str) + f". " + _("check_file_path_instruction", "Please verify file path and presence."))
            return pd.DataFrame() # Return empty DataFrame to prevent downstream errors
        except Exception as e:
            st.error(_("error_loading_data").format(file_path_str) + f" - {_('exception_detail_prefix','Exception')}: {e}")
            return pd.DataFrame()

    # Load all raw dataframes, using actual column names from config.COLUMN_MAP for date parsing
    stability_date_cols_parse = [config.COLUMN_MAP.get("date")] if config.COLUMN_MAP.get("date") else None
    df_stability_raw = load_data_main(config.STABILITY_DATA_FILE, date_cols_actual_names=stability_date_cols_parse)

    # For safety_data.csv, 'month' is text (Jan, Feb), no date parsing needed here for that column.
    df_safety_raw = load_data_main(config.SAFETY_DATA_FILE, date_cols_actual_names=None)

    df_engagement_raw = load_data_main(config.ENGAGEMENT_DATA_FILE, date_cols_actual_names=None)

    stress_date_cols_parse = [config.COLUMN_MAP.get("date")] if config.COLUMN_MAP.get("date") else None
    df_stress_raw = load_data_main(config.STRESS_DATA_FILE, date_cols_actual_names=stress_date_cols_parse)

    # --- Sidebar Filters for Dashboard ---
    st.sidebar.header(_("filters_header"))
    def get_unique_options_from_dfs_list(dfs_list_input: List[pd.DataFrame], column_conceptual_key: str) -> List[str]:
        actual_col_name = config.COLUMN_MAP.get(column_conceptual_key)
        if not actual_col_name: return [] # If key not in map
        all_options_set = set()
        for df_item in dfs_list_input:
            if not df_item.empty and actual_col_name in df_item.columns:
                all_options_set.update(df_item[actual_col_name].dropna().astype(str).tolist())
        return sorted(list(all_options_set))

    # Get filter options using conceptual keys
    all_raw_dataframes_list = [df for df in [df_stability_raw, df_safety_raw, df_engagement_raw, df_stress_raw] if not df.empty]
    sites_options_list = get_unique_options_from_dfs_list(all_raw_dataframes_list, "site")
    regions_options_list = get_unique_options_from_dfs_list(all_raw_dataframes_list, "region")
    departments_options_list = get_unique_options_from_dfs_list(all_raw_dataframes_list, "department")
    fcs_options_list = get_unique_options_from_dfs_list(all_raw_dataframes_list, "fc")
    shifts_options_list = get_unique_options_from_dfs_list(all_raw_dataframes_list, "shift")

    # Ensure unique keys for Streamlit widgets if they appear multiple times in script logic
    selected_sites = st.sidebar.multiselect(_("select_site"), options=sites_options_list, default=config.DEFAULT_SITES, key="filter_multiselect_sites")
    selected_regions = st.sidebar.multiselect(_("select_region"), options=regions_options_list, default=config.DEFAULT_REGIONS, key="filter_multiselect_regions")
    selected_departments = st.sidebar.multiselect(_("select_department"), options=departments_options_list, default=config.DEFAULT_DEPARTMENTS, key="filter_multiselect_departments")
    selected_fcs = st.sidebar.multiselect(_("select_fc"), options=fcs_options_list, default=config.DEFAULT_FUNCTIONAL_CATEGORIES, key="filter_multiselect_fcs")
    selected_shifts = st.sidebar.multiselect(_("select_shift"), options=shifts_options_list, default=config.DEFAULT_SHIFTS, key="filter_multiselect_shifts")

    # --- Filter DataFrames Utility ---
    def apply_all_filters_to_df(df_to_filter: pd.DataFrame, col_map: Dict[str, str], 
                              selections: Dict[str, List[str]]) -> pd.DataFrame:
        if df_to_filter.empty: return df_to_filter.copy() # Return copy if empty
        df_filtered = df_to_filter.copy() # Work on a copy
        for concept_key, selected_opts_list in selections.items():
            actual_col_in_df = col_map.get(concept_key) # Get actual CSV column name
            if actual_col_in_df and selected_opts_list and actual_col_in_df in df_filtered.columns:
                # Ensure comparison is string-to-string as filter options are strings
                df_filtered = df_filtered[df_filtered[actual_col_in_df].astype(str).isin([str(opt) for opt in selected_opts_list])]
        return df_filtered

    filter_selections_active_map = {
        'site': selected_sites, 'region': selected_regions, 'department': selected_departments,
        'fc': selected_fcs, 'shift': selected_shifts
    }

    # Apply filters to create the DataFrames for visualization
    df_stability_filtered = apply_all_filters_to_df(df_stability_raw, config.COLUMN_MAP, filter_selections_active_map)
    df_safety_filtered = apply_all_filters_to_df(df_safety_raw, config.COLUMN_MAP, filter_selections_active_map)
    df_engagement_filtered = apply_all_filters_to_df(df_engagement_raw, config.COLUMN_MAP, filter_selections_active_map)
    df_stress_filtered = apply_all_filters_to_df(df_stress_raw, config.COLUMN_MAP, filter_selections_active_map)

    # --- Main Dashboard Area ---
    st.title(_("dashboard_title"))
    st.markdown(_("dashboard_subtitle"))
    st.caption(_("alignment_note"))
    st.markdown("---")
    st.info(_("psych_safety_note")) # Global note regarding data handling and privacy
    st.markdown("---")

    # Helper for dummy previous values (for MVP demonstration purposes)
    def get_dummy_prev_val(curr_val: Optional[Union[int, float, np.number]], 
                           factor: float = 0.1, is_percent: bool = False, 
                           variation_abs: Optional[Union[int, float]] = None) -> Optional[float]:
        if pd.isna(curr_val) or not isinstance(curr_val, (int,float,np.number)): return None
        
        current_float_val = float(curr_val) # Ensure it's a float for calculations
        if variation_abs is not None or (abs(current_float_val) < 10 and not is_percent) : # Prefer absolute for small numbers or if specified
             abs_var_val = variation_abs if variation_abs is not None else (1.0 if current_float_val >= 0 else -1.0) 
             change_amount = abs_var_val * np.random.uniform(-1, 1)
        else: # Percentage based variation for larger numbers
            change_amount = current_float_val * factor * np.random.uniform(-0.7, 0.7) # More subtle % variation
        
        previous_val_calculated = current_float_val - change_amount
        if is_percent: return round(max(0.0, min(100.0, previous_val_calculated)),1) # Clamp percentages
        return round(previous_val_calculated, 1) if not pd.isna(previous_val_calculated) else None


    # --- 1. Laboral Stability Panel ---
    st.header(_("stability_panel_title"))
    agg_trend_stability = pd.DataFrame() # Initialize for insights
    avg_rotation_current = float('nan')

    if not df_stability_filtered.empty:
        cols_metrics_stab = st.columns(4) # For KPI Gauge + 3 Retention Metrics
        
        # Rotation Rate
        rot_rate_actual_col = config.COLUMN_MAP.get("rotation_rate", "rotation_rate")
        avg_rotation_current = df_stability_filtered[rot_rate_actual_col].mean() if rot_rate_actual_col in df_stability_filtered.columns else float('nan')
        prev_avg_rotation_val = get_dummy_prev_val(avg_rotation_current, 0.05, True)

        with cols_metrics_stab[0]:
            # Format the target value for help text string replacement
            target_rot_fmt = config.STABILITY_ROTATION_RATE["target"]
            target_rot_str = f"{target_rot_fmt:.0f}" if target_rot_fmt % 1 == 0 else f"{target_rot_fmt:.1f}"
            help_text_rot_final = _("rotation_rate_metric_help").format(target=target_rot_str)

            viz.display_metric_card(st, "rotation_rate_gauge", avg_rotation_current, st.session_state.selected_lang_code, unit="%",
                                    higher_is_better=False, target_value=config.STABILITY_ROTATION_RATE["target"],
                                    threshold_good=config.STABILITY_ROTATION_RATE["good"],
                                    threshold_warning=config.STABILITY_ROTATION_RATE["warning"],
                                    previous_value=prev_avg_rotation_val, help_text_key=help_text_rot_final) # Pass formatted help text directly
            if pd.notna(avg_rotation_current):
                st.plotly_chart(viz.create_kpi_gauge(
                    avg_rotation_current, "rotation_rate_gauge", st.session_state.selected_lang_code, unit="%", higher_is_worse=True,
                    threshold_good=config.STABILITY_ROTATION_RATE["good"], threshold_warning=config.STABILITY_ROTATION_RATE["warning"],
                    target_line_value=config.STABILITY_ROTATION_RATE["target"], previous_value=prev_avg_rotation_val
                ), use_container_width=True)
            
        # Retention Metrics
        retention_metric_definitions = [
            ("retention_6m", "retention_6m_metric"), 
            ("retention_12m", "retention_12m_metric"), 
            ("retention_18m", "retention_18m_metric")
        ]
        for i, (col_map_key_retention, label_key_retention) in enumerate(retention_metric_definitions):
            actual_col_name_retention = config.COLUMN_MAP.get(col_map_key_retention, col_map_key_retention)
            value_retention = df_stability_filtered[actual_col_name_retention].mean() if actual_col_name_retention in df_stability_filtered.columns else float('nan')
            previous_value_retention = get_dummy_prev_val(value_retention, 0.03, True)
            with cols_metrics_stab[i+1]: # Place in subsequent columns
                target_ret_fmt = config.STABILITY_RETENTION["good"]
                target_ret_str = f"{target_ret_fmt:.0f}" if target_ret_fmt % 1 == 0 else f"{target_ret_fmt:.1f}"
                help_text_ret_final = _("retention_metric_help").format(target=target_ret_str)

                viz.display_metric_card(st, label_key_retention, value_retention, st.session_state.selected_lang_code, unit="%", higher_is_better=True,
                                        target_value=config.STABILITY_RETENTION["good"], 
                                        threshold_good=config.STABILITY_RETENTION["good"], 
                                        threshold_warning=config.STABILITY_RETENTION["warning"], # Remember, for higher_is_better=True, warning is below good
                                        previous_value=previous_value_retention,
                                        help_text_key=help_text_ret_final)
        st.markdown("<br>", unsafe_allow_html=True) # Add some space before the trend chart

        # Hires vs. Exits Trend Chart
        date_actual_col_stability = config.COLUMN_MAP.get("date", "date")
        hires_actual_col_stability = config.COLUMN_MAP.get("hires", "hires")
        exits_actual_col_stability = config.COLUMN_MAP.get("exits", "exits")

        if all(col_name in df_stability_filtered.columns for col_name in [date_actual_col_stability, hires_actual_col_stability, exits_actual_col_stability]):
            trend_df_for_stability = df_stability_filtered[[date_actual_col_stability, hires_actual_col_stability, exits_actual_col_stability]].copy()
            if not pd.api.types.is_datetime64_any_dtype(trend_df_for_stability[date_actual_col_stability]):
                trend_df_for_stability[date_actual_col_stability] = pd.to_datetime(trend_df_for_stability[date_actual_col_stability], errors='coerce')
            trend_df_for_stability.dropna(subset=[date_actual_col_stability], inplace=True)
            trend_df_for_stability.sort_values(by=date_actual_col_stability, inplace=True)
            
            if not trend_df_for_stability.empty:
                agg_trend_stability = trend_df_for_stability.groupby(pd.Grouper(key=date_actual_col_stability, freq='M')).agg(
                    Hires_Total_Agg=(hires_actual_col_stability, 'sum'), 
                    Exits_Total_Agg=(exits_actual_col_stability, 'sum')
                ).reset_index()
                
                # Map uses TEXT_STRING keys for legend (e.g. "hires_label") to the *new aggregated column names*
                map_for_stability_trend = {"hires_label": "Hires_Total_Agg", "exits_label": "Exits_Total_Agg"}
                units_for_stability_trend = {"Hires_Total_Agg": "", "Exits_Total_Agg": ""} # Units defined by axis title
                
                st.plotly_chart(viz.create_trend_chart(
                    agg_trend_stability, date_actual_col_stability, map_for_stability_trend, 
                    "hires_vs_exits_chart_title", st.session_state.selected_lang_code,
                    y_axis_title_key="people_count_label", x_axis_title_key="month_axis_label",
                    show_average_line=True, rolling_avg_window=3, value_col_units_map=units_for_stability_trend
                ), use_container_width=True)
            else: st.warning(_("no_data_hires_exits"))
        else: st.warning(_("no_data_hires_exits"))
        
        # Actionable Insights for Stability Panel
        action_insights_stability = insights.generate_stability_insights(
            df_stability_filtered, 
            avg_rotation_current, 
            agg_trend_stability if 'agg_trend_stability' in locals() and not agg_trend_stability.empty else pd.DataFrame(), # Pass valid DF
            st.session_state.selected_lang_code
        )
        if action_insights_stability:
            st.markdown("---") # Separator for insights section
            st.subheader(_("actionable_insights_title"))
            for insight_item_stability in action_insights_stability:
                st.markdown(f"💡 {insight_item_stability}")
    else: st.info(_("no_data_available"))
    st.markdown("---")


    # --- 2. Safety Pulse Module ---
    st.header(_("safety_pulse_title"))
    total_inc_current_period = float('nan')
    current_dwa_val = float('nan')

    if not df_safety_filtered.empty:
        cols_layout_safety_main = st.columns([2, 1, 1]) # Bar Chart, Metric Card 1, Metric Card 2
        month_col_name_safety = config.COLUMN_MAP.get("month", "month")
        incidents_col_name_safety = config.COLUMN_MAP.get("incidents", "incidents")
        near_misses_col_name_safety = config.COLUMN_MAP.get("near_misses", "near_misses")
        days_no_acc_col_name_safety = config.COLUMN_MAP.get("days_without_accidents", "days_without_accidents")
        active_alerts_col_name_safety = config.COLUMN_MAP.get("active_alerts", "active_alerts")

        with cols_layout_safety_main[0]: # Bar Chart
            if all(c in df_safety_filtered.columns for c in [month_col_name_safety, incidents_col_name_safety, near_misses_col_name_safety]):
                summary_safety_df = df_safety_filtered.groupby(month_col_name_safety, as_index=False).agg(
                    Incidents_Sum_Agg=(incidents_col_name_safety, 'sum'), 
                    Near_Misses_Sum_Agg=(near_misses_col_name_safety, 'sum')
                )
                try: 
                    month_order_cat_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    summary_safety_df[month_col_name_safety] = pd.Categorical(summary_safety_df[month_col_name_safety].astype(str), categories=month_order_cat_list, ordered=True)
                    summary_safety_df.sort_values(month_col_name_safety, inplace=True)
                    summary_safety_df.dropna(subset=[month_col_name_safety], inplace=True) # If month couldn't be categorized
                except Exception: summary_safety_df.sort_values(by=month_col_name_safety, errors='ignore', inplace=True)
                
                if not summary_safety_df.empty:
                    map_safety_bars_viz = {"incidents_label": "Incidents_Sum_Agg", "near_misses_label": "Near_Misses_Sum_Agg"}
                    st.plotly_chart(viz.create_comparison_bar_chart(
                        summary_safety_df, month_col_name_safety, map_safety_bars_viz, "monthly_incidents_chart_title", st.session_state.selected_lang_code,
                        x_axis_title_key="month_axis_label", y_axis_title_key="count_label",
                        barmode='stack', show_total_for_stacked=True, data_label_format_str=".0f"
                    ), use_container_width=True)
                    # Update total_inc_current_period for insights if summary_safety_df is not empty
                    if not summary_safety_df.empty: 
                        total_inc_current_period = summary_safety_df['Incidents_Sum_Agg'].sum() # Sum over the period shown in chart
                else: st.warning(_("no_data_incidents_near_misses"))
            else: st.warning(_("no_data_incidents_near_misses"))
        
        current_dwa_val = df_safety_filtered[days_no_acc_col_name_safety].max() if days_no_acc_col_name_safety in df_safety_filtered.columns else float('nan')
        with cols_layout_safety_main[1]: # Days Without Accidents Metric Card
            previous_dwa_value = get_dummy_prev_val(current_dwa_val, 0.1, variation_abs=10) # Use variation_abs for counts
            viz.display_metric_card(st, "days_without_accidents_metric", current_dwa_val, st.session_state.selected_lang_code, unit=" "+_("days_unit"),
                                   higher_is_better=True, help_text_key="days_no_incident_help",
                                   threshold_good=config.SAFETY_DAYS_NO_INCIDENTS["good"], 
                                   threshold_warning=config.SAFETY_DAYS_NO_INCIDENTS["warning"],
                                   previous_value=previous_dwa_value)
        with cols_layout_safety_main[2]: # Active Safety Alerts Metric Card
            active_alerts_current_val = df_safety_filtered[active_alerts_col_name_safety].sum() if active_alerts_col_name_safety in df_safety_filtered.columns else float('nan')
            previous_active_alerts_val = get_dummy_prev_val(active_alerts_current_val, 0.2, variation_abs=1)
            previous_active_alerts_val = int(previous_active_alerts_val) if pd.notna(previous_active_alerts_val) else None # Ensure integer for count
            viz.display_metric_card(st, "active_safety_alerts_metric", active_alerts_current_val, st.session_state.selected_lang_code, unit="",
                                   higher_is_better=False, target_value=0, # Target is zero active alerts
                                   threshold_good=0, threshold_warning=1, 
                                   previous_value=previous_active_alerts_val)

        # Actionable Insights for Safety Pulse
        action_insights_safety_list = insights.generate_safety_insights(df_safety_filtered, current_dwa_val, total_inc_current_period, st.session_state.selected_lang_code)
        if action_insights_safety_list:
            st.markdown("---")
            st.subheader(_("actionable_insights_title"))
            for insight_item_safety in action_insights_safety_list: st.markdown(f"💡 {insight_item_safety}")
    else: st.info(_("no_data_available"))
    st.markdown("---")

    # --- 3. Employee Engagement & Commitment ---
    st.header(_("engagement_title"))
    avg_enps_for_insight, avg_climate_for_insight, participation_for_insight = float('nan'), float('nan'), float('nan')

    if not df_engagement_filtered.empty:
        cols_layout_engagement_main = st.columns([2,1]) 
        with cols_layout_engagement_main[0]: # Radar Chart
            radar_data_points_engagement = []
            radar_targets_localized_eng_radar = {}
            
            # Iterate through the conceptual keys defined in config.COLUMN_MAP["engagement_radar_dims_cols"]
            for conceptual_key_radar, actual_col_radar in config.COLUMN_MAP["engagement_radar_dims_cols"].items():
                if actual_col_radar in df_engagement_filtered.columns: # Check if the actual column exists in the dataframe
                    avg_score_radar = df_engagement_filtered[actual_col_radar].mean()
                    # Get the display label key (e.g., 'initiative_label') using the conceptual_key_radar
                    label_key_for_display_radar = config.COLUMN_MAP["engagement_radar_dims_labels"].get(conceptual_key_radar, actual_col_radar)
                    display_name_for_radar = _(label_key_for_display_radar, actual_col_radar.replace('_', ' ').title())
                    
                    if pd.notna(avg_score_radar):
                        radar_data_points_engagement.append({"Dimension": display_name_for_radar, "Score": avg_score_radar})
                        radar_targets_localized_eng_radar[display_name_for_radar] = config.ENGAGEMENT_RADAR_DIM_TARGET 
            
            if radar_data_points_engagement:
                df_radar_viz_eng = pd.DataFrame(radar_data_points_engagement)
                st.plotly_chart(viz.create_enhanced_radar_chart(
                    df_radar_viz_eng, "Dimension", "Score", "engagement_dimensions_radar_title", st.session_state.selected_lang_code,
                    range_max_override=config.ENGAGEMENT_RADAR_DIM_SCALE_MAX, 
                    target_values_map=radar_targets_localized_eng_radar, fill_opacity=0.4
                ), use_container_width=True)
            elif any(config.COLUMN_MAP["engagement_radar_dims_cols"].get(k) in df_engagement_filtered.columns for k in config.COLUMN_MAP["engagement_radar_dims_cols"]):
                 st.warning(_("no_data_radar")) # Data columns exist, but filtering resulted in no data for radar points
            else: st.warning(_("no_data_radar_columns")) # Essential columns for radar are missing entirely
        
        with cols_layout_engagement_main[1]: # Metric cards for engagement
            kpis_engagement_config_list = [ # (column_map_key, label_key, unit, higher_is_better, threshold_config_dict, help_text_key)
                ("labor_climate_score", "labor_climate_score_metric", "", True, config.ENGAGEMENT_CLIMATE_SCORE, None),
                ("enps_score", "enps_metric", "", True, config.ENGAGEMENT_ENPS, "enps_metric_help"),
                ("participation_rate", "survey_participation_metric", "%", True, config.ENGAGEMENT_PARTICIPATION, None),
                ("recognitions_count", "recognitions_count_metric", "", True, None, None) # No specific thresholds from config for count
            ]
            
            for col_map_k_eng_card, label_k_eng_card, unit_eng_card, hib_eng_card, thresholds_eng_card_dict, help_k_eng_card in kpis_engagement_config_list:
                actual_col_name_eng_card = config.COLUMN_MAP[col_map_k_eng_card]
                is_count_metric_card_eng = "count" in col_map_k_eng_card 
                
                current_val_metric_eng_card = float('nan')
                if actual_col_name_eng_card in df_engagement_filtered.columns:
                    if is_count_metric_card_eng:
                        current_val_metric_eng_card = df_engagement_filtered[actual_col_name_eng_card].sum()
                    else:
                        current_val_metric_eng_card = df_engagement_filtered[actual_col_name_eng_card].mean()
                
                prev_val_metric_eng_card = get_dummy_prev_val(current_val_metric_eng_card, 0.05, (unit_eng_card=="%"), 
                                                             variation_abs=5 if is_count_metric_card_eng else None) # Adjust variation for counts
                
                thresh_good_eng_card = thresholds_eng_card_dict.get("good") if thresholds_eng_card_dict else None
                thresh_warn_eng_card = thresholds_eng_card_dict.get("warning") if thresholds_eng_card_dict else None
                
                # Format help text if target placeholder is present
                help_text_formatted_card_eng = ""
                if help_k_eng_card and thresh_good_eng_card is not None and "{target}" in _(help_k_eng_card, ""):
                    target_fmt_eng_card = ".0f" if float(thresh_good_eng_card) % 1 == 0 else ".1f"
                    help_text_formatted_card_eng = _(help_k_eng_card).format(target=f"{float(thresh_good_eng_card):{target_fmt_eng_card}}")
                elif help_k_eng_card:
                    help_text_formatted_card_eng = _(help_k_eng_card)

                viz.display_metric_card(st, label_k_eng_card, current_val_metric_eng_card, st.session_state.selected_lang_code, unit=unit_eng_card, 
                                        higher_is_better=hib_eng_card, target_value=thresh_good_eng_card, 
                                        threshold_good=thresh_good_eng_card, threshold_warning=thresh_warn_eng_card,
                                        previous_value=prev_val_metric_eng_card, help_text_key=help_text_formatted_card_eng)
                
                # Store values for insights function
                if col_map_k_eng_card == "enps_score": avg_enps_for_insight = current_val_metric_eng_card
                if col_map_k_eng_card == "labor_climate_score": avg_climate_for_insight = current_val_metric_eng_card
                if col_map_k_eng_card == "participation_rate": participation_for_insight = current_val_metric_eng_card

        # Actionable Insights for Engagement Panel
        action_insights_engagement_list = insights.generate_engagement_insights(avg_enps_for_insight, avg_climate_for_insight, participation_for_insight, st.session_state.selected_lang_code)
        if action_insights_engagement_list:
            st.markdown("---") # Separator
            st.subheader(_("actionable_insights_title"))
            for insight_item_eng in action_insights_engagement_list: st.markdown(f"💡 {insight_item_eng}")
    else: st.info(_("no_data_available"))
    st.markdown("---")

    # --- 4. Operational Stress Dashboard ---
    st.header(_("stress_title"))
    avg_stress_current_val_for_insight = float('nan')
    df_stress_trends_for_insight_func = pd.DataFrame()

    if not df_stress_filtered.empty:
        cols_layout_stress_page = st.columns([1, 2]) 
        
        stress_lvl_actual_col_stress = config.COLUMN_MAP.get("stress_level_survey")
        overtime_actual_col_stress = config.COLUMN_MAP.get("overtime_hours")
        unfilled_actual_col_stress = config.COLUMN_MAP.get("unfilled_shifts")
        date_actual_col_stress_main = config.COLUMN_MAP.get("date")
        workload_actual_col_stress = config.COLUMN_MAP.get("workload_perception")
        psych_actual_col_stress = config.COLUMN_MAP.get("psychological_signals")

        avg_stress_current_val_for_insight = df_stress_filtered[stress_lvl_actual_col_stress].mean() if stress_lvl_actual_col_stress in df_stress_filtered.columns else float('nan')
        
        with cols_layout_stress_page[0]: # Stress Semaforo (KPI Visual)
            st.subheader(_("overall_stress_indicator_title"))
            st.plotly_chart(viz.create_stress_semaforo_visual(
                avg_stress_current_val_for_insight, st.session_state.selected_lang_code, scale_max=config.STRESS_LEVEL_PSYCHOSOCIAL["max_scale"]
            ), use_container_width=True)
            # Dynamic help text for stress indicator
            target_stress_indicator_help = config.STRESS_LEVEL_PSYCHOSOCIAL['low']
            help_text_stress_indicator = _("stress_indicator_help").format(target=f"{target_stress_indicator_help:.1f}")
            st.caption(help_text_stress_indicator)

        with cols_layout_stress_page[1]: # Shift Load Bar Chart
            if all(c in df_stress_filtered.columns for c in [date_actual_col_stress_main, overtime_actual_col_stress, unfilled_actual_col_stress]):
                df_shiftload_trend_stress = df_stress_filtered[[date_actual_col_stress_main, overtime_actual_col_stress, unfilled_actual_col_stress]].copy()
                if not pd.api.types.is_datetime64_any_dtype(df_shiftload_trend_stress[date_actual_col_stress_main]):
                    df_shiftload_trend_stress[date_actual_col_stress_main] = pd.to_datetime(df_shiftload_trend_stress[date_actual_col_stress_main], errors='coerce')
                df_shiftload_trend_stress.dropna(subset=[date_actual_col_stress_main], inplace=True)
                df_shiftload_trend_stress.sort_values(by=date_actual_col_stress_main, inplace=True)
                
                if not df_shiftload_trend_stress.empty:
                    sl_summary_agg_for_chart = df_shiftload_trend_stress.groupby(pd.Grouper(key=date_actual_col_stress_main, freq='M')).agg(
                       Overtime_Agg_Data=(overtime_actual_col_stress, 'sum'), Unfilled_Agg_Data=(unfilled_actual_col_stress, 'sum')).reset_index()
                    # Map TEXT_STRING label keys to the NEW aggregated column names
                    map_sl_stress_bars_to_viz = {"overtime_label": "Overtime_Agg_Data", "unfilled_shifts_label": "Unfilled_Agg_Data"}
                    st.plotly_chart(viz.create_comparison_bar_chart(
                        sl_summary_agg_for_chart, date_actual_col_stress_main, map_sl_stress_bars_to_viz, "monthly_shift_load_chart_title", st.session_state.selected_lang_code,
                        x_axis_title_key="month_axis_label", y_axis_title_key="hours_or_shifts_label",
                        barmode='group', data_label_format_str=".0f"
                    ), use_container_width=True)
                else: st.warning(_("no_data_shift_load"))
            else: st.warning(_("no_data_shift_load"))
        st.markdown("---") # Separator

        # Workload vs Psych Signals Trend (Full Width below the two columns)
        if all(c in df_stress_filtered.columns for c in [date_actual_col_stress_main, workload_actual_col_stress, psych_actual_col_stress]):
            df_wp_ps_trend_stress_chart = df_stress_filtered[[date_actual_col_stress_main, workload_actual_col_stress, psych_actual_col_stress]].copy()
            if not pd.api.types.is_datetime64_any_dtype(df_wp_ps_trend_stress_chart[date_actual_col_stress_main]):
                df_wp_ps_trend_stress_chart[date_actual_col_stress_main] = pd.to_datetime(df_wp_ps_trend_stress_chart[date_actual_col_stress_main], errors='coerce')
            df_wp_ps_trend_stress_chart.dropna(subset=[date_actual_col_stress_main], inplace=True)
            df_wp_ps_trend_stress_chart.sort_values(by=date_actual_col_stress_main, inplace=True)
            
            if not df_wp_ps_trend_stress_chart.empty:
                # These specific aggregated column names MUST match what `insights.generate_stress_insights` expects for trend analysis
                df_stress_trends_for_insight_func = df_wp_ps_trend_stress_chart.groupby(pd.Grouper(key=date_actual_col_stress_main, freq='M')).agg(
                    Workload_Avg=(workload_actual_col_stress, 'mean'), 
                    Psych_Signals_Avg=(psych_actual_col_stress, 'mean')
                ).reset_index()
                
                map_wp_ps_stress_trend_to_viz = {"workload_perception_label": "Workload_Avg", "psychological_signals_label": "Psych_Signals_Avg"}
                unit_map_wp_ps_stress_viz = {"Workload_Avg": "", "Psych_Signals_Avg": ""} # Units are scores, no specific suffix
                st.plotly_chart(viz.create_trend_chart(
                    df_stress_trends_for_insight_func, date_actual_col_stress_main, map_wp_ps_stress_trend_to_viz, 
                    "workload_vs_psych_chart_title", st.session_state.selected_lang_code,
                    y_axis_title_key="average_score_label", x_axis_title_key="month_axis_label",
                    show_average_line=True, rolling_avg_window=3, value_col_units_map=unit_map_wp_ps_stress_viz
                ), use_container_width=True)
            else: st.warning(_("no_data_workload_psych"))
        else: st.warning(_("no_data_workload_psych"))

        # Actionable Insights for Stress panel
        action_insights_stress_list = insights.generate_stress_insights(avg_stress_current_val_for_insight, df_stress_trends_for_insight_func, st.session_state.selected_lang_code)
        if action_insights_stress_list:
            st.markdown("---")
            st.subheader(_("actionable_insights_title"))
            for insight_item_stress in action_insights_stress_list: st.markdown(f"💡 {insight_item_stress}")
    else: st.info(_("no_data_available"))
    st.markdown("---")

    # --- 5. Interactive Plant Map (Placeholder) ---
    st.header(_("plant_map_title"))
    st.markdown(config.PLACEHOLDER_TEXT_PLANT_MAP, unsafe_allow_html=True)
    st.warning(_("This module is a placeholder for future development.", "Module currently in development."))
    st.markdown("---")

    # --- 6. Predictive AI Insights (Placeholder) ---
    st.header(_("ai_insights_title"))
    st.markdown(config.PLACEHOLDER_TEXT_AI_INSIGHTS, unsafe_allow_html=True)
    st.warning(_("This module is a placeholder for future development.", "Module currently in development."))
    st.markdown("---")


# --- GLOSSARY PAGE ---
elif app_mode_selected == glossary_nav_label_loc:
    st.title(_("glossary_page_title"))
    st.markdown(_("glossary_intro"))
    st.markdown("---")

    search_term_for_glossary_input = st.text_input(_("search_term_label"), key="glossary_search_text_field") # Unique key
    
    # GLOSSARY_TERMS imported from glossary_data.py
    # Sort by the English term key for consistent display order
    sorted_glossary_data_from_file = dict(sorted(GLOSSARY_TERMS.items()))

    num_glossary_terms_displayed = 0
    if sorted_glossary_data_from_file:
        for term_key_english, definitions_for_term in sorted_glossary_data_from_file.items():
            display_this_term_in_glossary = True
            if search_term_for_glossary_input: # Apply search filter if text is entered
                search_text_lower = search_term_for_glossary_input.lower()
                match_in_english_key = search_text_lower in term_key_english.lower()
                match_in_english_def = search_text_lower in definitions_for_term.get("EN", "").lower() if definitions_for_term.get("EN") else False
                match_in_spanish_def = search_text_lower in definitions_for_term.get("ES", "").lower() if definitions_for_term.get("ES") else False
                if not (match_in_english_key or match_in_english_def or match_in_spanish_def):
                    display_this_term_in_glossary = False
            
            if display_this_term_in_glossary:
                num_glossary_terms_displayed +=1
                # Display term in English as the expander header (it's the key)
                with st.expander(term_key_english, expanded=(search_term_for_glossary_input != "")): # Expand if search term exists
                    primary_display_lang_key = st.session_state.selected_lang_code.upper() # 'EN' or 'ES'
                    secondary_display_lang_key = "ES" if primary_display_lang_key == "EN" else "EN"

                    # Display definition in the primary selected language
                    if primary_display_lang_key in definitions_for_term and definitions_for_term[primary_display_lang_key]:
                        st.markdown(f"**{_('definition_label')}:**") # Uses current_lang_texts for "Definition"
                        st.markdown(definitions_for_term[primary_display_lang_key])
                    
                    # Optionally display the definition in the other language as a caption
                    if secondary_display_lang_key in definitions_for_term and definitions_for_term[secondary_display_lang_key]:
                         # Add a separator if primary definition was shown
                         if primary_display_lang_key in definitions_for_term and definitions_for_term[primary_display_lang_key]:
                             st.markdown("---")
                         # Get the full name of the secondary language (e.g., "Español")
                         secondary_lang_full_name = _(f"language_name_full_{secondary_display_lang_key.upper()}", secondary_display_lang_key)
                         st.caption(f"*{secondary_lang_full_name}:* {definitions_for_term[secondary_display_lang_key]}")
                    # Fallback if primary language def is missing but English exists
                    elif "EN" in definitions_for_term and definitions_for_term["EN"] and primary_display_lang_key != "EN":
                        st.markdown(f"**{config.TEXT_STRINGS['EN'].get('definition_label', 'Definition')}:**")
                        st.markdown(definitions_for_term["EN"])

        if search_term_for_glossary_input and num_glossary_terms_displayed == 0:
            st.info(_("no_term_found")) # Displayed if search yields no results
    elif not GLOSSARY_TERMS: # Check if GLOSSARY_TERMS dictionary itself is empty
        st.warning(_("glossary_empty_message")) # Message if no terms are defined at all

# --- Optional Modules Display (Always in Sidebar for toggling) ---
# This section should be outside the main 'if app_mode_selected' block so sidebar items are always present.
st.sidebar.markdown("---")
st.sidebar.markdown(f"## {_('optional_modules_header')}")
# Use a unique key for the checkbox here as well
show_optional_modules_in_sidebar_toggle = st.sidebar.checkbox(
    _('show_optional_modules'), 
    key="sidebar_optional_modules_toggle_checkbox", 
    value=False # Default to collapsed/hidden
)
if show_optional_modules_in_sidebar_toggle :
    with st.sidebar.expander(_('optional_modules_title'), expanded=True): # Expander within the sidebar
        # Fallback to DEFAULT_LANG's list if current language doesn't have one (shouldn't happen with full translations)
        optional_list_markdown_content = _('optional_modules_list', 
                                          default_text_override=config.TEXT_STRINGS[config.DEFAULT_LANG].get('optional_modules_list',""))
        st.markdown(optional_list_markdown_content, unsafe_allow_html=True) # Allow markdown for lists

# --- Footer / App Info (Always in Sidebar) ---
st.sidebar.markdown("---")
st.sidebar.caption(f"{_(config.APP_TITLE_KEY)} {config.APP_VERSION}") # Use app_title_key from config
st.sidebar.caption(_("Built with Streamlit, Plotly, and Pandas.", "Constructido con Streamlit, Plotly y Pandas."))
st.sidebar.caption(_("Data Last Updated: (N/A for sample data)", "Última Actualización de Datos: (N/A para datos de muestra)"))
