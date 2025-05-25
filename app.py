# app.py
import streamlit as st
import pandas as pd
import numpy as np
import visualizations as viz # Custom visualization functions
import config                  # Main config (excluding glossary terms)
import insights                # New module for generating insights
from glossary_data import GLOSSARY_TERMS # Import glossary from its own file
from typing import List, Dict, Optional, Any, Union, Callable # Added Callable

# --- Page Configuration (Applied once at the top) ---
initial_lang_code_for_config = st.session_state.get('selected_lang_code', config.DEFAULT_LANG)
if initial_lang_code_for_config not in config.TEXT_STRINGS:
    initial_lang_code_for_config = config.DEFAULT_LANG

st.set_page_config(
    page_title=config.TEXT_STRINGS[initial_lang_code_for_config].get("app_title", "Vital Signs Dashboard"),
    page_icon=config.APP_ICON,
    layout="wide"
)

# --- Language Selection & Localization Helper (Stubs assuming ui_components.py is not used) ---
def display_language_selector_stub(st_session_state: Any, get_localized_text_func: Callable[[str, Optional[str]], str]) -> str:
    st.sidebar.markdown("---")
    available_langs = list(config.TEXT_STRINGS.keys())
    if 'selected_lang_code' not in st_session_state:
        st_session_state.selected_lang_code = config.DEFAULT_LANG

    def update_lang_callback():
        st_session_state.selected_lang_code = st_session_state._app_lang_selector_key_widget_stub

    selected_lang = st.sidebar.selectbox(
        label=f"{config.TEXT_STRINGS['EN'].get('language_selector', 'Language')} / {config.TEXT_STRINGS['ES'].get('language_selector', 'Idioma')}",
        options=available_langs,
        index=available_langs.index(st_session_state.selected_lang_code),
        format_func=lambda x: config.TEXT_STRINGS[x].get(f"language_name_full_{x.upper()}", x.upper()),
        key="_app_lang_selector_key_widget_stub", # Ensure unique key
        on_change=update_lang_callback
    )
    return st_session_state.selected_lang_code

selected_lang_code = display_language_selector_stub(st.session_state, lambda k,d=None: config.TEXT_STRINGS[st.session_state.selected_lang_code].get(k,d or k) )
current_lang_texts = config.TEXT_STRINGS.get(selected_lang_code, config.TEXT_STRINGS[config.DEFAULT_LANG])

def _(text_key: str, default_text_override: Optional[str] = None) -> str:
    return current_lang_texts.get(text_key, default_text_override if default_text_override is not None else text_key)

# --- App Navigation (Stub assuming ui_components.py is not used) ---
def display_navigation_stub() -> str:
    st.sidebar.markdown("---")
    dashboard_nav_label = _("dashboard_nav_label", "Dashboard")
    glossary_nav_label = _("glossary_nav_label", "Glossary")
    selected_mode = st.sidebar.radio(
        label=_("navigation_label", "Navigation"),
        options=[dashboard_nav_label, glossary_nav_label],
        key="app_navigation_mode_radio_stub" # Ensure unique key
    )
    st.sidebar.markdown("---")
    return selected_mode
app_mode_selected = display_navigation_stub()


# --- DASHBOARD MODE ---
if app_mode_selected == _("dashboard_nav_label", "Dashboard"): # Use localized label
    # --- Data Loading ---
    @st.cache_data
    def load_data_main(file_path_str: str, date_cols_actual_names: Optional[List[str]] = None):
        try:
            df = pd.read_csv(file_path_str, parse_dates=date_cols_actual_names if date_cols_actual_names else False)
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].notna().any():
                    try: df[col] = df[col].astype(str).str.strip()
                    except AttributeError: pass # Handles non-string objects
            return df
        except FileNotFoundError:
            st.error(_("error_loading_data").format(file_path_str) + f". " + _("check_file_path_instruction", "Please verify file path and presence."))
            return pd.DataFrame()
        except Exception as e:
            st.error(_("error_loading_data").format(file_path_str) + f" - {_('exception_detail_prefix','Exception')}: {e}")
            return pd.DataFrame()

    stability_date_cols_parse = [config.COLUMN_MAP.get("date")] if config.COLUMN_MAP.get("date") else None
    df_stability_raw = load_data_main(config.STABILITY_DATA_FILE, date_cols_actual_names=stability_date_cols_parse)
    df_safety_raw = load_data_main(config.SAFETY_DATA_FILE, date_cols_actual_names=None) # Month is text
    df_engagement_raw = load_data_main(config.ENGAGEMENT_DATA_FILE, date_cols_actual_names=None)
    stress_date_cols_parse = [config.COLUMN_MAP.get("date")] if config.COLUMN_MAP.get("date") else None
    df_stress_raw = load_data_main(config.STRESS_DATA_FILE, date_cols_actual_names=stress_date_cols_parse)

    # --- Sidebar Filters (Stub assuming ui_components.py is not used) ---
    def display_sidebar_filters_stub(all_raw_dfs: List[pd.DataFrame]) -> Dict[str, List[str]]:
        st.sidebar.header(_("filters_header"))
        def get_unique_opts(dfs_list: List[pd.DataFrame], col_key: str) -> List[str]:
            actual_col = config.COLUMN_MAP.get(col_key)
            if not actual_col: return []
            opts_set = set()
            for df_item in dfs_list:
                if not df_item.empty and actual_col in df_item.columns:
                    opts_set.update(df_item[actual_col].dropna().astype(str).tolist())
            return sorted(list(opts_set))

        sites_opts = get_unique_opts(all_raw_dfs, "site")
        regions_opts = get_unique_opts(all_raw_dfs, "region")
        depts_opts = get_unique_opts(all_raw_dfs, "department")
        fcs_opts = get_unique_opts(all_raw_dfs, "fc")
        shifts_opts = get_unique_opts(all_raw_dfs, "shift")

        sel_sites = st.sidebar.multiselect(_("select_site"), options=sites_opts, default=config.DEFAULT_SITES, key="stub_filter_sites")
        sel_regions = st.sidebar.multiselect(_("select_region"), options=regions_opts, default=config.DEFAULT_REGIONS, key="stub_filter_regions")
        sel_depts = st.sidebar.multiselect(_("select_department"), options=depts_opts, default=config.DEFAULT_DEPARTMENTS, key="stub_filter_depts")
        sel_fcs = st.sidebar.multiselect(_("select_fc"), options=fcs_opts, default=config.DEFAULT_FUNCTIONAL_CATEGORIES, key="stub_filter_fcs")
        sel_shifts = st.sidebar.multiselect(_("select_shift"), options=shifts_opts, default=config.DEFAULT_SHIFTS, key="stub_filter_shifts")
        
        return {'site': sel_sites, 'region': sel_regions, 'department': sel_depts, 'fc': sel_fcs, 'shift': sel_shifts}

    all_dataframes_list = [df for df in [df_stability_raw, df_safety_raw, df_engagement_raw, df_stress_raw] if not df.empty]
    filter_selections_active_map = display_sidebar_filters_stub(all_dataframes_list)

    def apply_all_filters_to_df(df_to_filter: pd.DataFrame, col_map: Dict[str, str], selections: Dict[str, List[str]]) -> pd.DataFrame:
        if df_to_filter.empty: return df_to_filter.copy()
        df_filtered = df_to_filter.copy()
        for concept_key, selected_opts_list in selections.items():
            actual_col_in_df = col_map.get(concept_key)
            if actual_col_in_df and selected_opts_list and actual_col_in_df in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[actual_col_in_df].astype(str).isin([str(opt) for opt in selected_opts_list])]
        return df_filtered

    df_stability_filtered = apply_all_filters_to_df(df_stability_raw, config.COLUMN_MAP, filter_selections_active_map)
    df_safety_filtered = apply_all_filters_to_df(df_safety_raw, config.COLUMN_MAP, filter_selections_active_map)
    df_engagement_filtered = apply_all_filters_to_df(df_engagement_raw, config.COLUMN_MAP, filter_selections_active_map)
    df_stress_filtered = apply_all_filters_to_df(df_stress_raw, config.COLUMN_MAP, filter_selections_active_map)

    # --- Main Dashboard Area ---
    st.title(_("dashboard_title"))
    st.markdown(_("dashboard_subtitle"))
    st.caption(_("alignment_note"))
    st.markdown("---")
    st.info(_("psych_safety_note"))
    st.markdown("---")

    def get_dummy_prev_val(curr_val: Optional[Union[int, float, np.number]], factor: float = 0.1, is_percent: bool = False, variation_abs: Optional[Union[int, float]] = None) -> Optional[float]:
        if pd.isna(curr_val) or not isinstance(curr_val, (int,float,np.number)): return None
        current_float_val = float(curr_val)
        if variation_abs is not None or (abs(current_float_val) < 10 and not is_percent) :
             abs_var_val = variation_abs if variation_abs is not None else (1.0 if current_float_val >= 0 else -1.0)
             change_amount = abs_var_val * np.random.uniform(-1, 1)
        else: change_amount = current_float_val * factor * np.random.uniform(-0.7, 0.7)
        previous_val_calculated = current_float_val - change_amount
        if is_percent: return round(max(0.0, min(100.0, previous_val_calculated)),1)
        return round(previous_val_calculated, 1) if not pd.isna(previous_val_calculated) else None

    # --- 1. Laboral Stability Panel ---
    st.header(_("stability_panel_title"))
    agg_trend_stability = pd.DataFrame()
    avg_rotation_current = float('nan')
    if not df_stability_filtered.empty:
        cols_metrics_stab = st.columns(4)
        rot_rate_actual_col = config.COLUMN_MAP.get("rotation_rate")
        if rot_rate_actual_col and rot_rate_actual_col in df_stability_filtered.columns: # Check if key and col exist
             avg_rotation_current = df_stability_filtered[rot_rate_actual_col].mean()
        prev_avg_rotation_val = get_dummy_prev_val(avg_rotation_current, 0.05, True)
        with cols_metrics_stab[0]:
            viz.display_metric_card(st, "rotation_rate_gauge", avg_rotation_current, selected_lang_code, unit="%",
                                    higher_is_better=False, target_value=config.STABILITY_ROTATION_RATE["target"],
                                    threshold_good=config.STABILITY_ROTATION_RATE["good"], threshold_warning=config.STABILITY_ROTATION_RATE["warning"],
                                    previous_value=prev_avg_rotation_val, help_text_key="rotation_rate_metric_help")
            if pd.notna(avg_rotation_current):
                st.plotly_chart(viz.create_kpi_gauge(
                    avg_rotation_current, "rotation_rate_gauge", selected_lang_code, unit="%", higher_is_worse=True,
                    threshold_good=config.STABILITY_ROTATION_RATE["good"], threshold_warning=config.STABILITY_ROTATION_RATE["warning"],
                    target_line_value=config.STABILITY_ROTATION_RATE["target"], previous_value=prev_avg_rotation_val
                ), use_container_width=True)
        retention_metric_definitions = [("retention_6m", "retention_6m_metric"),("retention_12m", "retention_12m_metric"),("retention_18m", "retention_18m_metric")]
        for i, (col_map_key_retention, label_key_retention) in enumerate(retention_metric_definitions):
            actual_col_name_retention = config.COLUMN_MAP.get(col_map_key_retention)
            value_retention = float('nan')
            if actual_col_name_retention and actual_col_name_retention in df_stability_filtered.columns:
                value_retention = df_stability_filtered[actual_col_name_retention].mean()
            previous_value_retention = get_dummy_prev_val(value_retention, 0.03, True)
            with cols_metrics_stab[i+1]:
                viz.display_metric_card(st, label_key_retention, value_retention, selected_lang_code, unit="%", higher_is_better=True,
                                        target_value=config.STABILITY_RETENTION["good"], threshold_good=config.STABILITY_RETENTION["good"],
                                        threshold_warning=config.STABILITY_RETENTION["warning"], previous_value=previous_value_retention,
                                        help_text_key="retention_metric_help")
        st.markdown("<br>", unsafe_allow_html=True)
        date_actual_col_stability = config.COLUMN_MAP.get("date")
        hires_actual_col_stability = config.COLUMN_MAP.get("hires")
        exits_actual_col_stability = config.COLUMN_MAP.get("exits")
        if date_actual_col_stability and hires_actual_col_stability and exits_actual_col_stability and \
           all(col_name in df_stability_filtered.columns for col_name in [date_actual_col_stability, hires_actual_col_stability, exits_actual_col_stability]):
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
                map_for_stability_trend = {"hires_label": "Hires_Total_Agg", "exits_label": "Exits_Total_Agg"}
                units_for_stability_trend = {"Hires_Total_Agg": "", "Exits_Total_Agg": ""}
                st.plotly_chart(viz.create_trend_chart(
                    agg_trend_stability, date_actual_col_stability, map_for_stability_trend,
                    "hires_vs_exits_chart_title", selected_lang_code,
                    y_axis_title_key="people_count_label", x_axis_title_key="month_axis_label",
                    show_average_line=True, rolling_avg_window=3, value_col_units_map=units_for_stability_trend
                ), use_container_width=True)
            else: st.warning(_("no_data_hires_exits"))
        else: st.warning(_("no_data_hires_exits"))
        action_insights_stability = insights.generate_stability_insights(
            df_stability_filtered, avg_rotation_current,
            agg_trend_stability if not agg_trend_stability.empty else pd.DataFrame(),
            selected_lang_code
        )
        if action_insights_stability:
            st.markdown("---")
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
        cols_layout_safety_main = st.columns([2, 1, 1])
        month_col_name_safety = config.COLUMN_MAP.get("month")
        incidents_col_name_safety = config.COLUMN_MAP.get("incidents")
        near_misses_col_name_safety = config.COLUMN_MAP.get("near_misses")
        days_no_acc_col_name_safety = config.COLUMN_MAP.get("days_without_accidents")
        active_alerts_col_name_safety = config.COLUMN_MAP.get("active_alerts")
        with cols_layout_safety_main[0]:
            if month_col_name_safety and incidents_col_name_safety and near_misses_col_name_safety and \
               all(c in df_safety_filtered.columns for c in [month_col_name_safety, incidents_col_name_safety, near_misses_col_name_safety]):
                summary_safety_df = df_safety_filtered.groupby(month_col_name_safety, as_index=False).agg(
                    Incidents_Sum_Agg=(incidents_col_name_safety, 'sum'),
                    Near_Misses_Sum_Agg=(near_misses_col_name_safety, 'sum')
                ).reset_index() # Added reset_index here
                try:
                    month_order_cat_list = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                    summary_safety_df[month_col_name_safety] = pd.Categorical(summary_safety_df[month_col_name_safety].astype(str), categories=month_order_cat_list, ordered=True)
                    summary_safety_df.sort_values(month_col_name_safety, inplace=True)
                    summary_safety_df.dropna(subset=[month_col_name_safety], inplace=True)
                except Exception: summary_safety_df.sort_values(by=month_col_name_safety, errors='ignore', inplace=True)
                if not summary_safety_df.empty:
                    map_safety_bars_viz = {"incidents_label": "Incidents_Sum_Agg", "near_misses_label": "Near_Misses_Sum_Agg"}
                    st.plotly_chart(viz.create_comparison_bar_chart(
                        summary_safety_df, month_col_name_safety, map_safety_bars_viz, "monthly_incidents_chart_title", selected_lang_code,
                        x_axis_title_key="month_axis_label", y_axis_title_key="count_label",
                        barmode='stack', show_total_for_stacked=True, data_label_format_str=".0f"
                    ), use_container_width=True)
                    if not summary_safety_df.empty: total_inc_current_period = summary_safety_df['Incidents_Sum_Agg'].sum()
                else: st.warning(_("no_data_incidents_near_misses"))
            else: st.warning(_("no_data_incidents_near_misses"))
        if days_no_acc_col_name_safety and days_no_acc_col_name_safety in df_safety_filtered.columns:
            current_dwa_val = df_safety_filtered[days_no_acc_col_name_safety].max()
        with cols_layout_safety_main[1]:
            previous_dwa_value = get_dummy_prev_val(current_dwa_val, 0.1, variation_abs=10)
            viz.display_metric_card(st, "days_without_accidents_metric", current_dwa_val, selected_lang_code, unit=" "+_("days_unit"),
                                   higher_is_better=True, help_text_key="days_no_incident_help",
                                   threshold_good=config.SAFETY_DAYS_NO_INCIDENTS["good"],
                                   threshold_warning=config.SAFETY_DAYS_NO_INCIDENTS["warning"],
                                   previous_value=previous_dwa_value)
        with cols_layout_safety_main[2]:
            active_alerts_current_val = float('nan')
            if active_alerts_col_name_safety and active_alerts_col_name_safety in df_safety_filtered.columns:
                active_alerts_current_val = df_safety_filtered[active_alerts_col_name_safety].sum()
            previous_active_alerts_val = get_dummy_prev_val(active_alerts_current_val, 0.2, variation_abs=1)
            previous_active_alerts_val = int(previous_active_alerts_val) if pd.notna(previous_active_alerts_val) else None
            viz.display_metric_card(st, "active_safety_alerts_metric", active_alerts_current_val, selected_lang_code, unit="",
                                   higher_is_better=False, target_value=0,
                                   threshold_good=0, threshold_warning=1,
                                   previous_value=previous_active_alerts_val)
        action_insights_safety_list = insights.generate_safety_insights(df_safety_filtered, current_dwa_val, total_inc_current_period, selected_lang_code)
        if action_insights_safety_list:
            st.markdown("---"); st.subheader(_("actionable_insights_title"))
            for insight_item_safety in action_insights_safety_list: st.markdown(f"💡 {insight_item_safety}")
    else: st.info(_("no_data_available"))
    st.markdown("---")

    # --- 3. Employee Engagement & Commitment ---
    st.header(_("engagement_title"))
    avg_enps_for_insight, avg_climate_for_insight, participation_for_insight = float('nan'), float('nan'), float('nan')
    if not df_engagement_filtered.empty:
        cols_layout_engagement_main = st.columns([2,1])
        with cols_layout_engagement_main[0]:
            radar_data_points_engagement = []
            radar_targets_localized_eng_radar = {}
            for conceptual_key_radar, actual_col_radar in config.COLUMN_MAP.get("engagement_radar_dims_cols", {}).items():
                if actual_col_radar and actual_col_radar in df_engagement_filtered.columns:
                    avg_score_radar = df_engagement_filtered[actual_col_radar].mean()
                    label_map = config.COLUMN_MAP.get("engagement_radar_dims_labels", {})
                    label_key_for_display_radar = label_map.get(conceptual_key_radar, actual_col_radar)
                    display_name_for_radar = _(label_key_for_display_radar, actual_col_radar.replace('_', ' ').title())
                    if pd.notna(avg_score_radar):
                        radar_data_points_engagement.append({"Dimension": display_name_for_radar, "Score": avg_score_radar})
                        radar_targets_localized_eng_radar[display_name_for_radar] = config.ENGAGEMENT_RADAR_DIM_TARGET
            if radar_data_points_engagement:
                df_radar_viz_eng = pd.DataFrame(radar_data_points_engagement)
                st.plotly_chart(viz.create_enhanced_radar_chart(
                    df_radar_viz_eng, "Dimension", "Score", "engagement_dimensions_radar_title", selected_lang_code,
                    range_max_override=config.ENGAGEMENT_RADAR_DIM_SCALE_MAX, target_values_map=radar_targets_localized_eng_radar
                ), use_container_width=True)
            elif any(config.COLUMN_MAP.get("engagement_radar_dims_cols", {}).get(k) in df_engagement_filtered.columns for k in config.COLUMN_MAP.get("engagement_radar_dims_cols", {})):
                 st.warning(_("no_data_radar"))
            else: st.warning(_("no_data_radar_columns"))
        with cols_layout_engagement_main[1]:
            kpis_engagement_config_list = [
                ("labor_climate_score", "labor_climate_score_metric", "", True, config.ENGAGEMENT_CLIMATE_SCORE, None),
                ("enps_score", "enps_metric", "", True, config.ENGAGEMENT_ENPS, "enps_metric_help"),
                ("participation_rate", "survey_participation_metric", "%", True, config.ENGAGEMENT_PARTICIPATION, None),
                ("recognitions_count", "recognitions_count_metric", "", True, None, None)]
            for col_map_k_eng_card, label_k_eng_card, unit_eng_card, hib_eng_card, thresholds_eng_card_dict, help_k_eng_card in kpis_engagement_config_list:
                actual_col_name_eng_card = config.COLUMN_MAP.get(col_map_k_eng_card)
                is_count_metric_card_eng = "count" in col_map_k_eng_card
                current_val_metric_eng_card = float('nan')
                if actual_col_name_eng_card and actual_col_name_eng_card in df_engagement_filtered.columns:
                    if is_count_metric_card_eng: current_val_metric_eng_card = df_engagement_filtered[actual_col_name_eng_card].sum()
                    else: current_val_metric_eng_card = df_engagement_filtered[actual_col_name_eng_card].mean()
                prev_val_metric_eng_card = get_dummy_prev_val(current_val_metric_eng_card, 0.05, (unit_eng_card=="%"), variation_abs=5 if is_count_metric_card_eng else None)
                thresh_good_eng_card = thresholds_eng_card_dict.get("good") if thresholds_eng_card_dict else None
                thresh_warn_eng_card = thresholds_eng_card_dict.get("warning") if thresholds_eng_card_dict else None
                viz.display_metric_card(st, label_k_eng_card, current_val_metric_eng_card, selected_lang_code, unit=unit_eng_card,
                                        higher_is_better=hib_eng_card, target_value=thresh_good_eng_card, threshold_good=thresh_good_eng_card,
                                        threshold_warning=thresh_warn_eng_card, previous_value=prev_val_metric_eng_card, help_text_key=help_k_eng_card)
                if col_map_k_eng_card == "enps_score": avg_enps_for_insight = current_val_metric_eng_card
                if col_map_k_eng_card == "labor_climate_score": avg_climate_for_insight = current_val_metric_eng_card
                if col_map_k_eng_card == "participation_rate": participation_for_insight = current_val_metric_eng_card
        action_insights_engagement_list = insights.generate_engagement_insights(avg_enps_for_insight, avg_climate_for_insight, participation_for_insight, selected_lang_code)
        if action_insights_engagement_list:
            st.markdown("---"); st.subheader(_("actionable_insights_title"))
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

        if stress_lvl_actual_col_stress and stress_lvl_actual_col_stress in df_stress_filtered.columns:
            avg_stress_current_val_for_insight = df_stress_filtered[stress_lvl_actual_col_stress].mean()
        with cols_layout_stress_page[0]:
            st.subheader(_("overall_stress_indicator_title"))
            st.plotly_chart(viz.create_stress_semaforo_visual(
                avg_stress_current_val_for_insight, selected_lang_code, scale_max=config.STRESS_LEVEL_PSYCHOSOCIAL["max_scale"]
            ), use_container_width=True)
            target_stress_indicator_help = config.STRESS_LEVEL_PSYCHOSOCIAL['low']
            help_text_stress_caption = _("stress_indicator_help").format(target=f"{target_stress_indicator_help:.1f}") if "{target}" in _("stress_indicator_help") else _("stress_indicator_help")
            st.caption(help_text_stress_caption)
        with cols_layout_stress_page[1]:
            if date_actual_col_stress_main and overtime_actual_col_stress and unfilled_actual_col_stress and \
               all(c in df_stress_filtered.columns for c in [date_actual_col_stress_main, overtime_actual_col_stress, unfilled_actual_col_stress]):
                df_shiftload_trend_stress = df_stress_filtered[[date_actual_col_stress_main, overtime_actual_col_stress, unfilled_actual_col_stress]].copy()
                if not pd.api.types.is_datetime64_any_dtype(df_shiftload_trend_stress[date_actual_col_stress_main]):
                    df_shiftload_trend_stress[date_actual_col_stress_main] = pd.to_datetime(df_shiftload_trend_stress[date_actual_col_stress_main], errors='coerce')
                df_shiftload_trend_stress.dropna(subset=[date_actual_col_stress_main], inplace=True)
                df_shiftload_trend_stress.sort_values(by=date_actual_col_stress_main, inplace=True)
                if not df_shiftload_trend_stress.empty:
                    sl_summary_agg_for_chart = df_shiftload_trend_stress.groupby(pd.Grouper(key=date_actual_col_stress_main, freq='M')).agg(
                       Overtime_Agg_Data=(overtime_actual_col_stress, 'sum'), Unfilled_Agg_Data=(unfilled_actual_col_stress, 'sum')).reset_index()
                    map_sl_stress_bars_to_viz = {"overtime_label": "Overtime_Agg_Data", "unfilled_shifts_label": "Unfilled_Agg_Data"}
                    st.plotly_chart(viz.create_comparison_bar_chart(
                        sl_summary_agg_for_chart, date_actual_col_stress_main, map_sl_stress_bars_to_viz, "monthly_shift_load_chart_title", selected_lang_code,
                        x_axis_title_key="month_axis_label", y_axis_title_key="hours_or_shifts_label", barmode='group', data_label_format_str=".0f"
                    ), use_container_width=True)
                else: st.warning(_("no_data_shift_load"))
            else: st.warning(_("no_data_shift_load"))
        st.markdown("---")
        if date_actual_col_stress_main and workload_actual_col_stress and psych_actual_col_stress and \
           all(c in df_stress_filtered.columns for c in [date_actual_col_stress_main, workload_actual_col_stress, psych_actual_col_stress]):
            df_wp_ps_trend_stress_chart = df_stress_filtered[[date_actual_col_stress_main, workload_actual_col_stress, psych_actual_col_stress]].copy()
            if not pd.api.types.is_datetime64_any_dtype(df_wp_ps_trend_stress_chart[date_actual_col_stress_main]):
                df_wp_ps_trend_stress_chart[date_actual_col_stress_main] = pd.to_datetime(df_wp_ps_trend_stress_chart[date_actual_col_stress_main], errors='coerce')
            df_wp_ps_trend_stress_chart.dropna(subset=[date_actual_col_stress_main], inplace=True)
            df_wp_ps_trend_stress_chart.sort_values(by=date_actual_col_stress_main, inplace=True)
            if not df_wp_ps_trend_stress_chart.empty:
                df_stress_trends_for_insight_func = df_wp_ps_trend_stress_chart.groupby(pd.Grouper(key=date_actual_col_stress_main, freq='M')).agg(
                    Workload_Avg=(workload_actual_col_stress, 'mean'), Psych_Signals_Avg=(psych_actual_col_stress, 'mean')).reset_index()
                map_wp_ps_stress_trend_to_viz = {"workload_perception_label": "Workload_Avg", "psychological_signals_label": "Psych_Signals_Avg"}
                unit_map_wp_ps_stress_viz = {"Workload_Avg": "", "Psych_Signals_Avg": ""}
                st.plotly_chart(viz.create_trend_chart(
                    df_stress_trends_for_insight_func, date_actual_col_stress_main, map_wp_ps_stress_trend_to_viz,
                    "workload_vs_psych_chart_title", selected_lang_code, y_axis_title_key="average_score_label", x_axis_title_key="month_axis_label",
                    show_average_line=True, rolling_avg_window=3, value_col_units_map=unit_map_wp_ps_stress_viz
                ), use_container_width=True)
            else: st.warning(_("no_data_workload_psych"))
        else: st.warning(_("no_data_workload_psych"))
        action_insights_stress_list = insights.generate_stress_insights(avg_stress_current_val_for_insight, df_stress_trends_for_insight_func, selected_lang_code)
        if action_insights_stress_list:
            st.markdown("---"); st.subheader(_("actionable_insights_title"))
            for insight_item_stress in action_insights_stress_list: st.markdown(f"💡 {insight_item_stress}")
    else: st.info(_("no_data_available"))
    st.markdown("---")


    # --- 5. Spatial Dynamics Analysis ---
    st.header(_("plant_map_title"))

    spatial_metric_options_display = {
        "stress_level_survey": _("Stress Level", default_text_override="Stress Level"),
        "incidents": _("Incidents", default_text_override="Incidents")
    }
    spatial_metric_options_keys = { # These should match conceptual keys in COLUMN_MAP
        "stress_level_survey": "stress_level_survey",
        "incidents": "incidents"
    }

    # Make sure default for selectbox is valid, even if language just changed.
    default_selectbox_metric = list(spatial_metric_options_keys.keys())[0] # Default to the first option
    if 'selected_spatial_metric' not in st.session_state:
        st.session_state.selected_spatial_metric = default_selectbox_metric
    # If current session state value isn't in options (e.g., lang change cleared text or something), reset
    if st.session_state.selected_spatial_metric not in spatial_metric_options_keys:
         st.session_state.selected_spatial_metric = default_selectbox_metric

    selected_display_metric_key_for_selectbox = st.selectbox(
        _("Select Metric for Spatial Analysis", default_text_override="Select Metric for Spatial Analysis"),
        options=list(spatial_metric_options_display.keys()),
        format_func=lambda x: spatial_metric_options_display[x],
        index=list(spatial_metric_options_display.keys()).index(st.session_state.selected_spatial_metric), # Ensure index is valid
        key="spatial_metric_selector_app" # Unique key for this selectbox
    )
    st.session_state.selected_spatial_metric = selected_display_metric_key_for_selectbox # Update session state
    
    selected_metric_col_key = spatial_metric_options_keys[st.session_state.selected_spatial_metric]
    metric_to_map_actual_col = config.COLUMN_MAP.get(selected_metric_col_key)
    
    df_for_spatial_analysis = pd.DataFrame()
    colorbar_title_key_for_metric_map = "value_axis_label" # Default

    if metric_to_map_actual_col:
        if selected_metric_col_key == "stress_level_survey":
            if not df_stress_filtered.empty and metric_to_map_actual_col in df_stress_filtered.columns:
                df_for_spatial_analysis = df_stress_filtered
                colorbar_title_key_for_metric_map = "stress_level_label_short"
        elif selected_metric_col_key == "incidents":
            if not df_safety_filtered.empty and metric_to_map_actual_col in df_safety_filtered.columns:
                df_for_spatial_analysis = df_safety_filtered
                colorbar_title_key_for_metric_map = "incident_count_label_short"
    
    if not df_for_spatial_analysis.empty and metric_to_map_actual_col and metric_to_map_actual_col in df_for_spatial_analysis.columns:
        x_coord_col_key = "location_x"; y_coord_col_key = "location_y"
        actual_x_coord_col = config.COLUMN_MAP.get(x_coord_col_key); actual_y_coord_col = config.COLUMN_MAP.get(y_coord_col_key)
        spatial_df_prepared = df_for_spatial_analysis.copy()
        
        has_real_coords = (actual_x_coord_col and actual_x_coord_col in spatial_df_prepared.columns and
                           actual_y_coord_col and actual_y_coord_col in spatial_df_prepared.columns and
                           spatial_df_prepared[[actual_x_coord_col, actual_y_coord_col]].dropna().shape[0] > 0) # Check for some non-NaN coords

        x_col_for_map_final, y_col_for_map_final = None, None

        if has_real_coords:
            x_col_for_map_final, y_col_for_map_final = actual_x_coord_col, actual_y_coord_col
            spatial_df_prepared.dropna(subset=[x_col_for_map_final, y_col_for_map_final, metric_to_map_actual_col], inplace=True)
        else:
            st.caption(_("simulating_coordinates_caption"))
            x_col_for_map_final, y_col_for_map_final = "_sim_x", "_sim_y"
            np.random.seed(42)
            cat_col_for_sim = None
            if config.COLUMN_MAP.get("site") in spatial_df_prepared.columns and spatial_df_prepared[config.COLUMN_MAP.get("site")].nunique() > 1:
                cat_col_for_sim = config.COLUMN_MAP.get("site")
            elif config.COLUMN_MAP.get("department") in spatial_df_prepared.columns and spatial_df_prepared[config.COLUMN_MAP.get("department")].nunique() > 1:
                cat_col_for_sim = config.COLUMN_MAP.get("department")
            
            if cat_col_for_sim and cat_col_for_sim in spatial_df_prepared.columns:
                unique_cats = spatial_df_prepared[cat_col_for_sim].dropna().unique()
                if len(unique_cats) > 0:
                    cat_to_x = {cat: i * 100 + np.random.randint(-20, 20) for i, cat in enumerate(unique_cats)}
                    cat_to_y = {cat: (i % 3) * 70 + np.random.randint(-15, 15) for i, cat in enumerate(unique_cats)}
                    spatial_df_prepared[x_col_for_map_final] = spatial_df_prepared[cat_col_for_sim].map(cat_to_x).fillna(np.random.uniform(0,200)) + np.random.normal(0, 15, size=len(spatial_df_prepared))
                    spatial_df_prepared[y_col_for_map_final] = spatial_df_prepared[cat_col_for_sim].map(cat_to_y).fillna(np.random.uniform(0,100)) + np.random.normal(0, 15, size=len(spatial_df_prepared))
                else:
                    spatial_df_prepared[x_col_for_map_final] = np.random.uniform(0, 200, size=len(spatial_df_prepared))
                    spatial_df_prepared[y_col_for_map_final] = np.random.uniform(0, 100, size=len(spatial_df_prepared))
            else:
                spatial_df_prepared[x_col_for_map_final] = np.random.uniform(0, 200, size=len(spatial_df_prepared))
                spatial_df_prepared[y_col_for_map_final] = np.random.uniform(0, 100, size=len(spatial_df_prepared))
            
            spatial_df_prepared.dropna(subset=[x_col_for_map_final, y_col_for_map_final, metric_to_map_actual_col], inplace=True)

        facility_dimensions_map = None
        if not spatial_df_prepared.empty and x_col_for_map_final in spatial_df_prepared and y_col_for_map_final in spatial_df_prepared and \
           spatial_df_prepared[x_col_for_map_final].notna().any() and spatial_df_prepared[y_col_for_map_final].notna().any():
             facility_dimensions_map = {
                 "x0": spatial_df_prepared[x_col_for_map_final].min() - 5, "y0": spatial_df_prepared[y_col_for_map_final].min() - 5,
                 "x1": spatial_df_prepared[x_col_for_map_final].max() + 5, "y1": spatial_df_prepared[y_col_for_map_final].max() + 5,
             }
        entry_exit_points_to_plot = config.EXAMPLE_ENTRY_EXIT_POINTS

        if not spatial_df_prepared.empty and x_col_for_map_final and y_col_for_map_final :
            st.subheader(_("Worker Density Heatmap", default_text_override="Worker Concentration"))
            density_map_fig = viz.create_worker_density_heatmap(
                df_input=spatial_df_prepared, x_col=x_col_for_map_final, y_col=y_col_for_map_final,
                title_key="", lang=selected_lang_code, # Title handled by st.subheader
                facility_dimensions=facility_dimensions_map, entry_exit_points=entry_exit_points_to_plot )
            st.plotly_chart(density_map_fig, use_container_width=True)
            st.markdown("---")

            if metric_to_map_actual_col in spatial_df_prepared.columns and not spatial_df_prepared[metric_to_map_actual_col].dropna().empty:
                metric_map_dynamic_title = f"{_('Metric Density', default_text_override='Metric Density')}: {spatial_metric_options_display[selected_display_metric]}"
                st.subheader(metric_map_dynamic_title)
                metric_heatmap_fig = viz.create_metric_density_heatmap(
                    df_input=spatial_df_prepared, x_col=x_col_for_map_final, y_col=y_col_for_map_final, z_col=metric_to_map_actual_col,
                    title_key="", lang=selected_lang_code, # Title handled by st.subheader
                    aggregation_func="avg" if selected_spatial_metric_key == "stress_level_survey" else "sum",
                    colorscale="Reds" if selected_spatial_metric_key == "stress_level_survey" else "YlOrRd", # More distinct for incidents
                    colorbar_title_key=colorbar_title_key_for_metric_map,
                    show_points=config.HEATMAP_SHOW_POINTS_OVERLAY,
                    facility_dimensions=facility_dimensions_map, entry_exit_points=entry_exit_points_to_plot )
                st.plotly_chart(metric_heatmap_fig, use_container_width=True)
            else: st.warning(_("heatmap_no_value_data") + f" (Metric: {selected_spatial_metric_key})")
        else: st.warning(_("heatmap_no_coordinate_data"))
    else:
        st.info(_("no_data_available") + f" {_('for_metric_key', 'for metric')}: {spatial_metric_options_display.get(selected_spatial_metric_key, selected_spatial_metric_key)}") # Make new key "for_metric_key"

    st.markdown("---")


    # --- 6. Predictive AI Insights (Placeholder) ---
    st.header(_("ai_insights_title"))
    st.markdown(config.PLACEHOLDER_TEXT_AI_INSIGHTS, unsafe_allow_html=True)
    module_name_ai = _("ai_insights_title").split(". ",1)[-1] if ". " in _("ai_insights_title") else _("ai_insights_title")
    st.warning(_("module_in_development_warning").format(module_name=module_name_ai) )
    st.markdown("---")


# --- GLOSSARY PAGE ---
elif app_mode_selected == _("glossary_nav_label", "Glossary"):
    # ... (Glossary logic remains the same) ...
    st.title(_("glossary_page_title")); st.markdown(_("glossary_intro")); st.markdown("---")
    search_term_for_glossary_input = st.text_input(_("search_term_label"), key="glossary_search_text_field")
    sorted_glossary_data_from_file = dict(sorted(GLOSSARY_TERMS.items())); num_glossary_terms_displayed = 0
    if sorted_glossary_data_from_file:
        for term_key_english, definitions_for_term in sorted_glossary_data_from_file.items():
            display_this_term_in_glossary = True
            if search_term_for_glossary_input:
                search_text_lower = search_term_for_glossary_input.lower()
                match_in_english_key = search_text_lower in term_key_english.lower()
                match_in_english_def = search_text_lower in definitions_for_term.get("EN", "").lower() if definitions_for_term.get("EN") else False
                match_in_spanish_def = search_text_lower in definitions_for_term.get("ES", "").lower() if definitions_for_term.get("ES") else False
                if not (match_in_english_key or match_in_english_def or match_in_spanish_def): display_this_term_in_glossary = False
            if display_this_term_in_glossary:
                num_glossary_terms_displayed +=1
                with st.expander(term_key_english, expanded=(search_term_for_glossary_input != "")):
                    primary_display_lang_key = selected_lang_code.upper(); secondary_display_lang_key = "ES" if primary_display_lang_key == "EN" else "EN"
                    if primary_display_lang_key in definitions_for_term and definitions_for_term[primary_display_lang_key]:
                        st.markdown(f"**{_('definition_label')}:**"); st.markdown(definitions_for_term[primary_display_lang_key])
                    if secondary_display_lang_key in definitions_for_term and definitions_for_term[secondary_display_lang_key]:
                         if primary_display_lang_key in definitions_for_term and definitions_for_term[primary_display_lang_key]: st.markdown("---")
                         secondary_lang_full_name = _(f"language_name_full_{secondary_display_lang_key.upper()}", secondary_display_lang_key)
                         st.caption(f"*{secondary_lang_full_name}:* {definitions_for_term[secondary_display_lang_key]}")
                    elif "EN" in definitions_for_term and definitions_for_term["EN"] and primary_display_lang_key != "EN":
                        st.markdown(f"**{config.TEXT_STRINGS['EN'].get('definition_label', 'Definition')}:**"); st.markdown(definitions_for_term["EN"])
        if search_term_for_glossary_input and num_glossary_terms_displayed == 0: st.info(_("no_term_found"))
    elif not GLOSSARY_TERMS: st.warning(_("glossary_empty_message"))

# --- Optional Modules & Footer Stubs ---
def display_optional_modules_toggle_stub():
    st.sidebar.markdown("---"); st.sidebar.markdown(f"## {_('optional_modules_header')}")
    show_optional = st.sidebar.checkbox( _('show_optional_modules'), key="sidebar_optional_modules_toggle_checkbox_stub", value=False )
    if show_optional :
        with st.sidebar.expander(_('optional_modules_title'), expanded=True):
            optional_list_markdown_content = _('optional_modules_list', default_text_override=config.TEXT_STRINGS[config.DEFAULT_LANG].get('optional_modules_list',""))
            st.markdown(optional_list_markdown_content, unsafe_allow_html=True)
def display_footer_stub():
    st.sidebar.markdown("---"); st.sidebar.caption(f"{_(config.APP_TITLE_KEY)} {config.APP_VERSION}")
    st.sidebar.caption(_("Built with Streamlit, Plotly, and Pandas.", "Constructido con Streamlit, Plotly y Pandas."))
    st.sidebar.caption(_("Data Last Updated: (N/A for sample data)", "Última Actualización de Datos: (N/A para datos de muestra)"))

display_optional_modules_toggle_stub()
display_footer_stub()
