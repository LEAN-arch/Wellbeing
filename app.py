import streamlit as st
import pandas as pd
import visualizations as viz # Your custom visualization functions
import config # Your configuration file

# --- Page Configuration (Applied once at the top) ---
initial_lang_code = config.LANG
if 'selected_lang_code' in st.session_state: # If language has been selected already
    initial_lang_code = st.session_state.selected_lang_code

st.set_page_config(
    page_title=config.TEXT_STRINGS[initial_lang_code].get("dashboard_title", config.APP_TITLE),
    page_icon=config.APP_ICON,
    layout="wide"
)

# --- Language Selection ---
st.sidebar.markdown("---") # Separator
available_langs = list(config.TEXT_STRINGS.keys())
if 'selected_lang_code' not in st.session_state:
    st.session_state.selected_lang_code = config.LANG # Default from config

def update_lang():
    st.session_state.selected_lang_code = st.session_state._lang_selector # _ to avoid direct manipulation by widget

selected_lang_code = st.sidebar.selectbox(
    label=config.TEXT_STRINGS["EN"].get("language_selector", "Language / Idioma:") + " / " + config.TEXT_STRINGS["ES"].get("language_selector", "Idioma / Language:"),
    options=available_langs,
    index=available_langs.index(st.session_state.selected_lang_code),
    format_func=lambda x: "English" if x == "EN" else "Español" if x == "ES" else x,
    key="_lang_selector", # Use a private key for session state tracking
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
        st.error(f"Error: File '{file_path}' not found. Please ensure it's in the same directory as the application (app.py).")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

df_stability_raw = load_data(config.STABILITY_DATA_FILE, date_cols=[config.COLUMN_DATE])
df_safety_raw = load_data(config.SAFETY_DATA_FILE)
df_engagement_raw = load_data(config.ENGAGEMENT_DATA_FILE)
df_stress_raw = load_data(config.STRESS_DATA_FILE, date_cols=[config.COLUMN_DATE])


# --- Sidebar Filters ---
st.sidebar.header(current_lang_texts.get("filters_header", "Filters"))

def get_unique_options(df, column_name):
    if not df.empty and column_name in df.columns:
        return sorted(df[column_name].dropna().astype(str).unique())
    return []

# Combine dataframes to get comprehensive filter options if necessary, or use a primary one
# For simplicity, using df_stability_raw assuming it contains all dimensional values
# or consider a separate dimensions table/file in a real scenario.
# Here, we'll use a combined approach for robust filter options.
all_dfs = [df for df in [df_stability_raw, df_safety_raw, df_engagement_raw, df_stress_raw] if not df.empty]
if all_dfs:
    combined_df_for_filters = pd.concat(all_dfs, ignore_index=True, sort=False)
else:
    combined_df_for_filters = pd.DataFrame() # Handle case where all dataframes are empty


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
        actual_col_name = getattr(config, col_config_key, None)
        if actual_col_name and selected_values and actual_col_name in filtered_df.columns:
            if filtered_df[actual_col_name].dtype == 'object' or pd.api.types.is_string_dtype(filtered_df[actual_col_name]):
                 selected_values_str = [str(v) for v in selected_values]
                 filtered_df = filtered_df[filtered_df[actual_col_name].astype(str).isin(selected_values_str)]
            else:
                 filtered_df = filtered_df[filtered_df[actual_col_name].isin(selected_values)]
    return filtered_df

filter_selections_map = {
    'COLUMN_SITE': selected_sites,
    'COLUMN_REGION': selected_regions,
    'COLUMN_DEPARTMENT': selected_departments,
    'COLUMN_FC': selected_fcs,
    'COLUMN_SHIFT': selected_shifts
}

df_stability = filter_dataframe_by_selections(df_stability_raw, filter_selections_map)
df_safety = filter_dataframe_by_selections(df_safety_raw, filter_selections_map)
df_engagement = filter_dataframe_by_selections(df_engagement_raw, filter_selections_map)
df_stress = filter_dataframe_by_selections(df_stress_raw, filter_selections_map)

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
    metric_cols = st.columns(4)

    avg_rotation = df_stability[config.COLUMN_ROTATION_RATE].mean() if config.COLUMN_ROTATION_RATE in df_stability.columns else float('nan')
    with metric_cols[0]:
        st.plotly_chart(viz.create_kpi_gauge(
            value=avg_rotation,
            title_key="rotation_rate_gauge",
            lang=st.session_state.selected_lang_code,
            unit="%",
            higher_is_worse=True,
            threshold_good=config.ROTATION_RATE_THRESHOLD_GOOD,
            threshold_warning=config.ROTATION_RATE_THRESHOLD_WARNING,
            # threshold_critical implicitly defined by warning
            target_line_value=config.ROTATION_RATE_TARGET
        ), use_container_width=True)
        rotation_caption = current_lang_texts.get("rotation_gauge_caption").format(
            good=config.ROTATION_RATE_THRESHOLD_GOOD,
            warn=config.ROTATION_RATE_THRESHOLD_WARNING,
            target=config.ROTATION_RATE_TARGET
        )
        st.caption(rotation_caption)

    with metric_cols[1]:
        ret_6m = df_stability[config.COLUMN_RETENTION_6M].mean() if config.COLUMN_RETENTION_6M in df_stability.columns else float('nan')
        viz.display_metric_card(st, "retention_6m", ret_6m, lang=st.session_state.selected_lang_code, unit="%", higher_is_better=True, target_value=config.RETENTION_THRESHOLD_GOOD, lower_threshold=config.RETENTION_THRESHOLD_WARNING)

    with metric_cols[2]:
        ret_12m = df_stability[config.COLUMN_RETENTION_12M].mean() if config.COLUMN_RETENTION_12M in df_stability.columns else float('nan')
        viz.display_metric_card(st, "retention_12m", ret_12m, lang=st.session_state.selected_lang_code, unit="%", higher_is_better=True, target_value=config.RETENTION_THRESHOLD_GOOD, lower_threshold=config.RETENTION_THRESHOLD_WARNING)

    with metric_cols[3]:
        ret_18m = df_stability[config.COLUMN_RETENTION_18M].mean() if config.COLUMN_RETENTION_18M in df_stability.columns else float('nan')
        viz.display_metric_card(st, "retention_18m", ret_18m, lang=st.session_state.selected_lang_code, unit="%", higher_is_better=True, target_value=config.RETENTION_THRESHOLD_GOOD, lower_threshold=config.RETENTION_THRESHOLD_WARNING)

    st.markdown("---") # Separator before the trend chart

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

    with col1:
        if config.COLUMN_MONTH in df_safety.columns and \
           config.COLUMN_INCIDENTS in df_safety.columns and \
           (config.COLUMN_NEAR_MISSES in df_safety.columns): # Near misses are optional

            y_cols_safety = [config.COLUMN_INCIDENTS]
            if config.COLUMN_NEAR_MISSES in df_safety.columns:
                y_cols_safety.append(config.COLUMN_NEAR_MISSES)

            safety_summary = df_safety.groupby(config.COLUMN_MONTH, as_index=False).agg(
                {col: 'sum' for col in y_cols_safety} # Dynamically sum available columns
            )
            try:
                month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                temp_month_dt = pd.to_datetime(safety_summary[config.COLUMN_MONTH].astype(str), format='%b', errors='coerce')
                if not temp_month_dt.isnull().all(): # if any converted successfully
                    safety_summary['month_dt_sorter'] = temp_month_dt
                    safety_summary = safety_summary.sort_values('month_dt_sorter').drop(columns=['month_dt_sorter'])
                else: # Fallback if '%b' doesn't work, try full month name or sort as is
                    try:
                        safety_summary[config.COLUMN_MONTH] = pd.Categorical(safety_summary[config.COLUMN_MONTH], categories=month_order, ordered=True)
                        safety_summary = safety_summary.sort_values(config.COLUMN_MONTH)
                    except:
                        safety_summary = safety_summary.sort_values(by=config.COLUMN_MONTH, errors='ignore')
            except Exception:
                safety_summary = safety_summary.sort_values(by=config.COLUMN_MONTH, errors='ignore')
            
            st.plotly_chart(viz.create_comparison_bar_chart(
                safety_summary, config.COLUMN_MONTH, y_cols_safety,
                "monthly_incidents_chart", lang=st.session_state.selected_lang_code,
                x_axis_title_key="month_axis", y_axis_title_key="count_axis", barmode='stack'
            ), use_container_width=True)
        else:
            st.warning(current_lang_texts.get("no_data_incidents_near_misses"))

    with col2:
        days_no_accidents = df_safety[config.COLUMN_DAYS_WITHOUT_ACCIDENTS].max() if config.COLUMN_DAYS_WITHOUT_ACCIDENTS in df_safety.columns else "N/A"
        viz.display_metric_card(st, "days_without_accidents_metric", days_no_accidents, lang=st.session_state.selected_lang_code, unit=" " + current_lang_texts.get("days_label", "days"), higher_is_better=True)

    with col3:
        active_alerts_count = df_safety[config.COLUMN_ACTIVE_ALERTS].sum() if config.COLUMN_ACTIVE_ALERTS in df_safety.columns else "N/A"
        viz.display_metric_card(st, "active_safety_alerts_metric", active_alerts_count, lang=st.session_state.selected_lang_code, higher_is_better=False, target_value=0) # Target is 0 alerts
else:
    st.info(current_lang_texts.get("no_data_available"))
st.markdown("---")


# 3. Employee Engagement & Commitment
st.header(current_lang_texts.get("engagement_title"))
if not df_engagement.empty:
    col1, col2 = st.columns([2,1]) # Radar larger

    with col1:
        radar_data_list = []
        radar_categories_present = [] # Keep track of categories with data
        # Example radar target values (on a 1-5 scale)
        radar_target_values = {"Initiative": 4.0, "Punctuality": 4.5, "Recognition": 3.8, "Feedback Culture": 4.2}
        # Translate target keys if needed
        localized_radar_target_values = {
            current_lang_texts.get(config.ENGAGEMENT_RADAR_LABELS_KEYS[k], k.title()): v 
            for k,v in radar_target_values.items() if k in config.ENGAGEMENT_RADAR_LABELS_KEYS
        }


        for internal_key, data_col_name in config.ENGAGEMENT_RADAR_DATA_COLS.items():
            if data_col_name in df_engagement.columns:
                avg_val = df_engagement[data_col_name].mean()
                label_key = config.ENGAGEMENT_RADAR_LABELS_KEYS.get(internal_key, data_col_name)
                display_label = current_lang_texts.get(label_key, data_col_name.replace('_data','').replace('_',' ').title())
                
                if pd.notna(avg_val):
                    radar_data_list.append({
                        "Dimension": display_label,
                        "Score": avg_val
                    })
                    radar_categories_present.append(display_label)
        
        if radar_data_list:
            df_radar = pd.DataFrame(radar_data_list)
            # Filter target values for only present categories to avoid plotting issues
            filtered_radar_targets = {k:v for k,v in localized_radar_target_values.items() if k in radar_categories_present}

            st.plotly_chart(viz.create_enhanced_radar_chart(
                df_radar, "Dimension", "Score",
                "engagement_dimensions_radar", lang=st.session_state.selected_lang_code,
                range_max_override=5, # Assuming a 1-5 scale
                target_values=filtered_radar_targets
            ), use_container_width=True)
        else:
            st.warning(current_lang_texts.get("no_data_radar_columns"))

    with col2:
        climate_score = df_engagement[config.COLUMN_LABOR_CLIMATE].mean() if config.COLUMN_LABOR_CLIMATE in df_engagement.columns else float('nan')
        viz.display_metric_card(st, "labor_climate_score_metric", climate_score, lang=st.session_state.selected_lang_code, unit="", higher_is_better=True, target_value=config.CLIMATE_SCORE_THRESHOLD_GOOD, lower_threshold=config.CLIMATE_SCORE_THRESHOLD_WARNING)

        nps = df_engagement[config.COLUMN_ENPS].mean() if config.COLUMN_ENPS in df_engagement.columns else float('nan')
        viz.display_metric_card(st, "enps_metric", nps, lang=st.session_state.selected_lang_code, unit="", higher_is_better=True, target_value=config.ENPS_THRESHOLD_GOOD, lower_threshold=config.ENPS_THRESHOLD_WARNING)

        participation = df_engagement[config.COLUMN_PARTICIPATION].mean() if config.COLUMN_PARTICIPATION in df_engagement.columns else float('nan')
        viz.display_metric_card(st, "survey_participation_metric", participation, lang=st.session_state.selected_lang_code, unit="%", higher_is_better=True, target_value=config.PARTICIPATION_THRESHOLD_GOOD)
        
        recognitions = df_engagement[config.COLUMN_RECOGNITIONS_COUNT].sum() if config.COLUMN_RECOGNITIONS_COUNT in df_engagement.columns else float('nan')
        viz.display_metric_card(st, "recognitions_count_metric", recognitions, lang=st.session_state.selected_lang_code, unit="", higher_is_better=True)
else:
    st.info(current_lang_texts.get("no_data_available"))
st.markdown("---")

# 4. Operational Stress Dashboard
st.header(current_lang_texts.get("stress_title"))
if not df_stress.empty:
    col1, col2 = st.columns([2,1]) # Bar chart | Semaforo + Trend

    with col1:
        st.subheader(current_lang_texts.get("overall_stress_indicator_title"))
        avg_stress_level = df_stress[config.COLUMN_STRESS_LEVEL_SURVEY].mean() if config.COLUMN_STRESS_LEVEL_SURVEY in df_stress.columns else float('nan')
        st.plotly_chart(viz.create_stress_semaforo_visual(avg_stress_level, lang=st.session_state.selected_lang_code), use_container_width=True)
        stress_caption = current_lang_texts.get("stress_semaforo_caption").format(
                max_scale=config.STRESS_LEVEL_MAX_SCALE,
                low=config.STRESS_LEVEL_THRESHOLD_LOW,
                medium=config.STRESS_LEVEL_THRESHOLD_MEDIUM
        )
        st.caption(stress_caption)

    with col2:
        if config.COLUMN_DATE in df_stress.columns and \
           config.COLUMN_OVERTIME_HOURS in df_stress.columns and \
           config.COLUMN_UNFILLED_SHIFTS in df_stress.columns:
            
            stress_trend_df = df_stress.copy()
            if not pd.api.types.is_datetime64_any_dtype(stress_trend_df[config.COLUMN_DATE]):
                stress_trend_df[config.COLUMN_DATE] = pd.to_datetime(stress_trend_df[config.COLUMN_DATE], errors='coerce')
            stress_trend_df = stress_trend_df.dropna(subset=[config.COLUMN_DATE])
            
            if not stress_trend_df.empty:
                stress_summary_monthly = stress_trend_df.groupby(pd.Grouper(key=config.COLUMN_DATE, freq='M')).agg(
                    Overtime=(config.COLUMN_OVERTIME_HOURS, 'sum'),
                    Unfilled_Shifts=(config.COLUMN_UNFILLED_SHIFTS, 'sum')
                ).reset_index()
                # Localize y-column names for the chart labels dictionary
                y_col_labels_stress = {
                    'Overtime': current_lang_texts.get('overtime_label', 'Overtime'),
                    'Unfilled_Shifts': current_lang_texts.get('unfilled_shifts_label', 'Unfilled Shifts')
                }
                
                fig_shift_load = px.bar(stress_summary_monthly, x=config.COLUMN_DATE, y=['Overtime', 'Unfilled_Shifts'],
                                        title=current_lang_texts.get("monthly_shift_load_chart"), barmode='group',
                                        color_discrete_sequence=config.COLOR_SCHEME_CATEGORICAL, labels=y_col_labels_stress,
                                        text_auto=True)
                fig_shift_load.update_traces(texttemplate='%{y:.0f}', textposition='outside')
                fig_shift_load.update_layout(
                    yaxis_title=current_lang_texts.get("hours_or_shifts_label"),
                    xaxis_title=current_lang_texts.get("month_axis"),
                    legend_title_text=current_lang_texts.get("metrics_legend"),
                    hovermode="x unified",
                    xaxis_tickangle=-30,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.5)'),
                    xaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig_shift_load, use_container_width=True)
            else:
                st.warning(current_lang_texts.get("no_data_shift_load"))
        else:
            st.warning(current_lang_texts.get("no_data_shift_load"))

    # Workload trends vs. psychological signals (Full width below the two columns)
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
                Psychological_Signals=(config.COLUMN_PSYCH_SIGNAL_SCORE, 'mean')
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
st.sidebar.caption(f"{current_lang_texts.get('dashboard_title')} v0.3.1 (MVP - Viz Refined)")
st.sidebar.caption("Built with Streamlit, Plotly, and Pandas.")
st.sidebar.markdown("---")
st.sidebar.caption(f"{current_lang_texts.get('dashboard_title')} v0.3.0 (MVP - Viz Enhanced)")
st.sidebar.caption("Built with Streamlit, Plotly, and Pandas.")
