# visualizations.py
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import config # Your configuration file
from typing import List, Dict, Optional, Any, Union # For type hinting

# --- Helper to get localized text ---
def get_lang_text(lang_code: str, key: str, default_text: Optional[str] = None) -> str:
    effective_lang_code = lang_code if lang_code in config.TEXT_STRINGS else config.DEFAULT_LANG
    text_dict = config.TEXT_STRINGS.get(effective_lang_code, config.TEXT_STRINGS[config.DEFAULT_LANG])
    # If default_text is provided, use it as the ultimate fallback, otherwise use the key.
    return text_dict.get(key, default_text if default_text is not None else key)


# --- Helper for status determination (remains crucial) ---
def get_status_by_thresholds(value: Optional[Union[int, float, np.number]],
                             higher_is_worse: bool,
                             threshold_good: Optional[Union[int, float, np.number]] = None,
                             threshold_warning: Optional[Union[int, float, np.number]] = None) -> Optional[str]:
    if pd.isna(value) or value is None: return None
    val_f = float(value)
    good_f = float(threshold_good) if threshold_good is not None and pd.notna(threshold_good) else None
    warn_f = float(threshold_warning) if threshold_warning is not None and pd.notna(threshold_warning) else None

    if higher_is_worse:
        if good_f is not None and val_f <= good_f: return "good"
        if warn_f is not None and (good_f is None or val_f > good_f) and val_f <= warn_f: return "warning"
        if (warn_f is not None and val_f > warn_f) or \
           (warn_f is None and good_f is not None and val_f > good_f): return "critical"
    else: # Higher is better
        if good_f is not None and val_f >= good_f: return "good"
        if warn_f is not None and (good_f is None or val_f < good_f) and val_f >= warn_f: return "warning"
        if (warn_f is not None and val_f < warn_f) or \
           (warn_f is None and good_f is not None and val_f < good_f): return "critical"
    return None # Default if no clear category matched

def get_semaforo_color(status: Optional[str]) -> str:
    if status == "good": return config.COLOR_STATUS_GOOD
    if status == "warning": return config.COLOR_STATUS_WARNING
    if status == "critical": return config.COLOR_STATUS_CRITICAL
    return config.COLOR_TEXT_SECONDARY


# --- KPI Gauge Visualization (More "Friendly" and Actionable) ---
def create_kpi_gauge(value: Optional[Union[int, float, np.number]], title_key: str, lang: str,
                     unit: str = "%", higher_is_worse: bool = True,
                     threshold_good: Optional[Union[int, float, np.number]] = None,
                     threshold_warning: Optional[Union[int, float, np.number]] = None,
                     target_line_value: Optional[Union[int, float, np.number]] = None,
                     max_value_override: Optional[Union[int, float, np.number]] = None,
                     previous_value: Optional[Union[int, float, np.number]] = None,
                     subtitle_key: Optional[str] = None) -> go.Figure:
    
    title_text = get_lang_text(lang, title_key)
    if subtitle_key:
        subtitle_text = get_lang_text(lang, subtitle_key)
        title_text = f"{title_text}<br><span style='font-size:0.7em;color:{config.COLOR_TEXT_SECONDARY};'>{subtitle_text}</span>"

    current_val_gauge = 0.0 
    delta_ref_val = None

    if pd.notna(value) and isinstance(value, (int, float, np.number)):
        current_val_gauge = float(value)
    if previous_value is not None and pd.notna(previous_value) and isinstance(previous_value, (int, float, np.number)):
        delta_ref_val = float(previous_value)

    # Dynamic max_value determination - ensure visibility of current value and target/thresholds
    if max_value_override is not None and pd.notna(max_value_override):
        max_axis_val = float(max_value_override)
    else:
        val_candidates = [1.0, (abs(current_val_gauge) * 1.25 if pd.notna(current_val_gauge) else 1.0) ]
        for t_val in [threshold_good, threshold_warning, target_line_value]:
            if t_val is not None and pd.notna(t_val): val_candidates.append(float(t_val) * 1.2) # Ensure threshold is visible
        max_axis_val = max(val_candidates) if val_candidates else 100.0
        if max_axis_val <= 0 : max_axis_val = 10.0 # Fallback for non-positive max

    steps_list = []
    num_t_good = float(threshold_good) if threshold_good is not None and pd.notna(threshold_good) else None
    num_t_warn = float(threshold_warning) if threshold_warning is not None and pd.notna(threshold_warning) else None
    
    if num_t_good is not None and num_t_warn is not None: # Ensure logical order of thresholds
        if higher_is_worse and num_t_warn < num_t_good: num_t_warn = num_t_good 
        if not higher_is_worse and num_t_warn > num_t_good: num_t_warn = num_t_good

    last_step_val = 0.0
    if higher_is_worse: 
        if num_t_good is not None:
            steps_list.append({'range': [last_step_val, num_t_good], 'color': config.COLOR_STATUS_GOOD, 'name': get_lang_text(lang, 'good_label')})
            last_step_val = num_t_good
        if num_t_warn is not None and num_t_warn > last_step_val:
            steps_list.append({'range': [last_step_val, num_t_warn], 'color': config.COLOR_STATUS_WARNING, 'name': get_lang_text(lang, 'warning_label')})
            last_step_val = num_t_warn
        steps_list.append({'range': [last_step_val, max_axis_val], 'color': config.COLOR_STATUS_CRITICAL, 'name': get_lang_text(lang, 'critical_label')})
    else: 
        if num_t_warn is not None: 
            steps_list.append({'range': [last_step_val, num_t_warn], 'color': config.COLOR_STATUS_CRITICAL, 'name': get_lang_text(lang, 'critical_label')})
            last_step_val = num_t_warn
        if num_t_good is not None and num_t_good > last_step_val:
            steps_list.append({'range': [last_step_val, num_t_good], 'color': config.COLOR_STATUS_WARNING, 'name': get_lang_text(lang, 'warning_label')})
            last_step_val = num_t_good
        steps_list.append({'range': [last_step_val, max_axis_val], 'color': config.COLOR_STATUS_GOOD, 'name': get_lang_text(lang, 'good_label')})

    if not steps_list: steps_list.append({'range': [0, max_axis_val], 'color': config.COLOR_NEUTRAL_INFO})
    
    num_target_val = float(target_line_value) if target_line_value is not None and pd.notna(target_line_value) else None
    
    # Determine current status for bar color
    gauge_bar_color = config.COLOR_NEUTRAL_INFO # Default bar color
    if pd.notna(current_val_gauge):
        current_status_for_bar = get_status_by_thresholds(current_val_gauge, higher_is_worse, num_t_good, num_t_warn)
        gauge_bar_color = get_semaforo_color(current_status_for_bar)


    fig = go.Figure(go.Indicator(
        mode="gauge+number" + ("+delta" if delta_ref_val is not None and pd.notna(current_val_gauge) else ""),
        value=current_val_gauge if pd.notna(current_val_gauge) else None,
        title={'text': title_text, 'font': {'size': 14, 'color': config.COLOR_TEXT_SECONDARY}}, # Slightly smaller main title
        number={'font': {'size': 28, 'color': gauge_bar_color}, 'suffix': unit if unit and unit != "N/A" and pd.notna(current_val_gauge) else "", 'valueformat': ".1f"},
        delta={'reference': delta_ref_val,
               'increasing': {'color': config.COLOR_STATUS_CRITICAL if higher_is_worse else config.COLOR_STATUS_GOOD},
               'decreasing': {'color': config.COLOR_STATUS_GOOD if higher_is_worse else config.COLOR_STATUS_CRITICAL},
               'font': {'size': 12}}, # Smaller delta font
        gauge={
            'axis': {'range': [0, max_axis_val], 'tickwidth': 1, 'tickcolor': "lightgray", 'nticks': 5, 'tickfont': {'size': 10}},
            'bar': {'color': gauge_bar_color, 'thickness': 0.3}, # Value bar uses status color
            'bgcolor': "rgba(255,255,255,0.7)", 'borderwidth': 1, 'bordercolor': "rgba(200,200,200,0.5)",
            'steps': steps_list,
            'threshold': {
                'line': {'color': config.COLOR_TARGET_LINE, 'width': 3},
                'thickness': 0.9, 'value': num_target_val
            } if num_target_val is not None else {}
        }
    ))
    fig.update_layout(height=190, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor='rgba(0,0,0,0)')
    return fig

# --- Trend Chart Visualization (More "Friendly" and Actionable) ---
def create_trend_chart(df_input: pd.DataFrame, date_col: str,
                       value_cols_map: Dict[str, str], # {localized_label_key_for_legend: actual_col_name_in_df}
                       title_key: str, lang: str,
                       y_axis_title_key: str = "value_axis_label", x_axis_title_key: str = "date_time_axis_label",
                       show_average_line: bool = False,
                       target_value_map: Optional[Dict[str, Union[int, float]]] = None, # {actual_col_name: target_val}
                       rolling_avg_window: Optional[int] = None,
                       value_col_units_map: Optional[Dict[str, str]] = None, # {actual_col_name: unit_string}
                       y_axis_format: Optional[str] = ",.1f") -> go.Figure: # Default format for y-axis ticks
    df = df_input.copy() 
    title_text = get_lang_text(lang, title_key)
    x_title_text = get_lang_text(lang, x_axis_title_key)
    y_title_text = get_lang_text(lang, y_axis_title_key)

    if df.empty or date_col not in df.columns or not value_cols_map:
        return go.Figure().update_layout(title_text=f"{title_text} ({get_lang_text(lang, 'no_data_for_selection')})",
            annotations=[dict(text=get_lang_text(lang, 'no_data_for_selection'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)])

    fig = go.Figure()
    colors = config.COLOR_SCHEME_CATEGORICAL
    plotted_actual_cols = []

    for i, (label_key, actual_col) in enumerate(value_cols_map.items()):
        if actual_col not in df.columns or not pd.api.types.is_numeric_dtype(df[actual_col]):
            continue
        plotted_actual_cols.append(actual_col)
        series_color = colors[i % len(colors)]
        legend_name = get_lang_text(lang, label_key, actual_col.replace('_',' ').title())
        unit = value_col_units_map.get(actual_col, "") if value_col_units_map else ""
        
        hover_format = y_axis_format if y_axis_format else ",.2f" # Default hover format

        fig.add_trace(go.Scatter(x=df[date_col], y=df[actual_col], mode='lines+markers', name=legend_name,
            line=dict(color=series_color, width=2.2), marker=dict(size=6, symbol="circle"),
            hovertemplate=f"<b>{legend_name}</b><br>{get_lang_text(lang, 'date_label', 'Date')}: %{{x|%Y-%m-%d}}<br>{get_lang_text(lang, 'value_label','Value')}: %{{y:{hover_format}}}{unit}<extra></extra>")) # Added more specific hover

    if rolling_avg_window and isinstance(rolling_avg_window, int) and rolling_avg_window > 0:
        for i, actual_col in enumerate(plotted_actual_cols):
            if len(df) >= rolling_avg_window:
                label_key = [k for k,v in value_cols_map.items() if v == actual_col][0]
                base_name = get_lang_text(lang, label_key)
                ma_name = f"{base_name} ({rolling_avg_window}-p MA)"
                unit = value_col_units_map.get(actual_col, "") if value_col_units_map else ""
                # Use a unique column name for rolling average to avoid clashes
                rolling_col_name_temp = f"_{actual_col}_rolling_avg_temp" 
                df[rolling_col_name_temp] = df[actual_col].rolling(window=rolling_avg_window, center=True, min_periods=1).mean()
                fig.add_trace(go.Scatter(x=df[date_col], y=df[rolling_col_name_temp], mode='lines', name=ma_name,
                    line=dict(color=colors[i % len(colors)], width=1.5, dash='longdashdot'), opacity=0.7, # Different dash
                    hovertemplate=f"<b>{ma_name}</b><br>{get_lang_text(lang, 'date_label', 'Date')}: %{{x|%Y-%m-%d}}<br>{get_lang_text(lang, 'value_label','Value')}: %{{y:{hover_format}}}{unit}<extra></extra>"))
    
    for i, actual_col in enumerate(plotted_actual_cols):
        label_key = [k for k,v in value_cols_map.items() if v == actual_col][0]
        series_name = get_lang_text(lang, label_key)
        series_color = colors[i % len(colors)]
        if show_average_line:
            avg = df[actual_col].mean()
            if pd.notna(avg):
                fig.add_hline(y=avg, line_dash="dash", line_color=series_color, opacity=0.5,
                              annotation_text=f"{get_lang_text(lang, 'average_label')} {series_name}: {avg:{y_axis_format if y_axis_format else ',.1f'}}",
                              annotation_position="bottom left", annotation_font_size=9)
        if target_value_map and actual_col in target_value_map and pd.notna(target_value_map[actual_col]):
            target = target_value_map[actual_col]
            fig.add_hline(y=target, line_dash="solid", line_color=config.COLOR_TARGET_LINE, line_width=2, opacity=0.8,
                          annotation_text=f"{get_lang_text(lang, 'target_label')} {series_name}: {target:{y_axis_format if y_axis_format else ',.1f'}}",
                          annotation_position="top right", annotation_font_size=10, annotation_font_color=config.COLOR_TARGET_LINE)

    fig.update_layout(
        title=dict(text=title_text, x=0.5, font_size=18),
        yaxis_title=y_title_text, xaxis_title=x_title_text,
        legend_title_text=get_lang_text(lang, "metrics_legend"), hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=config.COLOR_TEXT_SECONDARY, namelength=-1),
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', type='date', # Light grid for x-axis too
                   rangeslider_visible=True if len(df[date_col].unique()) > 12 else False, # Show only for more than a year of monthly data
                   rangeselector=dict(buttons=list([
                        dict(count=3, label="3M", step="month", stepmode="backward"), dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"), dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all")]),font_size=10, y=1.1, x=0, xanchor='left')),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickformat=(y_axis_format if y_axis_format else None)),
        legend=dict(orientation="h", yanchor="top", y=1.0, xanchor="center", x=0.5, font_size=10,traceorder="normal"), # Legend below title
        margin=dict(l=50, r=30, t=80, b=70) # Adjust for rangeslider if present
    )
    return fig

# --- Comparison Bar Chart Visualization (More "Friendly" and Actionable) ---
def create_comparison_bar_chart(df_input: pd.DataFrame, x_col: str,
                                y_cols_map: Dict[str, str], # {TEXT_STRING_KEY_FOR_LABEL: ACTUAL_COLUMN_NAME_IN_DF}
                                title_key: str, lang: str,
                                y_axis_title_key: str = "count_label", x_axis_title_key: str = "category_axis_label",
                                barmode: str = 'group', show_total_for_stacked: bool = False,
                                data_label_format_str: str = ".0f") -> go.Figure: # Default to integer format
    df = df_input.copy() 
    title_text = get_lang_text(lang, title_key)
    x_title_text = get_lang_text(lang, x_axis_title_key)
    y_title_text = get_lang_text(lang, y_axis_title_key)
    
    # For px.bar, the `y` argument should be a list of column names that exist in the dataframe passed to it.
    # We rename columns in a temporary df_plot for prettier legend names directly handled by px.bar.
    df_plot = df[[x_col]].copy() # Start with x column
    y_cols_for_plotting = [] # Will store the (potentially renamed) y column names for px.bar
    original_cols_for_sum = [] # Store original y col names for accurate sum in stacked mode

    for label_key_for_legend, actual_col_name_in_df in y_cols_map.items():
        if actual_col_name_in_df in df.columns and pd.api.types.is_numeric_dtype(df[actual_col_name_in_df]):
            # Get the localized display name for the legend
            display_name_for_legend = get_lang_text(lang, label_key_for_legend, actual_col_name_in_df.replace('_', ' ').title())
            df_plot[display_name_for_legend] = df[actual_col_name_in_df] # Add column with display name
            y_cols_for_plotting.append(display_name_for_legend)
            original_cols_for_sum.append(actual_col_name_in_df) # Used for sum if display name differs

    if df_plot.empty or x_col not in df_plot.columns or not y_cols_for_plotting:
        return go.Figure().update_layout(
            title_text=f"{title_text} ({get_lang_text(lang, 'no_data_for_selection')})",
            annotations=[dict(text=get_lang_text(lang, 'no_data_for_selection'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)]
        )

    fig = px.bar(df_plot, x=x_col, y=y_cols_for_plotting, # Use the list of (potentially renamed) y columns
                 title=None, barmode=barmode,
                 color_discrete_sequence=config.COLOR_SCHEME_CATEGORICAL
                )

    fmt = data_label_format_str if isinstance(data_label_format_str, str) and data_label_format_str else ".0f"
    texttemplate_str_final = f'%{{y:{fmt}}}'
    hovertemplate_str_final = f'<b>%{{x}}</b><br>%{{fullData.name}}: %{{y:{fmt}}}<extra></extra>' # %{fullData.name} gets legend entry

    fig.update_traces(
        texttemplate=texttemplate_str_final,
        textposition='outside' if barmode != 'stack' else 'inside',
        textfont=dict(size=9, color= config.COLOR_TEXT_SECONDARY if barmode=='stack' else 'black'), # Dimmer for inside labels
        insidetextanchor='middle' if barmode == 'stack' else 'end', # End for outside to prevent overlap
        hovertemplate=hovertemplate_str_final,
        marker_line_width=1.5, marker_line_color='rgba(0,0,0,0.5)' # Subtle border on bars
    )

    if barmode == 'stack' and show_total_for_stacked and original_cols_for_sum:
        # Summing from the original DataFrame `df` using `original_cols_for_sum` for accuracy
        df_temp_for_total = df.copy() # Make a copy to add total column
        df_temp_for_total['_total_val_on_stack_'] = df_temp_for_total[original_cols_for_sum].sum(axis=1, numeric_only=True)
        
        current_annotations = list(fig.layout.annotations or []) # Preserve any existing annotations
        for _, row in df_temp_for_total.iterrows():
            x_val = row[x_col]
            total_y_val = row['_total_val_on_stack_']
            if pd.notna(total_y_val):
                 current_annotations.append(dict(
                    x=x_val, y=total_y_val, text=f"{total_y_val:{fmt}}",
                    font=dict(size=10, color=config.COLOR_TARGET_LINE), # Make total stand out
                    showarrow=False, yanchor='bottom', yshift=3, xanchor='center'
                ))
        fig.update_layout(annotations=current_annotations)

    fig.update_layout(
        title=dict(text=title_text, x=0.5, font_size=16), # Smaller title if lots of charts
        yaxis_title=y_title_text, xaxis_title=x_title_text,
        legend_title_text="", # Often cleaner without "Metrics" legend title if obvious
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=11, namelength=-1),
        xaxis_tickangle=-30 if not df_plot.empty and df_plot[x_col].nunique() > 7 else 0,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font_size=9, title_text=""),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.08)', zeroline=True, zerolinecolor='rgba(0,0,0,0.2)'),
        xaxis=dict(showgrid=False, type='category', zeroline=True, zerolinecolor='rgba(0,0,0,0.2)'),
        bargap=0.2 if barmode == 'group' else 0.1, # Adjust gap between bars
        margin=dict(l=50, r=20, t=60, b=40) # Adjusted margins
    )
    return fig

# --- Metric Card (UX Enhanced - keep as before, it was quite good) ---
def display_metric_card(st_object, label_key: str, value: Optional[Union[int, float, np.number]], lang: str,
                        previous_value: Optional[Union[int, float, np.number]] = None, unit: str = "",
                        higher_is_better: Optional[bool] = None, help_text_key: Optional[str] = None,
                        target_value: Optional[Union[int, float, np.number]] = None,
                        threshold_good: Optional[Union[int, float, np.number]] = None,
                        threshold_warning: Optional[Union[int, float, np.number]] = None):
    label_text_orig = get_lang_text(lang, label_key)
    raw_help_text_template = get_lang_text(lang, help_text_key, "") if help_text_key else ""
    help_text_final_str = raw_help_text_template
    if target_value is not None and pd.notna(target_value) and isinstance(target_value, (int, float, np.number)) and "{target}" in raw_help_text_template:
        target_float = float(target_value)
        target_format_spec = ".0f" if target_float % 1 == 0 and abs(target_float) >=1 else ".1f"
        try: help_text_final_str = raw_help_text_template.format(target=f"{target_float:{target_format_spec}}")
        except (KeyError, ValueError): help_text_final_str = raw_help_text_template 
    val_display_str, delta_text_str, delta_color_str, status_icon_str = "N/A", None, "normal", "❓ "
    if pd.notna(value) and isinstance(value, (int, float, np.number)):
        val_raw_float = float(value)
        if unit == "%": val_display_str = f"{val_raw_float:,.1f}%"
        elif unit == get_lang_text(lang, 'days_unit') or (abs(val_raw_float) >= 1000 and val_raw_float == int(val_raw_float)):
            val_display_str = f"{val_raw_float:,.0f}{(' ' + unit) if unit and unit != '%' else ''}"
        elif abs(val_raw_float) < 1 and val_raw_float != 0: 
            val_display_str = f"{val_raw_float:,.2f}{(' ' + unit) if unit and unit != '%' else ''}"
        else: val_display_str = f"{val_raw_float:,.1f}{(' ' + unit) if unit and unit != '%' else ''}"
        if unit == "%" and not val_display_str.endswith("%"): val_display_str += "%"
        if previous_value is not None and pd.notna(previous_value) and isinstance(previous_value, (int, float, np.number)):
            prev_val_raw_float = float(previous_value)
            delta_absolute = val_raw_float - prev_val_raw_float
            delta_sign = "+" if delta_absolute >= 1e-9 else ("" if abs(delta_absolute) < 1e-9 else "-")
            delta_unit_for_display = unit if unit != '%' else ''
            if unit == "%": delta_abs_formatted = f"{abs(delta_absolute):.1f}%"
            elif abs(delta_absolute) >= 1000 and delta_absolute == int(delta_absolute): delta_abs_formatted = f"{abs(delta_absolute):,.0f}{(' '+delta_unit_for_display) if delta_unit_for_display else ''}"
            else: delta_abs_formatted = f"{abs(delta_absolute):,.1f}{(' '+delta_unit_for_display) if delta_unit_for_display else ''}"
            if abs(prev_val_raw_float) > 1e-9:
                 delta_percent_val = (delta_absolute / abs(prev_val_raw_float)) * 100
                 delta_text_str = f"{delta_sign}{delta_abs_formatted} ({delta_sign}{abs(delta_percent_val):,.0f}%)"
            else: delta_text_str = f"{delta_sign}{delta_abs_formatted} ({get_lang_text(lang,'prev_period_label_short','Prev 0')})"
            if higher_is_better is not None:
                if delta_absolute > 1e-9: delta_color_str = "normal" if higher_is_better else "inverse"
                elif delta_absolute < -1e-9: delta_color_str = "inverse" if higher_is_better else "normal"
                else: delta_color_str = "off"
        current_status_text = get_status_by_thresholds(val_raw_float, higher_is_better, threshold_good, threshold_warning)
        if current_status_text == "good": status_icon_str = "✅ "
        elif current_status_text == "warning": status_icon_str = "⚠️ "
        elif current_status_text == "critical": status_icon_str = "❗ "
        elif target_value is not None and higher_is_better is not None and pd.notna(target_value):
            if (higher_is_better and val_raw_float >= float(target_value)) or \
               (not higher_is_better and val_raw_float <= float(target_value)):
                status_icon_str = "🎯 "
            else: status_icon_str = "" 
        else: status_icon_str = "" 
    st_object.metric(label=status_icon_str + label_text_orig, value=val_display_str, delta=delta_text_str, delta_color=delta_color_str, help=help_text_final_str)

# --- Radar Chart Visualization (UX Enhanced) ---
def create_enhanced_radar_chart(df_radar_input: pd.DataFrame, categories_col: str, values_col: str,
                                 title_key: str, lang: str, group_col: Optional[str] = None,
                                 range_max_override: Optional[Union[int, float]] = None,
                                 target_values_map: Optional[Dict[str, Union[int, float]]] = None, 
                                 fill_opacity: float = 0.3): # Slightly less fill for clarity with target
    title_text = get_lang_text(lang, title_key)
    df_radar = df_radar_input.copy()
    if df_radar.empty or categories_col not in df_radar.columns or values_col not in df_radar.columns or df_radar[categories_col].nunique() == 0:
        return go.Figure().update_layout(title_text=f"{title_text} ({get_lang_text(lang, 'no_data_radar')})",
            annotations=[dict(text=get_lang_text(lang, 'no_data_radar'),showarrow=False, xref="paper", yref="paper", x=0.5,y=0.5)])
    all_categories_ordered_list = df_radar[categories_col].unique()
    all_r_vals = df_radar[values_col].dropna().tolist()
    if target_values_map: all_r_vals.extend([v for v in target_values_map.values() if pd.notna(v) and isinstance(v,(int,float))])
    valid_r_vals = [float(v) for v in all_r_vals if isinstance(v, (int,float)) and pd.notna(v)]
    max_data_val_for_range = max(valid_r_vals) if valid_r_vals else 0.0
    radial_range_max_val = float(range_max_override) if range_max_override is not None and pd.notna(range_max_override) else (max_data_val_for_range * 1.15 if max_data_val_for_range > 0 else config.ENGAGEMENT_RADAR_DIM_SCALE_MAX or 5.0)
    radial_range_max_val = max(radial_range_max_val, 1.0)
    fig = go.Figure()
    colors_list = config.COLOR_SCHEME_CATEGORICAL
    if group_col and group_col in df_radar.columns and df_radar[group_col].nunique() > 0:
        for i, (name_group, group_data_df) in enumerate(df_radar.groupby(group_col)):
            current_group_ordered_df = pd.DataFrame({categories_col: all_categories_ordered_list}).merge(
                group_data_df, on=categories_col, how='left').fillna({values_col: 0})
            fig.add_trace(go.Scatterpolar(
                r=current_group_ordered_df[values_col], theta=current_group_ordered_df[categories_col],
                fill='toself', name=str(name_group), line_color=colors_list[i % len(colors_list)], opacity=fill_opacity,
                hovertemplate='<b>%{theta}</b><br>' + f'{str(name_group)}: %{{r:.1f}}<extra></extra>' ))
    else:
        single_series_ordered_df = pd.DataFrame({categories_col: all_categories_ordered_list}).merge(
            df_radar, on=categories_col, how='left').fillna({values_col: 0})
        fig.add_trace(go.Scatterpolar(
            r=single_series_ordered_df[values_col], theta=single_series_ordered_df[categories_col],
            fill='toself', name=get_lang_text(lang, "average_score_label"), line_color=colors_list[0],
            opacity=fill_opacity + 0.15, hovertemplate='<b>%{theta}</b>: %{r:.1f}<extra></extra>'))
    if target_values_map: 
        target_r_values_ordered = [target_values_map.get(cat, 0) for cat in all_categories_ordered_list]
        fig.add_trace(go.Scatterpolar(
            r=target_r_values_ordered, theta=all_categories_ordered_list, mode='lines', name=get_lang_text(lang, "target_label"),
            line=dict(color=config.COLOR_TARGET_LINE, dash='longdash', width=2), hoverinfo='skip'))
    fig.update_layout(
        title=dict(text=title_text, x=0.5, font_size=16), # Slightly smaller radar title
        polar=dict(bgcolor="rgba(255,255,255,0.0)",
                   radialaxis=dict(visible=True, range=[0, radial_range_max_val], showline=True, linecolor='rgba(0,0,0,0.1)', gridcolor="rgba(0,0,0,0.1)", tickfont_size=9),
                   angularaxis=dict(showline=True, linecolor='rgba(0,0,0,0.1)', gridcolor="rgba(0,0,0,0.05)", tickfont_size=10, direction="clockwise")),
        showlegend=True, legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5, font_size=9, itemsizing='constant'), # Legend below chart
        margin=dict(l=40, r=40, t=70, b=80), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

# --- Stress Semáforo Visual (Bullet Gauge Style) ---
def create_stress_semaforo_visual(stress_level_value: Optional[Union[int, float, np.number]], lang: str,
                                  scale_max: float = config.STRESS_LEVEL_PSYCHOSOCIAL["max_scale"]) -> go.Figure:
    val_for_indicator, color_for_status_viz, text_for_status_viz = None, config.COLOR_TEXT_SECONDARY, get_lang_text(lang, 'status_na_label')

    if pd.notna(stress_level_value) and isinstance(stress_level_value, (int, float, np.number)):
        val_float_stress_viz = float(stress_level_value)
        val_for_indicator = val_float_stress_viz
        status_stress_viz = get_status_by_thresholds(val_float_stress_viz, higher_is_worse=True,
                                           threshold_good=config.STRESS_LEVEL_PSYCHOSOCIAL["low"],
                                           threshold_warning=config.STRESS_LEVEL_PSYCHOSOCIAL["medium"])
        color_for_status_viz = get_semaforo_color(status_stress_viz)
        if status_stress_viz == "good": text_for_status_viz = get_lang_text(lang, 'low_label')
        elif status_stress_viz == "warning": text_for_status_viz = get_lang_text(lang, 'moderate_label')
        elif status_stress_viz == "critical": text_for_status_viz = get_lang_text(lang, 'high_label')
        else: text_for_status_viz = f"{val_float_stress_viz:.1f}" if pd.notna(val_float_stress_viz) else get_lang_text(lang, 'status_na_label')
            
    num_config_stress_viz = {'font': {'size': 20, 'color': color_for_status_viz}, 'valueformat': ".1f"}
    if val_for_indicator is not None: num_config_stress_viz['suffix'] = f" / {scale_max:.0f}"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val_for_indicator, 
        domain={'x': [0.05, 0.95], 'y': [0.0, 0.6]}, # Position indicator lower
        title={'text': f"<b style='color:{color_for_status_viz}; font-size:1.2em;'>{text_for_status_viz.upper()}</b>", 'font': {'size': 14}, 'align': "center"}, # Title for status
        number=num_config_stress_viz,
        gauge={
            'shape': "bullet",
            'axis': {'range': [0, scale_max], 'visible': True, 'showticklabels': True,
                     'tickvals': [0, config.STRESS_LEVEL_PSYCHOSOCIAL["low"], config.STRESS_LEVEL_PSYCHOSOCIAL["medium"], scale_max],
                     'ticktext': ["0", f"{config.STRESS_LEVEL_PSYCHOSOCIAL['low']:.1f}", f"{config.STRESS_LEVEL_PSYCHOSOCIAL['medium']:.1f}", f"{scale_max:.0f}"],
                     'tickfont': {'size':8, 'color': config.COLOR_TEXT_SECONDARY}, 'tickmode': 'array'},
            'steps': [
                {'range': [0, config.STRESS_LEVEL_PSYCHOSOCIAL["low"]], 'color': config.COLOR_STATUS_GOOD},
                {'range': [config.STRESS_LEVEL_PSYCHOSOCIAL["low"], config.STRESS_LEVEL_PSYCHOSOCIAL["medium"]], 'color': config.COLOR_STATUS_WARNING},
                {'range': [config.STRESS_LEVEL_PSYCHOSOCIAL["medium"], scale_max], 'color': config.COLOR_STATUS_CRITICAL}
            ],
            'bar': {'color': color_for_status_viz, 'thickness': 0.6}, 
            'bgcolor': "rgba(240, 240, 240, 0.8)", 'borderwidth': 0.5, 'bordercolor': "rgba(200,200,200,0.6)"
        }))
    # Make it compact - title in Streamlit subheader will give it its overall name "Psychosocial Stress Index"
    fig.update_layout(height=90, margin=dict(t=5, b=10, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)')
    return fig
