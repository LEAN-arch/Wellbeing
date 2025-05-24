# visualizations.py
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import config # Your configuration file
from typing import List, Dict, Optional, Any, Union # For type hinting

# --- Helper to get localized text ---
def get_lang_text(lang_code: str, key: str, default_text: Optional[str] = None) -> str:
    """Retrieves localized text safely. Falls back to the key itself if not found and no default_text."""
    effective_lang_code = lang_code if lang_code in config.TEXT_STRINGS else config.DEFAULT_LANG
    text_dict = config.TEXT_STRINGS.get(effective_lang_code, config.TEXT_STRINGS[config.DEFAULT_LANG])
    return text_dict.get(key, default_text if default_text is not None else key)


# --- Helper for getting correct status text based on thresholds ---
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
    return None

def get_semaforo_color(status: Optional[str]) -> str:
    if status == "good": return config.COLOR_STATUS_GOOD
    if status == "warning": return config.COLOR_STATUS_WARNING
    if status == "critical": return config.COLOR_STATUS_CRITICAL
    return config.COLOR_TEXT_SECONDARY


# --- KPI Gauge Visualization ---
def create_kpi_gauge(value: Optional[Union[int, float, np.number]], title_key: str, lang: str,
                     unit: str = "%", higher_is_worse: bool = True,
                     threshold_good: Optional[Union[int, float, np.number]] = None,
                     threshold_warning: Optional[Union[int, float, np.number]] = None,
                     target_line_value: Optional[Union[int, float, np.number]] = None,
                     max_value_override: Optional[Union[int, float, np.number]] = None,
                     previous_value: Optional[Union[int, float, np.number]] = None) -> go.Figure:
    title_text = get_lang_text(lang, title_key)
    current_val_gauge = 0.0 
    delta_ref_val = None

    if pd.notna(value) and isinstance(value, (int, float, np.number)):
        current_val_gauge = float(value)
    if previous_value is not None and pd.notna(previous_value) and isinstance(previous_value, (int, float, np.number)):
        delta_ref_val = float(previous_value)

    if max_value_override is not None and pd.notna(max_value_override):
        max_axis_val = float(max_value_override)
    else:
        val_candidates = [1.0] 
        if pd.notna(current_val_gauge): val_candidates.append(abs(current_val_gauge) * 1.3)
        all_thresholds_for_max = [threshold_good, threshold_warning, target_line_value]
        valid_thresholds_for_max = [float(t) for t in all_thresholds_for_max if t is not None and pd.notna(t)]
        if valid_thresholds_for_max: val_candidates.append(max(valid_thresholds_for_max) * 1.2)
        if not valid_thresholds_for_max and (not pd.notna(current_val_gauge) or current_val_gauge == 0):
            val_candidates.append(100.0)

        max_axis_val = max(val_candidates) if val_candidates else 100.0
        if pd.notna(current_val_gauge) and current_val_gauge > max_axis_val and max_value_override is None:
             max_axis_val = current_val_gauge * 1.1
        if max_axis_val <= 0 : max_axis_val = 10.0 # Ensure positive range for gauge if needed

    steps_list = []
    num_t_good = float(threshold_good) if threshold_good is not None and pd.notna(threshold_good) else None
    num_t_warn = float(threshold_warning) if threshold_warning is not None and pd.notna(threshold_warning) else None
    
    if num_t_good is not None and num_t_warn is not None: # Ensure logical order of thresholds
        if higher_is_worse and num_t_warn < num_t_good: num_t_warn = num_t_good 
        if not higher_is_worse and num_t_warn > num_t_good: num_t_warn = num_t_good

    last_step_val = 0.0
    if higher_is_worse: 
        if num_t_good is not None:
            steps_list.append({'range': [last_step_val, num_t_good], 'color': config.COLOR_STATUS_GOOD})
            last_step_val = num_t_good
        if num_t_warn is not None and num_t_warn > last_step_val:
            steps_list.append({'range': [last_step_val, num_t_warn], 'color': config.COLOR_STATUS_WARNING})
            last_step_val = num_t_warn
        steps_list.append({'range': [last_step_val, max_axis_val], 'color': config.COLOR_STATUS_CRITICAL})
    else: 
        if num_t_warn is not None: 
            steps_list.append({'range': [last_step_val, num_t_warn], 'color': config.COLOR_STATUS_CRITICAL})
            last_step_val = num_t_warn
        if num_t_good is not None and num_t_good > last_step_val:
            steps_list.append({'range': [last_step_val, num_t_good], 'color': config.COLOR_STATUS_WARNING})
            last_step_val = num_t_good
        steps_list.append({'range': [last_step_val, max_axis_val], 'color': config.COLOR_STATUS_GOOD})

    if not steps_list: steps_list.append({'range': [0, max_axis_val], 'color': config.COLOR_NEUTRAL_INFO}) # Fallback
    
    num_target = float(target_line_value) if target_line_value is not None and pd.notna(target_line_value) else None

    fig = go.Figure(go.Indicator(
        mode="gauge+number" + ("+delta" if delta_ref_val is not None and pd.notna(current_val_gauge) else ""),
        value=current_val_gauge if pd.notna(current_val_gauge) else None, # Let Plotly handle None display for number
        title={'text': title_text, 'font': {'size': 15, 'color': config.COLOR_TEXT_SECONDARY}},
        number={'font': {'size': 26}, 'suffix': unit if unit and unit != "N/A" and pd.notna(current_val_gauge) else "", 'valueformat': ".1f"},
        delta={'reference': delta_ref_val,
               'increasing': {'color': config.COLOR_STATUS_CRITICAL if higher_is_worse else config.COLOR_STATUS_GOOD},
               'decreasing': {'color': config.COLOR_STATUS_GOOD if higher_is_worse else config.COLOR_STATUS_CRITICAL},
               'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, max_axis_val], 'tickwidth': 1, 'tickcolor': "darkgray", 'nticks': 5},
            'bar': {'color': 'rgba(0,0,0,0.7)', 'thickness': 0.1}, 
            'bgcolor': "white", 'borderwidth': 0, 'bordercolor': "lightgray",
            'steps': steps_list,
            'threshold': {
                'line': {'color': config.COLOR_TARGET_LINE, 'width': 2.5},
                'thickness': 0.75, 'value': num_target
            } if num_target is not None else {}
        }
    ))
    fig.update_layout(height=180, margin=dict(l=15, r=15, t=40, b=5), paper_bgcolor='rgba(0,0,0,0)')
    return fig

# --- Trend Chart Visualization ---
def create_trend_chart(df_input: pd.DataFrame, date_col: str,
                       value_cols_map: Dict[str, str], title_key: str, lang: str,
                       y_axis_title_key: str = "value_axis_label", x_axis_title_key: str = "date_time_axis_label",
                       show_average_line: bool = False,
                       target_value_map: Optional[Dict[str, Union[int, float]]] = None,
                       rolling_avg_window: Optional[int] = None,
                       value_col_units_map: Optional[Dict[str, str]] = None) -> go.Figure:
    df = df_input.copy() 
    title_text = get_lang_text(lang, title_key)
    x_title_text = get_lang_text(lang, x_axis_title_key)
    y_title_text = get_lang_text(lang, y_axis_title_key)

    if df.empty or date_col not in df.columns or not value_cols_map:
        return go.Figure().update_layout(
            title_text=f"{title_text} ({get_lang_text(lang, 'no_data_for_selection')})",
            annotations=[dict(text=get_lang_text(lang, 'no_data_for_selection'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5, font_size=12)]
        )

    fig = go.Figure()
    colors = config.COLOR_SCHEME_CATEGORICAL
    plotted_actual_cols = [] 

    for i, (series_display_key, actual_col_name) in enumerate(value_cols_map.items()):
        if actual_col_name not in df.columns or not pd.api.types.is_numeric_dtype(df[actual_col_name]):
            continue 
        plotted_actual_cols.append(actual_col_name)
        
        series_color = colors[i % len(colors)]
        series_name_display = get_lang_text(lang, series_display_key, actual_col_name.replace('_', ' ').title())
        unit = value_col_units_map.get(actual_col_name, "") if value_col_units_map else ""
        
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df[actual_col_name], mode='lines+markers', name=series_name_display,
            line=dict(color=series_color, width=2), marker=dict(size=5, symbol="circle-open"),
            hovertemplate=(f"<b>{series_name_display}</b><br>" +
                           f"{x_title_text}: %{{x|%b %d, %Y}}<br>" + 
                           f"{y_title_text}: %{{y:,.2f}}{unit}<extra></extra>")
        ))

    if rolling_avg_window and isinstance(rolling_avg_window, int) and rolling_avg_window > 0:
        for i, actual_col_name in enumerate(plotted_actual_cols):
            if len(df) >= rolling_avg_window : 
                original_display_key = [k for k,v in value_cols_map.items() if v == actual_col_name][0]
                base_series_name_display = get_lang_text(lang, original_display_key)
                rolling_legend_name = f"{base_series_name_display} ({rolling_avg_window}-p MA)"
                unit = value_col_units_map.get(actual_col_name, "") if value_col_units_map else ""
                # Ensure column for rolling average is distinct if original actual_col_name could clash with a key
                rolling_col_name = f'_{actual_col_name}_rolling_avg'
                df[rolling_col_name] = df[actual_col_name].rolling(window=rolling_avg_window, center=True, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=df[date_col], y=df[rolling_col_name], mode='lines', name=rolling_legend_name,
                    line=dict(color=colors[i % len(colors)], width=1.5, dash='dashdot'), opacity=0.7,
                    hovertemplate=(f"<b>{rolling_legend_name}</b><br>" +
                                   f"{x_title_text}: %{{x|%b %d, %Y}}<br>" +
                                   f"{y_title_text}: %{{y:,.2f}}{unit}<extra></extra>")
                ))
    
    for i, actual_col_name in enumerate(plotted_actual_cols):
        original_display_key = [k for k,v in value_cols_map.items() if v == actual_col_name][0]
        series_name_display = get_lang_text(lang, original_display_key)
        series_color = colors[i % len(colors)]
        if show_average_line:
            avg_val = df[actual_col_name].mean()
            if pd.notna(avg_val):
                fig.add_hline(y=avg_val, line_dash="dot", line_color=series_color, opacity=0.6,
                              annotation_text=f"{get_lang_text(lang, 'average_label')} {series_name_display}: {avg_val:,.1f}",
                              annotation_position="bottom right" if i % 2 == 0 else "top left",
                              annotation_font_size=9, annotation_font_color=series_color, annotation_bgcolor="rgba(255,255,255,0.7)")
        if target_value_map and actual_col_name in target_value_map and pd.notna(target_value_map[actual_col_name]):
            target_val = target_value_map[actual_col_name]
            fig.add_hline(y=target_val, line_dash="dash", line_color=config.COLOR_TARGET_LINE, line_width=1.5,
                          annotation_text=f"{get_lang_text(lang, 'target_label')} {series_name_display}: {target_val:,.1f}",
                          annotation_position="top right" if i % 2 == 0 else "bottom left",
                          annotation_font_size=9, annotation_font_color=config.COLOR_TARGET_LINE,
                          annotation_bgcolor="rgba(255,255,255,0.7)")

    fig.update_layout(
        title=dict(text=title_text, x=0.5, font_size=18),
        yaxis_title=y_title_text, xaxis_title=x_title_text,
        legend_title_text=get_lang_text(lang, "metrics_legend"), hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.9)", font_size=11, bordercolor=config.COLOR_TEXT_SECONDARY, namelength=-1),
        xaxis=dict(showgrid=False, type='date',
                   rangeslider_visible=len(df[date_col].unique()) > 10, 
                   rangeselector=dict(buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"), dict(step="all")
                    ]), font=dict(size=10), y=1.05, x=0, xanchor='left')),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'), 
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5, font_size=10),
        margin=dict(l=60, r=20, t=110, b=40) 
    )
    return fig

# --- Comparison Bar Chart Visualization (Definitive Fix) ---
def create_comparison_bar_chart(df_input: pd.DataFrame, x_col: str,
                                y_cols_map: Dict[str, str], # {TEXT_STRING_KEY_FOR_LABEL: ACTUAL_COLUMN_NAME_IN_DF}
                                title_key: str, lang: str,
                                y_axis_title_key: str = "count_label", x_axis_title_key: str = "category_axis_label",
                                barmode: str = 'group', show_total_for_stacked: bool = False,
                                data_label_format_str: str = ".0f") -> go.Figure:
    df = df_input.copy() # Work with a copy
    title_text = get_lang_text(lang, title_key)
    x_title_text = get_lang_text(lang, x_axis_title_key)
    y_title_text = get_lang_text(lang, y_axis_title_key)

    # 1. Filter y_cols_map to only include valid, numeric columns from df_input
    valid_y_cols_map = {}
    actual_y_cols_to_use_for_sum = [] # Keep original names for accurate summation
    for label_key, actual_name in y_cols_map.items():
        if actual_name in df.columns and pd.api.types.is_numeric_dtype(df[actual_name]):
            valid_y_cols_map[label_key] = actual_name
            actual_y_cols_to_use_for_sum.append(actual_name)

    if df.empty or x_col not in df.columns or not valid_y_cols_map:
        return go.Figure().update_layout(
            title_text=f"{title_text} ({get_lang_text(lang, 'no_data_for_selection')})",
            annotations=[dict(text=get_lang_text(lang, 'no_data_for_selection'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)]
        )

    # 2. Create a DataFrame for plotting, where y-columns are renamed to their display names (for Plotly Express legend)
    df_plot = df[[x_col] + actual_y_cols_to_use_for_sum].copy() # Select only necessary columns
    
    y_display_names_for_px = []
    plotly_labels_arg = {} # For px.bar's 'labels' argument to control legend/axis names

    for label_key, actual_name in valid_y_cols_map.items():
        display_name = get_lang_text(lang, label_key, actual_name.replace('_', ' ').title())
        if display_name != actual_name: # If renaming is needed for display
             # If multiple original columns map to the same display name, this will overwrite.
             # This shouldn't happen if label_keys are unique.
            df_plot.rename(columns={actual_name: display_name}, inplace=True) 
        y_display_names_for_px.append(display_name)
        plotly_labels_arg[display_name] = display_name # Map new column name to itself for clarity for px.bar labels


    # Ensure px.bar `y` uses the (potentially renamed) display names from df_plot
    if not y_display_names_for_px or not all(name in df_plot.columns for name in y_display_names_for_px):
        return go.Figure().update_layout(title_text=f"{title_text} (Error: Y-cols mismatch for plot)")

    try:
        fig = px.bar(df_plot, x=x_col, y=y_display_names_for_px,
                     title=None, barmode=barmode,
                     color_discrete_sequence=config.COLOR_SCHEME_CATEGORICAL,
                     labels=plotly_labels_arg # Ensures legend uses the display names
                    )
    except Exception as e_px:
        # This is a fallback if px.bar itself fails
        # print(f"PX.BAR FAILED: {e_px}, df_plot columns: {df_plot.columns}, y_display_names_for_px: {y_display_names_for_px}")
        return go.Figure().update_layout(title_text=f"{title_text} (Chart Error)")


    # Ensure a valid format specifier for templates
    fmt_spec = data_label_format_str
    if not (isinstance(fmt_spec, str) and fmt_spec and (fmt_spec.startswith(".") or fmt_spec.startswith(","))):
        fmt_spec = ".0f" # Default to integer

    # These templates expect %{fullData.name} to be the legend entry
    # and %{y} to be the value for that bar segment.
    texttemplate_final = f'%{{y:{fmt_spec}}}'
    hovertemplate_final = f'<b>%{{x}}</b><br>%{{fullData.name}}: %{{y:{fmt_spec}}}<extra></extra>'

    try:
        fig.update_traces(
            texttemplate=texttemplate_final,
            textposition='outside' if barmode != 'stack' else 'inside',
            textfont_size=9,
            insidetextanchor='middle' if barmode == 'stack' else 'auto',
            hovertemplate=hovertemplate_final
            # selector=dict(type='bar') # Not needed if all traces are bars
        )
    except Exception as e_update: # Catch errors specifically from update_traces
        # print(f"UPDATE_TRACES FAILED: {e_update}")
        # Fallback to simpler hovertemplate without formatted y if update_traces fails
        fig.update_traces(hovertemplate='<b>%{x}</b><br>%{fullData.name}: %{y}<extra></extra>')


    if barmode == 'stack' and show_total_for_stacked and y_display_names_for_px:
        # Summation should be on the numeric columns in df_plot, which now have display names
        if all(col in df_plot.columns for col in y_display_names_for_px):
            df_plot['_total_val_'] = df_plot[y_display_names_for_px].sum(axis=1, numeric_only=True)
            annotations_list = [
                dict(x=row[x_col], y=row['_total_val_'],
                     text=f"{row['_total_val_']:{fmt_spec}}",
                     font=dict(size=10, color=config.COLOR_TEXT_SECONDARY),
                     showarrow=False, yanchor='bottom', yshift=3, xanchor='center')
                for _, row in df_plot.iterrows() if pd.notna(row['_total_val_'])
            ]
            if annotations_list:
                current_layout_annotations = fig.layout.annotations or ()
                fig.update_layout(annotations=list(current_layout_annotations) + annotations_list)

    fig.update_layout(
        title=dict(text=title_text, x=0.5, font_size=18),
        yaxis_title=y_title_text, xaxis_title=x_title_text,
        legend_title_text=get_lang_text(lang, "metrics_legend"), hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.9)", font_size=11, namelength=-1),
        xaxis_tickangle=-30 if not df_plot.empty and df_plot[x_col].nunique() > 6 else 0,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font_size=10),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
        xaxis=dict(showgrid=False, type='category'),
        margin=dict(l=50, r=20, t=70, b=50)
    )
    return fig


# --- Metric Card (UX Enhanced) ---
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
        target_format_spec = ".0f" if target_float % 1 == 0 and abs(target_float) >=1 else ".1f" # Avoid ".0" for integer targets
        try:
            help_text_final_str = raw_help_text_template.format(target=f"{target_float:{target_format_spec}}")
        except (KeyError, ValueError): 
            help_text_final_str = raw_help_text_template 

    val_display_str, delta_text_str, delta_color_str, status_icon_str = "N/A", None, "normal", "❓ "

    if pd.notna(value) and isinstance(value, (int, float, np.number)):
        val_raw_float = float(value)
        # Value Formatting logic
        if unit == "%": val_display_str = f"{val_raw_float:,.1f}%"
        elif unit == get_lang_text(lang, 'days_unit') or (abs(val_raw_float) >= 1000 and val_raw_float == int(val_raw_float)): # For days or large whole numbers
            val_display_str = f"{val_raw_float:,.0f}{(' ' + unit) if unit and unit != '%' else ''}"
        elif abs(val_raw_float) < 1 and val_raw_float != 0: # For small decimal values not percentages
            val_display_str = f"{val_raw_float:,.2f}{(' ' + unit) if unit and unit != '%' else ''}"
        else: # Default for most other numbers
            val_display_str = f"{val_raw_float:,.1f}{(' ' + unit) if unit and unit != '%' else ''}"
        if unit == "%" and not val_display_str.endswith("%"): val_display_str += "%"


        if previous_value is not None and pd.notna(previous_value) and isinstance(previous_value, (int, float, np.number)):
            prev_val_raw_float = float(previous_value)
            delta_absolute = val_raw_float - prev_val_raw_float
            delta_sign = "+" if delta_absolute >= 1e-9 else ("" if abs(delta_absolute) < 1e-9 else "-") # More robust zero check
            
            delta_unit_for_display = unit if unit != '%' else '' # For absolute delta, don't add % if unit is already %
            # Formatting for the absolute part of delta
            if unit == "%": delta_abs_formatted = f"{abs(delta_absolute):.1f}%"
            elif abs(delta_absolute) >= 1000 and delta_absolute == int(delta_absolute): delta_abs_formatted = f"{abs(delta_absolute):,.0f}{(' '+delta_unit_for_display) if delta_unit_for_display else ''}"
            else: delta_abs_formatted = f"{abs(delta_absolute):,.1f}{(' '+delta_unit_for_display) if delta_unit_for_display else ''}"
            
            if abs(prev_val_raw_float) > 1e-9: # Avoid division by zero
                 delta_percent_val = (delta_absolute / abs(prev_val_raw_float)) * 100
                 delta_text_str = f"{delta_sign}{delta_abs_formatted} ({delta_sign}{abs(delta_percent_val):,.0f}%)"
            else: delta_text_str = f"{delta_sign}{delta_abs_formatted} ({get_lang_text(lang,'prev_period_label_short','Prev 0')})"

            if higher_is_better is not None: # Determine delta color
                if delta_absolute > 1e-9: delta_color_str = "normal" if higher_is_better else "inverse"
                elif delta_absolute < -1e-9: delta_color_str = "inverse" if higher_is_better else "normal"
                else: delta_color_str = "off"
        
        current_status_text = get_status_by_thresholds(val_raw_float, higher_is_better, threshold_good, threshold_warning)
        if current_status_text == "good": status_icon_str = "✅ "
        elif current_status_text == "warning": status_icon_str = "⚠️ "
        elif current_status_text == "critical": status_icon_str = "❗ "
        elif target_value is not None and higher_is_better is not None and pd.notna(target_value): # If no threshold status, check target
            if (higher_is_better and val_raw_float >= float(target_value)) or \
               (not higher_is_better and val_raw_float <= float(target_value)):
                status_icon_str = "🎯 " # Icon for meeting a general target
            else: status_icon_str = "" # No icon if target not met and no other status
        else: status_icon_str = "" # Default: no icon if no status applies
            
    st_object.metric(label=status_icon_str + label_text_orig, value=val_display_str, delta=delta_text_str, delta_color=delta_color_str, help=help_text_final_str)

# --- Radar Chart Visualization ---
def create_enhanced_radar_chart(df_radar_input: pd.DataFrame, categories_col: str, values_col: str,
                                 title_key: str, lang: str, group_col: Optional[str] = None,
                                 range_max_override: Optional[Union[int, float]] = None,
                                 target_values_map: Optional[Dict[str, Union[int, float]]] = None, 
                                 fill_opacity: float = 0.35):
    title_text = get_lang_text(lang, title_key)
    df_radar = df_radar_input.copy()

    if df_radar.empty or categories_col not in df_radar.columns or values_col not in df_radar.columns or df_radar[categories_col].nunique() == 0:
        return go.Figure().update_layout(title_text=f"{title_text} ({get_lang_text(lang, 'no_data_radar')})",
            annotations=[dict(text=get_lang_text(lang, 'no_data_radar'),showarrow=False, xref="paper", yref="paper", x=0.5,y=0.5)])

    all_categories_ordered_list = df_radar[categories_col].unique()
    
    all_r_vals = df_radar[values_col].dropna().tolist() # Ensure no NaNs from data for max calculation
    if target_values_map: all_r_vals.extend([v for v in target_values_map.values() if pd.notna(v) and isinstance(v,(int,float))])
    
    valid_r_vals = [float(v) for v in all_r_vals if isinstance(v, (int,float)) and pd.notna(v)] 
    max_data_val_for_range = max(valid_r_vals) if valid_r_vals else 0.0 
    
    radial_range_max_val = float(range_max_override) if range_max_override is not None and pd.notna(range_max_override) else (max_data_val_for_range * 1.2 if max_data_val_for_range > 0 else 5.0)
    radial_range_max_val = max(radial_range_max_val, 1.0) # Ensure positive range

    fig = go.Figure()
    colors_list = config.COLOR_SCHEME_CATEGORICAL

    if group_col and group_col in df_radar.columns and df_radar[group_col].nunique() > 0:
        for i, (name_group, group_data_df) in enumerate(df_radar.groupby(group_col)):
            current_group_ordered_df = pd.DataFrame({categories_col: all_categories_ordered_list}).merge(
                group_data_df, on=categories_col, how='left').fillna({values_col: 0}) # Ensure all categories are present
            fig.add_trace(go.Scatterpolar(
                r=current_group_ordered_df[values_col], theta=current_group_ordered_df[categories_col],
                fill='toself', name=str(name_group), line_color=colors_list[i % len(colors_list)], opacity=fill_opacity,
                hovertemplate='<b>%{theta}</b><br>' + f'{str(name_group)}: %{{r:.1f}}<extra></extra>' ))
    else: # Single series
        single_series_ordered_df = pd.DataFrame({categories_col: all_categories_ordered_list}).merge(
            df_radar, on=categories_col, how='left').fillna({values_col: 0})
        fig.add_trace(go.Scatterpolar(
            r=single_series_ordered_df[values_col], theta=single_series_ordered_df[categories_col],
            fill='toself', name=get_lang_text(lang, "average_score_label"), line_color=colors_list[0],
            opacity=fill_opacity + 0.1, hovertemplate='<b>%{theta}</b>: %{r:.1f}<extra></extra>'))

    if target_values_map: # Keys are localized category names
        target_r_values_ordered = [target_values_map.get(cat, 0) for cat in all_categories_ordered_list] # Default to 0 if category not in map
        fig.add_trace(go.Scatterpolar(
            r=target_r_values_ordered, theta=all_categories_ordered_list, mode='lines', name=get_lang_text(lang, "target_label"),
            line=dict(color=config.COLOR_TARGET_LINE, dash='longdash', width=1.5), hoverinfo='skip'))

    fig.update_layout(
        title=dict(text=title_text, x=0.5, font_size=18),
        polar=dict(bgcolor="rgba(255,255,255,0.05)",
                   radialaxis=dict(visible=True, range=[0, radial_range_max_val], showline=False, gridcolor="rgba(0,0,0,0.1)", tickfont_size=9),
                   angularaxis=dict(showline=False, gridcolor="rgba(0,0,0,0.1)", tickfont_size=10, direction="clockwise")),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font_size=9),
        margin=dict(l=50, r=50, t=100, b=60), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

# --- Stress Semáforo Visual (Bullet Gauge Style) ---
def create_stress_semaforo_visual(stress_level_value: Optional[Union[int, float, np.number]], lang: str,
                                  scale_max: float = config.STRESS_LEVEL_PSYCHOSOCIAL["max_scale"]) -> go.Figure:
    val_for_bar_indicator, color_for_status, text_for_status = None, config.COLOR_TEXT_SECONDARY, get_lang_text(lang, 'status_na_label')

    if pd.notna(stress_level_value) and isinstance(stress_level_value, (int, float, np.number)):
        val_float_stress = float(stress_level_value)
        val_for_bar_indicator = val_float_stress 
        status_stress = get_status_by_thresholds(val_float_stress, higher_is_worse=True,
                                           threshold_good=config.STRESS_LEVEL_PSYCHOSOCIAL["low"],
                                           threshold_warning=config.STRESS_LEVEL_PSYCHOSOCIAL["medium"])
        
        color_for_status = get_semaforo_color(status_stress)
        if status_stress == "good": text_for_status = get_lang_text(lang, 'low_label')
        elif status_stress == "warning": text_for_status = get_lang_text(lang, 'moderate_label')
        elif status_stress == "critical": text_for_status = get_lang_text(lang, 'high_label')
        else: text_for_status = f"{val_float_stress:.1f}" if pd.notna(val_float_stress) else get_lang_text(lang, 'status_na_label')
            
    num_config_stress = {'font': {'size': 22, 'color': color_for_status}, 'valueformat': ".1f"}
    if val_for_bar_indicator is not None: num_config_stress['suffix'] = f" / {scale_max:.0f}"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=val_for_bar_indicator, # Plotly handles None by not showing number typically
        domain={'x': [0.05, 0.95], 'y': [0.1, 0.7]},
        title={'text': f"<b style='color:{color_for_status}; font-size:1.1em;'>{text_for_status.upper()}</b>", 'font': {'size': 15}, 'align': "center"},
        number=num_config_stress,
        gauge={
            'shape': "bullet",
            'axis': {'range': [0, scale_max], 'visible': True, 'showticklabels': True,
                     'tickvals': [0, config.STRESS_LEVEL_PSYCHOSOCIAL["low"], config.STRESS_LEVEL_PSYCHOSOCIAL["medium"], scale_max],
                     'ticktext': ["0", 
                                  f"{config.STRESS_LEVEL_PSYCHOSOCIAL['low']:.1f}", 
                                  f"{config.STRESS_LEVEL_PSYCHOSOCIAL['medium']:.1f}", 
                                  f"{scale_max:.0f}"],
                     'tickfont': {'size':9, 'color': config.COLOR_TEXT_SECONDARY}, 'tickmode': 'array'},
            'steps': [
                {'range': [0, config.STRESS_LEVEL_PSYCHOSOCIAL["low"]], 'color': config.COLOR_STATUS_GOOD},
                {'range': [config.STRESS_LEVEL_PSYCHOSOCIAL["low"], config.STRESS_LEVEL_PSYCHOSOCIAL["medium"]], 'color': config.COLOR_STATUS_WARNING},
                {'range': [config.STRESS_LEVEL_PSYCHOSOCIAL["medium"], scale_max], 'color': config.COLOR_STATUS_CRITICAL}
            ],
            'bar': {'color': color_for_status, 'thickness': 0.5}, 
            'bgcolor': "rgba(230, 230, 230, 0.7)", 'borderwidth': 0.5, 'bordercolor': "rgba(150,150,150,0.5)"
        }))
    fig.update_layout(height=100, margin=dict(t=15, b=15, l=5, r=5), paper_bgcolor='rgba(0,0,0,0)')
    return fig
