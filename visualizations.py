# visualizations.py
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import config # Your configuration file
from typing import List, Dict, Optional, Any, Union # For type hinting

# --- Text Localization Helper (Centralized and Robust) ---
def get_lang_text(lang_code: str, key: str, default_text_override: Optional[str] = None) -> str:
    """
    Retrieves localized text safely from config.TEXT_STRINGS.
    Falls back to DEFAULT_LANG if lang_code is invalid.
    Falls back to default_text_override if provided and key is missing.
    Falls back to the key itself as the last resort.
    """
    # Determine effective language code, falling back to DEFAULT_LANG
    effective_lang_code = lang_code if lang_code in config.TEXT_STRINGS else config.DEFAULT_LANG
    
    # Get the text dictionary for the effective language, fallback to English if that also fails
    # (though DEFAULT_LANG should always be in TEXT_STRINGS)
    text_dict = config.TEXT_STRINGS.get(effective_lang_code, config.TEXT_STRINGS.get(config.DEFAULT_LANG, {}))
    
    # Get the text for the key
    localized_text = text_dict.get(key)
    
    if localized_text is not None:
        return localized_text
    elif default_text_override is not None:
        return default_text_override
    else:
        # print(f"Warning: Localization key '{key}' not found for lang '{lang_code}' or default. Returning key itself.")
        return key # Last resort: return the key


# --- Status and Color Helpers (Remain Essential) ---
def get_status_by_thresholds(value: Optional[Union[int, float, np.number]],
                             higher_is_worse: bool,
                             threshold_good: Optional[Union[int, float, np.number]] = None,
                             threshold_warning: Optional[Union[int, float, np.number]] = None) -> Optional[str]:
    if pd.isna(value) or value is None: return None
    val_f = float(value)
    good_f = float(threshold_good) if threshold_good is not None and pd.notna(threshold_good) else None
    warn_f = float(threshold_warning) if threshold_warning is not None and pd.notna(threshold_warning) else None

    if higher_is_worse: # Lower values are better
        if good_f is not None and val_f <= good_f: return "good"
        if warn_f is not None and (good_f is None or val_f > good_f) and val_f <= warn_f: return "warning" # Between good and warning
        if (warn_f is not None and val_f > warn_f) or \
           (warn_f is None and good_f is not None and val_f > good_f): return "critical" # Worse than warning/good
    else: # Higher values are better
        if good_f is not None and val_f >= good_f: return "good"
        if warn_f is not None and (good_f is None or val_f < good_f) and val_f >= warn_f: return "warning" # Between warning and good
        if (warn_f is not None and val_f < warn_f) or \
           (warn_f is None and good_f is not None and val_f < good_f): return "critical" # Worse than warning/good
    return None # If value falls in undefined gap or no thresholds are set

def get_semaforo_color(status: Optional[str]) -> str:
    if status == "good": return config.COLOR_STATUS_GOOD
    if status == "warning": return config.COLOR_STATUS_WARNING
    if status == "critical": return config.COLOR_STATUS_CRITICAL
    return config.COLOR_TEXT_SECONDARY # Default color for N/A or None status


# --- KPI Gauge Visualization (SME Platinum Edition) ---
def create_kpi_gauge(value: Optional[Union[int, float, np.number]], title_key: str, lang: str,
                     unit: str = "%", higher_is_worse: bool = True,
                     threshold_good: Optional[Union[int, float, np.number]] = None,
                     threshold_warning: Optional[Union[int, float, np.number]] = None,
                     target_line_value: Optional[Union[int, float, np.number]] = None,
                     max_value_override: Optional[Union[int, float, np.number]] = None,
                     previous_value: Optional[Union[int, float, np.number]] = None,
                     subtitle_key: Optional[str] = None) -> go.Figure:
    
    title_base_text = get_lang_text(lang, title_key)
    full_title_text = title_base_text
    if subtitle_key:
        subtitle_localized = get_lang_text(lang, subtitle_key)
        full_title_text = f"{title_base_text}<br><span style='font-size:0.7em;color:{config.COLOR_TEXT_SECONDARY};font-weight:normal;'>{subtitle_localized}</span>"

    current_numeric_val = float(value) if pd.notna(value) and isinstance(value, (int,float,np.number)) else 0.0
    value_for_indicator = float(value) if pd.notna(value) and isinstance(value, (int,float,np.number)) else None # Pass None for Plotly formatting

    delta_config = {}
    if previous_value is not None and pd.notna(previous_value) and isinstance(previous_value, (int,float,np.number)) and value_for_indicator is not None:
        delta_ref_numeric = float(previous_value)
        delta_actual_val = current_numeric_val - delta_ref_numeric
        
        delta_color_increasing = config.COLOR_STATUS_CRITICAL if higher_is_worse else config.COLOR_STATUS_GOOD
        delta_color_decreasing = config.COLOR_STATUS_GOOD if higher_is_worse else config.COLOR_STATUS_CRITICAL
        
        delta_text_color = config.COLOR_TEXT_SECONDARY # Default if no change
        if abs(delta_actual_val) > 1e-9: # A small tolerance for zero change
            delta_text_color = delta_color_increasing if delta_actual_val > 0 else delta_color_decreasing
        
        delta_config = {
            'reference': delta_ref_numeric,
            'increasing': {'color': delta_color_increasing, 'symbol': "▲"},
            'decreasing': {'color': delta_color_decreasing, 'symbol': "▼"},
            'font': {'size': 12, 'color': delta_text_color}
        }

    # Determine max_value for gauge axis
    if max_value_override is not None and pd.notna(max_value_override):
        axis_max = float(max_value_override)
    else:
        max_candidates = [1.0]
        if pd.notna(current_numeric_val): max_candidates.append(abs(current_numeric_val) * 1.4)
        for ref_point in [threshold_good, threshold_warning, target_line_value]:
            if ref_point is not None and pd.notna(ref_point): max_candidates.append(float(ref_point) * 1.25)
        if not any(pd.notna(p) for p in [threshold_good, threshold_warning, target_line_value]) and (not pd.notna(current_numeric_val) or current_numeric_val == 0):
            max_candidates.append(100.0 if unit == "%" else 10.0) # Default scale max
        axis_max = max(max_candidates) if max_candidates else 100.0
        if axis_max <= current_numeric_val and pd.notna(current_numeric_val): axis_max = current_numeric_val * 1.1
        if axis_max <= 0: axis_max = 10.0 # Ensure positive range for the axis

    # Define gauge steps based on thresholds
    steps_config = []
    num_t_good = float(threshold_good) if threshold_good is not None and pd.notna(threshold_good) else None
    num_t_warn = float(threshold_warning) if threshold_warning is not None and pd.notna(threshold_warning) else None
    
    if num_t_good is not None and num_t_warn is not None: # Ensure logical order for thresholds
        if higher_is_worse and num_t_warn < num_t_good: num_t_warn = num_t_good
        if not higher_is_worse and num_t_warn > num_t_good: num_t_warn = num_t_good

    current_range_start = 0.0
    if higher_is_worse:
        if num_t_good is not None:
            steps_config.append({'range': [current_range_start, num_t_good], 'color': config.COLOR_STATUS_GOOD})
            current_range_start = num_t_good
        if num_t_warn is not None and num_t_warn > current_range_start:
            steps_config.append({'range': [current_range_start, num_t_warn], 'color': config.COLOR_STATUS_WARNING})
            current_range_start = num_t_warn
        steps_config.append({'range': [current_range_start, axis_max], 'color': config.COLOR_STATUS_CRITICAL})
    else: # Higher is better
        if num_t_warn is not None: # Warning defines the lower 'bad' range
            steps_config.append({'range': [current_range_start, num_t_warn], 'color': config.COLOR_STATUS_CRITICAL})
            current_range_start = num_t_warn
        if num_t_good is not None and num_t_good > current_range_start: # Good is above warning
            steps_config.append({'range': [current_range_start, num_t_good], 'color': config.COLOR_STATUS_WARNING}) # Mid-range
            current_range_start = num_t_good
        steps_config.append({'range': [current_range_start, axis_max], 'color': config.COLOR_STATUS_GOOD}) # Top range is good
    if not steps_config: steps_config.append({'range': [0, axis_max], 'color': config.COLOR_NEUTRAL_INFO})
    
    target_line_float = float(target_line_value) if target_line_value is not None and pd.notna(target_line_value) else None
    
    # Color number based on its status
    number_status = get_status_by_thresholds(current_numeric_val, higher_is_worse, num_t_good, num_t_warn)
    value_display_color = get_semaforo_color(number_status) if number_status else config.COLOR_TARGET_LINE


    number_value_format = ".1f" # Default to one decimal
    if unit != "%" and pd.notna(value_for_indicator) and float(value_for_indicator) == int(value_for_indicator) and abs(float(value_for_indicator)) >= 1:
        number_value_format = ".0f" # Show as integer if it's a whole number (and not for percentages usually)


    fig = go.Figure(go.Indicator(
        mode="gauge+number" + ("+delta" if delta_config else ""),
        value=value_for_indicator, # Let Plotly handle None by showing N/A if 'value' is None
        title={'text': full_title_text, 'font': {'size': 13, 'color': config.COLOR_TEXT_SECONDARY}},
        number={'font': {'size': 28, 'color': value_display_color}, # Color value based on status
                'suffix': unit if unit and pd.notna(value_for_indicator) else "", 
                'valueformat': number_value_format},
        delta=delta_config if delta_config else None, # Pass None if empty
        gauge={
            'axis': {'range': [0, axis_max], 'tickwidth': 1, 'tickcolor': "rgba(0,0,0,0.2)", 'nticks': 5, 'tickfont':{'size':9}},
            'bar': {'color': "rgba(0,0,0,0.65)", 'thickness': 0.12, 'line':{'color':"rgba(0,0,0,0.8)", 'width':0.5}},
            'bgcolor': "rgba(255,255,255,0.0)", 
            'borderwidth': 0.5, 'bordercolor': "rgba(0,0,0,0.1)",
            'steps': steps_config,
            'threshold': {
                'line': {'color': config.COLOR_TARGET_LINE, 'width': 2.5},
                'thickness': 0.8, 'value': target_line_float
            } if target_line_float is not None else {}
        }
    ))
    fig.update_layout(height=175, margin=dict(l=15, r=15, t=40, b=15), paper_bgcolor='rgba(0,0,0,0)')
    return fig

# --- Trend Chart Visualization (SME Platinum Edition) ---
def create_trend_chart(df_input: pd.DataFrame, date_col: str,
                       value_cols_map: Dict[str, str], # {text_key_for_legend: actual_col_name_in_df}
                       title_key: str, lang: str,
                       y_axis_title_key: str = "value_axis_label", x_axis_title_key: str = "date_time_axis_label",
                       show_average_line: bool = False,
                       target_value_map: Optional[Dict[str, Union[int, float]]] = None, # {actual_col_name: target_value}
                       rolling_avg_window: Optional[int] = None,
                       value_col_units_map: Optional[Dict[str, str]] = None, # {actual_col_name: unit_string}
                       y_axis_format_str: Optional[str] = ",.1f") -> go.Figure: # Default d3 format for y-axis
    df = df_input.copy() 
    title_text = get_lang_text(lang, title_key)
    x_title = get_lang_text(lang, x_axis_title_key)
    y_title = get_lang_text(lang, y_axis_title_key)

    if df.empty or date_col not in df.columns or not value_cols_map:
        return go.Figure().update_layout(title_text=f"{title_text} ({get_lang_text(lang, 'no_data_for_selection')})",
            annotations=[dict(text=get_lang_text(lang, 'no_data_for_selection'), showarrow=False,xref="paper",yref="paper",x=0.5,y=0.5)])

    fig = go.Figure()
    colors = px.colors.qualitative.Set2 # Another visually distinct palette
    plotted_actual_cols = []

    for i, (legend_key, actual_col) in enumerate(value_cols_map.items()):
        if actual_col not in df.columns or not pd.api.types.is_numeric_dtype(df[actual_col]):
            continue
        plotted_actual_cols.append(actual_col)
        
        color = colors[i % len(colors)]
        name = get_lang_text(lang, legend_key, actual_col.replace('_',' ').title())
        unit = value_col_units_map.get(actual_col, "") if value_col_units_map else ""
        y_fmt = y_axis_format_str if y_axis_format_str else ",.2f"

        fig.add_trace(go.Scatter(x=df[date_col], y=df[actual_col], mode='lines+markers', name=name,
            line=dict(color=color, width=2), marker=dict(size=5, symbol="circle"),
            hovertemplate=f"<b>{name}</b><br>{get_lang_text(lang, 'date_label', 'Date')}: %{{x|%Y-%m-%d}}<br>{y_title}: %{{y:{y_fmt}}}{unit}<extra></extra>"))

    if rolling_avg_window and isinstance(rolling_avg_window, int) and rolling_avg_window > 0:
        for i, actual_col in enumerate(plotted_actual_cols):
            if len(df) >= rolling_avg_window :
                legend_key = [k for k,v in value_cols_map.items() if v == actual_col][0]
                base_name = get_lang_text(lang, legend_key)
                ma_name = f"{base_name} ({rolling_avg_window}-p MA)"
                unit = value_col_units_map.get(actual_col, "") if value_col_units_map else ""
                # Use a temporary, unique column name for the rolling average
                rolling_col_temp = f"__{actual_col}_rolling_avg_temp__"
                df[rolling_col_temp] = df[actual_col].rolling(window=rolling_avg_window, center=True, min_periods=1).mean()
                fig.add_trace(go.Scatter(x=df[date_col], y=df[rolling_col_temp], mode='lines', name=ma_name,
                    line=dict(color=colors[i % len(colors)], width=1.5, dash='longdashdot'), opacity=0.75,
                    hovertemplate=f"<b>{ma_name}</b><br>{get_lang_text(lang, 'date_label', 'Date')}: %{{x|%Y-%m-%d}}<br>{y_title}: %{{y:{y_fmt}}}{unit}<extra></extra>"))
    
    for i, actual_col in enumerate(plotted_actual_cols):
        legend_key = [k for k,v in value_cols_map.items() if v == actual_col][0]
        series_name = get_lang_text(lang, legend_key)
        line_color_for_annotations = colors[i % len(colors)]
        if show_average_line:
            avg = df[actual_col].mean()
            if pd.notna(avg):
                fig.add_hline(y=avg, line_dash="dot", line_color=line_color_for_annotations, opacity=0.6,
                              annotation_text=f"{get_lang_text(lang, 'average_label')} {series_name}: {avg:{y_fmt if y_fmt else ',.1f'}}",
                              annotation_position="bottom right" if i%2==0 else "top left",
                              annotation_font=dict(size=9, color=line_color_for_annotations), annotation_bgcolor="rgba(255,255,255,0.8)")
        if target_value_map and actual_col in target_value_map and pd.notna(target_value_map[actual_col]):
            target = target_value_map[actual_col]
            fig.add_hline(y=target, line_dash="solid", line_color=config.COLOR_TARGET_LINE, line_width=1.5, opacity=1.0,
                          annotation_text=f"{get_lang_text(lang, 'target_label')} {series_name}: {target:{y_fmt if y_fmt else ',.1f'}}",
                          annotation_position="top right" if i%2==0 else "bottom left",
                          annotation_font=dict(size=10, color=config.COLOR_TARGET_LINE,family="Arial Black"), annotation_bgcolor="rgba(255,255,255,0.8)")

    fig.update_layout(
        title=dict(text=title_text, x=0.03, y=0.95, xanchor='left', yanchor='top', font_size=17),
        yaxis_title=y_title, xaxis_title=x_title,
        legend_title_text="", hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=11, bordercolor=config.COLOR_TEXT_SECONDARY, namelength=-1),
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', type='date',
                   showspikes=True, spikemode='across+marker', spikesnap='cursor', spikethickness=1, spikedash='solid', spikecolor='rgba(0,0,0,0.3)',
                   rangeslider_visible= len(df[date_col].unique()) > 15, # Show slider if data is ample
                   rangeselector=dict(buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="todate" if not df.empty and df[date_col].max() > pd.Timestamp.now() - pd.DateOffset(months=1) else "backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"), dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"), dict(count=1, label=get_lang_text(lang, "1y_range_label", "1Y"), step="year", stepmode="backward"),
                        dict(step="all", label=get_lang_text(lang, "all_range_label", "All"))]),
                        font_size=10, bgcolor='rgba(230,230,230,0.7)', borderwidth=1, bordercolor='rgba(0,0,0,0.1)',
                        y=1.18, x=0.01, xanchor='left')), # Slightly above legend
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickformat=y_axis_format_str),
        legend=dict(orientation="h", yanchor="top", y=1.09, xanchor="right", x=1, font_size=10, traceorder="normal", bgcolor="rgba(255,255,255,0.6)"),
        margin=dict(l=60, r=30, t=100, b=50) # Top margin for title, bottom for slider if shown
    )
    return fig

# --- Comparison Bar Chart Visualization (SME Platinum Edition) ---
def create_comparison_bar_chart(df_input: pd.DataFrame, x_col: str,
                                y_cols_map: Dict[str, str], # {TEXT_STRING_KEY_FOR_LABEL: ACTUAL_COLUMN_NAME_IN_DF}
                                title_key: str, lang: str,
                                y_axis_title_key: str = "count_label", x_axis_title_key: str = "category_axis_label",
                                barmode: str = 'group', show_total_for_stacked: bool = False,
                                data_label_format_str: str = ".0f") -> go.Figure:
    df = df_input.copy() 
    title_text = get_lang_text(lang, title_key)
    x_title = get_lang_text(lang, x_axis_title_key)
    y_title = get_lang_text(lang, y_axis_title_key)
    
    # Filter and prepare y-columns and their display names for Plotly Express
    y_cols_for_px = [] # Actual column names in df to pass to px.bar's `y`
    labels_for_px = {} # Maps actual_col_name -> display_name for px.bar's `labels` arg

    for label_key, actual_name in y_cols_map.items():
        if actual_name in df.columns and pd.api.types.is_numeric_dtype(df[actual_name]):
            y_cols_for_px.append(actual_name)
            labels_for_px[actual_name] = get_lang_text(lang, label_key, actual_name.replace('_', ' ').title())
    
    if df.empty or x_col not in df.columns or not y_cols_for_px:
        return go.Figure().update_layout(title_text=f"{title_text} ({get_lang_text(lang, 'no_data_for_selection')})",
            annotations=[dict(text=get_lang_text(lang, 'no_data_for_selection'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)])

    fig = px.bar(df, x=x_col, y=y_cols_for_px, # Use original column names for 'y'
                 title=None, barmode=barmode,
                 color_discrete_sequence=px.colors.qualitative.Pastel if barmode == 'stack' else config.COLOR_SCHEME_CATEGORICAL, # Softer for stacks
                 labels=labels_for_px # Tell px.bar how to name them in legend/tooltips
                )
    
    fmt = data_label_format_str if (isinstance(data_label_format_str, str) and data_label_format_str) else ".0f"
    texttemplate_str = f'%{{y:{fmt}}}'
    # Hovertemplate should use trace.name, which Plotly Express sets based on `labels` or column name
    
    # Update traces to customize hovertemplate using trace.name (set by px via labels)
    for trace in fig.data:
        if trace.type == 'bar':
             trace.hovertemplate = f"<b>%{{x}}</b><br>{trace.name}: %{{y:{fmt}}}<extra></extra>"
             trace.texttemplate = texttemplate_str
             trace.textposition = 'outside' if barmode != 'stack' else 'inside'
             trace.textfont = dict(size=9, color= 'rgba(0,0,0,0.7)' if barmode == 'stack' else 'black')
             trace.insidetextanchor = 'middle' if barmode == 'stack' else 'auto'
             trace.marker.line.width = 0.5
             trace.marker.line.color = 'rgba(0,0,0,0.5)'


    if barmode == 'stack' and show_total_for_stacked and y_cols_for_px:
        df_total_sum = df.copy()
        df_total_sum['_sum_for_stack_'] = df_total_sum[y_cols_for_px].sum(axis=1, numeric_only=True)
        annotations_total = [
            dict(x=r[x_col], y=r['_sum_for_stack_'], text=f"{r['_sum_for_stack_']:{fmt}}",
                 font=dict(size=9, color=config.COLOR_TARGET_LINE), showarrow=False, 
                 yanchor='bottom', yshift=3, xanchor='center')
            for i, r in df_total_sum.iterrows() if pd.notna(r['_sum_for_stack_'])
        ]
        if annotations_total:
            current_annotations = list(fig.layout.annotations or [])
            fig.update_layout(annotations=current_annotations + annotations_total)

    fig.update_layout(
        title=dict(text=title_text, x=0.05, xanchor='left', font_size=16),
        yaxis_title=y_title, xaxis_title=x_title,
        legend_title_text="", hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=11, namelength=-1),
        xaxis_tickangle=-30 if len(df[x_col].unique()) > 7 else 0,
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, font_size=9, title_text=""), # Legend below
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.08)', zeroline=True, zerolinecolor='rgba(0,0,0,0.2)'),
        xaxis=dict(showgrid=False, type='category', linecolor='rgba(0,0,0,0.2)',
                   # Add some padding to x-axis if few bars to prevent them being too wide
                   # automargin=True, # This might help with labels, or specific padding
                  ),
        bargap=0.2 if barmode == 'group' else 0.05, # Adjust group gap
        bargroupgap=0.1 if barmode == 'group' else 0, # Gap within groups
        margin=dict(l=50, r=20, t=60, b=100 if len(y_cols_map)>2 else 60)
    )
    return fig

# --- Metric Card (UX Enhanced - Keep Robust Version) ---
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
        # More careful formatting for target in help text
        target_format_spec = ".0f" if target_float == int(target_float) and abs(target_float) >= 1 else ".1f"
        try: help_text_final_str = raw_help_text_template.format(target=f"{target_float:{target_format_spec}}")
        except (KeyError, ValueError): help_text_final_str = raw_help_text_template 
    val_display_str, delta_text_str, delta_color_str, status_icon_str = "N/A", None, "normal", "❓ "

    if pd.notna(value) and isinstance(value, (int, float, np.number)):
        val_raw_float = float(value)
        # Value Formatting
        if unit == "%": val_display_str = f"{val_raw_float:,.1f}%"
        elif unit == get_lang_text(lang, 'days_unit') or (abs(val_raw_float) >= 1000 and val_raw_float == int(val_raw_float)):
            val_display_str = f"{val_raw_float:,.0f}{(' ' + unit) if unit and unit != '%' else ''}"
        elif abs(val_raw_float) < 1 and val_raw_float != 0 and unit != "%": 
            val_display_str = f"{val_raw_float:,.2f}{(' ' + unit) if unit and unit != '%' else ''}" # More precision for small non-%
        else: val_display_str = f"{val_raw_float:,.1f}{(' ' + unit) if unit and unit != '%' else ''}"
        if unit == "%" and not val_display_str.endswith("%"): val_display_str += "%" # Ensure % only once if unit is %

        # Delta Calculation
        if previous_value is not None and pd.notna(previous_value) and isinstance(previous_value, (int, float, np.number)):
            prev_val_raw_float = float(previous_value)
            delta_absolute_val = val_raw_float - prev_val_raw_float
            delta_sign_str = "+" if delta_absolute_val >= 1e-9 else ("" if abs(delta_absolute_val) < 1e-9 else "-")
            
            delta_unit_str_display = unit if unit != '%' else '' # Unit for abs part of delta
            # Formatting for absolute part of delta, matching main value's style
            if unit == "%": delta_abs_formatted_str = f"{abs(delta_absolute_val):.1f}%"
            elif abs(delta_absolute_val) >=1000 and delta_absolute_val == int(delta_absolute_val) : delta_abs_formatted_str = f"{abs(delta_absolute_val):,.0f}{(' '+delta_unit_str_display) if delta_unit_str_display else ''}"
            else: delta_abs_formatted_str = f"{abs(delta_absolute_val):,.1f}{(' '+delta_unit_str_display) if delta_unit_str_display else ''}"

            if abs(prev_val_raw_float) > 1e-9: # Avoid division by zero for percentage change
                 delta_percentage_change = (delta_absolute_val / abs(prev_val_raw_float)) * 100
                 delta_text_str = f"{delta_sign_str}{delta_abs_formatted_str} ({delta_sign_str}{abs(delta_percentage_change):,.0f}%)"
            else: delta_text_str = f"{delta_sign_str}{delta_abs_formatted_str} ({get_lang_text(lang,'prev_period_label_short','Prev 0')})"

            # Delta Color Logic
            if higher_is_better is not None:
                if delta_absolute_val > 1e-9: delta_color_str = "normal" if higher_is_better else "inverse"
                elif delta_absolute_val < -1e-9: delta_color_str = "inverse" if higher_is_better else "normal"
                else: delta_color_str = "off" # No change
        
        # Icon Logic
        current_status_for_icon = get_status_by_thresholds(val_raw_float, higher_is_better, threshold_good, threshold_warning)
        if current_status_for_icon == "good": status_icon_str = "✅ "
        elif current_status_for_icon == "warning": status_icon_str = "⚠️ "
        elif current_status_for_icon == "critical": status_icon_str = "❗ "
        elif target_value is not None and higher_is_better is not None and pd.notna(target_value): # If no threshold status, check target
            if (higher_is_better and val_raw_float >= float(target_value)) or \
               (not higher_is_better and val_raw_float <= float(target_value)):
                status_icon_str = "🎯 " # Icon for meeting general target
            # else: status_icon_str = "👀 " # Optional: icon for not meeting target if not warning/critical
        else: status_icon_str = "" # No icon if no clear status or target indication
            
    st_object.metric(label=status_icon_str + label_text_orig, value=val_display_str, delta=delta_text_str, delta_color=delta_color_str, help=help_text_final_str)


# --- Radar Chart Visualization (UX Enhanced) ---
def create_enhanced_radar_chart(df_radar_input: pd.DataFrame, categories_col: str, values_col: str,
                                 title_key: str, lang: str, group_col: Optional[str] = None,
                                 range_max_override: Optional[Union[int, float]] = None,
                                 target_values_map: Optional[Dict[str, Union[int, float]]] = None, 
                                 fill_opacity: float = 0.3):
    title_text = get_lang_text(lang, title_key)
    df_radar = df_radar_input.copy()
    if df_radar.empty or categories_col not in df_radar.columns or values_col not in df_radar.columns or df_radar[categories_col].nunique() == 0:
        return go.Figure().update_layout(title_text=f"{title_text} ({get_lang_text(lang, 'no_data_radar')})",
            annotations=[dict(text=get_lang_text(lang, 'no_data_radar'),showarrow=False, xref="paper", yref="paper", x=0.5,y=0.5)])
    
    all_categories_ordered_list = df_radar[categories_col].unique() # These are display names already
    
    # Calculate max for radial axis, considering data and targets
    all_r_values_for_scale = []
    if values_col in df_radar.columns and not df_radar[values_col].dropna().empty:
         all_r_values_for_scale.extend(df_radar[values_col].dropna().tolist())
    if target_values_map: 
        all_r_values_for_scale.extend([v for v in target_values_map.values() if pd.notna(v) and isinstance(v,(int,float))])
    
    valid_r_values_float = [float(v) for v in all_r_values_for_scale if isinstance(v, (int,float)) and pd.notna(v)]
    max_val_from_data_or_target = max(valid_r_values_float) if valid_r_values_float else 0.0
    
    # Use config for default scale if available
    default_radar_scale_max = config.ENGAGEMENT_RADAR_DIM_SCALE_MAX if pd.notna(config.ENGAGEMENT_RADAR_DIM_SCALE_MAX) else 5.0
    
    radial_axis_max = float(range_max_override) if range_max_override is not None and pd.notna(range_max_override) else \
                     (max_val_from_data_or_target * 1.15 if max_val_from_data_or_target > 0 else default_radar_scale_max)
    radial_axis_max = max(radial_axis_max, 1.0) # Ensure a minimum positive range

    fig = go.Figure()
    colors_radar_palette = px.colors.qualitative.Vivid # More vivid palette for radar

    has_groups_radar = group_col and group_col in df_radar.columns and df_radar[group_col].nunique() > 0
    
    if has_groups_radar:
        for i, (group_name_radar, group_data_radar_df) in enumerate(df_radar.groupby(group_col)):
            # Merge ensures all categories are present, in order, for each group
            current_group_data_ordered = pd.DataFrame({categories_col: all_categories_ordered_list}).merge(
                group_data_radar_df, on=categories_col, how='left').fillna({values_col: 0}) # Fill missing category scores with 0
            fig.add_trace(go.Scatterpolar(
                r=current_group_data_ordered[values_col], theta=current_group_data_ordered[categories_col],
                fill='toself', name=str(group_name_radar), 
                line_color=colors_radar_palette[i % len(colors_radar_palette)], opacity=fill_opacity,
                hovertemplate='<b>%{theta}</b><br>' + f'{str(group_name_radar)}: %{{r:.1f}}<extra></extra>' ))
    else: # Single series (or no valid group_col)
        if values_col in df_radar.columns and not df_radar[values_col].dropna().empty:
            single_series_data_ordered = pd.DataFrame({categories_col: all_categories_ordered_list}).merge(
                df_radar, on=categories_col, how='left').fillna({values_col: 0})
            fig.add_trace(go.Scatterpolar(
                r=single_series_data_ordered[values_col], theta=single_series_data_ordered[categories_col],
                fill='toself', name=get_lang_text(lang, "average_score_label"), 
                line_color=colors_radar_palette[0], opacity=fill_opacity + 0.1, # Make main slightly more opaque
                hovertemplate='<b>%{theta}</b>: %{r:.1f}<extra></extra>'))
        # If df_radar is empty for the values_col, no data trace is added. Target trace might still appear.

    if target_values_map: # Keys in target_values_map are already localized display names from app.py
        target_r_vals_ordered = [target_values_map.get(cat, 0) for cat in all_categories_ordered_list] # Default to 0 if cat not in map
        fig.add_trace(go.Scatterpolar(
            r=target_r_vals_ordered, theta=all_categories_ordered_list, mode='lines', 
            name=get_lang_text(lang, "target_label"),
            line=dict(color=config.COLOR_TARGET_LINE, dash='dashdot', width=2.5), # More prominent target
            hoverinfo='skip')) 
    
    show_legend_final_radar = has_groups_radar or (target_values_map and ( (not has_groups_radar and values_col in df_radar.columns and not df_radar[values_col].dropna().empty) or has_groups_radar ) )


    fig.update_layout(
        title=dict(text=title_text, x=0.5, y=0.95, yanchor='top', font_size=16), 
        polar=dict(bgcolor="rgba(250,250,250,0.0)", # Very light transparent for modern feel
                   radialaxis=dict(visible=True, range=[0, radial_axis_max], showline=True, 
                                   linecolor='rgba(0,0,0,0.15)', gridcolor="rgba(0,0,0,0.1)", 
                                   tickfont_size=8, nticks=5, showticklabels=True, layer='below traces'), # Ticks below data
                   angularaxis=dict(showline=True, linecolor='rgba(0,0,0,0.15)', gridcolor="rgba(0,0,0,0.05)", 
                                    tickfont_size=9, direction="clockwise", showticklabels=True, layer='below traces')),
        showlegend=show_legend_final_radar, 
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, font_size=9), 
        margin=dict(l=30, r=30, t=50, b=50), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


# --- Stress Semáforo Visual (SME Platinum Edition) ---
def create_stress_semaforo_visual(stress_level_value: Optional[Union[int, float, np.number]], lang: str,
                                  scale_max: float = config.STRESS_LEVEL_PSYCHOSOCIAL["max_scale"]) -> go.Figure:
    # Note: `title_key` is typically "overall_stress_indicator_title" and used as st.subheader in app.py
    
    current_stress_val_num, status_color, status_text_display = None, config.COLOR_TEXT_SECONDARY, get_lang_text(lang, 'status_na_label')

    if pd.notna(stress_level_value) and isinstance(stress_level_value, (int, float, np.number)):
        stress_float = float(stress_level_value)
        current_stress_val_num = stress_float
        stress_status = get_status_by_thresholds(stress_float, higher_is_worse=True,
                                           threshold_good=config.STRESS_LEVEL_PSYCHOSOCIAL["low"],
                                           threshold_warning=config.STRESS_LEVEL_PSYCHOSOCIAL["medium"])
        status_color = get_semaforo_color(stress_status)
        if stress_status == "good": status_text_display = get_lang_text(lang, 'low_label')
        elif stress_status == "warning": status_text_display = get_lang_text(lang, 'moderate_label')
        elif stress_status == "critical": status_text_display = get_lang_text(lang, 'high_label')
        else: # Fallback if status is None
            status_text_display = f"{stress_float:.1f}" if pd.notna(stress_float) else get_lang_text(lang, 'status_na_label')
            
    # Configuration for the number part of the indicator
    indicator_number_config = {
        'font': {'size': 22, 'color': status_color},
        'valueformat': ".1f" # Apply formatting to the displayed number
    }
    if current_stress_val_num is not None: 
        indicator_number_config['suffix'] = f" / {scale_max:.0f}" # Add scale only if value is present
    
    # This is the text that appears largest as part of the indicator itself
    indicator_title_main_text = f"<b style='color:{status_color}; font-size:1.0em;'>{status_text_display.upper()}</b>"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number", 
        value=current_stress_val_num, # This is the primary value Plotly uses to format and display number
        domain={'x': [0.0, 1.0], 'y': [0.0, 1.0]}, # Fill the allocated space
        title={ # Title displayed by the indicator itself, usually above number
            'text': indicator_title_main_text, 
            'font': {'size': 14}, # Adjust font size for indicator title
            'align': 'center' # Explicitly align if supported
        },
        number=indicator_number_config,
        gauge={
            'shape': "bullet",
            'axis': {'range': [0, scale_max], 'visible': True, 'showticklabels': True,
                     'tickvals': [0, config.STRESS_LEVEL_PSYCHOSOCIAL["low"], config.STRESS_LEVEL_PSYCHOSOCIAL["medium"], scale_max],
                     'ticktext': ["0", f"{config.STRESS_LEVEL_PSYCHOSOCIAL['low']:.0f}", f"{config.STRESS_LEVEL_PSYCHOSOCIAL['medium']:.0f}", f"{scale_max:.0f}"],
                     'tickfont': {'size':9, 'color': config.COLOR_TEXT_SECONDARY}, 'tickmode': 'array'},
            'steps': [ # Color bands for context
                {'range': [0, config.STRESS_LEVEL_PSYCHOSOCIAL["low"]], 'color': "rgba(46, 204, 113, 0.5)"}, # Softer fill
                {'range': [config.STRESS_LEVEL_PSYCHOSOCIAL["low"], config.STRESS_LEVEL_PSYCHOSOCIAL["medium"]], 'color': "rgba(241, 196, 15, 0.5)"},
                {'range': [config.STRESS_LEVEL_PSYCHOSOCIAL["medium"], scale_max], 'color': "rgba(231, 76, 60, 0.5)"}
            ],
            'bar': {'color': status_color, 'thickness': 0.6, 'line':{'color':'rgba(0,0,0,0.4)', 'width':0.5}},
            'bgcolor': "rgba(255,255,255,0)", 
            'borderwidth': 0.5, 
            'bordercolor': "rgba(0,0,0,0.1)"
        }))
    fig.update_layout(height=80, margin=dict(t=10, b=10, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)') # Even more compact
    return fig
