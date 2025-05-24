# visualizations.py
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import config
from typing import List, Dict, Optional, Any, Union

# --- Helper to get localized text ---
def get_lang_text(lang_code: str, key: str, default_text: Optional[str] = None) -> str:
    effective_lang_code = lang_code if lang_code in config.TEXT_STRINGS else config.DEFAULT_LANG
    text_dict = config.TEXT_STRINGS.get(effective_lang_code, config.TEXT_STRINGS.get(config.DEFAULT_LANG, {}))
    return text_dict.get(key, default_text if default_text is not None else key)

# --- Helper for status determination ---
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
    else: 
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


# --- KPI Gauge Visualization (SME Platinum Edition) ---
def create_kpi_gauge(value: Optional[Union[int, float, np.number]], title_key: str, lang: str,
                     unit: str = "%", higher_is_worse: bool = True,
                     threshold_good: Optional[Union[int, float, np.number]] = None,
                     threshold_warning: Optional[Union[int, float, np.number]] = None,
                     target_line_value: Optional[Union[int, float, np.number]] = None,
                     max_value_override: Optional[Union[int, float, np.number]] = None,
                     previous_value: Optional[Union[int, float, np.number]] = None,
                     subtitle_key: Optional[str] = None) -> go.Figure:
    
    title_base = get_lang_text(lang, title_key)
    title_final = title_base
    if subtitle_key:
        subtitle_str = get_lang_text(lang, subtitle_key)
        title_final = f"{title_base}<br><span style='font-size:0.8em;color:{config.COLOR_TEXT_SECONDARY};font-weight:normal;'>{subtitle_str}</span>"
    current_val_numeric = float(value) if pd.notna(value) and isinstance(value, (int,float,np.number)) else 0.0
    current_val_for_display = float(value) if pd.notna(value) and isinstance(value, (int,float,np.number)) else None
    delta_obj = {} 
    if previous_value is not None and pd.notna(previous_value) and isinstance(previous_value, (int, float, np.number)) and current_val_for_display is not None:
        delta_ref_val = float(previous_value)
        delta_val = current_val_numeric - delta_ref_val
        increasing_color = config.COLOR_STATUS_CRITICAL if higher_is_worse else config.COLOR_STATUS_GOOD
        decreasing_color = config.COLOR_STATUS_GOOD if higher_is_worse else config.COLOR_STATUS_CRITICAL
        delta_font_color = config.COLOR_TEXT_SECONDARY 
        if abs(delta_val) > 1e-9 : delta_font_color = increasing_color if delta_val > 0 else decreasing_color
        delta_obj = {
            'reference': delta_ref_val,
            'increasing': {'color': increasing_color, 'symbol': "▲"}, 
            'decreasing': {'color': decreasing_color, 'symbol': "▼"}, 
            'font': {'size': 12, 'color': delta_font_color} 
        }
    if max_value_override is not None and pd.notna(max_value_override):
        axis_max_val = float(max_value_override)
    else:
        val_candidates_for_max = [1.0]
        if pd.notna(current_val_numeric): val_candidates_for_max.append(abs(current_val_numeric) * 1.4)
        ref_points_for_max = [threshold_good, threshold_warning, target_line_value]
        valid_ref_points_for_max = [float(p) for p in ref_points_for_max if p is not None and pd.notna(p)]
        if valid_ref_points_for_max: val_candidates_for_max.append(max(valid_ref_points_for_max) * 1.25)
        if not valid_ref_points_for_max and (not pd.notna(current_val_numeric) or current_val_numeric == 0):
             val_candidates_for_max.append(100.0 if unit == "%" else 10.0)
        axis_max_val = max(val_candidates_for_max) if val_candidates_for_max else 100.0
        if axis_max_val <= (current_val_numeric if pd.notna(current_val_numeric) else 0):
            axis_max_val = (current_val_numeric * 1.1) if pd.notna(current_val_numeric) and current_val_numeric > 0 else (axis_max_val * 1.1 or 10.0)
        if axis_max_val <= 0: axis_max_val = 10.0
    gauge_steps = []
    num_t_good = float(threshold_good) if threshold_good is not None and pd.notna(threshold_good) else None
    num_t_warn = float(threshold_warning) if threshold_warning is not None and pd.notna(threshold_warning) else None
    if num_t_good is not None and num_t_warn is not None:
        if higher_is_worse and num_t_warn < num_t_good: num_t_warn = num_t_good
        if not higher_is_worse and num_t_warn > num_t_good: num_t_warn = num_t_good
    range_start = 0.0
    if higher_is_worse:
        if num_t_good is not None:
            gauge_steps.append({'range': [range_start, num_t_good], 'color': config.COLOR_STATUS_GOOD, 'name': get_lang_text(lang, 'good_label')})
            range_start = num_t_good
        if num_t_warn is not None and num_t_warn > range_start:
            gauge_steps.append({'range': [range_start, num_t_warn], 'color': config.COLOR_STATUS_WARNING, 'name': get_lang_text(lang, 'warning_label')})
            range_start = num_t_warn
        gauge_steps.append({'range': [range_start, axis_max_val], 'color': config.COLOR_STATUS_CRITICAL, 'name': get_lang_text(lang, 'critical_label')})
    else:
        if num_t_warn is not None:
            gauge_steps.append({'range': [range_start, num_t_warn], 'color': config.COLOR_STATUS_CRITICAL, 'name': get_lang_text(lang, 'critical_label')})
            range_start = num_t_warn
        if num_t_good is not None and num_t_good > range_start:
            gauge_steps.append({'range': [range_start, num_t_good], 'color': config.COLOR_STATUS_WARNING, 'name': get_lang_text(lang, 'warning_label')})
            range_start = num_t_good
        gauge_steps.append({'range': [range_start, axis_max_val], 'color': config.COLOR_STATUS_GOOD, 'name': get_lang_text(lang, 'good_label')})
    if not gauge_steps: gauge_steps.append({'range': [0, axis_max_val], 'color': config.COLOR_NEUTRAL_INFO})
    target_line_val_float = float(target_line_value) if target_line_value is not None and pd.notna(target_line_value) else None
    number_status = get_status_by_thresholds(current_val_numeric, higher_is_worse, num_t_good, num_t_warn)
    value_display_color = get_semaforo_color(number_status) if number_status else config.COLOR_TARGET_LINE
    number_format_str = ".1f" 
    if unit == "%": number_format_str = ".1f" 
    elif pd.notna(current_val_for_display) and isinstance(current_val_for_display, (int, float)) and float(current_val_for_display) == int(current_val_for_display) and abs(float(current_val_for_display)) >=1 :
        number_format_str = ".0f"
    fig = go.Figure(go.Indicator(
        mode="gauge+number" + ("+delta" if delta_obj else ""), value=current_val_for_display,
        title={'text': title_final, 'font': {'size': 13, 'color': config.COLOR_TEXT_SECONDARY}},
        number={'font': {'size': 30, 'color': value_display_color}, 
                'suffix': unit if unit and pd.notna(current_val_for_display) else "", 'valueformat': number_format_str},
        delta=delta_obj if delta_obj else None, 
        gauge={
            'axis': {'range': [0, axis_max_val], 'tickwidth': 1, 'tickcolor': "rgba(0,0,0,0.2)", 'nticks': 5, 'tickfont':{'size':9}},
            'bar': {'color': "rgba(0,0,0,0.8)", 'thickness': 0.15, 'line':{'color':"rgba(0,0,0,1)", 'width':0.5}},
            'bgcolor': "rgba(255,255,255,0.0)", 'borderwidth': 0.5, 'bordercolor': "rgba(0,0,0,0.1)",
            'steps': gauge_steps,
            'threshold': {
                'line': {'color': config.COLOR_TARGET_LINE, 'width': 2}, 
                'thickness': 0.75, 'value': target_line_val_float
            } if target_line_val_float is not None else {}
        }
    ))
    fig.update_layout(height=170, margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor='rgba(0,0,0,0)')
    return fig

# --- Trend Chart Visualization (SME Platinum) ---
def create_trend_chart(df_input: pd.DataFrame, date_col: str,
                       value_cols_map: Dict[str, str], title_key: str, lang: str,
                       y_axis_title_key: str = "value_axis_label", x_axis_title_key: str = "date_time_axis_label",
                       show_average_line: bool = False,
                       target_value_map: Optional[Dict[str, Union[int, float]]] = None,
                       rolling_avg_window: Optional[int] = None,
                       value_col_units_map: Optional[Dict[str, str]] = None,
                       y_axis_format_str: Optional[str] = ",.1f") -> go.Figure:
    df = df_input.copy() 
    title_text = get_lang_text(lang, title_key)
    x_title = get_lang_text(lang, x_axis_title_key)
    y_title = get_lang_text(lang, y_axis_title_key)
    if df.empty or date_col not in df.columns or not value_cols_map:
        return go.Figure().update_layout(title_text=f"{title_text} ({get_lang_text(lang, 'no_data_for_selection')})",
            annotations=[dict(text=get_lang_text(lang, 'no_data_for_selection'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5, font_size=12)])
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    plotted_actual_cols = []
    for i, (legend_key, actual_col) in enumerate(value_cols_map.items()):
        if actual_col not in df.columns or not pd.api.types.is_numeric_dtype(df[actual_col]): continue
        plotted_actual_cols.append(actual_col)
        color = colors[i % len(colors)]
        name = get_lang_text(lang, legend_key, actual_col.replace('_',' ').title())
        unit = value_col_units_map.get(actual_col, "") if value_col_units_map else ""
        y_fmt = y_axis_format_str if y_axis_format_str else ",.2f"
        fig.add_trace(go.Scatter(x=df[date_col], y=df[actual_col], mode='lines+markers', name=name,
            line=dict(color=color, width=2.2), marker=dict(size=6, symbol="circle"),
            hovertemplate=f"<b>{name}</b><br>{get_lang_text(lang, 'date_label', 'Date')}: %{{x|%b %d, %Y}}<br>{y_title}: %{{y:{y_fmt}}}{unit}<extra></extra>"))
    if rolling_avg_window and isinstance(rolling_avg_window, int) and rolling_avg_window > 0:
        for i, actual_col in enumerate(plotted_actual_cols):
            if len(df) >= rolling_avg_window :
                legend_key = next((k for k,v in value_cols_map.items() if v == actual_col), actual_col) 
                base_name = get_lang_text(lang, legend_key)
                ma_name = f"{base_name} ({rolling_avg_window}-p MA)"
                unit = value_col_units_map.get(actual_col, "") if value_col_units_map else ""
                rolling_col_temp = f"__{actual_col}_rolling_avg_temp__" 
                df[rolling_col_temp] = df[actual_col].rolling(window=rolling_avg_window, center=True, min_periods=1).mean()
                fig.add_trace(go.Scatter(x=df[date_col], y=df[rolling_col_temp], mode='lines', name=ma_name,
                    line=dict(color=colors[i % len(colors)], width=1.5, dash='longdashdot'), opacity=0.75,
                    hovertemplate=f"<b>{ma_name}</b><br>{get_lang_text(lang, 'date_label', 'Date')}: %{{x|%b %d, %Y}}<br>{y_title}: %{{y:{y_fmt}}}{unit}<extra></extra>"))
    for i, actual_col in enumerate(plotted_actual_cols):
        legend_key = next((k for k,v in value_cols_map.items() if v == actual_col), actual_col)
        series_name = get_lang_text(lang, legend_key)
        color_for_annotations = colors[i % len(colors)]
        if show_average_line:
            avg = df[actual_col].mean()
            if pd.notna(avg):
                fig.add_hline(y=avg, line_dash="dot", line_color=color_for_annotations, opacity=0.6,
                              annotation_text=f"{get_lang_text(lang, 'average_label')} {series_name}: {avg:{y_fmt if y_fmt else ',.1f'}}",
                              annotation_position="bottom right" if i%2==0 else "top left",
                              annotation_font=dict(size=9, color=color_for_annotations), annotation_bgcolor="rgba(255,255,255,0.8)")
        if target_value_map and actual_col in target_value_map and pd.notna(target_value_map[actual_col]):
            target = target_value_map[actual_col]
            fig.add_hline(y=target, line_dash="solid", line_color=config.COLOR_TARGET_LINE, line_width=1.5, opacity=1.0,
                          annotation_text=f"{get_lang_text(lang, 'target_label')} {series_name}: {target:{y_fmt if y_fmt else ',.1f'}}",
                          annotation_position="top left" if i%2==0 else "bottom right",
                          annotation_font=dict(size=10, color=config.COLOR_TARGET_LINE, family="Arial Black"), annotation_bgcolor="rgba(255,255,255,0.8)")
    fig.update_layout(
        title=dict(text=title_text, x=0.03, y=0.95, xanchor='left', yanchor='top', font_size=17),
        yaxis_title=y_title, xaxis_title=x_title, legend_title_text="", hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=11, bordercolor=config.COLOR_TEXT_SECONDARY, namelength=-1),
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', type='date',
                   showspikes=True, spikemode='across+marker', spikesnap='cursor', spikethickness=1, spikedash='solid', spikecolor='rgba(0,0,0,0.3)',
                   rangeslider_visible= len(df[date_col].unique()) > 15, 
                   rangeselector=dict(buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="todate" if not df.empty and date_col in df.columns and df[date_col].max() > pd.Timestamp.now() - pd.DateOffset(months=1) else "backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"), dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"), dict(count=1, label=get_lang_text(lang, "1y_range_label", "1Y"), step="year", stepmode="backward"),
                        dict(step="all", label=get_lang_text(lang, "all_range_label", "All"))]),
                        font_size=10, bgcolor='rgba(230,230,230,0.7)', borderwidth=1, bordercolor='rgba(0,0,0,0.1)',
                        y=1.18, x=0.01, xanchor='left')),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickformat=(y_axis_format_str if y_axis_format_str else None)),
        legend=dict(orientation="h", yanchor="top", y=1.09, xanchor="right", x=1, font_size=10,traceorder="normal", bgcolor="rgba(255,255,255,0.6)"),
        margin=dict(l=60, r=30, t=100, b=50 if not (not df.empty and date_col in df.columns and len(df[date_col].unique()) > 15) else 80) 
    )
    return fig

# --- Comparison Bar Chart Visualization (Final Fix for insidetextanchor) ---
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
    
    actual_y_cols_for_plotting = []
    plotly_bar_labels_arg = {} 

    for text_key_for_legend, actual_col_name_from_map in y_cols_map.items():
        if actual_col_name_from_map in df.columns and pd.api.types.is_numeric_dtype(df[actual_col_name_from_map]):
            actual_y_cols_for_plotting.append(actual_col_name_from_map)
            plotly_bar_labels_arg[actual_col_name_from_map] = get_lang_text(lang, text_key_for_legend, actual_col_name_from_map.replace('_', ' ').title())

    if df.empty or x_col not in df.columns or not actual_y_cols_for_plotting:
        return go.Figure().update_layout(
            title_text=f"{title_text} ({get_lang_text(lang, 'no_data_for_selection')})",
            annotations=[dict(text=get_lang_text(lang, 'no_data_for_selection'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)]
        )
    
    try:
        fig = px.bar(df, x=x_col, y=actual_y_cols_for_plotting, 
                     title=None, barmode=barmode,
                     color_discrete_sequence=px.colors.qualitative.Pastel if barmode == 'stack' else config.COLOR_SCHEME_CATEGORICAL,
                     labels=plotly_bar_labels_arg,
                     text_auto=True # Enable auto text from px.bar initially
                    )
    except Exception as e_px:
        # print(f"PX.BAR EXCEPTION: {e_px}") # For local debugging
        return go.Figure().update_layout(title_text=f"{title_text} (Chart Gen Error: {str(e_px)[:50]})")
    
    final_fmt_spec = data_label_format_str
    if not (isinstance(final_fmt_spec, str) and final_fmt_spec and (final_fmt_spec.startswith(".") or final_fmt_spec.startswith(","))):
        final_fmt_spec = ".0f" 

    # Refine traces created by px.bar
    for trace in fig.data:
        if hasattr(trace, 'type') and trace.type == 'bar':
            trace.texttemplate = f'%{{y:{final_fmt_spec}}}'
            # Ensure trace.name is available and used correctly. px.bar sets trace.name from the 'y' column or its mapping in 'labels'.
            current_trace_name = trace.name if trace.name else "Value" # Fallback trace name
            trace.hovertemplate = f'<b>%{{x}}</b><br>{current_trace_name}: %{{y:{final_fmt_spec}}}<extra></extra>'
            
            trace.textposition = 'outside' if barmode != 'stack' else 'inside'
            trace.textfont = dict(size=9, color=config.COLOR_TEXT_SECONDARY if barmode == 'stack' else 'black')
            
            # **** CRITICAL FIX for insidetextanchor: ONLY set if textposition is 'inside' ****
            if trace.textposition == 'inside':
                trace.insidetextanchor = 'middle'  # Valid values: 'start', 'middle', 'end'
            elif hasattr(trace, 'insidetextanchor'):
                # If not 'inside', it's safer to let Plotly handle default or remove if it was set to something invalid.
                # However, deleting attributes can be tricky. Omitting 'else' branch is safer here.
                # If a trace object previously had an 'insidetextanchor' from a default, it might persist.
                # Best practice: Do not set insidetextanchor if textposition is not 'inside'.
                # For this attempt, we only set it when 'inside'.
                pass

            if hasattr(trace, 'marker') and hasattr(trace.marker, 'line'):
                trace.marker.line.width = 0.5
                trace.marker.line.color = 'rgba(0,0,0,0.3)'

    if barmode == 'stack' and show_total_for_stacked and actual_y_cols_for_plotting:
        df_for_total_calc = df.copy() 
        df_for_total_calc['_total_stacked_'] = df_for_total_calc[actual_y_cols_for_plotting].sum(axis=1, numeric_only=True)
        annotations_list_total = [
            dict(x=row[x_col], y=row['_total_stacked_'], 
                 text=f"{row['_total_stacked_']:{final_fmt_spec}}",
                 font=dict(size=10, color=config.COLOR_TARGET_LINE), 
                 showarrow=False, yanchor='bottom', yshift=4, xanchor='center')
            for _, row in df_for_total_calc.iterrows() if pd.notna(row['_total_stacked_'])
        ]
        if annotations_list_total:
            current_layout_annotations = list(fig.layout.annotations or ())
            fig.update_layout(annotations=current_layout_annotations + annotations_list_total)

    fig.update_layout(
        title=dict(text=title_text, x=0.03, y=0.97, xanchor='left', yanchor='top', font_size=16),
        yaxis_title=y_title, xaxis_title=x_title,
        legend_title_text="", hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=11, namelength=-1),
        xaxis_tickangle=-30 if not df.empty and df[x_col].nunique() > 7 else 0,
        legend=dict(orientation="h", yanchor="top", y=-0.2 if len(actual_y_cols_for_plotting)>3 else -0.15, xanchor="center", x=0.5, font_size=9),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.08)'),
        xaxis=dict(showgrid=False, type='category', linecolor='rgba(0,0,0,0.2)'),
        bargap=0.2, 
        bargroupgap=0.05 if barmode == 'group' else 0,
        margin=dict(l=50, r=20, t=60, b=100 if len(actual_y_cols_for_plotting)>2 else 70)
    )
    return fig


# --- Metric Card (UX Enhanced - Robust Version) ---
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
        target_format_spec = ".0f" if target_float == int(target_float) and abs(target_float) >=1 else ".1f"
        try: help_text_final_str = raw_help_text_template.format(target=f"{target_float:{target_format_spec}}")
        except (KeyError, ValueError): help_text_final_str = raw_help_text_template 
    val_display_str, delta_text_str, delta_color_str, status_icon_str = "N/A", None, "normal", "❓ "
    if pd.notna(value) and isinstance(value, (int, float, np.number)):
        val_raw_float = float(value)
        if unit == "%": val_display_str = f"{val_raw_float:,.1f}%"
        elif unit == get_lang_text(lang, 'days_unit') or (abs(val_raw_float) >= 1000 and val_raw_float == int(val_raw_float)):
            val_display_str = f"{val_raw_float:,.0f}{(' ' + unit) if unit and unit != '%' else ''}"
        elif abs(val_raw_float) < 1 and val_raw_float != 0 and unit != "%": 
            val_display_str = f"{val_raw_float:,.2f}{(' ' + unit) if unit and unit != '%' else ''}"
        else: val_display_str = f"{val_raw_float:,.1f}{(' ' + unit) if unit and unit != '%' else ''}"
        if unit == "%" and not val_display_str.endswith("%"): val_display_str += "%"
        if previous_value is not None and pd.notna(previous_value) and isinstance(previous_value, (int, float, np.number)):
            prev_val_raw_float = float(previous_value)
            delta_absolute_val = val_raw_float - prev_val_raw_float
            delta_sign_str = "+" if delta_absolute_val >= 1e-9 else ("" if abs(delta_absolute_val) < 1e-9 else "-")
            delta_unit_for_display = unit if unit != '%' else ''
            if unit == "%": delta_abs_formatted = f"{abs(delta_absolute_val):.1f}%"
            elif abs(delta_absolute_val) >= 1000 and delta_absolute_val == int(delta_absolute_val): delta_abs_formatted = f"{abs(delta_absolute_val):,.0f}{(' '+delta_unit_for_display) if delta_unit_for_display else ''}"
            else: delta_abs_formatted = f"{abs(delta_absolute_val):,.1f}{(' '+delta_unit_for_display) if delta_unit_for_display else ''}"
            if abs(prev_val_raw_float) > 1e-9:
                 delta_percent_val = (delta_absolute_val / abs(prev_val_raw_float)) * 100
                 delta_text_str = f"{delta_sign_str}{delta_abs_formatted} ({delta_sign_str}{abs(delta_percent_val):,.0f}%)"
            else: delta_text_str = f"{delta_sign_str}{delta_abs_formatted} ({get_lang_text(lang,'prev_period_label_short','Prev 0')})"
            if higher_is_better is not None:
                if delta_absolute_val > 1e-9: delta_color_str = "normal" if higher_is_better else "inverse"
                elif delta_absolute_val < -1e-9: delta_color_str = "inverse" if higher_is_better else "normal"
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

# --- Radar Chart Visualization (SME Platinum) ---
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
    all_categories_ordered_list = df_radar[categories_col].unique()
    all_r_vals_radar = []
    if values_col in df_radar.columns and not df_radar[values_col].dropna().empty:
        all_r_vals_radar.extend(df_radar[values_col].dropna().tolist())
    if target_values_map: all_r_vals_radar.extend([v for v in target_values_map.values() if pd.notna(v) and isinstance(v,(int,float))])
    valid_r_vals_radar = [float(v) for v in all_r_vals_radar if isinstance(v, (int,float)) and pd.notna(v)]
    max_data_val_for_radar_range = max(valid_r_vals_radar) if valid_r_vals_radar else 0.0
    default_max_scale_radar = config.ENGAGEMENT_RADAR_DIM_SCALE_MAX if pd.notna(config.ENGAGEMENT_RADAR_DIM_SCALE_MAX) else 5.0
    radial_range_max_final = float(range_max_override) if range_max_override is not None and pd.notna(range_max_override) else \
                             (max_data_val_for_radar_range * 1.25 if max_data_val_for_radar_range > 0 else default_max_scale_radar)
    radial_range_max_final = max(radial_range_max_final, 1.0) 
    fig = go.Figure()
    colors_list_radar = px.colors.qualitative.Vivid 
    has_groups_on_radar = group_col and group_col in df_radar.columns and df_radar[group_col].nunique() > 0
    plot_data_exists = False
    if has_groups_on_radar:
        for i, (name_grp_radar_plot, group_data_radar_df) in enumerate(df_radar.groupby(group_col)):
            if not group_data_radar_df.empty and not group_data_radar_df[values_col].dropna().empty: plot_data_exists = True
            current_grp_ordered_df = pd.DataFrame({categories_col: all_categories_ordered_list}).merge(
                group_data_radar_df, on=categories_col, how='left').fillna({values_col: 0})
            fig.add_trace(go.Scatterpolar(
                r=current_grp_ordered_df[values_col], theta=current_grp_ordered_df[categories_col],
                fill='toself', name=str(name_grp_radar_plot), line_color=colors_list_radar[i % len(colors_list_radar)], opacity=fill_opacity,
                hovertemplate='<b>%{theta}</b><br>' + f'{str(name_grp_radar_plot)}: %{{r:.1f}}<extra></extra>' ))
    else:
        if values_col in df_radar.columns and not df_radar[values_col].dropna().empty:
            plot_data_exists = True
            single_series_ordered_df = pd.DataFrame({categories_col: all_categories_ordered_list}).merge(
                df_radar, on=categories_col, how='left').fillna({values_col: 0})
            fig.add_trace(go.Scatterpolar(
                r=single_series_ordered_df[values_col], theta=single_series_ordered_df[categories_col],
                fill='toself', name=get_lang_text(lang, "average_score_label"), line_color=colors_list_radar[0],
                opacity=fill_opacity + 0.15, hovertemplate='<b>%{theta}</b>: %{r:.1f}<extra></extra>'))
    if target_values_map: 
        target_r_values_ordered = [target_values_map.get(cat, 0) for cat in all_categories_ordered_list]
        fig.add_trace(go.Scatterpolar(
            r=target_r_values_ordered, theta=all_categories_ordered_list, mode='lines', name=get_lang_text(lang, "target_label"),
            line=dict(color=config.COLOR_TARGET_LINE, dash='dashdot', width=2.5), hoverinfo='skip')) 
    show_legend_final_radar = has_groups_on_radar or (target_values_map and plot_data_exists)
    fig.update_layout(
        title=dict(text=title_text, x=0.5, y=0.97, yanchor='top', font_size=16), 
        polar=dict(bgcolor="rgba(248,248,248,0.0)", 
                   radialaxis=dict(visible=True, range=[0, radial_range_max_final], showline=True, linecolor='rgba(0,0,0,0.15)', gridcolor="rgba(0,0,0,0.1)", tickfont_size=8, nticks=5, showticklabels=True, layer='below traces'),
                   angularaxis=dict(showline=True, linecolor='rgba(0,0,0,0.15)', gridcolor="rgba(0,0,0,0.05)", tickfont_size=9, direction="clockwise", showticklabels=True, layer='below traces')),
        showlegend=show_legend_final_radar, 
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, font_size=9, itemsizing='constant'), 
        margin=dict(l=30, r=30, t=50, b=70), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

# --- Stress Semáforo Visual (SME Platinum - Corrected title property) ---
def create_stress_semaforo_visual(stress_level_value: Optional[Union[int, float, np.number]], lang: str,
                                  scale_max: float = config.STRESS_LEVEL_PSYCHOSOCIAL["max_scale"]) -> go.Figure:
    display_num_stress, color_for_status_s, text_for_status_s = None, config.COLOR_TEXT_SECONDARY, get_lang_text(lang, 'status_na_label')
    if pd.notna(stress_level_value) and isinstance(stress_level_value, (int, float, np.number)):
        val_float_s_viz = float(stress_level_value)
        display_num_stress = val_float_s_viz 
        status_s_viz = get_status_by_thresholds(val_float_s_viz, higher_is_worse=True,
                                           threshold_good=config.STRESS_LEVEL_PSYCHOSOCIAL["low"],
                                           threshold_warning=config.STRESS_LEVEL_PSYCHOSOCIAL["medium"])
        color_for_status_s = get_semaforo_color(status_s_viz)
        if status_s_viz == "good": text_for_status_s = get_lang_text(lang, 'low_label')
        elif status_s_viz == "warning": text_for_status_s = get_lang_text(lang, 'moderate_label')
        elif status_s_viz == "critical": text_for_status_s = get_lang_text(lang, 'high_label')
        else: text_for_status_s = f"{val_float_s_viz:.1f}" if pd.notna(val_float_s_viz) else get_lang_text(lang, 'status_na_label')
    num_config_s = {'font': {'size': 20, 'color': color_for_status_s}, 'valueformat': ".1f"}
    if display_num_stress is not None: num_config_s['suffix'] = f" / {scale_max:.0f}"
    indicator_title_obj_stress = {
        'text': f"<b style='color:{color_for_status_s}; font-size:1em;'>{text_for_status_s.upper()}</b>", 
        'font': {'size': 12} 
    }
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=display_num_stress, 
        domain={'x': [0.0, 1.0], 'y': [0.0, 0.8]}, 
        title=indicator_title_obj_stress,
        number=num_config_s,
        gauge={
            'shape': "bullet",
            'axis': {'range': [0, scale_max], 'visible': True, 'showticklabels': True, 'layer':'below traces',
                     'tickvals': [0, config.STRESS_LEVEL_PSYCHOSOCIAL["low"], config.STRESS_LEVEL_PSYCHOSOCIAL["medium"], scale_max],
                     'ticktext': ["0", f"{config.STRESS_LEVEL_PSYCHOSOCIAL['low']:.1f}", 
                                  f"{config.STRESS_LEVEL_PSYCHOSOCIAL['medium']:.1f}", f"{scale_max:.0f}"],
                     'tickfont': {'size':8, 'color': config.COLOR_TEXT_SECONDARY}, 'tickmode': 'array'},
            'steps': [ 
                {'range': [0, config.STRESS_LEVEL_PSYCHOSOCIAL["low"]], 'color': "rgba(46, 204, 113, 0.4)"}, 
                {'range': [config.STRESS_LEVEL_PSYCHOSOCIAL["low"], config.STRESS_LEVEL_PSYCHOSOCIAL["medium"]], 'color': "rgba(241, 196, 15, 0.4)"},
                {'range': [config.STRESS_LEVEL_PSYCHOSOCIAL["medium"], scale_max], 'color': "rgba(231, 76, 60, 0.4)"}
            ],
            'bar': {'color': color_for_status_s, 'thickness': 0.4, 'line':{'color':'rgba(0,0,0,0.3)', 'width':0.5}}, 
            'bgcolor': "rgba(255,255,255,0)", 'borderwidth': 0.5, 'bordercolor': "rgba(0,0,0,0.1)"
        }))
    fig.update_layout(height=80, margin=dict(t=10, b=5, l=5, r=5), paper_bgcolor='rgba(0,0,0,0)') 
    return fig
