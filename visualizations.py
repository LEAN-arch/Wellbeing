# visualizations.py
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import config
from typing import List, Dict, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)

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


# --- Centralized Layout Helper (Corrected for ValueError and legend handling) ---
def _apply_standard_layout(fig: go.Figure,
                           lang: str,
                           title_text_direct: str,
                           x_axis_title_key: Optional[str] = None,
                           y_axis_title_key: Optional[str] = None,
                           legend_params: Optional[Dict[str, Any]] = None,
                           margin_params: Optional[Dict[str, int]] = None,
                           extra_layout_updates: Optional[Dict[str, Any]] = None) -> None:
    layout_settings = {
        "title": dict(text=title_text_direct, x=0.03, y=0.97,
                      xanchor='left', yanchor='top',
                      font=dict(size=config.FONT_SIZE_TITLE_DEFAULT,
                                family=config.FONT_FAMILY_DEFAULT)),
        "paper_bgcolor": config.COLOR_PAPER_BACKGROUND,
        "plot_bgcolor": config.COLOR_PLOT_BACKGROUND,
        "font": dict(family=config.FONT_FAMILY_DEFAULT,
                     size=config.FONT_SIZE_BODY_DEFAULT,
                     color=config.COLOR_TEXT_PRIMARY),
        "hovermode": "x unified",
        "hoverlabel": dict(
            bgcolor=config.COLOR_HOVER_LABEL_BACKGROUND,
            font_size=config.FONT_SIZE_HOVER_LABEL,
            bordercolor=config.COLOR_TEXT_SECONDARY,
            namelength=-1
        ),
        "margin": margin_params if margin_params is not None else config.DEFAULT_CHART_MARGINS,
        "xaxis": {"title": {}}, # Initialize to allow merging/setting title.text
        "yaxis": {"title": {}}  # Initialize to allow merging/setting title.text
    }

    if x_axis_title_key:
        layout_settings["xaxis"]["title"]["text"] = get_lang_text(lang, x_axis_title_key, x_axis_title_key)
        layout_settings["xaxis"]["gridcolor"] = config.COLOR_GRID_SECONDARY
        layout_settings["xaxis"]["linecolor"] = config.COLOR_AXIS_LINE
        layout_settings["xaxis"]["zerolinecolor"] = config.COLOR_GRID_SECONDARY
        layout_settings["xaxis"]["zerolinewidth"] = 1
    if y_axis_title_key:
        layout_settings["yaxis"]["title"]["text"] = get_lang_text(lang, y_axis_title_key, y_axis_title_key)
        layout_settings["yaxis"]["gridcolor"] = config.COLOR_GRID_PRIMARY
        layout_settings["yaxis"]["linecolor"] = config.COLOR_AXIS_LINE
        layout_settings["yaxis"]["zerolinecolor"] = config.COLOR_GRID_PRIMARY
        layout_settings["yaxis"]["zerolinewidth"] = 1

    show_legend_flag = True
    if legend_params is not None:
        if "showlegend" in legend_params:
            show_legend_flag = legend_params.pop("showlegend")
        if show_legend_flag:
            final_legend_settings = {
                "orientation": "h", "yanchor": "top", "y": -0.15,
                "xanchor": "center", "x": 0.5,
                "font_size": config.FONT_SIZE_LEGEND,
                "traceorder": "normal",
                "bgcolor": config.COLOR_LEGEND_BACKGROUND,
                "bordercolor": config.COLOR_LEGEND_BORDER,
                "borderwidth": 1
            }
            final_legend_settings.update(legend_params)
            layout_settings["legend"] = final_legend_settings
    layout_settings["showlegend"] = show_legend_flag

    if extra_layout_updates:
        for key, val in extra_layout_updates.items():
            if isinstance(val, dict) and isinstance(layout_settings.get(key), dict):
                current_sub_dict = layout_settings.get(key, {})
                current_sub_dict.update(val)
                layout_settings[key] = current_sub_dict
            else:
                layout_settings[key] = val
    try:
        fig.update_layout(**layout_settings)
    except Exception as e:
        logger.error(f"Error applying layout settings in _apply_standard_layout: {e}")
        logger.error(f"Layout settings dump for problematic chart ('{title_text_direct}'): {layout_settings}")
        raise


# --- Standardized No Data Figure ---
def _create_no_data_figure(lang: str, title_key_for_base: str,
                           message_key: str = "no_data_for_selection") -> go.Figure:
    fig = go.Figure()
    no_data_message = get_lang_text(lang, message_key, "No data available")
    base_title = get_lang_text(lang, title_key_for_base, title_key_for_base)
    title_with_no_data_msg = f"{base_title} ({no_data_message})"
    _apply_standard_layout(
        fig, lang, title_text_direct=title_with_no_data_msg,
        extra_layout_updates={
            "title": dict(x=0.5, y=0.9, xanchor='center', yanchor='middle',
                          font_size=config.FONT_SIZE_TITLE_DEFAULT - 2),
            "xaxis_visible": False,
            "yaxis_visible": False,
            "showlegend": False,
            "plot_bgcolor": config.COLOR_PAPER_BACKGROUND,
        }
    )
    fig.add_annotation(
        text=no_data_message, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=config.FONT_SIZE_BODY_DEFAULT + 2, color=config.COLOR_TEXT_SECONDARY)
    )
    return fig

# --- KPI Gauge Visualization ---
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
        title_final = f"{title_base}<br><span style='font-size:{config.FONT_SIZE_SUBTITLE}px;color:{config.COLOR_TEXT_SECONDARY};font-weight:normal;'>{subtitle_str}</span>"
    current_val_numeric = float(value) if pd.notna(value) and isinstance(value, (int,float,np.number)) else 0.0
    current_val_for_display = float(value) if pd.notna(value) and isinstance(value, (int,float,np.number)) else None
    delta_obj = {}
    if previous_value is not None and pd.notna(previous_value) and isinstance(previous_value, (int, float, np.number)) and current_val_for_display is not None:
        delta_ref_val = float(previous_value)
        delta_val = current_val_numeric - delta_ref_val
        increasing_color = config.COLOR_STATUS_CRITICAL if higher_is_worse else config.COLOR_STATUS_GOOD
        decreasing_color = config.COLOR_STATUS_GOOD if higher_is_worse else config.COLOR_STATUS_CRITICAL
        delta_font_color = config.COLOR_TEXT_SECONDARY
        if abs(delta_val) > config.EPSILON : delta_font_color = increasing_color if delta_val > 0 else decreasing_color
        delta_obj = {
            'reference': delta_ref_val,
            'increasing': {'color': increasing_color, 'symbol': "▲"},
            'decreasing': {'color': decreasing_color, 'symbol': "▼"},
            'font': {'size': config.FONT_SIZE_GAUGE_DELTA, 'color': delta_font_color}
        }
    if max_value_override is not None and pd.notna(max_value_override):
        axis_max_val = float(max_value_override)
    else:
        val_candidates_for_max = [1.0]
        if pd.notna(current_val_numeric): val_candidates_for_max.append(abs(current_val_numeric) * 1.4)
        ref_points_for_max = [threshold_good, threshold_warning, target_line_value]
        valid_ref_points_for_max = [float(p) for p in ref_points_for_max if p is not None and pd.notna(p)]
        if valid_ref_points_for_max: val_candidates_for_max.append(max(valid_ref_points_for_max) * 1.25)
        if not valid_ref_points_for_max and (not pd.notna(current_val_numeric) or abs(current_val_numeric) < config.EPSILON):
             val_candidates_for_max.append(100.0 if unit == "%" else 10.0)
        axis_max_val = max(val_candidates_for_max) if val_candidates_for_max else 100.0
        if axis_max_val <= (abs(current_val_numeric) if pd.notna(current_val_numeric) else 0.0):
            axis_max_val = (abs(current_val_numeric) * 1.1) if pd.notna(current_val_numeric) and abs(current_val_numeric) > config.EPSILON else (axis_max_val * 1.1 or 10.0)
        if axis_max_val <= config.EPSILON: axis_max_val = 10.0
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
        if range_start < axis_max_val : gauge_steps.append({'range': [range_start, axis_max_val], 'color': config.COLOR_STATUS_CRITICAL, 'name': get_lang_text(lang, 'critical_label')})
    else:
        if num_t_warn is not None:
            gauge_steps.append({'range': [range_start, num_t_warn], 'color': config.COLOR_STATUS_CRITICAL, 'name': get_lang_text(lang, 'critical_label')})
            range_start = num_t_warn
        if num_t_good is not None and num_t_good > range_start:
            gauge_steps.append({'range': [range_start, num_t_good], 'color': config.COLOR_STATUS_WARNING, 'name': get_lang_text(lang, 'warning_label')})
            range_start = num_t_good
        if range_start < axis_max_val: gauge_steps.append({'range': [range_start, axis_max_val], 'color': config.COLOR_STATUS_GOOD, 'name': get_lang_text(lang, 'good_label')})
    if not gauge_steps and axis_max_val > config.EPSILON : gauge_steps.append({'range': [0, axis_max_val], 'color': config.COLOR_NEUTRAL_INFO})
    target_line_val_float = float(target_line_value) if target_line_value is not None and pd.notna(target_line_value) else None
    number_status = get_status_by_thresholds(current_val_numeric, higher_is_worse, num_t_good, num_t_warn)
    value_display_color = get_semaforo_color(number_status) if number_status else config.COLOR_TARGET_LINE
    number_format_str = ".1f"
    if unit == "%": number_format_str = ".1f"
    elif pd.notna(current_val_for_display) and isinstance(current_val_for_display, (int, float)) and \
         abs(float(current_val_for_display) - int(current_val_for_display)) < config.EPSILON and abs(float(current_val_for_display)) >=1 :
        number_format_str = ".0f"
    fig = go.Figure(go.Indicator(
        mode="gauge+number" + ("+delta" if delta_obj else ""), value=current_val_for_display,
        title={'text': title_final, 'font': {'size': config.FONT_SIZE_TITLE_GAUGE, 'color': config.COLOR_TEXT_SECONDARY}},
        number={'font': {'size': config.FONT_SIZE_GAUGE_NUMBER, 'color': value_display_color},
                'suffix': unit if unit and pd.notna(current_val_for_display) else "", 'valueformat': number_format_str},
        delta=delta_obj if delta_obj else None,
        gauge={
            'axis': {'range': [0, axis_max_val], 'tickwidth': 1, 'tickcolor': config.COLOR_GAUGE_TICK, 'nticks': 5,
                     'tickfont':{'size': config.FONT_SIZE_AXIS_TICKS_GAUGE}},
            'bar': {'color': config.COLOR_GAUGE_NEEDLE_BASE, 'thickness': 0.15,
                    'line':{'color':config.COLOR_GAUGE_NEEDLE_BORDER, 'width':0.5}},
            'bgcolor': config.COLOR_GAUGE_BACKGROUND,
            'borderwidth': 0.5, 'bordercolor': config.COLOR_GAUGE_BORDERCOLOR,
            'steps': gauge_steps,
            'threshold': {'line': {'color': config.COLOR_TARGET_LINE, 'width': 2},
                          'thickness': 0.75, 'value': target_line_val_float
            } if target_line_val_float is not None else {}
        }
    ))
    fig.update_layout(height=170, margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor=config.COLOR_PAPER_BACKGROUND)
    return fig

# --- Trend Chart Visualization ---
def create_trend_chart(df_input: pd.DataFrame, date_col: str,
                       value_cols_map: Dict[str, str], title_key: str, lang: str,
                       y_axis_title_key: str = "value_axis_label", x_axis_title_key: str = "date_time_axis_label",
                       show_average_line: bool = False,
                       target_value_map: Optional[Dict[str, Union[int, float]]] = None,
                       rolling_avg_window: Optional[int] = None,
                       value_col_units_map: Optional[Dict[str, str]] = None,
                       y_axis_format_str: Optional[str] = ",.1f") -> go.Figure:
    df = df_input.copy()
    if df.empty or date_col not in df.columns or not value_cols_map:
        return _create_no_data_figure(lang, title_key)
    fig = go.Figure()
    colors = config.COLOR_SCHEME_CATEGORICAL_SET2
    y_title_str = get_lang_text(lang, y_axis_title_key, "Value")
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
            hovertemplate=f"<b>{name}</b><br>{get_lang_text(lang, 'date_label', 'Date')}: %{{x|%b %d, %Y}}<br>{y_title_str}: %{{y:{y_fmt}}}{unit}<extra></extra>"))
    if rolling_avg_window and isinstance(rolling_avg_window, int) and rolling_avg_window > 0:
        for i, actual_col in enumerate(plotted_actual_cols):
            if len(df) >= rolling_avg_window :
                legend_key = next((k for k,v in value_cols_map.items() if v == actual_col), actual_col)
                base_name = get_lang_text(lang, legend_key, actual_col.replace('_',' ').title())
                ma_name = f"{base_name} ({rolling_avg_window}-p MA)"
                unit = value_col_units_map.get(actual_col, "") if value_col_units_map else ""
                rolling_col_temp = f"__{actual_col}_rolling_avg_temp__"
                df[rolling_col_temp] = df[actual_col].rolling(window=rolling_avg_window, center=True, min_periods=1).mean()
                fig.add_trace(go.Scatter(x=df[date_col], y=df[rolling_col_temp], mode='lines', name=ma_name,
                    line=dict(color=colors[i % len(colors)], width=1.5, dash='longdashdot'), opacity=0.75,
                    hovertemplate=f"<b>{ma_name}</b><br>{get_lang_text(lang, 'date_label', 'Date')}: %{{x|%b %d, %Y}}<br>{y_title_str}: %{{y:{y_fmt}}}{unit}<extra></extra>"))
    for i, actual_col in enumerate(plotted_actual_cols):
        legend_key = next((k for k,v in value_cols_map.items() if v == actual_col), actual_col)
        series_name = get_lang_text(lang, legend_key, actual_col.replace('_',' ').title())
        color_for_annotations = colors[i % len(colors)]
        y_fmt = y_axis_format_str if y_axis_format_str else ",.1f"
        if show_average_line:
            avg = df[actual_col].mean()
            if pd.notna(avg):
                fig.add_hline(y=avg, line_dash="dot", line_color=color_for_annotations, opacity=0.6,
                              annotation_text=f"{get_lang_text(lang, 'average_label')} {series_name}: {avg:{y_fmt}}",
                              annotation_position="bottom right" if i%2==0 else "top left",
                              annotation_font=dict(size=config.FONT_SIZE_ANNOTATION_SMALL, color=color_for_annotations),
                              annotation_bgcolor=config.COLOR_ANNOTATION_BG)
        if target_value_map and actual_col in target_value_map and pd.notna(target_value_map[actual_col]):
            target = target_value_map[actual_col]
            fig.add_hline(y=target, line_dash="solid", line_color=config.COLOR_TARGET_LINE, line_width=1.5, opacity=1.0,
                          annotation_text=f"{get_lang_text(lang, 'target_label')} {series_name}: {target:{y_fmt}}",
                          annotation_position="top left" if i%2==0 else "bottom right",
                          annotation_font=dict(size=config.FONT_SIZE_ANNOTATION_TARGET, color=config.COLOR_TARGET_LINE,
                                               family=config.FONT_FAMILY_TARGET_ANNOTATION),
                          annotation_bgcolor=config.COLOR_ANNOTATION_BG)
    title_text_direct = get_lang_text(lang, title_key)
    margin_params = dict(l=60, r=30, t=100, b=50 if not (len(df[date_col].unique()) > 15) else 80)
    legend_params_trend = { "showlegend": True, "orientation": "h", "yanchor": "top", "y": 1.09, "xanchor": "right", "x": 1, "title_text": ""}
    _apply_standard_layout(fig, lang, title_text_direct=title_text_direct,
                           x_axis_title_key=x_axis_title_key, y_axis_title_key=y_axis_title_key,
                           legend_params=legend_params_trend, margin_params=margin_params)
    fig.update_layout( # Specific updates for this chart type
        xaxis_gridcolor=config.COLOR_GRID_SECONDARY, # gridcolor is under xaxis/yaxis directly
        yaxis_gridcolor=config.COLOR_GRID_PRIMARY,
        yaxis_tickformat=(y_axis_format_str if y_axis_format_str else None)
    )
    fig.update_xaxes( # Use update_xaxes to modify the existing xaxis object
        type='date',
        showspikes=True,
        spikemode='across+marker',
        spikesnap='cursor',
        spikethickness=1,
        spikedash='solid',
        spikecolor=config.COLOR_SPIKE_LINE,
        rangeslider_visible=len(df[date_col].unique()) > 15 if date_col in df.columns and not df.empty else False,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="todate" if not df.empty and date_col in df.columns and not df[date_col].empty and df[date_col].max() > pd.Timestamp.now() - pd.DateOffset(months=1) else "backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"), dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"), dict(count=1, label=get_lang_text(lang, "1y_range_label", "1Y"), step="year", stepmode="backward"),
                dict(step="all", label=get_lang_text(lang, "all_range_label", "All"))
            ]),
            font_size=config.FONT_SIZE_RANGESELECTOR_BUTTONS,
            bgcolor=config.COLOR_RANGESELECTOR_BACKGROUND,
            borderwidth=1, bordercolor=config.COLOR_RANGESELECTOR_BORDER,
            y=1.18, x=0.01, xanchor='left'
        ) if date_col in df.columns and not df.empty else None
    )
    return fig

# --- Comparison Bar Chart Visualization ---
def create_comparison_bar_chart(df_input: pd.DataFrame, x_col: str,
                                y_cols_map: Dict[str, str],
                                title_key: str, lang: str,
                                y_axis_title_key: str = "count_label", x_axis_title_key: str = "category_axis_label",
                                barmode: str = 'group', show_total_for_stacked: bool = False,
                                data_label_format_str: str = ".0f") -> go.Figure:
    df = df_input.copy()
    actual_y_cols_for_plotting = []
    plotly_bar_labels_arg = {}
    for text_key_for_legend, actual_col_name_from_map in y_cols_map.items():
        if actual_col_name_from_map in df.columns and pd.api.types.is_numeric_dtype(df[actual_col_name_from_map]):
            actual_y_cols_for_plotting.append(actual_col_name_from_map)
            plotly_bar_labels_arg[actual_col_name_from_map] = get_lang_text(lang, text_key_for_legend, actual_col_name_from_map.replace('_', ' ').title())
    if df.empty or x_col not in df.columns or not actual_y_cols_for_plotting:
        return _create_no_data_figure(lang, title_key)
    try:
        fig = px.bar(df, x=x_col, y=actual_y_cols_for_plotting, title=None, barmode=barmode,
                     color_discrete_sequence=px.colors.qualitative.Pastel if barmode == 'stack' else config.COLOR_SCHEME_CATEGORICAL,
                     labels=plotly_bar_labels_arg, text_auto=False)
    except Exception as e_px:
        logger.error(f"PX.BAR EXCEPTION in create_comparison_bar_chart for '{title_key}': {e_px}", exc_info=True)
        return _create_no_data_figure(lang, title_key, message_key="chart_generation_error_label")
    final_fmt_spec = data_label_format_str
    if not (isinstance(final_fmt_spec, str) and final_fmt_spec and (final_fmt_spec.startswith(".") or final_fmt_spec.startswith(","))):
        final_fmt_spec = ".0f"
    for trace in fig.data:
        if hasattr(trace, 'type') and trace.type == 'bar':
            trace.texttemplate = f'%{{y:{final_fmt_spec}}}'
            trace_name_for_hover = trace.name if trace.name else "Value"
            trace.hovertemplate = f'<b>%{{x}}</b><br>{trace_name_for_hover}: %{{y:{final_fmt_spec}}}<extra></extra>'
            current_text_position = 'outside' if barmode != 'stack' else 'inside'
            trace.textposition = current_text_position
            trace.textfont = dict(size=config.FONT_SIZE_BAR_TEXT,
                                  color=config.COLOR_BAR_TEXT_INSIDE if current_text_position == 'inside' else config.COLOR_BAR_TEXT_OUTSIDE)
            if current_text_position == 'inside': trace.insidetextanchor = 'middle'
            if hasattr(trace, 'marker') and hasattr(trace.marker, 'line'):
                trace.marker.line.width = 0.5
                trace.marker.line.color = config.COLOR_BAR_MARKER_BORDER
    if barmode == 'stack' and show_total_for_stacked and actual_y_cols_for_plotting:
        df_total_sum_calc = df.copy()
        df_total_sum_calc['_total_stacked_'] = df_total_sum_calc[actual_y_cols_for_plotting].sum(axis=1, numeric_only=True)
        annotations_list_total = [
            dict(x=row[x_col], y=row['_total_stacked_'], text=f"{row['_total_stacked_']:{final_fmt_spec}}",
                 font=dict(size=config.FONT_SIZE_ANNOTATION_SMALL, color=config.COLOR_TARGET_LINE),
                 showarrow=False, yanchor='bottom', yshift=4, xanchor='center')
            for _, row in df_total_sum_calc.iterrows() if pd.notna(row['_total_stacked_'])
        ]
        if annotations_list_total:
            fig.update_layout(annotations=list(fig.layout.annotations or []) + annotations_list_total)
    title_text_direct = get_lang_text(lang, title_key)
    margin_params = dict(l=50, r=20, t=60, b=100 if len(actual_y_cols_for_plotting)>2 else 70)
    legend_params_bar = {"showlegend": True, "orientation":"h", "yanchor":"top", "y":-0.2 if len(actual_y_cols_for_plotting)>3 else -0.15,
                         "xanchor":"center", "x":0.5, "title_text":""}
    _apply_standard_layout(fig, lang, title_text_direct=title_text_direct,
                           x_axis_title_key=x_axis_title_key, y_axis_title_key=y_axis_title_key,
                           legend_params=legend_params_bar, margin_params=margin_params)
    fig.update_layout(
        xaxis_tickangle=-30 if not df.empty and x_col in df.columns and df[x_col].nunique() > 7 else 0,
        yaxis_gridcolor=config.COLOR_GRID_PRIMARY,
        xaxis_type='category',
        xaxis_showgrid=False,
        xaxis_linecolor=config.COLOR_AXIS_LINE,
        bargap=0.2, bargroupgap=0.05 if barmode == 'group' else 0,
    )
    return fig

# --- Metric Card ---
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
        target_format_spec = ".0f" if abs(target_float - int(target_float)) < config.EPSILON and abs(target_float) >=1 else ".1f"
        try: help_text_final_str = raw_help_text_template.format(target=f"{target_float:{target_format_spec}}")
        except (KeyError, ValueError) as e:
             logger.warning(f"Error formatting help text for key '{help_text_key}' with target: {e}")
             help_text_final_str = raw_help_text_template
    val_display_str, delta_text_str, delta_color_str, status_icon_str = get_lang_text(lang,"status_na_label","N/A"), None, "normal", "❓ "
    if pd.notna(value) and isinstance(value, (int, float, np.number)):
        val_raw_float = float(value)
        if unit == "%": val_display_str = f"{val_raw_float:,.1f}%"
        elif unit == get_lang_text(lang, 'days_unit') or (abs(val_raw_float) >= 1000 and abs(val_raw_float - int(val_raw_float)) < config.EPSILON):
            val_display_str = f"{val_raw_float:,.0f}{(' ' + unit) if unit and unit != '%' else ''}"
        elif abs(val_raw_float) < 1 and val_raw_float != 0 and unit != "%":
            val_display_str = f"{val_raw_float:,.2f}{(' ' + unit) if unit and unit != '%' else ''}"
        else: val_display_str = f"{val_raw_float:,.1f}{(' ' + unit) if unit and unit != '%' else ''}"
        if unit == "%" and not val_display_str.endswith("%"): val_display_str += "%"
        if previous_value is not None and pd.notna(previous_value) and isinstance(previous_value, (int, float, np.number)):
            prev_val_raw_float = float(previous_value)
            delta_absolute_val = val_raw_float - prev_val_raw_float
            delta_sign_str = "+" if delta_absolute_val >= config.EPSILON else ("" if abs(delta_absolute_val) < config.EPSILON else "-")
            delta_unit_for_display = unit if unit != '%' else ''
            delta_abs_formatted = ""
            if unit == "%": delta_abs_formatted = f"{abs(delta_absolute_val):.1f}%"
            elif abs(delta_absolute_val) >= 1000 and abs(delta_absolute_val - int(delta_absolute_val)) < config.EPSILON:
                delta_abs_formatted = f"{abs(delta_absolute_val):,.0f}{(' '+delta_unit_for_display) if delta_unit_for_display else ''}"
            else: delta_abs_formatted = f"{abs(delta_absolute_val):,.1f}{(' '+delta_unit_for_display) if delta_unit_for_display else ''}"
            if abs(prev_val_raw_float) > config.EPSILON:
                 delta_percent_val = (delta_absolute_val / abs(prev_val_raw_float)) * 100
                 delta_text_str = f"{delta_sign_str}{delta_abs_formatted} ({delta_sign_str}{abs(delta_percent_val):,.0f}%)"
            else: delta_text_str = f"{delta_sign_str}{delta_abs_formatted} ({get_lang_text(lang,'prev_period_label_short','Prev 0')})"
            if higher_is_better is not None:
                if delta_absolute_val > config.EPSILON: delta_color_str = "normal" if higher_is_better else "inverse"
                elif delta_absolute_val < -config.EPSILON: delta_color_str = "inverse" if higher_is_better else "normal"
                else: delta_color_str = "off"
        status_logic_higher_is_worse = not higher_is_better if higher_is_better is not None else True
        current_status_text = get_status_by_thresholds(val_raw_float, status_logic_higher_is_worse, threshold_good, threshold_warning)
        if current_status_text == "good": status_icon_str = "✅ "
        elif current_status_text == "warning": status_icon_str = "⚠️ "
        elif current_status_text == "critical": status_icon_str = "❗ "
        elif target_value is not None and higher_is_better is not None and pd.notna(target_value):
            target_float_comp = float(target_value)
            if (higher_is_better and val_raw_float >= target_float_comp - config.EPSILON) or \
               (not higher_is_better and val_raw_float <= target_float_comp + config.EPSILON):
                status_icon_str = "🎯 "
            else: status_icon_str = ""
        else: status_icon_str = ""
    st_object.metric(label=status_icon_str + label_text_orig, value=val_display_str, delta=delta_text_str, delta_color=delta_color_str, help=help_text_final_str)

# --- Radar Chart Visualization (Enhanced Visibility) ---
def create_enhanced_radar_chart(df_radar_input: pd.DataFrame, categories_col: str, values_col: str,
                                 title_key: str, lang: str, group_col: Optional[str] = None,
                                 range_max_override: Optional[Union[int, float]] = None,
                                 target_values_map: Optional[Dict[str, Union[int, float]]] = None,
                                 fill_opacity: float = config.RADAR_FILL_OPACITY,
                                 line_width: float = 2.5
                                 ) -> go.Figure:
    df_radar = df_radar_input.copy()
    if df_radar.empty or categories_col not in df_radar.columns or values_col not in df_radar.columns or df_radar[categories_col].nunique() == 0:
        return _create_no_data_figure(lang, title_key, message_key="no_data_radar")

    all_categories_ordered_list = df_radar[categories_col].unique()
    all_r_vals_radar = []
    if values_col in df_radar.columns and not df_radar[values_col].dropna().empty:
        all_r_vals_radar.extend(df_radar[values_col].dropna().tolist())
    if target_values_map: all_r_vals_radar.extend([v for v in target_values_map.values() if pd.notna(v) and isinstance(v,(int,float))])

    valid_r_vals_radar = [float(v) for v in all_r_vals_radar if isinstance(v, (int,float)) and pd.notna(v)]
    max_data_val_for_radar_range = max(valid_r_vals_radar) if valid_r_vals_radar else 0.0

    default_max_scale_radar = config.ENGAGEMENT_RADAR_DIM_SCALE_MAX
    if not (isinstance(default_max_scale_radar, (int, float)) and pd.notna(default_max_scale_radar)):
        default_max_scale_radar = 5.0

    radial_range_max_final = float(range_max_override) if range_max_override is not None and pd.notna(range_max_override) else \
                             (max_data_val_for_radar_range * 1.25 if max_data_val_for_radar_range > config.EPSILON else default_max_scale_radar)
    radial_range_max_final = max(radial_range_max_final, 1.0)

    fig = go.Figure()
    colors_list_radar = config.COLOR_SCHEME_RADAR_DEFAULT

    has_groups_on_radar = group_col and group_col in df_radar.columns and df_radar[group_col].nunique() > 0
    plot_data_exists = False

    all_teams_name_variants = ["ALL TEAM", "ALL TEAMS", "AVERAGE", "OVERALL", "GLOBAL", "PROMEDIO", "TODOS"]
    primary_group_name = None
    if has_groups_on_radar:
        unique_groups = df_radar[group_col].unique()
        for variant in all_teams_name_variants:
            matching_groups = [g for g in unique_groups if str(g).strip().upper() == variant.upper()]
            if matching_groups:
                primary_group_name = matching_groups[0]
                break

    plotted_groups_color_idx = 0

    if has_groups_on_radar:
        group_names_to_plot = list(df_radar[group_col].unique())
        if primary_group_name and primary_group_name in group_names_to_plot:
            group_names_to_plot.remove(primary_group_name)
            group_names_to_plot.insert(0, primary_group_name)

        for name_grp_radar_plot in group_names_to_plot:
            group_data_radar_df = df_radar[df_radar[group_col] == name_grp_radar_plot]
            if not group_data_radar_df.empty and not group_data_radar_df[values_col].dropna().empty:
                plot_data_exists = True
                current_grp_ordered_df = pd.DataFrame({categories_col: all_categories_ordered_list}).merge(
                    group_data_radar_df, on=categories_col, how='left').fillna({values_col: 0})

                is_primary = (name_grp_radar_plot == primary_group_name)
                current_line_width = line_width + 0.75 if is_primary else line_width
                current_color = config.COLOR_RADAR_PRIMARY_TRACE if is_primary else colors_list_radar[plotted_groups_color_idx % len(colors_list_radar)]
                current_fill_opacity = fill_opacity + 0.1 if is_primary else fill_opacity

                fig.add_trace(go.Scatterpolar(
                    r=current_grp_ordered_df[values_col], theta=current_grp_ordered_df[categories_col],
                    fill='toself', name=str(name_grp_radar_plot),
                    line=dict(color=current_color, width=current_line_width),
                    fillcolor=current_color,
                    opacity=current_fill_opacity,
                    hovertemplate='<b>%{theta}</b><br>' + f'{str(name_grp_radar_plot)}: %{{r:.1f}}<extra></extra>' ))
                if not is_primary:
                    plotted_groups_color_idx += 1
    else:
        if values_col in df_radar.columns and not df_radar[values_col].dropna().empty:
            plot_data_exists = True
            single_series_ordered_df = pd.DataFrame({categories_col: all_categories_ordered_list}).merge(
                df_radar, on=categories_col, how='left').fillna({values_col: 0})
            fig.add_trace(go.Scatterpolar(
                r=single_series_ordered_df[values_col], theta=single_series_ordered_df[categories_col],
                fill='toself', name=get_lang_text(lang, "average_score_label"),
                line=dict(color=config.COLOR_RADAR_PRIMARY_TRACE, width=line_width + 0.75),
                fillcolor=config.COLOR_RADAR_PRIMARY_TRACE,
                opacity=fill_opacity + 0.1,
                hovertemplate='<b>%{theta}</b>: %{r:.1f}<extra></extra>'))

    if target_values_map:
        target_r_values_ordered = [target_values_map.get(cat, 0) for cat in all_categories_ordered_list]
        fig.add_trace(go.Scatterpolar(
            r=target_r_values_ordered, theta=all_categories_ordered_list, mode='lines', name=get_lang_text(lang, "target_label"),
            line=dict(color=config.COLOR_RADAR_TARGET_LINE, dash='dashdot', width=line_width - 0.5),
            hoverinfo='skip'))

    show_legend_final_radar = has_groups_on_radar or (target_values_map and plot_data_exists)

    title_text_direct = get_lang_text(lang, title_key)
    margin_bottom = 120 if show_legend_final_radar and len(all_categories_ordered_list) > 5 else (90 if show_legend_final_radar else 70)
    margin_params = dict(l=50, r=50, t=70, b=margin_bottom)
    legend_params_radar = {
        "showlegend": show_legend_final_radar,
        "orientation":"h", "yanchor":"bottom", "y": -0.45, "xanchor":"center", "x":0.5,
        "font_size": config.FONT_SIZE_LEGEND,
        "itemsizing": 'constant',
        "title_text": get_lang_text(lang, "metrics_legend", "Legend") if has_groups_on_radar and show_legend_final_radar else "",
        "tracegroupgap": 10
    }
    _apply_standard_layout(fig, lang, title_text_direct=title_text_direct,
                           legend_params=legend_params_radar, margin_params=margin_params,
                           extra_layout_updates={"title": {"x":0.5, "xanchor":"center"}})

    fig.update_layout(
        polar=dict(
            bgcolor=config.COLOR_RADAR_POLAR_BACKGROUND,
            radialaxis=dict(
                visible=True, range=[0, radial_range_max_final], showline=True,
                linecolor=config.COLOR_RADAR_AXIS_LINE,
                gridcolor=config.COLOR_RADAR_GRID_LINE,
                tickfont=dict(size=config.FONT_SIZE_RADAR_TICK, color=config.COLOR_RADAR_TICK_LABEL),
                angle=90,
                nticks=max(3, int(radial_range_max_final / (1 if radial_range_max_final <=5 else 2) )) if radial_range_max_final > 0 else 3 ,
                showticklabels=True, layer='below traces'
            ),
            angularaxis=dict(
                showline=True, linecolor=config.COLOR_RADAR_AXIS_LINE,
                gridcolor=config.COLOR_RADAR_ANGULAR_GRID_LINE,
                tickfont=dict(size=config.FONT_SIZE_RADAR_ANGULAR_TICK, color=config.COLOR_RADAR_TICK_LABEL),
                direction="clockwise",
                showticklabels=True, layer='below traces',
            )
        )
    )
    return fig

# --- Facility Heatmap Visualization ---
def create_metric_density_heatmap(
    df_input: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    title_key: str,
    lang: str,
    aggregation_func: str = "avg",
    xbins: int = config.HEATMAP_NBINSX_DEFAULT,
    ybins: int = config.HEATMAP_NBINSY_DEFAULT,
    colorscale: str = config.HEATMAP_COLORSCALE_DEFAULT,
    colorbar_title_key: str = "value_axis_label",
    show_points: bool = config.HEATMAP_SHOW_POINTS_OVERLAY,
    point_size: int = config.HEATMAP_POINT_SIZE,
    point_opacity: float = config.HEATMAP_POINT_OPACITY,
    facility_dimensions: Optional[Dict[str, float]] = None,
    entry_exit_points: Optional[List[Dict[str, Any]]] = None
) -> go.Figure:

    df = df_input.copy()
    title_text = get_lang_text(lang, title_key)

    if df.empty or not all(c in df.columns for c in [x_col, y_col, z_col]):
        logger.warning(f"Metric Heatmap: Missing required columns. Got x:'{x_col}', y:'{y_col}', z:'{z_col}' from available: {list(df.columns)}")
        return _create_no_data_figure(lang, title_key, message_key="heatmap_no_coordinate_data")

    df.dropna(subset=[x_col, y_col, z_col], inplace=True)
    if df.empty:
        logger.warning(f"Metric Heatmap: No valid data after NaNs drop for {x_col}, {y_col}, {z_col}")
        return _create_no_data_figure(lang, title_key, message_key="heatmap_no_value_data")

    try:
        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        df[z_col] = pd.to_numeric(df[z_col], errors='coerce')
        df.dropna(subset=[x_col, y_col, z_col], inplace=True)
        if df.empty: raise ValueError("All data became NaN after numeric conversion for heatmap.")
    except ValueError as e:
        logger.error(f"Metric Heatmap: Error converting columns to numeric: {e}")
        return _create_no_data_figure(lang, title_key, message_key="chart_generation_error_label")

    fig = go.Figure()
    histfunc_map = {"avg": "avg", "average": "avg", "sum": "sum", "total": "sum", "count": "count", "min": "min", "max": "max"}
    plotly_histfunc = histfunc_map.get(aggregation_func.lower(), "avg")
    x_min_data, x_max_data = df[x_col].min(), df[x_col].max()
    y_min_data, y_max_data = df[y_col].min(), df[y_col].max()
    xbins_config = {}
    if xbins and xbins > 0 and (x_max_data - x_min_data > config.EPSILON): xbins_config['size'] = (x_max_data - x_min_data) / xbins
    ybins_config = {}
    if ybins and ybins > 0 and (y_max_data - y_min_data > config.EPSILON): ybins_config['size'] = (y_max_data - y_min_data) / ybins
    colorbar_config = dict(
        title=dict(text=get_lang_text(lang, colorbar_title_key, "Value"), side="right",
                   font=dict(size=config.FONT_SIZE_AXIS_TITLE, color=config.COLOR_TEXT_PRIMARY)),
        tickfont=dict(size=config.FONT_SIZE_AXIS_TICKS, color=config.COLOR_TEXT_PRIMARY),
        thickness=15, len=0.75, bgcolor=config.COLOR_LEGEND_BACKGROUND, # Consistent bg for colorbar
        bordercolor=config.COLOR_LEGEND_BORDER, borderwidth=1 )

    fig.add_trace(go.Histogram2d(
        x=df[x_col], y=df[y_col], z=df[z_col], histfunc=plotly_histfunc,
        xbins=xbins_config if xbins_config else None, ybins=ybins_config if ybins_config else None,
        colorscale=colorscale, showscale=True, colorbar=colorbar_config,
        zmin=df[z_col].min() if plotly_histfunc != "count" and not df[z_col].empty else None,
        zmax=df[z_col].max() if plotly_histfunc != "count" and not df[z_col].empty else None,
        name=get_lang_text(lang, aggregation_func + "_label", aggregation_func.title()) # For potential legend
    ))
    if show_points:
        hover_texts = [f"{z_val:.1f}" for z_val in df[z_col]] # Using .1f for consistency
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[y_col], mode='markers',
            marker=dict(size=point_size, color=df[z_col], colorscale=colorscale, opacity=point_opacity, showscale=False),
            text=hover_texts, hoverinfo='x+y+text',
            name=get_lang_text(lang, "individual_data_points_label", "Data Points") ))

    shapes = []
    if facility_dimensions:
        min_x_boundary = facility_dimensions.get("x0", x_min_data if not pd.isna(x_min_data) else 0)
        min_y_boundary = facility_dimensions.get("y0", y_min_data if not pd.isna(y_min_data) else 0)
        max_x_boundary = facility_dimensions.get("x1", x_max_data if not pd.isna(x_max_data) else 100) # Default if data bounds are NaN
        max_y_boundary = facility_dimensions.get("y1", y_max_data if not pd.isna(y_max_data) else 100)
        shapes.append(
            dict(type="rect", xref="x", yref="y",
                 x0=min_x_boundary, y0=min_y_boundary,
                 x1=max_x_boundary, y1=max_y_boundary,
                 line=dict(color=config.FACILITY_OUTLINE_COLOR, width=2),
                 layer="below" )
        )
    
    annotations_list_combined = [] # Initialize for possible text annotations for E/E points
    if entry_exit_points:
        ee_x = [p['x'] for p in entry_exit_points if 'x' in p and 'y' in p]
        ee_y = [p['y'] for p in entry_exit_points if 'x' in p and 'y' in p]
        ee_texts = []
        for p in entry_exit_points:
            if 'x' in p and 'y' in p:
                point_type_key = f"{p.get('type', 'point').lower()}_point_label_short"
                point_type_localized = get_lang_text(lang, point_type_key, p.get('type', 'Point').title())
                ee_texts.append(f"{p.get('name', 'P')} ({point_type_localized})")
        
        # Plot E/E points as a separate trace
        fig.add_trace(go.Scatter(
            x=ee_x, y=ee_y, text=ee_texts, mode='markers+text', # Ensure text is shown
            marker=dict(
                symbol=config.ENTRY_EXIT_POINT_SYMBOL, color=config.ENTRY_EXIT_POINT_COLOR,
                size=config.ENTRY_EXIT_POINT_SIZE,
                line=dict(color=config.ENTRY_EXIT_POINT_BORDER_COLOR, width=config.ENTRY_EXIT_POINT_BORDER_WIDTH)
            ),
            textfont=dict(color=config.ENTRY_EXIT_LABEL_COLOR, size=config.ENTRY_EXIT_LABEL_SIZE),
            textposition="top center",
            hoverinfo='text', name=get_lang_text(lang, "entry_exit_points_legend_label", "Key Locations")
        ))

    extra_updates_for_layout = {"shapes": shapes} if shapes else {}
    # Add annotations if they were created (not used by default in this simpler E/E plotting)
    # if annotations_list_combined: extra_updates_for_layout["annotations"] = annotations_list_combined
    
    _apply_standard_layout(
        fig, lang, title_text_direct=title_text,
        x_axis_title_key="x_coordinate_label", y_axis_title_key="y_coordinate_label",
        legend_params={"showlegend": show_points or bool(entry_exit_points), 
                       "orientation": "v", "x": 1.02, "y": 1, "xanchor": "left", "yanchor": "top"},
        extra_layout_updates=extra_updates_for_layout
    )
    fig.update_layout(
        xaxis_constrain='domain', yaxis_scaleanchor='x', yaxis_scaleratio=1,
        autosize=True # Helps fit within Streamlit columns
    )
    fig.update_xaxes(gridcolor=config.COLOR_GRID_SECONDARY) # Explicitly ensure grid colors are applied
    fig.update_yaxes(gridcolor=config.COLOR_GRID_PRIMARY)
    return fig

# --- Spatial Dynamics: Worker Density Heatmap (using Histogram2dContour) ---
def create_worker_density_heatmap(
    df_input: pd.DataFrame,
    x_col: str, y_col: str,
    title_key: str, lang: str,
    ncontours: int = 20, # Increased for smoother density contours
    colorscale: str = "Blues",
    show_points: bool = False,
    point_size: int = 2, # Slightly larger for better visibility if shown
    point_color: str = 'rgba(0,0,0,0.2)', # More subtle point color
    facility_dimensions: Optional[Dict[str, float]] = None,
    entry_exit_points: Optional[List[Dict[str, Any]]] = None
) -> go.Figure:
    df = df_input.copy()
    title_text = get_lang_text(lang, title_key)

    if df.empty or not all(c in df.columns for c in [x_col, y_col]):
        logger.warning(f"Worker Density Heatmap: Missing coordinate columns: {x_col}, {y_col}")
        return _create_no_data_figure(lang, title_key, message_key="heatmap_no_coordinate_data")

    df.dropna(subset=[x_col, y_col], inplace=True)
    if df.empty:
        logger.warning(f"Worker Density Heatmap: No valid data after NaNs drop for {x_col}, {y_col}")
        return _create_no_data_figure(lang, title_key, message_key="heatmap_no_coordinate_data")
    
    try:
        df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
        df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
        df.dropna(subset=[x_col, y_col], inplace=True)
        if df.empty: raise ValueError("All coordinate data became NaN after numeric conversion.")
    except ValueError as e:
        logger.error(f"Worker Density Heatmap: Error converting columns to numeric: {e}")
        return _create_no_data_figure(lang, title_key, message_key="chart_generation_error_label")

    fig = go.Figure()
    fig.add_trace(go.Histogram2dContour(
        x = df[x_col], y = df[y_col],
        colorscale = colorscale, reversescale = False, xaxis = 'x', yaxis = 'y',
        showscale = True, ncontours = ncontours,
        line=dict(width=0.3, color='rgba(0,0,0,0.1)'), # Even more subtle contour lines
        histnorm="density", # Use "density" for a normalized representation
        colorbar=dict(
            title=dict(text=get_lang_text(lang, "density_label_short", "Density"), side="right",
                       font=dict(size=config.FONT_SIZE_AXIS_TITLE, color=config.COLOR_TEXT_PRIMARY)),
            tickfont=dict(size=config.FONT_SIZE_AXIS_TICKS, color=config.COLOR_TEXT_PRIMARY),
            thickness=15, len=0.75, bgcolor=config.COLOR_LEGEND_BACKGROUND,
            bordercolor=config.COLOR_LEGEND_BORDER, borderwidth=1 ),
        name=get_lang_text(lang, "density_label_short", "Density") ))
    if show_points:
        fig.add_trace(go.Scatter(
            x=df[x_col], y=df[y_col], mode='markers',
            marker=dict(color=point_color, size=point_size, opacity=0.5),
            hoverinfo='skip', name=get_lang_text(lang, "individual_data_points_label", "Locations") ))

    if entry_exit_points:
        ee_x = [p['x'] for p in entry_exit_points if 'x' in p and 'y' in p]
        ee_y = [p['y'] for p in entry_exit_points if 'x' in p and 'y' in p]
        ee_texts = []
        for p in entry_exit_points:
            if 'x' in p and 'y' in p:
                point_type_key = f"{p.get('type', 'point').lower()}_point_label_short"
                point_type_localized = get_lang_text(lang, point_type_key, p.get('type', 'Point').title())
                ee_texts.append(f"{p.get('name', 'P')} ({point_type_localized})")
        
        fig.add_trace(go.Scatter(
            x=ee_x, y=ee_y, text=ee_texts, mode='markers+text',
            marker=dict(
                symbol=config.ENTRY_EXIT_POINT_SYMBOL, color=config.ENTRY_EXIT_POINT_COLOR,
                size=config.ENTRY_EXIT_POINT_SIZE + 2, # Make E/E points slightly larger
                line=dict(color=config.ENTRY_EXIT_POINT_BORDER_COLOR, width=config.ENTRY_EXIT_POINT_BORDER_WIDTH)
            ),
            textfont=dict(color=config.ENTRY_EXIT_LABEL_COLOR, size=config.ENTRY_EXIT_LABEL_SIZE),
            textposition="bottom center", # Place below marker for less overlap with density
            hoverinfo='text', name=get_lang_text(lang, "entry_exit_points_legend_label", "Key Locations")
        ))

    shapes = []
    if facility_dimensions:
        x_min_data_for_bounds, x_max_data_for_bounds = (df[x_col].min(), df[x_col].max()) if not df[x_col].empty else (0,0)
        y_min_data_for_bounds, y_max_data_for_bounds = (df[y_col].min(), df[y_col].max()) if not df[y_col].empty else (0,0)
        shapes.append(
            dict(type="rect", xref="x", yref="y",
                 x0=facility_dimensions.get("x0", x_min_data_for_bounds), y0=facility_dimensions.get("y0", y_min_data_for_bounds),
                 x1=facility_dimensions.get("x1", x_max_data_for_bounds), y1=facility_dimensions.get("y1", y_max_data_for_bounds),
                 line=dict(color=config.FACILITY_OUTLINE_COLOR, width=2), layer="below" ))

    _apply_standard_layout(
        fig, lang, title_text_direct=title_text,
        x_axis_title_key="x_coordinate_label", y_axis_title_key="y_coordinate_label",
        legend_params={"showlegend": bool(entry_exit_points) or show_points, 
                       "orientation": "v", "x": 1.02, "y": 1, "xanchor": "left", "yanchor": "top"}, # Position legend top-right outside plot
        extra_layout_updates={"shapes": shapes} if shapes else {}
    )
    fig.update_layout(
        autosize=True, xaxis_constrain='domain', yaxis_scaleanchor="x", yaxis_scaleratio=1,
        xaxis_gridcolor=config.COLOR_GRID_SECONDARY,
        yaxis_gridcolor=config.COLOR_GRID_PRIMARY,
    )
    return fig

# --- Stress Semáforo Visual ---
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
    num_config_s = {'font': {'size': config.FONT_SIZE_STRESS_SEMAFORO_NUMBER, 'color': color_for_status_s}, 'valueformat': ".1f"}
    if display_num_stress is not None: num_config_s['suffix'] = f" / {scale_max:.0f}"
    indicator_title_obj_stress = {
        'text': f"<b style='color:{color_for_status_s}; font-size:1em;'>{text_for_status_s.upper()}</b>",
        'font': {'size': config.FONT_SIZE_STRESS_SEMAFORO_TITLE}
    }
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=display_num_stress, domain={'x': [0.0, 1.0], 'y': [0.0, 0.8]},
        title=indicator_title_obj_stress, number=num_config_s,
        gauge={
            'shape': "bullet",
            'axis': {'range': [0, scale_max], 'visible': True, 'showticklabels': True,
                     'tickvals': [0, config.STRESS_LEVEL_PSYCHOSOCIAL["low"], config.STRESS_LEVEL_PSYCHOSOCIAL["medium"], scale_max],
                     'ticktext': ["0", f"{config.STRESS_LEVEL_PSYCHOSOCIAL['low']:.1f}",
                                  f"{config.STRESS_LEVEL_PSYCHOSOCIAL['medium']:.1f}", f"{scale_max:.0f}"],
                     'tickfont': {'size':config.FONT_SIZE_STRESS_SEMAFORO_AXIS_TICK, 'color': config.COLOR_TEXT_SECONDARY},
                     'tickmode': 'array'},
            'steps': [
                {'range': [0, config.STRESS_LEVEL_PSYCHOSOCIAL["low"]], 'color': config.COLOR_STRESS_BULLET_LOW},
                {'range': [config.STRESS_LEVEL_PSYCHOSOCIAL["low"], config.STRESS_LEVEL_PSYCHOSOCIAL["medium"]], 'color': config.COLOR_STRESS_BULLET_MEDIUM},
                {'range': [config.STRESS_LEVEL_PSYCHOSOCIAL["medium"], scale_max], 'color': config.COLOR_STRESS_BULLET_HIGH}
            ],
            'bar': {'color': color_for_status_s, 'thickness': 0.4,
                    'line':{'color':config.COLOR_STRESS_BULLET_BAR_BORDER, 'width':0.5}},
            'bgcolor': config.COLOR_STRESS_BULLET_BACKGROUND,
            'borderwidth': 0.5, 'bordercolor': config.COLOR_STRESS_BULLET_BORDER
        }))
    fig.update_layout(height=80, margin=dict(t=10, b=5, l=5, r=5), paper_bgcolor=config.COLOR_PAPER_BACKGROUND)
    return fig
