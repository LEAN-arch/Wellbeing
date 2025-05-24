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
    text_dict = config.TEXT_STRINGS.get(effective_lang_code, config.TEXT_STRINGS[config.DEFAULT_LANG])
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

    if higher_is_worse: # Lower value is better
        if good_f is not None and val_f <= good_f: return "good"
        if warn_f is not None and (good_f is None or val_f > good_f) and val_f <= warn_f: return "warning"
        if (warn_f is not None and val_f > warn_f) or \
           (warn_f is None and good_f is not None and val_f > good_f): return "critical"
    else: # Higher value is better
        if good_f is not None and val_f >= good_f: return "good"
        if warn_f is not None and (good_f is None or val_f < good_f) and val_f >= warn_f: return "warning"
        if (warn_f is not None and val_f < warn_f) or \
           (warn_f is None and good_f is not None and val_f < good_f): return "critical"
    return None # No clear status matched

def get_semaforo_color(status: Optional[str]) -> str:
    if status == "good": return config.COLOR_STATUS_GOOD
    if status == "warning": return config.COLOR_STATUS_WARNING
    if status == "critical": return config.COLOR_STATUS_CRITICAL
    return config.COLOR_TEXT_SECONDARY


# --- KPI Gauge Visualization (Substantially Improved) ---
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
    current_val_for_display = float(value) if pd.notna(value) and isinstance(value, (int,float,np.number)) else None # Pass None to Indicator for its formatting

    delta_obj = {}
    if previous_value is not None and pd.notna(previous_value) and isinstance(previous_value, (int,float,np.number)) and current_val_for_display is not None:
        delta_ref_val = float(previous_value)
        delta_obj = {
            'reference': delta_ref_val,
            'increasing': {'color': config.COLOR_STATUS_CRITICAL if higher_is_worse else config.COLOR_STATUS_GOOD, 'symbol': "▲"},
            'decreasing': {'color': config.COLOR_STATUS_GOOD if higher_is_worse else config.COLOR_STATUS_CRITICAL, 'symbol': "▼"},
            'font': {'size': 12}
        }
        # Add current status color to delta text itself if value changed
        delta_val = current_val_numeric - delta_ref_val
        if abs(delta_val) > 1e-9 : # If there is a change
            delta_obj['font']['color'] = delta_obj['increasing']['color'] if delta_val > 0 else delta_obj['decreasing']['color']
        else: # No change, neutral color
            delta_obj['font']['color'] = config.COLOR_TEXT_SECONDARY


    # Dynamic max_value for the gauge axis
    if max_value_override is not None and pd.notna(max_value_override):
        axis_max_val = float(max_value_override)
    else:
        val_candidates_for_max = [1.0]
        if pd.notna(current_val_numeric): val_candidates_for_max.append(abs(current_val_numeric) * 1.4) # Give more headroom
        
        ref_points_for_max = [threshold_good, threshold_warning, target_line_value]
        valid_ref_points_for_max = [float(p) for p in ref_points_for_max if p is not None and pd.notna(p)]
        if valid_ref_points_for_max: val_candidates_for_max.append(max(valid_ref_points_for_max) * 1.25)
        
        if not valid_ref_points_for_max and (not pd.notna(current_val_numeric) or current_val_numeric == 0):
             val_candidates_for_max.append(100.0 if unit == "%" else 10.0) # Contextual default

        axis_max_val = max(val_candidates_for_max)
        if axis_max_val <= (current_val_numeric if pd.notna(current_val_numeric) else 0):
            axis_max_val = (current_val_numeric * 1.2) if pd.notna(current_val_numeric) and current_val_numeric > 0 else (axis_max_val * 1.2 or 10.0)
        if axis_max_val <= 0 : axis_max_val = 10.0

    # Gauge steps for coloring background
    gauge_color_steps = []
    num_th_good_val = float(threshold_good) if threshold_good is not None and pd.notna(threshold_good) else None
    num_th_warn_val = float(threshold_warning) if threshold_warning is not None and pd.notna(threshold_warning) else None
    
    if num_th_good_val is not None and num_th_warn_val is not None: # Ensure logical threshold order
        if higher_is_worse and num_th_warn_val < num_th_good_val: num_th_warn_val = num_th_good_val
        if not higher_is_worse and num_th_warn_val > num_t_good_val: num_t_warn_val = num_t_good_val
        
    current_step_range_start = 0.0
    if higher_is_worse:
        if num_th_good_val is not None:
            gauge_color_steps.append({'range': [current_step_range_start, num_th_good_val], 'color': config.COLOR_STATUS_GOOD})
            current_step_range_start = num_th_good_val
        if num_th_warn_val is not None and num_t_warn_val > current_step_range_start:
            gauge_color_steps.append({'range': [current_step_range_start, num_th_warn_val], 'color': config.COLOR_STATUS_WARNING})
            current_step_range_start = num_t_warn_val
        gauge_color_steps.append({'range': [current_step_range_start, axis_max_val], 'color': config.COLOR_STATUS_CRITICAL})
    else: 
        if num_th_warn_val is not None: 
            gauge_color_steps.append({'range': [current_step_range_start, num_th_warn_val], 'color': config.COLOR_STATUS_CRITICAL})
            current_step_range_start = num_t_warn_val
        if num_th_good_val is not None and num_th_good_val > current_step_range_start:
            gauge_color_steps.append({'range': [current_step_range_start, num_th_good_val], 'color': config.COLOR_STATUS_WARNING})
            current_step_range_start = num_th_good_val
        gauge_color_steps.append({'range': [current_step_range_start, axis_max_val], 'color': config.COLOR_STATUS_GOOD})

    if not gauge_color_steps: gauge_color_steps.append({'range': [0, axis_max_val], 'color': config.COLOR_NEUTRAL_INFO})
    
    target_line_val_float = float(target_line_value) if target_line_value is not None and pd.notna(target_line_value) else None
    
    current_status_for_number_color = get_status_by_thresholds(current_val_numeric, higher_is_worse, num_th_good_val, num_th_warn_val)
    number_display_color = get_semaforo_color(current_status_for_number_color) if current_status_for_number_color else config.COLOR_TARGET_LINE


    fig = go.Figure(go.Indicator(
        mode="gauge+number" + ("+delta" if delta_obj else ""),
        value=current_val_for_display,
        title={'text': title_final, 'font': {'size': 13, 'color': config.COLOR_TEXT_SECONDARY}},
        number={'font': {'size': 30, 'color': number_display_color}, # Color number based on status
                'suffix': unit if unit and pd.notna(current_val_for_display) else "", 'valueformat': ".1f"},
        delta=delta_obj,
        gauge={
            'axis': {'range': [0, axis_max_val], 'tickwidth': 1, 'tickcolor': "rgba(0,0,0,0.2)", 'nticks': 5, 'tickfont':{'size':9}},
            'bar': {'color': "rgba(0,0,0,0.7)", 'thickness': 0.1, 'line':{'color':"rgba(0,0,0,1)", 'width':0.5}}, # Thin, dark pointer bar
            'bgcolor': "rgba(255,255,255,0.0)", # Make gauge area transparent if steps are good
            'borderwidth': 0.5, 'bordercolor': "rgba(0,0,0,0.1)",
            'steps': gauge_color_steps,
            'threshold': {
                'line': {'color': config.COLOR_TARGET_LINE, 'width': 2},
                'thickness': 0.75, 'value': target_line_val_float
            } if target_line_val_float is not None else {}
        }
    ))
    fig.update_layout(height=170, margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor='rgba(0,0,0,0)')
    return fig

# --- Trend Chart Visualization (Substantially Improved) ---
def create_trend_chart(df_input: pd.DataFrame, date_col: str,
                       value_cols_map: Dict[str, str], title_key: str, lang: str,
                       y_axis_title_key: str = "value_axis_label", x_axis_title_key: str = "date_time_axis_label",
                       show_average_line: bool = False,
                       target_value_map: Optional[Dict[str, Union[int, float]]] = None, # {actual_col_name: target_val}
                       rolling_avg_window: Optional[int] = None,
                       value_col_units_map: Optional[Dict[str, str]] = None, # {actual_col_name: unit_string}
                       y_axis_format_str: Optional[str] = ",.1f") -> go.Figure:
    df = df_input.copy()
    title_text = get_lang_text(lang, title_key)
    x_title_text = get_lang_text(lang, x_axis_title_key)
    y_title_text = get_lang_text(lang, y_axis_title_key)

    if df.empty or date_col not in df.columns or not value_cols_map:
        return go.Figure().update_layout(title_text=f"{title_text} ({get_lang_text(lang, 'no_data_for_selection')})",
            annotations=[dict(text=get_lang_text(lang, 'no_data_for_selection'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5, font_size=12)])

    fig = go.Figure()
    colors = px.colors.qualitative.D3 # Another good categorical palette
    plotted_actual_cols_list = []

    for i, (display_label_key, actual_data_col) in enumerate(value_cols_map.items()):
        if actual_data_col not in df.columns or not pd.api.types.is_numeric_dtype(df[actual_data_col]): continue
        plotted_actual_cols_list.append(actual_data_col)
        
        line_color = colors[i % len(colors)]
        legend_display_name = get_lang_text(lang, display_label_key, actual_data_col.replace('_',' ').title())
        unit_str = value_col_units_map.get(actual_data_col, "") if value_col_units_map else ""
        
        fig.add_trace(go.Scatter(x=df[date_col], y=df[actual_data_col], mode='lines+markers', name=legend_display_name,
            line=dict(color=line_color, width=2.5), marker=dict(size=6, symbol="circle"),
            hovertemplate=(f"<b>{legend_display_name}</b><br>" +
                           f"{get_lang_text(lang, 'date_label','Date')}: %{{x|%b %d, %Y}}<br>" + # Use %b for abbreviated month
                           f"{y_title_text}: %{{y:{y_axis_format_str if y_axis_format_str else ',.2f'}}}{unit_str}<extra></extra>")
        ))

    if rolling_avg_window and isinstance(rolling_avg_window, int) and rolling_avg_window > 0:
        for i, actual_data_col in enumerate(plotted_actual_cols_list):
            if len(df) >= rolling_avg_window :
                display_label_key = [k for k,v in value_cols_map.items() if v == actual_data_col][0]
                base_name = get_lang_text(lang, display_label_key)
                ma_legend_name = f"{base_name} ({rolling_avg_window}-p MA)"
                unit_str = value_col_units_map.get(actual_data_col, "") if value_col_units_map else ""
                # Use unique name for temp column
                temp_rolling_col = f"_rolling_{actual_data_col}"
                df[temp_rolling_col] = df[actual_data_col].rolling(window=rolling_avg_window, center=True, min_periods=1).mean()
                fig.add_trace(go.Scatter(x=df[date_col], y=df[temp_rolling_col], mode='lines', name=ma_legend_name,
                    line=dict(color=colors[i % len(colors)], width=1.5, dash='longdash'), opacity=0.6, # Softer rolling line
                    hovertemplate=(f"<b>{ma_legend_name}</b><br>" +
                                   f"{get_lang_text(lang, 'date_label','Date')}: %{{x|%b %d, %Y}}<br>" +
                                   f"{y_title_text}: %{{y:{y_axis_format_str if y_axis_format_str else ',.2f'}}}{unit_str}<extra></extra>")
                ))
    
    for i, actual_data_col in enumerate(plotted_actual_cols_list):
        display_label_key = [k for k,v in value_cols_map.items() if v == actual_data_col][0]
        series_name_disp = get_lang_text(lang, display_label_key)
        line_color = colors[i % len(colors)]
        if show_average_line:
            avg_value = df[actual_data_col].mean()
            if pd.notna(avg_value):
                fig.add_hline(y=avg_value, line_dash="dashdot", line_color=line_color, opacity=0.5,
                              annotation_text=f"{get_lang_text(lang, 'average_label')} {series_name_disp}: {avg_value:{y_axis_format_str if y_axis_format_str else ',.1f'}}",
                              annotation_position="bottom right" if i%2==0 else "bottom left", annotation_font_size=9, annotation_bgcolor="rgba(255,255,255,0.75)")
        if target_value_map and actual_data_col in target_value_map and pd.notna(target_value_map[actual_data_col]):
            target_val_line = target_value_map[actual_data_col]
            fig.add_hline(y=target_val_line, line_dash="solid", line_color=config.COLOR_TARGET_LINE, line_width=1.5, opacity=0.9,
                          annotation_text=f"{get_lang_text(lang, 'target_label')} {series_name_disp}: {target_val_line:{y_axis_format_str if y_axis_format_str else ',.1f'}}",
                          annotation_position="top left" if i%2==0 else "top right", annotation_font_size=10, annotation_font_color=config.COLOR_TARGET_LINE, annotation_bgcolor="rgba(255,255,255,0.75)")

    fig.update_layout(
        title=dict(text=title_text, x=0.05, xanchor='left', font_size=17), # Title left-aligned for more "report" feel
        yaxis_title=y_title_text, xaxis_title=x_title_text,
        legend_title_text="", hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=11, bordercolor=config.COLOR_TEXT_SECONDARY, namelength=-1),
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', type='date',
                   showspikes=True, spikemode='across', spikesnap='cursor', spikethickness=1, spikedash='dot', spikecolor=config.COLOR_TARGET_LINE, # Add spikes for x-axis
                   rangeslider_visible= len(df[date_col].unique()) > 20, # Slider for many points
                   rangeselector=dict(buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="todate" if df[date_col].max() > pd.Timestamp.now() - pd.DateOffset(months=1) else "backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"), dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"), dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all")]),font_size=10, y=1.15, x=0, xanchor='left')),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)', tickformat=y_axis_format_str),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5, font_size=10), # Legend below plot
        margin=dict(l=50, r=30, t=60, b=100) # Increased bottom margin for legend if many items
    )
    return fig

# --- Comparison Bar Chart Visualization (More "Friendly" and Actionable) ---
def create_comparison_bar_chart(df_input: pd.DataFrame, x_col: str,
                                y_cols_map: Dict[str, str], # {TEXT_STRING_KEY_FOR_LABEL: ACTUAL_COLUMN_NAME_IN_DF}
                                title_key: str, lang: str,
                                y_axis_title_key: str = "count_label", x_axis_title_key: str = "category_axis_label",
                                barmode: str = 'group', show_total_for_stacked: bool = False,
                                data_label_format_str: str = ".0f",
                                add_bar_trendline: Optional[str] = None) -> go.Figure: # actual_col_name for trendline
    df = df_input.copy() 
    title_text = get_lang_text(lang, title_key)
    x_title_text = get_lang_text(lang, x_axis_title_key)
    y_title_text = get_lang_text(lang, y_axis_title_key)
    
    df_plot = df[[x_col]].copy()
    y_cols_for_plotting_display_names = []
    actual_y_cols_used = []

    for label_k, actual_col in y_cols_map.items():
        if actual_col in df.columns and pd.api.types.is_numeric_dtype(df[actual_col]):
            display_n = get_lang_text(lang, label_k, actual_col.replace('_', ' ').title())
            df_plot[display_n] = df[actual_col]
            y_cols_for_plotting_display_names.append(display_n)
            actual_y_cols_used.append(actual_col) # Keep track of original cols that were plotted

    if df_plot.empty or x_col not in df_plot.columns or not y_cols_for_plotting_display_names:
        return go.Figure().update_layout(title_text=f"{title_text} ({get_lang_text(lang, 'no_data_for_selection')})",
            annotations=[dict(text=get_lang_text(lang, 'no_data_for_selection'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)])

    fig = px.bar(df_plot, x=x_col, y=y_cols_for_plotting_display_names, title=None, barmode=barmode,
                 color_discrete_sequence=config.COLOR_SCHEME_CATEGORICAL)
    
    fmt_specifier = data_label_format_str if (isinstance(data_label_format_str, str) and data_label_format_str) else ".0f"
    text_tmpl = f'%{{y:{fmt_specifier}}}'
    hover_tmpl = f'<b>%{{x}}</b><br>%{{fullData.name}}: %{{y:{fmt_specifier}}}<extra></extra>'

    fig.update_traces(
        texttemplate=text_tmpl, textposition='outside' if barmode != 'stack' else 'inside',
        textfont=dict(size=9, color=config.COLOR_TEXT_SECONDARY if barmode=='stack' else 'black'),
        insidetextanchor='middle' if barmode == 'stack' else 'end', hovertemplate=hover_tmpl,
        marker_line_width=0.5, marker_line_color='rgba(0,0,0,0.4)') # Subtle bar outline

    if barmode == 'stack' and show_total_for_stacked and actual_y_cols_used: # Sum from original df cols
        df_plot['_total_'] = df[actual_y_cols_used].sum(axis=1, numeric_only=True)
        annotations_total = [
            dict(x=row[x_col], y=row['_total_'], text=f"{row['_total_']:{fmt_specifier}}",
                 font=dict(size=10, color=config.COLOR_TARGET_LINE), showarrow=False, 
                 yanchor='bottom', yshift=4, xanchor='center')
            for _, row in df_plot.iterrows() if pd.notna(row['_total_'])
        ]
        current_annotations = list(fig.layout.annotations or [])
        fig.update_layout(annotations=current_annotations + annotations_total)
    
    # Optional: Add a trendline for a specific y-column (usually for single series or primary series in group)
    if add_bar_trendline and add_bar_trendline in y_cols_map: # add_bar_trendline is label_key
        actual_col_for_trend = y_cols_map[add_bar_trendline]
        # Ensure x_col can be used for OLS trendline (e.g., convert categorical months to numbers if needed)
        if actual_col_for_trend in df_plot.columns: # Check if this col (possibly renamed) exists in df_plot
            df_plot_trend = df_plot.copy()
            # If x_col is categorical month names, map to numbers for trendline
            if df_plot_trend[x_col].dtype == 'object' or pd.api.types.is_categorical_dtype(df_plot_trend[x_col]):
                 try:
                    month_map = {name: i for i, name in enumerate(pd.to_datetime(df_plot_trend[x_col], format='%b', errors='coerce').dt.strftime('%b').unique() if pd.notna(name))}
                    if month_map: df_plot_trend['_x_numeric_'] = df_plot_trend[x_col].map(month_map)
                 except: df_plot_trend['_x_numeric_'] = pd.Series(range(len(df_plot_trend))) # Fallback
            else: df_plot_trend['_x_numeric_'] = df_plot_trend[x_col] # Assume it's numeric or date already

            if '_x_numeric_' in df_plot_trend.columns and pd.api.types.is_numeric_dtype(df_plot_trend['_x_numeric_']):
                trendline_fig = px.scatter(df_plot_trend, x='_x_numeric_', y=actual_col_for_trend,
                                          trendline="ols", trendline_color_override=config.COLOR_TARGET_LINE)
                if len(trendline_fig.data) > 1: # OLS line is the second trace
                    trendline_trace = trendline_fig.data[1]
                    trendline_trace.name = f"{get_lang_text(lang, add_bar_trendline)} Trend"
                    trendline_trace.line.dash = 'dash'
                    trendline_trace.showlegend = True
                    fig.add_trace(trendline_trace)


    fig.update_layout(
        title=dict(text=title_text, x=0.05, xanchor='left', font_size=16),
        yaxis_title=y_title_text, xaxis_title=x_title_text,
        legend_title_text="", hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=11, namelength=-1),
        xaxis_tickangle=-30 if len(df_plot[x_col].unique()) > 6 else 0,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, font_size=9, title_text=""),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.08)'),
        xaxis=dict(showgrid=False, type='category', linecolor='rgba(0,0,0,0.2)'),
        bargap=0.15, margin=dict(l=50, r=20, t=50, b=80 if len(y_cols_for_plotting_display_names) > 2 else 50)
    )
    return fig

# --- Metric Card (UX Enhanced - Keep previous robust version) ---
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

# --- Radar Chart Visualization (UX Enhanced - keep robust previous version) ---
def create_enhanced_radar_chart(df_radar_input: pd.DataFrame, categories_col: str, values_col: str,
                                 title_key: str, lang: str, group_col: Optional[str] = None,
                                 range_max_override: Optional[Union[int, float]] = None,
                                 target_values_map: Optional[Dict[str, Union[int, float]]] = None, 
                                 fill_opacity: float = 0.3): # Default less opaque
    title_text = get_lang_text(lang, title_key)
    df_radar = df_radar_input.copy()
    if df_radar.empty or categories_col not in df_radar.columns or values_col not in df_radar.columns or df_radar[categories_col].nunique() == 0:
        return go.Figure().update_layout(title_text=f"{title_text} ({get_lang_text(lang, 'no_data_radar')})",
            annotations=[dict(text=get_lang_text(lang, 'no_data_radar'),showarrow=False, xref="paper", yref="paper", x=0.5,y=0.5)])
    all_categories_ordered_list = df_radar[categories_col].unique()
    all_r_vals_radar = df_radar[values_col].dropna().tolist() # From data
    if target_values_map: # Add target values to consider for max range
        all_r_vals_radar.extend([v for v in target_values_map.values() if pd.notna(v) and isinstance(v,(int,float))])
    valid_r_vals_radar = [float(v) for v in all_r_vals_radar if isinstance(v, (int,float)) and pd.notna(v)]
    max_data_val_for_radar_range = max(valid_r_vals_radar) if valid_r_vals_radar else 0.0
    radial_range_max_final = float(range_max_override) if range_max_override is not None and pd.notna(range_max_override) else (max_data_val_for_radar_range * 1.2 if max_data_val_for_radar_range > 0 else config.ENGAGEMENT_RADAR_DIM_SCALE_MAX or 5.0)
    radial_range_max_final = max(radial_range_max_final, 1.0) # Min range of 1
    fig = go.Figure()
    colors_radar = config.COLOR_SCHEME_CATEGORICAL
    if group_col and group_col in df_radar.columns and df_radar[group_col].nunique() > 0:
        for i, (name_grp_radar, group_data_radar) in enumerate(df_radar.groupby(group_col)):
            current_grp_ordered_radar = pd.DataFrame({categories_col: all_categories_ordered_list}).merge(
                group_data_radar, on=categories_col, how='left').fillna({values_col: 0})
            fig.add_trace(go.Scatterpolar(
                r=current_grp_ordered_radar[values_col], theta=current_grp_ordered_radar[categories_col],
                fill='toself', name=str(name_grp_radar), line_color=colors_radar[i % len(colors_radar)], opacity=fill_opacity,
                hovertemplate='<b>%{theta}</b><br>' + f'{str(name_grp_radar)}: %{{r:.1f}}<extra></extra>' ))
    else:
        single_series_ordered_radar = pd.DataFrame({categories_col: all_categories_ordered_list}).merge(
            df_radar, on=categories_col, how='left').fillna({values_col: 0})
        fig.add_trace(go.Scatterpolar(
            r=single_series_ordered_radar[values_col], theta=single_series_ordered_radar[categories_col],
            fill='toself', name=get_lang_text(lang, "average_score_label"), line_color=colors_radar[0],
            opacity=fill_opacity + 0.15, hovertemplate='<b>%{theta}</b>: %{r:.1f}<extra></extra>')) # Main series slightly more opaque
    if target_values_map: 
        target_r_ordered_radar = [target_values_map.get(cat, 0) for cat in all_categories_ordered_list]
        fig.add_trace(go.Scatterpolar(
            r=target_r_ordered_radar, theta=all_categories_ordered_list, mode='lines', name=get_lang_text(lang, "target_label"),
            line=dict(color=config.COLOR_TARGET_LINE, dash='longdash', width=2), hoverinfo='skip')) # Thicker target line
    fig.update_layout(
        title=dict(text=title_text, x=0.5, font_size=16), # Slightly smaller title
        polar=dict(bgcolor="rgba(245,245,245,0.3)", # Very light grey bg for polar area
                   radialaxis=dict(visible=True, range=[0, radial_range_max_final], showline=True, linecolor='rgba(0,0,0,0.1)', gridcolor="rgba(0,0,0,0.1)", tickfont_size=8, angle=30, Ntickbins=5),
                   angularaxis=dict(showline=True, linecolor='rgba(0,0,0,0.1)', gridcolor="rgba(0,0,0,0.05)", tickfont_size=9, direction="clockwise", period=len(all_categories_ordered_list))),
        showlegend=True, legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5, font_size=9, itemsizing='constant'), # Legend below, constant size
        margin=dict(l=40, r=40, t=60, b=70), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

# --- Stress Semáforo Visual (More "Friendly" and Actionable) ---
def create_stress_semaforo_visual(stress_level_value: Optional[Union[int, float, np.number]], lang: str,
                                  scale_max: float = config.STRESS_LEVEL_PSYCHOSOCIAL["max_scale"]) -> go.Figure:
    display_num_stress, color_for_status_stress, text_for_status_stress = None, config.COLOR_TEXT_SECONDARY, get_lang_text(lang, 'status_na_label')

    if pd.notna(stress_level_value) and isinstance(stress_level_value, (int, float, np.number)):
        val_float_s = float(stress_level_value)
        display_num_stress = val_float_s
        status_s = get_status_by_thresholds(val_float_s, higher_is_worse=True,
                                           threshold_good=config.STRESS_LEVEL_PSYCHOSOCIAL["low"],
                                           threshold_warning=config.STRESS_LEVEL_PSYCHOSOCIAL["medium"])
        color_for_status_stress = get_semaforo_color(status_s)
        if status_s == "good": text_for_status_stress = get_lang_text(lang, 'low_label')
        elif status_s == "warning": text_for_status_stress = get_lang_text(lang, 'moderate_label')
        elif status_s == "critical": text_for_status_stress = get_lang_text(lang, 'high_label')
        else: text_for_status_stress = f"{val_float_s:.1f}" if pd.notna(val_float_s) else get_lang_text(lang, 'status_na_label')
            
    num_config_s = {'font': {'size': 24, 'color': color_for_status_stress}, 'valueformat': ".1f"} # Slightly larger number
    if display_num_stress is not None: num_config_s['suffix'] = f" / {scale_max:.0f}"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=display_num_stress, 
        domain={'x': [0, 1], 'y': [0, 1]}, # Use full domain, adjust height in layout
        title={'text': f"<b style='color:{color_for_status_stress}; font-size:1.15em;'>{text_for_status_stress.upper()}</b>", 
               'font': {'size': 13}, 'align': "center"}, # Title integrated with value
        number=num_config_s,
        gauge={
            'shape': "bullet",
            'axis': {'range': [0, scale_max], 'visible': True, 'showticklabels': True, 'layer': 'below traces',
                     'tickvals': [0, config.STRESS_LEVEL_PSYCHOSOCIAL["low"], config.STRESS_LEVEL_PSYCHOSOCIAL["medium"], scale_max],
                     'ticktext': ["0", f"{config.STRESS_LEVEL_PSYCHOSOCIAL['low']:.0f}", f"{config.STRESS_LEVEL_PSYCHOSOCIAL['medium']:.0f}", f"{scale_max:.0f}"],
                     'tickfont': {'size':9, 'color': config.COLOR_TEXT_SECONDARY}, 'tickmode': 'array'},
            'steps': [ # Color bands for context
                {'range': [0, config.STRESS_LEVEL_PSYCHOSOCIAL["low"]], 'color': config.COLOR_STATUS_GOOD},
                {'range': [config.STRESS_LEVEL_PSYCHOSOCIAL["low"], config.STRESS_LEVEL_PSYCHOSOCIAL["medium"]], 'color': config.COLOR_STATUS_WARNING},
                {'range': [config.STRESS_LEVEL_PSYCHOSOCIAL["medium"], scale_max], 'color': config.COLOR_STATUS_CRITICAL}
            ],
            'bar': {'color': color_for_status_stress, 'thickness': 0.4, 'line':{'color':'white', 'width':1}}, # Bar showing current level, slightly thinner
            'bgcolor': "rgba(255,255,255,0.1)", 'borderwidth': 1, 'bordercolor': "rgba(0,0,0,0.1)"
        }))
    # Compact height suitable for embedding within columns or alongside other metrics
    fig.update_layout(height=100, margin=dict(t=25, b=25, l=15, r=15), paper_bgcolor='rgba(0,0,0,0)')
    return fig
