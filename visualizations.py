# visualizations.py
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import config
from typing import List, Dict, Optional, Any, Union

# --- Helper to get localized text ---
def get_lang_text(lang_code: str, key: str, default_text: str = "") -> str:
    text_dict = config.TEXT_STRINGS.get(lang_code, config.TEXT_STRINGS[config.DEFAULT_LANG])
    return text_dict.get(key, default_text or key) # Fallback to key if default_text is empty

# --- Helper for status determination ---
def get_status_by_thresholds(value: Optional[Union[int, float]], higher_is_worse: bool,
                             threshold_good: Optional[Union[int, float]] = None,
                             threshold_warning: Optional[Union[int, float]] = None) -> Optional[str]:
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
    return None # Undefined if no clear zone matches

def get_semaforo_color(status: Optional[str]) -> str:
    if status == "good": return config.COLOR_STATUS_GOOD
    if status == "warning": return config.COLOR_STATUS_WARNING
    if status == "critical": return config.COLOR_STATUS_CRITICAL
    return config.COLOR_GRAY_TEXT

# --- KPI Gauge ---
def create_kpi_gauge(value: Optional[Union[int, float]], title_key: str, lang: str,
                     unit: str = "%", higher_is_worse: bool = True,
                     threshold_good: Optional[Union[int, float]] = None,
                     threshold_warning: Optional[Union[int, float]] = None,
                     target_line_value: Optional[Union[int, float]] = None,
                     max_value_override: Optional[Union[int, float]] = None,
                     previous_value: Optional[Union[int, float]] = None) -> go.Figure:
    title_text = get_lang_text(lang, title_key, title_key)
    current_val_gauge = 0.0
    delta_ref_val = None

    if pd.notna(value) and isinstance(value, (int, float, np.number)):
        current_val_gauge = float(value)
        if previous_value is not None and pd.notna(previous_value) and isinstance(previous_value, (int, float, np.number)):
            delta_ref_val = float(previous_value)

    # Dynamic max_value determination
    if max_value_override is not None:
        max_axis_val = float(max_value_override)
    else:
        val_candidates = [1.0] # Start with a minimum to avoid zero range
        if pd.notna(current_val_gauge): val_candidates.append(current_val_gauge * 1.25)
        for t in [threshold_good, threshold_warning, target_line_value]:
            if t is not None and pd.notna(t): val_candidates.append(float(t) * 1.2)
        
        if not any(pd.notna(t) for t in [threshold_good, threshold_warning, target_line_value]) and \
           (not pd.notna(current_val_gauge) or current_val_gauge == 0):
            val_candidates.append(100.0) # Default max for empty/zero cases with no thresholds

        max_axis_val = max(val_candidates) if val_candidates else 100.0
        if max_axis_val <= (current_val_gauge if pd.notna(current_val_gauge) else 0): # Ensure value is visible
             max_axis_val = (current_val_gauge * 1.1) if pd.notna(current_val_gauge) and current_val_gauge > 0 else max_axis_val * 1.1 or 10.0

    # Gauge steps coloring
    steps_list = []
    num_t_good = float(threshold_good) if threshold_good is not None and pd.notna(threshold_good) else None
    num_t_warn = float(threshold_warning) if threshold_warning is not None and pd.notna(threshold_warning) else None
    
    # Logic: Ensure good < warn if higher_is_worse, or warn < good if not.
    # This simplifies step definition.
    # If only one threshold is given, it might be the "good" boundary or "warning" boundary.
    
    last_step_val = 0.0
    if higher_is_worse: # Green -> Yellow -> Red
        if num_t_good is not None:
            steps_list.append({'range': [last_step_val, num_t_good], 'color': config.COLOR_STATUS_GOOD})
            last_step_val = num_t_good
        if num_t_warn is not None and num_t_warn > last_step_val: # Ensure warning is distinct and higher
            steps_list.append({'range': [last_step_val, num_t_warn], 'color': config.COLOR_STATUS_WARNING})
            last_step_val = num_t_warn
        steps_list.append({'range': [last_step_val, max_axis_val], 'color': config.COLOR_STATUS_CRITICAL})
    else: # Lower is worse: Red -> Yellow -> Green
        if num_t_warn is not None: # Warn is the lower (bad) threshold
            steps_list.append({'range': [last_step_val, num_t_warn], 'color': config.COLOR_STATUS_CRITICAL})
            last_step_val = num_t_warn
        if num_t_good is not None and num_t_good > last_step_val: # Good is higher than warning
            steps_list.append({'range': [last_step_val, num_t_good], 'color': config.COLOR_STATUS_WARNING}) # This is the 'acceptable' range
            last_step_val = num_t_good
        steps_list.append({'range': [last_step_val, max_axis_val], 'color': config.COLOR_STATUS_GOOD})

    if not steps_list: steps_list.append({'range': [0, max_axis_val], 'color': config.COLOR_NEUTRAL_INFO})

    num_target = float(target_line_value) if target_line_value is not None and pd.notna(target_line_value) else None

    fig = go.Figure(go.Indicator(
        mode="gauge+number" + ("+delta" if delta_ref_val is not None else ""),
        value=current_val_gauge if pd.notna(current_val_gauge) else 0,
        title={'text': title_text, 'font': {'size': 15, 'color': config.COLOR_TEXT_SECONDARY}}, # Slightly smaller title
        number={'font': {'size': 26}, 'suffix': unit if unit and unit != "N/A" else "", 'valueformat': ".1f"},
        delta={'reference': delta_ref_val,
               'increasing': {'color': config.COLOR_STATUS_CRITICAL if higher_is_worse else config.COLOR_STATUS_GOOD},
               'decreasing': {'color': config.COLOR_STATUS_GOOD if higher_is_worse else config.COLOR_STATUS_CRITICAL},
               'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, max_axis_val], 'tickwidth': 1, 'tickcolor': "darkgray", 'nticks': 5},
            'bar': {'color': 'rgba(0,0,0,0.7)', 'thickness': 0.1}, # Main value pointer
            'bgcolor': "white", 'borderwidth': 0,
            'steps': steps_list,
            'threshold': {
                'line': {'color': config.COLOR_TARGET_LINE, 'width': 3},
                'thickness': 0.75, 'value': num_target
            } if num_target is not None else {}
        }
    ))
    fig.update_layout(height=180, margin=dict(l=15, r=15, t=40, b=5), paper_bgcolor='rgba(0,0,0,0)') # Transparent bg
    return fig

# --- Trend Chart ---
def create_trend_chart(df_input: pd.DataFrame, date_col: str,
                       value_cols_map: Dict[str, str], # {display_key: actual_col_name_in_df}
                       title_key: str, lang: str,
                       y_axis_title_key: str = "value_axis_label",
                       x_axis_title_key: str = "date_time_axis_label",
                       show_average_line: bool = False,
                       target_value_map: Optional[Dict[str, Union[int, float]]] = None, # {actual_col_name: target_value}
                       rolling_avg_window: Optional[int] = None,
                       value_col_units_map: Optional[Dict[str, str]] = None # {actual_col_name: unit_string}
                       ) -> go.Figure:
    df = df_input.copy()
    title_text = get_lang_text(lang, title_key, title_key)
    x_title_text = get_lang_text(lang, x_axis_title_key, "Date/Time")
    y_title_text = get_lang_text(lang, y_axis_title_key, "Value")

    if df.empty or date_col not in df.columns or not value_cols_map:
        return go.Figure().update_layout(title_text=f"{title_text} ({get_lang_text(lang, 'no_data_for_selection')})",
            annotations=[dict(text=get_lang_text(lang, 'no_data_for_selection'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5, font_size=12)])

    fig = go.Figure()
    colors = config.COLOR_SCHEME_CATEGORICAL
    processed_actual_cols = [] # To keep track of successfully added series for avg/target lines

    for i, (display_key, actual_col) in enumerate(value_cols_map.items()):
        if actual_col not in df.columns or not pd.api.types.is_numeric_dtype(df[actual_col]):
            continue
        processed_actual_cols.append(actual_col)
        series_color = colors[i % len(colors)]
        series_name = get_lang_text(lang, display_key, actual_col.replace('_', ' ').title())
        unit = value_col_units_map.get(actual_col, "") if value_col_units_map else ""
        
        fig.add_trace(go.Scatter(
            x=df[date_col], y=df[actual_col], mode='lines+markers', name=series_name,
            line=dict(color=series_color, width=2), marker=dict(size=6, symbol="circle-open"), # More subtle markers
            hovertemplate=f"<b>{series_name}</b><br>{x_title_text}: %{{x|%b %d, %Y}}<br>{y_title_text}: %{{y:,.2f}}{unit}<extra></extra>"
        ))

    if rolling_avg_window and isinstance(rolling_avg_window, int) and rolling_avg_window > 0:
        for i, actual_col in enumerate(processed_actual_cols):
            if len(df) >= rolling_avg_window : # Check if enough data for rolling mean
                df[f'{actual_col}_rolling'] = df[actual_col].rolling(window=rolling_avg_window, center=True, min_periods=1).mean()
                display_key = [k for k,v in value_cols_map.items() if v == actual_col][0] # Find original display key
                series_name_base = get_lang_text(lang, display_key)
                rolling_name = f"{series_name_base} ({rolling_avg_window}-p MA)"
                unit = value_col_units_map.get(actual_col, "") if value_col_units_map else ""
                fig.add_trace(go.Scatter(
                    x=df[date_col], y=df[f'{actual_col}_rolling'], mode='lines', name=rolling_name,
                    line=dict(color=colors[i % len(colors)], width=1.5, dash='dashdot'), opacity=0.8,
                    hovertemplate=f"<b>{rolling_name}</b><br>{x_title_text}: %{{x|%b %d, %Y}}<br>{y_title_text}: %{{y:,.2f}}{unit}<extra></extra>"
                ))
    
    for i, actual_col in enumerate(processed_actual_cols):
        display_key = [k for k,v in value_cols_map.items() if v == actual_col][0]
        series_name = get_lang_text(lang, display_key)
        series_color = colors[i % len(colors)]
        if show_average_line:
            avg = df[actual_col].mean()
            if pd.notna(avg):
                fig.add_hline(y=avg, line_dash="dot", line_color=series_color, opacity=0.6,
                              annotation_text=f"{get_lang_text(lang, 'average_label')} {series_name}: {avg:,.1f}",
                              annotation_position="bottom right" if i%2==0 else "top left",
                              annotation_font_size=9, annotation_font_color=series_color, annotation_bgcolor="rgba(255,255,255,0.7)")
        if target_value_map and actual_col in target_value_map and pd.notna(target_value_map[actual_col]):
            target = target_value_map[actual_col]
            fig.add_hline(y=target, line_dash="dash", line_color=config.COLOR_TARGET_LINE, line_width=1.5,
                          annotation_text=f"{get_lang_text(lang, 'target_label')} {series_name}: {target:,.1f}",
                          annotation_position="top right" if i%2==0 else "bottom left",
                          annotation_font_size=9, annotation_font_color=config.COLOR_TARGET_LINE, annotation_bgcolor="rgba(255,255,255,0.7)")

    fig.update_layout(
        title=dict(text=title_text, x=0.5, font_size=18),
        yaxis_title=y_title_text, xaxis_title=x_title_text,
        legend_title_text=get_lang_text(lang, "metrics_legend"), hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.9)", font_size=11, bordercolor=config.COLOR_TEXT_SECONDARY, namelength=-1),
        xaxis=dict(showgrid=False, type='date',
                   rangeslider_visible=len(df)>10, # Show slider only for longer series
                   rangeselector=dict(buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"), dict(step="all")
                    ]), font=dict(size=10), y=1.05, x=0, xanchor='left')),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'), # Lighter grid
        legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5, font_size=10),
        margin=dict(l=50, r=20, t=100, b=30) # Adjusted for legend above
    )
    return fig

# --- Comparison Bar Chart ---
def create_comparison_bar_chart(df_input: pd.DataFrame, x_col: str,
                                y_cols_map: Dict[str, str], # {display_key: actual_col_name_in_df}
                                title_key: str, lang: str,
                                y_axis_title_key: str = "count_label",
                                x_axis_title_key: str = "category_axis_label",
                                barmode: str = 'group', show_total_for_stacked: bool = False,
                                data_label_format_str: str = ".0f") -> go.Figure:
    df = df_input.copy()
    title_text = get_lang_text(lang, title_key)
    x_title_text = get_lang_text(lang, x_axis_title_key)
    y_title_text = get_lang_text(lang, y_axis_title_key)

    # Create a new df for plotting with y-columns renamed to their display names
    df_plot = df[[x_col]].copy()
    y_display_names = []
    y_cols_for_sum = [] # Keep track of display names used for sum in stacked bar

    for display_key, actual_col in y_cols_map.items():
        if actual_col in df.columns and pd.api.types.is_numeric_dtype(df[actual_col]):
            display_name = get_lang_text(lang, display_key, actual_col.replace('_', ' ').title())
            df_plot[display_name] = df[actual_col] # Add column with display name
            y_display_names.append(display_name)
            y_cols_for_sum.append(display_name) # Use display name for sum if needed

    if df_plot.empty or x_col not in df_plot.columns or not y_display_names:
        return go.Figure().update_layout(title_text=f"{title_text} ({get_lang_text(lang, 'no_data_for_selection')})",
            annotations=[dict(text=get_lang_text(lang, 'no_data_for_selection'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)])

    fig = px.bar(df_plot, x=x_col, y=y_display_names, title=None, barmode=barmode,
                 color_discrete_sequence=config.COLOR_SCHEME_CATEGORICAL)

    fmt = data_label_format_str if isinstance(data_label_format_str, str) and data_label_format_str else ".0f"
    hover_fmt = fmt  # Can be different if needed, e.g. include currency
    
    try:
        fig.update_traces(
            texttemplate=f'%{{y:{fmt}}}',
            textposition='outside' if barmode != 'stack' else 'inside',
            textfont_size=9, insidetextanchor='middle' if barmode == 'stack' else 'auto',
            hovertemplate=f'<b>%{{x}}</b><br>%{{fullData.name}}: %{{y:{hover_fmt}}}<extra></extra>', # fullData.name uses legend entries
            selector=dict(type='bar')
        )
    except Exception as e: # Fallback simple hover if complex one fails
         print(f"Warning: update_traces failed in bar chart - {e}. Using simpler hover.")
         fig.update_traces(hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y}<extra></extra>", selector=dict(type='bar'))


    if barmode == 'stack' and show_total_for_stacked and y_cols_for_sum:
        df_plot['_total_'] = df_plot[y_cols_for_sum].sum(axis=1)
        annotations = [
            dict(x=row[x_col], y=row['_total_'], text=f"{row['_total_']:{fmt}}",
                 font=dict(size=9, color=config.COLOR_TEXT_SECONDARY),
                 showarrow=False, yanchor='bottom', yshift=2, xanchor='center')
            for _, row in df_plot.iterrows() if pd.notna(row['_total_'])
        ]
        if annotations: fig.update_layout(annotations=annotations)

    fig.update_layout(
        title=dict(text=title_text, x=0.5, font_size=18),
        yaxis_title=y_title_text, xaxis_title=x_title_text,
        legend_title_text=get_lang_text(lang, "metrics_legend"), hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.9)", font_size=11, namelength=-1),
        xaxis_tickangle=-30 if len(df_plot[x_col].unique()) > 7 else 0,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font_size=10),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
        xaxis=dict(showgrid=False, type='category'),
        margin=dict(l=50, r=20, t=70, b=50)
    )
    return fig

# --- Metric Card (UX Enhanced) ---
def display_metric_card(st_object, label_key: str, value: Optional[Union[int, float]], lang: str,
                        previous_value: Optional[Union[int, float]] = None, unit: str = "",
                        higher_is_better: Optional[bool] = None,
                        help_text_key: Optional[str] = None,
                        target_value: Optional[Union[int, float]] = None,
                        threshold_good: Optional[Union[int, float]] = None,
                        threshold_warning: Optional[Union[int, float]] = None):
    label_text_orig = get_lang_text(lang, label_key, label_key)
    help_text = get_lang_text(lang, help_text_key, "") if help_text_key else None
    
    val_display = "N/A"
    delta_str = None
    delta_clr = "normal"
    status_icon = "❓ " # Default for N/A

    if pd.notna(value) and isinstance(value, (int, float, np.number)):
        val_raw = float(value)
        # Value Formatting
        if unit == "%": val_display = f"{val_raw:,.1f}%"
        elif unit == get_lang_text(lang, 'days_unit') or (abs(val_raw) >= 1000 and val_raw % 1 == 0):
            val_display = f"{val_raw:,.0f}{(' ' + unit) if unit and unit != '%' else ''}"
        else: val_display = f"{val_raw:,.1f}{(' ' + unit) if unit and unit != '%' else ''}"
        if unit == "%" and not val_display.endswith("%"): val_display += "%" # Ensure % for percentage unit

        # Delta Calculation & Formatting
        if previous_value is not None and pd.notna(previous_value) and isinstance(previous_value, (int, float, np.number)):
            prev_val_raw = float(previous_value)
            delta_abs = val_raw - prev_val_raw
            sign = "+" if delta_abs >= 1e-9 else ("" if abs(delta_abs) < 1e-9 else "-")
            
            formatted_delta_abs_unit = unit if unit != '%' else '' # For absolute delta unit
            if unit == "%": formatted_delta_abs_str = f"{abs(delta_abs):.1f}%"
            else: formatted_delta_abs_str = f"{abs(delta_abs):,.1f}{(' '+formatted_delta_abs_unit) if formatted_delta_abs_unit else ''}"

            if abs(prev_val_raw) > 1e-9: # Avoid div by zero for percentage
                delta_perc = (delta_abs / abs(prev_val_raw)) * 100
                delta_str = f"{sign}{formatted_delta_abs_str} ({sign}{abs(delta_perc):.0f}%)"
            else:
                delta_str = f"{sign}{formatted_delta_abs_str} ({get_lang_text(lang, 'prev_period_label_short', 'Prev 0')})"

            if higher_is_better is not None:
                if delta_abs > 1e-9: delta_clr = "normal" if higher_is_better else "inverse"
                elif delta_abs < -1e-9: delta_clr = "inverse" if higher_is_better else "normal"
                else: delta_clr = "off"
        
        # Icon based on thresholds/target
        current_status = get_status_by_thresholds(val_raw, higher_is_better, threshold_good, threshold_warning)
        if current_status == "good": status_icon = "✅ "
        elif current_status == "warning": status_icon = "⚠️ "
        elif current_status == "critical": status_icon = "❗ "
        elif target_value is not None and higher_is_better is not None and pd.notna(target_value): # Fallback to target
            if (higher_is_better and val_raw >= float(target_value)) or \
               (not higher_is_better and val_raw <= float(target_value)):
                status_icon = "🎯 " # Target met
            else: status_icon = "⚠️ " # Target not met could also be a warning
        else: status_icon = "" # No icon if no clear status
            
    st_object.metric(label=status_icon + label_text_orig, value=val_display, delta=delta_str, delta_color=delta_clr, help=help_text)

# --- Radar Chart ---
def create_enhanced_radar_chart(df_radar_input: pd.DataFrame, categories_col: str, values_col: str,
                                 title_key: str, lang: str, group_col: Optional[str] = None,
                                 range_max_override: Optional[Union[int, float]] = None,
                                 target_values_map: Optional[Dict[str, Union[int, float]]] = None, # Keys are category names (localized)
                                 fill_opacity: float = 0.35):
    title_text = get_lang_text(lang, title_key)
    df_radar = df_radar_input.copy()

    if df_radar.empty or categories_col not in df_radar.columns or values_col not in df_radar.columns or df_radar[categories_col].nunique() == 0:
        return go.Figure().update_layout(title_text=f"{title_text} ({get_lang_text(lang, 'no_data_radar')})",
            annotations=[dict(text=get_lang_text(lang, 'no_data_radar'),showarrow=False, xref="paper", yref="paper", x=0.5,y=0.5)])

    all_categories = df_radar[categories_col].unique()
    
    # Calculate overall range for radial axis
    all_vals_for_range = df_radar[values_col].dropna().tolist()
    if target_values_map: all_vals_for_range.extend([v for v in target_values_map.values() if pd.notna(v) and isinstance(v, (int,float))])
    max_val_in_data = max(all_vals_for_range) if all_vals_for_range else 0
    
    radial_range_max = range_max_override if range_max_override is not None else (max_val_in_data * 1.2 if max_val_in_data > 0 else 5.0)
    radial_range_max = max(radial_range_max, 1.0) # Ensure at least 1.0

    fig = go.Figure()
    colors = config.COLOR_SCHEME_CATEGORICAL

    if group_col and group_col in df_radar.columns and df_radar[group_col].nunique() > 0:
        for i, (name, group) in enumerate(df_radar.groupby(group_col)):
            # Ensure all categories are present for consistent plotting for each group
            current_group_data = pd.DataFrame({categories_col: all_categories}).merge(
                group, on=categories_col, how='left'
            ).fillna({values_col: 0}) # Fill missing dimension values with 0 for this group
            fig.add_trace(go.Scatterpolar(
                r=current_group_data[values_col], theta=current_group_data[categories_col],
                fill='toself', name=str(name), line_color=colors[i % len(colors)], opacity=fill_opacity,
                hovertemplate='<b>%{theta}</b><br>' + f'{str(name)}: %{{r:.1f}}' + '<extra></extra>'
            ))
    else: # Single series
        # Ensure all categories are present and ordered correctly
        single_series_data = pd.DataFrame({categories_col: all_categories}).merge(
            df_radar, on=categories_col, how='left'
        ).fillna({values_col: 0})
        fig.add_trace(go.Scatterpolar(
            r=single_series_data[values_col], theta=single_series_data[categories_col],
            fill='toself', name=get_lang_text(lang, "average_score_label"), line_color=colors[0],
            opacity=fill_opacity + 0.1, hovertemplate='<b>%{theta}</b>: %{r:.1f}<extra></extra>'
        ))

    if target_values_map:
        target_vals_ordered = [target_values_map.get(cat, 0) for cat in all_categories] # Use ordered categories
        fig.add_trace(go.Scatterpolar(
            r=target_vals_ordered, theta=all_categories, mode='lines', name=get_lang_text(lang, "target_label"),
            line=dict(color=config.COLOR_TARGET_LINE, dash='longdash', width=2), hoverinfo='skip'
        ))

    fig.update_layout(
        title=dict(text=title_text, x=0.5, font_size=18),
        polar=dict(bgcolor="rgba(255,255,255,0.05)",
                   radialaxis=dict(visible=True, range=[0, radial_range_max], showline=False, gridcolor="rgba(0,0,0,0.1)", tickfont_size=9),
                   angularaxis=dict(showline=False, gridcolor="rgba(0,0,0,0.1)", tickfont_size=10, direction="clockwise")),
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font_size=9),
        margin=dict(l=40, r=40, t=80, b=40), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# --- Stress Semáforo Visual ---
def create_stress_semaforo_visual(stress_level_value: Optional[Union[int, float]], lang: str,
                                  scale_max: float = config.STRESS_LEVEL_MAX_SCALE) -> go.Figure:
    val_for_bar = 0.0
    actual_display_num = None # Number to show on indicator; None if stress_level_value is NaN
    color_for_status = config.COLOR_TEXT_SECONDARY # Default for N/A
    text_for_status = get_lang_text(lang, 'status_na_label', 'N/A')

    if pd.notna(stress_level_value) and isinstance(stress_level_value, (int, float, np.number)):
        val_for_bar = float(stress_level_value)
        actual_display_num = float(stress_level_value)
        
        # Determine status and color using thresholds for "higher is worse"
        status = get_status_by_thresholds(val_for_bar, higher_is_worse=True,
                                           threshold_good=config.STRESS_LEVEL_PSYCHOSOCIAL["low"],
                                           threshold_warning=config.STRESS_LEVEL_PSYCHOSOCIAL["medium"])
        if status == "good": text_for_status, color_for_status = get_lang_text(lang, 'low_label'), config.COLOR_STATUS_GOOD
        elif status == "warning": text_for_status, color_for_status = get_lang_text(lang, 'moderate_label'), config.COLOR_STATUS_WARNING
        elif status == "critical": text_for_status, color_for_status = get_lang_text(lang, 'high_label'), config.COLOR_STATUS_CRITICAL
        else: text_for_status = f"{get_lang_text(lang, 'value_axis_label')}: {val_for_bar:.1f}" # Fallback
            
    # Config for the number display part of the indicator
    number_display_config = {
        'font': {'size': 22, 'color': color_for_status},
        'valueformat': ".1f"
    }
    if actual_display_num is not None: # Add suffix only if number is valid
        number_display_config['suffix'] = f" / {scale_max:.0f}"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=actual_display_num, # Value displayed by the number part, Plotly handles None
        domain={'x': [0, 1], 'y': [0.2, 0.8]},
        title={'text': f"<b style='color:{color_for_status}; font-size:1.1em;'>{text_for_status}</b>", 'font': {'size': 16}, 'align': "center"},
        number=number_display_config,
        gauge={
            'shape': "bullet",
            'axis': {'range': [0, scale_max], 'visible': True, 'showticklabels': True,
                     'tickvals': [0, config.STRESS_LEVEL_PSYCHOSOCIAL["low"], config.STRESS_LEVEL_PSYCHOSOCIAL["medium"], scale_max],
                     'ticktext': ["0", f"{config.STRESS_LEVEL_PSYCHOSOCIAL['low']:.0f}", f"{config.STRESS_LEVEL_PSYCHOSOCIAL['medium']:.0f}", f"{scale_max:.0f}"],
                     'tickfont': {'size':9, 'color': config.COLOR_TEXT_SECONDARY}, 'tickmode': 'array'},
            'steps': [
                {'range': [0, config.STRESS_LEVEL_PSYCHOSOCIAL["low"]], 'color': config.COLOR_STATUS_GOOD},
                {'range': [config.STRESS_LEVEL_PSYCHOSOCIAL["low"], config.STRESS_LEVEL_PSYCHOSOCIAL["medium"]], 'color': config.COLOR_STATUS_WARNING},
                {'range': [config.STRESS_LEVEL_PSYCHOSOCIAL["medium"], scale_max], 'color': config.COLOR_STATUS_CRITICAL}
            ],
            # The 'bar' shows the current value on the scale
            'bar': {'color': color_for_status if pd.notna(actual_display_num) else 'rgba(200,200,200,0.7)', 'thickness': 0.5},
            'bgcolor': "rgba(255,255,255,0.7)", 'borderwidth': 0.5, 'bordercolor': "lightgray"
        }))
    fig.update_layout(height=110, margin=dict(t=15, b=20, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)')
    return fig
