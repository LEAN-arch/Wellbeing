# visualizations.py
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import config

# --- Helper to get localized text ---
def get_lang_text(lang_code, key, default_text=""):
    """Retrieves localized text safely."""
    text_dict = config.TEXT_STRINGS.get(lang_code, config.TEXT_STRINGS["EN"])
    return text_dict.get(key, default_text)

# --- Helper for getting correct status text based on thresholds ---
def get_status_by_thresholds(value, higher_is_worse, threshold_good=None, threshold_warning=None):
    """Determines 'good', 'warning', 'critical' status based on value and thresholds."""
    if pd.isna(value) or value is None:
        return None

    val_float = float(value)
    good_thresh = float(threshold_good) if threshold_good is not None and pd.notna(threshold_good) else None
    warn_thresh = float(threshold_warning) if threshold_warning is not None and pd.notna(threshold_warning) else None

    if higher_is_worse: # Lower is better (e.g., rotation, incidents, stress)
        if good_thresh is not None and val_float <= good_thresh: return "good"
        if warn_thresh is not None and val_float <= warn_thresh: return "warning" # Assumes warn_thresh > good_thresh
        if warn_thresh is not None and val_float > warn_thresh : return "critical"
        if good_thresh is not None and warn_thresh is None and val_float > good_thresh: return "critical" # Only good defined, anything above is critical
    else: # Higher is better (e.g., retention, eNPS)
        if good_thresh is not None and val_float >= good_thresh: return "good"
        if warn_thresh is not None and val_float >= warn_thresh: return "warning" # Assumes warn_thresh < good_thresh
        if warn_thresh is not None and val_float < warn_thresh: return "critical"
        if good_thresh is not None and warn_thresh is None and val_float < good_thresh: return "critical" # Only good defined, anything below is critical
    return None # No clear status if thresholds are ambiguous or value is between non-contiguous thresholds


def get_semaforo_color(status):
    """Maps status string to configured color."""
    if status == "good": return config.COLOR_GREEN_SEMAFORO
    if status == "warning": return config.COLOR_YELLOW_SEMAFORO
    if status == "critical": return config.COLOR_RED_SEMAFORO
    return config.COLOR_GRAY_TEXT

# --- Enhanced KPI Gauge ---
def create_kpi_gauge(value, title_key, lang, unit="%", higher_is_worse=True,
                     threshold_good=None, threshold_warning=None,
                     target_line_value=None, max_value_override=None,
                     previous_value=None):
    title_text = get_lang_text(lang, title_key, title_key)
    
    current_value_for_gauge = 0.0
    delta_ref = None
    
    if pd.notna(value) and isinstance(value, (int, float)):
        current_value_for_gauge = float(value)
        if previous_value is not None and pd.notna(previous_value):
            delta_ref = float(previous_value)

    if max_value_override is not None:
        max_val = float(max_value_override)
    else:
        max_val_candidates = [1.0]
        if pd.notna(current_value_for_gauge): max_val_candidates.append(current_value_for_gauge * 1.25)
        
        # Use valid numeric thresholds for max_val calculation
        valid_thresh_for_max = [t for t in [threshold_good, threshold_warning, target_line_value] if t is not None and pd.notna(t)]
        if valid_thresh_for_max:
             max_val_candidates.append(max(float(t) for t in valid_thresh_for_max) * 1.2)
        else: # No thresholds, and value might be 0 or NaN
            if not pd.notna(current_value_for_gauge) or current_value_for_gauge == 0:
                max_val_candidates.append(100.0)


        max_val = max(max_val_candidates) if max_val_candidates else 100.0
        if max_val <=0 : max_val = 100.0 # Safety for 0 or negative max
        if pd.notna(current_value_for_gauge) and current_value_for_gauge > max_val and max_value_override is None:
            max_val = current_value_for_gauge * 1.1

    steps = []
    num_t_good = float(threshold_good) if threshold_good is not None and pd.notna(threshold_good) else None
    num_t_warn = float(threshold_warning) if threshold_warning is not None and pd.notna(threshold_warning) else None

    current_range_start = 0.0
    if higher_is_worse:
        if num_t_good is not None:
            steps.append({'range': [current_range_start, num_t_good], 'color': config.COLOR_GREEN_SEMAFORO})
            current_range_start = num_t_good
        if num_t_warn is not None and num_t_warn > current_range_start: # Ensure warning is above good
            steps.append({'range': [current_range_start, num_t_warn], 'color': config.COLOR_YELLOW_SEMAFORO})
            current_range_start = num_t_warn
        steps.append({'range': [current_range_start, max_val], 'color': config.COLOR_RED_SEMAFORO})
    else: # Lower is worse (e.g. retention higher is better)
        if num_t_warn is not None: # Here, warning is the lower undesirable bound
            steps.append({'range': [current_range_start, num_t_warn], 'color': config.COLOR_RED_SEMAFORO})
            current_range_start = num_t_warn
        if num_t_good is not None and num_t_good > current_range_start: # Good is above warning
            steps.append({'range': [current_range_start, num_t_good], 'color': config.COLOR_YELLOW_SEMAFORO})
            current_range_start = num_t_good
        steps.append({'range': [current_range_start, max_val], 'color': config.COLOR_GREEN_SEMAFORO})

    if not steps: steps.append({'range': [0, max_val], 'color': config.COLOR_NEUTRAL_METRIC})
    
    num_target_line = float(target_line_value) if target_line_value is not None and pd.notna(target_line_value) else None

    fig = go.Figure(go.Indicator(
        mode="gauge+number" + ("+delta" if delta_ref is not None else ""),
        value=current_value_for_gauge if pd.notna(current_value_for_gauge) else 0,
        title={'text': title_text, 'font': {'size': 16, 'color': config.COLOR_GRAY_TEXT}},
        number={'font': {'size': 28}, 'suffix': unit if unit and unit != "N/A" else "", 'valueformat': ".1f"},
        delta={'reference': delta_ref,
               'increasing': {'color': config.COLOR_RED_SEMAFORO if higher_is_worse else config.COLOR_GREEN_SEMAFORO},
               'decreasing': {'color': config.COLOR_GREEN_SEMAFORO if higher_is_worse else config.COLOR_RED_SEMAFORO},
               'font': {'size': 16}, 'valueformat': ".1f"},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1, 'tickcolor': "darkgray", 'nticks': 5},
            'bar': {'color': 'rgba(0,0,0,0.5)', 'thickness': 0.15},
            'bgcolor': "white", 'borderwidth': 1, 'bordercolor': "lightgray",
            'steps': steps,
            'threshold': {
                'line': {'color': config.COLOR_TARGET_LINE, 'width': 3},
                'thickness': 0.8, 'value': num_target_line
            } if num_target_line is not None else {}
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=5), paper_bgcolor='white')
    return fig

# --- Enhanced Trend Chart with Annotations and Selectors ---
def create_trend_chart(df, date_col, value_cols_map, title_key, lang, # value_cols_map: {'Series Name Key': 'actual_col_name'}
                       y_axis_title_key="value_axis_label", x_axis_title_key="date_time_axis_label",
                       show_average_line=False, target_value_map=None, # target_value_map: {'Actual Col Name': target_val}
                       rolling_avg_window=None, value_col_units_map=None): # value_col_units_map: {'Actual Col Name': unit_str}
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = get_lang_text(lang, title_key, title_key)
    x_title_text = get_lang_text(lang, x_axis_title_key, "Date/Time")
    y_title_text = get_lang_text(lang, y_axis_title_key, "Value")

    if df.empty or date_col not in df.columns or not value_cols_map:
        return go.Figure().update_layout(
            title_text=f"{title_text} ({lang_texts.get('no_data_for_selection', 'No data')})",
            annotations=[dict(text=lang_texts.get('no_data_for_selection', 'No data'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5, font_size=12)]
        )

    fig = go.Figure()
    colors = config.COLOR_SCHEME_CATEGORICAL
    processed_value_cols = []

    # Main series lines
    for i, (series_name_key, actual_col_name) in enumerate(value_cols_map.items()):
        if actual_col_name not in df.columns or not pd.api.types.is_numeric_dtype(df[actual_col_name]):
            continue # Skip if column is missing or not numeric
        processed_value_cols.append(actual_col_name)

        series_color = colors[i % len(colors)]
        series_name = get_lang_text(lang, series_name_key, actual_col_name.replace('_', ' ').title())
        unit_suffix = value_col_units_map.get(actual_col_name, "") if value_col_units_map else ""

        fig.add_trace(go.Scatter(
            x=df[date_col], y=df[actual_col_name], mode='lines+markers', name=series_name,
            line=dict(color=series_color, width=2), marker=dict(size=5, symbol="circle"),
            hovertemplate=f"<b>{series_name}</b><br>{x_title_text}: %{{x|%Y-%m-%d}}<br>{y_title_text}: %{{y:.2f}}{unit_suffix}<extra></extra>"
        ))

    # Rolling average lines
    if rolling_avg_window and isinstance(rolling_avg_window, int) and rolling_avg_window > 1:
        for i, actual_col_name in enumerate(processed_value_cols):
            series_color = colors[i % len(colors)]
            series_name_key = next((k for k, v in value_cols_map.items() if v == actual_col_name), actual_col_name) # Get key back
            base_series_name = get_lang_text(lang, series_name_key, actual_col_name.replace('_', ' ').title())
            rolling_avg_name = f"{base_series_name} ({rolling_avg_window}-p MA)" # p for period
            unit_suffix = value_col_units_map.get(actual_col_name, "") if value_col_units_map else ""

            if len(df) >= rolling_avg_window:
                df_copy = df.copy()
                df_copy[f'{actual_col_name}_rolling_avg'] = df_copy[actual_col_name].rolling(window=rolling_avg_window, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=df_copy[date_col], y=df_copy[f'{actual_col_name}_rolling_avg'], mode='lines',
                    name=rolling_avg_name,
                    line=dict(color=series_color, width=1.5, dash='dot'), # Changed dash
                    opacity=0.7,
                    hovertemplate=f"<b>{rolling_avg_name}</b><br>{x_title_text}: %{{x|%Y-%m-%d}}<br>{y_title_text}: %{{y:.2f}}{unit_suffix}<extra></extra>"
                ))

    # Static lines (Average and Target)
    for i, actual_col_name in enumerate(processed_value_cols):
        series_color = colors[i % len(colors)]
        series_name_key = next((k for k, v in value_cols_map.items() if v == actual_col_name), actual_col_name)
        series_name = get_lang_text(lang, series_name_key, actual_col_name.replace('_', ' ').title())

        if show_average_line:
            avg_val = df[actual_col_name].mean()
            if pd.notna(avg_val):
                fig.add_hline(y=avg_val, line_dash="longdash",
                              annotation_text=f"{get_lang_text(lang, 'average_label', 'Avg')} {series_name}: {avg_val:.1f}",
                              annotation_position="bottom right" if i % 2 == 0 else "top left",
                              line_color=series_color, opacity=0.5,
                              annotation=dict(font=dict(size=9, color=series_color), bgcolor="rgba(255,255,255,0.8)", bordercolor=series_color, borderwidth=0.5))

        if target_value_map and actual_col_name in target_value_map and pd.notna(target_value_map[actual_col_name]):
            target_val = target_value_map[actual_col_name]
            fig.add_hline(y=target_val, line_dash="dash",
                          annotation_text=f"{get_lang_text(lang, 'target_label', 'Target')} {series_name}: {target_val:.1f}",
                          annotation_position="top right" if i % 2 == 0 else "bottom left",
                          line_color=config.COLOR_TARGET_LINE, line_width=1.5,
                          annotation=dict(font=dict(size=9, color=config.COLOR_TARGET_LINE), bgcolor="rgba(255,255,255,0.8)", bordercolor=config.COLOR_TARGET_LINE, borderwidth=0.5))


    fig.update_layout(
        title=dict(text=title_text, x=0.5),
        yaxis_title=y_title_text,
        xaxis_title=x_title_text,
        legend_title_text=get_lang_text(lang, "metrics_legend", "Metrics"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=config.COLOR_GRAY_TEXT),
        xaxis=dict(
            showgrid=False, type='date', # Ensure x-axis is treated as date
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ]),
                font=dict(size=10)
            )
        ),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(l=50, r=30, t=80, b=30)
    )
    return fig

# --- Enhanced Comparison Bar Chart ---
def create_comparison_bar_chart(df, x_col, y_cols_map, title_key, lang, # y_cols_map: {'Series Name Key': 'actual_col_name'}
                                y_axis_title_key="count_axis_label", x_axis_title_key="category_axis_label",
                                barmode='group', show_total_for_stacked=False, data_label_format_str=".0f"):
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = get_lang_text(lang, title_key, title_key)
    x_title_text = get_lang_text(lang, x_axis_title_key, "Category")
    y_title_text = get_lang_text(lang, y_axis_title_key, "Count")

    y_actual_col_names = [col_name for col_name in y_cols_map.values() if col_name in df.columns and pd.api.types.is_numeric_dtype(df[col_name])]
    
    if df.empty or x_col not in df.columns or not y_actual_col_names:
        return go.Figure().update_layout(
            title_text=f"{title_text} ({lang_texts.get('no_data_for_selection', 'No data')})",
            annotations=[dict(text=lang_texts.get('no_data_for_selection', 'No data'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5, font_size=12)]
        )

    # Create df_for_plot with display names for y-columns, makes Plotly Express legend nicer
    df_for_plot = df[[x_col] + y_actual_col_names].copy()
    display_y_cols = []
    rename_map = {}
    for series_key, actual_col in y_cols_map.items():
        if actual_col in y_actual_col_names:
            display_name = get_lang_text(lang, series_key, actual_col.replace('_',' ').title())
            display_y_cols.append(display_name)
            rename_map[actual_col] = display_name
    df_for_plot.rename(columns=rename_map, inplace=True)
    
    fig = px.bar(df_for_plot, x=x_col, y=display_y_cols, title=None, barmode=barmode,
                 color_discrete_sequence=config.COLOR_SCHEME_CATEGORICAL
                 )

    fig.update_traces(
        texttemplate=f'%{{y:{data_label_format_str}}}',
        textposition='outside' if barmode != 'stack' else 'inside',
        textfont_size=9, # Smaller for less clutter
        insidetextanchor='middle' if barmode == 'stack' else 'auto',
        hovertemplate=f'<b>%{{x}}</b><br>%{{fullData.name}}: %{{y:{data_label_format_str}}}<extra></extra>'
    )

    if barmode == 'stack' and show_total_for_stacked and display_y_cols:
        # Calculate total using the display_y_cols names in df_for_plot
        df_for_plot['_total_stacked_'] = df_for_plot[display_y_cols].sum(axis=1)
        
        annotations = []
        # Iterate using df_for_plot which has the x_col and _total_stacked_
        for _, row in df_for_plot.iterrows():
            x_val = row[x_col]
            total_val = row['_total_stacked_']
            if pd.notna(total_val):
                 annotations.append(dict(
                                    x=x_val, y=total_val,
                                    text=f"{total_val:{data_label_format_str}}",
                                    font=dict(size=10, color=config.COLOR_GRAY_TEXT), # Totals slightly dimmer
                                    showarrow=False, yanchor='bottom', yshift=3, xanchor='center'
                                    ))
        fig.update_layout(annotations=annotations)

    fig.update_layout(
        title=dict(text=title_text, x=0.5, font_size=18),
        yaxis_title=y_title_text,
        xaxis_title=x_title_text,
        legend_title_text=get_lang_text(lang, "metrics_legend", "Metrics"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=12, namelength=-1), # Show full legend item name
        xaxis_tickangle=-30 if len(df_for_plot[x_col].unique()) > 6 else 0, # Use df_for_plot
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=9)),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)'),
        xaxis=dict(showgrid=False, type='category') # Critical for months/text
    )
    return fig

# --- Enhanced Metric Card ---
def display_metric_card(st_object, label_key, value, lang,
                        previous_value=None, unit="", higher_is_better=None,
                        help_text_key=None, target_value=None,
                        threshold_good=None, threshold_warning=None):
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    label_text_orig = get_lang_text(lang, label_key, label_key)
    help_text_content = get_lang_text(lang, help_text_key, "") if help_text_key else None

    formatted_value = "N/A"
    delta_text = None
    delta_color = "normal"
    icon = ""

    if pd.notna(value) and isinstance(value, (int, float, np.number)): # Accept numpy numbers too
        raw_value = float(value)

        val_to_format = raw_value
        if unit == "%":
            formatted_value = f"{val_to_format:,.1f}%"
        elif unit == get_lang_text(lang, 'days_label') or (abs(val_to_format) >= 1000 and val_to_format % 1 == 0):
            formatted_value = f"{val_to_format:,.0f}{(' ' + unit) if unit else ''}"
        elif abs(val_to_format) < 1 and abs(val_to_format) > 0: # For small decimals
            formatted_value = f"{val_to_format:,.2f}{(' ' + unit) if unit else ''}"
        else:
            formatted_value = f"{val_to_format:,.1f}{(' ' + unit) if unit else ''}"
        
        # Delta
        if previous_value is not None and pd.notna(previous_value) and isinstance(previous_value, (int,float, np.number)):
            prev_raw_value = float(previous_value)
            delta_abs = raw_value - prev_raw_value
            sign = "+" if delta_abs >= 1e-9 else ("" if abs(delta_abs) < 1e-9 else "-")
            
            delta_formatted_abs = f"{abs(delta_abs):,.1f}{unit if unit != '%' else ''}" # Exclude % for unit consistency in delta string
            if unit=="%": delta_formatted_abs += "%"


            if abs(prev_raw_value) > 1e-9 :
                 delta_percentage = (delta_abs / abs(prev_raw_value)) * 100
                 delta_text = f"{sign}{delta_formatted_abs} ({sign}{abs(delta_percentage):,.0f}%)"
            else:
                 delta_text = f"{sign}{delta_formatted_abs} ({_('prev_period_label_short','Prev 0')})" # Use localized "Prev 0"

            if higher_is_better is not None:
                if delta_abs > 1e-9 : delta_color = "normal" if higher_is_better else "inverse"
                elif delta_abs < -1e-9 : delta_color = "inverse" if higher_is_better else "normal"
                else: delta_color = "off"
        
        status = get_status_by_thresholds(raw_value, higher_is_better, threshold_good, threshold_warning)
        if status == "good": icon = "✅ "
        elif status == "warning": icon = "⚠️ "
        elif status == "critical": icon = "❗ "
        elif target_value is not None and higher_is_better is not None:
            if (higher_is_better and raw_value >= float(target_value)) or \
               (not higher_is_better and raw_value <= float(target_value)):
                icon = "🎯 " # Icon for hitting a general target
    
    elif pd.isna(value) or value is None:
         formatted_value = "N/A"
         icon = "❓ "

    st_object.metric(label=icon + label_text_orig, value=formatted_value, delta=delta_text, delta_color=delta_color, help=help_text_content)


# --- Enhanced Radar Chart ---
def create_enhanced_radar_chart(df_radar_input, categories_col, values_col, title_key, lang,
                                 group_col=None, range_max_override=None, target_values_map=None,
                                 fill_opacity=0.4): # Reduced default fill opacity for better target visibility
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = get_lang_text(lang, title_key, title_key)
    df_radar = df_radar_input.copy()

    if df_radar.empty or categories_col not in df_radar.columns or values_col not in df_radar.columns or df_radar[categories_col].nunique() == 0:
        return go.Figure().update_layout(
            title_text=f"{title_text} ({lang_texts.get('no_data_radar', 'No data')})",
            annotations=[dict(text=lang_texts.get('no_data_radar', 'No data for radar chart'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5, font_size=12)]
        )

    all_r_values = df_radar[values_col].dropna().tolist()
    if target_values_map: # target_values_map uses display names as keys
        all_r_values.extend([v for v in target_values_map.values() if pd.notna(v) and isinstance(v, (int,float))])
    
    valid_r_values = [v for v in all_r_values if isinstance(v, (int, float))]
    current_max_val = max(valid_r_values) if valid_r_values else 0
    
    range_max = range_max_override if range_max_override is not None else (current_max_val * 1.15 if current_max_val > 0 else 5.0)
    range_max = max(range_max, 1.0) # Minimum range of 1

    fig = go.Figure()
    color_sequence = config.COLOR_SCHEME_CATEGORICAL
    all_categories_ordered = df_radar[categories_col].unique()

    if group_col and group_col in df_radar.columns and df_radar[group_col].nunique() > 0:
        for i, (group_name, group_data) in enumerate(df_radar.groupby(group_col)):
            group_df_ordered = pd.DataFrame({categories_col: all_categories_ordered})
            group_df_ordered = pd.merge(group_df_ordered, group_data, on=categories_col, how='left').fillna({values_col: 0})
            fig.add_trace(go.Scatterpolar(
                r=group_df_ordered[values_col], theta=group_df_ordered[categories_col],
                fill='toself', name=str(group_name),
                line_color=color_sequence[i % len(color_sequence)], opacity=fill_opacity,
                hovertemplate='<b>%{theta}</b><br>' + f'{str(group_name)}: %{{r:.1f}}' + '<extra></extra>'
            ))
    else:
        df_radar_ordered = pd.DataFrame({categories_col: all_categories_ordered})
        df_radar_ordered = pd.merge(df_radar_ordered, df_radar, on=categories_col, how='left').fillna({values_col: 0})
        fig.add_trace(go.Scatterpolar(
            r=df_radar_ordered[values_col], theta=df_radar_ordered[categories_col],
            fill='toself', name=get_lang_text(lang, "average_score_label", "Average Score"),
            line_color=color_sequence[0], opacity=fill_opacity + 0.1,
            hovertemplate='<b>%{theta}</b>: %{r:.1f}<extra></extra>'
        ))
    
    if target_values_map:
        target_r_values = [target_values_map.get(cat, 0) for cat in all_categories_ordered]
        fig.add_trace(go.Scatterpolar(
            r=target_r_values, theta=all_categories_ordered, mode='lines',
            name=get_lang_text(lang, "target_label", "Target"),
            line=dict(color=config.COLOR_TARGET_LINE, dash='dash', width=2),
            hoverinfo='skip'
        ))

    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=18)),
        polar=dict(
            bgcolor="rgba(255,255,255,0.0)",
            radialaxis=dict(
                visible=True, range=[0, range_max], showline=False, showticklabels=True,
                gridcolor="rgba(0,0,0,0.15)", linecolor="rgba(0,0,0,0.15)", tickfont=dict(size=9)
            ),
            angularaxis=dict(
                showline=False, showticklabels=True, gridcolor="rgba(0,0,0,0.15)",
                linecolor="rgba(0,0,0,0.15)", tickfont=dict(size=10), direction="clockwise"
            )
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font=dict(size=9)),
        margin=dict(l=50, r=50, t=100, b=60), # Make space for long category labels on radar
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


# --- Enhanced Stress Semáforo (Visual Bar) ---
def create_stress_semaforo_visual(stress_level, lang, scale_max=config.STRESS_LEVEL_MAX_SCALE):
    title_text_unused = get_lang_text(lang, "overall_stress_indicator_title", "Average Stress Level") # Title is within indicator

    val_for_gauge = 0.0
    raw_value = stress_level
    semaforo_color = config.COLOR_GRAY_TEXT
    status_text = get_lang_text(lang, 'stress_na', 'N/A')

    if pd.notna(raw_value) and isinstance(raw_value, (int, float, np.number)):
        val_for_gauge = float(raw_value)
        status = get_status_by_thresholds(val_for_gauge, higher_is_worse=True,
                                           threshold_good=config.STRESS_LEVEL_THRESHOLD_LOW,
                                           threshold_warning=config.STRESS_LEVEL_THRESHOLD_MEDIUM)
        if status == "good": status_text, semaforo_color = get_lang_text(lang, 'low_label', 'Low'), config.COLOR_GREEN_SEMAFORO
        elif status == "warning": status_text, semaforo_color = get_lang_text(lang, 'moderate_label', 'Moderate'), config.COLOR_YELLOW_SEMAFORO
        elif status == "critical": status_text, semaforo_color = get_lang_text(lang, 'high_label', 'High'), config.COLOR_RED_SEMAFORO
        else: status_text = f"{get_lang_text(lang, 'value_axis_label')}: {val_for_gauge:.1f}"

    gauge_value_clamped = max(0.0, min(float(scale_max), val_for_gauge)) if pd.notna(val_for_gauge) else 0.0
    # Displayed number for indicator should be the raw value, not clamped, unless it's NaN
    display_number_value = val_for_gauge if pd.notna(val_for_gauge) else None


    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_value_clamped, # This is the value that positions the bar in steps
        domain={'x': [0, 1], 'y': [0.2, 0.8]},
        title={'text': f"<b style='color:{semaforo_color}; font-size:1.1em;'>{status_text}</b>", 'font': {'size': 16}, 'align': "center"},
        number={
            'valueformat': ".1f" if display_number_value is not None else "",
            'font': {'size': 22, 'color': semaforo_color},
            'suffix': f" / {scale_max:.0f}" if display_number_value is not None else "" , # Suffix shown only if number is displayed
            'value': display_number_value # The number displayed can be the actual stress_level
        },
        gauge={
            'shape': "bullet",
            'axis': {'range': [0, scale_max], 'visible': True, 'showticklabels': True,
                     'tickvals': [0, config.STRESS_LEVEL_THRESHOLD_LOW, config.STRESS_LEVEL_THRESHOLD_MEDIUM, scale_max],
                     'ticktext': ["0", f"{config.STRESS_LEVEL_THRESHOLD_LOW:.0f}", f"{config.STRESS_LEVEL_THRESHOLD_MEDIUM:.0f}", f"{scale_max:.0f}"],
                     'tickfont': {'size':9, 'color': config.COLOR_GRAY_TEXT}, 'tickmode': 'array'
            },
            'steps': [
                {'range': [0, config.STRESS_LEVEL_THRESHOLD_LOW], 'color': config.COLOR_GREEN_SEMAFORO, 'name':get_lang_text(lang, 'low_label', 'Low')},
                {'range': [config.STRESS_LEVEL_THRESHOLD_LOW, config.STRESS_LEVEL_THRESHOLD_MEDIUM], 'color': config.COLOR_YELLOW_SEMAFORO, 'name': get_lang_text(lang, 'moderate_label', 'Moderate')},
                {'range': [config.STRESS_LEVEL_THRESHOLD_MEDIUM, scale_max], 'color': config.COLOR_RED_SEMAFORO, 'name': get_lang_text(lang, 'high_label', 'High')}
            ],
            'bar': {'color': 'rgba(50,50,50,0.8)', 'thickness': 0.3}, # Darker bar indicates actual value position on the colored scale
            'bgcolor': "rgba(255,255,255,0.8)", 'borderwidth': 1, 'bordercolor': "lightgray"
        }))
    fig.update_layout(height=110, margin=dict(t=15, b=15, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)') # Slightly less height
    return fig
