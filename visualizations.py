# visualizations.py
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import config

# --- Helper to get localized text ---
def get_lang_text(lang_code, key, default_text=""):
    """Retrieves localized text safely."""
    text_dict = config.TEXT_STRINGS.get(lang_code, config.TEXT_STRINGS["EN"]) # Fallback to EN
    return text_dict.get(key, default_text)

# --- Helper for getting correct status text based on thresholds ---
def get_status_by_thresholds(value, higher_is_worse, threshold_good=None, threshold_warning=None):
    """Determines 'good', 'warning', 'critical' status based on value and thresholds."""
    if pd.isna(value) or value is None:
        return None # Indicate no specific status

    # Ensure thresholds are numeric before comparison
    num_threshold_good = float(threshold_good) if threshold_good is not None and pd.notna(threshold_good) else None
    num_threshold_warning = float(threshold_warning) if threshold_warning is not None and pd.notna(threshold_warning) else None
    val = float(value)

    if higher_is_worse: # Lower value is better
        if num_threshold_good is not None and val <= num_threshold_good:
            return "good"
        # Check warning only if good threshold is defined and value is greater than it, or if no good threshold is defined
        if num_threshold_warning is not None and (num_threshold_good is None or val > num_threshold_good) and val <= num_threshold_warning:
            return "warning"
        # If value is greater than warning threshold, or greater than good threshold (if warning is not defined)
        if (num_threshold_warning is not None and val > num_threshold_warning) or \
           (num_threshold_warning is None and num_threshold_good is not None and val > num_threshold_good):
            return "critical"
    else: # Higher value is better
        if num_threshold_good is not None and val >= num_threshold_good:
            return "good"
        if num_threshold_warning is not None and (num_threshold_good is None or val < num_threshold_good) and val >= num_threshold_warning:
            return "warning"
        if (num_threshold_warning is not None and val < num_threshold_warning) or \
           (num_threshold_warning is None and num_threshold_good is not None and val < num_threshold_good):
            return "critical"
    return None # No specific status if thresholds don't define a clear zone

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

    # Determine dynamic max_value
    if max_value_override is not None:
        max_val = float(max_value_override)
    else:
        max_val_candidates = [1.0]
        if pd.notna(current_value_for_gauge): max_val_candidates.append(current_value_for_gauge * 1.25)
        if threshold_warning is not None and pd.notna(threshold_warning): max_val_candidates.append(float(threshold_warning) * 1.2)
        elif threshold_good is not None and pd.notna(threshold_good): max_val_candidates.append(float(threshold_good) * (1.5 if higher_is_worse else 1.3))
        if target_line_value is not None and pd.notna(target_line_value): max_val_candidates.append(float(target_line_value) * 1.1)
        
        max_val = max(max_val_candidates) if max_val_candidates else 100.0
        if max_val <=0 : max_val = 100.0
        if pd.notna(current_value_for_gauge) and current_value_for_gauge > max_val and max_value_override is None:
            max_val = current_value_for_gauge * 1.1

    steps = []
    # Ensure thresholds are numeric for step calculation
    num_threshold_good = float(threshold_good) if threshold_good is not None and pd.notna(threshold_good) else None
    num_threshold_warning = float(threshold_warning) if threshold_warning is not None and pd.notna(threshold_warning) else None

    current_range_start = 0.0
    if higher_is_worse: # Green -> Yellow -> Red
        if num_threshold_good is not None:
            steps.append({'range': [current_range_start, num_threshold_good], 'color': config.COLOR_GREEN_SEMAFORO})
            current_range_start = num_threshold_good
        if num_threshold_warning is not None and (num_threshold_good is None or num_threshold_warning > current_range_start):
            steps.append({'range': [current_range_start, num_threshold_warning], 'color': config.COLOR_YELLOW_SEMAFORO})
            current_range_start = num_threshold_warning
        steps.append({'range': [current_range_start, max_val], 'color': config.COLOR_RED_SEMAFORO})
    else: # Lower is worse: Red -> Yellow -> Green
        if num_threshold_warning is not None: # This warning threshold is the upper limit of "bad"
            steps.append({'range': [current_range_start, num_threshold_warning], 'color': config.COLOR_RED_SEMAFORO})
            current_range_start = num_threshold_warning
        if num_threshold_good is not None and (num_threshold_warning is None or num_threshold_good > current_range_start):
            steps.append({'range': [current_range_start, num_threshold_good], 'color': config.COLOR_YELLOW_SEMAFORO})
            current_range_start = num_threshold_good
        steps.append({'range': [current_range_start, max_val], 'color': config.COLOR_GREEN_SEMAFORO})

    if not steps: # Fallback if no thresholds provided
        steps.append({'range': [0, max_val], 'color': config.COLOR_NEUTRAL_METRIC})

    # Determine gauge's own threshold line
    gauge_main_threshold = target_line_value
    if gauge_main_threshold is None: # Fallback if no specific target
        gauge_main_threshold = num_threshold_warning if num_threshold_warning is not None else num_threshold_good
        
    gauge_indicator = go.Indicator(
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
            'bar': {'color': 'rgba(0,0,0,0.6)', 'thickness': 0.2},
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "lightgray",
            'steps': steps,
            'threshold': {
                'line': {'color': config.COLOR_TARGET_LINE, 'width': 3},
                'thickness': 0.8,
                'value': float(gauge_main_threshold)
            } if gauge_main_threshold is not None and pd.notna(gauge_main_threshold) else {}
        }
    )
    fig = go.Figure(gauge_indicator)
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=5), paper_bgcolor='white')
    return fig

# --- Enhanced Trend Chart with Annotations and Selectors ---
def create_trend_chart(df, date_col, value_cols, title_key, lang,
                       y_axis_title_key="value_axis_label", x_axis_title_key="date_time_axis_label",
                       show_average_line=False, target_value_map=None, highlight_peaks_dips=False,
                       rolling_avg_window=None, value_col_units=None):
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = get_lang_text(lang, title_key, title_key)
    x_title_text = get_lang_text(lang, x_axis_title_key, "Date/Time")
    y_title_text = get_lang_text(lang, y_axis_title_key, "Value")

    value_cols_numeric = [col for col in value_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if df.empty or date_col not in df.columns or not value_cols_numeric:
        return go.Figure().update_layout(
            title_text=f"{title_text} ({lang_texts.get('no_data_for_selection', 'No data')})",
            annotations=[dict(text=lang_texts.get('no_data_for_selection', 'No data for current selection'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5, font_size=12)]
        )

    fig = go.Figure()
    colors = config.COLOR_SCHEME_CATEGORICAL

    # Main series lines
    for i, col in enumerate(value_cols_numeric):
        series_color = colors[i % len(colors)]
        series_name_key = f"{col.lower().replace(' ', '_').replace('%','').replace('(','').replace(')','').replace('&','and')}_label"
        series_name = get_lang_text(lang, series_name_key, col.replace('_', ' ').title())

        fig.add_trace(go.Scatter(
            x=df[date_col], y=df[col], mode='lines+markers', name=series_name,
            line=dict(color=series_color, width=2.5), marker=dict(size=5, symbol="circle"),
            hovertemplate=f"<b>{series_name}</b><br>{x_title_text}: %{{x}}<br>{y_title_text}: %{{y:.2f}}{value_col_units.get(col, '') if value_col_units else ''}<extra></extra>"
        ))

    # Rolling average lines
    if rolling_avg_window and isinstance(rolling_avg_window, int) and rolling_avg_window > 1:
        for i, col in enumerate(value_cols_numeric):
            series_color = colors[i % len(colors)]
            series_name_key = f"{col.lower().replace(' ', '_').replace('%','').replace('(','').replace(')','').replace('&','and')}_label"
            base_series_name = get_lang_text(lang, series_name_key, col.replace('_', ' ').title())
            rolling_avg_name = f"{base_series_name} ({rolling_avg_window}-period MA)"
            
            if len(df) >= rolling_avg_window: # Enough data points
                df_copy = df.copy() # Avoid SettingWithCopyWarning
                df_copy[f'{col}_rolling_avg'] = df_copy[col].rolling(window=rolling_avg_window, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=df_copy[date_col], y=df_copy[f'{col}_rolling_avg'], mode='lines',
                    name=rolling_avg_name,
                    line=dict(color=series_color, width=1.5, dash='dash'),
                    opacity=0.8,
                    hovertemplate=f"<b>{rolling_avg_name}</b><br>{x_title_text}: %{{x}}<br>{y_title_text}: %{{y:.2f}}{value_col_units.get(col, '') if value_col_units else ''}<extra></extra>"
                ))

    # Static lines (Average and Target) - plot after data lines
    for i, col in enumerate(value_cols_numeric):
        series_color = colors[i % len(colors)]
        series_name_key = f"{col.lower().replace(' ', '_').replace('%','').replace('(','').replace(')','').replace('&','and')}_label"
        series_name = get_lang_text(lang, series_name_key, col.replace('_', ' ').title())

        if show_average_line:
            avg_val = df[col].mean()
            if pd.notna(avg_val):
                fig.add_hline(y=avg_val, line_dash="dot",
                              annotation_text=f"{get_lang_text(lang, 'average_label', 'Avg')} {series_name}: {avg_val:.1f}",
                              annotation_position="bottom right" if i % 2 == 0 else "bottom left",
                              line_color=series_color, opacity=0.6,
                              annotation=dict(font=dict(size=9, color=series_color), bgcolor="rgba(255,255,255,0.7)", bordercolor=series_color, borderwidth=0.5, yshift=5 if i%2==0 else -5))

        if target_value_map and col in target_value_map and pd.notna(target_value_map[col]):
            target_val = target_value_map[col]
            fig.add_hline(y=target_val, line_dash="dashdot",
                          annotation_text=f"{get_lang_text(lang, 'target_label', 'Target')} {series_name}: {target_val:.1f}",
                          annotation_position="top left" if i % 2 == 0 else "top right",
                          line_color=config.COLOR_TARGET_LINE, line_width=2,
                          annotation=dict(font=dict(size=9, color=config.COLOR_TARGET_LINE), bgcolor="rgba(255,255,255,0.7)", bordercolor=config.COLOR_TARGET_LINE, borderwidth=0.5, yshift=-5 if i%2==0 else 5))

    # Highlight Peaks/Dips (simplified example, focus on global for less clutter)
    if highlight_peaks_dips and len(df) > 2:
         for i, col in enumerate(value_cols_numeric):
            series_color = colors[i % len(colors)]
            if pd.api.types.is_numeric_dtype(df[col]) and not df[col].empty:
                max_val = df[col].max()
                min_val = df[col].min()
                if pd.notna(max_val):
                    max_point_idx = df[col].idxmax()
                    max_point_date = df.loc[max_point_idx, date_col]
                    fig.add_annotation(x=max_point_date, y=max_val, text=f"Max: {max_val:.1f}",
                                       showarrow=True, arrowhead=2, arrowcolor=series_color,
                                       ax=20, ay=-40, bgcolor="rgba(255,255,255,0.8)", font=dict(color=series_color))
                if pd.notna(min_val):
                    min_point_idx = df[col].idxmin()
                    min_point_date = df.loc[min_point_idx, date_col]
                    fig.add_annotation(x=min_point_date, y=min_val, text=f"Min: {min_val:.1f}",
                                       showarrow=True, arrowhead=2, arrowcolor=series_color,
                                       ax=-20, ay=40, bgcolor="rgba(255,255,255,0.8)", font=dict(color=series_color))


    fig.update_layout(
        title=dict(text=title_text, x=0.5), # Centered title
        yaxis_title=y_title_text,
        xaxis_title=x_title_text,
        legend_title_text=get_lang_text(lang, "metrics_legend", "Metrics"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=config.COLOR_GRAY_TEXT),
        xaxis=dict(
            showgrid=False,
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
        margin=dict(l=50, r=30, t=80, b=30) # Adjusted margins
    )
    return fig

# --- Enhanced Comparison Bar Chart ---
def create_comparison_bar_chart(df, x_col, y_cols, title_key, lang,
                                y_axis_title_key="count_axis_label", x_axis_title_key="category_axis_label",
                                barmode='group', show_total_for_stacked=False, text_auto_format_str="{:.0f}"): # Default to integer format for labels
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = get_lang_text(lang, title_key, title_key)
    x_title_text = get_lang_text(lang, x_axis_title_key, "Category")
    y_title_text = get_lang_text(lang, y_axis_title_key, "Count")

    y_cols_list = y_cols if isinstance(y_cols, list) else [y_cols]
    y_cols_list = [col for col in y_cols_list if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    if df.empty or x_col not in df.columns or not y_cols_list:
        return go.Figure().update_layout(
            title_text=f"{title_text} ({lang_texts.get('no_data_for_selection', 'No data')})",
            annotations=[dict(text=lang_texts.get('no_data_for_selection', 'No data for current selection'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5, font_size=12)]
        )

    y_col_labels = {col: get_lang_text(lang, f"{col.lower().replace(' ', '_').replace('(','').replace(')','').replace('%','').replace('&', 'and')}_label", col.replace('_', ' ').title()) for col in y_cols_list}
    
    fig = px.bar(df, x=x_col, y=y_cols_list, title=None, barmode=barmode, # Title set in layout
                 color_discrete_sequence=config.COLOR_SCHEME_CATEGORICAL,
                 labels=y_col_labels
                 )

    fig.update_traces(
        texttemplate=f'%{{y:{text_auto_format_str.split(":")[1] if ":" in text_auto_format_str else ".0f"}}}', # Extract format part or default
        textposition='outside' if barmode != 'stack' else 'inside',
        textfont_size=10,
        insidetextanchor='middle',
        hovertemplate='<b>%{x}</b><br>%{fullData.name}: %{y}<extra></extra>' # Use fullData.name for legend entry
    )

    if barmode == 'stack' and show_total_for_stacked and y_cols_list:
        df_total = df.copy()
        df_total['_total_stacked_'] = df_total[y_cols_list].sum(axis=1)
        
        annotations = []
        for i, row_val in df_total[x_col].items(): # Use items for index if df index is not simple range
            total_val = df_total.loc[i, '_total_stacked_']
            if pd.notna(total_val):
                 annotations.append(dict(x=row_val, y=total_val, 
                                         text=f"{total_val:{text_auto_format_str.split(':')[1] if ':' in text_auto_format_str else '.0f'}}",
                                         font=dict(size=11, color=config.COLOR_GRAY_TEXT),
                                         showarrow=False, yanchor='bottom', yshift=3, xanchor='center'))
        fig.update_layout(annotations=annotations)


    fig.update_layout(
        title=dict(text=title_text, x=0.5),
        yaxis_title=y_title_text,
        xaxis_title=x_title_text,
        legend_title_text=get_lang_text(lang, "metrics_legend", "Metrics"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=config.COLOR_GRAY_TEXT),
        xaxis_tickangle=-45 if len(df[x_col].unique()) > 6 else 0, # Rotate if many categories
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=10)),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)'),
        xaxis=dict(showgrid=False, type='category') # Explicitly category for months/text
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

    if pd.notna(value) and isinstance(value, (int, float)):
        raw_value = float(value) # Ensure float for comparisons

        # Value Formatting
        val_to_format = raw_value
        if unit == "%":
            formatted_value = f"{val_to_format:,.1f}%"
        elif unit == get_lang_text(lang, 'days_label') or abs(val_to_format) >= 1000 and val_to_format % 1 == 0: # days or large integers
            formatted_value = f"{val_to_format:,.0f}{(' ' + unit) if unit else ''}"
        else: # Other numbers with one decimal place
            formatted_value = f"{val_to_format:,.1f}{(' ' + unit) if unit else ''}"

        # Delta Calculation
        if previous_value is not None and pd.notna(previous_value) and isinstance(previous_value, (int,float)):
            prev_raw_value = float(previous_value)
            delta_abs = raw_value - prev_raw_value
            
            sign = "+" if delta_abs >= 1e-9 else ("" if abs(delta_abs) < 1e-9 else "-") # Handle zero delta with no sign
            
            if abs(prev_raw_value) > 1e-9 : # Avoid division by zero for percentage
                 delta_percentage = (delta_abs / abs(prev_raw_value)) * 100
                 delta_text = f"{sign}{abs(delta_abs):,.1f}{unit} ({sign}{abs(delta_percentage):,.0f}%)"
            else:
                 delta_text = f"{sign}{abs(delta_abs):,.1f}{unit} (Prev 0)" # Or another indicator

            # Delta Color
            if higher_is_better is not None:
                if delta_abs > 1e-9 : delta_color = "normal" if higher_is_better else "inverse"
                elif delta_abs < -1e-9 : delta_color = "inverse" if higher_is_better else "normal"
                else: delta_color = "off"
        
        # Icon based on Thresholds
        status = get_status_by_thresholds(raw_value, higher_is_better, threshold_good, threshold_warning)
        if status == "good": icon = "✅ "
        elif status == "warning": icon = "⚠️ "
        elif status == "critical": icon = "❗ "
        elif target_value is not None and higher_is_better is not None: # Fallback to target icon if no detailed status
            if (higher_is_better and raw_value >= target_value) or \
               (not higher_is_better and raw_value <= target_value):
                icon = "👍 " # Generic positive if meeting target
    
    elif pd.isna(value):
         formatted_value = "N/A"
         icon = "❓ "

    st_object.metric(label=icon + label_text_orig, value=formatted_value, delta=delta_text, delta_color=delta_color, help=help_text_content)

# --- Enhanced Radar Chart ---
def create_enhanced_radar_chart(df_radar_input, categories_col, values_col, title_key, lang,
                                 group_col=None, range_max_override=None, target_values_map=None,
                                 fill_opacity=0.5): # Adjusted default opacity
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = get_lang_text(lang, title_key, title_key)
    df_radar = df_radar_input.copy()

    if df_radar.empty or categories_col not in df_radar.columns or values_col not in df_radar.columns or df_radar[categories_col].empty:
        return go.Figure().update_layout(
            title_text=f"{title_text} ({lang_texts.get('no_data_radar', 'No data')})",
            annotations=[dict(text=lang_texts.get('no_data_radar', 'No data for radar chart'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5, font_size=12)]
        )

    all_r_values = df_radar[values_col].dropna().tolist()
    if target_values_map:
        all_r_values.extend([v for v in target_values_map.values() if pd.notna(v)])
    
    valid_r_values = [v for v in all_r_values if isinstance(v, (int, float))]
    current_max_val = max(valid_r_values) if valid_r_values else 0
    
    range_max = range_max_override if range_max_override is not None else (current_max_val * 1.2 if current_max_val > 0 else 5.0)
    range_max = max(range_max, 1.0)

    fig = go.Figure()
    color_sequence = config.COLOR_SCHEME_CATEGORICAL
    all_categories_ordered = df_radar[categories_col].unique()

    # Add actual data traces
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
    else: # Single trace
        df_radar_ordered = pd.DataFrame({categories_col: all_categories_ordered})
        df_radar_ordered = pd.merge(df_radar_ordered, df_radar, on=categories_col, how='left').fillna({values_col: 0})
        fig.add_trace(go.Scatterpolar(
            r=df_radar_ordered[values_col], theta=df_radar_ordered[categories_col],
            fill='toself', name=get_lang_text(lang, "average_score_label", "Average Score"),
            line_color=color_sequence[0], opacity=fill_opacity + 0.1, # Make single trace slightly less transparent
            hovertemplate='<b>%{theta}</b>: %{r:.1f}<extra></extra>'
        ))
    
    # Add target line AFTER data traces
    if target_values_map:
        target_r_values = [target_values_map.get(cat, 0) for cat in all_categories_ordered] # Default to 0 if category not in map
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
                gridcolor="rgba(0,0,0,0.1)", linecolor="rgba(0,0,0,0.1)", tickfont=dict(size=9)
            ),
            angularaxis=dict(
                showline=False, showticklabels=True, gridcolor="rgba(0,0,0,0.1)",
                linecolor="rgba(0,0,0,0.1)", tickfont=dict(size=10), direction="clockwise"
            )
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font=dict(size=9)),
        margin=dict(l=50, r=50, t=100, b=60),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

# --- Enhanced Stress Semáforo (Visual Bar) ---
def create_stress_semaforo_visual(stress_level, lang, scale_max=config.STRESS_LEVEL_MAX_SCALE):
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = get_lang_text(lang, "overall_stress_indicator_title", "Overall Stress Level") # Using a defined key

    val_for_gauge = 0.0
    raw_value = stress_level
    semaforo_color = config.COLOR_GRAY_TEXT
    status_text = get_lang_text(lang, 'stress_na', 'N/A')
    formatted_value = status_text # Initialize for display if N/A

    if pd.notna(raw_value) and isinstance(raw_value, (int, float)):
        val_for_gauge = float(raw_value)
        formatted_value = f"{raw_value:.1f}"
        
        status = get_status_by_thresholds(raw_value, higher_is_worse=True, # Stress is higher is worse
                                           threshold_good=config.STRESS_LEVEL_THRESHOLD_LOW,
                                           threshold_warning=config.STRESS_LEVEL_THRESHOLD_MEDIUM)
        if status == "good": status_text, semaforo_color = get_lang_text(lang, 'stress_low', 'Low'), config.COLOR_GREEN_SEMAFORO
        elif status == "warning": status_text, semaforo_color = get_lang_text(lang, 'stress_medium', 'Moderate'), config.COLOR_YELLOW_SEMAFORO
        elif status == "critical": status_text, semaforo_color = get_lang_text(lang, 'stress_high', 'High'), config.COLOR_RED_SEMAFORO
        else: status_text = f"{get_lang_text(lang, 'value_axis_label')}: {raw_value:.1f}" # Default if no status

    gauge_value_clamped = max(0.0, min(float(scale_max), val_for_gauge)) if pd.notna(val_for_gauge) else 0.0

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_value_clamped,
        domain={'x': [0, 1], 'y': [0.2, 0.8]},
        title={'text': f"<b style='color:{semaforo_color}; font-size:1.2em;'>{status_text}</b>", 'font': {'size': 16}, 'align': "center"},
        number={'valueformat': ".1f", 'font': {'size': 24, 'color': semaforo_color}, 'suffix': f" / {scale_max:.0f}"},
        gauge={
            'shape': "bullet",
            'axis': {'range': [0, scale_max], 'visible': True, 'showticklabels': True,
                     'tickvals': [0, config.STRESS_LEVEL_THRESHOLD_LOW, config.STRESS_LEVEL_THRESHOLD_MEDIUM, scale_max],
                     'ticktext': ["0", f"{config.STRESS_LEVEL_THRESHOLD_LOW:.0f}", f"{config.STRESS_LEVEL_THRESHOLD_MEDIUM:.0f}", f"{scale_max:.0f}"],
                     'tickfont': {'size':9, 'color': config.COLOR_GRAY_TEXT}, 'tickmode': 'array'
            },
            'steps': [
                {'range': [0, config.STRESS_LEVEL_THRESHOLD_LOW], 'color': config.COLOR_GREEN_SEMAFORO, 'name':get_lang_text(lang, 'stress_low', 'Low')},
                {'range': [config.STRESS_LEVEL_THRESHOLD_LOW, config.STRESS_LEVEL_THRESHOLD_MEDIUM], 'color': config.COLOR_YELLOW_SEMAFORO, 'name': get_lang_text(lang, 'stress_medium', 'Moderate')},
                {'range': [config.STRESS_LEVEL_THRESHOLD_MEDIUM, scale_max], 'color': config.COLOR_RED_SEMAFORO, 'name': get_lang_text(lang, 'stress_high', 'High')}
            ],
            'bar': {'color': 'rgba(0,0,0,0.7)', 'thickness': 0.4}, # Neutral dark bar to show actual value position
            'bgcolor': "white", 'borderwidth': 1, 'bordercolor': "lightgray"
        }))
    fig.update_layout(height=120, margin=dict(t=20, b=20, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)')
    return fig
