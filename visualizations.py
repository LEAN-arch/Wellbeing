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

    val_float = float(value)
    good_thresh = float(threshold_good) if threshold_good is not None and pd.notna(threshold_good) else None
    warn_thresh = float(threshold_warning) if threshold_warning is not None and pd.notna(threshold_warning) else None

    if higher_is_worse: # Lower value is better (e.g., rotation, incidents, stress)
        if good_thresh is not None and val_float <= good_thresh: return "good"
        if warn_thresh is not None and val_float <= warn_thresh: # Assumes warn_thresh > good_thresh if both are defined
            if good_thresh is None or val_float > good_thresh : # Value is between good and warning, or just under warning if good not set
                return "warning"
        # If value is greater than warning threshold, or greater than good threshold (if warning is not defined)
        if (warn_thresh is not None and val_float > warn_thresh) or \
           (warn_thresh is None and good_thresh is not None and val_float > good_thresh):
            return "critical"
    else: # Higher value is better (e.g., retention, eNPS)
        if good_thresh is not None and val_float >= good_thresh: return "good"
        if warn_thresh is not None and val_float >= warn_thresh: # Assumes warn_thresh < good_thresh if both defined
             if good_thresh is None or val_float < good_thresh: # Value is between warning and good
                return "warning"
        # If value is less than warning threshold, or less than good threshold (if warning is not defined)
        if (warn_thresh is not None and val_float < warn_thresh) or \
           (warn_thresh is None and good_thresh is not None and val_float < good_thresh):
            return "critical"
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
    
    if pd.notna(value) and isinstance(value, (int, float, np.number)):
        current_value_for_gauge = float(value)
        if previous_value is not None and pd.notna(previous_value) and isinstance(previous_value, (int,float, np.number)):
            delta_ref = float(previous_value)

    if max_value_override is not None:
        max_val = float(max_value_override)
    else:
        max_val_candidates = [1.0]
        if pd.notna(current_value_for_gauge): max_val_candidates.append(current_value_for_gauge * 1.25)
        
        valid_thresh_for_max = [t for t in [threshold_good, threshold_warning, target_line_value] if t is not None and pd.notna(t)]
        if valid_thresh_for_max:
             max_val_candidates.append(max(float(t) for t in valid_thresh_for_max) * 1.2)
        else:
            if not pd.notna(current_value_for_gauge) or current_value_for_gauge == 0:
                max_val_candidates.append(100.0)

        max_val = max(max_val_candidates) if max_val_candidates else 100.0
        if max_val <=0 : max_val = 100.0
        if pd.notna(current_value_for_gauge) and current_value_for_gauge > max_val and max_value_override is None:
            max_val = current_value_for_gauge * 1.1

    steps = []
    num_t_good = float(threshold_good) if threshold_good is not None and pd.notna(threshold_good) else None
    num_t_warn = float(threshold_warning) if threshold_warning is not None and pd.notna(threshold_warning) else None

    current_range_start = 0.0
    if higher_is_worse: # Green -> Yellow -> Red
        if num_t_good is not None:
            steps.append({'range': [current_range_start, num_t_good], 'color': config.COLOR_GREEN_SEMAFORO})
            current_range_start = num_t_good
        if num_t_warn is not None and num_t_warn > current_range_start: # Ensure warning is above good
            steps.append({'range': [current_range_start, num_t_warn], 'color': config.COLOR_YELLOW_SEMAFORO})
            current_range_start = num_t_warn
        steps.append({'range': [current_range_start, max_val], 'color': config.COLOR_RED_SEMAFORO})
    else: # Lower is worse: Red -> Yellow -> Green
        if num_t_warn is not None: # Here, warning is the lower undesirable bound
            steps.append({'range': [current_range_start, num_t_warn], 'color': config.COLOR_RED_SEMAFORO})
            current_range_start = num_t_warn
        if num_t_good is not None and num_t_good > current_range_start: # Good is above warning
            steps.append({'range': [current_range_start, num_t_good], 'color': config.COLOR_YELLOW_SEMAFORO})
            current_range_start = num_t_good
        steps.append({'range': [current_range_start, max_val], 'color': config.COLOR_GREEN_SEMAFORO})

    if not steps: steps.append({'range': [0, max_val], 'color': config.COLOR_NEUTRAL_METRIC}) # Fallback
    
    num_target_line = float(target_line_value) if target_line_value is not None and pd.notna(target_line_value) else None

    fig = go.Figure(go.Indicator(
        mode="gauge+number" + ("+delta" if delta_ref is not None else ""),
        value=current_value_for_gauge if pd.notna(current_value_for_gauge) else 0, # Display 0 for NaN on gauge value
        title={'text': title_text, 'font': {'size': 16, 'color': config.COLOR_GRAY_TEXT}},
        number={'font': {'size': 28}, 'suffix': unit if unit and unit != "N/A" else "", 'valueformat': ".1f"},
        delta={'reference': delta_ref,
               'increasing': {'color': config.COLOR_RED_SEMAFORO if higher_is_worse else config.COLOR_GREEN_SEMAFORO},
               'decreasing': {'color': config.COLOR_GREEN_SEMAFORO if higher_is_worse else config.COLOR_RED_SEMAFORO},
               'font': {'size': 16}, 'valueformat': ".1f"},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1, 'tickcolor': "darkgray", 'nticks': 5},
            'bar': {'color': 'rgba(0,0,0,0.5)', 'thickness': 0.15}, # Value indicator
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
def create_trend_chart(df_input, date_col, value_cols_map, title_key, lang, # value_cols_map: {'Series Name Key': 'actual_col_name'}
                       y_axis_title_key="value_axis_label", x_axis_title_key="date_time_axis_label",
                       show_average_line=False, target_value_map=None, # target_value_map: {'Actual Col Name': target_val}
                       rolling_avg_window=None, value_col_units_map=None): # value_col_units_map: {'Actual Col Name': unit_str}
    df = df_input.copy() # Work on a copy
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = get_lang_text(lang, title_key, title_key)
    x_title_text = get_lang_text(lang, x_axis_title_key, "Date/Time")
    y_title_text = get_lang_text(lang, y_axis_title_key, "Value")

    if df.empty or date_col not in df.columns or not value_cols_map:
        return go.Figure().update_layout(
            title_text=f"{title_text} ({lang_texts.get('no_data_for_selection', 'No data')})",
            annotations=[dict(text=lang_texts.get('no_data_for_selection', 'No data for current selection'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5, font_size=12)]
        )

    fig = go.Figure()
    colors = config.COLOR_SCHEME_CATEGORICAL
    processed_value_cols_actual_names = [] # Keep track of columns successfully processed for later use

    # Main series lines
    for i, (series_label_key, actual_col_name) in enumerate(value_cols_map.items()):
        if actual_col_name not in df.columns or not pd.api.types.is_numeric_dtype(df[actual_col_name]):
            # st.warning(f"Column {actual_col_name} for series {series_label_key} not found or not numeric.")
            continue 
        processed_value_cols_actual_names.append(actual_col_name)

        series_color = colors[i % len(colors)]
        series_name = get_lang_text(lang, series_label_key, actual_col_name.replace('_', ' ').title())
        unit_suffix = value_col_units_map.get(actual_col_name, "") if value_col_units_map else ""

        fig.add_trace(go.Scatter(
            x=df[date_col], y=df[actual_col_name], mode='lines+markers', name=series_name,
            line=dict(color=series_color, width=2), marker=dict(size=5, symbol="circle"),
            hovertemplate=f"<b>{series_name}</b><br>{x_title_text}: %{{x|%Y-%m-%d}}<br>{y_title_text}: %{{y:.2f}}{unit_suffix}<extra></extra>"
        ))

    # Rolling average lines
    if rolling_avg_window and isinstance(rolling_avg_window, int) and rolling_avg_window > 1:
        for i, actual_col_name in enumerate(processed_value_cols_actual_names): # Iterate using actual names
            series_color = colors[i % len(colors)] # Match color with original series
            
            # Find the original label key for this actual_col_name to get the base series name
            series_label_key_for_rolling = None
            for k, v in value_cols_map.items():
                if v == actual_col_name:
                    series_label_key_for_rolling = k
                    break
            
            base_series_name = get_lang_text(lang, series_label_key_for_rolling, actual_col_name.replace('_', ' ').title()) if series_label_key_for_rolling else actual_col_name
            rolling_avg_name = f"{base_series_name} ({rolling_avg_window}-p MA)"
            unit_suffix = value_col_units_map.get(actual_col_name, "") if value_col_units_map else ""

            if len(df) >= rolling_avg_window:
                df[f'{actual_col_name}_rolling_avg'] = df[actual_col_name].rolling(window=rolling_avg_window, min_periods=1, center=True).mean() # Centered MA
                fig.add_trace(go.Scatter(
                    x=df[date_col], y=df[f'{actual_col_name}_rolling_avg'], mode='lines',
                    name=rolling_avg_name,
                    line=dict(color=series_color, width=1.5, dash='dot'),
                    opacity=0.7,
                    hovertemplate=f"<b>{rolling_avg_name}</b><br>{x_title_text}: %{{x|%Y-%m-%d}}<br>{y_title_text}: %{{y:.2f}}{unit_suffix}<extra></extra>"
                ))

    # Static lines (Average and Target)
    for i, actual_col_name in enumerate(processed_value_cols_actual_names):
        series_color = colors[i % len(colors)]
        series_label_key_for_static = None
        for k,v in value_cols_map.items():
            if v == actual_col_name:
                series_label_key_for_static = k
                break
        series_name = get_lang_text(lang, series_label_key_for_static, actual_col_name.replace('_', ' ').title()) if series_label_key_for_static else actual_col_name

        if show_average_line:
            avg_val = df[actual_col_name].mean()
            if pd.notna(avg_val):
                fig.add_hline(y=avg_val, line_dash="longdash",
                              annotation_text=f"{get_lang_text(lang, 'average_label', 'Avg')} {series_name}: {avg_val:.1f}",
                              annotation_position="bottom right" if i % 2 == 0 else "top left", # Alternate positions
                              line_color=series_color, opacity=0.5,
                              annotation_font_size=9, annotation_font_color=series_color,
                              annotation_bgcolor="rgba(255,255,255,0.75)")

        if target_value_map and actual_col_name in target_value_map and pd.notna(target_value_map[actual_col_name]):
            target_val = target_value_map[actual_col_name]
            fig.add_hline(y=target_val, line_dash="dash",
                          annotation_text=f"{get_lang_text(lang, 'target_label', 'Target')} {series_name}: {target_val:.1f}", # Format target value
                          annotation_position="top left" if i % 2 == 0 else "bottom right",
                          line_color=config.COLOR_TARGET_LINE, line_width=1.5,
                          annotation_font_size=9, annotation_font_color=config.COLOR_TARGET_LINE,
                          annotation_bgcolor="rgba(255,255,255,0.75)")


    fig.update_layout(
        title=dict(text=title_text, x=0.5),
        yaxis_title=y_title_text,
        xaxis_title=x_title_text,
        legend_title_text=get_lang_text(lang, "metrics_legend", "Metrics"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=config.COLOR_GRAY_TEXT, namelength=-1), # show full legend item names
        xaxis=dict(
            showgrid=False, type='date',
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
        margin=dict(l=60, r=30, t=90, b=30) # Ensure enough margin for titles and axis labels
    )
    return fig

# --- Enhanced Comparison Bar Chart ---
def create_comparison_bar_chart(df_input, x_col, y_cols_map, title_key, lang, # y_cols_map: {'Series Name Key': 'actual_col_name'}
                                y_axis_title_key="count_axis_label", x_axis_title_key="category_axis_label",
                                barmode='group', show_total_for_stacked=False, data_label_format_str=".0f"):
    df = df_input.copy()
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = get_lang_text(lang, title_key, title_key)
    x_title_text = get_lang_text(lang, x_axis_title_key, "Category")
    y_title_text = get_lang_text(lang, y_axis_title_key, "Count")

    # Prepare DataFrame for Plotly Express by renaming columns to their desired display names
    df_for_plot = df.copy()
    y_display_names = []
    rename_cols_for_plot = {}

    for series_key, actual_col_name in y_cols_map.items():
        if actual_col_name in df.columns and pd.api.types.is_numeric_dtype(df[actual_col_name]):
            display_name = get_lang_text(lang, series_key, actual_col_name.replace('_', ' ').title())
            rename_cols_for_plot[actual_col_name] = display_name
            y_display_names.append(display_name)
        # else: st.warning(f"Bar chart: Column '{actual_col_name}' for series key '{series_key}' not found or not numeric in DataFrame.") # Optional warning

    if not y_display_names or x_col not in df.columns: # Check if we have any y-columns to plot or x_col is missing
         return go.Figure().update_layout(
            title_text=f"{title_text} ({lang_texts.get('no_data_for_selection', 'No data')})",
            annotations=[dict(text=lang_texts.get('no_data_for_selection', 'No data for current selection'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5, font_size=12)]
        )

    df_for_plot.rename(columns=rename_cols_for_plot, inplace=True)
    
    fig = px.bar(df_for_plot, x=x_col, y=y_display_names, title=None, barmode=barmode,
                 color_discrete_sequence=config.COLOR_SCHEME_CATEGORICAL,
                 # labels argument in px.bar maps existing column names to new display names
                 # but we already renamed, so direct display names work
                 )

    fig.update_traces(
        texttemplate=f'%{{y:{data_label_format_str}}}',
        textposition='outside' if barmode != 'stack' else 'inside',
        textfont_size=9,
        insidetextanchor='middle' if barmode == 'stack' else 'auto',
        hovertemplate='<b>%{x}</b><br>%{fullData.name}: %{y}<extra></extra>' # Uses legend name (which is display_name)
    )

    if barmode == 'stack' and show_total_for_stacked and y_display_names:
        df_for_plot['_total_stacked_'] = df_for_plot[y_display_names].sum(axis=1)
        annotations = []
        for _, row in df_for_plot.iterrows(): # Iterate over df_for_plot
            x_val = row[x_col]
            total_val = row['_total_stacked_']
            if pd.notna(total_val):
                 annotations.append(dict(
                    x=x_val, y=total_val,
                    text=f"{total_val:{data_label_format_str}}", # Apply format
                    font=dict(size=10, color=config.COLOR_GRAY_TEXT),
                    showarrow=False, yanchor='bottom', yshift=3, xanchor='center'
                ))
        fig.update_layout(annotations=annotations)

    fig.update_layout(
        title=dict(text=title_text, x=0.5, font_size=18),
        yaxis_title=y_title_text,
        xaxis_title=x_title_text,
        legend_title_text=get_lang_text(lang, "metrics_legend", "Metrics"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=12, namelength=-1),
        xaxis_tickangle=-30 if len(df_for_plot[x_col].unique()) > 6 else 0,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=9)),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)'),
        xaxis=dict(showgrid=False, type='category')
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

    if pd.notna(value) and isinstance(value, (int, float, np.number)):
        raw_value = float(value)
        val_to_format = raw_value

        # Value Formatting
        if unit == "%": formatted_value = f"{val_to_format:,.1f}%"
        elif unit == get_lang_text(lang, 'days_label') or (abs(val_to_format) >= 1000 and val_to_format % 1 == 0):
            formatted_value = f"{val_to_format:,.0f}{(' ' + unit) if unit and unit != '%' else ''}"
        elif abs(val_to_format) < 1 and abs(val_to_format) > 0: # Small decimals
            formatted_value = f"{val_to_format:,.2f}{(' ' + unit) if unit and unit != '%' else ''}"
        else: # Other numbers
            formatted_value = f"{val_to_format:,.1f}{(' ' + unit) if unit and unit != '%' else ''}"
        
        # Ensure '%' is appended if unit is '%' but not in formatted_value
        if unit == "%" and not formatted_value.endswith("%"): formatted_value += "%"

        # Delta Calculation
        if previous_value is not None and pd.notna(previous_value) and isinstance(previous_value, (int,float, np.number)):
            prev_raw_value = float(previous_value)
            delta_abs_val = raw_value - prev_raw_value # Absolute change in value
            sign = "+" if delta_abs_val >= 1e-9 else ("" if abs(delta_abs_val) < 1e-9 else "-")
            
            # Use the same formatting logic for delta's absolute part as for the main value
            delta_display_unit = unit if unit != '%' else '' # Unit for the absolute part of delta
            if unit == "%": formatted_delta_abs = f"{abs(delta_abs_val):.1f}%"
            elif unit == get_lang_text(lang, 'days_label') or (abs(delta_abs_val) >=1000 and delta_abs_val %1 ==0):
                formatted_delta_abs = f"{abs(delta_abs_val):.0f}{(' ' + delta_display_unit) if delta_display_unit else ''}"
            else:
                formatted_delta_abs = f"{abs(delta_abs_val):.1f}{(' ' + delta_display_unit) if delta_display_unit else ''}"


            if abs(prev_raw_value) > 1e-9 :
                 delta_percentage_val = (delta_abs_val / abs(prev_raw_value)) * 100
                 delta_text = f"{sign}{formatted_delta_abs} ({sign}{abs(delta_percentage_val):,.0f}%)"
            else: # Avoid division by zero
                 delta_text = f"{sign}{formatted_delta_abs} ({get_lang_text(lang,'prev_period_label_short','Prev 0')})"

            if higher_is_better is not None:
                if delta_abs_val > 1e-9 : delta_color = "normal" if higher_is_better else "inverse"
                elif delta_abs_val < -1e-9 : delta_color = "inverse" if higher_is_better else "normal"
                else: delta_color = "off"
        
        status = get_status_by_thresholds(raw_value, higher_is_better, threshold_good, threshold_warning)
        if status == "good": icon = "✅ "
        elif status == "warning": icon = "⚠️ "
        elif status == "critical": icon = "❗ "
        elif target_value is not None and higher_is_better is not None and pd.notna(target_value): # Check if target is met
            if (higher_is_better and raw_value >= float(target_value)) or \
               (not higher_is_better and raw_value <= float(target_value)):
                icon = "🎯 "
    
    elif pd.isna(value) or value is None: # Ensure explicit None also results in N/A
         formatted_value = "N/A"
         icon = "❓ "

    st_object.metric(label=icon + label_text_orig, value=formatted_value, delta=delta_text, delta_color=delta_color, help=help_text_content)

# --- Enhanced Radar Chart ---
def create_enhanced_radar_chart(df_radar_input, categories_col, values_col, title_key, lang,
                                 group_col=None, range_max_override=None, target_values_map=None,
                                 fill_opacity=0.4):
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = get_lang_text(lang, title_key, title_key)
    df_radar = df_radar_input.copy()

    if df_radar.empty or categories_col not in df_radar.columns or values_col not in df_radar.columns or df_radar[categories_col].nunique() == 0:
        return go.Figure().update_layout(
            title_text=f"{title_text} ({lang_texts.get('no_data_radar', 'No data')})",
            annotations=[dict(text=lang_texts.get('no_data_radar', 'No data for radar chart'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5, font_size=12)]
        )

    # Ensure categories are unique and consistently ordered for all traces
    all_categories_ordered = df_radar[categories_col].unique()


    # Calculate range_max considering data and targets (localized target keys need to be resolved first)
    all_r_values_for_range = df_radar[values_col].dropna().tolist()
    if target_values_map: # target_values_map uses localized display names as keys
        all_r_values_for_range.extend([v for v in target_values_map.values() if pd.notna(v) and isinstance(v, (int,float))])
    
    valid_r_values_for_range = [v for v in all_r_values_for_range if isinstance(v, (int, float))]
    current_max_data_val = max(valid_r_values_for_range) if valid_r_values_for_range else 0
    
    range_max = range_max_override if range_max_override is not None else (current_max_data_val * 1.15 if current_max_data_val > 0 else 5.0)
    range_max = max(range_max, 1.0) # Minimum range for visibility


    fig = go.Figure()
    color_sequence = config.COLOR_SCHEME_CATEGORICAL

    # Add actual data traces
    if group_col and group_col in df_radar.columns and df_radar[group_col].nunique() > 0:
        for i, (group_name, group_data) in enumerate(df_radar.groupby(group_col)):
            # Create DataFrame for this group, ensuring all categories are present and in order
            group_df_ordered = pd.DataFrame({categories_col: all_categories_ordered})
            group_df_ordered = pd.merge(group_df_ordered, group_data, on=categories_col, how='left').fillna({values_col: 0}) # Fill missing with 0

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
            line_color=color_sequence[0], opacity=fill_opacity + 0.1, # Make single trace slightly more opaque
            hovertemplate='<b>%{theta}</b>: %{r:.1f}<extra></extra>'
        ))
    
    # Add target line AFTER data traces
    if target_values_map:
        # `all_categories_ordered` are the display names. `target_values_map` should have keys matching these.
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
            bgcolor="rgba(255,255,255,0.05)",
            radialaxis=dict(
                visible=True, range=[0, range_max], showline=False, showticklabels=True,
                gridcolor="rgba(0,0,0,0.15)", linecolor="rgba(0,0,0,0.15)", tickfont=dict(size=9)
            ),
            angularaxis=dict( # These are the category labels
                showline=False, showticklabels=True, gridcolor="rgba(0,0,0,0.15)",
                linecolor="rgba(0,0,0,0.15)", tickfont=dict(size=10), direction="clockwise"
            )
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font=dict(size=9)),
        margin=dict(l=50, r=50, t=100, b=60), # Give space for title and legend
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


# --- Enhanced Stress Semáforo (Visual Bar) - Corrected ---
def create_stress_semaforo_visual(stress_level, lang, scale_max=config.STRESS_LEVEL_MAX_SCALE):
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    # The title is integrated into the indicator for better alignment with the visual.
    # If a separate Streamlit subheader is used, it will be redundant or should be styled differently.

    val_for_gauge_bar = 0.0  # Position of the bar on the gauge
    display_number = None    # Number to display; None will typically show as "N/A" or similar in Plotly
    semaforo_color = config.COLOR_GRAY_TEXT
    status_text = get_lang_text(lang, 'stress_na', 'N/A')

    if pd.notna(stress_level) and isinstance(stress_level, (int, float, np.number)):
        val_for_gauge_bar = float(stress_level)
        display_number = float(stress_level)
        
        status = get_status_by_thresholds(val_for_gauge_bar, higher_is_worse=True,
                                           threshold_good=config.STRESS_LEVEL_THRESHOLD_LOW,
                                           threshold_warning=config.STRESS_LEVEL_THRESHOLD_MEDIUM)
        if status == "good": status_text, semaforo_color = get_lang_text(lang, 'low_label'), config.COLOR_GREEN_SEMAFORO
        elif status == "warning": status_text, semaforo_color = get_lang_text(lang, 'moderate_label'), config.COLOR_YELLOW_SEMAFORO
        elif status == "critical": status_text, semaforo_color = get_lang_text(lang, 'high_label'), config.COLOR_RED_SEMAFORO
        else: status_text = f"{get_lang_text(lang, 'value_axis_label')}: {val_for_gauge_bar:.1f}"

    # Clamp the bar value to the gauge axis to prevent overflow if value is outside [0, scale_max]
    gauge_bar_clamped = max(0.0, min(float(scale_max), val_for_gauge_bar)) if pd.notna(val_for_gauge_bar) else 0.0

    number_config = {
        'font': {'size': 22, 'color': semaforo_color},
        'valueformat': ".1f" # Apply formatting even if value is None (Plotly handles None)
    }
    # Add suffix only if display_number is valid
    if display_number is not None:
        number_config['suffix'] = f" / {scale_max:.0f}"


    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=display_number, # The numeric value to display (can be None)
        domain={'x': [0, 1], 'y': [0.2, 0.8]},
        title={'text': f"<b style='color:{semaforo_color}; font-size:1.1em;'>{status_text}</b>", 'font': {'size': 16}, 'align': "center"},
        number=number_config,
        gauge={
            'shape': "bullet",
            'axis': {'range': [0, scale_max], 'visible': True, 'showticklabels': True,
                     'tickvals': [0, config.STRESS_LEVEL_THRESHOLD_LOW, config.STRESS_LEVEL_THRESHOLD_MEDIUM, scale_max],
                     'ticktext': ["0", f"{config.STRESS_LEVEL_THRESHOLD_LOW:.0f}", f"{config.STRESS_LEVEL_THRESHOLD_MEDIUM:.0f}", f"{scale_max:.0f}"],
                     'tickfont': {'size':9, 'color': config.COLOR_GRAY_TEXT}, 'tickmode': 'array'
            },
            'steps': [ # Background colors for the ranges
                {'range': [0, config.STRESS_LEVEL_THRESHOLD_LOW], 'color': config.COLOR_GREEN_SEMAFORO, 'name':get_lang_text(lang, 'low_label')},
                {'range': [config.STRESS_LEVEL_THRESHOLD_LOW, config.STRESS_LEVEL_THRESHOLD_MEDIUM], 'color': config.COLOR_YELLOW_SEMAFORO, 'name': get_lang_text(lang, 'moderate_label')},
                {'range': [config.STRESS_LEVEL_THRESHOLD_MEDIUM, scale_max], 'color': config.COLOR_RED_SEMAFORO, 'name': get_lang_text(lang, 'high_label')}
            ],
            # The bar whose 'value' attribute is set by the Indicator's main 'value', but this positions it
            # Here we use 'gauge_bar_clamped' which is essentially the indicator 'value' if not None, clamped.
            # Let the main 'value' of the Indicator determine the bar position, so this is implicitly handled.
            'bar': {'color': semaforo_color if pd.notna(display_number) else 'rgba(128,128,128,0.5)', 'thickness': 0.6}, # Bar is colored by status. Grey if N/A.
            'bgcolor': "rgba(255,255,255,0.8)", 'borderwidth': 1, 'bordercolor': "lightgray"
        }))
    fig.update_layout(height=110, margin=dict(t=15, b=15, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)')
    return fig
