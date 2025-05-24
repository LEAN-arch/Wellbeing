# visualizations.py
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import config

# --- Helper to get localized text ---
def get_lang_text(lang_code, key, default_text=""):
    return config.TEXT_STRINGS.get(lang_code, config.TEXT_STRINGS["EN"]).get(key, default_text)

# --- Enhanced KPI Gauge ---
def create_kpi_gauge(value, title_key, lang, unit="%", higher_is_worse=True,
                     threshold_good=None, threshold_warning=None,
                     target_line_value=None, max_value_override=None,
                     previous_value=None):
    title_text = get_lang_text(lang, title_key, title_key)
    
    current_value_for_gauge = 0
    delta_ref = None
    delta_mode = "off" # "off", "increasing", "decreasing"

    if pd.notna(value) and isinstance(value, (int, float)):
        current_value_for_gauge = value
        if previous_value is not None and pd.notna(previous_value):
            delta_ref = previous_value
            delta_mode = "number" # Shows absolute difference

    # Max value determination (simplified for clarity, can be made more robust)
    if max_value_override is not None:
        max_val = max_value_override
    else:
        temp_max = [1.0]
        if pd.notna(current_value_for_gauge): temp_max.append(float(current_value_for_gauge) * 1.3)
        if threshold_warning is not None and pd.notna(threshold_warning): temp_max.append(float(threshold_warning) * 1.2)
        elif threshold_good is not None and pd.notna(threshold_good): temp_max.append(float(threshold_good) * (1.5 if higher_is_worse else 1.2) )
        max_val = max(temp_max) if temp_max else 100.0
        if max_val <=0 : max_val = 100.0


    steps = []
    if higher_is_worse:
        good_color, warn_color, crit_color = config.COLOR_GREEN_SEMAFORO, config.COLOR_YELLOW_SEMAFORO, config.COLOR_RED_SEMAFORO
    else:
        good_color, warn_color, crit_color = config.COLOR_RED_SEMAFORO, config.COLOR_YELLOW_SEMAFORO, config.COLOR_GREEN_SEMAFORO # Reversed logic for "lower is worse"

    if threshold_good is not None and pd.notna(threshold_good):
        if higher_is_worse:
            steps.append({'range': [0, threshold_good], 'color': good_color})
            if threshold_warning is not None and pd.notna(threshold_warning) and threshold_warning > threshold_good:
                steps.append({'range': [threshold_good, threshold_warning], 'color': warn_color})
                steps.append({'range': [threshold_warning, max_val], 'color': crit_color})
            else:
                steps.append({'range': [threshold_good, max_val], 'color': crit_color}) # if only good thresh, rest is critical
        else: # lower is worse
            steps.append({'range': [0, threshold_good], 'color': crit_color}) # assumes threshold_good is actually a "bad" limit
            if threshold_warning is not None and pd.notna(threshold_warning) and threshold_warning > threshold_good:
                steps.append({'range': [threshold_good, threshold_warning], 'color': warn_color})
                steps.append({'range': [threshold_warning, max_val], 'color': good_color})
            else:
                steps.append({'range': [threshold_good, max_val], 'color': good_color}) # if only good thresh, rest is good
    else: # No thresholds for steps
        steps.append({'range': [0, max_val], 'color': config.COLOR_NEUTRAL_METRIC})


    fig = go.Figure(go.Indicator(
        mode="gauge+number" + ("+delta" if delta_ref is not None else ""),
        value=current_value_for_gauge,
        title={'text': title_text, 'font': {'size': 16, 'color': config.COLOR_GRAY_TEXT}},
        number={'font': {'size': 28}, 'suffix': unit if unit and unit != "N/A" else "", 'valueformat': ".1f"},
        delta={'reference': delta_ref, 'increasing': {'color': config.COLOR_RED_SEMAFORO if higher_is_worse else config.COLOR_GREEN_SEMAFORO},
               'decreasing': {'color': config.COLOR_GREEN_SEMAFORO if higher_is_worse else config.COLOR_RED_SEMAFORO},
               'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1, 'tickcolor': "darkgray", 'nticks': 5},
            'bar': {'color': 'rgba(0,0,0,0.6)', 'thickness': 0.2}, # Thinner, more subtle bar for value
            'bgcolor': "white",
            'borderwidth': 0, # Cleaner look
            'steps': steps,
            'threshold': {
                'line': {'color': config.COLOR_TARGET_LINE, 'width': 4}, # Prominent target
                'thickness': 0.9,
                'value': target_line_value
            } if target_line_value is not None and pd.notna(target_line_value) else {}
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=5), paper_bgcolor='rgba(0,0,0,0)')
    return fig

# --- Enhanced Trend Chart with Annotations ---
def create_trend_chart(df, date_col, value_cols, title_key, lang,
                       y_axis_title_key="value_axis", x_axis_title_key="date_time_axis",
                       show_average_line=False, target_value_map=None, highlight_peaks_dips=False,
                       rolling_avg_window=None): # e.g., 3 for 3-month rolling average
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = get_lang_text(lang, title_key, title_key)
    x_title_text = get_lang_text(lang, x_axis_title_key, "Date/Time")
    y_title_text = get_lang_text(lang, y_axis_title_key, "Value")

    value_cols = [col for col in value_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])] # Ensure numeric
    if df.empty or not date_col in df.columns or not value_cols:
        return go.Figure().update_layout(title_text=f"{title_text} ({lang_texts.get('no_data_for_selection')})", annotations=[dict(text=lang_texts.get('no_data_for_selection'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)])

    fig = go.Figure()

    for i, col in enumerate(value_cols):
        color = config.COLOR_SCHEME_CATEGORICAL[i % len(config.COLOR_SCHEME_CATEGORICAL)]
        series_name = lang_texts.get(f"{col.lower()}_label", col.replace('_', ' ').title())

        fig.add_trace(go.Scatter(x=df[date_col], y=df[col], mode='lines+markers', name=series_name,
                                 line=dict(color=color, width=2), marker=dict(size=6)))

        if rolling_avg_window and isinstance(rolling_avg_window, int) and rolling_avg_window > 1:
            df[f'{col}_rolling_avg'] = df[col].rolling(window=rolling_avg_window, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=df[date_col], y=df[f'{col}_rolling_avg'], mode='lines',
                                     name=f"{series_name} ({rolling_avg_window}-period MA)",
                                     line=dict(color=color, width=1, dash='dash'),
                                     opacity=0.7))

        if show_average_line:
            avg_val = df[col].mean()
            if pd.notna(avg_val):
                fig.add_hline(y=avg_val, line_dash="dot",
                              annotation_text=f"{lang_texts.get('average_label')} {series_name}: {avg_val:.1f}",
                              annotation_position="bottom right" if i % 2 == 0 else "bottom left",
                              line_color=color, opacity=0.5)

        if target_value_map and col in target_value_map and pd.notna(target_value_map[col]):
            fig.add_hline(y=target_value_map[col], line_dash="dashdot", # Distinct dash for target
                          annotation_text=f"{lang_texts.get('target_label')} {series_name}: {target_value_map[col]}",
                          annotation_position="top left" if i % 2 == 0 else "top right",
                          line_color=config.COLOR_TARGET_LINE, line_width=2)

        if highlight_peaks_dips and len(df) > 2:
            # Simple peak/dip detection (can be more sophisticated)
            max_point = df.loc[df[col].idxmax()]
            min_point = df.loc[df[col].idxmin()]
            fig.add_annotation(x=max_point[date_col], y=max_point[col], text="Peak", showarrow=True, arrowhead=1, ax=0, ay=-30, bordercolor="#c7c7c7", borderwidth=1, bgcolor="#ff7f0e", opacity=0.8)
            fig.add_annotation(x=min_point[date_col], y=min_point[col], text="Dip", showarrow=True, arrowhead=1, ax=0, ay=30, bordercolor="#c7c7c7", borderwidth=1, bgcolor="#1f77b4", opacity=0.8)


    fig.update_layout(
        title_text=title_text,
        yaxis_title=y_title_text,
        xaxis_title=x_title_text,
        legend_title_text=lang_texts.get("metrics_legend", "Metrics"),
        hovermode="x unified",
        xaxis=dict(showgrid=False, rangeslider_visible=True, rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ]))
        ),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_traces(hovertemplate='%{y:.2f}<extra></extra>') # Use default or customize per trace
    return fig


# --- Enhanced Comparison Bar Chart ---
def create_comparison_bar_chart(df, x_col, y_cols, title_key, lang,
                                y_axis_title_key="count_axis", x_axis_title_key="category_axis",
                                barmode='group', text_auto_format='.2s', show_total_for_stacked=False):
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = get_lang_text(lang, title_key, title_key)
    x_title_text = get_lang_text(lang, x_axis_title_key, "Category")
    y_title_text = get_lang_text(lang, y_axis_title_key, "Count")

    y_cols_list = y_cols if isinstance(y_cols, list) else [y_cols]
    y_cols_list = [col for col in y_cols_list if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]


    if df.empty or not x_col in df.columns or not y_cols_list:
        return go.Figure().update_layout(title_text=f"{title_text} ({lang_texts.get('no_data_for_selection')})", annotations=[dict(text=lang_texts.get('no_data_for_selection'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)])

    y_col_labels = {col: lang_texts.get(f"{col.lower().replace(' ', '_').replace('(','').replace(')','').replace('%','').replace('&', 'and')}_label", col.replace('_', ' ').title()) for col in y_cols_list}

    fig = px.bar(df, x=x_col, y=y_cols_list, title=title_text, barmode=barmode,
                 color_discrete_sequence=config.COLOR_SCHEME_CATEGORICAL,
                 labels=y_col_labels,
                 text_auto=True) # This handles individual segment labels

    fig.update_traces(texttemplate=f'%{{y:{text_auto_format}}}', 
                      textposition='outside' if barmode != 'stack' else 'inside',
                      textfont_size=10)

    if barmode == 'stack' and show_total_for_stacked and y_cols_list:
        df['total_stacked'] = df[y_cols_list].sum(axis=1)
        # Adding annotations for total stack - a bit more manual
        annotations = []
        for i, row in df.iterrows():
            annotations.append(dict(x=row[x_col], y=row['total_stacked'], text=f"{row['total_stacked']:{text_auto_format}}",
                                  font=dict(family='Arial', size=11, color='black'),
                                  showarrow=False, yanchor='bottom', yshift=5)) # Shift text above bar
        fig.update_layout(annotations=annotations)

    fig.update_layout(
        yaxis_title=y_title_text,
        xaxis_title=x_title_text,
        legend_title_text=lang_texts.get("metrics_legend", "Metrics"),
        hovermode="x unified",
        xaxis_tickangle=-30 if len(df[x_col].unique()) > 7 else 0,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)'),
        xaxis=dict(showgrid=False, type='category') # Ensure x-axis treats months as categories
    )
    return fig


# --- Enhanced Metric Card ---
def display_metric_card(st_object, label_key, value, lang,
                        previous_value=None, unit="", higher_is_better=None,
                        help_text_key=None, target_value=None,
                        threshold_good=None, threshold_warning=None): # For semantic icon
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    label_text_orig = get_lang_text(lang, label_key, label_key)
    label_text = label_text_orig
    help_text_content = get_lang_text(lang, help_text_key, "") if help_text_key else None

    formatted_value = "N/A"
    delta_text = None
    delta_color = "normal"
    icon = ""

    if pd.notna(value) and isinstance(value, (int, float)):
        raw_value = value # Keep raw value for comparisons
        # Formatting logic
        if abs(value) >= 10000 and value % 1 == 0: # Large whole numbers
            formatted_value = f"{value:,.0f}{unit}"
        elif abs(value) >=100 and value % 1 == 0: # Medium whole numbers
            formatted_value = f"{value:,.0f}{unit}"
        elif isinstance(value, float) or (isinstance(value, int) and (unit == "%" or unit=="")):
             # For percentages, small numbers, or if explicit float, show decimal
            formatted_value = f"{value:,.1f}{unit if unit != '%' else ''}" # Avoid double % if already in unit
            if unit == "%" and not formatted_value.endswith("%"):
                formatted_value += "%"
        else: # Other integers
            formatted_value = f"{int(value)}{unit}"


        if previous_value is not None and pd.notna(previous_value) and isinstance(previous_value, (int,float)):
            delta_abs = value - previous_value
            if previous_value != 0:
                 delta_percentage = ((value - previous_value) / abs(previous_value)) * 100
                 delta_text = f"{delta_abs:+,.1f}{unit} ({delta_percentage:+.0f}%)"
            else: # Avoid division by zero
                 delta_text = f"{delta_abs:+,.1f}{unit} (Prev 0)"

            if higher_is_better is not None:
                if delta_abs > 1e-9 : delta_color = "normal" if higher_is_better else "inverse" # Check for meaningful change
                elif delta_abs < -1e-9 : delta_color = "inverse" if higher_is_better else "normal"
                else: delta_color = "off"
        
        # Icon based on good/warning thresholds
        if higher_is_better is not None:
            if threshold_good is not None:
                if (higher_is_better and raw_value >= threshold_good) or \
                   (not higher_is_better and raw_value <= threshold_good):
                    icon = "✅ " # Met or exceeded good target
            if threshold_warning is not None: # Warning takes precedence if both hit (e.g. value IS warning)
                if (higher_is_better and raw_value < threshold_warning and (threshold_good is None or raw_value < threshold_good)) or \
                   (not higher_is_better and raw_value > threshold_warning and (threshold_good is None or raw_value > threshold_good)):
                    icon = "⚠️ " # In warning zone
        elif target_value is not None and higher_is_better is not None: # Fallback to target if specific thresholds not given
            if (higher_is_better and raw_value >= target_value) or (not higher_is_better and raw_value <= target_value):
                icon = "👍 "
    
    elif value == "N/A" or pd.isna(value) :
         formatted_value = "N/A"
         icon = "❓ "

    st_object.metric(label=icon + label_text, value=formatted_value, delta=delta_text, delta_color=delta_color, help=help_text_content)


# --- Enhanced Radar Chart ---
def create_enhanced_radar_chart(df_radar_input, categories_col, values_col, title_key, lang,
                                 group_col=None, range_max_override=None, target_values_map=None, # {category_name: target}
                                 fill_opacity=0.6):
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = get_lang_text(lang, title_key, title_key)
    df_radar = df_radar_input.copy()

    if df_radar.empty or not all(col in df_radar.columns for col in [categories_col, values_col]):
        return go.Figure().update_layout(title_text=f"{title_text} ({lang_texts.get('no_data_radar')})", annotations=[dict(text=lang_texts.get('no_data_radar'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)])

    # Calculate range_max considering data and targets
    all_r_values = df_radar[values_col].tolist()
    if target_values_map:
        all_r_values.extend(list(target_values_map.values()))
    
    # Filter out NaNs and non-numeric if any, then find max, default to 5
    valid_r_values = [v for v in all_r_values if pd.notna(v) and isinstance(v, (int, float))]
    current_max_val = max(valid_r_values) if valid_r_values else 0
    
    range_max = range_max_override if range_max_override is not None else (current_max_val * 1.15 if current_max_val > 0 else 5)
    range_max = max(range_max, 1.0) # ensure it's at least 1


    fig = go.Figure()
    # Ensure categories are unique and consistently ordered
    all_categories_ordered = df_radar[categories_col].unique()
    num_categories = len(all_categories_ordered)

    color_sequence = config.COLOR_SCHEME_CATEGORICAL

    if group_col and group_col in df_radar.columns and df_radar[group_col].nunique() > 0:
        for i, (group_name, group_data) in enumerate(df_radar.groupby(group_col)):
            group_df_ordered = pd.DataFrame({categories_col: all_categories_ordered})
            group_df_ordered = pd.merge(group_df_ordered, group_data, on=categories_col, how='left').fillna(0)

            fig.add_trace(go.Scatterpolar(
                r=group_df_ordered[values_col], theta=group_df_ordered[categories_col],
                fill='toself', name=str(group_name),
                line_color=color_sequence[i % len(color_sequence)], opacity=fill_opacity,
                hovertemplate='<b>%{theta}</b><br>' + f'{str(group_name)}: %{{r:.1f}}' + '<extra></extra>'
            ))
    else:
        df_radar_ordered = pd.DataFrame({categories_col: all_categories_ordered})
        df_radar_ordered = pd.merge(df_radar_ordered, df_radar, on=categories_col, how='left').fillna(0)
        fig.add_trace(go.Scatterpolar(
            r=df_radar_ordered[values_col], theta=df_radar_ordered[categories_col],
            fill='toself', name=get_lang_text(lang, "average_score_label", "Average Score"),
            line_color=color_sequence[0], opacity=fill_opacity,
            hovertemplate='<b>%{theta}</b>: %{r:.1f}<extra></extra>'
        ))
    
    if target_values_map:
        target_r_values = [target_values_map.get(cat, 0) for cat in all_categories_ordered]
        fig.add_trace(go.Scatterpolar(
            r=target_r_values, theta=all_categories_ordered, mode='lines',
            name=get_lang_text(lang, "target_label", "Target"),
            line=dict(color=config.COLOR_TARGET_LINE, dash='dash', width=2.5),
            hoverinfo='skip'
        ))

    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=18)),
        polar=dict(
            bgcolor="rgba(255,255,255,0.0)", # Transparent background
            radialaxis=dict(
                visible=True, range=[0, range_max], showline=False, showticklabels=True,
                gridcolor="rgba(0,0,0,0.1)", linecolor="rgba(0,0,0,0.1)", tickfont=dict(size=10)
            ),
            angularaxis=dict(
                showline=False, showticklabels=True, gridcolor="rgba(0,0,0,0.1)",
                linecolor="rgba(0,0,0,0.1)", tickfont=dict(size=11), direction="clockwise",
                # Rotate category labels for better readability if many categories
                tickangle = 0 if num_categories <= 6 else (360 / num_categories) / 2 # Experimental
            )
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(l=60, r=60, t=100, b=60), # Increased margins for labels
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


# --- Enhanced Stress Semáforo (Visual Bar) ---
def create_stress_semaforo_visual(stress_level, lang):
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = get_lang_text(lang, "overall_stress_indicator_title", "Overall Stress Level")

    val_for_gauge = 0.0
    semaforo_color = config.COLOR_GRAY_TEXT
    status_text = get_lang_text(lang, 'stress_na', 'N/A')

    if pd.notna(stress_level) and isinstance(stress_level, (int, float)):
        val_for_gauge = float(stress_level)
        if val_for_gauge <= config.STRESS_LEVEL_THRESHOLD_LOW:
            status_text = get_lang_text(lang, 'stress_low', 'Low')
            semaforo_color = config.COLOR_GREEN_SEMAFORO
        elif val_for_gauge <= config.STRESS_LEVEL_THRESHOLD_MEDIUM:
            status_text = get_lang_text(lang, 'stress_medium', 'Moderate')
            semaforo_color = config.COLOR_YELLOW_SEMAFORO
        else:
            status_text = get_lang_text(lang, 'stress_high', 'High')
            semaforo_color = config.COLOR_RED_SEMAFORO
    else:
        val_for_gauge = 0.0

    max_scale = config.STRESS_LEVEL_MAX_SCALE

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val_for_gauge,
        domain={'x': [0, 1], 'y': [0.2, 0.8]}, # Give some vertical padding
        title={'text': f"<b>{status_text}</b>", 'font': {'size': 20, 'color': semaforo_color}, 'align': "center"}, # Status more prominent
        number={'valueformat': ".1f", 'font': {'size': 28, 'color': semaforo_color}, 'suffix': f" / {max_scale:.0f}"},
        gauge={
            'shape': "bullet",
            'axis': {'range': [0, max_scale], 'visible': True, 'showticklabels': True,
                     'tickvals': [0, config.STRESS_LEVEL_THRESHOLD_LOW, config.STRESS_LEVEL_THRESHOLD_MEDIUM, max_scale],
                     'ticktext': ["0", f"{config.STRESS_LEVEL_THRESHOLD_LOW:.0f}", f"{config.STRESS_LEVEL_THRESHOLD_MEDIUM:.0f}", f"{max_scale:.0f}"],
                     'tickfont': {'size':10, 'color': config.COLOR_GRAY_TEXT}},
            'steps': [
                {'range': [0, config.STRESS_LEVEL_THRESHOLD_LOW], 'color': config.COLOR_GREEN_SEMAFORO},
                {'range': [config.STRESS_LEVEL_THRESHOLD_LOW, config.STRESS_LEVEL_THRESHOLD_MEDIUM], 'color': config.COLOR_YELLOW_SEMAFORO},
                {'range': [config.STRESS_LEVEL_THRESHOLD_MEDIUM, max_scale], 'color': config.COLOR_RED_SEMAFORO}
            ],
            'bar': {'color': 'rgba(0,0,0,0.7)', 'thickness': 0.3} # Value indicator bar
        }))
    fig.update_layout(height=150, margin=dict(t=20, b=30, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)')
    # Add a subtitle (main title is within indicator for this visual)
    # You could use st.subheader in app.py for the overall title.
    return fig
