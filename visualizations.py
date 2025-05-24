# visualizations.py
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import config

def create_kpi_gauge(value, title_key, lang, unit="%", higher_is_worse=True,
                     low_threshold=None, medium_threshold=None, high_threshold=None, target_threshold=None, max_value_override=None):
    """
    Creates an enhanced KPI gauge.
    - target_threshold: The specific value to show as a line on the gauge.
    - max_value_override: Manually set the max of the gauge axis.
    """
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = lang_texts.get(title_key, title_key)

    # Use provided thresholds or fall back to generic ones if defined broadly
    lt = low_threshold if low_threshold is not None else (config.ROTATION_RATE_LOW if "rotation" in title_key.lower() else None)
    mt = medium_threshold if medium_threshold is not None else (config.ROTATION_RATE_MEDIUM if "rotation" in title_key.lower() else None)
    ht = high_threshold if high_threshold is not None else (config.ROTATION_RATE_HIGH if "rotation" in title_key.lower() else None)


    current_value_for_gauge = 0
    display_value_text = "N/A"
    if pd.notna(value) and isinstance(value, (int, float)):
        current_value_for_gauge = value
        display_value_text = f"{value:.1f}{unit if unit else ''}"
    
    # Determine dynamic max_value if not provided
    if max_value_override is not None:
        max_val = max_value_override
    else:
        max_val_candidates = [1] # Ensure max_val is at least 1
        if pd.notna(current_value_for_gauge): max_val_candidates.append(current_value_for_gauge * 1.25)
        if ht is not None: max_val_candidates.append(ht * 1.2)
        elif mt is not None: max_val_candidates.append(mt * 1.5)
        elif lt is not None: max_val_candidates.append(lt * 2)
        else: max_val_candidates.append(100) # Default max if no thresholds
        max_val = max(max_val_candidates)
        if current_value_for_gauge > max_val: # If current value exceeds calculated max
            max_val = current_value_for_gauge * 1.1

    steps = []
    # Define steps based on higher_is_worse and provided thresholds
    # Ensure thresholds are numeric and ordered correctly for step definitions
    valid_thresholds = sorted([t for t in [lt, mt, ht] if t is not None and pd.notna(t)])

    if higher_is_worse: # e.g., Rotation Rate - Green, Yellow, Red
        if valid_thresholds:
            last_thresh = 0
            if len(valid_thresholds) >= 1: # At least a 'low' threshold defines 'good'
                steps.append({'range': [last_thresh, valid_thresholds[0]], 'color': config.COLOR_GREEN_SEMAFORO})
                last_thresh = valid_thresholds[0]
            if len(valid_thresholds) >= 2: # A 'medium' threshold defines 'warning'
                steps.append({'range': [last_thresh, valid_thresholds[1]], 'color': config.COLOR_YELLOW_SEMAFORO})
                last_thresh = valid_thresholds[1]
            steps.append({'range': [last_thresh, max_val], 'color': config.COLOR_RED_SEMAFORO}) # Rest is 'critical'
        else: # No specific thresholds, full bar is neutral
             steps.append({'range': [0, max_val], 'color': config.COLOR_NEUTRAL_METRIC})
    else: # Lower is worse e.g., eNPS, Retention - Red, Yellow, Green
        if valid_thresholds:
            last_thresh = 0
            if len(valid_thresholds) >= 1: # At least a 'low' threshold defines 'critical'
                steps.append({'range': [last_thresh, valid_thresholds[0]], 'color': config.COLOR_RED_SEMAFORO})
                last_thresh = valid_thresholds[0]
            if len(valid_thresholds) >= 2: # A 'medium' threshold defines 'warning'
                steps.append({'range': [last_thresh, valid_thresholds[1]], 'color': config.COLOR_YELLOW_SEMAFORO})
                last_thresh = valid_thresholds[1]
            steps.append({'range': [last_thresh, max_val], 'color': config.COLOR_GREEN_SEMAFORO}) # Rest is 'good'
        else:
            steps.append({'range': [0, max_val], 'color': config.COLOR_NEUTRAL_METRIC})


    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_value_for_gauge,
        title={'text': title_text, 'font': {'size': 16, 'color': config.COLOR_GRAY_TEXT}},
        number={'font': {'size': 28}, 'suffix': unit if unit else "", 'valueformat': ".1f"},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': config.COLOR_NEUTRAL_METRIC if pd.isna(value) else 'rgba(0,0,0,0)', 'thickness': 0.05}, # Bar is value indicator
            'bgcolor': "rgba(0,0,0,0)", # Transparent background for steps to show
            'borderwidth': 1,
            'bordercolor': "lightgray",
            'steps': steps,
            'threshold': {
                'line': {'color': config.COLOR_TARGET_LINE, 'width': 3},
                'thickness': 0.8,
                'value': target_threshold if target_threshold is not None else (ht if higher_is_worse and ht else (lt if not higher_is_worse and lt else None))
            } if target_threshold is not None or (ht if higher_is_worse else lt) is not None else {}
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=5), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def create_trend_chart(df, date_col, value_cols, title_key, lang,
                       y_axis_title_key="value_axis", x_axis_title_key="date_time_axis",
                       show_average_line=False, target_value_map=None): # target_value_map: { 'col_name': target_value }
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = lang_texts.get(title_key, title_key)
    x_title_text = lang_texts.get(x_axis_title_key, "Date/Time")
    y_title_text = lang_texts.get(y_axis_title_key, "Value")

    value_cols = [col for col in value_cols if col in df.columns] # Ensure columns exist
    if df.empty or not date_col in df.columns or not value_cols:
        return go.Figure().update_layout(title_text=f"{title_text} ({lang_texts.get('no_data_for_selection')})", annotations=[dict(text=lang_texts.get('no_data_for_selection'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)])

    fig = px.line(df, x=date_col, y=value_cols, markers=True, color_discrete_sequence=config.COLOR_SCHEME_CATEGORICAL)

    for i, col in enumerate(value_cols):
        if show_average_line and pd.api.types.is_numeric_dtype(df[col]):
            avg_val = df[col].mean()
            if pd.notna(avg_val):
                fig.add_hline(y=avg_val, line_dash="dot",
                              annotation_text=f"{lang_texts.get('average_label', 'Avg')} {col}: {avg_val:.1f}",
                              annotation_position="bottom right" if i % 2 == 0 else "bottom left", # Alternate position
                              line_color=config.COLOR_SCHEME_CATEGORICAL[i % len(config.COLOR_SCHEME_CATEGORICAL)], opacity=0.7)

        if target_value_map and col in target_value_map and pd.notna(target_value_map[col]):
            fig.add_hline(y=target_value_map[col], line_dash="dash",
                          annotation_text=f"{lang_texts.get('target_label', 'Target')} {col}: {target_value_map[col]}",
                          annotation_position="top left" if i % 2 == 0 else "top right",
                          line_color=config.COLOR_SCHEME_CATEGORICAL[i % len(config.COLOR_SCHEME_CATEGORICAL)], line_width=2, opacity=0.9)

    fig.update_layout(
        title_text=title_text,
        yaxis_title=y_title_text,
        xaxis_title=x_title_text,
        legend_title_text=lang_texts.get("metrics_legend", "Metrics"),
        hovermode="x unified",
        xaxis=dict(showgrid=False), # Cleaner look
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.5)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_traces(hovertemplate='%{y:.2f}<extra></extra>')
    return fig

def create_comparison_bar_chart(df, x_col, y_cols, title_key, lang,
                                y_axis_title_key="count_axis", x_axis_title_key="category_axis",
                                barmode='group', text_auto_format='.2s'):
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = lang_texts.get(title_key, title_key)
    x_title_text = lang_texts.get(x_axis_title_key, "Category")
    y_title_text = lang_texts.get(y_axis_title_key, "Count")

    y_cols_list = y_cols if isinstance(y_cols, list) else [y_cols]
    y_cols_list = [col for col in y_cols_list if col in df.columns]

    if df.empty or not x_col in df.columns or not y_cols_list:
        return go.Figure().update_layout(title_text=f"{title_text} ({lang_texts.get('no_data_for_selection')})", annotations=[dict(text=lang_texts.get('no_data_for_selection'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)])

    # Create localized labels for y-columns
    y_col_labels = {col: lang_texts.get(f"{col.lower().replace(' ', '_')}_label", col.replace('_', ' ').title()) for col in y_cols_list}


    fig = px.bar(df, x=x_col, y=y_cols_list, title=title_text, barmode=barmode,
                 color_discrete_sequence=config.COLOR_SCHEME_CATEGORICAL,
                 labels=y_col_labels, # Apply localized labels
                 text_auto=True)
    
    fig.update_traces(texttemplate=f'%{{y:{text_auto_format}}}', textposition='outside' if barmode != 'stack' else 'inside')

    fig.update_layout(
        yaxis_title=y_title_text,
        xaxis_title=x_title_text,
        legend_title_text=lang_texts.get("metrics_legend", "Metrics"),
        hovermode="x unified",
        xaxis_tickangle=-30,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.5)'),
        xaxis=dict(showgrid=False)
    )
    return fig


def display_metric_card(st_object, label_key, value, lang,
                        previous_value=None, unit="", higher_is_better=None,
                        help_text_key=None, target_value=None):
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    label_text = lang_texts.get(label_key, label_key)
    help_text_content = lang_texts.get(help_text_key, "") if help_text_key else None

    formatted_value = "N/A"
    delta_text = None
    delta_color = "normal" # 'normal' (green for up), 'inverse' (red for up), 'off' (grey)

    if pd.notna(value) and isinstance(value, (int, float)):
        formatted_value = f"{value:,.0f}{unit}" if value % 1 == 0 and abs(value) >= 1000 else f"{value:,.1f}{unit}"

        if previous_value is not None and pd.notna(previous_value) and isinstance(previous_value, (int,float)):
            delta = value - previous_value
            delta_percentage = ((value - previous_value) / previous_value) * 100 if previous_value != 0 else (float('inf') if value > 0 else 0)
            
            sign = "+" if delta >= 0 else ""
            delta_text = f"{sign}{delta:,.1f}{unit} ({sign}{delta_percentage:.1f}%)"

            if higher_is_better is not None:
                if delta > 0: delta_color = "normal" if higher_is_better else "inverse"
                elif delta < 0: delta_color = "inverse" if higher_is_better else "normal"
                else: delta_color = "off"
        
        # Optionally add styling if value is against target
        if target_value is not None and higher_is_better is not None:
            if (higher_is_better and value < target_value) or (not higher_is_better and value > target_value):
                label_text = f"⚠️ {label_text}" # Add warning emoji
            elif (higher_is_better and value >= target_value) or (not higher_is_better and value <= target_value):
                 label_text = f"✅ {label_text}" # Add success emoji
    elif value == "N/A" or pd.isna(value) :
         formatted_value = "N/A"


    st_object.metric(label=label_text, value=formatted_value, delta=delta_text, delta_color=delta_color, help=help_text_content)


def create_enhanced_radar_chart(df_radar_input, categories_col, values_col, title_key, lang,
                                 group_col=None, range_max=None, higher_is_better_map=None):
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = lang_texts.get(title_key, title_key)
    df_radar = df_radar_input.copy()

    if df_radar.empty or not all(col in df_radar.columns for col in [categories_col, values_col]):
        return go.Figure().update_layout(title_text=f"{title_text} ({lang_texts.get('no_data_radar')})", annotations=[dict(text=lang_texts.get('no_data_radar'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)])

    if range_max is None:
        valid_max = df_radar[values_col].max() if not df_radar[values_col].empty and pd.notna(df_radar[values_col].max()) else 5
        range_max = valid_max * 1.1 if pd.notna(valid_max) else 5


    fig = go.Figure()
    color_sequence = config.COLOR_SCHEME_CATEGORICAL * (df_radar[group_col].nunique() // len(config.COLOR_SCHEME_CATEGORICAL) + 1) if group_col and group_col in df_radar.columns else [config.COLOR_NEUTRAL_METRIC]

    if group_col and group_col in df_radar.columns and df_radar[group_col].nunique() > 0:
        for i, (group_name, group_data) in enumerate(df_radar.groupby(group_col)):
            unique_categories = df_radar[categories_col].unique()
            group_data_ordered = pd.DataFrame({categories_col: unique_categories}).merge(
                group_data, on=categories_col, how='left'
            ).fillna(0) # Fill NaN values with 0 for plotting

            fig.add_trace(go.Scatterpolar(
                r=group_data_ordered[values_col],
                theta=group_data_ordered[categories_col],
                fill='toself',
                name=str(group_name),
                line_color=color_sequence[i % len(color_sequence)],
                opacity=0.7,
                hovertemplate='<b>%{theta}</b><br>' + f'{group_name}: %{{r:.1f}}' + '<extra></extra>'
            ))
    else:
        fig.add_trace(go.Scatterpolar(
            r=df_radar[values_col],
            theta=df_radar[categories_col],
            fill='toself',
            name=lang_texts.get("average_score_label", "Average Score"),
            line_color=color_sequence[0],
            opacity=0.7,
            hovertemplate='<b>%{theta}</b>: %{r:.1f}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=18)),
        polar=dict(
            bgcolor="rgba(255,255,255,0.1)",
            radialaxis=dict(
                visible=True,
                range=[0, range_max],
                showline=True,
                showticklabels=True,
                gridcolor="rgba(0,0,0,0.1)",
                linecolor="rgba(0,0,0,0.2)",
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                showline=True,
                showticklabels=True,
                gridcolor="rgba(0,0,0,0.1)",
                linecolor="rgba(0,0,0,0.2)",
                tickfont=dict(size=11),
                direction="clockwise" # Standard for radar
            )
        ),
        showlegend=(group_col is not None and df_radar[group_col].nunique() > 1),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=100, b=50), # Adjusted margins for legend
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_stress_semaforo_visual(stress_level, lang):
    """Creates a visual 'semaforo' (traffic light) bar for stress level."""
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_key = "overall_stress_indicator_title" # This specific key might be better passed or hardcoded if always the same
    title_text = lang_texts.get(title_key, "Overall Stress Indicator")

    val_for_gauge = 0
    semaforo_color = config.COLOR_GRAY_TEXT
    status_text = lang_texts.get('stress_na', 'N/A')

    if pd.notna(stress_level):
        val_for_gauge = float(stress_level)
        if val_for_gauge <= config.STRESS_LEVEL_LOW_THRESHOLD:
            status_text = lang_texts.get('stress_low', 'Low')
            semaforo_color = config.COLOR_GREEN_SEMAFORO
        elif val_for_gauge <= config.STRESS_LEVEL_MEDIUM_THRESHOLD:
            status_text = lang_texts.get('stress_medium', 'Moderate')
            semaforo_color = config.COLOR_YELLOW_SEMAFORO
        else:
            status_text = lang_texts.get('stress_high', 'High')
            semaforo_color = config.COLOR_RED_SEMAFORO
    else: # Handle NaN input
        val_for_gauge = 0 # Or some other representation for N/A on the gauge

    fig = go.Figure(go.Indicator(
        mode="gauge+number", # Using gauge for a horizontal bar effect + number
        value=val_for_gauge,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{title_text}: <b>{status_text}</b>", 'font': {'size': 16, 'color': config.COLOR_GRAY_TEXT}, 'align': "left"},
        number={'valueformat': ".1f", 'font': {'size': 24, 'color': semaforo_color}, 'suffix': " / "+str(config.STRESS_LEVEL_HIGH)},
        gauge={
            'shape': "bullet",
            'axis': {'range': [0, config.STRESS_LEVEL_HIGH], 'visible': True, 'tickvals': [0, config.STRESS_LEVEL_LOW_THRESHOLD, config.STRESS_LEVEL_MEDIUM_THRESHOLD, config.STRESS_LEVEL_HIGH]},
            'threshold': { # Not using this as a 'target' but rather for segmentation in steps
                'line': {'color': "rgba(0,0,0,0)", 'width': 0}, # Make threshold line invisible
                'thickness': 0,
                'value': config.STRESS_LEVEL_MEDIUM_THRESHOLD},
            'steps': [
                {'range': [0, config.STRESS_LEVEL_LOW_THRESHOLD], 'color': config.COLOR_GREEN_SEMAFORO, 'name': lang_texts.get('stress_low')},
                {'range': [config.STRESS_LEVEL_LOW_THRESHOLD, config.STRESS_LEVEL_MEDIUM_THRESHOLD], 'color': config.COLOR_YELLOW_SEMAFORO, 'name': lang_texts.get('stress_medium')},
                {'range': [config.STRESS_LEVEL_MEDIUM_THRESHOLD, config.STRESS_LEVEL_HIGH], 'color': config.COLOR_RED_SEMAFORO, 'name': lang_texts.get('stress_high')}
            ],
            'bar': {'color': semaforo_color, 'thickness': 0.5} # Main bar representing the current value
        }))
    fig.update_layout(height=120, margin=dict(t=40, b=20, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)')
    return fig
