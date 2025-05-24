# visualizations.py
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import config

def create_kpi_gauge(value, title_key, lang, unit="%", higher_is_worse=True,
                     threshold_good=None, threshold_warning=None, threshold_critical=None, # Use these more specific names
                     target_line_value=None, max_value_override=None):
    """
    Creates an enhanced KPI gauge with clearer threshold logic.
    - threshold_good: The point up to which (or from which if higher_is_better) is considered good.
    - threshold_warning: The point up to which (or from which) is warning.
    - threshold_critical: Point beyond which is critical (can also be inferred).
    - target_line_value: A specific value for the 'threshold' line on the gauge.
    - max_value_override: Manually set the max of the gauge axis.
    """
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = lang_texts.get(title_key, title_key)

    current_value_for_gauge = 0
    if pd.notna(value) and isinstance(value, (int, float)):
        current_value_for_gauge = value
    
    # Determine dynamic max_value
    if max_value_override is not None:
        max_val = max_value_override
    else:
        max_val_candidates = [1.0] # Ensure max_val is at least 1, make it float for calcs
        if pd.notna(current_value_for_gauge): max_val_candidates.append(float(current_value_for_gauge) * 1.25)
        
        # Consider thresholds for max_val calculation
        if threshold_critical is not None and pd.notna(threshold_critical): max_val_candidates.append(float(threshold_critical) * 1.2)
        elif threshold_warning is not None and pd.notna(threshold_warning): max_val_candidates.append(float(threshold_warning) * 1.5)
        elif threshold_good is not None and pd.notna(threshold_good): max_val_candidates.append(float(threshold_good) * 2.0)
        else: max_val_candidates.append(100.0) # Default max if no thresholds

        max_val = max(max_val_candidates)
        if pd.notna(current_value_for_gauge) and float(current_value_for_gauge) > max_val :
            max_val = float(current_value_for_gauge) * 1.1
        if max_val <=0: max_val = 100.0 # Safety net if all inputs are 0 or negative

    steps = []
    if higher_is_worse: # e.g., Rotation Rate (Green -> Yellow -> Red)
        if threshold_good is not None and pd.notna(threshold_good):
            steps.append({'range': [0, threshold_good], 'color': config.COLOR_GREEN_SEMAFORO, 'name': lang_texts.get("good_label", "Good")})
            if threshold_warning is not None and pd.notna(threshold_warning) and threshold_warning > threshold_good:
                steps.append({'range': [threshold_good, threshold_warning], 'color': config.COLOR_YELLOW_SEMAFORO, 'name': lang_texts.get("warning_label", "Warning")})
                steps.append({'range': [threshold_warning, max_val], 'color': config.COLOR_RED_SEMAFORO, 'name': lang_texts.get("critical_label", "Critical")})
            else: # Only good threshold provided, rest is considered not good (e.g. red)
                steps.append({'range': [threshold_good, max_val], 'color': config.COLOR_RED_SEMAFORO, 'name': lang_texts.get("critical_label", "Critical")})
        elif threshold_warning is not None and pd.notna(threshold_warning): # Only warning threshold
            steps.append({'range': [0, threshold_warning], 'color': config.COLOR_YELLOW_SEMAFORO, 'name': lang_texts.get("warning_label", "Warning")})
            steps.append({'range': [threshold_warning, max_val], 'color': config.COLOR_RED_SEMAFORO, 'name': lang_texts.get("critical_label", "Critical")})
        else: # No thresholds provided for coloring steps
            steps.append({'range': [0, max_val], 'color': config.COLOR_NEUTRAL_METRIC})
    else: # Lower is worse (e.g., eNPS - Red -> Yellow -> Green)
        if threshold_warning is not None and pd.notna(threshold_warning): # Requires at least a warning (low end)
            steps.append({'range': [0, threshold_warning], 'color': config.COLOR_RED_SEMAFORO, 'name': lang_texts.get("critical_label", "Critical")})
            if threshold_good is not None and pd.notna(threshold_good) and threshold_good > threshold_warning:
                steps.append({'range': [threshold_warning, threshold_good], 'color': config.COLOR_YELLOW_SEMAFORO, 'name': lang_texts.get("warning_label", "Warning")})
                steps.append({'range': [threshold_good, max_val], 'color': config.COLOR_GREEN_SEMAFORO, 'name': lang_texts.get("good_label", "Good")})
            else: # Only warning provided, rest is warning too or neutral if only one threshold is 'good' type
                 steps.append({'range': [threshold_warning, max_val], 'color': config.COLOR_YELLOW_SEMAFORO, 'name': lang_texts.get("warning_label", "Warning")})
        elif threshold_good is not None and pd.notna(threshold_good): # Only good threshold, implies all below is not good
            steps.append({'range': [0, threshold_good], 'color': config.COLOR_RED_SEMAFORO, 'name': lang_texts.get("critical_label", "Critical")}) # Or yellow
            steps.append({'range': [threshold_good, max_val], 'color': config.COLOR_GREEN_SEMAFORO, 'name': lang_texts.get("good_label", "Good")})
        else: # No thresholds
            steps.append({'range': [0, max_val], 'color': config.COLOR_NEUTRAL_METRIC})

    # Determine actual threshold line on the gauge
    # If target_line_value is provided, use it. Otherwise, use a relevant critical/target threshold.
    gauge_threshold_value = target_line_value
    if gauge_threshold_value is None:
        if higher_is_worse:
            gauge_threshold_value = threshold_warning if threshold_warning is not None else threshold_good
        else: # lower is worse
            gauge_threshold_value = threshold_warning if threshold_warning is not None else threshold_good

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_value_for_gauge,
        title={'text': title_text, 'font': {'size': 16, 'color': config.COLOR_GRAY_TEXT}},
        number={'font': {'size': 28}, 'suffix': unit if unit and unit != "N/A" else "", 'valueformat': ".1f"},
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1, 'tickcolor': "darkgray"},
            'bar': {'color': config.COLOR_NEUTRAL_METRIC if pd.isna(value) else 'rgba(0,0,0,0.4)', 'thickness': 0.1}, # Make value bar distinct from steps
            'bgcolor': "white",
            'borderwidth': 1,
            'bordercolor': "lightgray",
            'steps': steps,
            'threshold': {
                'line': {'color': config.COLOR_TARGET_LINE, 'width': 3},
                'thickness': 0.8, # Thickness of the threshold line segment
                'value': gauge_threshold_value
            } if gauge_threshold_value is not None and pd.notna(gauge_threshold_value) else {} # Only show threshold if value is valid
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=5), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def create_trend_chart(df, date_col, value_cols, title_key, lang,
                       y_axis_title_key="value_axis", x_axis_title_key="date_time_axis",
                       show_average_line=False, target_value_map=None):
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = lang_texts.get(title_key, title_key)
    x_title_text = lang_texts.get(x_axis_title_key, "Date/Time")
    y_title_text = lang_texts.get(y_axis_title_key, "Value")

    value_cols = [col for col in value_cols if col in df.columns]
    if df.empty or not date_col in df.columns or not value_cols:
        return go.Figure().update_layout(title_text=f"{title_text} ({lang_texts.get('no_data_for_selection')})", annotations=[dict(text=lang_texts.get('no_data_for_selection'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)])

    fig = px.line(df, x=date_col, y=value_cols, markers=True, color_discrete_sequence=config.COLOR_SCHEME_CATEGORICAL)

    for i, col in enumerate(value_cols):
        color_for_annotations = config.COLOR_SCHEME_CATEGORICAL[i % len(config.COLOR_SCHEME_CATEGORICAL)]
        if show_average_line and pd.api.types.is_numeric_dtype(df[col]):
            avg_val = df[col].mean()
            if pd.notna(avg_val):
                fig.add_hline(y=avg_val, line_dash="dot",
                              annotation_text=f"{lang_texts.get('average_label', 'Avg')} {current_lang_texts.get(col.lower()+'_label', col)}: {avg_val:.1f}",
                              annotation_position="bottom right" if i % 2 == 0 else "bottom left",
                              line_color=color_for_annotations, opacity=0.7)

        if target_value_map and col in target_value_map and pd.notna(target_value_map[col]):
            fig.add_hline(y=target_value_map[col], line_dash="dash",
                          annotation_text=f"{lang_texts.get('target_label', 'Target')} {current_lang_texts.get(col.lower()+'_label', col)}: {target_value_map[col]}",
                          annotation_position="top left" if i % 2 == 0 else "top right",
                          line_color=color_for_annotations, line_width=2, opacity=0.9) # Use same color as series for target

    fig.update_layout(
        title_text=title_text,
        yaxis_title=y_title_text,
        xaxis_title=x_title_text,
        legend_title_text=lang_texts.get("metrics_legend", "Metrics"),
        hovermode="x unified",
        xaxis=dict(showgrid=False),
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

    y_col_labels = {col: current_lang_texts.get(f"{col.lower().replace(' ', '_').replace('(','').replace(')','').replace('%','')}_label", col.replace('_', ' ').title()) for col in y_cols_list}

    fig = px.bar(df, x=x_col, y=y_cols_list, title=title_text, barmode=barmode,
                 color_discrete_sequence=config.COLOR_SCHEME_CATEGORICAL,
                 labels=y_col_labels,
                 text_auto=True)
    
    fig.update_traces(texttemplate=f'%{{y:{text_auto_format}}}', textposition='outside' if barmode != 'stack' else 'inside', textfont_size=10)

    fig.update_layout(
        yaxis_title=y_title_text,
        xaxis_title=x_title_text,
        legend_title_text=lang_texts.get("metrics_legend", "Metrics"),
        hovermode="x unified",
        xaxis_tickangle=-30 if len(df[x_col].unique()) > 5 else 0, # Angle only if many categories
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.5)'),
        xaxis=dict(showgrid=False, type='category') # Ensure x-axis is treated as categorical for months
    )
    return fig


def display_metric_card(st_object, label_key, value, lang,
                        previous_value=None, unit="", higher_is_better=None,
                        help_text_key=None, target_value=None, lower_threshold=None, upper_threshold=None):
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    label_text_orig = lang_texts.get(label_key, label_key)
    label_text = label_text_orig
    help_text_content = lang_texts.get(help_text_key, "") if help_text_key else None

    formatted_value = "N/A"
    delta_text = None
    delta_color = "normal"

    icon = ""

    if pd.notna(value) and isinstance(value, (int, float)):
        raw_value = value
        formatted_value = f"{value:,.0f}{unit}" if value % 1 == 0 and abs(value) >= 1000 else f"{value:,.1f}{unit}"
        if unit == "%" and "%" not in formatted_value: formatted_value += "%"


        if previous_value is not None and pd.notna(previous_value) and isinstance(previous_value, (int,float)):
            delta_abs = value - previous_value
            # Ensure previous_value is not zero for percentage calculation or handle it
            if previous_value != 0:
                 delta_percentage = ((value - previous_value) / abs(previous_value)) * 100 # Use abs for denominator robustness
                 delta_text = f"{delta_abs:+,.1f}{unit} ({delta_percentage:+.1f}%)"
            else:
                 delta_text = f"{delta_abs:+,.1f}{unit} (Prev 0)"


            if higher_is_better is not None:
                if delta_abs > 0: delta_color = "normal" if higher_is_better else "inverse"
                elif delta_abs < 0: delta_color = "inverse" if higher_is_better else "normal"
                else: delta_color = "off"
        
        # Icon based on thresholds (if provided and higher_is_better is defined)
        if higher_is_better is not None:
            if upper_threshold is not None and value >= upper_threshold:
                icon = "✅ " if higher_is_better else "⚠️ "
            elif lower_threshold is not None and value < lower_threshold:
                icon = "⚠️ " if higher_is_better else "✅ "
            elif target_value is not None: # More generic target check if no explicit good/bad thresholds
                if (higher_is_better and value >= target_value) or (not higher_is_better and value <= target_value):
                     icon = "👍 " # Simpler icon if meeting generic target
    
    elif value == "N/A" or pd.isna(value) :
         formatted_value = "N/A"

    st_object.metric(label=icon + label_text, value=formatted_value, delta=delta_text, delta_color=delta_color, help=help_text_content)


def create_enhanced_radar_chart(df_radar_input, categories_col, values_col, title_key, lang,
                                 group_col=None, range_max_override=None, target_values=None): # target_values is a dict: {category_name: target_value}
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = lang_texts.get(title_key, title_key)
    df_radar = df_radar_input.copy()

    if df_radar.empty or not all(col in df_radar.columns for col in [categories_col, values_col]):
        return go.Figure().update_layout(title_text=f"{title_text} ({lang_texts.get('no_data_radar')})", annotations=[dict(text=lang_texts.get('no_data_radar'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)])

    # Determine range_max
    current_max_val = 0
    if not df_radar[values_col].empty and pd.notna(df_radar[values_col].max()):
        current_max_val = df_radar[values_col].max()
    
    if target_values:
        max_target = max(target_values.values()) if target_values else 0
        current_max_val = max(current_max_val, max_target)

    range_max = range_max_override if range_max_override is not None else (current_max_val * 1.1 if current_max_val > 0 else 5)
    range_max = max(range_max, 1) # ensure it's at least 1


    fig = go.Figure()
    color_sequence = config.COLOR_SCHEME_CATEGORICAL * (df_radar[group_col].nunique() // len(config.COLOR_SCHEME_CATEGORICAL) + 1) if group_col and group_col in df_radar.columns else [config.COLOR_NEUTRAL_METRIC]

    # Ensure categories are unique and consistently ordered for all traces
    all_categories_ordered = df_radar[categories_col].unique()

    if group_col and group_col in df_radar.columns and df_radar[group_col].nunique() > 0:
        for i, (group_name, group_data) in enumerate(df_radar.groupby(group_col)):
            # Create a complete category list for this group
            group_df_ordered = pd.DataFrame({categories_col: all_categories_ordered})
            group_df_ordered = pd.merge(group_df_ordered, group_data, on=categories_col, how='left').fillna(0)

            fig.add_trace(go.Scatterpolar(
                r=group_df_ordered[values_col],
                theta=group_df_ordered[categories_col],
                fill='toself',
                name=str(group_name),
                line_color=color_sequence[i % len(color_sequence)],
                opacity=0.6, # Slightly less opacity
                hovertemplate='<b>%{theta}</b><br>' + f'{group_name}: %{{r:.1f}}' + '<extra></extra>'
            ))
    else:
        # Ensure categories are in consistent order for single trace as well
        df_radar_ordered = pd.DataFrame({categories_col: all_categories_ordered})
        df_radar_ordered = pd.merge(df_radar_ordered, df_radar, on=categories_col, how='left').fillna(0)

        fig.add_trace(go.Scatterpolar(
            r=df_radar_ordered[values_col],
            theta=df_radar_ordered[categories_col],
            fill='toself',
            name=current_lang_texts.get("average_score_label", "Average Score"),
            line_color=color_sequence[0],
            opacity=0.7,
            hovertemplate='<b>%{theta}</b>: %{r:.1f}<extra></extra>'
        ))
    
    # Add target line if target_values are provided
    if target_values:
        target_r_values = [target_values.get(cat, 0) for cat in all_categories_ordered] # Get target for each category
        fig.add_trace(go.Scatterpolar(
            r=target_r_values,
            theta=all_categories_ordered,
            mode='lines',
            name=lang_texts.get("target_label", "Target"),
            line=dict(color=config.COLOR_TARGET_LINE, dash='dash', width=2),
            hoverinfo='skip' # Don't show hover for target line, keep focus on actuals
        ))


    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=18)),
        polar=dict(
            bgcolor="rgba(255,255,255,0.05)", # Lighter background
            radialaxis=dict(
                visible=True,
                range=[0, range_max],
                showline=False, # Cleaner look
                showticklabels=True,
                gridcolor="rgba(0,0,0,0.15)",
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                showline=False, # Cleaner look
                showticklabels=True,
                gridcolor="rgba(0,0,0,0.15)",
                tickfont=dict(size=11),
                direction="clockwise"
            )
        ),
        showlegend=True, # Always show legend if there's more than one trace (or target)
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=100, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_stress_semaforo_visual(stress_level, lang):
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = lang_texts.get("overall_stress_indicator_title", "Overall Stress Indicator")

    val_for_gauge = 0.0 # Ensure float for gauge
    semaforo_color = config.COLOR_GRAY_TEXT
    status_text = lang_texts.get('stress_na', 'N/A')

    if pd.notna(stress_level) and isinstance(stress_level, (int, float)):
        val_for_gauge = float(stress_level)
        if val_for_gauge <= config.STRESS_LEVEL_THRESHOLD_LOW:
            status_text = lang_texts.get('stress_low', 'Low')
            semaforo_color = config.COLOR_GREEN_SEMAFORO
        elif val_for_gauge <= config.STRESS_LEVEL_THRESHOLD_MEDIUM:
            status_text = lang_texts.get('stress_medium', 'Moderate')
            semaforo_color = config.COLOR_YELLOW_SEMAFORO
        else:
            status_text = lang_texts.get('stress_high', 'High')
            semaforo_color = config.COLOR_RED_SEMAFORO
    else: # Handle NaN input
        val_for_gauge = 0.0 

    max_scale = config.STRESS_LEVEL_MAX_SCALE

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val_for_gauge,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{title_text}: <b style='color:{semaforo_color};'>{status_text}</b>", 'font': {'size': 16}, 'align': "center"},
        number={'valueformat': ".1f", 'font': {'size': 26, 'color': semaforo_color}, 'suffix': f" / {max_scale:.0f}"},
        gauge={
            'shape': "bullet",
            'axis': {'range': [0, max_scale], 'visible': True, 
                     'tickvals': [0, config.STRESS_LEVEL_THRESHOLD_LOW, config.STRESS_LEVEL_THRESHOLD_MEDIUM, max_scale],
                     'ticktext': [0, f"{config.STRESS_LEVEL_THRESHOLD_LOW:.1f}", f"{config.STRESS_LEVEL_THRESHOLD_MEDIUM:.1f}", f"{max_scale:.0f}"],
                     'tickfont': {'size':10}
                    },
            'steps': [
                {'range': [0, config.STRESS_LEVEL_THRESHOLD_LOW], 'color': config.COLOR_GREEN_SEMAFORO},
                {'range': [config.STRESS_LEVEL_THRESHOLD_LOW, config.STRESS_LEVEL_THRESHOLD_MEDIUM], 'color': config.COLOR_YELLOW_SEMAFORO},
                {'range': [config.STRESS_LEVEL_THRESHOLD_MEDIUM, max_scale], 'color': config.COLOR_RED_SEMAFORO}
            ],
            'bar': {'color': semaforo_color, 'thickness': 0.6} # Bar represents current value
        }))
    fig.update_layout(height=100, margin=dict(t=35, b=25, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)') # Reduced height
    return fig
