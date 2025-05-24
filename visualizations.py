# visualizations.py
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import config

# --- Helper to get localized text ---
def get_lang_text(lang_code, key, default_text=""):
    """Retrieves localized text safely."""
    # Handle case where lang_code might not be in TEXT_STRINGS if there's an error elsewhere
    text_dict = config.TEXT_STRINGS.get(lang_code, config.TEXT_STRINGS["EN"])
    return text_dict.get(key, default_text)

# --- Helper for getting correct status text based on thresholds ---
def get_status_by_thresholds(value, higher_is_worse, threshold_good=None, threshold_warning=None):
    """Determines 'good', 'warning', 'critical' status based on value and thresholds."""
    if pd.isna(value) or value is None:
        return None # Indicate no specific status

    if higher_is_worse:
        if threshold_good is not None and pd.notna(threshold_good) and value <= threshold_good:
            return "good"
        elif threshold_warning is not None and pd.notna(threshold_warning) and value <= threshold_warning:
             # If good isn't met but warning is below/equal
            return "warning"
        # If none of the above conditions for 'good' or 'warning' met (i.e., value > warning or good), it's critical
        # Special case: if only 'good' threshold is provided for higher_is_worse, values > good are critical
        if threshold_warning is None and threshold_good is not None and pd.notna(threshold_good) and value > threshold_good:
             return "critical"
        elif threshold_warning is not None and pd.notna(threshold_warning) and value > threshold_warning:
             return "critical"
        # Default if no thresholds provided or conditions missed - implies a neutral state if needed
        return None

    else: # Lower is worse
        if threshold_good is not None and pd.notna(threshold_good) and value >= threshold_good:
            return "good"
        elif threshold_warning is not None and pd.notna(threshold_warning) and value >= threshold_warning: # Between warning and good (implies higher)
             return "warning"
        # If none of the above conditions for 'good' or 'warning' met (i.e., value < warning or good), it's critical
        if threshold_warning is not None and pd.notna(threshold_warning) and value < threshold_warning:
             return "critical"
        elif threshold_good is not None and pd.notna(threshold_good) and value < threshold_good: # If only 'good' and value < good, critical
            return "critical"
        # Default if no thresholds or conditions missed
        return None


def get_semaforo_color(status):
    """Maps status string to configured color."""
    if status == "good": return config.COLOR_GREEN_SEMAFORO
    if status == "warning": return config.COLOR_YELLOW_SEMAFORO
    if status == "critical": return config.COLOR_RED_SEMAFORO
    return config.COLOR_GRAY_TEXT # Default for None/N/A

# --- Enhanced KPI Gauge ---
def create_kpi_gauge(value, title_key, lang, unit="%", higher_is_worse=True,
                     threshold_good=None, threshold_warning=None, # Used for steps coloring
                     target_line_value=None, max_value_override=None,
                     previous_value=None):
    """
    Creates a visually actional KPI gauge.
    - value: Current metric value (numeric or None).
    - title_key: Localization key for the title.
    - lang: Language code.
    - unit: The unit displayed (e.g., "%").
    - higher_is_worse: If True (like rotation), lower values are good. If False (like eNPS), higher are good.
    - threshold_good: Boundary for 'good' range.
    - threshold_warning: Boundary for 'warning' range (implies values beyond warning are 'critical').
    - target_line_value: A specific value to highlight with a prominent line.
    - max_value_override: Manually set the max axis value.
    - previous_value: Value from previous period for delta comparison.
    """
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = lang_texts.get(title_key, title_key)

    current_value_for_gauge = 0
    if pd.notna(value) and isinstance(value, (int, float)):
        current_value_for_gauge = float(value) # Ensure float for calculations
    
    # Determine dynamic max_value
    if max_value_override is not None:
        max_val = max_value_override
    else:
        max_val_candidates = [1.0] # Ensure max_val is at least 1.0
        if pd.notna(current_value_for_gauge): max_val_candidates.append(current_value_for_gauge * 1.2) # Add padding to current value
        
        # Consider thresholds for max_val calculation, prioritize higher thresholds
        if threshold_warning is not None and pd.notna(threshold_warning): max_val_candidates.append(float(threshold_warning) * 1.1)
        if threshold_good is not None and pd.notna(threshold_good): max_val_candidates.append(float(threshold_good) * 1.5)
        if target_line_value is not None and pd.notna(target_line_value): max_val_candidates.append(float(target_line_value) * 1.1)

        # If none of the thresholds exist, or they are small, use a default or consider the current value more
        if not any(pd.notna(t) for t in [threshold_good, threshold_warning]) and pd.notna(current_value_for_gauge):
             max_val_candidates.append(current_value_for_gauge * 1.5 if current_value_for_gauge > 0 else 100) # Fallback

        max_val = max(max_val_candidates) if max_val_candidates else 100.0 # Default safeguard

        # Final check: ensure current value is comfortably within the max unless max is overridden
        if pd.notna(current_value_for_gauge) and current_value_for_gauge > max_val and max_value_override is None:
            max_val = current_value_for_gauge * 1.1

        if max_val <= 1e-6: max_val = 100.0 # Prevent max being zero or negative if inputs are non-positive

    steps = []
    # Define steps based on higher_is_worse and provided thresholds (must be numeric)
    # Order threshold points ascendingly to define ranges
    numeric_thresholds = sorted([t for t in [threshold_good, threshold_warning] if t is not None and pd.notna(t)])

    # Remove duplicates in case good == warning, etc.
    unique_numeric_thresholds = sorted(list(set(numeric_thresholds)))

    # Map colors to sorted unique thresholds
    color_map = {}
    if higher_is_worse: # Green -> Yellow -> Red
        if len(unique_numeric_thresholds) >= 1: # Good <= first threshold
            color_map[unique_numeric_thresholds[0]] = config.COLOR_GREEN_SEMAFORO
        if len(unique_numeric_thresholds) >= 2: # Yellow between first and second
            color_map[unique_numeric_thresholds[1]] = config.COLOR_YELLOW_SEMAFORO
            # Rest is Red >= second
        # If only one threshold, use it as the transition point from good/warning to critical/warning depending on higher_is_worse

        current_range_start = 0.0
        for thresh in unique_numeric_thresholds:
            color = config.COLOR_YELLOW_SEMAFORO # Default to yellow if warning exists but good doesn't quite map like expected
            if thresh in color_map: color = color_map[thresh] # Get specific color if mapped

            # The color of a step [A, B] should correspond to the region based on thresholds
            # Logic simplified: define segments based on ordered thresholds and assign color to segment
            # Segment [0, t1] has color based on t1. Segment [t1, t2] color based on t2 etc.
            # For Higher is Worse (0 -> good -> warning -> critical)
            if threshold_good is not None and pd.notna(threshold_good): # [0, good] is GREEN
                 if thresh == threshold_good:
                     steps.append({'range': [current_range_start, thresh], 'color': config.COLOR_GREEN_SEMAFORO})
                     current_range_start = thresh
            if threshold_warning is not None and pd.notna(threshold_warning): # [good, warning] is YELLOW (if good exists), [0, warning] if good doesn't
                 if thresh == threshold_warning:
                      steps.append({'range': [current_range_start, thresh], 'color': config.COLOR_YELLOW_SEMAFORO})
                      current_range_start = thresh

        # The segment after the last threshold is the final color
        final_color = config.COLOR_RED_SEMAFORO # Critical (Higher is Worse)
        if not unique_numeric_thresholds: final_color = config.COLOR_NEUTRAL_METRIC # No thresholds = Neutral

        steps.append({'range': [current_range_start, max_val], 'color': final_color})


    else: # Lower is worse (e.g., eNPS - Red -> Yellow -> Green)
        # For Lower is Worse (0 -> critical -> warning -> good)
        # Steps [0, t1] get one color, [t1, t2] get another...
        current_range_start = 0.0
        # Assumes threshold_warning is lower (bad) and threshold_good is higher (good)
        if threshold_warning is not None and pd.notna(threshold_warning): # [0, warning] is RED
             steps.append({'range': [current_range_start, threshold_warning], 'color': config.COLOR_RED_SEMAFORO})
             current_range_start = threshold_warning
        if threshold_good is not None and pd.notna(threshold_good) and (threshold_warning is None or threshold_good > threshold_warning): # [warning, good] is YELLOW
             steps.append({'range': [current_range_start, threshold_good], 'color': config.COLOR_YELLOW_SEMAFORO})
             current_range_start = threshold_good

        # The segment after the last threshold is the final color
        final_color = config.COLOR_GREEN_SEMAFORO # Good (Lower is Worse)
        if not unique_numeric_thresholds: final_color = config.COLOR_NEUTRAL_METRIC # No thresholds = Neutral

        steps.append({'range': [current_range_start, max_val], 'color': final_color})


    # Determine threshold line on the gauge - Use target_line_value if provided, else critical (higher) or good (lower)
    gauge_threshold_value = target_line_value
    if gauge_threshold_value is None:
         if higher_is_worse and threshold_warning is not None: gauge_threshold_value = threshold_warning
         elif not higher_is_worse and threshold_good is not None: gauge_threshold_value = threshold_good
         # Fallback if none available

    fig = go.Figure(go.Indicator(
        mode="gauge+number" + ("+delta" if previous_value is not None and pd.notna(previous_value) and pd.notna(value) else ""), # Only show delta if values available
        value=current_value_for_gauge if pd.notna(current_value_for_gauge) else 0, # Pass 0 or specific NA val for gauge visual
        title={'text': title_text, 'font': {'size': 16, 'color': config.COLOR_GRAY_TEXT}},
        number={'font': {'size': 28}, 'suffix': unit if unit and unit != "N/A" else "", 'valueformat': ".1f"},
        delta={'reference': previous_value if pd.notna(previous_value) else 0,
               'increasing': {'color': config.COLOR_RED_SEMAFORO if higher_is_worse else config.COLOR_GREEN_SEMAFORO},
               'decreasing': {'color': config.COLOR_GREEN_SEMAFORO if higher_is_worse else config.COLOR_RED_SEMAFORO},
               'font': {'size': 16}, 'valueformat': ".1f"}, # Format delta same as value
        gauge={
            'axis': {'range': [0, max_val], 'tickwidth': 1, 'tickcolor': "darkgray", 'nticks': 5},
            'bar': {'color': 'rgba(0,0,0,0.6)', 'thickness': 0.2},
            'bgcolor': "white", # Changed back to white, transparent can look odd without a background behind st.plotly_chart
            'borderwidth': 1,
            'bordercolor': "lightgray",
            'steps': steps,
            'threshold': {
                'line': {'color': config.COLOR_TARGET_LINE, 'width': 3},
                'thickness': 0.8,
                'value': gauge_threshold_value
            } if gauge_threshold_value is not None and pd.notna(gauge_threshold_value) else {}
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=5), paper_bgcolor='white')
    return fig


# --- Enhanced Trend Chart with Annotations and Selectors ---
def create_trend_chart(df, date_col, value_cols, title_key, lang,
                       y_axis_title_key="value_axis_label", x_axis_title_key="date_time_axis_label",
                       show_average_line=False, target_value_map=None, highlight_peaks_dips=False, # Peak/Dip simplified placeholder
                       rolling_avg_window=None, value_col_units=None): # value_col_units = {col: unit} for hovertemplate
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = get_lang_text(lang, title_key, title_key)
    x_title_text = get_lang_text(lang, x_axis_title_key, "Date/Time")
    y_title_text = get_lang_text(lang, y_axis_title_key, "Value")

    # Ensure columns exist and are numeric (except date)
    value_cols_numeric = [col for col in value_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if df.empty or date_col not in df.columns or not value_cols_numeric:
        return go.Figure().update_layout(title_text=f"{title_text} ({lang_texts.get('no_data_for_selection')})", annotations=[dict(text=lang_texts.get('no_data_for_selection'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)])

    fig = go.Figure()
    colors = config.COLOR_SCHEME_CATEGORICAL

    for i, col in enumerate(value_cols_numeric):
        series_color = colors[i % len(colors)]
        series_name = get_lang_text(lang, f"{col.lower().replace(' ', '_').replace('%','')}_label", col.replace('_', ' ').title())

        fig.add_trace(go.Scatter(x=df[date_col], y=df[col], mode='lines+markers', name=series_name,
                                 line=dict(color=series_color, width=2), marker=dict(size=6))) # Marker size slight adjusted

        if rolling_avg_window and isinstance(rolling_avg_window, int) and rolling_avg_window > 1:
            # Calculate rolling average, skip if data is sparse
            if len(df) >= rolling_avg_window / 2: # Only show rolling average if enough data points
                df[f'{col}_rolling_avg'] = df[col].rolling(window=rolling_avg_window, min_periods=1).mean()
                fig.add_trace(go.Scatter(x=df[date_col], y=df[f'{col}_rolling_avg'], mode='lines',
                                        name=f"{series_name} ({rolling_avg_window} {get_lang_text(lang, 'average_label')})",
                                        line=dict(color=series_color, width=1.5, dash='dash'),
                                        opacity=0.8))

    # Add static lines (Average and Target) AFTER all data traces so they appear above
    for i, col in enumerate(value_cols_numeric):
         series_color = colors[i % len(colors)]
         series_name = get_lang_text(lang, f"{col.lower().replace(' ', '_').replace('%','')}_label", col.replace('_', ' ').title())

         if show_average_line:
            avg_val = df[col].mean()
            if pd.notna(avg_val):
                fig.add_hline(y=avg_val, line_dash="dot",
                              annotation_text=f"{get_lang_text(lang, 'average_label')} {series_name}: {avg_val:.1f}",
                              annotation_position="bottom right" if i % 2 == 0 else "bottom left",
                              line_color=series_color, opacity=0.6,
                              # Annotation font and style consistent with layout
                              annotation=dict(font=dict(size=10, color=series_color), bgcolor="rgba(255,255,255,0.7)", bordercolor=series_color, borderwidth=0.5, padx=3, pady=3))


         if target_value_map and col in target_value_map and pd.notna(target_value_map[col]):
             target_val = target_value_map[col]
             fig.add_hline(y=target_val, line_dash="dash",
                           annotation_text=f"{get_lang_text(lang, 'target_label')} {series_name}: {target_val}", # .1f or .0f as appropriate
                           annotation_position="top left" if i % 2 == 0 else "top right",
                           line_color=config.COLOR_TARGET_LINE, line_width=2, opacity=0.9,
                           annotation=dict(font=dict(size=10, color=config.COLOR_TARGET_LINE), bgcolor="rgba(255,255,255,0.7)", bordercolor=config.COLOR_TARGET_LINE, borderwidth=0.5, padx=3, pady=3))

    # Highlight peaks/dips (simple example - needs proper statistical logic for robustness)
    if highlight_peaks_dips and len(df) > 1: # Needs at least two points to have trend
        # Example: Highlight first/last point and global min/max
        if not df.empty:
             first_point = df.iloc[0]
             last_point = df.iloc[-1]
             
             for col in value_cols_numeric:
                 series_name = get_lang_text(lang, f"{col.lower().replace(' ', '_').replace('%','')}_label", col.replace('_', ' ').title())
                 
                 if pd.notna(first_point[col]):
                      fig.add_annotation(x=first_point[date_col], y=first_point[col], text=f"Start", showarrow=False, bgcolor="rgba(255, 180, 0, 0.8)", font=dict(size=9, color='white'))
                 if pd.notna(last_point[col]):
                      fig.add_annotation(x=last_point[date_col], y=last_point[col], text=f"End", showarrow=False, bgcolor="rgba(0, 150, 255, 0.8)", font=dict(size=9, color='white'))

                 if len(df) > 2 and pd.api.types.is_numeric_dtype(df[col]):
                      global_max_loc = df[col].idxmax()
                      global_min_loc = df[col].idxmin()
                      global_max_point = df.loc[global_max_loc]
                      global_min_point = df.loc[global_min_loc]

                      # Avoid annotating start/end point again if they are min/max
                      if global_max_loc != 0 and global_max_loc != len(df)-1 and pd.notna(global_max_point[col]):
                           fig.add_annotation(x=global_max_point[date_col], y=global_max_point[col], text=f"Peak {global_max_point[col]:.1f}", showarrow=True, arrowhead=1, ax=0, ay=-40, font=dict(size=9, color='black'), bgcolor="rgba(255,255,255,0.7)", bordercolor='black', borderwidth=0.5)
                      if global_min_loc != 0 and global_min_loc != len(df)-1 and pd.notna(global_min_point[col]):
                           fig.add_annotation(x=global_min_point[date_col], y=global_min_point[col], text=f"Dip {global_min_point[col]:.1f}", showarrow=True, arrowhead=1, ax=0, ay=40, font=dict(size=9, color='black'), bgcolor="rgba(255,255,255,0.7)", bordercolor='black', borderwidth=0.5)


    # Dynamically update hover template to include units if provided
    hover_templates = []
    for col in value_cols_numeric:
        unit = value_col_units.get(col, '') if value_col_units else ''
        series_name = get_lang_text(lang, f"{col.lower().replace(' ', '_').replace('%','')}_label", col.replace('_', ' ').title())
        hover_templates.append(f"{series_name}: %{{y:.2f}}{unit}<extra></extra>")

    if rolling_avg_window and isinstance(rolling_avg_window, int) and rolling_avg_window > 1:
         for col in value_cols_numeric:
              series_name_roll = f"{get_lang_text(lang, f'{col.lower().replace(' ', '_').replace('%','')}_label', col.replace('_', ' ').title())} ({rolling_avg_window} {get_lang_text(lang, 'average_label')})"
              unit = value_col_units.get(col, '') if value_col_units else ''
              hover_templates.append(f"{series_name_roll}: %{{y:.2f}}{unit}<extra></extra>")


    fig.update_layout(
        title_text=title_text,
        yaxis_title=y_title_text,
        xaxis_title=x_title_text,
        legend_title_text=lang_texts.get("metrics_legend", "Metrics"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="gray"),
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
                ]))
        ),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9))
    )
    # Set hovertemplate using the list generated above
    if hover_templates:
        fig.update_traces(hovertemplate= '<br>'.join(hover_templates) + '<extra></extra>')


    return fig

# --- Enhanced Comparison Bar Chart ---
def create_comparison_bar_chart(df, x_col, y_cols, title_key, lang,
                                y_axis_title_key="count_axis_label", x_axis_title_key="category_axis_label",
                                barmode='group', show_total_for_stacked=False):
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = get_lang_text(lang, title_key, title_key)
    x_title_text = get_lang_text(lang, x_axis_title_key, "Category")
    y_title_text = get_lang_text(lang, y_axis_title_key, "Count")

    y_cols_list = y_cols if isinstance(y_cols, list) else [y_cols]
    y_cols_list = [col for col in y_cols_list if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    if df.empty or x_col not in df.columns or not y_cols_list:
        return go.Figure().update_layout(title_text=f"{title_text} ({lang_texts.get('no_data_for_selection')})", annotations=[dict(text=lang_texts.get('no_data_for_selection'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)])

    y_col_labels = {col: get_lang_text(lang, f"{col.lower().replace(' ', '_').replace('(','').replace(')','').replace('%','').replace('&', 'and')}_label", col.replace('_', ' ').title()) for col in y_cols_list}

    fig = px.bar(df, x=x_col, y=y_cols_list, title=title_text, barmode=barmode,
                 color_discrete_sequence=config.COLOR_SCHEME_CATEGORICAL,
                 labels=y_col_labels # Apply localized labels to axis/legend
                 )

    # Add individual segment labels using go.Bar trace update (px.bar text_auto sometimes has issues with stacking)
    # This ensures labels are consistently placed.
    fig.update_traces(texttemplate='%{y}', textposition='inside' if barmode == 'stack' else 'outside', textfont_size=10, insidetextanchor='middle', selector=dict(type='bar')) # For individual bars


    if barmode == 'stack' and show_total_for_stacked and y_cols_list:
        # Calculate total and add as annotations
        df_total = df.copy()
        df_total['total_stacked'] = df_total[y_cols_list].sum(axis=1)
        # Ensure x_col is sorted for consistent annotation placement
        df_total = df_total.sort_values(by=x_col)

        annotations = []
        for i, row in df_total.iterrows():
            annotations.append(dict(x=row[x_col], y=row['total_stacked'],
                                  text=f"{row['total_stacked']:.0f}", # Format as integer
                                  font=dict(family='Arial', size=11, color='black'),
                                  showarrow=False, yanchor='bottom', yshift=5))
        fig.update_layout(annotations=annotations)


    fig.update_layout(
        title_text=title_text,
        yaxis_title=y_title_text,
        xaxis_title=x_title_text,
        legend_title_text=lang_texts.get("metrics_legend", "Metrics"),
        hovermode="x unified",
        xaxis_tickangle=-30 if len(df[x_col].unique()) > 7 else 0,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=9)),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)'),
        xaxis=dict(showgrid=False, type='category')
    )
    return fig


# --- Enhanced Metric Card ---
def display_metric_card(st_object, label_key, value, lang,
                        previous_value=None, unit="", higher_is_better=None,
                        help_text_key=None, target_value=None,
                        threshold_good=None, threshold_warning=None):
    """
    Displays an enhanced metric card with delta, color, and icons based on performance.
    Uses more granular threshold check.
    """
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    label_text_orig = get_lang_text(lang, label_key, label_key)
    help_text_content = get_lang_text(lang, help_text_key, "") if help_text_key else None

    formatted_value = "N/A"
    delta_text = None
    delta_color = "normal"
    icon = "" # Prepend icon based on thresholds/targets

    if pd.notna(value) and isinstance(value, (int, float)):
        raw_value = value

        # --- Value Formatting ---
        if unit == "%": formatted_value = f"{value:,.1f}%"
        elif unit in [get_lang_text(lang, 'days_label'), get_lang_text(lang, 'hours_label')]: formatted_value = f"{value:,.0f} {unit}" # Whole numbers for counts/days
        elif abs(value) >= 1000: formatted_value = f"{value:,.0f}" # Large counts/scores
        else: formatted_value = f"{value:,.1f}" # Default decimal for scores/rates
        
        # Ensure unit is added if not part of format string and not %
        if unit and unit != "%" and not formatted_value.endswith(unit) and not formatted_value == "N/A":
             formatted_value += f" {unit}"


        # --- Delta Calculation ---
        if previous_value is not None and pd.notna(previous_value) and isinstance(previous_value, (int,float)):
            delta_abs = value - previous_value
            if previous_value != 0:
                 delta_percentage = (delta_abs / abs(previous_value)) * 100
                 # Use plus/minus for text
                 sign = "+" if delta_abs >= 0 else ""
                 delta_text = f"{sign}{delta_abs:,.1f} ({sign}{delta_percentage:,.0f}%)"
            else: # Avoid division by zero percentage, just show absolute change
                 sign = "+" if delta_abs >= 0 else ""
                 delta_text = f"{sign}{delta_abs:,.1f} (Prev 0)"

            # --- Delta Color ---
            if higher_is_better is not None:
                # Check if change is significant before coloring delta aggressively? (Optional improvement)
                if delta_abs > 1e-9 : delta_color = "normal" if higher_is_better else "inverse"
                elif delta_abs < -1e-9 : delta_color = "inverse" if higher_is_better else "normal"
                else: delta_color = "off"
            # If higher_is_better is None, color is always normal (gray)
        
        # --- Icon based on Thresholds ---
        # Check status relative to defined thresholds
        status = get_status_by_thresholds(raw_value, higher_is_better=higher_is_better,
                                            threshold_good=threshold_good, threshold_warning=threshold_warning)
        if status == "good": icon = "✅ "
        elif status == "warning": icon = "⚠️ "
        elif status == "critical": icon = "❗ " # Using a stronger icon for critical


    elif pd.isna(value): # Handle explicit NA
         formatted_value = "N/A"
         icon = "❓ "

    st_object.metric(label=icon + label_text_orig, value=formatted_value, delta=delta_text, delta_color=delta_color, help=help_text_content)


# --- Enhanced Radar Chart ---
def create_enhanced_radar_chart(df_radar_input, categories_col, values_col, title_key, lang,
                                 group_col=None, range_max_override=None, target_values_map=None, # {category_name: target_value}
                                 fill_opacity=0.6):
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = get_lang_text(lang, title_key, title_key)
    df_radar = df_radar_input.copy()

    if df_radar.empty or categories_col not in df_radar.columns or values_col not in df_radar.columns or df_radar[categories_col].empty:
        return go.Figure().update_layout(title_text=f"{title_text} ({lang_texts.get('no_data_radar')})", annotations=[dict(text=lang_texts.get('no_data_radar'), showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)])

    # Calculate range_max considering data and targets
    all_r_values = df_radar[values_col].tolist()
    if target_values_map:
        all_r_values.extend(list(target_values_map.values()))
    
    # Filter out NaNs and non-numeric, then find max, default to 5
    valid_r_values = [v for v in all_r_values if pd.notna(v) and isinstance(v, (int, float))]
    current_max_val = max(valid_r_values) if valid_r_values else 0
    
    range_max = range_max_override if range_max_override is not None else (current_max_val * 1.2 if current_max_val > 0 else 5.0)
    range_max = max(range_max, 1.0)


    fig = go.Figure()
    color_sequence = config.COLOR_SCHEME_CATEGORICAL

    # Ensure categories are unique and consistently ordered
    all_categories_ordered = df_radar[categories_col].unique()

    # Add actual data traces
    if group_col and group_col in df_radar.columns and df_radar[group_col].nunique() > 0:
        for i, (group_name, group_data) in enumerate(df_radar.groupby(group_col)):
            group_df_ordered = pd.DataFrame({categories_col: all_categories_ordered})
            # Merge with data, filling NaN scores (for missing categories in a group) with 0
            group_df_ordered = pd.merge(group_df_ordered, group_data, on=categories_col, how='left').fillna({values_col: 0})

            fig.add_trace(go.Scatterpolar(
                r=group_df_ordered[values_col], theta=group_df_ordered[categories_col],
                fill='toself', name=str(group_name),
                line_color=color_sequence[i % len(color_sequence)], opacity=fill_opacity,
                hovertemplate='<b>%{theta}</b><br>' + f'{str(group_name)}: %{{r:.1f}}' + '<extra></extra>'
            ))
    else: # Single trace (e.g., overall average)
        df_radar_ordered = pd.DataFrame({categories_col: all_categories_ordered})
        df_radar_ordered = pd.merge(df_radar_ordered, df_radar, on=categories_col, how='left').fillna({values_col: 0}) # Fill NaNs with 0

        fig.add_trace(go.Scatterpolar(
            r=df_radar_ordered[values_col], theta=df_radar_ordered[categories_col],
            fill='toself', name=get_lang_text(lang, "average_score_label", "Average Score"),
            line_color=color_sequence[0], opacity=fill_opacity,
            hovertemplate='<b>%{theta}</b>: %{r:.1f}<extra></extra>'
        ))
    
    # Add target line AFTER data traces
    if target_values_map:
        # Create R values for the target line, ordered by the same categories
        target_r_values = [target_values_map.get(cat, 0) for cat in all_categories_ordered]
        fig.add_trace(go.Scatterpolar(
            r=target_r_values, theta=all_categories_ordered, mode='lines',
            name=get_lang_text(lang, "target_label", "Target"),
            line=dict(color=config.COLOR_TARGET_LINE, dash='dash', width=2.5),
            hoverinfo='skip' # Don't show hover for target line
        ))

    fig.update_layout(
        title=dict(text=title_text, x=0.5, font=dict(size=18)),
        polar=dict(
            bgcolor="rgba(255,255,255,0.05)",
            radialaxis=dict(
                visible=True, range=[0, range_max], showline=False, showticklabels=True,
                gridcolor="rgba(0,0,0,0.1)", linecolor="rgba(0,0,0,0.1)", tickfont=dict(size=10)
            ),
            angularaxis=dict(
                showline=False, showticklabels=True, gridcolor="rgba(0,0,0,0.1)",
                linecolor="rgba(0,0,0,0.1)", tickfont=dict(size=11), direction="clockwise",
                 tickangle = 0 # Reset angular ticks, default auto-rotate might be sufficient with Plotly
            )
        ),
        showlegend=True, # Always show legend if there's more than one trace (incl. target)
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, font=dict(size=10)),
        margin=dict(l=50, r=50, t=100, b=50),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


# --- Enhanced Stress Semáforo (Visual Bar) ---
def create_stress_semaforo_visual(stress_level, lang, scale_max=config.STRESS_LEVEL_MAX_SCALE):
    """
    Creates a visual 'semaforo' (traffic light) bar for stress level.
    Shows value, status, and indicates position on a fixed scale (e.g., 1-10).
    """
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = get_lang_text(lang, "overall_stress_indicator_title", "Average Stress Level")

    val_for_gauge = 0.0 # Default for calculation/gauge position
    raw_value = stress_level # Keep raw value for NA check
    semaforo_color = config.COLOR_GRAY_TEXT
    status_text = get_lang_text(lang, 'stress_na', 'N/A')
    formatted_value = status_text

    if pd.notna(raw_value) and isinstance(raw_value, (int, float)):
        val_for_gauge = float(raw_value) # Use for position on gauge
        formatted_value = f"{raw_value:.1f}"
        
        # Determine status and color based on thresholds
        status = get_status_by_thresholds(raw_value, higher_is_worse=True, # Stress is higher is worse
                                           threshold_good=config.STRESS_LEVEL_THRESHOLD_LOW,
                                           threshold_warning=config.STRESS_LEVEL_THRESHOLD_MEDIUM)
        if status == "good":
            status_text = get_lang_text(lang, 'stress_low', 'Low')
            semaforo_color = config.COLOR_GREEN_SEMAFORO
        elif status == "warning":
            status_text = get_lang_text(lang, 'stress_medium', 'Moderate')
            semaforo_color = config.COLOR_YELLOW_SEMAFORO
        elif status == "critical":
            status_text = get_lang_text(lang, 'stress_high', 'High')
            semaforo_color = config.COLOR_RED_SEMAFORO
        else: # Should not happen if thresholds cover scale, but as fallback
            status_text = f"Value {raw_value:.1f}" # Show value if unexpected
            semaforo_color = config.COLOR_GRAY_TEXT

    # Clamp gauge value within the scale range to avoid indicator going off-scale
    gauge_value_clamped = max(0.0, min(float(scale_max), val_for_gauge)) if pd.notna(val_for_gauge) else 0.0


    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=gauge_value_clamped, # Value shown on the gauge axis
        domain={'x': [0, 1], 'y': [0.3, 0.7]}, # Give some vertical padding
        title={'text': f"<b>{status_text}</b>", 'font': {'size': 18, 'color': semaforo_color}, 'align': "center"},
        number={ # Display the actual value (not necessarily clamped)
            'valueformat': ".1f",
            'font': {'size': 26, 'color': semaforo_color},
            'suffix': f" / {scale_max:.0f}",
            'prefix': "" # No prefix by default
        },
        gauge={
            'shape': "bullet",
            'axis': {
                'range': [0, scale_max],
                'visible': True, 'showticklabels': True,
                # Define explicit ticks at boundaries for clarity
                'tickvals': [0, config.STRESS_LEVEL_THRESHOLD_LOW, config.STRESS_LEVEL_THRESHOLD_MEDIUM, scale_max],
                # Define text for ticks, linking to threshold meaning
                'ticktext': ["0", get_lang_text(lang, 'low_label'), get_lang_text(lang, 'moderate_label'), f"{scale_max:.0f}"], # Using labels
                'tickfont': {'size':10, 'color': config.COLOR_GRAY_TEXT},
                'tickmode': 'array' # Use the specified tickvals/ticktext
            },
            'steps': [
                {'range': [0, config.STRESS_LEVEL_THRESHOLD_LOW], 'color': config.COLOR_GREEN_SEMAFORO},
                {'range': [config.STRESS_LEVEL_THRESHOLD_LOW, config.STRESS_LEVEL_THRESHOLD_MEDIUM], 'color': config.COLOR_YELLOW_SEMAFORO},
                {'range': [config.STRESS_LEVEL_THRESHOLD_MEDIUM, scale_max], 'color': config.COLOR_RED_SEMAFORO}
            ],
             'bar': {'color': semaforo_color, 'thickness': 0.6} # The bar shows current region/color, thickness can vary
            # Alternative bar: make it a dark color representing the *actual value's position* on the gauge scale:
            # 'bar': {'color': 'rgba(50, 50, 50, 0.8)', 'thickness': 0.6}
            # In this version, the BAR is the current value position. The STEPS color the background range.
            # Let's revert bar to represent value:
            'bar': {'color': config.COLOR_NEUTRAL_METRIC if pd.isna(raw_value) else 'rgba(0,0,0,0.7)', 'thickness': 0.5}, # Dark bar shows actual value on the scale
            'bgcolor': "white", # Full background white
            'borderwidth': 1,
            'bordercolor': "lightgray"
        }))

    # The title is shown slightly below the plot, within the gauge
    # For accessibility, use st.subheader or st.caption in app.py as main text titles.

    fig.update_layout(height=150, margin=dict(t=25, b=25, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)') # Use transparent paper bgcolor

    return fig
