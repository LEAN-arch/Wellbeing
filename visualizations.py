# visualizations.py
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import config # Import your configuration

def create_kpi_gauge(value, title_key, lang, max_value=100, low_threshold=config.ROTATION_RATE_LOW_THRESHOLD, high_threshold=config.ROTATION_RATE_HIGH_THRESHOLD):
    """Creates a KPI gauge. Title is a key for localization."""
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = lang_texts.get(title_key, title_key)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value if pd.notna(value) else 0, # Handle NaN for value
        title={'text': title_text, 'font': {'size': 18}},
        number={'suffix': "%" if "%" in title_text else "", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, max_value], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': config.COLOR_NEUTRAL_METRIC},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, low_threshold], 'color': config.COLOR_GREEN_SEMAFORO},
                {'range': [low_threshold, high_threshold], 'color': config.COLOR_YELLOW_SEMAFORO},
                {'range': [high_threshold, max_value], 'color': config.COLOR_RED_SEMAFORO}],
            'threshold': {
                'line': {'color': config.COLOR_RED_SEMAFORO, 'width': 4},
                'thickness': 0.75,
                'value': high_threshold if pd.notna(value) else 0 # Ensure threshold value is valid
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=20))
    # For screen readers, a text description should accompany this visual gauge.
    # This could be added as a st.caption() near where the gauge is displayed in app.py
    return fig

def create_line_chart(df, x_col, y_cols, title_key, lang, y_axis_title="Value", x_axis_title_key="date_time_axis"):
    """Creates a generic line chart for trends, with localized titles."""
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = lang_texts.get(title_key, title_key)
    x_title_text = lang_texts.get(x_axis_title_key, "Date/Time")
    y_title_text = y_axis_title # Keep Y axis title direct for now, or localize

    if df.empty or not all(col in df.columns for col in [x_col] + y_cols):
        fig = go.Figure().update_layout(
            title_text=f"{title_text} ({lang_texts.get('no_data_available', 'No data available')})",
            xaxis_visible=False, yaxis_visible=False,
            annotations=[dict(text=lang_texts.get('no_data_available', 'No data available'), showarrow=False)]
        )
        return fig

    fig = px.line(df, x=x_col, y=y_cols, title=title_text, markers=True,
                  color_discrete_sequence=config.COLOR_SCHEME_CATEGORICAL)
    fig.update_layout(
        yaxis_title=y_title_text,
        xaxis_title=x_title_text,
        legend_title_text=lang_texts.get("metrics_legend", "Metrics"),
        hovermode="x unified" # Improves hover interactivity
    )
    # Future: Add more detailed aria-label for screen readers, summarizing the chart data/trends.
    return fig

def create_bar_chart(df, x_col, y_cols, title_key, lang, y_axis_title="Count", x_axis_title_key="category_axis", barmode='group'):
    """Creates a generic bar chart, with localized titles."""
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = lang_texts.get(title_key, title_key)
    x_title_text = lang_texts.get(x_axis_title_key, "Category")
    y_title_text = y_axis_title

    if df.empty or not all(col in df.columns for col in [x_col] + y_cols):
        fig = go.Figure().update_layout(
            title_text=f"{title_text} ({lang_texts.get('no_data_available', 'No data available')})",
            xaxis_visible=False, yaxis_visible=False,
            annotations=[dict(text=lang_texts.get('no_data_available', 'No data available'), showarrow=False)]
        )
        return fig

    if not isinstance(y_cols, list):
        y_cols = [y_cols]

    fig = px.bar(df, x=x_col, y=y_cols, title=title_text, barmode=barmode,
                 color_discrete_sequence=config.COLOR_SCHEME_CATEGORICAL,
                 labels={col: col.replace('_', ' ').title() for col in y_cols}) # Auto-labels from column names
    fig.update_layout(
        yaxis_title=y_title_text,
        xaxis_title=x_title_text,
        legend_title_text=lang_texts.get("metrics_legend", "Metrics"),
        hovermode="x unified"
    )
    return fig

def display_metric_card(st_object, label_key, value, lang, delta=None, delta_color="normal", help_text_key=None):
    """Displays a single metric card, with localized label and optional help text."""
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    label_text = lang_texts.get(label_key, label_key)
    help_text = lang_texts.get(help_text_key, "") if help_text_key else ""

    # Format value nicely if it's numeric
    if isinstance(value, (int, float)) and pd.notna(value):
        if abs(value) >= 1000 and value % 1 == 0: # Whole number >= 1000
             formatted_value = f"{int(value):,}"
        elif isinstance(value, float):
             formatted_value = f"{value:.1f}" if "Score" in label_text or "Rate" in label_text or "%" in label_text else f"{value:.0f}"
        else:
             formatted_value = str(value)
        if "%" in label_text and not "%" in formatted_value and formatted_value != "N/A":
             formatted_value += "%"

    else:
        formatted_value = str(value) if pd.notna(value) else "N/A"


    st_object.metric(label=label_text, value=formatted_value, delta=delta, delta_color=delta_color, help=help_text if help_text else None)


def create_radar_chart(df_radar, categories_col, values_col, title_key, lang, group_col=None):
    """Creates a radar chart with localized title.
    df_radar should be a DataFrame with columns for categories and their values.
    Example:
        Metric         Average Score
        Initiative     4.2
        Punctuality    4.5
    """
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    title_text = lang_texts.get(title_key, title_key)

    if df_radar.empty or not all(col in df_radar.columns for col in [categories_col, values_col]):
        fig = go.Figure().update_layout(
            title_text=f"{title_text} ({lang_texts.get('no_data_radar', 'No data for radar')})",
            xaxis_visible=False, yaxis_visible=False,
            annotations=[dict(text=lang_texts.get('no_data_radar', 'No data for radar'), showarrow=False)]
        )
        return fig

    fig = go.Figure()

    if group_col and group_col in df_radar.columns:
        for group_name, group_data in df_radar.groupby(group_col):
            fig.add_trace(go.Scatterpolar(
                r=group_data[values_col],
                theta=group_data[categories_col],
                fill='toself',
                name=str(group_name),
                hovertemplate='<b>%{theta}</b>: %{r:.1f}<extra></extra>' # Formatted hover
            ))
    else:
        fig.add_trace(go.Scatterpolar(
            r=df_radar[values_col],
            theta=df_radar[categories_col],
            fill='toself',
            name=title_text, # Or a more generic name if not grouped
            hovertemplate='<b>%{theta}</b>: %{r:.1f}<extra></extra>'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, df_radar[values_col].max() if not df_radar[values_col].empty and pd.notna(df_radar[values_col].max()) else 5] # Default max if data is odd
            )),
        showlegend=(group_col is not None and df_radar[group_col].nunique() > 1),
        title=title_text,
        colorway=config.COLOR_SCHEME_CATEGORICAL
    )
    return fig


def create_stress_semaforo(stress_level, lang):
    """Creates a textual stress 'semaforo' (traffic light) with localized terms."""
    lang_texts = config.TEXT_STRINGS.get(lang, config.TEXT_STRINGS["EN"])
    if pd.isna(stress_level) or stress_level is None:
        return f"⚪ {lang_texts.get('stress_na', 'N/A')}"
    
    level_text = ""
    icon = ""
    if stress_level <= config.STRESS_LEVEL_LOW_THRESHOLD:
        level_text = lang_texts.get('stress_low', 'Low')
        icon = "🟢"
    elif stress_level <= config.STRESS_LEVEL_MEDIUM_THRESHOLD:
        level_text = lang_texts.get('stress_medium', 'Medium')
        icon = "🟡"
    else:
        level_text = lang_texts.get('stress_high', 'High')
        icon = "🔴"
    return f"{icon} {level_text} ({stress_level:.1f})"