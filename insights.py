# insights.py
import pandas as pd
import config # To access thresholds if needed, and TEXT_STRINGS
from typing import List, Dict, Optional, Any, Union

def _get_insight_text(lang_code: str, key: str, default_text: Optional[str] = None, **kwargs) -> str:
    """Helper to get localized text from config and format it with provided kwargs."""
    effective_lang_code = lang_code if lang_code in config.TEXT_STRINGS else config.DEFAULT_LANG
    text_dict = config.TEXT_STRINGS.get(effective_lang_code, config.TEXT_STRINGS[config.DEFAULT_LANG])
    base_text = text_dict.get(key, default_text or key) # Fallback to key if text not found

    # Prepare kwargs for formatting: replace None with 'N/A' or a suitable string for display
    # This helps avoid TypeErrors with format specifiers like :.1f if the value is None.
    formatted_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, (float, int, np.number)) and pd.notna(v):
            # Check if key implies percentage to format it nicely for insights
            if "thresh" in k or "val" in k or "target" in k : # Simple heuristic
                # Assume these might need .1f formatting
                formatted_kwargs[k] = f"{float(v):.1f}" 
            else:
                formatted_kwargs[k] = v # Pass as is
        elif v is None or pd.isna(v):
            formatted_kwargs[k] = get_lang_text(lang_code, "status_na_label", "N/A") # Localized "N/A"
        else:
            formatted_kwargs[k] = v # Pass other types (like strings) as is

    try:
        return base_text.format(**formatted_kwargs)
    except KeyError as e:
        # This means a placeholder {key_name} in base_text was not found in formatted_kwargs
        # print(f"Insight text formatting WARNING (KeyError): For localization key '{key}', placeholder '{e}' missing in provided arguments. Kwargs: {kwargs}")
        return base_text # Return unformatted base_text
    except TypeError as e:
        # This can happen if a format specifier is incompatible with the type of the argument EVEN after basic formatting above.
        # print(f"Insight text formatting WARNING (TypeError): For localization key '{key}'. Error: {e}. Base: '{base_text}'. Formatted Kwargs: {formatted_kwargs}")
        return base_text # Return unformatted base_text
    except ValueError as e: # Catches issues like trying to format non-numeric with numeric specifier
        # print(f"Insight text formatting WARNING (ValueError): For localization key '{key}'. Error: {e}. Base: '{base_text}'. Formatted Kwargs: {formatted_kwargs}")
        return base_text


def generate_stability_insights(df_stability_filtered: pd.DataFrame, avg_rotation_val: Optional[float],
                                hires_exits_trend_df: Optional[pd.DataFrame], lang: str) -> List[str]:
    insights_list = []
    # Re-define local _ helper to use the robust _get_insight_text
    _ = lambda k, default=None, **kw: _get_insight_text(lang, k, default, **kw)

    # Rotation Insights with Target
    rot_good_thresh = config.STABILITY_ROTATION_RATE["good"]
    rot_warn_thresh = config.STABILITY_ROTATION_RATE["warning"]
    # Ensure avg_rotation_val is float or handled if None for comparisons
    current_avg_rotation = float(avg_rotation_val) if pd.notna(avg_rotation_val) else None


    if current_avg_rotation is not None: # Check after potential float conversion
        if current_avg_rotation > rot_warn_thresh:
            insights_list.append(f"🚨 **{_('rotation_high_alert')}:** " +
                                 _('rotation_high_insight_v2',
                                   rotation_val=current_avg_rotation, warn_thresh=rot_warn_thresh, target_thresh=config.STABILITY_ROTATION_RATE["target"]))
        elif current_avg_rotation > rot_good_thresh: # If it's just above target/good, but not warning
            insights_list.append(f"⚠️ **{_('rotation_moderate_warn')}:** " +
                                 _('rotation_moderate_insight_v2',
                                   rotation_val=current_avg_rotation, good_thresh=rot_good_thresh, target_thresh=config.STABILITY_ROTATION_RATE["target"])) # Using good_thresh as it's compared against this
        else: # Rotation is good
            insights_list.append(f"✅ **{_('rotation_good_status')}:** " +
                                 _('rotation_good_insight_v2',
                                   rotation_val=current_avg_rotation, good_thresh=rot_good_thresh))
    else:
        insights_list.append(_('rotation_no_data', "Rotation data not available for insights."))

    # Retention Insights (Example: focusing on 12-month retention if available)
    ret12_col_actual = config.COLUMN_MAP.get("retention_12m")
    if ret12_col_actual and ret12_col_actual in df_stability_filtered.columns:
        avg_retention_12m_val = df_stability_filtered[ret12_col_actual].mean()
        current_avg_ret12 = float(avg_retention_12m_val) if pd.notna(avg_retention_12m_val) else None

        if current_avg_ret12 is not None:
            ret_good_thresh = config.STABILITY_RETENTION["good"]
            ret_warn_thresh = config.STABILITY_RETENTION["warning"]
            if current_avg_ret12 < ret_warn_thresh: # Lower is worse for retention warning
                insights_list.append(f"📉 **{_('retention_low_alert')}:** " +
                                     _('retention_low_insight_v2',
                                       retention_val=current_avg_ret12, warn_thresh=ret_warn_thresh, target_thresh=ret_good_thresh))
            elif current_avg_ret12 < ret_good_thresh: # Below target but not warning
                insights_list.append(f"📈 **{_('retention_needs_improvement')}:** " +
                                     _('retention_improvement_insight_v2',
                                       retention_val=current_avg_ret12, target_thresh=ret_good_thresh))
            else: # Retention is good
                insights_list.append(f"👍 **{_('retention_good_status', 'Good 12m Retention')}:** " +
                                      _('retention_good_insight_detail', retention_val=current_avg_ret12, target_thresh=ret_good_thresh))

    # Hires vs Exits Trend
    # Make sure 'hires_exits_trend_df' can be None
    if hires_exits_trend_df is not None and not hires_exits_trend_df.empty and \
       'Hires_Total_Agg' in hires_exits_trend_df.columns and 'Exits_Total_Agg' in hires_exits_trend_df.columns:
        hires_exits_trend_df['NetChange'] = hires_exits_trend_df['Hires_Total_Agg'] - hires_exits_trend_df['Exits_Total_Agg']
        avg_net_chg = hires_exits_trend_df['NetChange'].mean()
        # Use .iloc[-1, hires_exits_trend_df.columns.get_loc('NetChange')] for robust access
        recent_net_chg_val = hires_exits_trend_df['NetChange'].iloc[-1] if not hires_exits_trend_df.empty else 0 
        
        if avg_net_chg < 0:
            insights_list.append(f"📉 **{_('net_loss_trend_warn', 'Net Employee Loss Trend')}:** " +
                                 _('net_loss_insight', avg_change=avg_net_chg))
        if recent_net_chg_val < -5 : # Example: significant recent loss
            insights_list.append(f"🔻 **{_('recent_loss_focus', 'Focus on Recent Departures')}:** "+
                                 _('recent_loss_insight_detail', recent_change=recent_net_chg_val))
    elif not insights_list: # Only add this if no other stability insights were generated
        insights_list.append(_("no_insights_generated"))

    return insights_list


def generate_safety_insights(df_safety_filtered: pd.DataFrame, current_dwa_val: Optional[float],
                             total_incidents_this_period: Optional[float], lang: str) -> List[str]:
    insights_list = []
    _ = lambda k, default=None, **kw: _get_insight_text(lang, k, default, **kw)
    
    inc_good_thresh = config.SAFETY_INCIDENTS["good"]
    inc_warn_thresh = config.SAFETY_INCIDENTS["warning"]
    inc_target_val = config.SAFETY_INCIDENTS["target"]
    current_total_incidents = float(total_incidents_this_period) if pd.notna(total_incidents_this_period) else None

    if current_total_incidents is not None:
        if current_total_incidents > inc_warn_thresh:
            insights_list.append(f"🚨 **{_('high_incidents_alert')}:** " +
                                 _('high_incidents_insight_v2', count=current_total_incidents, warn_thresh=inc_warn_thresh))
        elif current_total_incidents > inc_target_val:
            insights_list.append(f"⚠️ **{_('moderate_incidents_warn')}:** " +
                                 _('moderate_incidents_insight_v2', count=current_total_incidents, target_thresh=inc_target_val))
        else: # Incidents at or below target (ideally 0)
            insights_list.append(f"✅ **{_('low_incidents_status')}:** " +
                                 _('low_incidents_insight_v2', count=current_total_incidents))
    
    dwa_good_thresh = config.SAFETY_DAYS_NO_INCIDENTS["good"]
    dwa_warn_thresh = config.SAFETY_DAYS_NO_INCIDENTS["warning"]
    current_dwa_float = float(current_dwa_val) if pd.notna(current_dwa_val) else None

    if current_dwa_float is not None:
        if current_dwa_float < dwa_warn_thresh: # Lower DWA is bad
            insights_list.append(f"⏱️ **{_('dwa_low_warn')}:** " +
                                 _('dwa_low_insight_v2', days=current_dwa_float, warn_thresh=dwa_warn_thresh))
        elif current_dwa_float < dwa_good_thresh: # Better than warning, but not yet "good"
            insights_list.append(f"⏳ **{_('dwa_needs_improvement','Days Without Incidents: Monitor')}:** " +
                                 _('dwa_improvement_insight', days=current_dwa_float, target_thresh=dwa_good_thresh))
        else: # Good DWA
            insights_list.append(f"👍 **{_('dwa_good_status', 'Good Days Without Incidents')}:** " +
                                 _('dwa_good_insight_detail', days=current_dwa_float, target_thresh=dwa_good_thresh, default_text="{days:.0f} days without incidents. Keep up the great work!"))


    if not insights_list: insights_list.append(_("no_insights_generated"))
    return insights_list

def generate_engagement_insights(avg_enps_val_in: Optional[float], avg_climate_val_in: Optional[float],
                                 participation_val_in: Optional[float], lang: str) -> List[str]:
    insights_list = []
    _ = lambda k, default=None, **kw: _get_insight_text(lang, k, default, **kw)

    avg_enps_curr = float(avg_enps_val_in) if pd.notna(avg_enps_val_in) else None
    avg_climate_curr = float(avg_climate_val_in) if pd.notna(avg_climate_val_in) else None
    participation_curr = float(participation_val_in) if pd.notna(participation_val_in) else None

    enps_good_thresh = config.ENGAGEMENT_ENPS["good"]
    enps_warn_thresh = config.ENGAGEMENT_ENPS["warning"]
    if avg_enps_curr is not None:
        if avg_enps_curr < enps_warn_thresh: insights_list.append(f"📉 **{_('enps_low_alert')}:** " + _('enps_low_insight', enps_val=avg_enps_curr))
        elif avg_enps_curr < enps_good_thresh: insights_list.append(f"🤔 **{_('enps_needs_focus')}:** " + _('enps_focus_insight', enps_val=avg_enps_curr))
        else: insights_list.append(f"👍 **{_('enps_good_status')}:** " + _('enps_good_insight', enps_val=avg_enps_curr))
    
    clim_good_thresh = config.ENGAGEMENT_CLIMATE_SCORE["good"]
    clim_warn_thresh = config.ENGAGEMENT_CLIMATE_SCORE["warning"]
    if avg_climate_curr is not None:
        if avg_climate_curr < clim_warn_thresh: insights_list.append(f"🚩 **{_('climate_low_alert')}:** " + _('climate_low_insight', climate_val=avg_climate_curr))
        elif avg_climate_curr < clim_good_thresh:
            insights_list.append(f"📊 **{_('climate_needs_focus','Climate Score: Needs Attention')}:** " + 
                                 _('climate_focus_insight', climate_val=avg_climate_curr, target_thresh=clim_good_thresh))
    
    part_good_thresh = config.ENGAGEMENT_PARTICIPATION["good"]
    if participation_curr is not None and participation_curr < part_good_thresh:
        insights_list.append(f"📝 **{_('participation_low_warn','Low Survey Participation')}:** " +
                             _('participation_low_insight', part_val=participation_curr, target_thresh=part_good_thresh))

    if not insights_list: insights_list.append(_("engagement_no_critical_insights"))
    return insights_list

def generate_stress_insights(avg_stress_val_in: Optional[float], df_stress_trend_data: pd.DataFrame, lang: str) -> List[str]:
    insights_list = []
    _ = lambda k, default=None, **kw: _get_insight_text(lang, k, default, **kw)
    
    current_avg_stress = float(avg_stress_val_in) if pd.notna(avg_stress_val_in) else None
    stress_low_threshold = config.STRESS_LEVEL_PSYCHOSOCIAL["low"]
    stress_medium_threshold = config.STRESS_LEVEL_PSYCHOSOCIAL["medium"]

    if current_avg_stress is not None:
        if current_avg_stress > stress_medium_threshold:
            insights_list.append(f"🥵 **{_('stress_high_alert')}:** " + _('stress_high_insight', stress_val=current_avg_stress))
        elif current_avg_stress > stress_low_threshold:
             insights_list.append(f"😟 **{_('stress_moderate_warn')}:** " + _('stress_moderate_insight', stress_val=current_avg_stress))
        else:
             insights_list.append(f"😌 **{_('stress_low_status')}:** " + _('stress_low_insight', stress_val=current_avg_stress))
    
    workload_agg_col = "Workload_Avg" 
    psych_signals_agg_col = "Psych_Signals_Avg" 

    if df_stress_trend_data is not None and not df_stress_trend_data.empty and \
       workload_agg_col in df_stress_trend_data.columns and psych_signals_agg_col in df_stress_trend_data.columns and \
       len(df_stress_trend_data) >= 2:
        
        if pd.api.types.is_numeric_dtype(df_stress_trend_data[workload_agg_col]) and \
           pd.api.types.is_numeric_dtype(df_stress_trend_data[psych_signals_agg_col]):
            
            # Get last two valid (non-NaN) periods for trend comparison
            trend_data_for_diff = df_stress_trend_data[[workload_agg_col, psych_signals_agg_col]].dropna().tail(2)
            if len(trend_data_for_diff) == 2:
                workload_diff = trend_data_for_diff[workload_agg_col].diff().iloc[-1]
                psych_diff = trend_data_for_diff[psych_signals_agg_col].diff().iloc[-1]

                if pd.notna(workload_diff) and pd.notna(psych_diff):
                    # Define "significant" change thresholds (can be made configurable)
                    if workload_diff > 0.2 and psych_diff < -0.2: # Example: workload up by 0.2 AND psych down by 0.2
                        insights_list.append(f"📈 **{_('stress_trend_warn')}:** " + _('stress_trend_insight'))
    
    if not insights_list: insights_list.append(_("no_insights_generated"))
    return insights_list
