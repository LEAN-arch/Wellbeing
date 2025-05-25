# insights.py
import pandas as pd
import numpy as np
import config # To access thresholds if needed, and TEXT_STRINGS
from typing import List, Dict, Optional, Any, Union

# --- Helper to get localized and formatted text for insights ---
def _get_insight_text(lang_code: str, key: str, default_text: Optional[str] = None, **kwargs) -> str:
    """Helper to get localized text from config and format it with provided kwargs."""
    effective_lang_code = lang_code if lang_code in config.TEXT_STRINGS else config.DEFAULT_LANG
    text_dict = config.TEXT_STRINGS.get(effective_lang_code, config.TEXT_STRINGS[config.DEFAULT_LANG])
    base_text = text_dict.get(key, default_text or key)

    formatted_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, (float, int, np.number)) and pd.notna(v):
            try:
                float_v = float(v)
                if float_v % 1 == 0 and abs(float_v) < 10000: # Integers (not excessively large)
                    formatted_kwargs[k] = f"{int(float_v)}"
                elif abs(float_v) < 0.01 and float_v != 0: # Very small numbers
                    formatted_kwargs[k] = f"{float_v:.2e}" # Scientific notation
                else: # Default to one decimal place for other floats
                    formatted_kwargs[k] = f"{float_v:.1f}"
            except (TypeError, ValueError):
                formatted_kwargs[k] = str(v) # Fallback if float conversion unexpectedly fails
        elif v is None or pd.isna(v):
            # Use localized "N/A" or a hardcoded default if that specific key is missing
            na_text_key = "status_na_label"
            formatted_kwargs[k] = config.TEXT_STRINGS.get(effective_lang_code, {}).get(na_text_key, "N/A")
        else:
            formatted_kwargs[k] = v # Pass non-numeric values as is

    try:
        return base_text.format(**formatted_kwargs)
    except KeyError as e:
        # Placeholder for actual logging if desired:
        # print(f"Insight text formatting WARNING (KeyError): For lang '{lang_code}', key '{key}', placeholder '{e}' missing. Using raw text.")
        return base_text # Return base text if a placeholder is missing
    except (TypeError, ValueError) as e:
        # print(f"Insight text formatting WARNING ({type(e).__name__}): For lang '{lang_code}', key '{key}'. Error: {e}. Using raw text.")
        return base_text


# --- Stability Insights ---
def generate_stability_insights(df_stability_filtered: pd.DataFrame, avg_rotation_val: Optional[float],
                                hires_exits_trend_df: Optional[pd.DataFrame], lang: str) -> List[str]:
    insights_list = []
    _ = lambda k, default=None, **kw: _get_insight_text(lang, k, default, **kw)

    rot_good_thresh = config.STABILITY_ROTATION_RATE["good"]
    rot_warn_thresh = config.STABILITY_ROTATION_RATE["warning"]
    rot_target_val = config.STABILITY_ROTATION_RATE["target"]
    current_avg_rotation = float(avg_rotation_val) if pd.notna(avg_rotation_val) else None

    if current_avg_rotation is not None:
        if current_avg_rotation > rot_warn_thresh:
            insights_list.append(f"🚨 **{_('rotation_high_alert')}:** " +
                                 _('rotation_high_insight_v2',
                                   rotation_val=current_avg_rotation, warn_thresh=rot_warn_thresh, target_thresh=rot_target_val))
        elif current_avg_rotation > rot_good_thresh:
            insights_list.append(f"⚠️ **{_('rotation_moderate_warn')}:** " +
                                 _('rotation_moderate_insight_v2',
                                   rotation_val=current_avg_rotation, good_thresh=rot_good_thresh, target_thresh=rot_target_val))
        else:
            insights_list.append(f"✅ **{_('rotation_good_status')}:** " +
                                 _('rotation_good_insight_v2',
                                   rotation_val=current_avg_rotation, good_thresh=rot_good_thresh, target_thresh=rot_target_val))
    else:
        insights_list.append(_('rotation_no_data'))

    ret12_col_actual = config.COLUMN_MAP.get("retention_12m")
    avg_retention_12m_val = None # Initialize
    if ret12_col_actual and ret12_col_actual in df_stability_filtered.columns:
        avg_retention_12m_val_series = df_stability_filtered[ret12_col_actual].mean()
        if pd.notna(avg_retention_12m_val_series):
             avg_retention_12m_val = float(avg_retention_12m_val_series)


    if avg_retention_12m_val is not None:
        ret_good_thresh = config.STABILITY_RETENTION["good"]
        ret_warn_thresh = config.STABILITY_RETENTION["warning"] # Used as the critical lower bound
        if avg_retention_12m_val < ret_warn_thresh: # Below warning is critical
            insights_list.append(f"📉 **{_('retention_low_alert')}:** " +
                                 _('retention_low_insight_v2',
                                   retention_val=avg_retention_12m_val, warn_thresh=ret_warn_thresh, target_thresh=ret_good_thresh))
        elif avg_retention_12m_val < ret_good_thresh: # Between warning and good
            insights_list.append(f"📈 **{_('retention_needs_improvement')}:** " +
                                 _('retention_improvement_insight_v2',
                                   retention_val=avg_retention_12m_val, target_thresh=ret_good_thresh))
        else: # At or above good
            insights_list.append(f"👍 **{_('retention_good_status', default='Good 12m Retention')}:** " +
                                 _('retention_good_insight_detail', retention_val=avg_retention_12m_val, target_thresh=ret_good_thresh))

    if hires_exits_trend_df is not None and not hires_exits_trend_df.empty and \
       'Hires_Total_Agg' in hires_exits_trend_df.columns and 'Exits_Total_Agg' in hires_exits_trend_df.columns:
        df_trend_copy = hires_exits_trend_df.copy() # To avoid SettingWithCopyWarning if modifying
        df_trend_copy['NetChange'] = df_trend_copy['Hires_Total_Agg'] - df_trend_copy['Exits_Total_Agg']
        avg_net_chg = df_trend_copy['NetChange'].mean()
        recent_net_chg_val = df_trend_copy['NetChange'].iloc[-1] if not df_trend_copy.empty else 0

        if pd.notna(avg_net_chg) and avg_net_chg < -config.EPSILON: # Use EPSILON for float comparison to 0
            insights_list.append(f"📉 **{_('net_loss_trend_warn', default='Net Employee Loss Trend')}:** " +
                                 _('net_loss_insight', avg_change=avg_net_chg))
        # Consider defining a threshold for "significant" recent loss
        if pd.notna(recent_net_chg_val) and recent_net_chg_val < -5 :
            insights_list.append(f"🔻 **{_('recent_loss_focus', default='Focus on Recent Departures')}:** "+
                                 _('recent_loss_insight_detail', recent_change=recent_net_chg_val))
    
    if not insights_list and (current_avg_rotation is not None or avg_retention_12m_val is not None):
         insights_list.append(_('no_insights_generated_stability', default_text=_('no_insights_generated')))
    elif not insights_list: # Truly no data for key metrics
         insights_list.append(_('rotation_no_data')) # A more specific no data message for stability

    return insights_list


# --- Safety Insights ---
def generate_safety_insights(df_safety_filtered: pd.DataFrame, current_dwa_val_in: Optional[float],
                             total_incidents_this_period_in: Optional[float], lang: str) -> List[str]:
    insights_list = []
    _ = lambda k, default=None, **kw: _get_insight_text(lang, k, default, **kw)
    
    inc_good_thresh = config.SAFETY_INCIDENTS["good"] # Max for "good"
    inc_warn_thresh = config.SAFETY_INCIDENTS["warning"] # Max for "warning"
    inc_target_val = config.SAFETY_INCIDENTS["target"]
    current_total_incidents = float(total_incidents_this_period_in) if pd.notna(total_incidents_this_period_in) else None

    if current_total_incidents is not None:
        if current_total_incidents > inc_warn_thresh:
            insights_list.append(f"🚨 **{_('high_incidents_alert')}:** " +
                                 _('high_incidents_insight_v2', count=current_total_incidents, warn_thresh=inc_warn_thresh))
        elif current_total_incidents > inc_target_val : # Changed from good_thresh to target_val as target is 0.
            insights_list.append(f"⚠️ **{_('moderate_incidents_warn')}:** " +
                                 _('moderate_incidents_insight_v2', count=current_total_incidents, target_thresh=inc_target_val))
        else: # This means incidents are at or below target (e.g. 0)
            insights_list.append(f"✅ **{_('low_incidents_status')}:** " +
                                 _('low_incidents_insight_v2', count=current_total_incidents))
    
    dwa_good_thresh = config.SAFETY_DAYS_NO_INCIDENTS["good"]
    dwa_warn_thresh = config.SAFETY_DAYS_NO_INCIDENTS["warning"] # Min for "warning"
    current_dwa_float = float(current_dwa_val_in) if pd.notna(current_dwa_val_in) else None

    if current_dwa_float is not None:
        if current_dwa_float < dwa_warn_thresh:
            insights_list.append(f"⏱️ **{_('dwa_low_warn')}:** " +
                                 _('dwa_low_insight_v2', days=current_dwa_float, warn_thresh=dwa_warn_thresh))
        elif current_dwa_float < dwa_good_thresh:
            insights_list.append(f"⏳ **{_('dwa_needs_improvement')}:** " +
                                 _('dwa_improvement_insight', days=current_dwa_float, target_thresh=dwa_good_thresh))
        else:
            insights_list.append(f"👍 **{_('dwa_good_status', default='Good Days Without Incidents')}:** " +
                                 _('dwa_good_insight_detail', days=current_dwa_float, target_thresh=dwa_good_thresh)) #target_thresh wasn't used but good for context

    if not insights_list and (current_total_incidents is not None or current_dwa_float is not None) :
        insights_list.append(_('no_insights_generated_safety', default_text=_('no_insights_generated')))
    elif not insights_list: # No safety data at all
        insights_list.append(_('no_insights_generated_safety', default_text="Insufficient data for safety insights."))

    return insights_list

# --- Engagement Insights ---
def generate_engagement_insights(avg_enps_val_in: Optional[float], avg_climate_val_in: Optional[float],
                                 participation_val_in: Optional[float], lang: str) -> List[str]:
    insights_list = []
    _ = lambda k, default=None, **kw: _get_insight_text(lang, k, default, **kw)

    avg_enps_curr = float(avg_enps_val_in) if pd.notna(avg_enps_val_in) else None
    avg_climate_curr = float(avg_climate_val_in) if pd.notna(avg_climate_val_in) else None
    participation_curr = float(participation_val_in) if pd.notna(participation_val_in) else None

    if avg_enps_curr is not None:
        enps_good_thresh = config.ENGAGEMENT_ENPS["good"]
        enps_warn_thresh = config.ENGAGEMENT_ENPS["warning"] # Lower bound for "warning"
        if avg_enps_curr < enps_warn_thresh: # Critical
            insights_list.append(f"📉 **{_('enps_low_alert')}:** " + _('enps_low_insight', enps_val=avg_enps_curr))
        elif avg_enps_curr < enps_good_thresh: # Warning / Needs focus
            insights_list.append(f"🤔 **{_('enps_needs_focus')}:** " + _('enps_focus_insight', enps_val=avg_enps_curr))
        else: # Good
            insights_list.append(f"👍 **{_('enps_good_status')}:** " + _('enps_good_insight', enps_val=avg_enps_curr))
    
    if avg_climate_curr is not None:
        clim_good_thresh = config.ENGAGEMENT_CLIMATE_SCORE["good"]
        clim_warn_thresh = config.ENGAGEMENT_CLIMATE_SCORE["warning"]
        if avg_climate_curr < clim_warn_thresh: # Critical
            insights_list.append(f"🚩 **{_('climate_low_alert')}:** " + _('climate_low_insight', climate_val=avg_climate_curr))
        elif avg_climate_curr < clim_good_thresh: # Warning / Needs focus
            insights_list.append(f"📊 **{_('climate_needs_focus')}:** " + 
                                 _('climate_focus_insight', climate_val=avg_climate_curr, target_thresh=clim_good_thresh))
    
    if participation_curr is not None:
        part_good_thresh = config.ENGAGEMENT_PARTICIPATION["good"]
        if participation_curr < part_good_thresh: # Only a warning if below target
            insights_list.append(f"📝 **{_('participation_low_warn')}:** " +
                                 _('participation_low_insight', part_val=participation_curr, target_thresh=part_good_thresh))

    if not insights_list and any(pd.notna(v) for v in [avg_enps_curr, avg_climate_curr, participation_curr]):
        insights_list.append(_("engagement_no_critical_insights"))
    elif not insights_list:
        insights_list.append(_("no_data_for_engagement_insights", default_text="Insufficient data for engagement insights."))

    return insights_list

# --- Stress Insights ---
def generate_stress_insights(avg_stress_val_in: Optional[float], df_stress_trend_data: Optional[pd.DataFrame], lang: str) -> List[str]:
    insights_list = []
    _ = lambda k, default=None, **kw: _get_insight_text(lang, k, default, **kw)
    
    current_avg_stress = float(avg_stress_val_in) if pd.notna(avg_stress_val_in) else None
    stress_low_threshold = config.STRESS_LEVEL_PSYCHOSOCIAL["low"]   # Below this is "low"
    stress_medium_threshold = config.STRESS_LEVEL_PSYCHOSOCIAL["medium"] # Below this is "moderate" (if above "low")

    if current_avg_stress is not None:
        if current_avg_stress > stress_medium_threshold: # High stress
            insights_list.append(f"🥵 **{_('stress_high_alert')}:** " + _('stress_high_insight', stress_val=current_avg_stress))
        elif current_avg_stress > stress_low_threshold: # Moderate stress
             insights_list.append(f"😟 **{_('stress_moderate_warn')}:** " + _('stress_moderate_insight', stress_val=current_avg_stress))
        else: # Low stress
             insights_list.append(f"😌 **{_('stress_low_status')}:** " + _('stress_low_insight', stress_val=current_avg_stress))
    
    # Trend analysis if data is available
    workload_agg_col = "Workload_Avg" 
    psych_signals_agg_col = "Psych_Signals_Avg"

    if df_stress_trend_data is not None and not df_stress_trend_data.empty and \
       workload_agg_col in df_stress_trend_data.columns and psych_signals_agg_col in df_stress_trend_data.columns and \
       len(df_stress_trend_data) >= 2: # Need at least 2 points for a trend
        
        # Ensure columns are numeric for diff()
        if pd.api.types.is_numeric_dtype(df_stress_trend_data[workload_agg_col]) and \
           pd.api.types.is_numeric_dtype(df_stress_trend_data[psych_signals_agg_col]):
            
            # Look at the last two available (aggregated) periods for simple trend
            trend_data_for_diff = df_stress_trend_data[[workload_agg_col, psych_signals_agg_col]].dropna().tail(2)
            if len(trend_data_for_diff) == 2:
                workload_diff = trend_data_for_diff[workload_agg_col].diff().iloc[-1]
                psych_diff = trend_data_for_diff[psych_signals_agg_col].diff().iloc[-1]

                # Check for significant adverse trend: workload increasing AND psych signals decreasing
                # Define what "significant" means (e.g., change > 0.2 points on scale)
                if pd.notna(workload_diff) and pd.notna(psych_diff):
                    # Example thresholds for a "concerning trend"
                    workload_increase_threshold = 0.2 
                    psych_decrease_threshold = -0.2 
                    if workload_diff > workload_increase_threshold and psych_diff < psych_decrease_threshold:
                        insights_list.append(f"📈📉 **{_('stress_trend_warn')}:** " + _('stress_trend_insight'))
    
    if not insights_list and current_avg_stress is not None :
        insights_list.append(_('no_insights_generated_stress', default_text=_('no_insights_generated')))
    elif not insights_list:
        insights_list.append(_("no_data_for_stress_insights", default_text="Insufficient data for stress insights."))

    return insights_list
