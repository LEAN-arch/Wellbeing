# insights.py
import pandas as pd
import config
from typing import List, Dict, Optional, Any, Union

# Localized text helper
def _get_insight_text(lang_code: str, key: str, default_text: Optional[str] = None, **kwargs) -> str:
    base_text = config.TEXT_STRINGS.get(lang_code, config.TEXT_STRINGS[config.DEFAULT_LANG]).get(key, default_text or key)
    try:
        return base_text.format(**kwargs) # Allow for dynamic values in insights
    except KeyError: # If a format key is missing
        return base_text # Return base text, might show {placeholder}


def generate_stability_insights(df_stability_filtered: pd.DataFrame, avg_rotation_val: Optional[float],
                                hires_exits_trend_df: Optional[pd.DataFrame], lang: str) -> List[str]:
    insights_list = []
    _ = lambda k, default=None, **kw: _get_insight_text(lang, k, default, **kw)

    # Rotation Insights with Target
    rot_good = config.STABILITY_ROTATION_RATE["good"]
    rot_warn = config.STABILITY_ROTATION_RATE["warning"]
    rot_target = config.STABILITY_ROTATION_RATE["target"]
    if pd.notna(avg_rotation_val):
        if avg_rotation_val > rot_warn:
            insights_list.append(f"🚨 **{_('rotation_high_alert')}:** " +
                                 _('rotation_high_insight_v2',
                                   rotation_val=avg_rotation_val, warn_thresh=rot_warn, target_thresh=rot_target,
                                   default_text="Rotation ({rotation_val:.1f}%) significantly exceeds warning ({warn_thresh}%). Target is {target_thresh}%. Prioritize root cause analysis for high-turnover areas (filter dashboard) and review exit interview data. Implement targeted retention programs for critical roles."))
        elif avg_rotation_val > rot_target: # Above target but not critical warning yet
            insights_list.append(f"⚠️ **{_('rotation_moderate_warn')}:** " +
                                 _('rotation_moderate_insight_v2',
                                   rotation_val=avg_rotation_val, target_thresh=rot_target,
                                   default_text="Rotation ({rotation_val:.1f}%) is above the target of {target_thresh}%. While not critical, focus on proactive retention for at-risk groups and monitor new hire stability."))
        else:
            insights_list.append(f"✅ **{_('rotation_good_status')}:** " +
                                 _('rotation_good_insight_v2',
                                   rotation_val=avg_rotation_val, target_thresh=rot_target,
                                   default_text="Rotation rate ({rotation_val:.1f}%) meets or is below target ({target_thresh}%). Maintain strong engagement and onboarding practices."))
    else:
        insights_list.append(_('rotation_no_data'))

    # Retention Insights
    ret12_col = config.COLUMN_MAP.get("retention_12m")
    if ret12_col and ret12_col in df_stability_filtered.columns:
        avg_ret12 = df_stability_filtered[ret12_col].mean()
        ret_good = config.STABILITY_RETENTION["good"]
        ret_warn = config.STABILITY_RETENTION["warning"]
        if pd.notna(avg_ret12):
            if avg_ret12 < ret_warn:
                insights_list.append(f"📉 **{_('retention_low_alert')}:** " +
                                     _('retention_low_insight_v2',
                                       retention_val=avg_ret12, warn_thresh=ret_warn, target_thresh=ret_good,
                                       default_text="12-month retention ({retention_val:.1f}%) is critically low (below {warn_thresh}%). Target is {target_thresh}%. Investigate early attrition causes: onboarding effectiveness, manager support, role clarity, and growth opportunities within the first year."))
            elif avg_ret12 < ret_good:
                insights_list.append(f"📈 **{_('retention_needs_improvement')}:** " +
                                     _('retention_improvement_insight_v2',
                                       retention_val=avg_ret12, target_thresh=ret_good,
                                       default_text="12-month retention ({retention_val:.1f}%) is below target ({target_thresh}%). Identify and address factors impacting new hire experience and commitment."))
            else:
                insights_list.append(f"👍 **{_('retention_good_status', 'Good 12m Retention')}:** " +
                                      _('retention_good_insight_detail', retention_val=avg_ret12, target_thresh=ret_good,
                                        default_text="12-month retention ({retention_val:.1f}%) meets/exceeds target. Current strategies are effective."))


    # Hires vs Exits Trend (More detailed insight based on net change)
    if hires_exits_trend_df is not None and not hires_exits_trend_df.empty and \
       'Hires_Total' in hires_exits_trend_df.columns and 'Exits_Total' in hires_exits_trend_df.columns:
        hires_exits_trend_df['NetChange'] = hires_exits_trend_df['Hires_Total'] - hires_exits_trend_df['Exits_Total']
        avg_net_change = hires_exits_trend_df['NetChange'].mean()
        recent_net_change = hires_exits_trend_df['NetChange'].iloc[-1] if len(hires_exits_trend_df) > 0 else 0
        
        if avg_net_change < 0:
            insights_list.append(f"📉 **{_('net_loss_trend_warn', 'Net Employee Loss Trend')}:** " +
                                 _('net_loss_insight', avg_change=avg_net_change,
                                   default_text="Averaging a net loss of {avg_change:.1f} employees per period. This indicates potential challenges in workforce replenishment or retention. Review talent acquisition strategies and drivers of attrition."))
        if recent_net_change < -5 : # Arbitrary threshold for "significant recent loss"
            insights_list.append(f"🔻 **{_('recent_loss_focus', 'Focus on Recent Departures')}:** "+
                                 _('recent_loss_insight_detail', recent_change=recent_net_change,
                                 default_text="Recent period shows a significant net loss of {recent_change:.0f} employees. Address immediate concerns contributing to departures."))


    if not insights_list: insights_list.append(_("no_insights_generated"))
    return insights_list

def generate_safety_insights(df_safety_filtered: pd.DataFrame, current_dwa: Optional[float], total_incidents_this_period: Optional[float], lang: str) -> List[str]:
    insights_list = []
    _ = lambda k, default=None, **kw: _get_insight_text(lang, k, default, **kw)
    
    inc_good = config.SAFETY_INCIDENTS["good"]
    inc_warn = config.SAFETY_INCIDENTS["warning"]
    inc_target = config.SAFETY_INCIDENTS["target"]

    if pd.notna(total_incidents_this_period):
        if total_incidents_this_period > inc_warn:
            insights_list.append(f"🚨 **{_('high_incidents_alert')}:** " +
                                 _('high_incidents_insight_v2', count=total_incidents_this_period, warn_thresh=inc_warn,
                                   default_text="{count:.0f} incidents this period (above warning {warn_thresh}). Critical review of high-risk tasks/areas needed. Implement immediate corrective actions and review safety training effectiveness."))
        elif total_incidents_this_period > inc_target: # Above target (0) but not critical warning
            insights_list.append(f"⚠️ **{_('moderate_incidents_warn')}:** " +
                                 _('moderate_incidents_insight_v2', count=total_incidents_this_period, target_thresh=inc_target,
                                   default_text="{count:.0f} incidents this period (target is {target_thresh}). Focus on proactive measures: near-miss analysis, safety walks, and toolbox talks."))
        else:
            insights_list.append(f"✅ **{_('low_incidents_status')}:** " +
                                 _('low_incidents_insight_v2', count=total_incidents_this_period,
                                   default_text="{count:.0f} incidents. Excellent! Reinforce positive safety behaviors and maintain high standards of hazard identification."))
    
    dwa_good = config.SAFETY_DAYS_NO_INCIDENTS["good"]
    dwa_warn = config.SAFETY_DAYS_NO_INCIDENTS["warning"]
    if pd.notna(current_dwa):
        if current_dwa < dwa_warn:
            insights_list.append(f"⏱️ **{_('dwa_low_warn')}:** " +
                                 _('dwa_low_insight_v2', days=current_dwa, warn_thresh=dwa_warn,
                                   default_text="{days:.0f} days without incidents is below warning level of {warn_thresh}. Review recent incident patterns and refresh safety focus in relevant areas."))
        elif current_dwa < dwa_good:
            insights_list.append(f"⏳ **{_('dwa_needs_improvement','Days Without Incidents: Monitor')}:** " +
                                 _('dwa_improvement_insight', days=current_dwa, target_thresh=dwa_good,
                                   default_text="{days:.0f} days without incidents. Aim to increase this towards the target of {target_thresh}+ days. Celebrate milestones reached."))

    if not insights_list: insights_list.append(_("no_insights_generated"))
    return insights_list

def generate_engagement_insights(avg_enps_val: Optional[float], avg_climate_val: Optional[float], participation_val: Optional[float], lang: str) -> List[str]:
    insights_list = []
    _ = lambda k, default=None, **kw: _get_insight_text(lang, k, default, **kw)

    enps_good = config.ENGAGEMENT_ENPS["good"]
    enps_warn = config.ENGAGEMENT_ENPS["warning"]
    if pd.notna(avg_enps_val):
        if avg_enps_val < enps_warn:
            insights_list.append(f"📉 **{_('enps_low_alert')}:** " + _('enps_low_insight', enps_val=avg_enps_val))
        elif avg_enps_val < enps_good:
            insights_list.append(f"🤔 **{_('enps_needs_focus')}:** " + _('enps_focus_insight', enps_val=avg_enps_val))
        else: insights_list.append(f"👍 **{_('enps_good_status')}:** " + _('enps_good_insight', enps_val=avg_enps_val))
    
    clim_good = config.ENGAGEMENT_CLIMATE_SCORE["good"]
    clim_warn = config.ENGAGEMENT_CLIMATE_SCORE["warning"]
    if pd.notna(avg_climate_val):
        if avg_climate_val < clim_warn:
            insights_list.append(f"🚩 **{_('climate_low_alert')}:** " + _('climate_low_insight', climate_val=avg_climate_val))
        elif avg_climate_val < clim_good:
            insights_list.append(f"📊 **{_('climate_needs_focus','Climate Score: Needs Attention')}:** " + 
                                 _('climate_focus_insight', climate_val=avg_climate_val, target_thresh=clim_good,
                                   default_text="Work Climate ({climate_val:.1f}) is below target ({target_thresh}). Address areas identified in survey feedback."))
    
    part_good = config.ENGAGEMENT_PARTICIPATION["good"]
    if pd.notna(participation_val) and participation_val < part_good:
        insights_list.append(f"📝 **{_('participation_low_warn','Low Survey Participation')}:** " +
                             _('participation_low_insight', part_val=participation_val, target_thresh=part_good,
                               default_text="Survey participation ({part_val:.0f}%) is below target ({target_thresh}%). Understand barriers to participation for more representative data."))


    if not insights_list: insights_list.append(_("engagement_no_critical_insights"))
    return insights_list

def generate_stress_insights(avg_stress_val: Optional[float], df_stress_trend_data: pd.DataFrame, lang: str) -> List[str]:
    insights_list = []
    _ = lambda k, default=None, **kw: _get_insight_text(lang, k, default, **kw)
    
    stress_low_thresh = config.STRESS_LEVEL_PSYCHOSOCIAL["low"]
    stress_med_thresh = config.STRESS_LEVEL_PSYCHOSOCIAL["medium"]

    if pd.notna(avg_stress_val):
        if avg_stress_val > stress_med_thresh:
            insights_list.append(f"🥵 **{_('stress_high_alert')}:** " + _('stress_high_insight', stress_val=avg_stress_val))
        elif avg_stress_val > stress_low_thresh:
             insights_list.append(f"😟 **{_('stress_moderate_warn')}:** " + _('stress_moderate_insight', stress_val=avg_stress_val))
        else:
             insights_list.append(f"😌 **{_('stress_low_status')}:** " + _('stress_low_insight', stress_val=avg_stress_val))
    
    # Trend Insight (Simplified: checks if Workload Avg went up AND Psych Signals Avg went down recently)
    # Requires the aggregated column names used in app.py: "Workload_Avg", "Psych_Signals_Avg"
    workload_trend_col = "Workload_Avg" 
    psych_trend_col = "Psych_Signals_Avg"

    if not df_stress_trend_data.empty and workload_trend_col in df_stress_trend_data.columns and \
       psych_trend_col in df_stress_trend_data.columns and len(df_stress_trend_data) >= 2:
        
        # Ensure columns are numeric before diff
        if pd.api.types.is_numeric_dtype(df_stress_trend_data[workload_trend_col]) and \
           pd.api.types.is_numeric_dtype(df_stress_trend_data[psych_trend_col]):
            
            workload_change = df_stress_trend_data[workload_trend_col].diff().iloc[-1]
            psych_change = df_stress_trend_data[psych_trend_col].diff().iloc[-1]

            if pd.notna(workload_change) and pd.notna(psych_change):
                if workload_change > 0.1 and psych_change < -0.1: # Arbitrary threshold for "significant" change
                    insights_list.append(f"📈 **{_('stress_trend_warn')}:** " + _('stress_trend_insight'))
    
    if not insights_list: insights_list.append(_("no_insights_generated"))
    return insights_list