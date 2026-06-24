"""
Behavioral signals from redrob_signals (availability, notice period, trust).
"""
from datetime import datetime

# Fixed current date representing the dataset snapshot context (June 20, 2026)
CURRENT_DATE = datetime(2026, 6, 20)

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        return None

def compute_behavioral_score(candidate):
    """
    Computes a behavioral multiplier and trust flags from redrob_signals.
    
    Formula components:
    1. Reachability multiplier (open to work, last active, recruiter response rate).
    2. Notice period penalty curve.
    3. Interview and offer follow-through boosts.
    4. Small trust boosts (verified phone, email, linkedin).
    """
    signals = candidate.get("redrob_signals", {})
    p = candidate.get("profile", {})
    yoe = p.get("years_of_experience", 0.0)
    
    # 1. Reachability Multiplier
    # Baseline multiplier is 1.0
    reachability = 1.0
    
    # Open to work flag
    if signals.get("open_to_work_flag", False):
        reachability *= 1.15
        
    # Recruiter response rate (fraction)
    resp_rate = signals.get("recruiter_response_rate", -1.0)
    if resp_rate >= 0.0:
        # Scale: 0.0 response rate -> 0.8x, 1.0 response rate -> 1.0x
        reachability *= (0.8 + 0.2 * resp_rate)
    else:
        # Default for missing signal
        reachability *= 0.9
        
    # Avg response time in hours
    resp_time = signals.get("avg_response_time_hours", -1.0)
    if 0.0 <= resp_time <= 12.0:
        reachability *= 1.05
    elif resp_time > 72.0:
        reachability *= 0.9
        
    # Active recency
    last_active_str = signals.get("last_active_date")
    last_active_dt = parse_date(last_active_str)
    
    days_since_active = 365 # Default if missing
    if last_active_dt:
        days_since_active = (CURRENT_DATE - last_active_dt).days
        
    if days_since_active <= 30:
        reachability *= 1.1
    elif days_since_active > 180:
        reachability *= 0.75 # Down-weight inactive
    elif days_since_active > 90:
        reachability *= 0.9
        
    # 2. Notice Period Penalty Curve
    # JD wants sub-30 day notice. 30+ bar gets higher.
    notice_days = signals.get("notice_period_days", 0)
    notice_mult = 1.0
    if notice_days <= 30:
        notice_mult = 1.0
    elif notice_days <= 90:
        # Linear drop from 1.0 to 0.85
        notice_mult = 1.0 - ((notice_days - 30) / 60.0) * 0.15
    else:
        # Linear drop from 0.85 to 0.60 (up to 180 days)
        notice_mult = 0.85 - (min(notice_days, 180) - 90) / 90.0 * 0.25
        
    # 3. Follow-through Signals
    follow_through_boost = 0.0
    int_rate = signals.get("interview_completion_rate", -1.0)
    if int_rate >= 0.0:
        follow_through_boost += int_rate * 0.05
        
    offer_rate = signals.get("offer_acceptance_rate", -1.0)
    if offer_rate >= 0.0:
        follow_through_boost += offer_rate * 0.05
        
    # 4. Small Trust boosts
    trust_boost = 0.0
    if signals.get("verified_email", False):
        trust_boost += 0.01
    if signals.get("verified_phone", False):
        trust_boost += 0.01
    if signals.get("linkedin_connected", False):
        trust_boost += 0.01
        
    # Combined behavioral multiplier
    behavior_multiplier = reachability * notice_mult
    
    # expected_salary check: YOE is high but salary expectations are extremely low (could be outlier/suspicious)
    sal = signals.get("expected_salary_range_inr_lpa", {})
    s_min = sal.get("min", 0.0)
    s_max = sal.get("max", 0.0)
    salary_anomaly = False
    if yoe >= 5.0 and s_max > 0.0 and s_max < 2.0:
        salary_anomaly = True
        behavior_multiplier *= 0.2 # Drastic down-weight (likely honeypot or garbage entry)

    return {
        "behavior_multiplier": float(behavior_multiplier),
        "reachability_mult": float(reachability),
        "notice_mult": float(notice_mult),
        "follow_through_boost": float(follow_through_boost),
        "trust_boost": float(trust_boost),
        "days_since_active": int(days_since_active),
        "salary_anomaly": salary_anomaly
    }
