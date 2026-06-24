"""
Explicit honeypot detection rules based on structural and historical profile consistency.
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

def check_is_honeypot(candidate):
    """
    Checks if a candidate profile is a honeypot (contains impossible contradictions).
    Returns (is_honeypot, reasons).
    """
    reasons = []
    
    p = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    yoe = p.get("years_of_experience", 0.0)
    
    # Rule 1: Expert Zero-duration skills (expert proficiency claimed with 0 duration)
    expert_zero = [s for s in skills if s.get("proficiency") == "expert" and s.get("duration_months") == 0]
    if len(expert_zero) >= 5:
        reasons.append(f"Expert Zero: claimed expert proficiency in {len(expert_zero)} skills with 0 duration months")
        
    # Rule 2: Career role Date vs Duration mismatch
    for idx, r in enumerate(career):
        sd = r.get("start_date")
        ed = r.get("end_date")
        dm = r.get("duration_months", 0)
        s_dt = parse_date(sd)
        
        if s_dt:
            # If current role, end date is June 20, 2026
            e_dt = CURRENT_DATE if r.get("is_current") else parse_date(ed)
            
            if e_dt:
                calc_dm = (e_dt.year - s_dt.year) * 12 + (e_dt.month - s_dt.month)
                # Flag if reported duration deviates from dates by more than 2 months
                if abs(calc_dm - dm) > 2:
                    reasons.append(
                        f"Career Date Mismatch: role at {r.get('company')} reported duration_months as {dm} "
                        f"but dates are {sd} to {ed or 'current'} (calculated {calc_dm} months)"
                    )
                    
    # Rule 3: Years of Experience vs Career sum duration mismatch
    total_career_months = sum(r.get("duration_months", 0) for r in career)
    total_career_years = total_career_months / 12.0
    if len(career) > 0 and abs(yoe - total_career_years) > 3.0:
        reasons.append(
            f"YOE Mismatch: reported years of experience is {yoe} "
            f"but sum of role durations is {total_career_years:.2f} years ({total_career_months} months)"
        )
        
    is_honeypot = len(reasons) > 0
    return is_honeypot, reasons
