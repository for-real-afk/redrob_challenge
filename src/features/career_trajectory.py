"""
Career trajectory analysis: consistency, leadership growth, seniority velocity, and domain continuity.
"""

LEADERSHIP_TITLES = {"lead", "manager", "architect", "principal", "cto", "head of", "director", "vp"}
TECH_ROLE_KEYWORDS = {"engineer", "scientist", "developer", "programmer", "analyst", "architect"}
DOMAIN_KEYWORDS = {"search", "ranking", "recommendation", "retrieval", "matching", "personalization", "marketplace ml", "nlp", "information retrieval"}

def compute_career_trajectory(candidate):
    """
    Computes trajectory features from candidate career history.
    """
    career = candidate.get("career_history", [])
    
    if not career:
        return {
            "trajectory_score": 0.5,
            "seniority_growth": 0.5,
            "leadership_growth": 0.0,
            "career_consistency": 0.5,
            "domain_continuity": 0.0
        }
        
    # 1. Leadership Growth
    has_leadership = 0.0
    for role in career:
        title = role.get("title", "").strip().lower()
        if any(lt in title for lt in LEADERSHIP_TITLES):
            has_leadership = 1.0
            break
            
    # 2. Career Consistency
    tech_roles = 0
    total_roles = len(career)
    for role in career:
        title = role.get("title", "").strip().lower()
        if any(tr in title for tr in TECH_ROLE_KEYWORDS):
            tech_roles += 1
            
    career_consistency = tech_roles / total_roles
    
    # 3. Domain Continuity (retrieval/search focus)
    domain_roles = 0
    for role in career:
        desc = role.get("description", "").strip().lower()
        title = role.get("title", "").strip().lower()
        full_text = f"{title} {desc}"
        if any(dkw in full_text for dkw in DOMAIN_KEYWORDS):
            domain_roles += 1
            
    domain_continuity = domain_roles / total_roles
    
    # 4. Promotion Velocity (Promotion Velocity)
    # Shorter average tenure (18-36 months) represents good velocity.
    total_months = sum(role.get("duration_months", 0) for role in career)
    avg_months_per_role = total_months / total_roles if total_roles > 0 else 0.0
    
    # Optimal promotion velocity is 1.5 to 3 years per role (18 to 36 months)
    promotion_velocity = 0.5
    if 18.0 <= avg_months_per_role <= 36.0:
        promotion_velocity = 1.0
    elif 12.0 <= avg_months_per_role < 18.0:
        promotion_velocity = 0.75
    elif 36.0 < avg_months_per_role <= 60.0:
        promotion_velocity = 0.6
        
    # Calculate composite trajectory score
    trajectory_score = 0.3 * career_consistency + 0.4 * domain_continuity + 0.2 * promotion_velocity + 0.1 * has_leadership
    
    return {
        "trajectory_score": float(trajectory_score),
        "promotion_velocity": float(promotion_velocity),
        "leadership_growth": float(has_leadership),
        "career_consistency": float(career_consistency),
        "domain_continuity": float(domain_continuity)
    }
