"""
Title taxonomy and alignment scoring for the Senior AI Engineer role.
"""
HIGH_VALUE_TITLES = {
    "search engineer", "ranking engineer", "recommendation engineer",
    "retrieval engineer", "applied scientist", "ml engineer",
    "relevance engineer", "discovery engineer", "machine learning engineer",
    "search and ranking", "recommendation system", "information retrieval"
}

def clean_title(title_str):
    if not title_str:
        return ""
    return title_str.strip().lower()

def compute_title_alignment(candidate):
    """
    Evaluates current, historical, and overall title alignment.
    Returns scores in [0.0, 1.0].
    """
    p = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    
    current_title = clean_title(p.get("current_title", ""))
    
    # 1. Current Title Alignment
    current_alignment = 0.0
    if any(hvt in current_title for hvt in HIGH_VALUE_TITLES):
        current_alignment = 1.0
    elif "software engineer" in current_title or "backend engineer" in current_title or "data engineer" in current_title:
        current_alignment = 0.5
        
    # 2. Historical Title Alignment
    matching_roles_count = 0
    total_roles = len(career)
    
    for role in career:
        title = clean_title(role.get("title", ""))
        if any(hvt in title for hvt in HIGH_VALUE_TITLES):
            matching_roles_count += 1
            
    historical_alignment = 0.0
    if total_roles > 0:
        historical_alignment = matching_roles_count / total_roles
        
    # 3. Overall Career Title Match
    ever_high_value = 0.0
    if current_alignment == 1.0 or matching_roles_count > 0:
        ever_high_value = 1.0
        
    return {
        "current_title_alignment": float(current_alignment),
        "historical_title_alignment": float(historical_alignment),
        "career_title_alignment": float(ever_high_value)
    }
