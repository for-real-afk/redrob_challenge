"""
Lexical feature matching (skills, keywords, and assessment scores).
"""
import re
from src.jd_parser import REQUIRED_SKILLS, PREFERRED_SKILLS

def compute_lexical_score(candidate):
    """
    Computes a lexical match score between the candidate and the job description.
    
    Formula components:
    1. Skill array matches weighted by proficiency and duration.
    2. Verified skill assessment scores.
    3. Keyword scanning over profile summary and career descriptions.
    """
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    assessment_scores = signals.get("skill_assessment_scores", {})
    
    # 1. Score based on the skills array
    skills_score = 0.0
    matched_skills = set()
    
    proficiency_map = {
        "expert": 1.0,
        "advanced": 0.8,
        "intermediate": 0.5,
        "beginner": 0.2
    }
    
    for s in skills:
        name = s.get("name", "").strip().lower()
        if not name:
            continue
            
        proficiency = s.get("proficiency", "intermediate").lower()
        prof_val = proficiency_map.get(proficiency, 0.5)
        
        # Duration weight: log/linear scale to discount thin keyword entries
        duration = s.get("duration_months", 0)
        # min 0.2, max 1.0 (reaches 1.0 at 36 months of use)
        dur_val = 0.2 + 0.8 * (min(duration, 36) / 36.0)
        
        # Base weight based on JD importance
        base_val = 0.0
        if name in REQUIRED_SKILLS:
            base_val = 1.0
            matched_skills.add(name)
        elif name in PREFERRED_SKILLS:
            base_val = 0.5
            matched_skills.add(name)
            
        skill_contrib = base_val * prof_val * dur_val
        
        # Incorporate skill assessment scores if present
        # If the candidate completed a test for this skill, boost it
        # Try exact match or suffix match
        assess_score = None
        for k, val in assessment_scores.items():
            if k.lower() == name or name in k.lower():
                assess_score = val
                break
                
        if assess_score is not None:
            # Verified test score (0-100) provides a substantial boost
            skill_contrib += (assess_score / 100.0) * 0.4
            
        skills_score += skill_contrib

    # 2. Text keyword scanning (for plain-language builders who don't list keywords as formal skills)
    summary = candidate.get("profile", {}).get("summary", "")
    career = candidate.get("career_history", [])
    descriptions = [r.get("description", "") for r in career]
    all_text = " ".join([summary] + descriptions).lower()
    
    text_bonus = 0.0
    text_matches = set()
    
    for word in REQUIRED_SKILLS:
        # Avoid double-counting if already in skills array, but give partial credit
        # Use regex to find word boundaries
        if re.search(r'\b' + re.escape(word) + r'\b', all_text):
            text_matches.add(word)
            if word not in matched_skills:
                text_bonus += 0.15
            else:
                text_bonus += 0.03
                
    for word in PREFERRED_SKILLS:
        if re.search(r'\b' + re.escape(word) + r'\b', all_text):
            text_matches.add(word)
            if word not in matched_skills:
                text_bonus += 0.07
            else:
                text_bonus += 0.015
                
    # Cap the text scan bonus so it doesn't inflate the score infinitely
    text_bonus = min(text_bonus, 1.5)
    
    total_score = skills_score + text_bonus
    
    # Return features dictionary for explainability
    return {
        "lexical_score": float(total_score),
        "matched_skills_count": len(matched_skills),
        "text_matches_count": len(text_matches),
        "skills_score": float(skills_score),
        "text_bonus": float(text_bonus)
    }
