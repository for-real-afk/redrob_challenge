"""
Career trajectory, consistency, company type, and seniority signals.
"""
from src.jd_parser import (
    IT_SERVICES_COMPANIES, PRODUCT_COMPANIES,
    NLP_IR_KEYWORDS, CV_SPEECH_ROBOTICS_KEYWORDS,
    TARGET_YOE_MIN, TARGET_YOE_MAX, TARGET_YOE_CENTER
)

def compute_career_score(candidate):
    """
    Computes career trajectory and fit signals.
    Returns a score and a dict of sub-features.
    """
    p = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    
    current_title = p.get("current_title", "").strip().lower()
    headline = p.get("headline", "").strip().lower()
    yoe = p.get("years_of_experience", 0.0)
    
    # 1. Title vs Substance Mismatch Check
    # Flag if current role is non-technical/non-AI (e.g., Marketing, Accountant, Operations)
    mismatch_keywords = {"marketing", "operations", "accountant", "customer support", "sales", "hr", "graphic designer", "recruiter", "finance"}
    is_mismatch_title = False
    for kw in mismatch_keywords:
        if kw in current_title or kw in headline:
            is_mismatch_title = True
            break
            
    # We also check if their skills list contains deep AI terms like "embeddings" or "fine-tuning llms"
    # mismatch penalty is applied in the combination step if this is true and AI skills are present.
    
    # 2. Company Type Signal (IT Services vs Product company)
    # Entire career in services -> heavy penalty.
    # Current services but prior product -> no penalty.
    all_companies = [r.get("company", "").strip().lower() for r in career]
    current_company = p.get("current_company", "").strip().lower()
    
    all_services = True
    has_product = False
    
    for r in career:
        comp = r.get("company", "").strip().lower()
        desc = r.get("description", "").strip().lower()
        ind = r.get("industry", "").strip().lower()
        
        # Check if company is in known IT services list
        is_services_firm = comp in IT_SERVICES_COMPANIES or "it services" in ind or "consulting" in ind
        
        # Check if company is in product company list or mentions product/startup keywords
        is_product_firm = comp in PRODUCT_COMPANIES or "product company" in desc or "startup" in desc
        
        if is_product_firm:
            has_product = True
            all_services = False
        elif not is_services_firm:
            # If it's a generic company not in IT services, treat it as neutral/product
            all_services = False
            
    # If they are currently at a services firm, but have a product company in history
    current_is_services = current_company in IT_SERVICES_COMPANIES
    services_penalty = 0.0
    if all_services and len(career) > 0:
        services_penalty = -0.4
    elif current_is_services and has_product:
        # Currently at services but has product experience - mild or no penalty
        services_penalty = -0.05
        
    # 3. Seniority Fit
    # Target band is 5-9 years. Soft distance scoring.
    if TARGET_YOE_MIN <= yoe <= TARGET_YOE_MAX:
        seniority_score = 1.0
    elif yoe < TARGET_YOE_MIN:
        # Discount if below 5 YOE
        seniority_score = max(0.1, 1.0 - (TARGET_YOE_MIN - yoe) * 0.2)
    else:
        # Discount if above 9 YOE (milder, since they are still experienced)
        seniority_score = max(0.3, 1.0 - (yoe - TARGET_YOE_MAX) * 0.08)
        
    # 4. Title-chasing company hoppers
    # Switches companies in < 18 months average
    n_roles = len(career)
    avg_duration = 0.0
    is_title_chaser = False
    
    if n_roles >= 3:
        total_duration = sum(r.get("duration_months", 0) for r in career)
        avg_duration = total_duration / n_roles
        
        # Check distinct companies to verify company hopping (not internal promotions)
        distinct_companies = len(set(all_companies))
        if avg_duration < 18.0 and distinct_companies >= 3:
            is_title_chaser = True
            
    title_chaser_penalty = -0.2 if is_title_chaser else 0.0
    
    # 5. External validation signal (GitHub)
    github_score = signals.get("github_activity_score", -1)
    github_boost = 0.0
    if github_score > 0:
        github_boost = (github_score / 100.0) * 0.1
        
    # 6. Domain exposure check (CV/Speech/Robotics only with zero NLP/IR)
    skill_names = [s.get("name", "").strip().lower() for s in skills]
    career_descs = [r.get("description", "").strip().lower() for r in career]
    all_text = " ".join([p.get("summary", "").strip().lower()] + career_descs + skill_names)
    
    cv_matches = sum(1 for kw in CV_SPEECH_ROBOTICS_KEYWORDS if kw in all_text)
    nlp_matches = sum(1 for kw in NLP_IR_KEYWORDS if kw in all_text)
    
    is_pure_cv_speech = (cv_matches > 2) and (nlp_matches == 0)
    domain_penalty = -0.35 if is_pure_cv_speech else 0.0
    
    # Career score: baseline is 1.0, modified by boosts and penalties
    base_score = 1.0
    combined_score = base_score + services_penalty + title_chaser_penalty + github_boost + domain_penalty
    # Bound between 0 and 1.5
    combined_score = max(0.0, min(1.5, combined_score))
    
    return {
        "career_score": float(combined_score),
        "is_mismatch_title": is_mismatch_title,
        "all_services": all_services,
        "services_penalty": float(services_penalty),
        "seniority_score": float(seniority_score),
        "is_title_chaser": is_title_chaser,
        "title_chaser_penalty": float(title_chaser_penalty),
        "github_boost": float(github_boost),
        "is_pure_cv_speech": is_pure_cv_speech,
        "domain_penalty": float(domain_penalty),
        "years_of_experience": float(yoe)
    }
