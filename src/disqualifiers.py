"""
Job Description disqualifiers, fake title inflation checks, and soft penalties.
Each function returns (is_triggered, weight_or_penalty, reason).
"""
import re
from datetime import datetime
from src.jd_parser import IT_SERVICES_COMPANIES, NLP_IR_KEYWORDS, CV_SPEECH_ROBOTICS_KEYWORDS

CURRENT_DATE = datetime(2026, 6, 20)

def parse_date(date_str):
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except:
        return None

def check_timeline_overlap(candidate):
    """
    Hard Fail: check if two full-time roles overlap by more than 3 months.
    """
    career = candidate.get("career_history", [])
    if len(career) < 2:
        return False, 1.0, ""
        
    intervals = []
    for role in career:
        sd = parse_date(role.get("start_date"))
        if not sd:
            continue
        ed = CURRENT_DATE if role.get("is_current") else parse_date(role.get("end_date"))
        if not ed:
            continue
        intervals.append((sd, ed, role.get("company", "Company")))
        
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            s1, e1, c1 = intervals[i]
            s2, e2, c2 = intervals[j]
            
            # Find intersection duration in days
            latest_start = max(s1, s2)
            earliest_end = min(e1, e2)
            overlap_days = (earliest_end - latest_start).days
            
            if overlap_days > 90:  # More than 3 months overlap
                return True, 0.0, f"Overlapping career timelines: {c1} and {c2} overlap by {overlap_days // 30} months"
                
    return False, 1.0, ""

def compute_inflation_confidence(candidate):
    """
    Computes fake title inflation confidence score.
    """
    p = candidate.get("profile", {})
    yoe = p.get("years_of_experience", 0.0)
    current_title = p.get("current_title", "").strip().lower()
    current_company_size = p.get("current_company_size", "")
    
    inflation_keywords = {"principal", "cto", "vp", "director", "architect"}
    has_inflation_title = any(kw in current_title for kw in inflation_keywords)
    
    if has_inflation_title and yoe < 2.0:
        # Check exceptions: Startup CTO, Founding Engineer
        is_founding_or_startup = "founding" in current_title or "startup" in current_title
        company_is_small = current_company_size in {"1-10", "11-50", "51-200"}
        
        if is_founding_or_startup or company_is_small:
            # Legitimate early-career leader
            return 0.40
        else:
            # High confidence fake title inflation
            return 0.98
            
    return 0.0

def check_fake_title_inflation(candidate):
    """
    Disqualifier: Hard fail if inflation confidence > 0.95. Else soft penalty.
    """
    confidence = compute_inflation_confidence(candidate)
    if confidence > 0.95:
        return True, 0.0, "Fake title inflation: Principal/VP/Director role with < 2 years total experience"
    elif confidence > 0.0:
        return True, 0.7, "Suspected title inflation (soft penalty): Lead/CTO title with low overall YOE"
    return False, 1.0, ""

def check_pure_research(candidate):
    """
    Soft Penalty: Pure research background with zero production deployment.
    """
    p = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    summary = p.get("summary", "").lower()
    
    career_texts = [r.get("description", "").lower() for r in career]
    all_text = " ".join([summary] + career_texts)
    
    research_words = {"phd", "postdoc", "researcher", "academic", "publications", "journals", "scientific", "laboratory", "fellow"}
    production_words = {"production", "shipped", "deployed", "scaled", "infrastructure", "software engineer", "pipeline", "cloud", "aws", "kubernetes", "fastapi"}
    
    has_research = any(w in all_text for w in research_words)
    has_production = any(w in all_text for w in production_words)
    
    if has_research and not has_production and len(career) <= 2:
        return True, 0.5, "Pure academic/research background with no production deployment experience"
        
    return False, 1.0, ""

def check_recent_only_langchain(candidate):
    """
    Soft Penalty: AI experience is <12 months of LangChain-calls-OpenAI work with no pre-LLM ML/production experience.
    """
    skills = candidate.get("skills", [])
    p = candidate.get("profile", {})
    yoe = p.get("years_of_experience", 0.0)
    
    llm_skills = {"langchain", "openai", "gpt", "prompt engineering", "llamaindex"}
    has_llm_skills = any(s.get("name", "").strip().lower() in llm_skills for s in skills)
    
    pre_llm_skills = {"machine learning", "data science", "nlp", "natural language processing", "scikit-learn", "pytorch", "tensorflow", "statistics"}
    
    max_pre_llm_duration = 0
    for s in skills:
        name = s.get("name", "").strip().lower()
        if name in pre_llm_skills:
            max_pre_llm_duration = max(max_pre_llm_duration, s.get("duration_months", 0))
            
    llm_durations = [s.get("duration_months", 0) for s in skills if s.get("name", "").strip().lower() in llm_skills]
    max_llm_duration = max(llm_durations) if llm_durations else 0
    
    if has_llm_skills and max_llm_duration > 0 and max_llm_duration <= 12:
        if max_pre_llm_duration < 12 and yoe < 3.0:
            return True, 0.4, "AI experience is limited to <12 months of LangChain/OpenAI calls without foundational pre-LLM ML experience"
            
    return False, 1.0, ""

def check_out_of_practice_senior(candidate):
    """
    Soft Penalty: Senior person who hasn't written production code in 18+ months.
    """
    career = candidate.get("career_history", [])
    if not career:
        return False, 1.0, ""
        
    current_roles = [r for r in career if r.get("is_current")]
    if not current_roles:
        current_roles = [career[0]]
        
    management_titles = {"director", "vp", "manager", "head of", "architect", "scrum master"}
    coding_keywords = {"code", "coding", "python", "java", "scala", "c++", "golang", "develop", "engineered", "implemented", "built", "programmed"}
    
    is_mgmt = False
    for r in current_roles:
        title = r.get("title", "").lower()
        desc = r.get("description", "").lower()
        
        has_mgmt_title = any(mt in title for mt in management_titles)
        has_coding_words = any(cw in desc for cw in coding_keywords)
        
        if has_mgmt_title and not has_coding_words and r.get("duration_months", 0) >= 18:
            is_mgmt = True
            
    if is_mgmt:
        return True, 0.3, "Senior candidate in management/architect role with no hands-on coding signals in the last 18+ months"
        
    return False, 1.0, ""

def check_services_only(candidate):
    """
    Soft Penalty: Career entirely within consulting/IT services.
    """
    career = candidate.get("career_history", [])
    if not career:
        return False, 1.0, ""
        
    all_services = True
    for r in career:
        comp = r.get("company", "").strip().lower()
        ind = r.get("industry", "").strip().lower()
        if comp not in IT_SERVICES_COMPANIES and "services" not in ind and "consulting" not in ind:
            all_services = False
            break
            
    if all_services:
        return True, 0.8, "Entire career spent within consulting/IT services firms (TCS, Infosys, Wipro, Accenture, Cognizant, etc.)"
        
    return False, 1.0, ""

def check_pure_cv_speech(candidate):
    """
    Soft Penalty: Pure computer-vision/speech/robotics specialist with no NLP/IR exposure.
    """
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    p = candidate.get("profile", {})
    
    skill_names = [s.get("name", "").strip().lower() for s in skills]
    career_descs = [r.get("description", "").strip().lower() for r in career]
    all_text = " ".join([p.get("summary", "").strip().lower()] + career_descs + skill_names)
    
    cv_matches = sum(1 for kw in CV_SPEECH_ROBOTICS_KEYWORDS if kw in all_text)
    nlp_matches = sum(1 for kw in NLP_IR_KEYWORDS if kw in all_text)
    
    if cv_matches > 3 and nlp_matches == 0:
        return True, 0.6, "Computer vision/speech/robotics specialist with no NLP or Information Retrieval exposure"
        
    return False, 1.0, ""

def check_title_chaser(candidate):
    """
    Soft Penalty: Career pattern of title-chasing.
    """
    career = candidate.get("career_history", [])
    if len(career) < 3:
        return False, 1.0, ""
        
    total_dur = sum(r.get("duration_months", 0) for r in career)
    avg_dur = total_dur / len(career)
    distinct_companies = len(set(r.get("company", "").lower().strip() for r in career))
    
    if avg_dur < 18.0 and distinct_companies >= 3:
        return True, 0.9, f"Title-chasing pattern: average tenure per role is {avg_dur:.1f} months across {distinct_companies} companies"
        
    return False, 1.0, ""

def check_proprietary_only(candidate):
    """
    Soft Penalty: 5+ years entirely on closed-source proprietary systems with zero external validation.
    """
    p = candidate.get("profile", {})
    yoe = p.get("years_of_experience", 0.0)
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    
    github_score = signals.get("github_activity_score", -1)
    
    if yoe >= 5.0 and github_score == -1:
        career_texts = [r.get("description", "").lower() for r in career]
        all_text = " ".join([p.get("summary", "").lower()] + career_texts)
        
        validation_keywords = {"open source", "oss", "paper", "talk", "conference", "patent", "publication", "meetup", "github", "contributor"}
        has_validation = any(w in all_text for w in validation_keywords)
        
        if not has_validation:
            return True, 0.9, "5+ years of experience on closed-source proprietary systems with zero external validation (no GitHub or public artifacts)"
            
    return False, 1.0, ""

def compute_confidence_score(candidate):
    """
    Computes candidate confidence score in [0.0, 1.0] based on completeness,
    verification status, and soft penalty warning deductions.
    """
    signals = candidate.get("redrob_signals", {})
    completeness = signals.get("profile_completeness_score", 50.0) / 100.0
    
    verification_boost = 0.0
    if signals.get("verified_email", False):
        verification_boost += 0.3
    if signals.get("verified_phone", False):
        verification_boost += 0.3
    if signals.get("linkedin_connected", False):
        verification_boost += 0.4
        
    base_confidence = 0.5 * completeness + 0.5 * verification_boost
    
    # Apply minor deductions based on soft penalties to scale down confidence
    deductions = 0.0
    for check_func in [
        check_pure_research, check_recent_only_langchain, check_out_of_practice_senior,
        check_services_only, check_pure_cv_speech, check_title_chaser, check_proprietary_only
    ]:
        is_trig, weight, _ = check_func(candidate)
        if is_trig:
            deductions += (1.0 - weight) * 0.15
            
    return float(max(0.1, min(1.0, base_confidence - deductions)))
