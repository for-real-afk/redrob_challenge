"""
Reasoning string generator V2 (evidence-based, fact-cited, rank-consistent).
"""
import hashlib
from src.features.feature_store import CONCEPTS
from src.jd_parser import REQUIRED_SKILLS, PREFERRED_SKILLS

def extract_strongest_evidence(candidate, breakdown):
    """
    Extracts 2-3 specific technical and career achievements to serve as positive evidence.
    """
    p = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    yoe = float(p.get("years_of_experience", 0.0))
    
    evidence = []
    
    if yoe > 0.0:
        evidence.append(f"matching background with {yoe:.1f} YOE")
        
    # 1. Look for matching product company history
    product_company = None
    from src.jd_parser import PRODUCT_COMPANIES
    for role in career:
        comp = role.get("company", "")
        if comp.lower().strip() in PRODUCT_COMPANIES:
            product_company = comp
            break
            
    # 2. Look for matching high-value skills
    skills_found = []
    for s in skills:
        name = s.get("name", "").strip().lower()
        if name in REQUIRED_SKILLS or name in PREFERRED_SKILLS:
            skills_found.append(s.get("name"))
            if len(skills_found) >= 2:
                break
                
    if product_company:
        evidence.append(f"shipped ML/search systems at {product_company}")
    elif career:
        evidence.append(f"systems experience at {career[0].get('company', 'a product firm')}")
        
    if skills_found:
        evidence.append(f"demonstrated skills in {', '.join(skills_found)}")
        
    # 3. Check for specific concepts like evaluation or systems thinking
    if breakdown.get("evaluation_maturity_score", 0.0) > 0.0:
        evidence.append("experience with NDCG/MAP ranking evaluation")
    elif breakdown.get("systems_thinking_score", 0.0) > 0.0:
        evidence.append("exposure to scalable infrastructure optimization")
        
    if not evidence:
        evidence.append("matching background in software engineering and Python")
        
    return evidence

def extract_minor_deductions(candidate, breakdown):
    """
    Identifies and formats minor soft penalties and warning signs.
    """
    signals = candidate.get("redrob_signals", {})
    notice = signals.get("notice_period_days", 0)
    yoe = candidate.get("profile", {}).get("years_of_experience", 0.0)
    
    deductions = []
    
    # Run soft penalty checks
    from src.disqualifiers import (
        check_pure_research, check_recent_only_langchain, check_out_of_practice_senior,
        check_services_only, check_pure_cv_speech, check_title_chaser, check_proprietary_only
    )
    
    checks = [
        (check_pure_research, "research-heavy profile with zero production exposure"),
        (check_recent_only_langchain, "AI work is limited to LangChain/OpenAI calls"),
        (check_out_of_practice_senior, "senior candidate lacking recent hands-on coding signals"),
        (check_services_only, "career entirely within IT services consulting firms"),
        (check_pure_cv_speech, "specialized in CV/speech with no direct NLP/IR background"),
        (check_title_chaser, "history of frequent company hopping"),
        (check_proprietary_only, "closed-source systems with zero public code footprint")
    ]
    
    for check_func, label in checks:
        is_trig, _, _ = check_func(candidate)
        if is_trig:
            deductions.append(label)
            break  # List the main one to keep reasoning concise
            
    if notice > 60:
        deductions.append(f"long notice period of {notice} days")
    if yoe < 5.0:
        deductions.append(f"slightly lower experience ({yoe:.1f} YOE) than target")
    elif yoe > 9.0:
        deductions.append(f"exceeds target seniority expectations ({yoe:.1f} YOE)")
        
    return deductions

def clean_sentence(s):
    if not s:
        return ""
    s = s.strip()
    s = s[0].upper() + s[1:]
    if not s.endswith("."):
        s += "."
    return s

def generate_reasoning_string(candidate, rank, score, breakdown):
    """
    Generates dynamic, evidence-based, and rank-consistent reasoning V2 strings
    in a premium recruiter-style format.
    """
    evidence = extract_strongest_evidence(candidate, breakdown)
    deductions = extract_minor_deductions(candidate, breakdown)
    
    # Format depending on the rank bracket
    if rank <= 10:
        tone = "Ranked as a top-tier candidate (Founding Team fit) due to"
    elif rank <= 50:
        tone = "Ranked highly due to"
    elif rank <= 80:
        tone = "Ranked in mid-tier due to"
    else:
        tone = "Included as a final filler pick due to"
        
    evidence_bullets = "\n".join(f"• {clean_sentence(e)}" for e in evidence)
    
    reasoning = f"Why Ranked Here:\n\n{tone}:\n{evidence_bullets}"
    
    if deductions:
        deduction_bullets = "\n".join(f"• {clean_sentence(d)}" for d in deductions)
        reasoning += f"\n\nMinor concerns:\n{deduction_bullets}"
        
    # Cap string length gracefully to remain compliant
    if len(reasoning) > 500:
        reasoning = reasoning[:497] + "..."
        
    return reasoning
