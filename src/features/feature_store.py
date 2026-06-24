"""
Feature Store pipeline extracting candidate features, dynamic interaction features,
query-level features, and explicit recruiter-style feature interactions.
"""
import re
from src.features.title_intelligence import compute_title_alignment
from src.features.career_trajectory import compute_career_trajectory
from src.features.company_intelligence import compute_company_intelligence
from src.features.lexical import compute_lexical_score
from src.features.behavioral import compute_behavioral_score
from src.jd_intelligence import get_jd_intelligence
from src.disqualifiers import compute_confidence_score

CONCEPT_VOCAB = {
    "retrieval": ["retrieval", "search", "dense retrieval", "hybrid search"],
    "ranking": ["ranking", "learning to rank", "xgbranker", "ranking evaluation"],
    "recommendation": ["recommendation", "recommendation system"],
    "systems": ["distributed systems", "inference optimization", "large-scale inference", "mlops"],
    "evaluation": ["ndcg", "mrr", "map", "a/b testing", "evaluation infrastructure"],
    "production_ml": ["production", "serving", "deployment", "latency", "monitoring"],
    "matching": ["matching", "marketplace", "personalization"],
    "personalization": ["personalization"],
    "marketplace_ml": ["marketplace", "marketplace products"]
}

CONCEPTS = CONCEPT_VOCAB

def extract_all_text(candidate):
    """
    Combines profile summary and career descriptions into a single clean text string.
    """
    p = candidate.get("profile", {})
    summary = p.get("summary", "")
    career = candidate.get("career_history", [])
    descriptions = [r.get("description", "") for r in career]
    return " ".join([summary] + descriptions).strip().lower()

def compute_evidence_density(text):
    """
    Calculates evidence density based on metrics, numbers, impact verbs, and production scale.
    """
    if not text:
        return 0.0
    
    # 1. Count impact verbs (distinct occurrences)
    impact_verbs = {
        "improved", "reduced", "optimized", "increased", "served", "scaled",
        "architected", "implemented", "deployed", "designed", "built", "engineered",
        "spearheaded", "led", "delivered", "saved"
    }
    verb_count = sum(1 for verb in impact_verbs if re.search(r'\b' + re.escape(verb) + r'\b', text))
    
    # 2. Count metric mentions (e.g. 10%, 5M, 100k, 500GB, 40ms, etc.)
    metric_count = len(re.findall(r'\b\d+(?:\.\d+)?\s*(?:%|m|k|gb|tb|tbps|users|queries|req|ms|s|x|ndcg|map|mrr)\b', text))
    
    # 3. Count production scale mentions
    scale_terms = {
        "million", "billion", "qps", "requests/day", "req/sec", "users", "queries/sec", 
        "scale", "scalability", "throughput", "concurrent", "active users", "load testing"
    }
    scale_count = sum(1 for term in scale_terms if re.search(r'\b' + re.escape(term) + r'\b', text))
    scale_count += len(re.findall(r'\b\d+\s*[mk]\b', text)) # e.g. 10M, 500k
    
    # 4. Count of all numeric indicators in text
    numbers_count = len(re.findall(r'\b\d+(?:\.\d+)?\b', text))
    
    # Bounded combination score
    verb_score = min(verb_count, 5) / 5.0
    metric_score = min(metric_count, 4) / 4.0
    scale_score = min(scale_count, 3) / 3.0
    numbers_score = min(numbers_count, 5) / 5.0
    
    score = 0.25 * verb_score + 0.25 * metric_score + 0.3 * scale_score + 0.2 * numbers_score
    return float(score)

def compute_concept_scores(text):
    """
    Computes scores for each concept ontology family.
    """
    scores = {}
    if not text:
        return {family: 0.0 for family in CONCEPT_VOCAB.keys()}
        
    for family, terms in CONCEPT_VOCAB.items():
        match_count = 0
        for term in terms:
            if re.search(r'\b' + re.escape(term) + r'\b', text):
                match_count += 1
        scores[family] = float(min(match_count, 3) / 3.0)
    return scores

def compute_hidden_gem_score(text, has_high_value_title, trajectory, concept_scores):
    """
    Identifies strong candidates that keyword systems miss. Avoids buzzwords and rewards core retrieval expertise.
    """
    if not text:
        return 0.0
    
    buzzwords = {"rag", "pinecone", "langchain", "openai", "gpt", "prompt engineering"}
    has_buzzwords = any(bw in text for bw in buzzwords)
    if has_buzzwords:
        return 0.0
        
    # Sum scores of core IR concept families
    retrieval_weight = concept_scores.get("retrieval", 0.0) + concept_scores.get("ranking", 0.0) + concept_scores.get("recommendation", 0.0)
    retrieval_strength = min(1.0, retrieval_weight / 2.0)
    
    score = 0.4 * retrieval_strength + 0.3 * trajectory.get("domain_continuity", 0.0) + 0.3 * float(has_high_value_title)
    return float(score)

def compute_systems_thinking_score(text):
    """
    Evaluates exposure to scalability, optimization, latency, monitoring, and architecture.
    """
    if not text:
        return 0.0
    signals = ["architecture", "distributed systems", "scalability", "optimization", "latency", "serving", "monitoring", "infrastructure"]
    match_count = sum(1 for term in signals if re.search(r'\b' + re.escape(term) + r'\b', text))
    return float(min(match_count, 4) / 4.0)

def compute_evaluation_maturity_score(text):
    """
    Evaluates exposure to offline evaluation metrics, experimentation, and online testing.
    """
    if not text:
        return 0.0
    signals = ["ndcg", "map", "mrr", "offline evaluation", "online evaluation", "a/b testing", "relevance metrics", "experimentation", "ab testing"]
    match_count = sum(1 for term in signals if re.search(r'\b' + re.escape(term) + r'\b', text))
    return float(min(match_count, 4) / 4.0)

def extract_features(candidate, semantic_score):
    """
    Extracts all candidate, query-level, interaction, and cross-features dynamically based on JD intelligence.
    """
    # 1. Fetch dynamic JD intelligence
    jd_intel = get_jd_intelligence()
    
    text = extract_all_text(candidate)
    
    # Extract sub-features
    title_feats = compute_title_alignment(candidate)
    trajectory_feats = compute_career_trajectory(candidate)
    company_feats = compute_company_intelligence(candidate)
    lexical_feats = compute_lexical_score(candidate)
    behavior_feats = compute_behavioral_score(candidate)
    
    # Extract base continuous scores
    lex_score = lexical_feats["lexical_score"]
    norm_lexical = min(lex_score, 3.0) / 3.0
    
    concept_scores = compute_concept_scores(text)
    systems_thinking_score = compute_systems_thinking_score(text)
    evaluation_maturity_score = compute_evaluation_maturity_score(text)
    evidence_density = compute_evidence_density(text)
    confidence_score = compute_confidence_score(candidate)
    
    # Hidden gem scoring using trajectory and ontologies
    hidden_gem = compute_hidden_gem_score(
        text, 
        title_feats["career_title_alignment"], 
        trajectory_feats, 
        concept_scores
    )
    
    # 2. JD-Candidate Interaction Features (Measure Candidate x JD)
    title_interaction = max(title_feats["current_title_alignment"], title_feats["historical_title_alignment"])
    
    # Concept Coverage overlap
    cand_concepts = [f for f, v in concept_scores.items() if v > 0.0]
    jd_concepts = jd_intel.get("concepts", [])
    concept_intersection = set(cand_concepts).intersection(set(jd_concepts))
    concept_coverage = len(concept_intersection) / len(jd_concepts) if jd_concepts else 0.0
    
    # Experience Alignment (YOE vs expectation center)
    exp_str = jd_intel.get("experience_expectation", "5-9")
    try:
        if "-" in exp_str:
            low, high = map(float, exp_str.split("-"))
            center = (low + high) / 2.0
        else:
            center = float(exp_str.replace("+", ""))
    except:
        center = 7.0
    yoe = float(candidate.get("profile", {}).get("years_of_experience", 0.0))
    experience_alignment = abs(yoe - center)
    
    # Seniority Alignment
    candidate_seniority = 2.0
    if trajectory_feats["promotion_velocity"] > 0.7 or title_feats["career_title_alignment"] > 0.5:
        candidate_seniority = 3.0
    if trajectory_feats["leadership_growth"] > 0.0:
        candidate_seniority = 4.0
    seniority_alignment = float(1.0 - abs(candidate_seniority - jd_intel["jd_seniority_level"]) / 4.0)
    
    # Domain Alignment
    cand_domains = [d for d in ["recruiting", "hr", "marketplace", "fintech", "saas", "e-commerce"] if d in text]
    jd_domains = jd_intel.get("domain", [])
    domain_intersection = set(cand_domains).intersection(set(jd_domains))
    domain_alignment = len(domain_intersection) / len(jd_domains) if jd_domains else 0.0
    
    # Systems Alignment
    cand_sys = [w for w in ["distributed", "scalability", "concurrency", "throughput", "latency", "docker", "kubernetes"] if w in text]
    jd_sys = jd_intel.get("systems_requirements", [])
    sys_intersection = set(cand_sys).intersection(set(jd_sys))
    systems_alignment = len(sys_intersection) / len(jd_sys) if jd_sys else 0.0
    
    # Evaluation Alignment
    cand_eval = [w for w in ["ndcg", "map", "mrr", "a/b testing", "ab testing", "offline evaluation", "benchmarks"] if w in text]
    jd_eval = jd_intel.get("evaluation_requirements", [])
    eval_intersection = set(cand_eval).intersection(set(jd_eval))
    evaluation_alignment = len(eval_intersection) / len(jd_eval) if jd_eval else 0.0
    
    # Behavior Alignment
    avail = 1.0 if candidate.get("redrob_signals", {}).get("open_to_work_flag", False) else 0.5
    notice = candidate.get("redrob_signals", {}).get("notice_period_days", 90)
    notice_val = 1.0 if notice <= 30 else (0.5 if notice <= 60 else 0.1)
    behavior_alignment = 0.5 * avail + 0.5 * notice_val
    
    # 3. Query-level Features (Extracted dynamically)
    jd_skill_count = jd_intel["jd_skill_count"]
    jd_concept_count = jd_intel["jd_concept_count"]
    jd_title_specificity = jd_intel["jd_title_specificity"]
    jd_seniority_level = jd_intel["jd_seniority_level"]
    jd_entropy = jd_intel["jd_entropy"]
    
    # 4. Explicit Cross-Feature Interactions
    retrieval_experience_x_production_ml = domain_alignment * concept_scores.get("production_ml", 0.0)
    title_alignment_x_systems_thinking = title_feats["current_title_alignment"] * systems_thinking_score
    evaluation_maturity_x_domain_alignment = evaluation_maturity_score * domain_alignment
    career_consistency_x_seniority_alignment = trajectory_feats["career_consistency"] * seniority_alignment
    systems_alignment_x_evidence_density = systems_alignment * evidence_density
    
    # Behavioral and availability base score
    behavioral_score = behavior_feats["behavior_multiplier"]
    
    # Capped prestige signals (supporting values)
    capped_product_ratio = float(company_feats["career_product_ratio"]) * 0.1
    capped_startup_ratio = float(company_feats["recent_product_ratio"]) * 0.1
    capped_company_type = float(company_feats["current_company_type"]) * 0.1
    
    return {
        # Candidate Features
        "semantic_score": float(semantic_score),
        "lexical_score": float(norm_lexical),
        "concept_score_retrieval": float(concept_scores.get("retrieval", 0.0)),
        "concept_score_ranking": float(concept_scores.get("ranking", 0.0)),
        "concept_score_recommendation": float(concept_scores.get("recommendation", 0.0)),
        "concept_score_systems": float(concept_scores.get("systems", 0.0)),
        "concept_score_evaluation": float(concept_scores.get("evaluation", 0.0)),
        "concept_score_production_ml": float(concept_scores.get("production_ml", 0.0)),
        "concept_score_matching": float(concept_scores.get("matching", 0.0)),
        "concept_score_personalization": float(concept_scores.get("personalization", 0.0)),
        "concept_score_marketplace_ml": float(concept_scores.get("marketplace_ml", 0.0)),
        "hidden_gem_score": float(hidden_gem),
        "trajectory_score": float(trajectory_feats["trajectory_score"]),
        "title_alignment": float(title_feats["current_title_alignment"]),
        "product_ratio": capped_product_ratio,
        "startup_ratio": capped_startup_ratio,
        "current_company_type": capped_company_type,
        "systems_thinking_score": float(systems_thinking_score),
        "evaluation_maturity_score": float(evaluation_maturity_score),
        "evidence_density_score": float(evidence_density),
        "behavioral_score": float(behavioral_score),
        "confidence_score": float(confidence_score),
        
        # JD Interaction Features
        "title_interaction": float(title_interaction),
        "concept_coverage": float(concept_coverage),
        "experience_alignment": float(experience_alignment),
        "seniority_alignment": float(seniority_alignment),
        "domain_alignment": float(domain_alignment),
        "systems_alignment": float(systems_alignment),
        "evaluation_alignment": float(evaluation_alignment),
        "behavior_alignment": float(behavior_alignment),
        
        # Query-level Features
        "jd_skill_count": float(jd_skill_count),
        "jd_concept_count": float(jd_concept_count),
        "jd_title_specificity": float(jd_title_specificity),
        "jd_seniority_level": float(jd_seniority_level),
        "jd_entropy": float(jd_entropy),
        
        # Explicit Feature Interactions
        "retrieval_experience_x_production_ml": float(retrieval_experience_x_production_ml),
        "title_alignment_x_systems_thinking": float(title_alignment_x_systems_thinking),
        "evaluation_maturity_x_domain_alignment": float(evaluation_maturity_x_domain_alignment),
        "career_consistency_x_seniority_alignment": float(career_consistency_x_seniority_alignment),
        "systems_alignment_x_evidence_density": float(systems_alignment_x_evidence_density)
    }
