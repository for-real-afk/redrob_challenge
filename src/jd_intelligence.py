"""
JD Intelligence Layer parsing Word JDs dynamically into structured recruiter queries and features.
"""
import os
import zipfile
import re
import xml.etree.ElementTree as ET

SKILL_VOCAB = {
    "python", "xgboost", "lightgbm", "sentence-transformers", "embeddings", "vector search",
    "dense retrieval", "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "faiss", "ndcg", "mrr", "map", "a/b testing", "lora", "qlora", "peft", "docker", "kubernetes",
    "distributed systems", "inference optimization", "mlops"
}

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

def parse_docx(path):
    """
    Extracts text from a docx file using Python's standard zipfile and ElementTree.
    """
    if not os.path.exists(path):
        return ""
    try:
        with zipfile.ZipFile(path) as doc:
            xml_content = doc.read('word/document.xml')
            root = ET.fromstring(xml_content)
            namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            text_parts = []
            for p in root.findall('.//w:p', namespaces):
                parts = [t.text for t in p.findall('.//w:t', namespaces) if t.text]
                if parts:
                    text_parts.append("".join(parts))
            return "\n".join(text_parts)
    except Exception as e:
        print(f"Warning: could not parse docx {path}: {e}")
        return ""

def analyze_jd(jd_text):
    """
    Dynamically extracts recruiter intelligence from job description text.
    """
    clean_text = jd_text.lower()
    
    # 1. Core Skills extraction
    core_skills = []
    for skill in SKILL_VOCAB:
        if re.search(r'\b' + re.escape(skill) + r'\b', clean_text):
            core_skills.append(skill)
            
    # 2. Concepts extraction
    concepts = []
    for family, keywords in CONCEPT_VOCAB.items():
        if any(re.search(r'\b' + re.escape(kw) + r'\b', clean_text) for kw in keywords):
            concepts.append(family)
            
    # 3. Domain extraction
    domain = []
    domains_words = ["recruiting", "hr", "marketplace", "fintech", "saas", "e-commerce", "applied ml"]
    for d in domains_words:
        if d in clean_text:
            domain.append(d)
            
    # 4. Seniority level
    seniority_level = "mid"
    if "senior" in clean_text or "founding team" in clean_text:
        seniority_level = "senior"
    elif "lead" in clean_text or "staff" in clean_text or "principal" in clean_text:
        seniority_level = "lead/staff"
    elif "junior" in clean_text or "intern" in clean_text:
        seniority_level = "junior"
        
    # 5. Leadership expectation
    leadership_expectation = "none"
    if any(word in clean_text for word in ["mentor", "mentoring", "leadership", "mentoring", "hiring", "lead the team"]):
        leadership_expectation = "medium"
    if "manage a team" in clean_text or "director" in clean_text or "vp" in clean_text:
        leadership_expectation = "high"
        
    # 6. Experience expectation
    experience_expectation = "5-9"
    yoe_match = re.search(r'(\d+)\s*[–-]\s*(\d+)\s*years', clean_text)
    if yoe_match:
        experience_expectation = f"{yoe_match.group(1)}-{yoe_match.group(2)}"
    else:
        yoe_single = re.search(r'(\d+)\s*\+\s*years', clean_text)
        if yoe_single:
            experience_expectation = f"{yoe_single.group(1)}+"

    # 7. Systems requirements
    systems_requirements = []
    systems_words = ["distributed", "scalability", "concurrency", "throughput", "latency", "docker", "kubernetes"]
    for w in systems_words:
        if w in clean_text:
            systems_requirements.append(w)
            
    # 8. Evaluation requirements
    evaluation_requirements = []
    eval_words = ["ndcg", "map", "mrr", "a/b testing", "ab testing", "offline evaluation", "benchmarks"]
    for w in eval_words:
        if w in clean_text:
            evaluation_requirements.append(w)
            
    # 9. Behavioral requirements
    behavioral_requirements = []
    behav_words = ["notice period", "expected salary", "relocation", "hybrid", "remote"]
    for w in behav_words:
        if w in clean_text:
            behavioral_requirements.append(w)
            
    # 10. Title signals
    title_signals = []
    title_words = ["ai engineer", "search engineer", "ranking engineer", "ml engineer", "applied scientist", "relevance engineer"]
    for w in title_words:
        if w in clean_text:
            title_signals.append(w)
            
    # Compute query-level features
    jd_skill_count = float(len(core_skills))
    jd_concept_count = float(len(concepts))
    
    # Title specificity: count how many high-value title keywords matches
    matched_titles = sum(1 for w in title_words if w in clean_text)
    jd_title_specificity = float(min(1.0, matched_titles / 5.0))
    
    # Seniority map
    seniority_map = {"junior": 1.0, "mid": 2.0, "senior": 3.0, "lead/staff": 4.0}
    jd_seniority_level = float(seniority_map.get(seniority_level, 2.0))
    
    # Entropy: ratio of preferred to core skills or warning flags
    jd_entropy = float(min(1.0, (len(behavioral_requirements) + len(systems_requirements)) / 10.0))
    
    return {
        "core_skills": core_skills,
        "concepts": concepts,
        "domain": domain,
        "seniority_level": seniority_level,
        "leadership_expectation": leadership_expectation,
        "experience_expectation": experience_expectation,
        "systems_requirements": systems_requirements,
        "evaluation_requirements": evaluation_requirements,
        "behavioral_requirements": behavioral_requirements,
        "title_signals": title_signals,
        
        # Computed features
        "jd_skill_count": jd_skill_count,
        "jd_concept_count": jd_concept_count,
        "jd_title_specificity": jd_title_specificity,
        "jd_seniority_level": jd_seniority_level,
        "jd_entropy": jd_entropy
    }

_CACHED_JD_INTEL = None

def get_jd_intelligence(docx_path=r"D:\projects\redrob\job_description.docx"):
    """
    Wrapper caching parsed JD intelligence. Falls back to default if file parsing fails.
    """
    global _CACHED_JD_INTEL
    if _CACHED_JD_INTEL is not None:
        return _CACHED_JD_INTEL
        
    text = parse_docx(docx_path)
    if not text:
        # High quality default matching the Senior AI Engineer JD
        text = (
            "Job Description: Senior AI Engineer — Founding Team. Pune/Noida, India. "
            "Experience Required: 5-9 years. Modern ML systems — embeddings, retrieval, ranking, "
            "LLMs, fine-tuning. Production experience with sentence-transformers, Pinecone, Weaviate, "
            "Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS. Evaluation metrics: NDCG, MRR, MAP, "
            "A/B testing, offline evaluation. Preferred: LoRA, QLoRA, PEFT, learning to rank, "
            "distributed systems, docker, kubernetes, hr-tech, recruiting. Notice period sub-30 days."
        )
        
    _CACHED_JD_INTEL = analyze_jd(text)
    return _CACHED_JD_INTEL
