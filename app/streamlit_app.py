"""
Streamlit sandbox web application for candidates ranking simulation.
"""
import streamlit as st
import json
import pandas as pd
from src.scoring import score_candidate
from src.reasoning import generate_reasoning_string
from src.honeypot import check_is_honeypot
from src.features.semantic import compute_jd_embedding, load_precomputed_embeddings
import numpy as np

# Page configuration
st.set_page_config(layout="wide", page_title="Redrob Ranker Sandbox")

st.title("Redrob Ranker Sandbox")
st.write("A reproducible ranking sandbox for validation on small candidate samples.")

# 1. Cache the embedding loader and model
@st.cache_resource
def get_semantic_scoring_resources():
    """
    Loads and caches the precomputed embeddings and the JD query vector.
    """
    embeddings, id_to_idx = load_precomputed_embeddings()
    jd_vector = compute_jd_embedding()
    jd_norm = np.linalg.norm(jd_vector)
    return embeddings, id_to_idx, jd_vector, jd_norm

# Load resources
try:
    embeddings, id_to_idx, jd_vector, jd_norm = get_semantic_scoring_resources()
    st.success("Precomputed embeddings and model loaded successfully!")
except Exception as e:
    st.error(f"Error loading precomputed embeddings. Did you run the precomputation script? Details: {e}")
    st.stop()

# 2. File Uploader
uploaded_file = st.file_uploader("Upload a candidate sample (.jsonl or .json)", type=["jsonl", "json"])

# Preloaded sample checkbox if no file uploaded
preload_sample = st.checkbox("Use preloaded sample_candidates.json", value=True)

candidates = []

if uploaded_file is not None:
    try:
        content = uploaded_file.read().decode("utf-8")
        if uploaded_file.name.endswith(".jsonl"):
            for line in content.split("\n"):
                if line.strip():
                    candidates.append(json.loads(line))
        else: # .json
            candidates = json.loads(content)
        st.info(f"Loaded {len(candidates)} candidates from upload.")
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
elif preload_sample:
    try:
        with open("sample_candidates.json", "r", encoding="utf-8") as f:
            candidates = json.load(f)
        st.info(f"Loaded {len(candidates)} candidates from sample_candidates.json.")
    except Exception as e:
        st.warning(f"Could not load sample_candidates.json: {e}")

# 3. Execution
if candidates:
    if len(candidates) > 100:
        st.warning("Sample size exceeds 100. Streamlit Cloud sandbox restricts parsing to 100 for latency check.")
        candidates = candidates[:100]
        
    if st.button("Run Ranking", type="primary"):
        ranked_list = []
        
        # We calculate semantic similarity for the uploaded candidates
        for c in candidates:
            cid = c.get("candidate_id")
            sem_score = 0.0
            
            # Lookup precomputed embedding if available in index
            if cid in id_to_idx:
                idx = id_to_idx[cid]
                cand_vec = embeddings[idx]
                cand_norm = np.linalg.norm(cand_vec)
                if cand_norm == 0:
                    cand_norm = 1e-9
                sem_score = float(np.dot(cand_vec, jd_vector) / (cand_norm * jd_norm))
            else:
                # If candidate is new and not precomputed, default to neutral semantic score
                sem_score = 0.45
                
            score, breakdown, is_disq, reasons = score_candidate(c, sem_score)
            is_hp, hp_reasons = check_is_honeypot(c)
            
            ranked_list.append({
                "candidate_id": cid,
                "anonymized_name": c.get("profile", {}).get("anonymized_name", "Anonymous"),
                "score": score,
                "is_honeypot": is_hp,
                "is_disqualified": is_disq,
                "disq_reasons": ", ".join(reasons + hp_reasons),
                "breakdown": breakdown,
                "candidate": c
            })
            
        # Sort and Rank
        ranked_list.sort(key=lambda x: (-x["score"], x["candidate_id"]))
        
        # Keep only positive/non-disqualified for submission table
        display_list = []
        rank_counter = 1
        for item in ranked_list:
            if not item["is_disqualified"] and item["score"] > 0:
                reasoning = generate_reasoning_string(item["candidate"], rank_counter, item["score"], item["breakdown"])
                display_list.append({
                    "Rank": rank_counter,
                    "Candidate ID": item["candidate_id"],
                    "Name": item["anonymized_name"],
                    "Score": item["score"],
                    "Honeypot?": "🚨 Flagged" if item["is_honeypot"] else "✅ Clean",
                    "Reasoning": reasoning,
                    "Item": item
                })
                rank_counter += 1
                
        # Show Ranked DataFrame
        st.subheader("Ranked Candidates (Top Fit)")
        if display_list:
            df = pd.DataFrame(display_list)[["Rank", "Candidate ID", "Name", "Score", "Honeypot?", "Reasoning"]]
            st.dataframe(df, use_container_width=True)
            
            # 4. Attribution/Transparency details
            st.subheader("Attribution & Score Transparency Details")
            for item in display_list:
                with st.expander(f"Rank {item['Rank']} | {item['Name']} ({item['Candidate ID']}) — Score: {item['Score']}"):
                    bd = item["Item"]["breakdown"]
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Lexical (Normalized)", f"{bd.get('norm_lexical', 0.0):.2f}")
                    col2.metric("Semantic (Normalized)", f"{bd.get('norm_semantic', 0.0):.2f}")
                    col3.metric("Seniority Alignment", f"{bd.get('seniority_alignment', 0.0):.2f}")
                    col4.metric("Behavioral Score", f"{bd.get('behavioral_score', 0.0):.2f}")
                    
                    cand_obj = item['Item']['candidate']
                    st.write("**Factual Details:**")
                    st.write(f"- Years of Experience: {cand_obj.get('profile', {}).get('years_of_experience', 0.0):.1f}")
                    st.write(f"- Last Active: {cand_obj.get('redrob_signals', {}).get('last_active_date', 'N/A')}")
                    st.write(f"- Stated Notice Period: {cand_obj.get('redrob_signals', {}).get('notice_period_days', 0)} days")
                    st.write(f"- Stated Expected Salary: {cand_obj.get('redrob_signals', {}).get('expected_salary_range_inr_lpa', {}).get('max', 0)} LPA max")
        else:
            st.warning("No candidates passed the disqualification filters with a positive score.")
            
        # Show Disqualified / Honeypot candidates at the bottom
        disq_list = [item for item in ranked_list if item["is_disqualified"]]
        if disq_list:
            st.subheader("Disqualified / Filtered Candidates")
            df_disq = pd.DataFrame([
                {
                    "Candidate ID": item["candidate_id"],
                    "Name": item["anonymized_name"],
                    "Honeypot?": "🚨 Yes" if item["is_honeypot"] else "No",
                    "Disqualification Reason": item["disq_reasons"]
                }
                for item in disq_list
            ])
            st.dataframe(df_disq, use_container_width=True)
else:
    st.info("Upload candidate profiles or use the preloaded sample checkbox to begin ranking.")
