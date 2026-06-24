"""
Decoupled offline script to compute sentence embeddings and candidate features
for the dynamically retrieved candidate pool.
"""
import os
import json
import numpy as np
import torch
import time
from sentence_transformers import SentenceTransformer

# Feature stores and disqualifiers
from src.features.feature_store import extract_features
from src.features.lexical import compute_lexical_score
from src.features.title_intelligence import compute_title_alignment
from src.features.behavioral import compute_behavioral_score
from src.honeypot import check_is_honeypot
from src.disqualifiers import check_timeline_overlap, check_fake_title_inflation

def main():
    torch.set_num_threads(1)
    
    start_time = time.time()
    candidates_path = r"D:\projects\redrob\candidates.jsonl"
    output_dir = r"D:\projects\redrob\precompute\artifacts"
    embeddings_path = os.path.join(output_dir, "embeddings.npz")
    features_path = os.path.join(output_dir, "cached_features.json")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Step 1: Reading and filtering candidates...")
    candidates = []
    
    # We read a subset first to evaluate pool stats for adaptive retrieval sizing
    sample_candidates = []
    
    count = 0
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            c = json.loads(line)
            count += 1
            
            # Hard Fails (Contradictions, overlaps, inflation > 0.95)
            is_hp, _ = check_is_honeypot(c)
            if is_hp:
                continue
                
            is_overlap, _, _ = check_timeline_overlap(c)
            if is_overlap:
                continue
                
            is_inflated, weight, _ = check_fake_title_inflation(c)
            if is_inflated and weight == 0.0:  # Hard fail
                continue
                
            candidates.append(c)
            if len(sample_candidates) < 5000:
                sample_candidates.append(c)
                
            if count % 20000 == 0:
                print(f"Read {count} candidates...")
                
    print(f"Total candidates read: {count}")
    print(f"Clean candidates pool: {len(candidates)}")
    
    # Step 2: Compute Pool Stats for Adaptive Sizing
    print("Evaluating pool statistics for adaptive retrieval...")
    lexical_scores = []
    title_matches = 0
    
    for c in sample_candidates:
        lex_feats = compute_lexical_score(c)
        lexical_scores.append(lex_feats["lexical_score"])
        
        title_feats = compute_title_alignment(c)
        if title_feats["career_title_alignment"] > 0.0:
            title_matches += 1
            
    avg_lexical = np.mean(lexical_scores) if lexical_scores else 0.0
    pct_title = title_matches / len(sample_candidates) if sample_candidates else 0.0
    
    # Adaptive Pool Sizes
    lexical_pool_size = 1500 if avg_lexical < 0.2 else 800
    title_pool_size = 1200 if pct_title > 0.1 else 500
    concept_pool_size = 1000
    behavioral_pool_size = 1000
    
    print(f"Pool stats: Avg Lexical={avg_lexical:.3f}, Pct Title={pct_title:.3f}")
    print(f"Adaptive Pool Sizes -> Lexical: {lexical_pool_size}, Title: {title_pool_size}, Concept: {concept_pool_size}, Behavioral: {behavioral_pool_size}")
    
    # Step 3: Multi-Source Candidate Retrieval
    print("Retrieving candidates across sources...")
    scored_candidates = []
    
    for c in candidates:
        cid = c.get("candidate_id")
        
        # 1. Lexical score
        lex_score = compute_lexical_score(c)["lexical_score"]
        
        # 2. Title score
        title_score = compute_title_alignment(c)["career_title_alignment"]
        
        # 3. Concept score (density of keywords in text)
        p = c.get("profile", {})
        career = c.get("career_history", [])
        desc_text = " ".join([p.get("summary", "")] + [role.get("description", "") for role in career]).lower()
        concept_terms = ["retrieval", "search", "ranking", "recommendation", "matching", "personalization", "ndcg", "map", "mrr"]
        concept_score = sum(1 for term in concept_terms if term in desc_text)
        
        # 4. Behavioral score
        behavior_score = compute_behavioral_score(c)["behavior_multiplier"]
        
        scored_candidates.append({
            "candidate_id": cid,
            "candidate": c,
            "lexical": float(lex_score),
            "title": float(title_score),
            "concept": float(concept_score),
            "behavioral": float(behavior_score)
        })
        
    # Get top pools
    top_lexical = sorted(scored_candidates, key=lambda x: -x["lexical"])[:lexical_pool_size]
    top_title = sorted(scored_candidates, key=lambda x: -x["title"])[:title_pool_size]
    top_concept = sorted(scored_candidates, key=lambda x: -x["concept"])[:concept_pool_size]
    top_behavioral = sorted(scored_candidates, key=lambda x: -x["behavioral"])[:behavioral_pool_size]
    
    # Union candidates
    union_ids = set()
    union_pool = []
    
    for pool in [top_lexical, top_title, top_concept, top_behavioral]:
        for item in pool:
            cid = item["candidate_id"]
            if cid not in union_ids:
                union_ids.add(cid)
                # Compute composite score for truncation
                composite = (0.4 * item["lexical"] + 
                             0.3 * item["title"] + 
                             0.2 * item["concept"] + 
                             0.1 * item["behavioral"])
                item["composite"] = composite
                union_pool.append(item)
                
    print(f"Union candidate pool size: {len(union_pool)}")
    
    # Truncate to MAX_UNION_POOL = 3000
    union_pool.sort(key=lambda x: -x["composite"])
    selected_pool = union_pool[:3000]
    print(f"Truncated selected pool size: {len(selected_pool)}")
    
    # Step 4: Extract embeddings and cache base candidate features
    texts = []
    candidate_ids = []
    cached_candidates = []
    
    for item in selected_pool:
        c = item["candidate"]
        cid = item["candidate_id"]
        
        p = c.get("profile", {})
        summary = p.get("summary", "")
        career = c.get("career_history", [])
        descriptions = [role.get("description", "") for role in career]
        full_text = " ".join([summary] + descriptions).strip()[:256] # Truncate for embedding model
        
        texts.append(full_text)
        candidate_ids.append(cid)
        cached_candidates.append(c)
        
    print("Initializing SentenceTransformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=128,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print("Saving embeddings matrix...")
    np.savez_compressed(
        embeddings_path,
        embeddings=embeddings.astype(np.float32),
        candidate_ids=np.array(candidate_ids)
    )
    
    # Cache candidate features
    print("Caching candidate profiles...")
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(cached_candidates, f, ensure_ascii=False, indent=2)
        
    elapsed = time.time() - start_time
    print(f"Step 4 complete! Precomputation files saved to {output_dir}")
    print(f"Total time elapsed: {elapsed:.2f}s ({elapsed/60.0:.2f} minutes).")

if __name__ == "__main__":
    main()
