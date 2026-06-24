#!/usr/bin/env python3
"""
Entrypoint for candidate discovery and ranking challenge.
Usage: python rank.py --candidates <candidates_file> --out <output_csv>
"""
import argparse
import csv
import sys
import time
import numpy as np
from src.data_loading import load_candidates
from src.scoring import score_and_rank_candidates
from src.reasoning import generate_reasoning_string
from validate_submission import validate_submission

# Basic retrieval scores
from src.features.lexical import compute_lexical_score
from src.features.title_intelligence import compute_title_alignment
from src.features.behavioral import compute_behavioral_score
from src.honeypot import check_is_honeypot
from src.disqualifiers import check_timeline_overlap, check_fake_title_inflation

def main():
    import torch
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description="Rank candidates for the Founding Team AI Engineer position.")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", required=True, help="Path to write the output CSV")
    args = parser.parse_args()
    
    start_time = time.perf_counter()
    print("Starting candidate discovery & ranking pipeline...")
    
    # 1. Load and parse candidates
    print(f"Loading candidates from {args.candidates}...")
    try:
        candidates = list(load_candidates(args.candidates))
        print(f"Successfully loaded {len(candidates)} candidates.")
    except Exception as e:
        print(f"Error loading candidates: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 2. Filter out Hard Fails (honeypots, overlaps, title inflation > 0.95)
    print("Running hard-fail ejections...")
    clean_candidates = []
    sample_candidates = []
    
    for c in candidates:
        is_hp, _ = check_is_honeypot(c)
        if is_hp:
            continue
            
        is_overlap, _, _ = check_timeline_overlap(c)
        if is_overlap:
            continue
            
        is_inflated, weight, _ = check_fake_title_inflation(c)
        if is_inflated and weight == 0.0:
            continue
            
        clean_candidates.append(c)
        if len(sample_candidates) < 5000:
            sample_candidates.append(c)
            
    print(f"Clean candidates remaining: {len(clean_candidates)}")
    
    # 3. Dynamic Adaptive Pool Size Sizing
    print("Running adaptive retrieval sizing...")
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
    
    lexical_pool_size = 1500 if avg_lexical < 0.2 else 800
    title_pool_size = 1200 if pct_title > 0.1 else 500
    concept_pool_size = 1000
    behavioral_pool_size = 1000
    
    # 4. Extract Multi-Source Retrieval Scores
    scored_candidates = []
    for c in clean_candidates:
        cid = c.get("candidate_id")
        
        lex_score = compute_lexical_score(c)["lexical_score"]
        title_score = compute_title_alignment(c)["career_title_alignment"]
        
        p = c.get("profile", {})
        career = c.get("career_history", [])
        desc_text = " ".join([p.get("summary", "")] + [role.get("description", "") for role in career]).lower()
        concept_terms = ["retrieval", "search", "ranking", "recommendation", "matching", "personalization", "ndcg", "map", "mrr"]
        concept_score = sum(1 for term in concept_terms if term in desc_text)
        
        behavior_score = compute_behavioral_score(c)["behavior_multiplier"]
        
        scored_candidates.append({
            "candidate_id": cid,
            "candidate": c,
            "lexical": float(lex_score),
            "title": float(title_score),
            "concept": float(concept_score),
            "behavioral": float(behavior_score)
        })
        
    # Pool selection
    top_lexical = sorted(scored_candidates, key=lambda x: -x["lexical"])[:lexical_pool_size]
    top_title = sorted(scored_candidates, key=lambda x: -x["title"])[:title_pool_size]
    top_concept = sorted(scored_candidates, key=lambda x: -x["concept"])[:concept_pool_size]
    top_behavioral = sorted(scored_candidates, key=lambda x: -x["behavioral"])[:behavioral_pool_size]
    
    union_ids = set()
    union_pool = []
    
    for pool in [top_lexical, top_title, top_concept, top_behavioral]:
        for item in pool:
            cid = item["candidate_id"]
            if cid not in union_ids:
                union_ids.add(cid)
                composite = (0.4 * item["lexical"] + 
                             0.3 * item["title"] + 
                             0.2 * item["concept"] + 
                             0.1 * item["behavioral"])
                item["composite"] = composite
                union_pool.append(item)
                
    # Truncate union pool to 3000
    union_pool.sort(key=lambda x: -x["composite"])
    selected_pool = [item["candidate"] for item in union_pool[:3000]]
    print(f"Adaptive union pool size: {len(selected_pool)} candidates passed to ranker.")
    
    # 5. Score and rank selected pool
    print("Scoring and ranking candidates using learning-to-rank model...")
    try:
        ranked_list = score_and_rank_candidates(selected_pool)
        print(f"Scored and ranked {len(ranked_list)} candidates.")
    except Exception as e:
        print(f"Error during scoring/ranking: {e}", file=sys.stderr)
        sys.exit(1)
        
    # 6. Filter to top 100 and generate dynamic justifications
    top_100 = ranked_list[:100]
    print("Generating evidence-based justifications...")
    
    output_rows = []
    for item in top_100:
        rank = item["rank"]
        score = item["score"]
        candidate = item["candidate"]
        cid = item["candidate_id"]
        
        reasoning = generate_reasoning_string(candidate, rank, score, item["breakdown"])
        
        output_rows.append({
            "candidate_id": cid,
            "rank": rank,
            "score": score,
            "reasoning": reasoning
        })
        
    # 7. Write to submission CSV
    print(f"Writing output CSV to {args.out}...")
    try:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"], quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            for row in output_rows:
                writer.writerow(row)
        print("CSV written successfully.")
    except Exception as e:
        print(f"Error writing CSV file: {e}", file=sys.stderr)
        sys.exit(1)
        
    elapsed_time = time.perf_counter() - start_time
    print(f"Ranking step completed in {elapsed_time:.2f} seconds.")
    
    # 8. Self-validate submission compliance
    print("Validating generated submission file...")
    validation_errors = validate_submission(args.out)
    if validation_errors:
        print("Validation FAILED! Please inspect errors:", file=sys.stderr)
        for err in validation_errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)
    else:
        print("Submission validation PASSED. File format is 100% compliant.")

if __name__ == "__main__":
    main()
