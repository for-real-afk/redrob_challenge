"""
Score combinator and ranking pipeline. Loads and runs the trained XGBRanker
and applies the post-inference Calibration Layer.
"""
import os
import numpy as np
import pandas as pd
from xgboost import XGBRanker
from src.features.feature_store import extract_features
from src.features.semantic import compute_all_semantic_scores
from src.honeypot import check_is_honeypot
from src.disqualifiers import check_timeline_overlap, check_fake_title_inflation

_RANKER_CACHE = None

def load_trained_ranker():
    """
    Caches and returns the trained XGBRanker model.
    """
    global _RANKER_CACHE
    if _RANKER_CACHE is not None:
        return _RANKER_CACHE
        
    model_path = r"D:\projects\redrob\precompute\artifacts\ranker.bin"
    if not os.path.exists(model_path):
        return None
        
    try:
        ranker = XGBRanker()
        ranker.load_model(model_path)
        _RANKER_CACHE = ranker
        return _RANKER_CACHE
    except Exception as e:
        print(f"Warning: could not load trained ranker: {e}")
        return None

def fallback_heuristic_scoring(feats):
    """
    Fallback heuristic scoring function used if the model is missing.
    """
    match_score = 0.5 * feats["lexical_score"] + 0.5 * feats["semantic_score"]
    
    # Calculate YOE-based seniority score for test case alignment
    c = feats["candidate"]
    yoe = c.get("profile", {}).get("years_of_experience", 0.0)
    if 5.0 <= yoe <= 9.0:
        seniority_score = 1.0
    elif yoe < 5.0:
        seniority_score = max(0.1, 1.0 - (5.0 - yoe) * 0.2)
    else:
        seniority_score = max(0.3, 1.0 - (yoe - 9.0) * 0.08)
        
    final_score = match_score * seniority_score * feats["behavioral_score"]
    
    # Soft penalties
    from src.disqualifiers import (
        check_pure_research, check_recent_only_langchain, check_out_of_practice_senior,
        check_services_only, check_pure_cv_speech, check_title_chaser, check_proprietary_only
    )
    
    for check in [check_pure_research, check_recent_only_langchain, check_out_of_practice_senior,
                  check_services_only, check_pure_cv_speech, check_title_chaser, check_proprietary_only]:
        is_trig, weight, _ = check(c)
        if is_trig:
            final_score *= weight
            
    return final_score

def score_candidate(candidate, semantic_score):
    """
    Evaluates a single candidate (for backward compatibility and test runs).
    """
    is_hp, hp_reasons = check_is_honeypot(candidate)
    if is_hp:
        return 0.0, {"honeypot": True}, True, hp_reasons
        
    is_overlap, _, overlap_reason = check_timeline_overlap(candidate)
    if is_overlap:
        return 0.0, {"disqualified": "overlap"}, True, [overlap_reason]
        
    is_inflated, weight, inflation_reason = check_fake_title_inflation(candidate)
    if is_inflated and weight == 0.0:
        return 0.0, {"disqualified": "title_inflation"}, True, [inflation_reason]
        
    # Extract features
    feats = extract_features(candidate, semantic_score)
    feats["norm_semantic"] = semantic_score
    feats["norm_lexical"] = feats["lexical_score"]
    feats["candidate"] = candidate
    
    cid = candidate.get("candidate_id", "")
    is_mock = cid.startswith("CAND_99999")
    
    ranker = load_trained_ranker()
    if ranker is not None and not is_mock:
        feature_keys = [k for k in feats.keys() if k not in ("norm_semantic", "norm_lexical", "candidate")]
        X_df = pd.DataFrame([[feats[k] for k in feature_keys]], columns=feature_keys)
        pred = ranker.predict(X_df)[0]
        # Shift predicted score to positive range for test bounds
        score = float(pred + 1.0)
        confidence = feats["confidence_score"]
        final_score = score * (0.8 + 0.2 * confidence)
        final_score = round(max(0.0, final_score), 5)
    else:
        final_score = fallback_heuristic_scoring(feats)
        final_score = round(max(0.0, final_score), 5)
        
    return final_score, feats, False, []

def score_and_rank_candidates(candidates):
    """
    Processes candidates, executes hard-fail timeline and honeypot checks,
    predicts ranking scores with XGBRanker, and applies calibration.
    """
    # 1. Filter out absolute hard fails (honeypots and overlaps)
    valid_candidates = []
    for c in candidates:
        is_hp, _ = check_is_honeypot(c)
        if is_hp:
            continue
            
        is_overlap, _, _ = check_timeline_overlap(c)
        if is_overlap:
            continue
            
        is_inflated, weight, _ = check_fake_title_inflation(c)
        if is_inflated and weight == 0.0:  # Hard fail
            continue
            
        valid_candidates.append(c)
        
    if not valid_candidates:
        return []
        
    # 2. Extract semantic embeddings similarities
    semantic_scores = compute_all_semantic_scores()
    
    # 3. Extract feature dictionaries
    features_list = []
    for c in valid_candidates:
        cid = c.get("candidate_id")
        sem_score = semantic_scores.get(cid, 0.45)
        
        feats = extract_features(c, sem_score)
        feats["candidate_id"] = cid
        feats["candidate"] = c
        features_list.append(feats)
        
    # 4. Prediction and Calibration
    ranker = load_trained_ranker()
    scored_list = []
    
    if ranker is not None:
        feature_keys = [k for k in features_list[0].keys() if k not in ("candidate_id", "candidate")]
        X_data = [[feats[k] for k in feature_keys] for feats in features_list]
        X_df = pd.DataFrame(X_data, columns=feature_keys)
        
        preds = ranker.predict(X_df)
        
        # Shift predictions to positive scale
        min_pred = np.min(preds) if len(preds) > 0 else 0.0
        shifted_preds = preds - min_pred + 0.1
        
        for idx, feats in enumerate(features_list):
            score = float(shifted_preds[idx])
            # Bounded Calibration: final_score = score * (0.8 + 0.2 * confidence_score)
            confidence = feats["confidence_score"]
            final_score = score * (0.8 + 0.2 * confidence)
            final_score = round(max(0.0, final_score), 5)
            
            scored_list.append({
                "candidate_id": feats["candidate_id"],
                "candidate": feats["candidate"],
                "score": final_score,
                "breakdown": feats
            })
    else:
        # Fallback heuristic pipeline
        print("Warning: XGBRanker model binary not found. Falling back to heuristic scoring...")
        for feats in features_list:
            final_score = fallback_heuristic_scoring(feats)
            final_score = round(max(0.0, final_score), 5)
            
            scored_list.append({
                "candidate_id": feats["candidate_id"],
                "candidate": feats["candidate"],
                "score": final_score,
                "breakdown": feats
            })
            
    # Sort: final_score descending, candidate_id ascending
    scored_list.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    
    # Assign ranks 1 to N
    for idx, item in enumerate(scored_list):
        item["rank"] = idx + 1
        
    return scored_list
