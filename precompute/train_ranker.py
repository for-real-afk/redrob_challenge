"""
Decoupled model training script. Loads precomputed embeddings and candidate features,
generates pairwise preference consensus, trains the XGBRanker, and exports reports.
"""
import os
import json
import numpy as np
import pandas as pd
import time
from xgboost import XGBRanker
from src.features.feature_store import extract_features
from src.features.semantic import compute_jd_embedding

def load_cached_data():
    embeddings_path = r"D:\projects\redrob\precompute\artifacts\embeddings.npz"
    features_path = r"D:\projects\redrob\precompute\artifacts\cached_features.json"
    
    if not os.path.exists(embeddings_path) or not os.path.exists(features_path):
        raise FileNotFoundError("Embeddings or cached features not found. Run build_embeddings.py first.")
        
    data = np.load(embeddings_path)
    embeddings = data["embeddings"]
    candidate_ids = data["candidate_ids"]
    id_to_idx = {cid: idx for idx, cid in enumerate(candidate_ids)}
    
    with open(features_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)
        
    return embeddings, candidate_ids, id_to_idx, candidates

def compute_semantic_scores(embeddings, id_to_idx):
    jd_vector = compute_jd_embedding()
    jd_norm = np.linalg.norm(jd_vector)
    matrix_norms = np.linalg.norm(embeddings, axis=1)
    matrix_norms[matrix_norms == 0] = 1e-9
    
    similarities = np.dot(embeddings, jd_vector) / (matrix_norms * jd_norm)
    return similarities

def recruiter_fit_generator(f_a, f_b):
    score_a = f_a["lexical_score"] + f_a["concept_coverage"] - f_a["experience_alignment"] * 0.1
    score_b = f_b["lexical_score"] + f_b["concept_coverage"] - f_b["experience_alignment"] * 0.1
    return 1 if score_a > score_b + 0.05 else (-1 if score_b > score_a + 0.05 else 0)

def hidden_gem_generator(f_a, f_b):
    score_a = f_a["hidden_gem_score"]
    score_b = f_b["hidden_gem_score"]
    return 1 if score_a > score_b + 0.1 else (-1 if score_b > score_a + 0.1 else 0)

def systems_thinking_generator(f_a, f_b):
    score_a = f_a["systems_thinking_score"]
    score_b = f_b["systems_thinking_score"]
    return 1 if score_a > score_b + 0.1 else (-1 if score_b > score_a + 0.1 else 0)

def title_alignment_generator(f_a, f_b):
    score_a = f_a["title_interaction"]
    score_b = f_b["title_interaction"]
    return 1 if score_a > score_b + 0.1 else (-1 if score_b > score_a + 0.1 else 0)

def production_ml_generator(f_a, f_b):
    score_a = f_a["retrieval_experience_x_production_ml"]
    score_b = f_b["retrieval_experience_x_production_ml"]
    return 1 if score_a > score_b + 0.1 else (-1 if score_b > score_a + 0.1 else 0)

def run_preference_pipeline(features_list, threshold=3):
    """
    Compares candidate pairs and counts preference votes.
    Establishes preference if votes >= threshold.
    """
    n = len(features_list)
    borda_scores = np.zeros(n)
    
    generators = [
        recruiter_fit_generator,
        hidden_gem_generator,
        systems_thinking_generator,
        title_alignment_generator,
        production_ml_generator
    ]
    
    total_pairs = 0
    agreed_pairs = 0
    generator_wins = [0] * len(generators)
    covered_indices = set()
    
    # Compare a representative set of pairs to audit and build targets
    # (comparing all pairs for 3000 candidates is 4.5M comparisons, so we sample pairs)
    np.random.seed(42)
    sample_indices = np.random.choice(n, size=min(n, 1000), replace=False)
    
    for idx_i, i in enumerate(sample_indices):
        for j in sample_indices[idx_i + 1:]:
            total_pairs += 1
            f_a = features_list[i]
            f_b = features_list[j]
            
            votes = 0
            decisions = []
            for idx_g, gen in enumerate(generators):
                dec = gen(f_a, f_b)
                decisions.append(dec)
                if dec == 1:
                    votes += 1
                elif dec == -1:
                    votes -= 1
                    
            # Check agreement threshold
            if votes >= threshold:
                borda_scores[i] += 1.0
                agreed_pairs += 1
                covered_indices.add(i)
                covered_indices.add(j)
                for idx_g, dec in enumerate(decisions):
                    if dec == 1:
                        generator_wins[idx_g] += 1
            elif votes <= -threshold:
                borda_scores[j] += 1.0
                agreed_pairs += 1
                covered_indices.add(i)
                covered_indices.add(j)
                for idx_g, dec in enumerate(decisions):
                    if dec == -1:
                        generator_wins[idx_g] += 1
                        
    pair_rate = agreed_pairs / total_pairs if total_pairs > 0 else 0.0
    unique_candidate_coverage = len(covered_indices) / len(sample_indices) if len(sample_indices) > 0 else 0.0
    return borda_scores, pair_rate, agreed_pairs, generator_wins, unique_candidate_coverage

def main():
    start_time = time.time()
    
    print("Step 1: Loading precomputed embeddings and features...")
    embeddings, candidate_ids, id_to_idx, candidates = load_cached_data()
    
    print("Step 2: Calculating semantic similarities...")
    similarities = compute_semantic_scores(embeddings, id_to_idx)
    
    print("Step 3: Extracting candidate features...")
    features_list = []
    for c in candidates:
        cid = c.get("candidate_id")
        idx = id_to_idx[cid]
        sem_score = similarities[idx]
        
        feats = extract_features(c, sem_score)
        feats["candidate_id"] = cid
        features_list.append(feats)
        
    # Step 4: Generating pairwise preference consensus
    print("Step 4: Running pairwise preference consensus pipeline...")
    threshold = 3
    borda_scores, pair_rate, num_pairs, generator_wins, unique_coverage = run_preference_pipeline(features_list, threshold=threshold)
    
    print(f"Agreement Rate at threshold={threshold}: {pair_rate:.2%} ({num_pairs} pairs, Unique coverage: {unique_coverage:.2%}).")
    
    # Adaptive threshold fallback if pairs are too sparse
    if pair_rate < 0.05:
        threshold = 2
        print(f"Warning: Pair rate is under 5%. Falling back to threshold={threshold}")
        borda_scores, pair_rate, num_pairs, generator_wins, unique_coverage = run_preference_pipeline(features_list, threshold=threshold)
        print(f"Fallback Agreement Rate at threshold={threshold}: {pair_rate:.2%} ({num_pairs} pairs, Unique coverage: {unique_coverage:.2%}).")
        
    # Audit Generator Influence
    total_wins = sum(generator_wins) or 1
    generator_names = ["Recruiter Fit", "Hidden Gem", "Systems Thinking", "Title Alignment", "Production ML"]
    influence_summary = []
    print("\nGenerator Influence Audit:")
    for name, wins in zip(generator_names, generator_wins):
        pct = wins / total_wins
        print(f"  - {name}: {pct:.2%} ({wins} wins)")
        influence_summary.append(f"| {name} | {pct:.2%} | {wins} |")
        
    # Normalizing target y into 5 grades [0, 4]
    max_borda = np.max(borda_scores) if np.max(borda_scores) > 0 else 1.0
    normalized_scores = borda_scores / max_borda
    y = np.digitize(normalized_scores, bins=[0.2, 0.4, 0.6, 0.8])
    
    # Prepare training dataframe
    feature_keys = [k for k in features_list[0].keys() if k != "candidate_id"]
    X_data = []
    for feats in features_list:
        X_data.append([feats[k] for k in feature_keys])
        
    X_df = pd.DataFrame(X_data, columns=feature_keys)
    
    # Step 5: Training XGBRanker
    print("\nStep 5: Training XGBRanker learning-to-rank model...")
    # For a single query ranking, group size is the entire pool
    group = [len(X_df)]
    
    ranker = XGBRanker(
        objective="rank:pairwise",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    ranker.fit(X_df, y, group=group)
    print("XGBRanker training complete.")
    
    # Save Model
    model_dir = r"D:\projects\redrob\precompute\artifacts"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "ranker.bin")
    ranker.save_model(model_path)
    print(f"Trained model saved to {model_path}")
    
    # Step 6: Export Feature Importance Report
    print("Step 6: Exporting Feature Importance Report...")
    importances = ranker.feature_importances_
    imp_df = pd.DataFrame({
        "Feature": feature_keys,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    
    report_content = f"""# Feature Importance Report
This report details the relative influence of the engineered features in the trained XGBRanker model.

## Model Training Hyperparameters
*   **Objective:** `rank:ndcg`
*   **Estimators:** `300`
*   **Max Depth:** `6`
*   **Learning Rate:** `0.05`

## Generator Influence Summary
This summary shows how much each generator contributed to the pairwise preferences during training:

| Generator | Relative Influence | Agreed Wins |
| :--- | :--- | :--- |
{"\n".join(influence_summary)}

## Feature Weights Summary
The relative contribution (Gain) of each candidate, interaction, and query feature in the ranking decision:

"""
    for idx, row in imp_df.iterrows():
        report_content += f"*   **{row['Feature']}**: `{row['Importance']:.4f}`\n"
        
    report_path = r"D:\projects\redrob\feature_importance_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"Feature Importance Report written to {report_path}")
    
    # Step 7: Failure Analysis Generation
    print("Step 7: Exporting Failure Analysis...")
    # Predict scores and sort candidates
    preds = ranker.predict(X_df)
    sorted_idx = np.argsort(-preds)
    
    # Let's inspect potential edge cases (false positives, false negatives)
    false_positives = []
    false_negatives = []
    
    # False Positive simulation: keyword matchers with lower concept/domain scores that ranked in top 50
    # False Negative simulation: high-value titles with low lexical/semantic scores ranked below 100
    for rank, idx in enumerate(sorted_idx[:50]):
        cand = candidates[idx]
        feats = features_list[idx]
        concept_keys = [k for k in feats.keys() if k.startswith("concept_score_")]
        concept_val = sum(feats[k] for k in concept_keys) / len(concept_keys) if concept_keys else 0.0
        if feats["lexical_score"] > 0.7 and concept_val < 0.2:
            false_positives.append(f"Rank {rank+1} | {cand.get('profile', {}).get('current_title')} with {feats['lexical_score']:.2f} lexical but low concept score")
            
    for rank, idx in enumerate(sorted_idx[100:300]):
        cand = candidates[idx]
        feats = features_list[idx]
        if feats["title_interaction"] == 1.0 and feats["semantic_score"] < 0.4:
            false_negatives.append(f"Rank {rank+101} | {cand.get('profile', {}).get('current_title')} with strong title alignment but lower semantic similarity")
            
    # Audit Generator Influence for report
    influence_bullets = []
    for name, wins in zip(generator_names, generator_wins):
        pct = wins / total_wins
        influence_bullets.append(f"  - **{name}:** {pct:.2%} ({wins} wins)")
    influence_str = "\n".join(influence_bullets)

    fa_content = f"""# Failure Analysis Report
Analysis of expected failure modes, false positives, false negatives, ambiguous ranking cases, and synthetic label diagnostics.

## Synthetic Pairwise Preference Label Diagnostics
- **Pair Volume (`num_pairs_generated`):** {num_pairs}
- **Pair Diversity (`unique_candidate_coverage`):** {unique_coverage:.2%} (percentage of unique candidate profiles covered by agreed preference labels)
- **Generator Influence Audit:**
{influence_str}

## Expected Failure Modes
1.  **High-Keyword / Low-Context Profiles (False Positives):** Candidates with strong keyword list overlaps but minimal description context (such as prompt engineers stuffing tools, e.g. listing "RAG, LLM" with no actual production impact).
2.  **Great Engineers / Poor Documentation (False Negatives):** Core ranking and systems engineers who did not write verbose descriptions in their resumes, resulting in low semantic and lexical scores despite having high real-world capability.
3.  **Domain Ambiguity:** Search vs. Recommendation systems engineers. Both profiles are highly suitable but are ranked relative to minor lexical differences or specific ontology matching.

## Simulated False Positives in Candidate Pool
{"\n".join([f"*   {fp}" for fp in false_positives[:10]]) or "*   No significant false positives detected."}

## Simulated False Negatives in Candidate Pool
{"\n".join([f"*   {fn}" for fn in false_negatives[:10]]) or "*   No significant false negatives detected."}
"""
    fa_path = r"D:\projects\redrob\failure_analysis.md"
    with open(fa_path, "w", encoding="utf-8") as f:
        f.write(fa_content)
    print(f"Failure Analysis Report written to {fa_path}")
    
    elapsed = time.time() - start_time
    print(f"Ranking model training completed in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
