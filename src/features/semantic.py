"""
Semantic feature matching using precomputed SentenceTransformer embeddings.
"""
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Global cache to prevent multiple loads of the embedding matrix
_EMBEDDINGS_CACHE = None

# Hardcoded Job Description semantic representation (fixed JD)
JD_SEMANTIC_ANCHOR = (
    "Senior AI Engineer, machine learning, deep learning. Deployed embeddings-based retrieval systems, "
    "vector databases, hybrid search (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS). "
    "Strong Python software engineering, production scale. Experience designing evaluation frameworks (NDCG, MRR, MAP, A/B testing). "
    "Hands-on ML product focus, building search, ranking, retrieval, or recommendation systems. Noida or Pune India."
)

def load_precomputed_embeddings():
    """
    Loads precomputed embeddings from disk and caches them in memory.
    """
    global _EMBEDDINGS_CACHE
    if _EMBEDDINGS_CACHE is not None:
        return _EMBEDDINGS_CACHE
        
    artifacts_path = r"D:\projects\redrob\precompute\artifacts\embeddings.npz"
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(
            f"Precomputed embeddings not found at {artifacts_path}. "
            "Please run `python precompute/build_embeddings.py` first."
        )
        
    data = np.load(artifacts_path)
    embeddings = data["embeddings"]
    candidate_ids = data["candidate_ids"]
    
    # Store in cache as mapping from candidate_id to row index and the raw matrix
    id_to_idx = {cid: idx for idx, cid in enumerate(candidate_ids)}
    _EMBEDDINGS_CACHE = (embeddings, id_to_idx)
    return _EMBEDDINGS_CACHE

def compute_jd_embedding():
    """
    Computes sentence embedding for the fixed Job Description requirements.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(JD_SEMANTIC_ANCHOR, convert_to_numpy=True)

def compute_all_semantic_scores():
    """
    Computes semantic similarity scores for all candidates at once.
    Returns a dict mapping candidate_id to similarity score.
    """
    embeddings, id_to_idx = load_precomputed_embeddings()
    jd_vector = compute_jd_embedding()
    
    # Normalize vectors for cosine similarity
    jd_norm = np.linalg.norm(jd_vector)
    matrix_norms = np.linalg.norm(embeddings, axis=1)
    
    # Avoid divide-by-zero
    matrix_norms[matrix_norms == 0] = 1e-9
    
    # Calculate similarities: dot product divided by norms
    similarities = np.dot(embeddings, jd_vector) / (matrix_norms * jd_norm)
    
    # Map back to candidate IDs
    scores = {}
    for cid, idx in id_to_idx.items():
        # Scale to 0-1 range
        scores[cid] = float(similarities[idx])
        
    return scores
