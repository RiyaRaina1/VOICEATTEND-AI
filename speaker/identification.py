"""Speaker identification and similarity ranking module.

Calculates cosine similarity between query embeddings and enrolled student profiles,
ranks candidates, applies confidence thresholds, and rejects uncertain matches.
"""

from typing import Any, Dict, List, Optional
import numpy as np

import config
from speaker.embedding_store import normalize_vector


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two 1D NumPy vectors.
    
    Formula:
        sim(u, v) = (u . v) / (||u|| * ||v||)
        
    Returns:
        float bounded in [-1.0, 1.0].
    """
    v1 = np.asarray(vec1, dtype=np.float32).flatten()
    v2 = np.asarray(vec2, dtype=np.float32).flatten()

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-9 or norm2 < 1e-9:
        return 0.0

    dot = float(np.dot(v1, v2))
    sim = dot / (norm1 * norm2)
    return float(np.clip(sim, -1.0, 1.0))


def identify_speaker(
    query_embedding: np.ndarray,
    enrolled_profiles: Dict[int, Dict[str, Any]],
    threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD,
) -> Dict[str, Any]:
    """Compare query embedding against all enrolled profiles and return match decision.
    
    Args:
        query_embedding: 1D float32 speaker embedding of test recording.
        enrolled_profiles: Dict of student profiles from embedding_store.
        threshold: Cosine similarity acceptance cutoff.
        
    Returns:
        Dict with keys:
            - matched: bool (True if top candidate >= threshold)
            - best_candidate: Dict of top student info or None
            - decision: 'accepted' | 'rejected' | 'no_candidates'
            - threshold: float
            - margin: float (score difference between 1st and 2nd candidate)
            - ranked_candidates: List of sorted candidates with scores
            - reason: Human-readable explanation string
    """
    if not enrolled_profiles:
        return {
            "matched": False,
            "best_candidate": None,
            "decision": "no_candidates",
            "threshold": threshold,
            "margin": 0.0,
            "ranked_candidates": [],
            "reason": "No enrolled student voice profiles found in database.",
        }

    q_norm = normalize_vector(query_embedding)
    ranked: List[Dict[str, Any]] = []

    for sid, profile in enrolled_profiles.items():
        profile_emb = profile.get("profile_embedding")
        if profile_emb is None:
            continue

        score = calculate_cosine_similarity(q_norm, profile_emb)
        ranked.append({
            "student_id": sid,
            "student_code": profile["student_code"],
            "name": profile["name"],
            "sample_count": profile.get("sample_count", 0),
            "score": round(score, 4),
        })

    if not ranked:
        return {
            "matched": False,
            "best_candidate": None,
            "decision": "no_candidates",
            "threshold": threshold,
            "margin": 0.0,
            "ranked_candidates": [],
            "reason": "No valid profile embeddings available for comparison.",
        }

    # Sort descending by similarity score
    ranked.sort(key=lambda x: x["score"], reverse=True)
    best = ranked[0]
    best_score = best["score"]

    margin = 0.0
    if len(ranked) > 1:
        margin = round(best_score - ranked[1]["score"], 4)

    if best_score >= threshold:
        decision = "accepted"
        matched = True
        reason = (
            f"Speaker verified as {best['name']} ({best['student_code']}) "
            f"with similarity score {best_score:.4f} (threshold: {threshold:.2f})."
        )
    else:
        decision = "rejected"
        matched = False
        reason = (
            f"Unknown or uncertain speaker. Top match {best['name']} score {best_score:.4f} "
            f"did not meet threshold {threshold:.2f} (gap: {threshold - best_score:.4f})."
        )

    return {
        "matched": matched,
        "best_candidate": best,
        "decision": decision,
        "threshold": threshold,
        "margin": margin,
        "ranked_candidates": ranked,
        "reason": reason,
    }
