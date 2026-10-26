"""Speaker embedding storage, serialization, and profile aggregation.

Provides safe binary serialization of NumPy embeddings into SQLite BLOBs,
deserialization, and aggregation of multiple enrollment embeddings into
a normalized student voice profile.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

from database import db


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Safely serialize float32 embedding vector to raw bytes for database storage.
    
    Avoids pickle / eval risks by using exact contiguous float32 binary format.
    """
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.flatten()
    return arr.tobytes()


def bytes_to_embedding(blob: bytes) -> np.ndarray:
    """Deserialize binary BLOB back into 1D float32 NumPy array."""
    if not blob:
        raise ValueError("Cannot deserialize empty embedding blob.")
    return np.frombuffer(blob, dtype=np.float32)


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a 1D vector to unit length."""
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return vec
    return (vec / norm).astype(np.float32)


def compute_student_profile(samples: List[Dict]) -> Optional[np.ndarray]:
    """Compute an aggregated, L2-normalized voice profile from a list of sample records.
    
    Args:
        samples: List of dictionary records from database voice_samples table.
        
    Returns:
        1D float32 normalized mean embedding array, or None if no samples.
    """
    embeddings = []
    for sample in samples:
        blob = sample.get("embedding_blob")
        if blob:
            emb = bytes_to_embedding(blob)
            embeddings.append(normalize_vector(emb))

    if not embeddings:
        return None

    # Compute centroid (mean vector) and re-normalize to unit hypersphere
    mean_vec = np.mean(embeddings, axis=0)
    return normalize_vector(mean_vec)


def get_all_enrolled_profiles(db_path=None) -> Dict[int, Dict]:
    """Retrieve enrolled voice profiles for all students.
    
    Returns:
        Dict mapping student_id to {
            "student_id": int,
            "student_code": str,
            "name": str,
            "sample_count": int,
            "profile_embedding": np.ndarray,
            "individual_embeddings": List[np.ndarray]
        }
    """
    all_samples = db.get_all_active_voice_samples(db_path=db_path)
    students_map: Dict[int, Dict] = {}

    for sample in all_samples:
        sid = sample["student_id"]
        if sid not in students_map:
            students_map[sid] = {
                "student_id": sid,
                "student_code": sample["student_code"],
                "name": sample["student_name"],
                "samples": [],
            }
        students_map[sid]["samples"].append(sample)

    profiles: Dict[int, Dict] = {}
    for sid, data in students_map.items():
        samples = data["samples"]
        profile_emb = compute_student_profile(samples)
        if profile_emb is not None:
            profiles[sid] = {
                "student_id": sid,
                "student_code": data["student_code"],
                "name": data["name"],
                "sample_count": len(samples),
                "profile_embedding": profile_emb,
                "individual_embeddings": [
                    bytes_to_embedding(s["embedding_blob"]) for s in samples
                ],
            }

    return profiles
