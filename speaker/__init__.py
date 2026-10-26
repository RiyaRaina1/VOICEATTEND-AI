"""Speaker recognition package for VoiceAttend."""
from .model_service import (
    load_model,
    extract_embedding,
    is_available,
    get_model_info,
)
from .embedding_store import (
    embedding_to_bytes,
    bytes_to_embedding,
    normalize_vector,
    compute_student_profile,
    get_all_enrolled_profiles,
)
from .identification import (
    calculate_cosine_similarity,
    identify_speaker,
)

__all__ = [
    "load_model",
    "extract_embedding",
    "is_available",
    "get_model_info",
    "embedding_to_bytes",
    "bytes_to_embedding",
    "normalize_vector",
    "compute_student_profile",
    "get_all_enrolled_profiles",
    "calculate_cosine_similarity",
    "identify_speaker",
]
