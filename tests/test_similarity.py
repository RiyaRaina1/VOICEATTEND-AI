"""Unit tests for cosine similarity, vector normalization, and speaker identification."""

import numpy as np

try:
    import pytest
except ImportError:
    class _Approx:
        def __init__(self, val, abs=1e-5):
            self.val = val
            self.abs = abs
        def __eq__(self, other):
            return abs(self.val - other) <= self.abs

    class PytestFallback:
        @staticmethod
        def approx(val, abs=1e-5):
            return _Approx(val, abs)

    pytest = PytestFallback()

from speaker import embedding_store, identification


def test_cosine_similarity_properties():
    """Verify mathematical properties of cosine similarity."""
    # Identical vectors -> similarity = 1.0
    v1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    assert pytest.approx(identification.calculate_cosine_similarity(v1, v1), abs=1e-5) == 1.0

    # Orthogonal vectors -> similarity = 0.0
    v_ortho1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_ortho2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert pytest.approx(identification.calculate_cosine_similarity(v_ortho1, v_ortho2), abs=1e-5) == 0.0

    # Opposing vectors -> similarity = -1.0
    v_opp = -v1
    assert pytest.approx(identification.calculate_cosine_similarity(v1, v_opp), abs=1e-5) == -1.0

    # Zero vector handling -> 0.0 (safe division)
    v_zero = np.zeros(4, dtype=np.float32)
    assert identification.calculate_cosine_similarity(v1, v_zero) == 0.0


def test_vector_normalization():
    """Verify L2 normalization produces unit norm vector."""
    v = np.array([3.0, 4.0], dtype=np.float32)
    normed = embedding_store.normalize_vector(v)
    assert pytest.approx(np.linalg.norm(normed), abs=1e-5) == 1.0
    assert pytest.approx(normed[0], abs=1e-5) == 0.6
    assert pytest.approx(normed[1], abs=1e-5) == 0.8


def test_profile_averaging():
    """Verify that multiple enrollment samples are aggregated and normalized."""
    s1 = np.array([1.0, 0.0], dtype=np.float32)
    s2 = np.array([0.0, 1.0], dtype=np.float32)

    samples = [
        {"embedding_blob": embedding_store.embedding_to_bytes(s1)},
        {"embedding_blob": embedding_store.embedding_to_bytes(s2)},
    ]

    profile = embedding_store.compute_student_profile(samples)
    assert profile is not None
    assert pytest.approx(np.linalg.norm(profile), abs=1e-5) == 1.0
    # Mean of [1, 0] and [0, 1] is [0.5, 0.5], normalized is [1/sqrt(2), 1/sqrt(2)]
    expected = 1.0 / np.sqrt(2.0)
    assert pytest.approx(profile[0], abs=1e-5) == expected
    assert pytest.approx(profile[1], abs=1e-5) == expected


def test_speaker_identification_acceptance_and_rejection():
    """Verify identification ranks candidates, accepts matches >= threshold, and rejects unknown matches."""
    # Synthetic 4-dimensional normalized student profiles
    p_alice = embedding_store.normalize_vector(np.array([1.0, 0.1, 0.0, 0.0], dtype=np.float32))
    p_bob = embedding_store.normalize_vector(np.array([0.0, 0.1, 1.0, 0.0], dtype=np.float32))

    enrolled = {
        1: {
            "student_id": 1,
            "student_code": "STU_ALICE",
            "name": "Alice",
            "profile_embedding": p_alice,
            "sample_count": 3,
        },
        2: {
            "student_id": 2,
            "student_code": "STU_BOB",
            "name": "Bob",
            "profile_embedding": p_bob,
            "sample_count": 3,
        },
    }

    # Query 1: Alice speaking (close to p_alice)
    query_alice = embedding_store.normalize_vector(np.array([0.98, 0.15, 0.0, 0.0], dtype=np.float32))
    result = identification.identify_speaker(query_alice, enrolled, threshold=0.72)

    assert result["matched"] is True
    assert result["decision"] == "accepted"
    assert result["best_candidate"]["student_id"] == 1
    assert result["best_candidate"]["name"] == "Alice"
    assert result["best_candidate"]["score"] > 0.95
    assert len(result["ranked_candidates"]) == 2

    # Query 2: Unknown speaker (distinct orthogonal direction)
    query_charlie = embedding_store.normalize_vector(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    result_unk = identification.identify_speaker(query_charlie, enrolled, threshold=0.72)

    assert result_unk["matched"] is False
    assert result_unk["decision"] == "rejected"
    assert "did not meet threshold" in result_unk["reason"]

    # Query 3: Empty enrollment dictionary
    result_empty = identification.identify_speaker(query_alice, {}, threshold=0.72)
    assert result_empty["matched"] is False
    assert result_empty["decision"] == "no_candidates"
