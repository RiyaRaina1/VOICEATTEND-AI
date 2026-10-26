"""Unit tests for VoiceAttend SQLite database operations and constraints."""

import sqlite3
from pathlib import Path
import numpy as np

try:
    import pytest
except ImportError:
    class _RaisesContext:
        def __init__(self, exc):
            self.exc = exc
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                raise AssertionError(f"Expected exception {self.exc} was not raised.")
            return issubclass(exc_type, self.exc)

    class PytestFallback:
        @staticmethod
        def raises(exc):
            return _RaisesContext(exc)
        @staticmethod
        def fixture(func):
            return func

    pytest = PytestFallback()

from database import db
from speaker.embedding_store import embedding_to_bytes, bytes_to_embedding


@pytest.fixture
def temp_db(tmp_path: Path):
    """Fixture providing a freshly initialized SQLite database in a temporary directory."""
    db_file = tmp_path / "test_voiceattend.db"
    db.init_db(db_path=db_file)
    return db_file


def test_student_crud(temp_db):
    """Test creating, reading, updating, and searching students."""
    # Create student
    sid = db.create_student("STU001", "Alice Smith", "MCA A", db_path=temp_db)
    assert sid > 0

    # Get by ID
    student = db.get_student_by_id(sid, db_path=temp_db)
    assert student is not None
    assert student["student_code"] == "STU001"
    assert student["name"] == "Alice Smith"
    assert student["class_section"] == "MCA A"
    assert student["sample_count"] == 0

    # Get by Code
    by_code = db.get_student_by_code("STU001", db_path=temp_db)
    assert by_code["id"] == sid

    # Update
    updated = db.update_student(sid, "Alice Johnson", "MCA B", db_path=temp_db)
    assert updated is True
    student = db.get_student_by_id(sid, db_path=temp_db)
    assert student["name"] == "Alice Johnson"
    assert student["class_section"] == "MCA B"

    # Search
    results = db.search_students("Alice", db_path=temp_db)
    assert len(results) == 1
    assert results[0]["id"] == sid

    results = db.search_students("STU001", db_path=temp_db)
    assert len(results) == 1


def test_duplicate_student_code_rejection(temp_db):
    """Test that duplicate student_code raises sqlite3.IntegrityError."""
    db.create_student("STU_DUP", "Original Student", db_path=temp_db)
    with pytest.raises(sqlite3.IntegrityError):
        db.create_student("STU_DUP", "Duplicate Student", db_path=temp_db)


def test_voice_samples_storage_and_cascade_delete(temp_db, tmp_path):
    """Test adding voice samples, binary embedding persistence, and cascade deletion."""
    sid = db.create_student("STU002", "Bob Vance", db_path=temp_db)

    # Create dummy audio file
    dummy_wav = tmp_path / "dummy.wav"
    dummy_wav.write_bytes(b"RIFFdummydata")

    # Embedding serialization
    raw_emb = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    emb_blob = embedding_to_bytes(raw_emb)

    sample_id = db.add_voice_sample(
        student_id=sid,
        audio_path=str(dummy_wav),
        embedding_blob=emb_blob,
        sample_number=1,
        model_identifier="test-model",
        db_path=temp_db,
    )
    assert sample_id > 0

    # Verify student sample count
    student = db.get_student_by_id(sid, db_path=temp_db)
    assert student["sample_count"] == 1

    samples = db.get_voice_samples_for_student(sid, db_path=temp_db)
    assert len(samples) == 1
    assert samples[0]["sample_number"] == 1
    # Verify embedding blob deserialization
    retrieved_emb = bytes_to_embedding(samples[0]["embedding_blob"])
    assert np.allclose(raw_emb, retrieved_emb)

    # Delete single sample
    deleted_sample = db.delete_voice_sample(sample_id, delete_file=True, db_path=temp_db)
    assert deleted_sample is True
    assert not dummy_wav.exists()

    # Re-add sample and test cascading deletion on student delete
    dummy_wav2 = tmp_path / "dummy2.wav"
    dummy_wav2.write_bytes(b"RIFFdummydata2")
    db.add_voice_sample(
        student_id=sid,
        audio_path=str(dummy_wav2),
        embedding_blob=emb_blob,
        sample_number=1,
        model_identifier="test-model",
        db_path=temp_db,
    )

    db.delete_student(sid, delete_audio_files=True, db_path=temp_db)
    assert db.get_student_by_id(sid, db_path=temp_db) is None
    assert not dummy_wav2.exists()


def test_attendance_sessions_and_duplicate_prevention(temp_db):
    """Test session creation and duplicate attendance constraint within the same session."""
    sess_id = db.create_attendance_session("Morning Session", db_path=temp_db)
    assert sess_id > 0

    active_sess = db.get_active_session(db_path=temp_db)
    assert active_sess["id"] == sess_id

    sid1 = db.create_student("STU_ATT1", "Student One", db_path=temp_db)
    sid2 = db.create_student("STU_ATT2", "Student Two", db_path=temp_db)

    # Check initially unmarked
    assert not db.is_student_marked(sid1, sess_id, db_path=temp_db)
    unmarked = db.get_unmarked_students(sess_id, db_path=temp_db)
    assert len(unmarked) == 2

    # Mark attendance for sid1
    rec_id = db.mark_attendance(
        student_id=sid1,
        session_id=sess_id,
        similarity_score=0.88,
        command_verified=True,
        decision="accepted",
        db_path=temp_db,
    )
    assert rec_id > 0
    assert db.is_student_marked(sid1, sess_id, db_path=temp_db)

    # Remaining unmarked should now be 1
    unmarked = db.get_unmarked_students(sess_id, db_path=temp_db)
    assert len(unmarked) == 1
    assert unmarked[0]["id"] == sid2

    # Attempt duplicate marking in same session -> Must raise IntegrityError
    with pytest.raises(sqlite3.IntegrityError):
        db.mark_attendance(
            student_id=sid1,
            session_id=sess_id,
            similarity_score=0.89,
            command_verified=True,
            decision="accepted",
            db_path=temp_db,
        )

    # Close session
    closed = db.close_session(sess_id, db_path=temp_db)
    assert closed is True
    assert db.get_active_session(db_path=temp_db) is None
