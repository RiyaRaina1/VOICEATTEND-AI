"""Unit and integration tests for speech command verification and attendance service logic."""

from pathlib import Path
import numpy as np

try:
    import pytest
except ImportError:
    class PytestFallback:
        @staticmethod
        def fixture(func):
            return func

    pytest = PytestFallback()

from audio import preprocessing
from database import db
from services import attendance_service
from speech import command_verifier


def test_transcribe_uses_only_supported_keywords():
    """Regression test for faster-whisper's removed metadata_errors argument."""
    captured = {}

    class FakeModel:
        def transcribe(self, audio_path, *, language=None, beam_size=5):
            captured.update(
                audio_path=audio_path,
                language=language,
                beam_size=beam_size,
            )
            return [], None

    command_verifier._transcribe_audio(FakeModel(), "sample.wav")

    assert captured == {
        "audio_path": "sample.wav",
        "language": "en",
        "beam_size": 3,
    }


@pytest.fixture
def temp_db(tmp_path: Path):
    db_file = tmp_path / "test_attendance.db"
    db.init_db(db_path=db_file)
    return db_file


def test_command_text_normalization():
    """Verify that speech text normalization removes punctuation, lowercases, and trims."""
    assert command_verifier.normalize_text("Present, Sir!") == "present sir"
    assert command_verifier.normalize_text("  Present   MA'AM  ") == "present ma am"
    assert command_verifier.normalize_text("Here!") == "here"
    assert command_verifier.normalize_text("") == ""


def test_audio_validation_constraints():
    """Verify audio duration and silence validation."""
    sr = 16000
    # Empty audio
    valid, msg = preprocessing.validate_audio(np.zeros(0, dtype=np.float32), sr=sr)
    assert not valid
    assert "empty" in msg.lower()

    # Audio too short (< 1.0s)
    short_audio = np.random.randn(int(0.5 * sr)).astype(np.float32) * 0.1
    valid, msg = preprocessing.validate_audio(short_audio, sr=sr, min_duration=1.0)
    assert not valid
    assert "too short" in msg.lower()

    # Audio too long (> 6.0s)
    long_audio = np.random.randn(int(7.0 * sr)).astype(np.float32) * 0.1
    valid, msg = preprocessing.validate_audio(long_audio, sr=sr, max_duration=6.0)
    assert not valid
    assert "too long" in msg.lower()

    # Pure silence (RMS ~ 0)
    silent_audio = np.zeros(int(2.0 * sr), dtype=np.float32)
    valid, msg = preprocessing.validate_audio(silent_audio, sr=sr)
    assert not valid
    assert "silent" in msg.lower() or "quiet" in msg.lower()

    # Valid audio (2.0s with sufficient signal energy)
    t = np.linspace(0, 2.0, int(2.0 * sr), endpoint=False)
    good_audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    valid, msg = preprocessing.validate_audio(good_audio, sr=sr)
    assert valid
    assert msg == ""


def test_live_attendance_validation_and_session_rejection(temp_db):
    """Verify attendance pipeline rejects nonexistent sessions and invalid audio."""
    # Test nonexistent session
    res = attendance_service.process_live_attendance(
        audio_source=np.zeros(0, dtype=np.float32),
        session_id=9999,
        db_path=temp_db,
    )
    assert res["status"] == "error"
    assert "not found" in res["message"].lower()

    # Test invalid audio (empty)
    sess_id = db.create_attendance_session("Test Session", db_path=temp_db)
    res_audio = attendance_service.process_live_attendance(
        audio_source=np.zeros(0, dtype=np.float32),
        session_id=sess_id,
        db_path=temp_db,
    )
    assert res_audio["status"] == "invalid_audio"
    assert res_audio["decision"] == "rejected"
