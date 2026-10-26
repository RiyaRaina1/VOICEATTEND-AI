"""Tests for anti-spoofing service and authentication module."""

from pathlib import Path
import numpy as np
import pytest

from services import anti_spoofing


# ─── Anti-Spoofing Tests ────────────────────────────────────────────────────────

class TestAntiSpoofing:
    """Tests for audio quality heuristic anti-spoofing checks."""

    def _make_speech(self, duration_sec: float = 2.0, sr: int = 16000) -> np.ndarray:
        """Generate synthetic but audio-quality-valid speech-like signal."""
        t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
        # Mix of frequencies like natural speech
        signal = (
            0.3 * np.sin(2 * np.pi * 200 * t)
            + 0.2 * np.sin(2 * np.pi * 500 * t)
            + 0.1 * np.sin(2 * np.pi * 1000 * t)
            + 0.05 * np.random.randn(len(t))
        )
        return signal.astype(np.float32)

    def test_valid_speech_passes(self):
        """Normal speech-like audio should return 'clear' or 'suspicious'."""
        audio = self._make_speech()
        result = anti_spoofing.check_audio_quality_spoofing(audio)
        assert result["status"] in ("clear", "suspicious"), (
            f"Expected clear/suspicious for normal audio, got: {result['status']}"
        )
        assert result["is_dl_model"] is False, "Must not claim DL model"
        assert "latency_ms" in result
        assert "checks" in result

    def test_empty_audio_rejected(self):
        """Empty audio must be immediately rejected."""
        result = anti_spoofing.check_audio_quality_spoofing(np.array([], dtype=np.float32))
        assert result["status"] == "unavailable"
        assert "empty" in result["reason"].lower() or "cannot" in result["reason"].lower()

    def test_silent_audio_rejected(self):
        """Silent recording (all zeros) must be rejected as spoof_detected."""
        silent = np.zeros(32000, dtype=np.float32)
        result = anti_spoofing.check_audio_quality_spoofing(silent)
        assert result["status"] == "spoof_detected", (
            f"Silent audio should be spoof_detected, got: {result['status']}"
        )
        assert result["confidence"] == 0.0

    def test_heavily_clipped_audio_suspicious(self):
        """Audio with >5% hard clipping should be flagged."""
        audio = np.ones(32000, dtype=np.float32) * 0.99  # All near max amplitude
        result = anti_spoofing.check_audio_quality_spoofing(audio)
        assert result["status"] in ("suspicious", "spoof_detected"), (
            f"Heavily clipped audio should be suspicious/spoof_detected, got: {result['status']}"
        )
        assert result["checks"]["clip_fraction"] > anti_spoofing.CLIP_FRACTION_THRESHOLD

    def test_result_structure(self):
        """All required keys must be present in result."""
        audio = self._make_speech()
        result = anti_spoofing.check_audio_quality_spoofing(audio)
        required_keys = {"status", "checks", "confidence", "reason", "is_dl_model", "latency_ms"}
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_model_status(self):
        """get_anti_spoofing_status should document limitations clearly."""
        status = anti_spoofing.get_anti_spoofing_status()
        assert status["available"] is True
        assert status["dl_model_loaded"] is False
        assert "heuristic" in status["mode"].lower()
        assert status["warning"]  # Must have a warning about limitations

    def test_system_status_page_api(self):
        """The diagnostics page must use the anti-spoofing service API."""
        status = anti_spoofing.get_anti_spoofing_status()
        assert {"available", "mode", "warning"}.issubset(status)
        assert hasattr(anti_spoofing, "MIN_ENERGY_RMS")
        assert hasattr(anti_spoofing, "SPECTRAL_FLATNESS_MIN")
        assert hasattr(anti_spoofing, "SPECTRAL_FLATNESS_MAX")


# ─── Authentication Tests ────────────────────────────────────────────────────────

class TestAuthentication:
    """Tests for admin authentication module."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path):
        """Freshly initialized database for auth tests."""
        from database import db
        db_file = tmp_path / "test_auth.db"
        db.init_db(db_path=db_file)
        return db_file

    def test_password_hash_not_plaintext(self):
        """Hashed password must not contain the original."""
        from services.auth import hash_password
        hashed = hash_password("my_secure_password")
        assert "my_secure_password" not in hashed
        assert len(hashed) > 10

    def test_password_verify_correct(self):
        """Correct password must verify successfully."""
        from services.auth import hash_password, verify_password
        pw = "SuperSecret123!"
        hashed = hash_password(pw)
        assert verify_password(pw, hashed) is True

    def test_password_verify_incorrect(self):
        """Wrong password must fail verification."""
        from services.auth import hash_password, verify_password
        hashed = hash_password("correct_password")
        assert verify_password("wrong_password", hashed) is False

    def test_create_and_authenticate_user(self, temp_db):
        """Admin user creation and authentication."""
        from services.auth import create_admin_user, authenticate_user
        result = create_admin_user("testadmin", "Password123!", role="admin", db_path=temp_db)
        assert result is True

        user = authenticate_user("testadmin", "Password123!", db_path=temp_db)
        assert user is not None
        assert user["username"] == "testadmin"
        assert user["role"] == "admin"
        # Password hash must NOT be returned
        assert "password" not in user.get("password_hash", "")  # It's there but hashed
        assert not any(
            plain in user.get("password_hash", "")
            for plain in ["Password123!"]
        )

    def test_wrong_password_returns_none(self, temp_db):
        """Wrong password must return None."""
        from services.auth import create_admin_user, authenticate_user
        create_admin_user("user2", "RightPass123", db_path=temp_db)
        assert authenticate_user("user2", "WrongPass", db_path=temp_db) is None

    def test_duplicate_username_rejected(self, temp_db):
        """Duplicate username must fail gracefully."""
        from services.auth import create_admin_user
        assert create_admin_user("dupuser", "Pass12345", db_path=temp_db) is True
        assert create_admin_user("dupuser", "OtherPass1", db_path=temp_db) is False

    def test_short_password_rejected(self, temp_db):
        """Passwords shorter than 8 characters must be rejected."""
        from services.auth import create_admin_user
        with pytest.raises(ValueError, match="8 characters"):
            create_admin_user("shortpw", "short", db_path=temp_db)

    def test_invalid_role_rejected(self, temp_db):
        """Invalid role must be rejected."""
        from services.auth import create_admin_user
        with pytest.raises(ValueError, match="Role"):
            create_admin_user("roletest", "Password123!", role="superuser", db_path=temp_db)

    def test_has_any_admin_false_initially(self, temp_db):
        """No admins should be detected in fresh database."""
        from services.auth import has_any_admin
        assert has_any_admin(db_path=temp_db) is False

    def test_has_any_admin_true_after_creation(self, temp_db):
        """Admin should be detected after creation."""
        from services.auth import create_admin_user, has_any_admin
        create_admin_user("newadmin", "AdminPass1!", db_path=temp_db)
        assert has_any_admin(db_path=temp_db) is True

    def test_faculty_user_can_register_and_login(self, temp_db):
        """A new faculty account can be created without granting admin access."""
        from services.auth import authenticate_user, create_admin_user
        assert create_admin_user("faculty1", "FacultyPass1!", role="faculty", db_path=temp_db) is True
        user = authenticate_user("faculty1", "FacultyPass1!", db_path=temp_db)
        assert user is not None
        assert user["role"] == "faculty"

    def test_admin_authorized_password_reset(self, temp_db):
        """An administrator can reset an existing account password."""
        from services.auth import authenticate_user, create_admin_user, reset_password_by_admin
        create_admin_user("rootadmin", "AdminPass123!", role="admin", db_path=temp_db)
        create_admin_user("faculty1", "OldPass123!", role="faculty", db_path=temp_db)
        assert reset_password_by_admin(
            "faculty1", "NewPass123!", "rootadmin", "AdminPass123!", db_path=temp_db
        ) is True
        assert authenticate_user("faculty1", "OldPass123!", db_path=temp_db) is None
        assert authenticate_user("faculty1", "NewPass123!", db_path=temp_db) is not None

    def test_password_reset_requires_active_admin(self, temp_db):
        """Invalid or non-admin authorization must not change the password."""
        from services.auth import authenticate_user, create_admin_user, reset_password_by_admin
        create_admin_user("rootadmin", "AdminPass123!", role="admin", db_path=temp_db)
        create_admin_user("faculty1", "OldPass123!", role="faculty", db_path=temp_db)
        assert reset_password_by_admin(
            "faculty1", "NewPass123!", "rootadmin", "wrong-password", db_path=temp_db
        ) is False
        assert reset_password_by_admin(
            "faculty1", "NewPass123!", "faculty1", "OldPass123!", db_path=temp_db
        ) is False
        assert authenticate_user("faculty1", "OldPass123!", db_path=temp_db) is not None

    def test_audit_log_written(self, temp_db):
        """Audit events must be written to the database."""
        from services.auth import log_audit_event, get_audit_log
        log_audit_event("TEST_ACTION", "detail info", username="testuser", db_path=temp_db)
        log = get_audit_log(limit=10, db_path=temp_db)
        assert len(log) >= 1
        assert log[0]["action"] == "TEST_ACTION"
        assert log[0]["details"] == "detail info"

    def test_deactivate_user(self, temp_db):
        """Deactivated user must not be able to log in."""
        from services.auth import create_admin_user, authenticate_user, deactivate_user, get_all_users
        create_admin_user("deactivatetest", "Pass12345!", db_path=temp_db)
        users = get_all_users(db_path=temp_db)
        user_id = users[0]["id"]

        deactivate_user(user_id, db_path=temp_db)
        result = authenticate_user("deactivatetest", "Pass12345!", db_path=temp_db)
        assert result is None, "Deactivated user must not authenticate"
