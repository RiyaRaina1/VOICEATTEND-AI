"""Attendance and enrollment business logic service for VoiceAttend.

Coordinates the end-to-end workflows:
- Student-specific voice sample recording, validation, embedding extraction, and persistence.
- Live attendance verification with structured pipeline:
    1. Session validation
    2. Audio preprocessing & quality validation
    3. Anti-spoofing / heuristic liveness check
    4. NLP command verification (faster-whisper)
    5. Speaker embedding extraction (SpeechBrain ECAPA-TDNN)
    6. Identity matching (cosine similarity)
    7. Ambiguous match handling
    8. Duplicate attendance prevention
    9. Database persistence

Returns structured results with decision, scores, latency, and rejection reasons.
Fail-safe: unavailable models are documented in the result — NEVER silently pass.
"""

import io
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import config
from audio import preprocessing, recorder
from database import db
from services import anti_spoofing
from speaker import embedding_store, identification, model_service
from speech import command_verifier

logger = logging.getLogger(__name__)

# Minimum cosine margin between top-1 and top-2 to avoid ambiguous match rejection.
# If margin < this, the match is considered ambiguous and rejected.
AMBIGUOUS_MARGIN_THRESHOLD = float(os.getenv("VOICEATTEND_AMBIGUOUS_MARGIN", "0.05"))


class ServiceError(Exception):
    """Raised for business logic errors during enrollment or attendance."""
    pass


# ==============================================================================
# Enrollment Service
# ==============================================================================


def enroll_student_sample(
    student_id: int,
    audio_source: Union[bytes, io.BytesIO, str, Path],
    db_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Enroll a new voice sample for a specific student.

    CRITICAL WORKFLOW:
    1. Look up student by primary key.
    2. Check that max sample limit is not exceeded.
    3. Validate and normalize the audio.
    4. Extract speaker embedding using the pretrained model.
    5. Save the audio file with a deterministic student identifier.
    6. Save the embedding to the database linked to the student.
    7. Atomically roll back (delete audio file) if embedding extraction or DB insert fails.

    Returns:
        Dict containing enrollment summary and updated sample counts.
    """
    # 1. Verify student exists
    student = db.get_student_by_id(student_id, db_path=db_path)
    if not student:
        raise ServiceError(f"Student with ID {student_id} does not exist.")

    existing_samples = db.get_voice_samples_for_student(student_id, db_path=db_path)
    current_count = len(existing_samples)

    if current_count >= config.MAX_ENROLLMENT_SAMPLES:
        raise ServiceError(
            f"Student {student['name']} already has {current_count} samples "
            f"(maximum allowed: {config.MAX_ENROLLMENT_SAMPLES}). "
            "Please delete an existing sample before adding a new one."
        )

    next_sample_number = current_count + 1
    saved_path: Optional[Path] = None

    try:
        # 2. Validate and save audio
        saved_path, audio_arr, duration = recorder.process_and_save_enrollment_audio(
            audio_source=audio_source,
            student_id=student_id,
            sample_number=next_sample_number,
        )

        # 3. Extract speaker embedding
        embedding = model_service.extract_embedding(audio_arr)
        embedding_blob = embedding_store.embedding_to_bytes(embedding)

        # 4. Insert into database
        model_info = model_service.get_model_info()
        model_id = str(model_info.get("source", config.SPEAKER_MODEL_SOURCE))

        sample_id = db.add_voice_sample(
            student_id=student_id,
            audio_path=str(saved_path),
            embedding_blob=embedding_blob,
            sample_number=next_sample_number,
            model_identifier=model_id,
            db_path=db_path,
        )

        total_samples = current_count + 1
        is_complete = total_samples >= config.DEFAULT_ENROLLMENT_TARGET

        return {
            "success": True,
            "sample_id": sample_id,
            "sample_number": next_sample_number,
            "student_id": student_id,
            "student_name": student["name"],
            "student_code": student["student_code"],
            "audio_path": str(saved_path),
            "duration": round(duration, 2),
            "total_samples": total_samples,
            "target_samples": config.DEFAULT_ENROLLMENT_TARGET,
            "is_complete": is_complete,
            "message": (
                f"Sample {next_sample_number} enrolled for {student['name']}. "
                f"({total_samples}/{config.DEFAULT_ENROLLMENT_TARGET} required samples)"
            ),
        }

    except Exception as e:
        # Cleanup audio file on failure so orphan files are not left on disk
        if saved_path and saved_path.exists():
            try:
                saved_path.unlink()
            except OSError:
                pass
        logger.exception("Enrollment failed.")
        raise ServiceError(f"Enrollment failed: {e}")


def delete_student_sample(sample_id: int, db_path: Optional[Path] = None) -> bool:
    """Delete an individual voice sample and associated audio file."""
    return db.delete_voice_sample(sample_id, delete_file=True, db_path=db_path)


# ==============================================================================
# Live Attendance Service
# ==============================================================================


def process_live_attendance(
    audio_source: Union[bytes, io.BytesIO, str, Path],
    session_id: int,
    similarity_threshold: float = config.DEFAULT_SIMILARITY_THRESHOLD,
    db_path: Optional[Path] = None,
    enable_anti_spoofing: bool = True,
    anti_spoofing_strict: bool = False,
) -> Dict[str, Any]:
    """Execute the end-to-end live attendance verification pipeline.

    Pipeline Steps:
    1. Session check: Ensure the attendance session exists and is active.
    2. Audio preprocessing & validation: Check audio duration, signal energy, and normalize.
    3. Anti-spoofing check: Heuristic audio quality spoofing detection.
    4. NLP command verification: Verify student said "Present" (independent of speaker ID).
    5. Speaker embedding extraction: Generate 192-d normalized embedding.
    6. Identity matching: Compare against all enrolled student profiles using cosine similarity.
    7. Ambiguity check: Reject if margin between top-1 and top-2 is too small.
    8. Duplicate attendance check: Prevent marking the same student twice in the session.
    9. Database persistence: Record attendance timestamp, score, and decision.

    Args:
        audio_source: Raw audio bytes, file path, or numpy array.
        session_id: ID of the active attendance session.
        similarity_threshold: Cosine similarity cutoff for speaker acceptance.
        db_path: Optional override for SQLite database path.
        enable_anti_spoofing: Whether to run anti-spoofing checks.
        anti_spoofing_strict: If True, reject on 'suspicious' (not just 'spoof_detected').

    Returns:
        Comprehensive outcome dict with:
            - status: str (success | error | command_failed | speaker_rejected | ...)
            - decision: str (accepted | rejected | duplicate)
            - message: str
            - pipeline: Dict of per-step results and latencies
            - total_latency_ms: float
    """
    pipeline_start = time.time()
    pipeline: Dict[str, Any] = {
        "session_check": None,
        "audio_preprocessing": None,
        "anti_spoofing": None,
        "command_verification": None,
        "speaker_embedding": None,
        "speaker_identification": None,
        "duplicate_check": None,
        "database_write": None,
    }

    # ── Step 1: Session validation ────────────────────────────────────────────
    t0 = time.time()
    session = db.get_session_by_id(session_id, db_path=db_path)
    pipeline["session_check"] = {
        "ok": session is not None,
        "session_id": session_id,
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }

    if not session:
        return _build_result(
            status="error",
            decision="rejected",
            message=f"Attendance session {session_id} not found.",
            pipeline=pipeline,
            pipeline_start=pipeline_start,
        )

    if session.get("status") != "active":
        return _build_result(
            status="session_closed",
            decision="rejected",
            message=f"Session '{session['session_name']}' is closed. Cannot mark attendance.",
            pipeline=pipeline,
            pipeline_start=pipeline_start,
        )

    # ── Step 2: Audio preprocessing & validation ──────────────────────────────
    t0 = time.time()
    try:
        audio_arr, duration = recorder.process_attendance_audio(audio_source)
        pipeline["audio_preprocessing"] = {
            "ok": True,
            "duration_sec": round(duration, 2),
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }
    except preprocessing.AudioValidationError as e:
        pipeline["audio_preprocessing"] = {
            "ok": False,
            "error": str(e),
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }
        return _build_result(
            status="invalid_audio",
            decision="rejected",
            message=str(e),
            pipeline=pipeline,
            pipeline_start=pipeline_start,
        )
    except Exception as e:
        pipeline["audio_preprocessing"] = {
            "ok": False,
            "error": str(e),
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }
        return _build_result(
            status="error",
            decision="rejected",
            message=f"Failed to process attendance audio: {e}",
            pipeline=pipeline,
            pipeline_start=pipeline_start,
        )

    # ── Step 3: Anti-spoofing check ───────────────────────────────────────────
    if enable_anti_spoofing:
        t0 = time.time()
        spoof_result = anti_spoofing.check_audio_quality_spoofing(audio_arr)
        pipeline["anti_spoofing"] = spoof_result

        if spoof_result["status"] == "spoof_detected":
            return _build_result(
                status="spoof_detected",
                decision="rejected",
                message=(
                    f"Anti-spoofing check FAILED: {spoof_result['reason']}"
                ),
                pipeline=pipeline,
                anti_spoofing=spoof_result,
                pipeline_start=pipeline_start,
            )

        if anti_spoofing_strict and spoof_result["status"] == "suspicious":
            return _build_result(
                status="spoof_suspicious",
                decision="rejected",
                message=(
                    "Strict anti-spoofing: suspicious audio rejected. "
                    f"Reason: {spoof_result['reason']}"
                ),
                pipeline=pipeline,
                anti_spoofing=spoof_result,
                pipeline_start=pipeline_start,
            )
    else:
        pipeline["anti_spoofing"] = {"status": "skipped", "ok": True}

    # ── Step 4: Speech command verification ───────────────────────────────────
    t0 = time.time()
    command_result = command_verifier.verify_command(
        audio_source=audio_arr,
        required_command=config.DEFAULT_ATTENDANCE_COMMAND,
        approved_variants=config.APPROVED_COMMAND_VARIANTS,
    )
    command_result["latency_ms"] = round((time.time() - t0) * 1000, 2)
    pipeline["command_verification"] = command_result

    if not command_result["verified"]:
        return _build_result(
            status="command_failed",
            decision="rejected",
            message=(
                f"Attendance command verification failed: {command_result['reason']}. "
                "The student must say 'Present' clearly."
            ),
            pipeline=pipeline,
            command_result=command_result,
            pipeline_start=pipeline_start,
        )

    # ── Step 5: Model availability and speaker embedding ──────────────────────
    if not model_service.is_available():
        info = model_service.get_model_info()
        pipeline["speaker_embedding"] = {
            "ok": False,
            "error": info.get("error", "Model unavailable"),
        }
        return _build_result(
            status="model_unavailable",
            decision="rejected",
            message=f"Speaker recognition model unavailable: {info.get('error')}",
            pipeline=pipeline,
            command_result=command_result,
            pipeline_start=pipeline_start,
        )

    t0 = time.time()
    try:
        query_embedding = model_service.extract_embedding(audio_arr)
        pipeline["speaker_embedding"] = {
            "ok": True,
            "embedding_dim": len(query_embedding),
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }
    except Exception as e:
        pipeline["speaker_embedding"] = {
            "ok": False,
            "error": str(e),
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }
        return _build_result(
            status="embedding_failed",
            decision="rejected",
            message=f"Failed to extract speaker embedding: {e}",
            pipeline=pipeline,
            command_result=command_result,
            pipeline_start=pipeline_start,
        )

    # ── Step 6: Retrieve enrolled profiles & identify speaker ──────────────────
    t0 = time.time()
    enrolled_profiles = embedding_store.get_all_enrolled_profiles(db_path=db_path)
    if not enrolled_profiles:
        pipeline["speaker_identification"] = {
            "ok": False,
            "error": "No enrolled profiles",
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }
        return _build_result(
            status="no_enrolled_students",
            decision="rejected",
            message="No students have completed voice enrollment in the database.",
            pipeline=pipeline,
            command_result=command_result,
            pipeline_start=pipeline_start,
        )

    id_result = identification.identify_speaker(
        query_embedding=query_embedding,
        enrolled_profiles=enrolled_profiles,
        threshold=similarity_threshold,
    )
    id_result["latency_ms"] = round((time.time() - t0) * 1000, 2)
    pipeline["speaker_identification"] = id_result

    if not id_result["matched"] or id_result["best_candidate"] is None:
        return _build_result(
            status="speaker_rejected",
            decision="rejected",
            message=id_result["reason"],
            pipeline=pipeline,
            command_result=command_result,
            identification=id_result,
            pipeline_start=pipeline_start,
        )

    # ── Step 7: Ambiguity margin check ────────────────────────────────────────
    margin = id_result.get("margin", 1.0)
    if len(enrolled_profiles) > 1 and margin < AMBIGUOUS_MARGIN_THRESHOLD:
        best = id_result["best_candidate"]
        return _build_result(
            status="ambiguous_match",
            decision="rejected",
            message=(
                f"Ambiguous speaker match: top candidate {best['name']} "
                f"(score={best['score']:.4f}) is too close to second candidate "
                f"(margin={margin:.4f} < {AMBIGUOUS_MARGIN_THRESHOLD}). "
                "Please re-record with better audio quality."
            ),
            pipeline=pipeline,
            command_result=command_result,
            identification=id_result,
            pipeline_start=pipeline_start,
        )

    best = id_result["best_candidate"]
    student_id = best["student_id"]
    score = best["score"]

    # ── Step 8: Duplicate attendance prevention ────────────────────────────────
    t0 = time.time()
    already_marked = db.is_student_marked(
        student_id=student_id, session_id=session_id, db_path=db_path
    )
    pipeline["duplicate_check"] = {
        "ok": not already_marked,
        "already_marked": already_marked,
        "latency_ms": round((time.time() - t0) * 1000, 2),
    }

    if already_marked:
        return _build_result(
            status="already_marked",
            decision="duplicate",
            message=(
                f"Duplicate attendance: {best['name']} ({best['student_code']}) "
                f"is already marked present in session '{session['session_name']}'."
            ),
            pipeline=pipeline,
            student=best,
            command_result=command_result,
            identification=id_result,
            pipeline_start=pipeline_start,
        )

    # ── Step 9: Persist attendance record ─────────────────────────────────────
    t0 = time.time()
    try:
        record_id = db.mark_attendance(
            student_id=student_id,
            session_id=session_id,
            similarity_score=score,
            command_verified=True,
            decision="accepted",
            db_path=db_path,
        )
        pipeline["database_write"] = {
            "ok": True,
            "record_id": record_id,
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }
    except Exception as e:
        pipeline["database_write"] = {
            "ok": False,
            "error": str(e),
            "latency_ms": round((time.time() - t0) * 1000, 2),
        }
        return _build_result(
            status="db_error",
            decision="rejected",
            message=f"Database error recording attendance: {e}",
            pipeline=pipeline,
            command_result=command_result,
            identification=id_result,
            pipeline_start=pipeline_start,
        )

    return _build_result(
        status="success",
        decision="accepted",
        message=(
            f"Attendance recorded for {best['name']} ({best['student_code']}). "
            f"Similarity: {score:.3f} | Command: '{command_result['command_matched']}'"
        ),
        pipeline=pipeline,
        student=best,
        similarity_score=score,
        threshold=similarity_threshold,
        command_result=command_result,
        identification=id_result,
        record_id=record_id,
        session_name=session["session_name"],
        pipeline_start=pipeline_start,
    )


def _build_result(
    status: str,
    decision: str,
    message: str,
    pipeline: Dict[str, Any],
    pipeline_start: float,
    **kwargs,
) -> Dict[str, Any]:
    """Build a standardized result dictionary."""
    total_latency_ms = round((time.time() - pipeline_start) * 1000, 2)
    result = {
        "status": status,
        "decision": decision,
        "message": message,
        "pipeline": pipeline,
        "total_latency_ms": total_latency_ms,
    }
    result.update(kwargs)
    return result
