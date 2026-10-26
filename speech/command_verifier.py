"""Speech-command verification (NLP) module for VoiceAttend.

Transcribes audio using a lightweight, offline-capable model (faster-whisper 'tiny' on CPU),
normalizes the transcription, and verifies whether the spoken phrase matches
the required attendance command ("present" or approved variants).
"""

import logging
import inspect
import re
import string
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

import config
from audio.preprocessing import load_audio, save_wav

logger = logging.getLogger(__name__)

# Cached model instance
_WHISPER_MODEL = None
_MODEL_LOAD_ERROR: Optional[str] = None
_PYAV_COMPATIBILITY_PATCHED = False


def _patch_pyav_compatibility() -> None:
    """Handle PyAV releases that removed ``metadata_errors`` from ``av.open``.

    faster-whisper 1.2.x still passes this option while decoding audio.  The
    compatibility wrapper retries without it only when PyAV rejects that
    specific keyword, preserving all other decoder errors.
    """
    global _PYAV_COMPATIBILITY_PATCHED
    if _PYAV_COMPATIBILITY_PATCHED:
        return

    import av

    original_open = av.open

    def open_compatible(*args, **kwargs):
        try:
            return original_open(*args, **kwargs)
        except TypeError as error:
            if "metadata_errors" not in kwargs or "metadata_errors" not in str(error):
                raise
            compatible_kwargs = dict(kwargs)
            compatible_kwargs.pop("metadata_errors", None)
            return original_open(*args, **compatible_kwargs)

    av.open = open_compatible
    _PYAV_COMPATIBILITY_PATCHED = True


def _transcribe_audio(model: Any, input_path: str):
    """Transcribe using only options supported by the installed faster-whisper API.

    faster-whisper has changed optional keyword arguments across releases.  In
    particular, ``metadata_errors`` is not accepted by current releases and
    causes the whole attendance verification to fail.  Filtering the options
    against the bound method signature keeps this call compatible with both
    older and newer installations.
    """
    requested_options = {
        "language": "en",
        "beam_size": 3,
        "vad_filter": True,
        "vad_parameters": {"min_silence_duration_ms": 500},
    }
    signature = inspect.signature(model.transcribe)
    accepts_arbitrary_keywords = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_arbitrary_keywords:
        supported_options = requested_options
    else:
        supported_options = {
            name: value
            for name, value in requested_options.items()
            if name in signature.parameters
        }
    return model.transcribe(input_path, **supported_options)


def normalize_text(text: str) -> str:
    """Normalize transcribed text: lowercase, remove punctuation, collapse whitespace.
    
    Example:
        "Present, Sir!" -> "present sir"
    """
    if not text:
        return ""
    # Lowercase
    normalized = text.lower()
    # Replace punctuation with space
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    normalized = normalized.translate(translator)
    # Collapse multiple whitespaces
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def load_command_model():
    """Load faster-whisper tiny model on CPU with caching."""
    global _WHISPER_MODEL, _MODEL_LOAD_ERROR
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL

    try:
        from faster_whisper import WhisperModel
        _patch_pyav_compatibility()

        model_dir = str(config.WHISPER_MODEL_DIR)
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading faster-whisper model '{config.WHISPER_MODEL_SIZE}' on CPU...")
        _WHISPER_MODEL = WhisperModel(
            model_size_or_path=config.WHISPER_MODEL_SIZE,
            device="cpu",
            compute_type="int8",
            download_root=model_dir,
        )
        _MODEL_LOAD_ERROR = None
        logger.info("Speech command model loaded successfully.")
        return _WHISPER_MODEL
    except ImportError:
        _MODEL_LOAD_ERROR = (
            "faster-whisper is not installed. Please install it with: "
            "pip install faster-whisper"
        )
        logger.warning(_MODEL_LOAD_ERROR)
        return None
    except Exception as e:
        _MODEL_LOAD_ERROR = f"Failed to load speech command model: {e}"
        logger.error(_MODEL_LOAD_ERROR)
        return None


def is_available() -> bool:
    """Check if the speech command model is loaded or loadable."""
    if _WHISPER_MODEL is not None:
        return True
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def get_model_status() -> Dict[str, Any]:
    """Return speech-command model diagnostics and status."""
    return {
        "available": is_available() and (_MODEL_LOAD_ERROR is None),
        "loaded": _WHISPER_MODEL is not None,
        "model_name": f"faster-whisper-{config.WHISPER_MODEL_SIZE}",
        "device": "cpu (int8 quantized)",
        "download_dir": str(config.WHISPER_MODEL_DIR),
        "error": _MODEL_LOAD_ERROR,
    }


def verify_command(
    audio_source: Union[str, Path, np.ndarray, bytes],
    required_command: str = config.DEFAULT_ATTENDANCE_COMMAND,
    approved_variants: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Transcribe audio and verify attendance command.
    
    Args:
        audio_source: File path, raw bytes, or float32 NumPy array.
        required_command: Target command (default 'present').
        approved_variants: List of acceptable variations.
        
    Returns:
        Dict with keys:
            - verified: bool
            - raw_transcription: str
            - normalized_transcription: str
            - confidence: float
            - command_matched: str or None
            - reason: str
    """
    if approved_variants is None:
        approved_variants = config.APPROVED_COMMAND_VARIANTS

    norm_required = normalize_text(required_command)
    norm_variants = {normalize_text(v) for v in approved_variants}
    norm_variants.add(norm_required)

    model = load_command_model()
    if model is None:
        error_msg = _MODEL_LOAD_ERROR or "Speech recognition model is unavailable."
        return {
            "verified": False,
            "raw_transcription": "",
            "normalized_transcription": "",
            "confidence": 0.0,
            "command_matched": None,
            "reason": f"Model unavailable: {error_msg}",
        }

    temp_wav_path: Optional[Path] = None
    try:
        if isinstance(audio_source, (str, Path)):
            input_path = str(audio_source)
        else:
            # Convert in-memory bytes or NumPy array to temporary WAV file
            if isinstance(audio_source, np.ndarray):
                audio_arr = audio_source
            else:
                audio_arr, _ = load_audio(audio_source)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                temp_wav_path = Path(tf.name)
            save_wav(audio_arr, temp_wav_path)
            input_path = str(temp_wav_path)

        # Transcribe with faster-whisper
        segments, info = _transcribe_audio(model, input_path)

        segment_texts = []
        confidences = []
        for segment in segments:
            segment_texts.append(segment.text)
            if hasattr(segment, "avg_logprob") and segment.avg_logprob is not None:
                # Convert log probability to approximate confidence (0.0 to 1.0)
                conf = float(np.exp(segment.avg_logprob))
                confidences.append(conf)

        raw_text = " ".join(segment_texts).strip()
        avg_conf = float(np.mean(confidences)) if confidences else (1.0 if raw_text else 0.0)
        normalized = normalize_text(raw_text)

        if not normalized:
            return {
                "verified": False,
                "raw_transcription": "",
                "normalized_transcription": "",
                "confidence": 0.0,
                "command_matched": None,
                "reason": "No speech detected in the recording (audio silent or unintelligible).",
            }

        # Check exact or variant match
        matched_command = None
        for variant in norm_variants:
            # Check if variant appears as an exact match or substring in normalized text
            if normalized == variant or variant in normalized:
                matched_command = variant
                break

        # Also support token-level check: e.g. "present" token present in tokens
        if not matched_command:
            tokens = set(normalized.split())
            if norm_required in tokens:
                matched_command = norm_required

        if matched_command:
            return {
                "verified": True,
                "raw_transcription": raw_text,
                "normalized_transcription": normalized,
                "confidence": round(avg_conf, 3),
                "command_matched": matched_command,
                "reason": f"Command verified: recognized '{matched_command}'.",
            }
        else:
            return {
                "verified": False,
                "raw_transcription": raw_text,
                "normalized_transcription": normalized,
                "confidence": round(avg_conf, 3),
                "command_matched": None,
                "reason": (
                    f"Spoken phrase '{normalized}' did not match attendance command "
                    f"'{required_command}' (variants: {', '.join(approved_variants[:3])})."
                ),
            }

    except Exception as e:
        logger.exception("Error during speech command verification.")
        return {
            "verified": False,
            "raw_transcription": "",
            "normalized_transcription": "",
            "confidence": 0.0,
            "command_matched": None,
            "reason": f"Speech recognition processing error: {e}",
        }
    finally:
        if temp_wav_path and temp_wav_path.exists():
            try:
                temp_wav_path.unlink()
            except OSError:
                pass
