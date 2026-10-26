"""Audio recorder and capture service for VoiceAttend.

Provides utilities to:
- Process and save captured audio from Streamlit microphone widgets.
- Generate deterministic, traceable filenames tied to student IDs.
- Record audio directly from local microphone when running via CLI/evaluation.
"""

import io
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

import config
from audio.preprocessing import (
    AudioValidationError,
    load_audio,
    normalize_audio,
    save_wav,
    validate_audio,
)

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except (ImportError, OSError):
    HAS_SOUNDDEVICE = False


def process_and_save_enrollment_audio(
    audio_source: Union[bytes, io.BytesIO, str, Path],
    student_id: int,
    sample_number: int,
    save_dir: Optional[Path] = None,
) -> Tuple[Path, np.ndarray, float]:
    """Validate, normalize, and save an enrollment audio recording for a student.
    
    Args:
        audio_source: In-memory bytes or path of recorded audio.
        student_id: Database ID of the student.
        sample_number: The 1-based index of this sample (e.g. 1 to 5).
        save_dir: Destination folder (defaults to config.ENROLLMENT_AUDIO_DIR).
        
    Returns:
        (saved_path, audio_array, duration_seconds)
        
    Raises:
        AudioValidationError: If audio fails duration or silence checks.
    """
    target_dir = Path(save_dir) if save_dir else config.ENROLLMENT_AUDIO_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    # Load audio array at target sample rate
    audio, sr = load_audio(audio_source, target_sr=config.AUDIO_SAMPLE_RATE)

    # Validate audio constraints
    is_valid, reason = validate_audio(audio, sr=sr)
    if not is_valid:
        raise AudioValidationError(reason)

    # Normalize audio levels
    audio = normalize_audio(audio)
    duration = float(len(audio) / sr)

    # Generate unique, traceable filename
    timestamp = int(time.time())
    unique_suffix = uuid.uuid4().hex[:8]
    filename = f"student_{student_id}_sample_{sample_number}_{timestamp}_{unique_suffix}.wav"
    output_path = target_dir / filename

    # Save as 16-bit PCM WAV
    save_wav(audio, output_path, sr=sr)

    return output_path, audio, duration


def process_attendance_audio(
    audio_source: Union[bytes, io.BytesIO, str, Path],
) -> Tuple[np.ndarray, float]:
    """Validate and normalize a live attendance recording in-memory.
    
    Returns:
        (audio_array, duration_seconds)
    """
    audio, sr = load_audio(audio_source, target_sr=config.AUDIO_SAMPLE_RATE)
    is_valid, reason = validate_audio(audio, sr=sr)
    if not is_valid:
        raise AudioValidationError(reason)

    audio = normalize_audio(audio)
    duration = float(len(audio) / sr)
    return audio, duration


def record_from_microphone(
    duration: float = 3.0,
    sr: int = config.AUDIO_SAMPLE_RATE,
) -> np.ndarray:
    """Record directly from default microphone using sounddevice.
    
    Used in local CLI tests and evaluation scripts.
    """
    if not HAS_SOUNDDEVICE:
        raise RuntimeError(
            "sounddevice is not installed or no audio input device is available. "
            "Please record using the Streamlit web interface or install sounddevice."
        )

    recording = sd.rec(
        int(duration * sr),
        samplerate=sr,
        channels=config.AUDIO_CHANNELS,
        dtype="float32",
    )
    sd.wait()
    return np.squeeze(recording)
