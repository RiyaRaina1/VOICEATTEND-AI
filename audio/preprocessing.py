"""Audio preprocessing and validation module for VoiceAttend.

Provides utilities for:
- Reading WAV/audio files and raw bytes.
- Converting audio to 16 kHz mono float32 arrays.
- Validating audio duration and minimum signal energy (silence rejection).
- Peak and RMS normalization.
- Saving processed audio in standard 16-bit PCM WAV format.
"""

import io
import math
import struct
import wave
from pathlib import Path
from typing import Tuple, Union

import numpy as np

import config

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


class AudioValidationError(ValueError):
    """Raised when recorded audio does not satisfy quality or duration constraints."""
    pass


def load_audio(
    source: Union[str, Path, bytes, io.BytesIO, np.ndarray],
    target_sr: int = config.AUDIO_SAMPLE_RATE,
) -> Tuple[np.ndarray, int]:
    """Load audio from file path, bytes, or numpy array into a 1D float32 numpy array normalized to [-1.0, 1.0].
    
    Returns:
        (audio_array, sample_rate)
    """
    if isinstance(source, np.ndarray):
        data = source.copy().astype(np.float32)
        sr = target_sr
    elif isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        if HAS_SOUNDFILE:
            data, sr = sf.read(str(path), dtype="float32")
        else:
            with open(path, "rb") as f:
                data, sr = _read_wav_fallback(f.read())
    elif isinstance(source, (bytes, bytearray)):
        if HAS_SOUNDFILE:
            bio = io.BytesIO(source)
            data, sr = sf.read(bio, dtype="float32")
        else:
            data, sr = _read_wav_fallback(source)
    elif isinstance(source, io.BytesIO):
        source.seek(0)
        raw_bytes = source.read()
        if HAS_SOUNDFILE:
            data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
        else:
            data, sr = _read_wav_fallback(raw_bytes)
    else:
        raise TypeError(f"Unsupported audio source type: {type(source)}")

    # Convert to mono if multichannel
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample to target_sr if necessary
    if sr != target_sr:
        data = _resample_linear(data, sr, target_sr)
        sr = target_sr

    return data.astype(np.float32), sr


def _read_wav_fallback(raw_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Fallback WAV parser using standard library wave module."""
    bio = io.BytesIO(raw_bytes)
    try:
        with wave.open(bio, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)

            if sampwidth == 1:
                # 8-bit unsigned PCM
                data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif sampwidth == 2:
                # 16-bit signed PCM
                data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                # 32-bit signed PCM
                data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                raise AudioValidationError(f"Unsupported WAV sample width: {sampwidth}")

            if n_channels > 1:
                data = data.reshape(-1, n_channels).mean(axis=1)

            return data, framerate
    except Exception as e:
        raise AudioValidationError(f"Failed to parse audio as WAV: {e}")


def _resample_linear(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Linear interpolation resampling for dependency-free resampling."""
    if orig_sr == target_sr:
        return data
    target_length = int(round(len(data) * float(target_sr) / orig_sr))
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    orig_indices = np.linspace(0, len(data) - 1, len(data))
    target_indices = np.linspace(0, len(data) - 1, target_length)
    return np.interp(target_indices, orig_indices, data).astype(np.float32)


def calculate_rms_energy(audio: np.ndarray) -> float:
    """Calculate the Root Mean Square (RMS) energy of an audio array."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def validate_audio(
    audio: np.ndarray,
    sr: int = config.AUDIO_SAMPLE_RATE,
    min_duration: float = config.MIN_AUDIO_DURATION_SEC,
    max_duration: float = config.MAX_AUDIO_DURATION_SEC,
    min_rms: float = config.MIN_AUDIO_RMS_THRESHOLD,
) -> Tuple[bool, str]:
    """Validate audio duration and signal energy.
    
    Returns:
        (is_valid: bool, error_message: str)
    """
    if len(audio) == 0:
        return False, "Audio recording is empty (0 samples)."

    duration = len(audio) / float(sr)
    if duration < min_duration:
        return False, f"Recording is too short ({duration:.2f}s). Minimum required is {min_duration:.1f}s."

    if duration > max_duration:
        return False, f"Recording is too long ({duration:.2f}s). Maximum allowed is {max_duration:.1f}s."

    rms = calculate_rms_energy(audio)
    if rms < min_rms:
        return False, (
            f"Recording is too quiet or silent (RMS energy: {rms:.4f} < {min_rms:.4f}). "
            "Please check microphone input and speak closer to the mic."
        )

    return True, ""


def normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak-normalize audio signal to avoid clipping and ensure consistent levels."""
    max_amp = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    if max_amp > 1e-6:
        return (audio / max_amp * target_peak).astype(np.float32)
    return audio.astype(np.float32)


def save_wav(
    audio: np.ndarray,
    output_path: Union[str, Path],
    sr: int = config.AUDIO_SAMPLE_RATE,
) -> Path:
    """Save float32 audio as a 16-bit PCM mono WAV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clip to [-1.0, 1.0] and convert to 16-bit signed PCM
    clipped = np.clip(audio, -1.0, 1.0)
    int16_data = (clipped * 32767.0).astype(np.int16)

    if HAS_SOUNDFILE:
        sf.write(str(output_path), int16_data, sr, subtype="PCM_16")
    else:
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(int16_data.tobytes())

    return output_path
