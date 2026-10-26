"""Audio processing package for VoiceAttend."""
from .preprocessing import (
    AudioValidationError,
    load_audio,
    validate_audio,
    normalize_audio,
    calculate_rms_energy,
    save_wav,
)
from .recorder import (
    process_and_save_enrollment_audio,
    process_attendance_audio,
    record_from_microphone,
    HAS_SOUNDDEVICE,
)

__all__ = [
    "AudioValidationError",
    "load_audio",
    "validate_audio",
    "normalize_audio",
    "calculate_rms_energy",
    "save_wav",
    "process_and_save_enrollment_audio",
    "process_attendance_audio",
    "record_from_microphone",
    "HAS_SOUNDDEVICE",
]
