"""VoiceAttend configuration settings.

Centralized configuration for audio recording, pretrained models,
database paths, enrollment parameters, similarity thresholds, and security.

Environment variable overrides:
    VOICEATTEND_DATA_DIR        — Override the data directory
    VOICEATTEND_DB_PATH         — Override the SQLite database path
    SPEAKER_MODEL_SOURCE        — Override SpeechBrain model HuggingFace ID
    VOICEATTEND_SIMILARITY_THRESHOLD — Override the cosine similarity threshold
    VOICEATTEND_AMBIGUOUS_MARGIN     — Override ambiguous match margin
    WHISPER_MODEL_SIZE          — Override Whisper model size (tiny/base/small)
    VOICEATTEND_ENABLE_AUTH     — Set to '1' to enable admin login (default: '0')
    VOICEATTEND_REQUIRE_AUTH    — Roles required to access the app (admin/faculty)
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("VOICEATTEND_DATA_DIR", BASE_DIR / "data"))

ENROLLMENT_AUDIO_DIR = DATA_DIR / "enrollment"
PROFILES_DIR = DATA_DIR / "profiles"
PRETRAINED_MODELS_DIR = DATA_DIR / "pretrained_models"
DB_PATH = Path(os.getenv("VOICEATTEND_DB_PATH", DATA_DIR / "voiceattend.db"))

# Ensure directories exist
for directory in (DATA_DIR, ENROLLMENT_AUDIO_DIR, PROFILES_DIR, PRETRAINED_MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Enrollment Settings
DEFAULT_ENROLLMENT_TARGET = 3
MAX_ENROLLMENT_SAMPLES = 5
MIN_ENROLLMENT_SAMPLES = 1

# Audio Requirements
AUDIO_SAMPLE_RATE = 16000  # 16 kHz mono standard for SpeechBrain & Whisper
AUDIO_CHANNELS = 1
MIN_AUDIO_DURATION_SEC = 1.0
MAX_AUDIO_DURATION_SEC = 6.0
MIN_AUDIO_RMS_THRESHOLD = 0.005  # Silence detection

# Speaker Recognition (SpeechBrain ECAPA-TDNN)
SPEAKER_MODEL_SOURCE = os.getenv(
    "SPEAKER_MODEL_SOURCE", "speechbrain/spkrec-ecapa-voxceleb"
)
SPEAKER_MODEL_DIR = PRETRAINED_MODELS_DIR / "spkrec-ecapa-voxceleb"

# Default cosine similarity threshold for ECAPA-TDNN (VoxCeleb normalized embeddings)
# Typical cosine similarity for same speaker is 0.70 - 0.85; different speaker is < 0.50
DEFAULT_SIMILARITY_THRESHOLD = float(os.getenv("VOICEATTEND_SIMILARITY_THRESHOLD", "0.72"))

# Minimum margin between top-1 and top-2 cosine scores to avoid ambiguous match rejection
DEFAULT_AMBIGUOUS_MARGIN = float(os.getenv("VOICEATTEND_AMBIGUOUS_MARGIN", "0.05"))

# Speech Command Verification (Whisper / faster-whisper / fallback)
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
WHISPER_MODEL_DIR = PRETRAINED_MODELS_DIR / "whisper"
DEFAULT_ATTENDANCE_COMMAND = "present"

# Approved command phrases (normalized)
APPROVED_COMMAND_VARIANTS = [
    "present",
    "present sir",
    "present ma'am",
    "present mam",
    "yes present",
    "i am present",
    "here",
    "yes sir",
    "yes ma'am",
]

# Authentication Settings
# VOICEATTEND_ENABLE_AUTH=1 enables admin login gate on all pages.
# Disable for single-user local installations.
ENABLE_AUTH = os.getenv("VOICEATTEND_ENABLE_AUTH", "0") == "1"

# Evaluation Settings
EVAL_DEFAULT_SPLIT_SAMPLES = (1, 3, 5)

# Application version
APP_VERSION = "2.0.0"
APP_NAME = "VoiceAttend"
