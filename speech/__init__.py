"""Speech command verification package for VoiceAttend."""
from .command_verifier import (
    normalize_text,
    load_command_model,
    verify_command,
    is_available,
    get_model_status,
)

__all__ = [
    "normalize_text",
    "load_command_model",
    "verify_command",
    "is_available",
    "get_model_status",
]
