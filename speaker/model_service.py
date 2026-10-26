"""Speaker recognition model service (Deep Learning) using SpeechBrain ECAPA-TDNN.

Loads the pretrained ECAPA-TDNN model trained on VoxCeleb, performs inference on CPU,
and extracts 1D L2-normalized speaker embeddings.

Provides a clean adapter interface:
- load_model()
- extract_embedding(audio_input) -> np.ndarray
- is_available() -> bool
- get_model_info() -> dict
"""

import logging
import os
from pathlib import Path
import shutil
from typing import Any, Dict, Optional, Union

import numpy as np

import config
from audio.preprocessing import load_audio
from speaker.embedding_store import normalize_vector

logger = logging.getLogger(__name__)

# Suppress HuggingFace hub symlink warning on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _ensure_windows_symlink_support() -> None:
    """Monkey-patch os.symlink on Windows to fall back to file copy if WinError 1314 occurs.
    
    Windows non-admin users without Developer Mode lack SeCreateSymbolicLinkPrivilege.
    SpeechBrain attempts to symlink downloaded HF snapshot files to savedir; falling back
    to shutil.copyfile avoids fatal [WinError 1314] errors.
    """
    orig_symlink = getattr(os, "symlink", None)
    if orig_symlink is None:
        return

    # Check if already patched
    if getattr(orig_symlink, "_is_safe_symlink", False):
        return

    def _safe_symlink(src, dst, *args, **kwargs):
        try:
            return orig_symlink(src, dst, *args, **kwargs)
        except OSError as err:
            winerror = getattr(err, "winerror", None) or getattr(err, "errno", None)
            if winerror == 1314 or "1314" in str(err):
                if os.path.isdir(src):
                    return shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    return shutil.copyfile(src, dst)
            raise

    _safe_symlink._is_safe_symlink = True  # type: ignore
    os.symlink = _safe_symlink


# Ensure patch is applied early
_ensure_windows_symlink_support()

# Cached model instance
_SPEAKER_CLASSIFIER = None
_MODEL_LOAD_ERROR: Optional[str] = None
_EMBEDDING_DIM: Optional[int] = None


def load_model():
    """Load the pretrained SpeechBrain ECAPA-TDNN model on CPU with caching."""
    global _SPEAKER_CLASSIFIER, _MODEL_LOAD_ERROR, _EMBEDDING_DIM
    if _SPEAKER_CLASSIFIER is not None:
        return _SPEAKER_CLASSIFIER

    _ensure_windows_symlink_support()

    try:
        import torch

        # Support SpeechBrain 1.0+ and legacy pretrained modules
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:
            from speechbrain.pretrained import EncoderClassifier

        save_dir = str(config.SPEAKER_MODEL_DIR)
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Loading SpeechBrain ECAPA-TDNN model from source: {config.SPEAKER_MODEL_SOURCE} (CPU)..."
        )
        _SPEAKER_CLASSIFIER = EncoderClassifier.from_hparams(
            source=config.SPEAKER_MODEL_SOURCE,
            savedir=save_dir,
            run_opts={"device": "cpu"},
        )
        _MODEL_LOAD_ERROR = None
        logger.info("SpeechBrain ECAPA-TDNN loaded successfully.")
        return _SPEAKER_CLASSIFIER

    except ImportError as e:
        _MODEL_LOAD_ERROR = (
            f"SpeechBrain or PyTorch is not installed ({e}). "
            "Please ensure torch, torchaudio, and speechbrain are installed."
        )
        logger.warning(_MODEL_LOAD_ERROR)
        return None
    except Exception as e:
        _MODEL_LOAD_ERROR = (
            f"Failed to initialize SpeechBrain model '{config.SPEAKER_MODEL_SOURCE}': {e}. "
            "If working offline, ensure model weights are cached in "
            f"'{config.SPEAKER_MODEL_DIR}'."
        )
        logger.error(_MODEL_LOAD_ERROR)
        return None


def is_available() -> bool:
    """Return True if the model is loaded and ready for embedding extraction."""
    if _SPEAKER_CLASSIFIER is not None:
        return True
    # Try lazy load if no previous fatal error
    if _MODEL_LOAD_ERROR is None:
        try:
            import torch
            import speechbrain
            return True
        except ImportError:
            return False
    return False


def get_model_info() -> Dict[str, Any]:
    """Return status and diagnostic metadata for the speaker model."""
    return {
        "available": is_available() and (_MODEL_LOAD_ERROR is None),
        "loaded": _SPEAKER_CLASSIFIER is not None,
        "source": config.SPEAKER_MODEL_SOURCE,
        "device": "cpu",
        "cache_dir": str(config.SPEAKER_MODEL_DIR),
        "embedding_dim": _EMBEDDING_DIM or "192 (standard ECAPA-TDNN)",
        "error": _MODEL_LOAD_ERROR,
    }


def extract_embedding(
    audio_input: Union[str, Path, np.ndarray, bytes],
) -> np.ndarray:
    """Extract a 1D L2-normalized float32 speaker embedding vector.
    
    Args:
        audio_input: Path to audio file, raw audio bytes, or 1D float32 NumPy array.
        
    Returns:
        1D float32 NumPy array of unit norm.
        
    Raises:
        RuntimeError: If model is not loaded or inference fails.
    """
    global _EMBEDDING_DIM
    classifier = load_model()
    if classifier is None:
        error_msg = _MODEL_LOAD_ERROR or "Speaker recognition model is not loaded."
        raise RuntimeError(
            f"Cannot extract speaker embedding: {error_msg}\n"
            "Production mode requires a functional pretrained model; "
            "mock embeddings are disallowed."
        )

    import torch

    # Prepare audio signal as 1D float32 array
    if isinstance(audio_input, np.ndarray):
        audio_arr = audio_input.astype(np.float32)
    else:
        audio_arr, _ = load_audio(audio_input, target_sr=config.AUDIO_SAMPLE_RATE)

    if len(audio_arr) == 0:
        raise ValueError("Audio array is empty; cannot extract embedding.")

    # Convert to torch tensor with shape [1, num_samples]
    wav_tensor = torch.from_numpy(audio_arr).unsqueeze(0).float()
    wav_lens = torch.tensor([1.0], dtype=torch.float32)

    with torch.no_grad():
        # encode_batch returns shape [batch, 1, emb_dim] or [batch, emb_dim]
        embeddings = classifier.encode_batch(wav_tensor, wav_lens)
        emb_np = embeddings.squeeze().detach().cpu().numpy().astype(np.float32)

    if emb_np.ndim > 1:
        emb_np = emb_np.flatten()

    _EMBEDDING_DIM = emb_np.shape[0]

    # Normalize to unit sphere (L2 norm)
    normalized_emb = normalize_vector(emb_np)
    return normalized_emb
