"""Anti-Spoofing / Liveness Detection service for VoiceAttend.

IMPORTANT DESIGN NOTES:
- This module provides audio-quality-based spoofing heuristics for CPU-only deployments.
- A real deep-learning anti-spoofing model (e.g., AASIST, RawBoost, LCNN) requires
  labelled spoof/genuine data and GPU-level compute for practical accuracy.
- Without a pretrained model checkpoint, we CANNOT claim genuine anti-spoofing.
- This module clearly distinguishes between:
    (a) AUDIO QUALITY CHECKS  — detectable offline, always available
    (b) HEURISTIC SPOOFING SIGNALS — weak indicators, clearly labeled as heuristics
    (c) DEEP LEARNING ANTI-SPOOFING — blocked: no pretrained model available

Status of deep-learning anti-spoofing: BLOCKED
Reason: No pretrained AASIST/LCNN checkpoint is bundled. Models require ~200MB downloads
        from ASVspoof challenge repositories. Once a checkpoint is available, integrate via
        services/anti_spoofing_dl.py and call check_spoofing_dl() from here.

Fail-safe policy: If this check cannot complete, it returns status='unavailable'.
                  The attendance engine treats 'unavailable' as a WARNING, not a pass.
                  A 'spoof_detected' result is ALWAYS a hard rejection.
"""

import logging
import time
from typing import Any, Dict

import numpy as np

import config

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────────────────────

# RMS energy (silence) threshold — rejects very quiet recordings
MIN_ENERGY_RMS = config.MIN_AUDIO_RMS_THRESHOLD

# Peak clipping threshold — hard clipping often signals recorded / replayed audio
CLIP_FRACTION_THRESHOLD = 0.05   # >5% of samples near ±1.0 suggests clipping

# Spectral flatness heuristic — synthetic/TTS audio often has anomalous flatness
# (0 = pure tone, 1 = white noise); human speech is typically 0.1 – 0.4
SPECTRAL_FLATNESS_MIN = 0.02
SPECTRAL_FLATNESS_MAX = 0.85

# DC offset heuristic — replayed audio through certain equipment may carry DC
DC_OFFSET_THRESHOLD = 0.05

# Minimum zero-crossing rate — extremely low ZCR may indicate synthetic audio
ZCR_MIN = 0.005


def _compute_spectral_flatness(audio: np.ndarray, frame_size: int = 1024) -> float:
    """Estimate spectral flatness (Wiener entropy) over short frames.

    Spectral flatness = geometric_mean(|X[k]|) / arithmetic_mean(|X[k]|)
    Range: [0, 1] — pure tone → 0, white noise → 1.
    Human speech: typically 0.05 – 0.40 depending on phoneme.
    """
    if len(audio) < frame_size:
        return 0.1  # Too short to analyse — assume neutral

    # Use middle portion of signal (more stable than edges)
    mid = len(audio) // 2
    start = max(0, mid - frame_size)
    frame = audio[start : start + frame_size].astype(np.float64)

    spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
    spectrum = np.maximum(spectrum, 1e-10)  # Avoid log(0)

    log_mean = np.mean(np.log(spectrum))
    arith_mean = np.mean(spectrum)

    if arith_mean < 1e-10:
        return 0.0

    flatness = float(np.exp(log_mean) / arith_mean)
    return max(0.0, min(1.0, flatness))


def _compute_zero_crossing_rate(audio: np.ndarray) -> float:
    """Compute normalised zero crossing rate."""
    if len(audio) < 2:
        return 0.0
    crossings = float(np.sum(np.abs(np.diff(np.sign(audio)))) / 2)
    return crossings / len(audio)


def check_audio_quality_spoofing(
    audio: np.ndarray,
    sr: int = config.AUDIO_SAMPLE_RATE,
) -> Dict[str, Any]:
    """Run audio quality and heuristic spoofing checks.

    Checks performed:
        1. Silence / energy check
        2. Clipping fraction (hard clip → replay artifact)
        3. Spectral flatness (anomalously flat/tonal → suspect)
        4. DC offset check
        5. Zero crossing rate check

    Returns:
        Dict with:
            - status: 'clear' | 'suspicious' | 'spoof_detected' | 'unavailable'
            - checks: Dict of individual check results
            - confidence: float [0, 1] — 1 = almost certainly genuine (heuristic only!)
            - reason: str
            - is_dl_model: False (this is heuristic only)
            - latency_ms: float
    """
    t0 = time.time()

    if audio is None or len(audio) == 0:
        return {
            "status": "unavailable",
            "checks": {},
            "confidence": 0.0,
            "reason": "Audio array is empty; cannot perform anti-spoofing checks.",
            "is_dl_model": False,
            "latency_ms": 0.0,
        }

    audio_f = audio.astype(np.float32)
    checks: Dict[str, Any] = {}
    suspicious_flags: int = 0

    # ── Check 1: Energy (silence) ───────────────────────────────────────────
    rms = float(np.sqrt(np.mean(np.square(audio_f))))
    checks["energy_rms"] = rms
    checks["energy_ok"] = rms >= MIN_ENERGY_RMS
    if not checks["energy_ok"]:
        latency_ms = round((time.time() - t0) * 1000, 2)
        return {
            "status": "spoof_detected",
            "checks": checks,
            "confidence": 0.0,
            "reason": f"Recording is silent or near-silent (RMS={rms:.4f}). Cannot verify presence.",
            "is_dl_model": False,
            "latency_ms": latency_ms,
        }

    # ── Check 2: Clipping fraction ──────────────────────────────────────────
    clip_threshold = 0.98
    n_clipped = int(np.sum(np.abs(audio_f) >= clip_threshold))
    clip_fraction = n_clipped / len(audio_f)
    checks["clip_fraction"] = round(clip_fraction, 4)
    checks["clip_ok"] = clip_fraction <= CLIP_FRACTION_THRESHOLD
    if not checks["clip_ok"]:
        suspicious_flags += 1
        logger.warning(
            "Anti-spoofing: High clipping fraction detected "
            f"({clip_fraction:.1%}). Possible replay or distorted audio."
        )

    # ── Check 3: Spectral flatness ──────────────────────────────────────────
    flatness = _compute_spectral_flatness(audio_f)
    checks["spectral_flatness"] = round(flatness, 4)
    checks["flatness_ok"] = SPECTRAL_FLATNESS_MIN <= flatness <= SPECTRAL_FLATNESS_MAX
    if not checks["flatness_ok"]:
        suspicious_flags += 1
        logger.warning(
            f"Anti-spoofing: Anomalous spectral flatness ({flatness:.4f}). "
            "Expected 0.02–0.85 for natural speech."
        )

    # ── Check 4: DC offset ─────────────────────────────────────────────────
    dc_offset = float(np.abs(np.mean(audio_f)))
    checks["dc_offset"] = round(dc_offset, 4)
    checks["dc_ok"] = dc_offset <= DC_OFFSET_THRESHOLD
    if not checks["dc_ok"]:
        suspicious_flags += 1
        logger.warning(
            f"Anti-spoofing: High DC offset ({dc_offset:.4f}). "
            "Possible replay through uncalibrated playback equipment."
        )

    # ── Check 5: Zero crossing rate ────────────────────────────────────────
    zcr = _compute_zero_crossing_rate(audio_f)
    checks["zero_crossing_rate"] = round(zcr, 4)
    checks["zcr_ok"] = zcr >= ZCR_MIN
    if not checks["zcr_ok"]:
        suspicious_flags += 1
        logger.warning(
            f"Anti-spoofing: Very low zero-crossing rate ({zcr:.4f}). "
            "Could indicate a near-static or synthetic signal."
        )

    latency_ms = round((time.time() - t0) * 1000, 2)

    # ── Aggregate decision ─────────────────────────────────────────────────
    # Scoring: each failed heuristic lowers confidence
    confidence = max(0.0, 1.0 - (suspicious_flags * 0.25))

    if suspicious_flags >= 3:
        status = "spoof_detected"
        reason = (
            f"Multiple spoofing heuristics triggered ({suspicious_flags}/4). "
            "Recording is likely replayed or synthetic. "
            "NOTE: This is a heuristic determination — not a deep-learning model result."
        )
    elif suspicious_flags >= 1:
        status = "suspicious"
        reason = (
            f"{suspicious_flags} heuristic spoofing indicator(s) detected. "
            "Audio passed basic quality checks but shows anomalies. "
            "NOTE: Heuristic-only check — deep-learning model not available."
        )
    else:
        status = "clear"
        reason = (
            "All audio quality and heuristic anti-spoofing checks passed. "
            "NOTE: Heuristic-only check — no deep-learning liveness model is active. "
            "This does NOT guarantee the recording is live speech."
        )

    return {
        "status": status,
        "checks": checks,
        "confidence": round(confidence, 3),
        "reason": reason,
        "suspicious_flags": suspicious_flags,
        "is_dl_model": False,
        "latency_ms": latency_ms,
    }


def get_anti_spoofing_status() -> Dict[str, Any]:
    """Return system status information about the anti-spoofing capability."""
    return {
        "available": True,
        "mode": "heuristic_audio_quality",
        "dl_model_loaded": False,
        "dl_model_name": None,
        "dl_model_error": (
            "No pretrained anti-spoofing model is bundled. "
            "AASIST/LCNN models require ASVspoof2019 checkpoints (~200MB). "
            "See services/anti_spoofing.py for integration instructions."
        ),
        "heuristic_checks": [
            "energy_silence",
            "clip_fraction",
            "spectral_flatness",
            "dc_offset",
            "zero_crossing_rate",
        ],
        "warning": (
            "Active anti-spoofing uses audio quality heuristics ONLY. "
            "A determined attacker with a high-quality recording can bypass this check. "
            "Deep-learning liveness detection (AASIST, RawBoost) is BLOCKED — model not available."
        ),
    }
