"""Helper script to create a sample synthetic evaluation dataset.

Generates reproducible WAV audio files for 2 enrolled students and 1 unknown speaker,
along with a dataset_manifest.csv for benchmarking evaluation/evaluate_speakers.py.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from audio.preprocessing import save_wav

EVAL_DIR = Path(__file__).resolve().parent / "sample_data"
MANIFEST_PATH = Path(__file__).resolve().parent / "dataset_manifest.csv"


def generate_synthetic_voice(base_freq: float, duration: float = 2.5, sr: int = 16000) -> np.ndarray:
    """Generate harmonic voice-like audio signal with amplitude envelope."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Fundamental frequency + formants / harmonics
    signal = (
        0.5 * np.sin(2 * np.pi * base_freq * t)
        + 0.3 * np.sin(2 * np.pi * (base_freq * 2) * t)
        + 0.15 * np.sin(2 * np.pi * (base_freq * 3) * t)
        + 0.05 * np.random.normal(0, 0.02, len(t))
    )
    # Smooth attack and release envelope to simulate natural speech onset
    envelope = np.ones_like(signal)
    fade_len = int(sr * 0.1)
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    envelope[:fade_len] = fade_in
    envelope[-fade_len:] = fade_out

    return (signal * envelope).astype(np.float32)


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    manifest_lines = ["speaker_id,audio_path,split,spoken_command"]

    speakers = [
        {"id": "student_01", "freq": 180.0, "enroll_count": 3, "test_count": 1},
        {"id": "student_02", "freq": 280.0, "enroll_count": 3, "test_count": 1},
        {"id": "unknown_01", "freq": 420.0, "enroll_count": 0, "test_count": 1},
    ]

    for spk in speakers:
        sid = spk["id"]
        freq = spk["freq"]

        # Enrollment samples
        for i in range(1, spk["enroll_count"] + 1):
            audio = generate_synthetic_voice(freq + (i * 2.0))
            filename = f"{sid}_enroll_{i}.wav"
            file_path = EVAL_DIR / filename
            save_wav(audio, file_path)
            rel_path = f"sample_data/{filename}"
            manifest_lines.append(f"{sid},{rel_path},enroll,present")

        # Testing samples
        for i in range(1, spk["test_count"] + 1):
            audio = generate_synthetic_voice(freq + 1.0)
            filename = f"{sid}_test_{i}.wav"
            file_path = EVAL_DIR / filename
            save_wav(audio, file_path)
            rel_path = f"sample_data/{filename}"
            manifest_lines.append(f"{sid},{rel_path},test,present")

    # Write manifest
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(manifest_lines) + "\n")

    print(f"Sample dataset generated successfully at {EVAL_DIR}")
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
