# VoiceAttend: Evaluation & Benchmarking Module

This module evaluates the performance, accuracy, and latency of the VoiceAttend system.

## 📊 Evaluated Metrics

1. **Top-1 Speaker Identification Accuracy (%)**: Percentage of genuine enrolled students correctly identified and accepted.
2. **Unknown Speaker Rejection Rate (%)**: Percentage of unenrolled/impostor voices correctly rejected.
3. **False Acceptance Rate (FAR) (%)**: Percentage of unauthorized or impostor voices wrongly accepted.
4. **False Rejection Rate (FRR) (%)**: Percentage of authorized enrolled students wrongly rejected.
5. **Enrollment Sample Count Impact**: Comparison of identification accuracy using **1 sample vs. 3 samples vs. 5 samples** per student profile.
6. **Command Recognition Accuracy (%)**: Percentage of speech utterances where the NLP verifier correctly recognizes the attendance command ("Present").
7. **Inference Latency (ms)**:
   - Average speech-command transcription latency.
   - Average speaker embedding extraction latency.

---

## 📁 Dataset Manifest Format

Evaluation data is specified via a standard CSV file with the following columns:

```csv
speaker_id,audio_path,split,spoken_command
student_01,data/eval/spk1_enroll_1.wav,enroll,present
student_01,data/eval/spk1_enroll_2.wav,enroll,present
student_01,data/eval/spk1_enroll_3.wav,enroll,present
student_01,data/eval/spk1_test_1.wav,test,present
student_02,data/eval/spk2_enroll_1.wav,enroll,present
student_02,data/eval/spk2_enroll_2.wav,enroll,present
student_02,data/eval/spk2_enroll_3.wav,enroll,present
student_02,data/eval/spk2_test_1.wav,test,present
unknown_01,data/eval/unk1_test_1.wav,test,present
unknown_02,data/eval/unk2_test_1.wav,test,good morning
```

### Manifest Column Definitions:
- `speaker_id`: Unique identifier for the speaker. Speakers intended as unauthorized impostors should have IDs starting with `unknown_` or should have no enrollment rows.
- `audio_path`: Relative or absolute path to the `.wav` file (16 kHz, 16-bit mono recommended).
- `split`: Either `enroll` (audio used to build the speaker voice profile) or `test` (audio used as query during evaluation).
- `spoken_command`: The text phrase uttered in the recording (e.g. `present`, `here`, or non-command phrases like `hello`).

> [!IMPORTANT]
> To preserve evaluation integrity, **never use the enrollment audio as the test audio**. Enrollment and testing recordings must be separate distinct utterances.

---

## 🚀 Running the Evaluation

To execute the benchmark:

```powershell
# Run with default settings
python evaluation/evaluate_speakers.py --dataset_csv evaluation/dataset_manifest.csv

# Run with custom threshold (e.g. 0.75) and custom sample splits
python evaluation/evaluate_speakers.py --dataset_csv evaluation/dataset_manifest.csv --threshold 0.75 --sample_splits 1 3 5
```

---

## 🎯 Threshold Calibration Guide

Cosine similarity for normalized SpeechBrain ECAPA-TDNN embeddings typically behaves as follows:

| Cosine Similarity Range | Interpretation | Recommended Action |
|:---|:---|:---|
| **> 0.80** | Very high confidence match | Accept unconditionally |
| **0.70 – 0.80** | Standard match zone | Default threshold range (0.72) |
| **0.60 – 0.70** | Ambiguous boundary zone | Potential acoustic mismatch or noise |
| **< 0.60** | Different speaker / Impostor | Reject |

### How to Calibrate:
- If legitimate students are being rejected (**high FRR**), slightly **decrease** the threshold (e.g. from 0.72 to 0.68).
- If different voices or impostors are occasionally accepted (**high FAR**), **increase** the threshold (e.g. from 0.72 to 0.76).
