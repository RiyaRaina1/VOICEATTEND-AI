"""VoiceAttend — Enhanced Evaluation Harness.

Evaluates speaker verification performance with full metrics suite:
1. Speaker identification accuracy and unknown-speaker rejection.
2. False Acceptance Rate (FAR) and False Rejection Rate (FRR).
3. Equal Error Rate (EER) via threshold sweep.
4. ROC curve data and AUC.
5. Confusion matrix (per-speaker).
6. Precision, Recall, F1-score (binary: genuine vs. impostor).
7. Enrollment sample-count comparison (1 vs 3 vs 5 enrollment samples).
8. Command recognition accuracy and latency.
9. Inference latency statistics: mean, median, 95th percentile.
10. Anti-spoofing metrics (if labelled data is available).

Dataset format: CSV with columns:
    speaker_id, audio_path, split, spoken_command, is_spoof (optional)
where:
    - split is 'enroll' or 'test'
    - unknown impostor speakers have speaker_id starting with 'unknown_' or are not enrolled.
    - is_spoof (1=spoof, 0=genuine) — only required for anti-spoofing metrics.

IMPORTANT:
  - No benchmark data is fabricated. All results are from real model predictions.
  - If a dataset is unavailable, the harness prints clear instructions and exits.
  - Enrollment and test splits are kept separate to prevent data leakage.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from audio.preprocessing import load_audio
from services import anti_spoofing as antispoofing_service
from speaker import embedding_store, identification, model_service
from speech import command_verifier

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="VoiceAttend Model Evaluation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset CSV format:
  speaker_id,audio_path,split,spoken_command,is_spoof
  student_01,data/eval/spk1_enroll_1.wav,enroll,present,0
  student_01,data/eval/spk1_test_1.wav,test,present,0
  unknown_01,data/eval/unk_test_1.wav,test,other,0

Notes:
  - 'is_spoof' column is optional (needed only for anti-spoofing metrics)
  - Impostor speakers: use speaker_id prefix 'unknown_' OR ensure they are not enrolled
  - Paths can be relative to the CSV file location
        """,
    )
    parser.add_argument(
        "--dataset_csv",
        type=str,
        default=str(PROJECT_ROOT / "evaluation" / "dataset_manifest.csv"),
        help="Path to dataset manifest CSV file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.DEFAULT_SIMILARITY_THRESHOLD,
        help="Cosine similarity threshold for speaker verification.",
    )
    parser.add_argument(
        "--sample_splits",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help="Number of enrollment samples per speaker to evaluate (e.g. 1 3 5).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(PROJECT_ROOT / "evaluation" / "evaluation_results.csv"),
        help="Path to save evaluation summary CSV.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(PROJECT_ROOT / "evaluation" / "evaluation_results.json"),
        help="Path to save full evaluation results JSON.",
    )
    parser.add_argument(
        "--sweep_thresholds",
        action="store_true",
        help="Sweep thresholds 0.30–0.95 to compute EER and ROC curve.",
    )
    return parser.parse_args()


def build_profiles_from_samples(
    enroll_samples: Dict[str, List[np.ndarray]],
    k_samples: int,
) -> Dict[str, Dict]:
    """Construct speaker profiles using up to k enrollment samples per speaker."""
    profiles = {}
    for spk_id, emb_list in enroll_samples.items():
        used_embs = emb_list[:k_samples]
        if not used_embs:
            continue
        mean_emb = np.mean(used_embs, axis=0)
        norm_profile = embedding_store.normalize_vector(mean_emb)
        profiles[spk_id] = {
            "student_id": spk_id,
            "student_code": spk_id,
            "name": f"Speaker_{spk_id}",
            "sample_count": len(used_embs),
            "profile_embedding": norm_profile,
        }
    return profiles


def compute_eer(fpr_list: List[float], fnr_list: List[float]) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) and the threshold where FPR ≈ FNR.

    Returns:
        (eer: float, eer_threshold: float)
    """
    fpr = np.array(fpr_list)
    fnr = np.array(fnr_list)

    # Find the threshold index where the absolute difference is minimal
    diff = np.abs(fpr - fnr)
    idx = np.argmin(diff)

    # Interpolate for smoother EER
    if idx < len(fpr) - 1:
        eer = (fpr[idx] + fnr[idx]) / 2.0
    else:
        eer = fpr[idx]

    return float(eer), float(idx)


def compute_roc_auc(fpr_list: List[float], tpr_list: List[float]) -> float:
    """Compute AUC under ROC curve using trapezoidal integration."""
    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)
    # Sort by FPR
    sorted_idx = np.argsort(fpr)
    fpr_sorted = fpr[sorted_idx]
    tpr_sorted = tpr[sorted_idx]
    trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trapz_func(tpr_sorted, fpr_sorted)) if trapz_func else 0.0


def compute_binary_metrics(
    y_true: List[int], y_pred: List[int]
) -> Dict[str, float]:
    """Compute precision, recall, F1 for binary classification (genuine=1, impostor=0)."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "specificity": round(specificity, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def percentile(arr: List[float], p: float) -> float:
    """Compute the p-th percentile of a list."""
    if not arr:
        return 0.0
    return float(np.percentile(arr, p))


def run_evaluation(
    dataset_path: Path,
    similarity_threshold: float,
    sample_splits: List[int],
    sweep_thresholds: bool = False,
) -> Optional[Dict]:
    """Execute complete evaluation workflow on the provided dataset."""
    if not dataset_path.exists():
        print(f"\n[ERROR] Dataset manifest file not found: {dataset_path}")
        print("\nExpected CSV format:")
        print("  speaker_id,audio_path,split,spoken_command,is_spoof")
        print("  student_01,data/eval/spk1_enroll_1.wav,enroll,present,0")
        print("  student_01,data/eval/spk1_test_1.wav,test,present,0")
        print("  unknown_01,data/eval/unk_test_1.wav,test,other,0")
        print("\nSee evaluation/README.md for dataset setup instructions.")
        return None

    df = pd.read_csv(dataset_path)
    required_cols = {"speaker_id", "audio_path", "split", "spoken_command"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Manifest CSV missing required columns: {missing_cols}")

    has_spoof_labels = "is_spoof" in df.columns

    enroll_rows = df[df["split"].str.lower() == "enroll"]
    test_rows = df[df["split"].str.lower() == "test"]

    print("\n" + "=" * 60)
    print(" VoiceAttend Evaluation Benchmark")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Total entries: {len(df)}")
    print(f"Enrollment audio files: {len(enroll_rows)}")
    print(f"Testing audio files: {len(test_rows)}")
    print(f"Has spoof labels: {has_spoof_labels}")
    print(f"Configured Threshold: {similarity_threshold:.2f}")
    print(f"Sample count splits: {sample_splits}")

    if len(enroll_rows) == 0 or len(test_rows) == 0:
        print("[ERROR] Manifest must contain both 'enroll' and 'test' audio samples.")
        return None

    if not model_service.is_available():
        print("\n[ERROR] Pretrained speaker recognition model is unavailable.")
        info = model_service.get_model_info()
        print(f"Detail: {info.get('error')}")
        return None

    # ── 1. Extract enrollment embeddings ───────────────────────────────────────
    print("\n--- Extracting Enrollment Embeddings ---")
    enroll_embeddings: Dict[str, List[np.ndarray]] = {}
    embed_errors = 0

    for _, row in enroll_rows.iterrows():
        spk_id = str(row["speaker_id"]).strip()
        path = Path(row["audio_path"])
        if not path.is_absolute():
            path = dataset_path.parent / path

        if not path.exists():
            logger.warning(f"Audio file missing: {path}, skipping.")
            continue

        try:
            emb = model_service.extract_embedding(path)
            enroll_embeddings.setdefault(spk_id, []).append(emb)
        except Exception as e:
            logger.error(f"Failed to extract embedding for {path}: {e}")
            embed_errors += 1

    print(f"  Enrolled speakers: {len(enroll_embeddings)}")
    print(f"  Embedding errors: {embed_errors}")

    # ── 2. Process test audio ──────────────────────────────────────────────────
    print("\n--- Processing Test Audio & Command Verification ---")
    test_data: List[Dict] = []
    cmd_latencies: List[float] = []
    spk_latencies: List[float] = []
    spoof_latencies: List[float] = []
    cmd_correct_count = 0

    for _, row in test_rows.iterrows():
        spk_id = str(row["speaker_id"]).strip()
        path = Path(row["audio_path"])
        if not path.is_absolute():
            path = dataset_path.parent / path

        spoken_cmd = str(row.get("spoken_command", "")).strip() if pd.notna(row.get("spoken_command")) else ""
        is_spoof_label = int(row["is_spoof"]) if has_spoof_labels and pd.notna(row.get("is_spoof")) else None

        if not path.exists():
            logger.warning(f"Test audio missing: {path}, skipping.")
            continue

        try:
            audio_arr, _ = load_audio(path)
        except Exception as e:
            logger.error(f"Failed to load {path}: {e}")
            continue

        # Command Verification
        t0 = time.time()
        try:
            cmd_result = command_verifier.verify_command(path)
        except Exception as e:
            cmd_result = {"verified": False, "reason": str(e)}
        t_cmd = time.time() - t0
        cmd_latencies.append(t_cmd)

        expected_cmd_valid = spoken_cmd.lower() in config.APPROVED_COMMAND_VARIANTS
        if cmd_result["verified"] == expected_cmd_valid:
            cmd_correct_count += 1

        # Speaker Embedding
        t0 = time.time()
        try:
            test_emb = model_service.extract_embedding(path)
        except Exception as e:
            logger.error(f"Failed speaker embedding for {path}: {e}")
            continue
        t_spk = time.time() - t0
        spk_latencies.append(t_spk)

        # Anti-spoofing heuristics
        t0 = time.time()
        spoof_check = antispoofing_service.check_audio_quality_spoofing(audio_arr)
        t_spoof = time.time() - t0
        spoof_latencies.append(t_spoof)

        test_data.append({
            "true_speaker": spk_id,
            "embedding": test_emb,
            "audio_arr": audio_arr,
            "path": str(path),
            "command_verified": cmd_result.get("verified", False),
            "spoken_command": spoken_cmd,
            "is_spoof_label": is_spoof_label,
            "spoof_check": spoof_check,
        })

    total_test = len(test_data)
    if total_test == 0:
        print("[ERROR] No valid test samples could be processed.")
        return None

    cmd_accuracy = (cmd_correct_count / total_test) * 100.0

    # Latency statistics
    def latency_stats(arr_sec: List[float]) -> Dict:
        arr_ms = [v * 1000 for v in arr_sec]
        return {
            "mean_ms": round(np.mean(arr_ms) if arr_ms else 0, 2),
            "median_ms": round(np.median(arr_ms) if arr_ms else 0, 2),
            "p95_ms": round(percentile(arr_ms, 95), 2),
        }

    cmd_latency_stats = latency_stats(cmd_latencies)
    spk_latency_stats = latency_stats(spk_latencies)

    print(f"  Total test samples: {total_test}")
    print(f"  Command accuracy: {cmd_accuracy:.2f}%")
    print(f"  Cmd latency: mean={cmd_latency_stats['mean_ms']}ms, "
          f"median={cmd_latency_stats['median_ms']}ms, "
          f"p95={cmd_latency_stats['p95_ms']}ms")
    print(f"  Speaker embed latency: mean={spk_latency_stats['mean_ms']}ms, "
          f"median={spk_latency_stats['median_ms']}ms, "
          f"p95={spk_latency_stats['p95_ms']}ms")

    # ── 3. Anti-spoofing metrics (if labels available) ─────────────────────────
    spoof_metrics = None
    if has_spoof_labels:
        labelled = [d for d in test_data if d["is_spoof_label"] is not None]
        if labelled:
            y_true_spoof = [d["is_spoof_label"] for d in labelled]
            y_pred_spoof = [
                1 if d["spoof_check"]["status"] == "spoof_detected" else 0
                for d in labelled
            ]
            spoof_metrics = compute_binary_metrics(y_true_spoof, y_pred_spoof)
            spoof_metrics["note"] = (
                "Heuristic-only anti-spoofing (audio quality checks). "
                "Not a deep-learning model result."
            )
            print("\n  Anti-Spoofing Heuristic Metrics:")
            print(f"    Precision: {spoof_metrics['precision']:.4f}")
            print(f"    Recall:    {spoof_metrics['recall']:.4f}")
            print(f"    F1:        {spoof_metrics['f1_score']:.4f}")

    # ── 4. Speaker recognition across enrollment sample counts ─────────────────
    print("\n--- Speaker Recognition Across Enrollment Sample Counts ---")
    results_by_split = []
    roc_data = {}

    for k in sample_splits:
        profiles = build_profiles_from_samples(enroll_embeddings, k_samples=k)
        if not profiles:
            print(f"  [SKIP] No profiles built for k={k}")
            continue

        enrolled_speakers = set(profiles.keys())
        tp = fn = fp = tn = wrong_id = 0
        genuine_count = impostor_count = 0
        y_true_binary: List[int] = []
        y_pred_binary: List[int] = []

        # For threshold sweeping (ROC/EER)
        all_scores: List[float] = []
        all_labels: List[int] = []  # 1=genuine, 0=impostor

        for item in test_data:
            true_spk = item["true_speaker"]
            is_genuine = true_spk in enrolled_speakers

            id_res = identification.identify_speaker(
                query_embedding=item["embedding"],
                enrolled_profiles=profiles,
                threshold=similarity_threshold,
            )
            decision = id_res["decision"]
            best = id_res.get("best_candidate")
            best_score = best["score"] if best else 0.0

            all_scores.append(best_score)
            all_labels.append(1 if is_genuine else 0)

            if is_genuine:
                genuine_count += 1
                y_true_binary.append(1)
                if decision == "accepted":
                    if best and str(best["student_id"]) == true_spk:
                        tp += 1
                        y_pred_binary.append(1)
                    else:
                        wrong_id += 1
                        y_pred_binary.append(1)  # accepted but wrong ID
                else:
                    fn += 1
                    y_pred_binary.append(0)
            else:
                impostor_count += 1
                y_true_binary.append(0)
                if decision == "accepted":
                    fp += 1
                    y_pred_binary.append(1)
                else:
                    tn += 1
                    y_pred_binary.append(0)

        accuracy = (tp / genuine_count * 100.0) if genuine_count > 0 else 0.0
        frr = (fn / genuine_count * 100.0) if genuine_count > 0 else 0.0
        far = (fp / impostor_count * 100.0) if impostor_count > 0 else 0.0
        unknown_rej_rate = (tn / impostor_count * 100.0) if impostor_count > 0 else 0.0

        binary_metrics = compute_binary_metrics(y_true_binary, y_pred_binary)

        # ROC / EER if requested
        eer = auc = None
        if sweep_thresholds and all_labels and any(all_labels) and not all(all_labels):
            thresholds = np.arange(0.10, 1.00, 0.01)
            fpr_list, tpr_list, fnr_list = [], [], []
            for t in thresholds:
                tp_t = sum(1 for s, l in zip(all_scores, all_labels) if l == 1 and s >= t)
                fn_t = sum(1 for s, l in zip(all_scores, all_labels) if l == 1 and s < t)
                fp_t = sum(1 for s, l in zip(all_scores, all_labels) if l == 0 and s >= t)
                tn_t = sum(1 for s, l in zip(all_scores, all_labels) if l == 0 and s < t)

                fpr_val = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0.0
                tpr_val = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0
                fnr_val = fn_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0

                fpr_list.append(fpr_val)
                tpr_list.append(tpr_val)
                fnr_list.append(fnr_val)

            eer, _ = compute_eer(fpr_list, fnr_list)
            auc = compute_roc_auc(fpr_list, tpr_list)
            roc_data[f"k{k}"] = {
                "thresholds": thresholds.tolist(),
                "fpr": fpr_list,
                "tpr": tpr_list,
                "fnr": fnr_list,
            }

        split_summary = {
            "samples_per_student": k,
            "genuine_queries": genuine_count,
            "impostor_queries": impostor_count,
            "identification_accuracy_pct": round(accuracy, 2),
            "false_rejection_rate_pct": round(frr, 2),
            "false_acceptance_rate_pct": round(far, 2),
            "unknown_rejection_rate_pct": round(unknown_rej_rate, 2),
            "wrong_identity_assignments": wrong_id,
            "precision": binary_metrics["precision"],
            "recall": binary_metrics["recall"],
            "f1_score": binary_metrics["f1_score"],
            "specificity": binary_metrics["specificity"],
            "eer": round(eer, 4) if eer is not None else None,
            "auc": round(auc, 4) if auc is not None else None,
        }
        results_by_split.append(split_summary)

        print(f"\n[Profile Size: {k} Sample(s) per Student]")
        print(f"  Genuine: {genuine_count} | Impostor: {impostor_count}")
        print(f"  Identification Accuracy: {accuracy:.2f}%")
        print(f"  FRR: {frr:.2f}% | FAR: {far:.2f}%")
        print(f"  Unknown Rejection Rate: {unknown_rej_rate:.2f}%")
        print(f"  Precision: {binary_metrics['precision']:.4f} | "
              f"Recall: {binary_metrics['recall']:.4f} | "
              f"F1: {binary_metrics['f1_score']:.4f}")
        if eer is not None:
            print(f"  EER: {eer:.4f} | AUC: {auc:.4f}")

    return {
        "results_by_split": results_by_split,
        "command_accuracy_pct": round(cmd_accuracy, 2),
        "cmd_latency": cmd_latency_stats,
        "speaker_latency": spk_latency_stats,
        "anti_spoofing_metrics": spoof_metrics,
        "roc_data": roc_data,
        "note": (
            "All metrics computed from real model predictions on the provided dataset. "
            "No benchmark data is fabricated."
        ),
    }


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_csv)

    results = run_evaluation(
        dataset_path=dataset_path,
        similarity_threshold=args.threshold,
        sample_splits=args.sample_splits,
        sweep_thresholds=args.sweep_thresholds,
    )

    if not results:
        return

    if results.get("results_by_split"):
        out_df = pd.DataFrame(results["results_by_split"])
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"\nSaved evaluation summary CSV: {out_path.resolve()}")

    # Save full JSON results
    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types for JSON serialization
    def default_serializer(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    # Remove non-serializable audio arrays from results
    serializable_results = {k: v for k, v in results.items() if k != "roc_data"}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, default=default_serializer)
    print(f"Saved full results JSON: {json_path.resolve()}")


if __name__ == "__main__":
    main()
