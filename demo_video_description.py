"""
Batch & single video demonstration and evaluation script:
Runs the full MECG multimodal pipeline on downloaded videos in data/videos/,
computes Grounding & Composite CR metrics, and displays structured reports.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.pipeline import MECGPipeline
from evaluation.metrics import calculate_composite_cr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MECG.Demo")


def load_dataset_metadata(csv_path: str = "data/processed/ComVID_cleaned.csv") -> Dict[str, Dict[str, Any]]:
    """Loads metadata index keyed by video_url_hash."""
    meta_by_hash = {}
    p = Path(csv_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                h = row.get("video_url_hash")
                if h and h not in meta_by_hash:
                    meta_by_hash[h] = row
    return meta_by_hash


def process_single_video(
    pipeline: MECGPipeline,
    target_video: Path,
    metadata_lookup: Dict[str, Dict[str, Any]],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Runs pipeline on a single video file and calculates metrics."""
    video_hash = target_video.stem
    row_info = metadata_lookup.get(video_hash, {})

    customer_text = row_info.get("customer_text") or "Not recommended. Product is defective."
    domain = row_info.get("domain") or "General"
    raw_aspects = row_info.get("aspects", "")
    gt_aspects = [a.strip() for a in raw_aspects.split(",") if a.strip()] if raw_aspects else ["Quality"]
    gt_emotion = row_info.get("emotion") or "dissatisfaction"
    gt_complaint = row_info.get("ground_truth_complaint") or ""

    if verbose:
        print("\n" + "=" * 80)
        print(f"PROCESSING VIDEO: {target_video.name}")
        print(f"Domain         : {domain}")
        print(f"Customer Input : \"{customer_text}\"")
        print("=" * 80)

    result = pipeline.process(
        video_path=target_video,
        customer_text=customer_text,
        video_id=video_hash,
        product_domain=domain,
        requested_output_language="en",
    )

    generated_text = result.get("generated_complaint", "")
    pred_aspects = result.get("aspects", [])
    pred_emotion = result.get("emotion", "dissatisfaction")

    cr_metrics = calculate_composite_cr(
        text=generated_text,
        target_aspects=gt_aspects,
        predicted_aspects=pred_aspects,
        target_emotion=gt_emotion,
        predicted_emotion=pred_emotion,
    )

    record = {
        "video_file": target_video.name,
        "video_hash": video_hash,
        "domain": domain,
        "customer_input": customer_text,
        "gold_ground_truth": gt_complaint,
        "generated_complaint": generated_text,
        "transcript": result.get("transcript_original", ""),
        "detected_language": result.get("detected_language", "en"),
        "predicted_aspects": pred_aspects,
        "gold_aspects": gt_aspects,
        "aspect_recall": cr_metrics["aspect_score"],
        "predicted_emotion": pred_emotion,
        "gold_emotion": gt_emotion,
        "emotion_score": cr_metrics["emotion_score"],
        "sentiment_score": cr_metrics["s_vader_norm"],
        "composite_cr": cr_metrics["cr_score"],
        "grounding_score": result.get("evidence_grounding_score", 0.0),
        "verified_claims_count": len(result.get("verified_claims", [])),
        "unsupported_claims_count": len(result.get("unsupported_claims", [])),
    }

    if verbose:
        print("\n--- GENERATED COMPLAINT ---")
        print(f"\"{generated_text}\"\n")
        print(f"CR Score: {cr_metrics['cr_score']:.4f} | Grounding: {result.get('evidence_grounding_score', 0.0)*100:.1f}% | Aspects: {pred_aspects}")
        print("=" * 80)

    return record


def print_summary_table(records: List[Dict[str, Any]]):
    """Prints ASCII table comparing all evaluated videos."""
    print("\n" + "=" * 96)
    print("MECG MULTIMODAL COMPLAINT GENERATION BATCH EVALUATION SUMMARY")
    print("=" * 96)
    print(f"{'Video File':<22} | {'Domain':<16} | {'Aspects (AS)':<12} | {'Emotion (ES)':<12} | {'S_VADER':<8} | {'CR Score':<8} | {'Grounding':<9}")
    print("-" * 96)

    total_as, total_es, total_sv, total_cr, total_gr = 0.0, 0.0, 0.0, 0.0, 0.0
    for r in records:
        print(
            f"{r['video_file']:<22} | "
            f"{r['domain'][:16]:<16} | "
            f"{r['aspect_recall']:<12.3f} | "
            f"{r['emotion_score']:<12.3f} | "
            f"{r['sentiment_score']:<8.3f} | "
            f"{r['composite_cr']:<8.3f} | "
            f"{r['grounding_score']*100:<8.1f}%"
        )
        total_as += r["aspect_recall"]
        total_es += r["emotion_score"]
        total_sv += r["sentiment_score"]
        total_cr += r["composite_cr"]
        total_gr += r["grounding_score"]

    n = max(1, len(records))
    print("-" * 96)
    print(
        f"{'AVERAGE (' + str(n) + ' videos)':<22} | "
        f"{'ALL DOMAINS':<16} | "
        f"{total_as/n:<12.3f} | "
        f"{total_es/n:<12.3f} | "
        f"{total_sv/n:<8.3f} | "
        f"{total_cr/n:<8.3f} | "
        f"{(total_gr/n)*100:<8.1f}%"
    )
    print("=" * 96 + "\n")


def run_batch_evaluation(video_path: str = None, run_all: bool = False, limit: int = None):
    video_dir = Path("data/videos")
    available_videos = sorted(list(video_dir.glob("*.mp4")))

    if not available_videos and not video_path:
        print("\n[!] No video found in data/videos/. Please download videos first using:")
        print("    !python main.py --phase 2 --limit 20  (or without limit for all)\n")
        return

    metadata_lookup = load_dataset_metadata()

    # Determine videos to process
    if video_path:
        targets = [Path(video_path)]
    elif run_all or limit:
        targets = available_videos[:limit] if limit else available_videos
    else:
        targets = [available_videos[0]]

    print(f"\n[>>] Initializing unified MECGPipeline for {len(targets)} video(s)...")
    pipeline = MECGPipeline()
    print("[>>] Pipeline ready. Commencing batch inference...\n")

    results = []
    for idx, vid in enumerate(targets, 1):
        print(f"\n>>> [{idx}/{len(targets)}] Evaluating {vid.name}...")
        rec = process_single_video(
            pipeline=pipeline,
            target_video=vid,
            metadata_lookup=metadata_lookup,
            verbose=(len(targets) <= 5),
        )
        results.append(rec)

    # Print summary table if more than 1 video
    if len(results) >= 1:
        print_summary_table(results)

    # Save output report
    output_path = Path("reports/all_videos_evaluation.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Detailed batch results saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MECG Batch Video Description & Evaluation")
    parser.add_argument("--video_path", type=str, default=None, help="Path to single mp4 video")
    parser.add_argument("--all", action="store_true", help="Process ALL videos in data/videos/")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos to evaluate")
    args = parser.parse_args()

    run_batch_evaluation(video_path=args.video_path, run_all=args.all, limit=args.limit)
