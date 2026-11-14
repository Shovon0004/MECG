"""
Interactive demonstration script:
Runs the full MECG pipeline on a real downloaded video from data/videos/
and displays the complete multimodal evidence-grounded complaint description.
"""

import argparse
import csv
import json
import logging
from pathlib import Path

from src.pipeline import MECGPipeline
from evaluation.metrics import calculate_composite_cr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MECG.Demo")


def run_demo(video_path: str = None, review_id: str = None):
    # 1. Locate video file
    video_dir = Path("data/videos")
    available_videos = sorted(list(video_dir.glob("*.mp4")))

    if not available_videos and not video_path:
        print("\n[!] No video found in data/videos/. Please download pilot videos first using:")
        print("    !python main.py --phase 2 --pilot\n")
        return

    target_video = Path(video_path) if video_path else available_videos[0]
    video_hash = target_video.stem

    # 2. Look up metadata from ComVID_cleaned.csv
    csv_path = Path("data/processed/ComVID_cleaned.csv")
    row_info = {}
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("video_url_hash") == video_hash or row.get("review_id") == review_id:
                    row_info = row
                    break

    customer_text = row_info.get("customer_text") or "Not recommended. Product is defective."
    domain = row_info.get("domain") or "Household Item"
    gt_aspects = [a.strip() for a in row_info.get("aspects", "").split(",") if a.strip()]
    gt_emotion = row_info.get("emotion") or "dissatisfaction"
    gt_complaint = row_info.get("ground_truth_complaint") or ""

    print("\n" + "=" * 80)
    print("MECG MULTIMODAL EVIDENCE-GROUNDED COMPLAINT GENERATION DEMO")
    print("=" * 80)
    print(f"Video File        : {target_video.name}")
    print(f"Product Domain    : {domain}")
    print(f"Customer Input    : \"{customer_text}\"")
    print(f"Gold Ground Truth : \"{gt_complaint}\"")
    print("=" * 80)
    print("\n[>>] Initializing MECG Pipeline & running multimodal inference...\n")

    pipeline = MECGPipeline()
    result = pipeline.process(
        video_path=target_video,
        customer_text=customer_text,
        video_id=video_hash,
        product_domain=domain,
        requested_output_language="en",
    )

    # 3. Compute Composite CR Score
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

    print("\n" + "=" * 80)
    print("EVIDENCE EXTRACTION & MULTIMODAL PERCEPTION RESULTS")
    print("=" * 80)
    print(f"1. Spoken Audio Transcript : \"{result.get('transcript_original', '(No speech detected)')}\"")
    print(f"2. Detected Language       : {result.get('detected_language', 'en')}")
    print(f"3. Predicted Aspects       : {result.get('aspects', [])}")
    print(f"4. Predicted Emotion       : {result.get('emotion', 'dissatisfaction')}")
    
    print("\n5. Visual Observations (Frames):")
    for obs in result.get("visual_evidence", []):
        ts = obs.get("timestamp_seconds", "0.0")
        desc = obs.get("observation", obs.get("description", ""))
        print(f"   - [t={ts}s] {desc}")

    print("\n" + "=" * 80)
    print("FINAL GENERATED EVIDENCE-GROUNDED COMPLAINT DESCRIPTION")
    print("=" * 80)
    print(f"\n\"{generated_text}\"\n")
    print("-" * 80)

    print("VERIFICATION & GROUNDING AUDIT:")
    print(f"- Evidence Grounding Score : {result.get('evidence_grounding_score', 1.0) * 100:.1f}%")
    print(f"- Verified Claims Count   : {len(result.get('verified_claims', []))}")
    print(f"- Unsupported Claims      : {len(result.get('unsupported_claims', []))}")
    for uc in result.get("unsupported_claims", []):
        print(f"  [X] Filtered: {uc.get('claim')}")

    print("\nCOMPOSITE COMPLAINT RETENTION (CR) BREAKDOWN:")
    print(f"- Composite CR Score      : {cr_metrics['cr_score']:.4f}")
    print(f"  * Normalized Sentiment  : {cr_metrics['s_vader_norm']:.4f}")
    print(f"  * Emotion Alignment     : {cr_metrics['emotion_score']:.4f}")
    print(f"  * Aspect Recall         : {cr_metrics['aspect_score']:.4f}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MECG Video Description Demo")
    parser.add_argument("--video_path", type=str, default=None, help="Path to mp4 video")
    parser.add_argument("--review_id", type=str, default=None, help="ComVID review ID")
    args = parser.parse_args()

    run_demo(video_path=args.video_path, review_id=args.review_id)
