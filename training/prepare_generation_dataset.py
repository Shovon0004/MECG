"""
Dataset preparation for Qwen3-VL evidence-grounded complaint generation.
Compiles multimodal inputs: frames, customer review, audio transcript,
domain, aspects, emotion, visual evidence, retrieved examples, and target complaint.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("MECG.PrepareGenDataset")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def build_generation_sample(
    row: Dict[str, Any],
    frames: List[Dict[str, Any]],
    transcript_info: Dict[str, Any],
    retrieved_examples: List[Dict[str, Any]],
    visual_evidence: List[Dict[str, Any]],
    requested_lang: str = "en",
) -> Dict[str, Any]:
    """Constructs a single structured training sample for evidence-grounded generation."""
    review_id = row.get("review_id", "")
    domain = row.get("domain", "General")
    customer_text = row.get("customer_text") or row.get("raw_text", "")
    ground_truth = row.get("ground_truth_complaint") or row.get("raw_ground_truth", "")
    aspects = row.get("aspects", "")
    emotion = row.get("emotion", "unknown")

    sample = {
        "review_id": review_id,
        "video_url_hash": row.get("video_url_hash", ""),
        "split": row.get("split", "train"),
        "product_domain": domain,
        "customer_review": customer_text,
        "audio_transcript_original": transcript_info.get("transcript", ""),
        "detected_audio_language": transcript_info.get("detected_language", "none"),
        "aspects": aspects,
        "emotion": emotion,
        "requested_output_language": requested_lang,
        "frames": frames,
        "visual_evidence": visual_evidence,
        "retrieved_context": retrieved_examples,
        "target_complaint": ground_truth,
    }
    return sample


def prepare_datasets(
    cleaned_csv_path: str = "data/processed/ComVID_cleaned.csv",
    transcripts_path: str = "data/processed/transcripts.jsonl",
    frames_dir: str = "data/processed/frames",
    output_dir: str = "data/processed",
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_out = out_dir / "generation_train.jsonl"
    val_out = out_dir / "generation_val.jsonl"

    # Load transcripts into map by video_id
    transcripts_map: Dict[str, Dict[str, Any]] = {}
    t_path = Path(transcripts_path)
    if t_path.exists():
        with open(t_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    transcripts_map[item.get("video_id", "")] = item

    frames_base = Path(frames_dir)

    train_count = 0
    val_count = 0

    with open(cleaned_csv_path, "r", encoding="utf-8") as f_in, \
         open(train_out, "w", encoding="utf-8") as f_train, \
         open(val_out, "w", encoding="utf-8") as f_val:

        reader = csv.DictReader(f_in)
        for row in reader:
            split = row.get("split", "train")
            v_hash = row.get("video_url_hash", "")
            rev_id = row.get("review_id", "")

            # Look up frames
            frame_dir = frames_base / v_hash
            frames = []
            if frame_dir.exists():
                meta_f = frame_dir / "metadata.json"
                if meta_f.exists():
                    try:
                        with open(meta_f, "r", encoding="utf-8") as mf:
                            frames = json.load(mf).get("frames", [])
                    except Exception:
                        pass
                if not frames:
                    for idx, img in enumerate(sorted(list(frame_dir.glob("*.jpg")))):
                        frames.append({"index": idx, "timestamp": round(idx * 2.0, 1), "file_path": str(img.resolve())})

            # Look up transcript
            t_info = transcripts_map.get(v_hash, transcripts_map.get(rev_id, {}))

            sample = build_generation_sample(
                row=row,
                frames=frames,
                transcript_info=t_info,
                retrieved_examples=[],
                visual_evidence=[],
                requested_lang="en",
            )

            line_json = json.dumps(sample, ensure_ascii=False) + "\n"
            if split == "train":
                f_train.write(line_json)
                train_count += 1
            elif split == "val":
                f_val.write(line_json)
                val_count += 1

    logger.info(f"Prepared generation datasets: {train_count} train samples -> {train_out}, {val_count} val samples -> {val_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned_csv", default="data/processed/ComVID_cleaned.csv")
    parser.add_argument("--transcripts", default="data/processed/transcripts.jsonl")
    parser.add_argument("--frames_dir", default="data/processed/frames")
    parser.add_argument("--output_dir", default="data/processed")
    args = parser.parse_args()

    prepare_datasets(
        cleaned_csv_path=args.cleaned_csv,
        transcripts_path=args.transcripts,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
    )
