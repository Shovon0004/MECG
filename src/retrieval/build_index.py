"""
Index builder for SigLIP 2 visual retrieval.
Indexes reference video frames from the training split strictly.
Prevents any validation or test set visual data leakage.
"""

import argparse
import csv
import logging
from pathlib import Path

from src.retrieval.visual_retriever import VisualRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MECG.BuildIndex")


def build_index(
    csv_path: str = "data/processed/ComVID_cleaned.csv",
    frames_dir: str = "data/processed/frames",
    output_dir: str = "outputs/retrieval_index",
    model_id: str = "google/siglip-base-patch16-224",
    limit: int = None,
    dry_run: bool = False,
):
    retriever = VisualRetriever(model_id=model_id, index_dir=output_dir)
    frames_base = Path(frames_dir)

    indexed_count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Rule: Prevent retrieval from validation/test data when building training indexes
            if row.get("split") != "train":
                continue

            v_hash = row.get("video_url_hash", "")
            video_frame_dir = frames_base / v_hash

            frame_files = sorted(list(video_frame_dir.glob("*.jpg")))
            if not frame_files:
                continue

            meta = {
                "review_id": row.get("review_id", ""),
                "domain": row.get("domain", ""),
                "aspects": row.get("aspects", ""),
                "ground_truth_complaint": row.get("ground_truth_complaint", ""),
                "split": "train",
            }

            retriever.add_video(video_hash=v_hash, frame_paths=frame_files, metadata=meta)
            indexed_count += 1
            if limit and indexed_count >= limit:
                break
            if dry_run and indexed_count >= 5:
                break

    idx_file, meta_file = retriever.save_index()
    logger.info(f"Built visual retrieval index with {indexed_count} training videos.")
    logger.info(f"Index: {idx_file} | Metadata: {meta_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="data/processed/ComVID_cleaned.csv")
    parser.add_argument("--frames_dir", default="data/processed/frames")
    parser.add_argument("--output_dir", default="outputs/retrieval_index")
    parser.add_argument("--model_id", default="google/siglip-base-patch16-224")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    build_index(
        csv_path=args.csv_path,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        model_id=args.model_id,
        limit=args.limit,
        dry_run=args.dry_run,
    )
