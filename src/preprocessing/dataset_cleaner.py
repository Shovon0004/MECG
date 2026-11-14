"""
Dataset cleaning pipeline for the Com-VID dataset.
Implements data hygiene, duplicate tracking, label normalization,
group-aware train/val/test splitting to prevent video leakage,
and preservation of raw data columns.
"""

import csv
import json
import logging
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import yaml
except ImportError:
    yaml = None

from src.preprocessing.normalizers import (
    clean_text_whitespace,
    compute_url_hash,
    detect_language_hint,
    normalize_aspects,
    normalize_emotion,
    parse_html_rating,
)

logger = logging.getLogger("MECG.Cleaner")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ComVIDCleaner:
    """
    Cleans and standardizes the Com-VID dataset.
    Ensures zero video leakage across splits and preserves all raw columns.
    """

    def __init__(
        self,
        raw_csv_path: Union[str, Path] = "data/Com-VID_data.csv",
        output_csv_path: Union[str, Path] = "data/processed/ComVID_cleaned.csv",
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        random_seed: int = 42,
    ):
        self.raw_csv_path = Path(raw_csv_path)
        self.output_csv_path = Path(output_csv_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.records: List[Dict[str, Any]] = []

    @classmethod
    def from_config(cls, config_path: Union[str, Path] = "configs/config.yaml") -> "ComVIDCleaner":
        """Instantiates cleaner using project YAML configuration."""
        cfg_path = Path(config_path)
        if cfg_path.exists() and yaml:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            paths = cfg.get("paths", {})
            splits = cfg.get("splits", {})
            return cls(
                raw_csv_path=paths.get("raw_csv", "data/Com-VID_data.csv"),
                output_csv_path=paths.get("cleaned_csv", "data/processed/ComVID_cleaned.csv"),
                train_ratio=splits.get("train_ratio", 0.80),
                val_ratio=splits.get("val_ratio", 0.10),
                test_ratio=splits.get("test_ratio", 0.10),
                random_seed=splits.get("random_seed", 42),
            )
        return cls()

    def load_raw_data(self) -> List[Dict[str, str]]:
        """Reads raw CSV records using Python standard csv library."""
        if not self.raw_csv_path.exists():
            raise FileNotFoundError(f"Raw dataset not found at: {self.raw_csv_path}")

        records: List[Dict[str, str]] = []
        with open(self.raw_csv_path, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            # Find matching headers (handling possible trailing spaces e.g. 'Ground-truth ')
            field_map = {}
            for col in reader.fieldnames or []:
                norm_col = col.strip()
                field_map[norm_col] = col

            for idx, row in enumerate(reader):
                cleaned_row = {
                    "raw_id": row.get(field_map.get("ID", "ID"), ""),
                    "raw_title": row.get(field_map.get("Title", "Title"), ""),
                    "raw_rating": row.get(field_map.get("Rating", "Rating"), ""),
                    "raw_video_source": row.get(field_map.get("Video-Source", "Video-Source"), ""),
                    "raw_text": row.get(field_map.get("Text", "Text"), ""),
                    "raw_ground_truth": row.get(field_map.get("Ground-truth", "Ground-truth "), ""),
                    "raw_emotion": row.get(field_map.get("Emotion", "Emotion"), ""),
                    "raw_aspect": row.get(field_map.get("Aspect", "Aspect"), ""),
                    "raw_domain": row.get(field_map.get("Domain", "Domain"), ""),
                    "row_index": idx,
                }
                records.append(cleaned_row)

        logger.info(f"Loaded {len(records)} raw records from {self.raw_csv_path}")
        return records

    def process(self) -> List[Dict[str, Any]]:
        """
        Executes complete cleaning pipeline:
        1. Parse HTML ratings and whitespace
        2. Normalize emotions and aspects
        3. Flag duplicate IDs and duplicate URLs
        4. Assign group-aware train/val/test splits
        """
        raw_records = self.load_raw_data()
        total_count = len(raw_records)

        # 1. First pass: count frequencies for ID and Video-Source to flag duplicates
        id_counter = Counter(r["raw_id"].strip() for r in raw_records if r["raw_id"].strip())
        url_counter = Counter(
            r["raw_video_source"].strip() for r in raw_records if r["raw_video_source"].strip()
        )

        intermediate_records: List[Dict[str, Any]] = []
        video_groups: Dict[str, List[int]] = defaultdict(list)

        for idx, r in enumerate(raw_records):
            review_id = r["raw_id"].strip()
            video_url = r["raw_video_source"].strip()
            has_video_url = bool(video_url)

            # Deterministic group hash: uses URL if present, otherwise ID fallback
            group_hash = compute_url_hash(video_url, fallback_id=review_id)
            video_groups[group_hash].append(idx)

            title_clean = clean_text_whitespace(r["raw_title"])
            customer_text = clean_text_whitespace(r["raw_text"])
            ground_truth = clean_text_whitespace(r["raw_ground_truth"])
            rating_stars = parse_html_rating(r["raw_rating"])
            emotion = normalize_emotion(r["raw_emotion"])
            aspect_list = normalize_aspects(r["raw_aspect"])
            domain = r["raw_domain"].strip()

            record: Dict[str, Any] = {
                "row_index": idx,
                "review_id": review_id,
                "title": title_clean,
                "rating_stars": rating_stars if rating_stars is not None else "",
                "video_url": video_url,
                "has_video_url": has_video_url,
                "video_url_hash": group_hash,
                "customer_text": customer_text,
                "ground_truth_complaint": ground_truth,
                "emotion": emotion,
                "aspects": ", ".join(aspect_list),
                "aspect_count": len(aspect_list),
                "domain": domain,
                "text_char_len": len(customer_text),
                "text_word_len": len(customer_text.split()),
                "ground_truth_char_len": len(ground_truth),
                "ground_truth_word_len": len(ground_truth.split()),
                "detected_language_hint": detect_language_hint(customer_text),
                # Duplicate tracking flags
                "is_duplicate_id": id_counter[review_id] > 1 if review_id else False,
                "id_occurrence_count": id_counter[review_id] if review_id else 0,
                "is_duplicate_url": url_counter[video_url] > 1 if video_url else False,
                "url_occurrence_count": url_counter[video_url] if video_url else 0,
                # Preserve raw values exactly
                "raw_id": r["raw_id"],
                "raw_title": r["raw_title"],
                "raw_rating": r["raw_rating"],
                "raw_video_source": r["raw_video_source"],
                "raw_text": r["raw_text"],
                "raw_ground_truth": r["raw_ground_truth"],
                "raw_emotion": r["raw_emotion"],
                "raw_aspect": r["raw_aspect"],
                "raw_domain": r["raw_domain"],
            }
            intermediate_records.append(record)

        # 2. Perform Group-Aware Train / Val / Test splitting
        # All records sharing the same video_url_hash are assigned to the exact same split.
        unique_groups = sorted(list(video_groups.keys()))
        rng = random.Random(self.random_seed)
        rng.shuffle(unique_groups)

        n_groups = len(unique_groups)
        n_train = int(n_groups * self.train_ratio)
        n_val = int(n_groups * self.val_ratio)

        train_groups = set(unique_groups[:n_train])
        val_groups = set(unique_groups[n_train : n_train + n_val])
        test_groups = set(unique_groups[n_train + n_val :])

        group_to_split: Dict[str, str] = {}
        for g in train_groups:
            group_to_split[g] = "train"
        for g in val_groups:
            group_to_split[g] = "val"
        for g in test_groups:
            group_to_split[g] = "test"

        for rec in intermediate_records:
            g_hash = rec["video_url_hash"]
            rec["split"] = group_to_split[g_hash]

        self.records = intermediate_records
        logger.info(
            f"Processed {total_count} records across {n_groups} unique video groups. "
            f"Split assignment: Train groups={len(train_groups)}, "
            f"Val groups={len(val_groups)}, Test groups={len(test_groups)}"
        )
        return self.records

    def save_cleaned_csv(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """Saves cleaned dataset to CSV."""
        if not self.records:
            self.process()

        out_path = Path(output_path) if output_path else self.output_csv_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(self.records[0].keys())

        with open(out_path, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.records)

        logger.info(f"Cleaned dataset saved successfully to {out_path} ({len(self.records)} rows)")
        return out_path
