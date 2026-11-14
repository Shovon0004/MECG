"""
Comprehensive dataset audit and quality reporting module for Com-VID.
Generates structured JSON reports covering duplicates, missing data,
label distributions, text statistics, and data leakage verification.
"""

import csv
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("MECG.Audit")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class DatasetAuditor:
    """Generates structured quality audit reports from cleaned Com-VID data."""

    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    @classmethod
    def from_csv(cls, csv_path: Union[str, Path] = "data/processed/ComVID_cleaned.csv") -> "DatasetAuditor":
        """Instantiates auditor by loading a cleaned CSV file."""
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Cleaned CSV not found at {path}")

        records = []
        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Cast numeric fields where appropriate
                if row.get("aspect_count"):
                    row["aspect_count"] = int(row["aspect_count"])
                if row.get("text_char_len"):
                    row["text_char_len"] = int(row["text_char_len"])
                if row.get("text_word_len"):
                    row["text_word_len"] = int(row["text_word_len"])
                if row.get("ground_truth_char_len"):
                    row["ground_truth_char_len"] = int(row["ground_truth_char_len"])
                if row.get("ground_truth_word_len"):
                    row["ground_truth_word_len"] = int(row["ground_truth_word_len"])
                if row.get("id_occurrence_count"):
                    row["id_occurrence_count"] = int(row["id_occurrence_count"])
                if row.get("url_occurrence_count"):
                    row["url_occurrence_count"] = int(row["url_occurrence_count"])
                row["is_duplicate_id"] = str(row.get("is_duplicate_id", "")).lower() == "true"
                row["is_duplicate_url"] = str(row.get("is_duplicate_url", "")).lower() == "true"
                row["has_video_url"] = str(row.get("has_video_url", "")).lower() == "true"
                records.append(row)
        return cls(records)

    def generate_report(self) -> Dict[str, Any]:
        """Calculates audit metrics and returns structured dictionary."""
        total_records = len(self.records)
        if total_records == 0:
            return {"error": "Empty dataset"}

        # 1. Duplicates & Identifiers
        id_counts = Counter(r["review_id"] for r in self.records if r.get("review_id"))
        dup_ids = {k: v for k, v in id_counts.items() if v > 1}

        url_counts = Counter(r["video_url"] for r in self.records if r.get("video_url"))
        dup_urls = {k: v for k, v in url_counts.items() if v > 1}
        zero_urls = [r["review_id"] for r in self.records if not r.get("video_url")]

        # 2. Missing Values Analysis
        missing_counts: Dict[str, int] = defaultdict(int)
        for r in self.records:
            if not r.get("raw_video_source"):
                missing_counts["raw_video_source"] += 1
            if not r.get("raw_aspect"):
                missing_counts["raw_aspect"] += 1
            if not r.get("rating_stars"):
                missing_counts["rating_stars"] += 1
            if not r.get("aspects"):
                missing_counts["normalized_aspects"] += 1

        # 3. Label Distributions
        domain_counts = Counter(r["domain"] for r in self.records)
        emotion_counts = Counter(r["emotion"] for r in self.records)
        rating_counts = Counter(r["rating_stars"] for r in self.records)

        # Aspect statistics
        all_aspects: List[str] = []
        aspect_len_dist: Counter = Counter()
        for r in self.records:
            aspect_str = r.get("aspects", "")
            if aspect_str:
                items = [x.strip() for x in aspect_str.split(",") if x.strip()]
                all_aspects.extend(items)
                aspect_len_dist[len(items)] += 1
            else:
                aspect_len_dist[0] += 1

        aspect_counter = Counter(all_aspects)

        # 4. Group-aware split audit & leakage check
        split_records = Counter(r["split"] for r in self.records)
        split_groups: Dict[str, set] = defaultdict(set)
        for r in self.records:
            split_groups[r["split"]].add(r["video_url_hash"])

        train_groups = split_groups.get("train", set())
        val_groups = split_groups.get("val", set())
        test_groups = split_groups.get("test", set())

        train_val_leakage = len(train_groups.intersection(val_groups))
        train_test_leakage = len(train_groups.intersection(test_groups))
        val_test_leakage = len(val_groups.intersection(test_groups))

        # 5. Text length statistics
        char_lens_text = [r.get("text_char_len", 0) for r in self.records]
        word_lens_text = [r.get("text_word_len", 0) for r in self.records]
        char_lens_gt = [r.get("ground_truth_char_len", 0) for r in self.records]
        word_lens_gt = [r.get("ground_truth_word_len", 0) for r in self.records]

        def get_stats(data: List[int]) -> Dict[str, float]:
            if not data:
                return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
            s = sorted(data)
            n = len(s)
            med = (s[n // 2] if n % 2 != 0 else (s[n // 2 - 1] + s[n // 2]) / 2.0)
            return {
                "min": int(min(data)),
                "max": int(max(data)),
                "mean": round(float(sum(data) / n), 2),
                "median": round(float(med), 2),
            }

        report: Dict[str, Any] = {
            "metadata": {
                "audit_timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset_name": "Com-VID",
                "total_records": total_records,
                "dataset_rules_enforced": [
                    "Preserve all original raw columns and values",
                    "Flag duplicate IDs and video URLs without deleting",
                    "Normalize known typos and punctuation without merging distinct aspect concepts",
                    "Group-aware split to strictly enforce 0% video leakage across splits",
                    "Distinguish customer text from future audio transcripts",
                    "Acknowledge that ground-truth complaint may mention non-visual defects"
                ]
            },
            "identifiers_and_duplicates": {
                "unique_review_ids": len(id_counts),
                "duplicate_review_ids_count": len(dup_ids),
                "sample_duplicate_ids": {k: dup_ids[k] for k in list(dup_ids.keys())[:5]},
                "unique_video_urls": len(url_counts),
                "duplicate_video_urls_count": len(dup_urls),
                "records_with_missing_video_url": len(zero_urls),
                "missing_url_review_ids": zero_urls,
            },
            "missing_values": dict(missing_counts),
            "distributions": {
                "domains": {k: {"count": v, "percentage": round(v / total_records * 100, 2)} for k, v in domain_counts.items()},
                "emotions": {k: {"count": v, "percentage": round(v / total_records * 100, 2)} for k, v in emotion_counts.items()},
                "ratings": {str(k): {"count": v, "percentage": round(v / total_records * 100, 2)} for k, v in rating_counts.items()},
                "aspects": {
                    "total_unique_normalized_aspects": len(aspect_counter),
                    "top_aspects": [
                        {"aspect": asp, "count": cnt, "percentage": round(cnt / total_records * 100, 2)}
                        for asp, cnt in aspect_counter.most_common(20)
                    ],
                    "aspect_count_per_review_distribution": {
                        f"{k}_aspects": v for k, v in sorted(aspect_len_dist.items())
                    }
                }
            },
            "splits": {
                "records_per_split": {
                    k: {"count": v, "percentage": round(v / total_records * 100, 2)}
                    for k, v in split_records.items()
                },
                "unique_video_groups_per_split": {
                    k: len(v) for k, v in split_groups.items()
                },
                "video_leakage_check": {
                    "train_val_overlap_count": train_val_leakage,
                    "train_test_overlap_count": train_test_leakage,
                    "val_test_overlap_count": val_test_leakage,
                    "is_leakage_free": (train_val_leakage == 0 and train_test_leakage == 0 and val_test_leakage == 0)
                }
            },
            "text_statistics": {
                "customer_text_characters": get_stats(char_lens_text),
                "customer_text_words": get_stats(word_lens_text),
                "ground_truth_characters": get_stats(char_lens_gt),
                "ground_truth_words": get_stats(word_lens_gt),
            }
        }
        return report

    def save_report(self, output_path: Union[str, Path] = "reports/dataset_audit.json") -> Path:
        """Saves audit report to JSON file."""
        report = self.generate_report()
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, mode="w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Audit report saved to {out_path}")
        return out_path
