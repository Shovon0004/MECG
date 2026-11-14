"""
Unit tests for MECG preprocessing, normalization, and data hygiene modules.
Uses standard Python unittest framework.
"""

import os
import unittest
from pathlib import Path

from src.preprocessing.dataset_cleaner import ComVIDCleaner
from src.preprocessing.normalizers import (
    clean_text_whitespace,
    compute_url_hash,
    detect_language_hint,
    normalize_aspects,
    normalize_emotion,
    parse_html_rating,
)


class TestNormalizers(unittest.TestCase):
    """Tests normalizer helper functions."""

    def test_parse_html_rating(self):
        self.assertEqual(
            parse_html_rating('<span class="a-icon-alt">1.0 out of 5 stars</span>'), 1.0
        )
        self.assertEqual(
            parse_html_rating('<span class="a-icon-alt">2.0 out of 5 stars</span>'), 2.0
        )
        self.assertEqual(
            parse_html_rating('<span class="s-icon-alt">1.0 out of 5 waste</span>'), 1.0
        )
        self.assertEqual(parse_html_rating("3/5"), 3.0)
        self.assertIsNone(parse_html_rating(""))
        self.assertIsNone(parse_html_rating("No rating provided"))

    def test_clean_text_whitespace(self):
        raw = "  First line.\nSecond line with   extra spaces\r\nand &amp; symbols.  "
        cleaned = clean_text_whitespace(raw)
        self.assertEqual(cleaned, "First line. Second line with extra spaces and & symbols.")

    def test_normalize_emotion(self):
        self.assertEqual(normalize_emotion("blame"), "blame")
        self.assertEqual(normalize_emotion("Blame"), "blame")
        self.assertEqual(normalize_emotion("disappointment"), "disappointment")
        self.assertEqual(normalize_emotion("dissatisfaction"), "dissatisfaction")
        self.assertEqual(normalize_emotion("frustration"), "frustration")
        self.assertEqual(normalize_emotion("   blame  "), "blame")

    def test_normalize_aspects_typos_and_casing(self):
        # Typos in Quality, Usability, Manufacturing
        raw = "Qualtiy, Usabilty, Manufaturing."
        normalized = normalize_aspects(raw)
        self.assertIn("Quality", normalized)
        self.assertIn("Usability", normalized)
        self.assertIn("Manufacturing", normalized)
        self.assertEqual(len(normalized), 3)

    def test_normalize_aspects_separators(self):
        # Semicolons and conjunctions
        raw = "Quality; Usability and Durability / Colour."
        normalized = normalize_aspects(raw)
        self.assertEqual(normalized, ["Colour", "Durability", "Quality", "Usability"])

    def test_semantic_distinction_preserved(self):
        # Ensure Usability, Functionality, Performance, Durability remain distinct
        raw = "Usability, Functionality, Performance, Durability"
        normalized = normalize_aspects(raw)
        self.assertEqual(len(normalized), 4)
        self.assertEqual(
            normalized, ["Durability", "Functionality", "Performance", "Usability"]
        )

    def test_compute_url_hash(self):
        url1 = "https://example.com/video1.m3u8"
        url2 = "https://example.com/video2.m3u8"
        h1 = compute_url_hash(url1)
        h2 = compute_url_hash(url2)
        self.assertEqual(len(h1), 16)
        self.assertNotEqual(h1, h2)
        # Empty fallback deterministic check
        hf1 = compute_url_hash("", fallback_id="REV001")
        hf2 = compute_url_hash("", fallback_id="REV001")
        self.assertEqual(hf1, hf2)

    def test_detect_language_hint(self):
        self.assertEqual(detect_language_hint("This is English text."), "en")
        self.assertEqual(detect_language_hint("প্লাস্টিকের বোতল ভাঙা"), "bn")
        self.assertEqual(detect_language_hint("यह उत्पाद खराब है"), "hi")


class TestCleanerAndLeakage(unittest.TestCase):
    """Tests full ComVIDCleaner pipeline and data leakage prevention."""

    @classmethod
    def setUpClass(cls):
        raw_csv = Path("data/Com-VID_data.csv")
        if not raw_csv.exists():
            raise unittest.SkipTest("Com-VID_data.csv not found in data/.")
        cls.cleaner = ComVIDCleaner(
            raw_csv_path="data/Com-VID_data.csv",
            output_csv_path="data/processed/ComVID_cleaned.csv",
            random_seed=42,
        )
        cls.records = cls.cleaner.process()

    def test_row_count_preservation(self):
        # Rule: Never delete duplicate records without a documented reason
        self.assertEqual(len(self.records), 1175)

    def test_raw_columns_preserved(self):
        sample = self.records[0]
        self.assertIn("raw_id", sample)
        self.assertIn("raw_title", sample)
        self.assertIn("raw_rating", sample)
        self.assertIn("raw_video_source", sample)
        self.assertIn("raw_text", sample)
        self.assertIn("raw_ground_truth", sample)
        self.assertIn("raw_emotion", sample)
        self.assertIn("raw_aspect", sample)
        self.assertIn("raw_domain", sample)

    def test_zero_video_leakage_across_splits(self):
        # Strict rule: 0% video leakage across train, val, and test
        splits_by_url_hash = {}
        for r in self.records:
            h = r["video_url_hash"]
            s = r["split"]
            if h in splits_by_url_hash:
                self.assertEqual(
                    splits_by_url_hash[h],
                    s,
                    f"Video leakage detected! Hash {h} assigned to both {splits_by_url_hash[h]} and {s}",
                )
            else:
                splits_by_url_hash[h] = s

    def test_duplicate_flags_populated(self):
        duplicate_id_records = [r for r in self.records if r["is_duplicate_id"]]
        self.assertGreater(len(duplicate_id_records), 0)
        duplicate_url_records = [r for r in self.records if r["is_duplicate_url"]]
        self.assertGreater(len(duplicate_url_records), 0)


if __name__ == "__main__":
    unittest.main()
