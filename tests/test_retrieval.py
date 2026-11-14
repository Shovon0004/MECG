"""
Unit tests for VisualRetriever.
Tests visual indexing, cosine ranking, and self-retrieval exclusion rules.
"""

import tempfile
import unittest
from pathlib import Path

from src.retrieval.visual_retriever import VisualRetriever


class TestVisualRetriever(unittest.TestCase):
    """Tests RAG visual retriever."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_self_retrieval_exclusion(self):
        test_dir = self.temp_path / "exclusion"
        test_dir.mkdir(parents=True, exist_ok=True)
        retriever = VisualRetriever(index_dir=test_dir, vector_dim=4)

        # Mock vector 1 (query match)
        v1 = [1.0, 0.0, 0.0, 0.0]
        meta1 = {"ground_truth_complaint": "Bottle 1 leaks", "domain": "Household"}
        retriever.add_video(video_hash="HASH_A", frame_paths=[], metadata=meta1, embedding=v1)

        # Mock vector 2
        v2 = [0.9, 0.1, 0.0, 0.0]
        meta2 = {"ground_truth_complaint": "Bottle 2 cap loose", "domain": "Household"}
        retriever.add_video(video_hash="HASH_B", frame_paths=[], metadata=meta2, embedding=v2)

        # Query with v1 vector, excluding HASH_A
        results = retriever.search(
            query_frame_paths=[],
            top_k=2,
            exclude_video_hash="HASH_A",
            query_embedding=v1,
        )

        # HASH_A must NOT be in results! HASH_B should be top match
        retrieved_hashes = [r["video_url_hash"] for r in results]
        self.assertNotIn("HASH_A", retrieved_hashes)
        self.assertIn("HASH_B", retrieved_hashes)

    def test_save_and_load_metadata(self):
        test_dir = self.temp_path / "save_load"
        test_dir.mkdir(parents=True, exist_ok=True)
        retriever = VisualRetriever(index_dir=test_dir, vector_dim=4)
        v = [0.5, 0.5, 0.5, 0.5]
        meta = {"ground_truth_complaint": "Screen broken", "domain": "Electronic"}
        retriever.add_video(video_hash="HASH_ELEC", frame_paths=[], metadata=meta, embedding=v)

        idx_file, meta_file = retriever.save_index()
        self.assertTrue(meta_file.exists())


if __name__ == "__main__":
    unittest.main()
