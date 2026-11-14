"""
Unit test for MECGPipeline end-to-end execution.
"""

import subprocess
import tempfile
import unittest
from pathlib import Path

from src.pipeline import MECGPipeline


class TestPipeline(unittest.TestCase):
    """Tests the complete end-to-end MECG pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)

        cls.synth_video = cls.temp_path / "test_pipeline_vid.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=10",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-an",
            str(cls.synth_video),
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_pipeline_execution_and_schema(self):
        pipeline = MECGPipeline()
        result = pipeline.process(
            video_path=self.synth_video,
            customer_text="The bottle cap is cracked and loose.",
            video_id="TEST_001",
            product_domain="Household Item",
            requested_output_language="en",
        )

        # Verify output schema
        required_keys = [
            "video_id",
            "detected_language",
            "requested_output_language",
            "transcript_original",
            "transcript_english",
            "visual_evidence",
            "aspects",
            "emotion",
            "retrieved_examples",
            "generated_complaint",
            "verified_claims",
            "unsupported_claims",
            "evidence_grounding_score",
        ]
        for k in required_keys:
            self.assertIn(k, result, f"Missing required pipeline key: {k}")

        self.assertEqual(result["video_id"], "TEST_001")
        self.assertEqual(result["requested_output_language"], "en")
        self.assertIsInstance(result["evidence_grounding_score"], (float, int))
        self.assertGreaterEqual(result["evidence_grounding_score"], 0.0)
        self.assertLessEqual(result["evidence_grounding_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
