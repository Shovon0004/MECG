"""
Unit tests for VideoValidator and VideoDownloader.
Validates synthetic video inspection, caching, failure handling, and report generation.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.vision.video_downloader import VideoDownloader
from src.vision.video_validator import VideoValidator


class TestVideoValidatorAndDownloader(unittest.TestCase):
    """Tests video inspection and downloading logic."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)

        # Generate a small 1-second synthetic MP4 test video with audio using ffmpeg
        cls.synth_video_path = cls.temp_path / "synthetic_test.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", "testsrc=duration=1:size=320x240:rate=10",
            "-f", "lavfi",
            "-i", "sine=frequency=1000:duration=1",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-c:a", "aac",
            str(cls.synth_video_path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to generate synthetic video: {proc.stderr}")

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_inspect_valid_video(self):
        info = VideoValidator.inspect(self.synth_video_path)
        self.assertTrue(info["is_valid"])
        self.assertEqual(info["width"], 320)
        self.assertEqual(info["height"], 240)
        self.assertAlmostEqual(info["duration_seconds"], 1.0, delta=0.2)
        self.assertTrue(info["has_audio"])
        self.assertEqual(info["video_codec"], "h264")
        self.assertGreater(info["file_size_bytes"], 0)

    def test_inspect_nonexistent_and_empty_file(self):
        # Nonexistent
        res_nonexistent = VideoValidator.inspect(self.temp_path / "does_not_exist.mp4")
        self.assertFalse(res_nonexistent["is_valid"])
        self.assertEqual(res_nonexistent["error"], "File does not exist")

        # Empty 0-byte file
        empty_file = self.temp_path / "empty.mp4"
        empty_file.touch()
        res_empty = VideoValidator.inspect(empty_file)
        self.assertFalse(res_empty["is_valid"])
        self.assertIn("0 bytes", res_empty["error"])

    def test_target_path_deduplication(self):
        downloader = VideoDownloader(video_dir=self.temp_path)
        url1 = "https://example.com/video_sample.m3u8"
        path1 = downloader.get_target_path(url1)
        path2 = downloader.get_target_path(url1)
        self.assertEqual(path1, path2)
        self.assertTrue(path1.name.endswith(".mp4"))

    def test_missing_url_handling(self):
        downloader = VideoDownloader(video_dir=self.temp_path)
        res = downloader.download_single_video("", review_id="REV_NO_URL")
        self.assertEqual(res["status"], "SKIPPED_NO_URL")
        self.assertIsNotNone(res["error"])

    def test_caching_and_resume(self):
        downloader = VideoDownloader(video_dir=self.temp_path)
        test_url = "https://example.com/stream.m3u8"
        target_path = downloader.get_target_path(test_url)

        # Copy synthetic video to target path to simulate existing cache
        import shutil
        shutil.copy(str(self.synth_video_path), str(target_path))

        # Test download invocation
        res = downloader.download_single_video(test_url, review_id="TEST_REV")
        self.assertEqual(res["status"], "CACHED")
        self.assertTrue(res["metadata"]["is_valid"])
        self.assertEqual(res["metadata"]["width"], 320)

    def test_report_saving(self):
        report_file = self.temp_path / "test_report.json"
        downloader = VideoDownloader(video_dir=self.temp_path, report_path=report_file)
        downloader.report["summary"]["total_queried"] = 1
        downloader.report["summary"]["cached_existing"] = 1
        saved = downloader.save_report()
        self.assertTrue(saved.exists())
        self.assertGreater(saved.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
