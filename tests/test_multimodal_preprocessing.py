"""
Unit tests for FrameExtractor, AudioExtractor, and WhisperTranscriber.
Verifies frame sampling, audio extraction, no-audio handling, and JSONL transcription logging.
"""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from src.audio.audio_extractor import AudioExtractor
from src.audio.transcriber import WhisperTranscriber
from src.vision.frame_extractor import FrameExtractor


class TestMultimodalPreprocessing(unittest.TestCase):
    """Tests frame extraction and audio extraction."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)

        # 1. Create synthetic video with audio (2 seconds)
        cls.video_with_audio = cls.temp_path / "video_with_audio.mp4"
        cmd_with_audio = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=2",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac",
            str(cls.video_with_audio),
        ]
        subprocess.run(cmd_with_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # 2. Create synthetic video WITHOUT audio (1 second)
        cls.video_no_audio = cls.temp_path / "video_no_audio.mp4"
        cmd_no_audio = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=10",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-an",
            str(cls.video_no_audio),
        ]
        subprocess.run(cmd_no_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_frame_extractor_uniform_sampling(self):
        frames_out = self.temp_path / "frames"
        extractor = FrameExtractor(frames_dir=frames_out, sample_frames=4, target_resolution=(224, 224))
        res = extractor.extract_frames(self.video_with_audio, video_id="SYNTH_01")

        self.assertEqual(res["total_frames_extracted"], 4)
        self.assertEqual(len(res["frames"]), 4)

        prev_ts = -1.0
        for f in res["frames"]:
            self.assertGreater(f["timestamp"], prev_ts)
            prev_ts = f["timestamp"]
            p = Path(f["file_path"])
            self.assertTrue(p.exists())
            self.assertGreater(p.stat().st_size, 0)

        # Verify caching
        res_cached = extractor.extract_frames(self.video_with_audio, video_id="SYNTH_01")
        self.assertEqual(res_cached["total_frames_extracted"], 4)

    def test_audio_extractor_valid_audio(self):
        audio_out = self.temp_path / "audio"
        extractor = AudioExtractor(audio_dir=audio_out, sample_rate=16000, channels=1)
        res = extractor.extract_audio(self.video_with_audio, video_id="SYNTH_AUDIO")

        self.assertTrue(res["has_audio"])
        self.assertIsNotNone(res["audio_path"])
        p = Path(res["audio_path"])
        self.assertTrue(p.exists())
        self.assertGreater(p.stat().st_size, 1000)

    def test_audio_extractor_no_audio(self):
        audio_out = self.temp_path / "audio"
        extractor = AudioExtractor(audio_dir=audio_out)
        res = extractor.extract_audio(self.video_no_audio, video_id="SYNTH_NO_AUDIO")

        self.assertFalse(res["has_audio"])
        self.assertIsNone(res["audio_path"])

    def test_transcriber_no_audio_handling(self):
        jsonl_out = self.temp_path / "transcripts.jsonl"
        transcriber = WhisperTranscriber(output_jsonl_path=jsonl_out)
        res = transcriber.transcribe(None, video_id="REV_NO_AUDIO")

        self.assertFalse(res["has_audio"])
        self.assertEqual(res["transcript"], "")
        self.assertTrue(jsonl_out.exists())

        # Verify jsonl line content
        with open(jsonl_out, "r", encoding="utf-8") as f:
            line = json.loads(f.readline())
        self.assertEqual(line["video_id"], "REV_NO_AUDIO")


if __name__ == "__main__":
    unittest.main()
