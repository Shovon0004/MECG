"""
Audio extraction module for MECG.
Extracts 16kHz mono WAV audio from product review videos for Whisper speech recognition.
Handles videos without audio gracefully.
"""

import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None

from src.vision.video_validator import VideoValidator

logger = logging.getLogger("MECG.AudioExtractor")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class AudioExtractor:
    """Extracts standardized 16kHz mono WAV audio tracks from videos."""

    def __init__(
        self,
        audio_dir: Union[str, Path] = "data/processed/audio",
        sample_rate: int = 16000,
        channels: int = 1,
    ):
        self.audio_dir = Path(audio_dir)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.channels = channels

    @classmethod
    def from_config(cls, config_path: Union[str, Path] = "configs/config.yaml") -> "AudioExtractor":
        """Instantiates AudioExtractor from YAML config."""
        cfg_path = Path(config_path)
        if cfg_path.exists() and yaml:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            paths = cfg.get("paths", {})
            v_proc = cfg.get("video_processing", {})
            return cls(
                audio_dir=paths.get("audio_dir", "data/processed/audio"),
                sample_rate=v_proc.get("audio_sample_rate", 16000),
                channels=v_proc.get("audio_channels", 1),
            )
        return cls()

    def extract_audio(
        self,
        video_path: Union[str, Path],
        video_id: str = "",
        force_extract: bool = False,
    ) -> Dict[str, Any]:
        """
        Extracts 16kHz mono WAV audio from video.
        Returns:
            {
                "video_id": str,
                "video_path": str,
                "audio_path": Optional[str],
                "has_audio": bool,
                "sample_rate": int,
                "channels": int,
                "duration_seconds": float,
                "error": Optional[str]
            }
        """
        v_path = Path(video_path)
        if not v_path.exists():
            raise FileNotFoundError(f"Video file not found: {v_path}")

        v_name = v_path.stem
        out_audio_path = self.audio_dir / f"{v_name}.wav"

        result: Dict[str, Any] = {
            "video_id": video_id or v_name,
            "video_path": str(v_path.resolve()),
            "audio_path": None,
            "has_audio": False,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "duration_seconds": 0.0,
            "error": None,
        }

        # Inspect stream presence
        meta = VideoValidator.inspect(v_path)
        if not meta.get("has_audio"):
            logger.info(f"Video {v_name} does not contain an audio stream.")
            result["has_audio"] = False
            return result

        result["has_audio"] = True
        result["duration_seconds"] = meta.get("duration_seconds", 0.0)

        # Check Cache
        if not force_extract and out_audio_path.exists() and out_audio_path.stat().st_size > 44:
            result["audio_path"] = str(out_audio_path.resolve())
            logger.info(f"Using cached audio for video: {v_name}")
            return result

        # Extract audio using FFmpeg
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(v_path.resolve()),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate),
            "-ac", str(self.channels),
            str(out_audio_path.resolve()),
        ]

        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0 and out_audio_path.exists() and out_audio_path.stat().st_size > 44:
            result["audio_path"] = str(out_audio_path.resolve())
            logger.info(f"Successfully extracted audio: {out_audio_path.name}")
        else:
            result["error"] = f"FFmpeg audio extraction failed: {proc.stderr.strip()}"
            logger.error(result["error"])

        return result
