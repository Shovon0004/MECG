"""
Frame extraction module for MECG.
Extracts N uniformly sampled timestamped frames from video using OpenCV or FFmpeg.
Stores frame images and metadata under data/processed/frames/{video_hash}/.
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import yaml
except ImportError:
    yaml = None

from src.preprocessing.normalizers import compute_url_hash
from src.vision.video_validator import VideoValidator

logger = logging.getLogger("MECG.FrameExtractor")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class FrameExtractor:
    """Samples timestamped frames uniformly across video duration."""

    def __init__(
        self,
        frames_dir: Union[str, Path] = "data/processed/frames",
        sample_frames: int = 8,
        target_resolution: Tuple[int, int] = (448, 448),
    ):
        self.frames_dir = Path(frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.sample_frames = sample_frames
        self.target_resolution = target_resolution

    @classmethod
    def from_config(cls, config_path: Union[str, Path] = "configs/config.yaml") -> "FrameExtractor":
        """Instantiates FrameExtractor from project configuration."""
        cfg_path = Path(config_path)
        if cfg_path.exists() and yaml:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            paths = cfg.get("paths", {})
            v_proc = cfg.get("video_processing", {})
            frame_size = v_proc.get("frame_size", [448, 448])
            return cls(
                frames_dir=paths.get("frames_dir", "data/processed/frames"),
                sample_frames=v_proc.get("sample_frames", 8),
                target_resolution=(frame_size[0], frame_size[1]),
            )
        return cls()

    def extract_frames(
        self,
        video_path: Union[str, Path],
        video_id: str = "",
        force_extract: bool = False,
    ) -> Dict[str, Any]:
        """
        Extracts sample_frames timestamped frames uniformly spaced across video duration.
        Uses FFmpeg for precision and codec robustness.
        Returns:
            {
                "video_id": str,
                "video_path": str,
                "total_frames_extracted": int,
                "duration_seconds": float,
                "frames": [
                    {
                        "index": int,
                        "timestamp": float,
                        "file_path": str
                    }, ...
                ]
            }
        """
        v_path = Path(video_path)
        if not v_path.exists():
            raise FileNotFoundError(f"Video file not found: {v_path}")

        v_name = v_path.stem  # e.g. {video_hash}
        out_subfolder = self.frames_dir / v_name
        meta_file = out_subfolder / "metadata.json"

        # Check Cache
        if not force_extract and meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("total_frames_extracted", 0) == self.sample_frames:
                    logger.info(f"Using cached frames for video: {v_name}")
                    return cached
            except Exception:
                pass

        out_subfolder.mkdir(parents=True, exist_ok=True)

        # Inspect duration via VideoValidator
        meta = VideoValidator.inspect(v_path)
        duration = meta.get("duration_seconds", 0.0)
        if duration <= 0:
            duration = 1.0  # Fallback minimum

        # Calculate uniform timestamps
        # For N frames, sample at (i + 0.5) * (duration / N)
        interval = duration / self.sample_frames
        timestamps = [round((i + 0.5) * interval, 3) for i in range(self.sample_frames)]

        extracted_frames: List[Dict[str, Any]] = []

        w, h = self.target_resolution
        for idx, ts in enumerate(timestamps):
            out_img_path = out_subfolder / f"frame_{idx:03d}_{ts:.2f}s.jpg"

            cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(ts),
                "-i", str(v_path.resolve()),
                "-frames:v", "1",
                "-vf", f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,format=yuvj420p",
                "-q:v", "2",
                str(out_img_path.resolve()),
            ]

            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode == 0 and out_img_path.exists():
                extracted_frames.append({
                    "index": idx,
                    "timestamp": ts,
                    "file_path": str(out_img_path.resolve()),
                })
            else:
                logger.warning(f"Could not extract frame at {ts}s: {proc.stderr.strip()}")

        result = {
            "video_id": video_id or v_name,
            "video_path": str(v_path.resolve()),
            "total_frames_extracted": len(extracted_frames),
            "target_frames": self.sample_frames,
            "duration_seconds": duration,
            "frames": extracted_frames,
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Extracted {len(extracted_frames)}/{self.sample_frames} frames for {v_name}")
        return result
