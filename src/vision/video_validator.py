"""
Video validation and metadata extraction using ffprobe.
Extracts duration, FPS, resolution, codec, and audio stream presence.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger("MECG.VideoValidator")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class VideoValidator:
    """Validates video files and extracts detailed stream metadata via ffprobe."""

    @staticmethod
    def inspect(video_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Runs ffprobe on the video file and returns a structured dictionary of metadata.
        Returns:
            {
                "is_valid": bool,
                "file_path": str,
                "file_size_bytes": int,
                "duration_seconds": float,
                "fps": float,
                "width": int,
                "height": int,
                "video_codec": str,
                "has_audio": bool,
                "audio_codec": Optional[str],
                "error": Optional[str]
            }
        """
        path = Path(video_path)
        result: Dict[str, Any] = {
            "is_valid": False,
            "file_path": str(path.resolve()),
            "file_size_bytes": 0,
            "duration_seconds": 0.0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "video_codec": "",
            "has_audio": False,
            "audio_codec": None,
            "error": None,
        }

        if not path.exists():
            result["error"] = "File does not exist"
            return result

        file_size = path.stat().st_size
        result["file_size_bytes"] = file_size
        if file_size == 0:
            result["error"] = "File size is 0 bytes (empty file)"
            return result

        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate,duration:format=duration",
            "-of", "json",
            str(path.resolve()),
        ]

        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
            if proc.returncode != 0:
                result["error"] = f"ffprobe error: {proc.stderr.strip()}"
                return result

            data = json.loads(proc.stdout)
            streams = data.get("streams", [])
            fmt = data.get("format", {})

            # Parse video stream
            video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
            audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

            if video_stream is None:
                result["error"] = "No video stream found in container"
                return result

            result["video_codec"] = video_stream.get("codec_name", "")
            result["width"] = int(video_stream.get("width", 0))
            result["height"] = int(video_stream.get("height", 0))

            # FPS parsing (e.g. "30/1" or "29.97")
            r_fps = video_stream.get("r_frame_rate", "")
            if "/" in r_fps:
                num, den = r_fps.split("/")
                try:
                    result["fps"] = round(float(num) / float(den), 2) if float(den) > 0 else 0.0
                except (ValueError, ZeroDivisionError):
                    result["fps"] = 0.0
            elif r_fps:
                try:
                    result["fps"] = round(float(r_fps), 2)
                except ValueError:
                    result["fps"] = 0.0

            # Duration parsing (stream level or format level)
            duration_str = video_stream.get("duration") or fmt.get("duration")
            if duration_str:
                try:
                    result["duration_seconds"] = round(float(duration_str), 2)
                except ValueError:
                    result["duration_seconds"] = 0.0

            if audio_stream:
                result["has_audio"] = True
                result["audio_codec"] = audio_stream.get("codec_name")

            result["is_valid"] = result["width"] > 0 and result["height"] > 0 and result["duration_seconds"] > 0
            return result

        except subprocess.TimeoutExpired:
            result["error"] = "ffprobe execution timed out"
            return result
        except Exception as e:
            result["error"] = f"Unexpected inspection error: {str(e)}"
            return result
