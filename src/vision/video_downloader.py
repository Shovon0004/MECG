"""
Robust video download and validation pipeline for the Com-VID dataset.
Supports direct MP4 URLs, Amazon HLS M3U8 streams, download caching,
deduplication via video_url_hash, and pilot execution mode.
"""

import csv
import json
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None

from src.preprocessing.normalizers import compute_url_hash
from src.vision.video_validator import VideoValidator

logger = logging.getLogger("MECG.VideoDownloader")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class VideoDownloader:
    """
    Downloads and caches product review videos from Amazon HLS (M3U8) and direct MP4 sources.
    Enforces deduplication, resume functionality, and validation.
    """

    def __init__(
        self,
        video_dir: Union[str, Path] = "data/videos",
        report_path: Union[str, Path] = "reports/video_download_report.json",
        timeout_seconds: int = 60,
        max_retries: int = 3,
        pilot_mode: bool = False,
        pilot_size: int = 5,
    ):
        self.video_dir = Path(video_dir)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.report_path = Path(report_path)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.pilot_mode = pilot_mode
        self.pilot_size = pilot_size

        # Report tracking dictionary
        self.report: Dict[str, Any] = {
            "metadata": {
                "started_at": datetime.now(timezone.utc).isoformat(),
                "completed_at": None,
                "video_dir": str(self.video_dir.resolve()),
                "pilot_mode": self.pilot_mode,
                "pilot_size": self.pilot_size if self.pilot_mode else None,
                "timeout_seconds": self.timeout_seconds,
                "max_retries": self.max_retries,
            },
            "summary": {
                "total_queried": 0,
                "downloaded_new": 0,
                "cached_existing": 0,
                "failed": 0,
                "skipped_no_url": 0,
            },
            "results": {},
        }

    @classmethod
    def from_config(cls, config_path: Union[str, Path] = "configs/config.yaml") -> "VideoDownloader":
        """Instantiates VideoDownloader using project YAML configuration."""
        cfg_path = Path(config_path)
        if cfg_path.exists() and yaml:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            paths = cfg.get("paths", {})
            v_proc = cfg.get("video_processing", {})
            return cls(
                video_dir=paths.get("video_dir", "data/videos"),
                report_path=paths.get("download_report", "reports/video_download_report.json"),
                timeout_seconds=v_proc.get("download_timeout_seconds", 60),
                max_retries=v_proc.get("max_download_retries", 3),
                pilot_mode=v_proc.get("pilot_mode", False),
                pilot_size=v_proc.get("pilot_size", 5),
            )
        return cls()

    def get_target_path(self, video_url: str, review_id: str = "") -> Path:
        """Returns the canonical target MP4 path named after the URL hash."""
        url_hash = compute_url_hash(video_url, fallback_id=review_id)
        return self.video_dir / f"{url_hash}.mp4"

    def download_single_video(
        self,
        video_url: str,
        review_id: str = "",
        force_download: bool = False,
    ) -> Dict[str, Any]:
        """
        Downloads a single video using FFmpeg.
        Handles direct MP4 or HLS (.m3u8) streams.
        Verifies downloaded video integrity.
        """
        url = video_url.strip() if video_url else ""
        url_hash = compute_url_hash(url, fallback_id=review_id)
        target_path = self.get_target_path(url, review_id=review_id)

        result: Dict[str, Any] = {
            "review_id": review_id,
            "video_url": url,
            "video_url_hash": url_hash,
            "target_path": str(target_path.resolve()),
            "status": "UNKNOWN",
            "metadata": {},
            "attempts": 0,
            "error": None,
        }

        # 1. Handle missing URL
        if not url:
            result["status"] = "SKIPPED_NO_URL"
            result["error"] = "Record does not provide a Video-Source URL"
            return result

        # 2. Check Cache / Resume
        if not force_download and target_path.exists() and target_path.stat().st_size > 0:
            validation = VideoValidator.inspect(target_path)
            if validation.get("is_valid"):
                result["status"] = "CACHED"
                result["metadata"] = validation
                logger.info(f"Video already cached and valid: {target_path.name}")
                return result
            else:
                logger.warning(f"Cached file {target_path.name} is corrupt: {validation.get('error')}. Re-downloading.")
                target_path.unlink(missing_ok=True)

        # 3. Attempt download with retries using FFmpeg
        temp_path = self.video_dir / f"tmp_{url_hash}_{int(time.time())}.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-timeout", str(self.timeout_seconds * 1_000_000),  # microseconds for protocol timeout
            "-i", url,
            "-c", "copy",
            "-bsf:a", "aac_adtstoasc",
            "-movflags", "+faststart",
            str(temp_path.resolve()),
        ]

        # Alternative transcode fallback command if stream-copy fails due to codec/container issues
        fallback_cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", url,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(temp_path.resolve()),
        ]

        attempts = 0
        success = False
        last_error = ""

        while attempts < self.max_retries and not success:
            attempts += 1
            result["attempts"] = attempts
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout_seconds,
                )
                if proc.returncode == 0 and temp_path.exists() and temp_path.stat().st_size > 0:
                    success = True
                else:
                    # Try fallback transcode command
                    proc_fb = subprocess.run(
                        fallback_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=self.timeout_seconds,
                    )
                    if proc_fb.returncode == 0 and temp_path.exists() and temp_path.stat().st_size > 0:
                        success = True
                    else:
                        last_error = proc.stderr.strip() or proc_fb.stderr.strip() or f"FFmpeg exited with code {proc.returncode}"
            except subprocess.TimeoutExpired:
                last_error = f"Download timed out after {self.timeout_seconds}s"
            except Exception as e:
                last_error = f"Exception during download: {str(e)}"

            if not success and attempts < self.max_retries:
                time.sleep(2)

        if success:
            # Atomically move temp file to target path
            shutil.move(str(temp_path), str(target_path))
            validation = VideoValidator.inspect(target_path)
            if validation.get("is_valid"):
                result["status"] = "DOWNLOADED"
                result["metadata"] = validation
                logger.info(f"Successfully downloaded & verified: {target_path.name}")
            else:
                target_path.unlink(missing_ok=True)
                result["status"] = "FAILED"
                result["error"] = f"Downloaded file failed validation: {validation.get('error')}"
                logger.error(f"Validation failure for {target_path.name}: {result['error']}")
        else:
            temp_path.unlink(missing_ok=True)
            result["status"] = "FAILED"
            result["error"] = last_error
            logger.error(f"Failed to download video {url}: {last_error}")

        return result

    def download_dataset(
        self,
        csv_path: Union[str, Path] = "data/processed/ComVID_cleaned.csv",
        limit: Optional[int] = None,
        force_download: bool = False,
    ) -> Dict[str, Any]:
        """
        Processes dataset records, downloading unique videos.
        Skips already processed video URLs to prevent duplicate downloads.
        """
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Cleaned CSV not found at: {path}")

        records = []
        with open(path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            records = list(reader)

        # Apply pilot mode or limit
        effective_limit = limit
        if effective_limit is None and self.pilot_mode:
            effective_limit = self.pilot_size

        if effective_limit is not None:
            records = records[:effective_limit]
            logger.info(f"Running in limited / pilot mode: processing first {len(records)} records")

        # Track processed URLs to avoid re-querying identical URLs in same batch
        seen_urls: Dict[str, Dict[str, Any]] = {}

        for i, row in enumerate(records, 1):
            url = row.get("video_url") or row.get("raw_video_source", "")
            rev_id = row.get("review_id") or row.get("raw_id", f"row_{i}")

            if url in seen_urls:
                # Reuse result of the same URL
                prev_res = dict(seen_urls[url])
                prev_res["review_id"] = rev_id
                self.report["results"][rev_id] = prev_res
                continue

            logger.info(f"[{i}/{len(records)}] Processing video for review {rev_id}...")
            res = self.download_single_video(url, review_id=rev_id, force_download=force_download)
            seen_urls[url] = res
            self.report["results"][rev_id] = res

            status = res["status"]
            if status == "DOWNLOADED":
                self.report["summary"]["downloaded_new"] += 1
            elif status == "CACHED":
                self.report["summary"]["cached_existing"] += 1
            elif status == "FAILED":
                self.report["summary"]["failed"] += 1
            elif status == "SKIPPED_NO_URL":
                self.report["summary"]["skipped_no_url"] += 1

        self.report["summary"]["total_queried"] = len(records)
        self.report["metadata"]["completed_at"] = datetime.now(timezone.utc).isoformat()
        self.save_report()
        return self.report

    def save_report(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """Saves download audit report to JSON."""
        target = Path(output_path) if output_path else self.report_path
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, mode="w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2)
        logger.info(f"Video download report saved to: {target}")
        return target
