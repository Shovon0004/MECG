"""
Evidence extraction pipeline connecting FrameExtractor and VideoAnalyzer.
Produces timestamp-aligned physical evidence logs from raw review videos.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.vision.frame_extractor import FrameExtractor
from src.vision.video_analyzer import VideoAnalyzer

logger = logging.getLogger("MECG.EvidenceExtractor")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class EvidenceExtractor:
    """Orchestrates video frame sampling and Qwen3-VL visual defect analysis."""

    def __init__(
        self,
        frame_extractor: Optional[FrameExtractor] = None,
        video_analyzer: Optional[VideoAnalyzer] = None,
    ):
        self.frame_extractor = frame_extractor or FrameExtractor.from_config()
        self.video_analyzer = video_analyzer or VideoAnalyzer.from_config()

    def extract_evidence(
        self,
        video_path: Union[str, Path],
        customer_text: str = "",
        product_domain: str = "",
        video_id: str = "",
    ) -> Dict[str, Any]:
        """
        Extracts timestamped visual evidence from video.
        Returns:
            {
                "video_id": str,
                "video_path": str,
                "duration_seconds": float,
                "frames_analyzed": int,
                "visible_observations": [...],
                "possible_aspects": [...],
                "limitations": [...]
            }
        """
        v_path = Path(video_path)
        if not v_path.exists():
            return {
                "video_id": video_id,
                "video_path": str(video_path),
                "error": f"Video file not found: {video_path}",
                "visible_observations": [],
                "possible_aspects": [],
                "limitations": ["Video file does not exist"],
            }

        # Step 1: Sample frames
        frame_result = self.frame_extractor.extract_frames(v_path, video_id=video_id)
        frames = frame_result.get("frames", [])

        # Step 2: Analyze frames with Qwen3-VL
        analysis = self.video_analyzer.analyze_frames(
            frames=frames,
            product_domain=product_domain,
            customer_review_hint=customer_text,
        )

        return {
            "video_id": video_id or v_path.stem,
            "video_path": str(v_path.resolve()),
            "duration_seconds": frame_result.get("duration_seconds", 0.0),
            "frames_analyzed": len(frames),
            "visible_observations": analysis.get("visible_observations", []),
            "possible_aspects": analysis.get("possible_aspects", []),
            "limitations": analysis.get("limitations", []),
        }
