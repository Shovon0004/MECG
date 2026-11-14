"""
Multilingual speech recognition module for MECG using Whisper.
Extracts timestamped transcription segments, detects spoken language and confidence,
preserves original transcripts, and appends to data/processed/transcripts.jsonl.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger("MECG.Transcriber")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class WhisperTranscriber:
    """
    Multilingual speech recognition supporting Faster-Whisper and HuggingFace Whisper.
    Detects language automatically, preserves original timestamps, and logs to JSONL.
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3-turbo",
        fallback_model_id: str = "openai/whisper-small",
        device: str = "auto",
        compute_type: str = "auto",
        output_jsonl_path: Union[str, Path] = "data/processed/transcripts.jsonl",
    ):
        self.model_id = model_id
        self.fallback_model_id = fallback_model_id
        self.output_jsonl_path = Path(output_jsonl_path)
        self.output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.compute_type = compute_type
        self.model = None

    @classmethod
    def from_config(cls, config_path: Union[str, Path] = "configs/config.yaml") -> "WhisperTranscriber":
        """Instantiates WhisperTranscriber from YAML config."""
        cfg_path = Path(config_path)
        if cfg_path.exists() and yaml:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            paths = cfg.get("paths", {})
            a_cfg = cfg.get("audio_transcription", {})
            return cls(
                model_id=a_cfg.get("model_id", "openai/whisper-large-v3-turbo"),
                fallback_model_id=a_cfg.get("fallback_model_id", "openai/whisper-small"),
                device=a_cfg.get("device", "auto"),
                compute_type=a_cfg.get("compute_type", "auto"),
                output_jsonl_path=paths.get("transcripts_file", "data/processed/transcripts.jsonl"),
            )
        return cls()

    def _init_model(self):
        """Initializes Whisper model on demand to conserve GPU VRAM."""
        if self.model is not None:
            return

        # Determine device
        import torch
        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device

        # Determine compute type
        if self.compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        else:
            compute_type = self.compute_type

        # Map standard openai/ model IDs to official faster-whisper (Systran) CTranslate2 format
        model_name = self.model_id
        if model_name == "openai/whisper-large-v3-turbo":
            model_name = "Systran/faster-whisper-large-v3-turbo"
        elif model_name.startswith("openai/whisper-"):
            model_name = model_name.replace("openai/whisper-", "")

        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading Faster-Whisper model: {model_name} on {device} ({compute_type})")
            self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception as e:
            logger.warning(f"Faster-whisper failed to load ({e}). Falling back to transformers pipeline...")
            from transformers import pipeline
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            self.model = pipeline(
                "automatic-speech-recognition",
                model=self.fallback_model_id,
                device=0 if device == "cuda" else -1,
                torch_dtype=torch_dtype,
                return_timestamps=True,
            )

    def transcribe(
        self,
        audio_path: Optional[Union[str, Path]],
        video_id: str = "",
        force_transcribe: bool = False,
    ) -> Dict[str, Any]:
        """
        Transcribes audio track and returns structured timestamped result.
        Returns:
            {
                "video_id": str,
                "audio_path": Optional[str],
                "has_audio": bool,
                "detected_language": str,
                "language_probability": float,
                "transcript": str,
                "segments": [
                    {
                        "start": float,
                        "end": float,
                        "text": str
                    }, ...
                ]
            }
        """
        result: Dict[str, Any] = {
            "video_id": video_id,
            "audio_path": str(audio_path) if audio_path else None,
            "has_audio": False,
            "detected_language": "none",
            "language_probability": 0.0,
            "transcript": "",
            "segments": [],
        }

        if not audio_path or not Path(audio_path).exists() or Path(audio_path).stat().st_size <= 44:
            logger.info(f"No valid audio for video {video_id}. Returning empty transcript.")
            self._append_to_jsonl(result)
            return result

        result["has_audio"] = True
        self._init_model()

        try:
            # Check faster-whisper vs transformers pipeline
            if hasattr(self.model, "transcribe"):
                segments, info = self.model.transcribe(str(audio_path), beam_size=5)
                result["detected_language"] = info.language
                result["language_probability"] = round(info.language_probability, 3)

                seg_list = []
                full_text_parts = []
                for s in segments:
                    seg_text = s.text.strip()
                    if seg_text:
                        full_text_parts.append(seg_text)
                        seg_list.append({
                            "start": round(s.start, 2),
                            "end": round(s.end, 2),
                            "text": seg_text,
                        })
                result["transcript"] = " ".join(full_text_parts)
                result["segments"] = seg_list
            else:
                # Transformers pipeline fallback
                out = self.model(str(audio_path))
                result["transcript"] = out.get("text", "").strip()
                chunks = out.get("chunks", [])
                seg_list = []
                for c in chunks:
                    ts = c.get("timestamp", (0.0, 0.0))
                    seg_list.append({
                        "start": ts[0] if ts[0] is not None else 0.0,
                        "end": ts[1] if ts[1] is not None else 0.0,
                        "text": c.get("text", "").strip(),
                    })
                result["segments"] = seg_list
                result["detected_language"] = "en"  # fallback default
                result["language_probability"] = 1.0

        except Exception as e:
            logger.error(f"Error during transcription of {audio_path}: {str(e)}")
            result["error"] = str(e)

        self._append_to_jsonl(result)
        return result

    def _append_to_jsonl(self, entry: Dict[str, Any]):
        """Appends transcription record to transcripts.jsonl atomically."""
        with open(self.output_jsonl_path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
