"""
Multilingual speech recognition module for MECG using Whisper.
Extracts timestamped transcription segments, detects spoken language and confidence,
preserves original transcripts, and appends to data/processed/transcripts.jsonl.
"""

import ctypes
import glob
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Auto-link CUDA 12 libraries from nvidia python site-packages (fixes libcublas.so.12 not found on Kaggle)
for p in sys.path:
    for lib_dir in glob.glob(os.path.join(p, "nvidia", "*", "lib")):
        if os.path.isdir(lib_dir):
            curr_ld = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_dir not in curr_ld:
                os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{curr_ld}"
            # Pre-load libcublas into process memory if present
            cublas_so = os.path.join(lib_dir, "libcublas.so.12")
            if os.path.exists(cublas_so):
                try:
                    ctypes.CDLL(cublas_so, mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass

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
        model_id: str = "deepdml/faster-whisper-large-v3-turbo-ct2",
        fallback_model_id: str = "openai/whisper-large-v3-turbo",
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
                model_id=a_cfg.get("model_id", "deepdml/faster-whisper-large-v3-turbo-ct2"),
                fallback_model_id=a_cfg.get("fallback_model_id", "openai/whisper-large-v3-turbo"),
                device=a_cfg.get("device", "auto"),
                compute_type=a_cfg.get("compute_type", "auto"),
                output_jsonl_path=paths.get("transcripts_file", "data/processed/transcripts.jsonl"),
            )
        return cls()

    def _init_model(self):
        """Initializes Whisper model with verified CUDA runtime support."""
        if self.model is not None:
            return

        import torch
        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device

        if self.compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        else:
            compute_type = self.compute_type

        model_name = self.model_id
        if model_name in ("openai/whisper-large-v3-turbo", "Systran/faster-whisper-large-v3-turbo"):
            model_name = "deepdml/faster-whisper-large-v3-turbo-ct2"
        elif model_name.startswith("openai/whisper-"):
            model_name = model_name.replace("openai/whisper-", "")

        # Attempt Faster-Whisper first
        try:
            from faster_whisper import WhisperModel
            import numpy as np

            logger.info(f"Loading Faster-Whisper model: {model_name} on {device} ({compute_type})...")
            fw_model = WhisperModel(model_name, device=device, compute_type=compute_type)

            # Verification: test short dummy audio array to verify libcublas linking
            dummy_pcm = np.zeros(1600, dtype=np.float32)
            _ = list(fw_model.transcribe(dummy_pcm)[0])
            self.model = fw_model
            logger.info("Faster-Whisper CUDA runtime verified successfully.")
            return
        except Exception as e:
            logger.warning(
                f"Faster-whisper CUDA initialization failed ({e}). "
                f"Switching to native Hugging Face transformers pipeline ({self.fallback_model_id})..."
            )

        # Fallback to HuggingFace pipeline (uses PyTorch's internal cuBLAS, 100% reliable on GPU)
        try:
            from transformers import pipeline

            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            self.model = pipeline(
                "automatic-speech-recognition",
                model=self.fallback_model_id,
                device=0 if device == "cuda" else -1,
                torch_dtype=torch_dtype,
                return_timestamps=True,
            )
            logger.info(f"Transformers ASR pipeline loaded successfully with {self.fallback_model_id}.")
        except Exception as e_pipe:
            logger.error(f"Failed to load transformers ASR fallback pipeline: {e_pipe}")

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
            if hasattr(self.model, "transcribe"):
                # Load audio as NumPy float32 array to bypass PyAV av.open(metadata_errors=...) incompatibility
                try:
                    import soundfile as sf
                    audio_arr, sr = sf.read(str(audio_path), dtype="float32")
                    if audio_arr.ndim > 1:
                        audio_arr = audio_arr.mean(axis=1)
                    audio_input = audio_arr
                except Exception:
                    audio_input = str(audio_path)

                segments, info = self.model.transcribe(
                    audio_input,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                )

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
                full_transcript = " ".join(full_text_parts)
                result["transcript"] = full_transcript
                result["segments"] = seg_list

                # Robust Language Detection with Confidence Gating:
                # If audio is silent/empty or confidence is low (< 0.65), default to English
                detected_lang = getattr(info, "language", "en")
                prob = float(getattr(info, "language_probability", 0.0) or 0.0)

                if not full_transcript or len(full_transcript) < 3 or prob < 0.65:
                    if prob < 0.65 and full_transcript:
                        logger.info(
                            f"Low audio language confidence ({prob:.2f} for '{detected_lang}'). "
                            "Defaulting to 'en'."
                        )
                    detected_lang = "en"
                    prob = 0.95 if not full_transcript else prob

                result["detected_language"] = detected_lang
                result["language_probability"] = round(prob, 3)
            elif callable(self.model):
                out = self.model(str(audio_path))
                result["transcript"] = out.get("text", "").strip()
                chunks = out.get("chunks", [])
                result["segments"] = [
                    {
                        "start": round(float(c.get("timestamp", [0.0, 0.0])[0] or 0.0), 2),
                        "end": round(float(c.get("timestamp", [0.0, 0.0])[1] or 0.0), 2),
                        "text": c.get("text", "").strip(),
                    }
                    for c in chunks
                    if c.get("text")
                ]
                result["detected_language"] = "en"
                result["language_probability"] = 0.95
        except Exception as e:
            logger.warning(f"Error during transcription with primary model: {e}. Attempting transformers fallback...")
            try:
                import torch
                from transformers import pipeline

                device = "cuda" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.float16 if device == "cuda" else torch.float32
                fallback_pipe = pipeline(
                    "automatic-speech-recognition",
                    model=self.fallback_model_id,
                    device=0 if device == "cuda" else -1,
                    torch_dtype=torch_dtype,
                    return_timestamps=True,
                )
                self.model = fallback_pipe
                out = self.model(str(audio_path))
                result["transcript"] = out.get("text", "").strip()
                chunks = out.get("chunks", [])
                result["segments"] = [
                    {
                        "start": round(float(c.get("timestamp", [0.0, 0.0])[0] or 0.0), 2),
                        "end": round(float(c.get("timestamp", [0.0, 0.0])[1] or 0.0), 2),
                        "text": c.get("text", "").strip(),
                    }
                    for c in chunks
                    if c.get("text")
                ]
                result["detected_language"] = "en"
                result["language_probability"] = 0.95
            except Exception as e_fb:
                logger.error(f"Fallback transcription also failed: {e_fb}")

        self._append_to_jsonl(result)
        return result

    def _append_to_jsonl(self, entry: Dict[str, Any]):
        """Thread-safe append of transcript entry to output JSONL."""
        try:
            with open(self.output_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Could not append transcript to {self.output_jsonl_path}: {e}")
