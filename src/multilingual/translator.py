"""
Multilingual translation module for MECG using NLLB-200.
Supports batch translation, disk-backed caching, on-demand GPU loading,
and no-op passthrough when source and target languages match.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None

from src.multilingual.language_detector import LanguageDetector

logger = logging.getLogger("MECG.Translator")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class NLLBTranslator:
    """
    Translates complaint texts and transcripts using facebook/nllb-200-distilled-600M.
    Features persistent disk caching and lazy model loading.
    """

    def __init__(
        self,
        model_id: str = "facebook/nllb-200-distilled-600M",
        default_target_lang: str = "eng_Latn",
        cache_path: Union[str, Path] = "data/processed/translation_cache.json",
        device: str = "auto",
        batch_size: int = 16,
    ):
        self.model_id = model_id
        self.default_target_lang = default_target_lang
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.batch_size = batch_size

        self.detector = LanguageDetector()
        self.tokenizer = None
        self.model = None
        self.cache: Dict[str, str] = {}
        self._load_cache()

    @classmethod
    def from_config(cls, config_path: Union[str, Path] = "configs/config.yaml") -> "NLLBTranslator":
        """Instantiates NLLBTranslator from project YAML config."""
        cfg_path = Path(config_path)
        if cfg_path.exists() and yaml:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            t_cfg = cfg.get("multilingual_translation", {})
            return cls(
                model_id=t_cfg.get("model_id", "facebook/nllb-200-distilled-600M"),
                default_target_lang=t_cfg.get("default_target_lang", "eng_Latn"),
                batch_size=t_cfg.get("batch_size", 16),
            )
        return cls()

    def _load_cache(self):
        """Loads cached translations from disk."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load translation cache: {e}. Starting with empty cache.")
                self.cache = {}

    def _save_cache(self):
        """Persists translation cache to disk."""
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist translation cache: {e}")

    def _make_cache_key(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Computes deterministic cache key."""
        raw = f"{src_lang}::{tgt_lang}::{text.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _init_model(self):
        """Lazily loads NLLB tokenizer and model on demand."""
        if self.model is not None and self.tokenizer is not None:
            return

        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device

        logger.info(f"Loading NLLB-200 model: {self.model_id} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        self.model.to(device)
        self.model.eval()
        self._device = device
        logger.info("NLLB-200 model successfully loaded.")

    def release_memory(self):
        """Unloads model from memory to free VRAM for subsequent pipeline stages."""
        if self.model is not None:
            import gc
            import torch
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("NLLB-200 model unloaded from memory.")

    def translate(
        self,
        text: str,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Translates single text. Detects source language if not provided.
        Returns:
            {
                "original_text": str,
                "translated_text": str,
                "source_lang": str,
                "target_lang": str,
                "cached": bool
            }
        """
        if not text or not text.strip():
            return {
                "original_text": text,
                "translated_text": "",
                "source_lang": src_lang or "unknown",
                "target_lang": tgt_lang or self.default_target_lang,
                "cached": False,
            }

        cleaned = text.strip()

        # Determine source language
        if not src_lang:
            det = self.detector.detect_language(cleaned)
            src_lang = det["nllb_code"]
        elif "_" not in src_lang:
            src_lang = self.detector.iso_to_nllb(src_lang)

        # Determine target language
        target = tgt_lang or self.default_target_lang
        if "_" not in target:
            target = self.detector.iso_to_nllb(target)

        # Optimization: No translation needed if source matches target
        if src_lang == target:
            return {
                "original_text": cleaned,
                "translated_text": cleaned,
                "source_lang": src_lang,
                "target_lang": target,
                "cached": False,
            }

        # Check Cache
        c_key = self._make_cache_key(cleaned, src_lang, target)
        if c_key in self.cache:
            return {
                "original_text": cleaned,
                "translated_text": self.cache[c_key],
                "source_lang": src_lang,
                "target_lang": target,
                "cached": True,
            }

        # Model Inference
        self._init_model()
        import torch

        self.tokenizer.src_lang = src_lang
        inputs = self.tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        target_lang_id = self.tokenizer.convert_tokens_to_ids(target)
        with torch.no_grad():
            gen_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=target_lang_id,
                max_length=512,
                num_beams=4,
            )

        translated = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0].strip()

        # Update Cache
        self.cache[c_key] = translated
        self._save_cache()

        return {
            "original_text": cleaned,
            "translated_text": translated,
            "source_lang": src_lang,
            "target_lang": target,
            "cached": False,
        }

    def translate_batch(
        self,
        texts: List[str],
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Batch translation with automatic caching and chunking."""
        results = []
        for text in texts:
            results.append(self.translate(text, src_lang=src_lang, tgt_lang=tgt_lang))
        return results
