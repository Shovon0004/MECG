"""
Language detection and code mapping module for MECG.
Detects language from text script and maps between ISO-639-1, Whisper tokens,
and NLLB-200 BCP-47 language codes.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger("MECG.LanguageDetector")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Default ISO to NLLB mapping
DEFAULT_ISO_TO_NLLB: Dict[str, str] = {
    "en": "eng_Latn",
    "bn": "ben_Beng",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "ur": "urd_Arab",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
}

# Reverse mapping: NLLB code to ISO code
DEFAULT_NLLB_TO_ISO: Dict[str, str] = {v: k for k, v in DEFAULT_ISO_TO_NLLB.items()}


class LanguageDetector:
    """Detects spoken/written language and maps across Whisper, ISO, and NLLB codes."""

    def __init__(self, languages_config_path: Union[str, Path] = "configs/languages.yaml"):
        self.config_path = Path(languages_config_path)
        self.lang_map = dict(DEFAULT_ISO_TO_NLLB)
        self._load_config()

    def _load_config(self):
        if self.config_path.exists() and yaml:
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f)
                langs = cfg.get("languages", {})
                for iso, data in langs.items():
                    nllb = data.get("nllb_code")
                    if nllb:
                        self.lang_map[iso] = nllb
            except Exception as e:
                logger.warning(f"Could not load languages config: {e}. Using defaults.")

    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detects primary language and script from text.
        Returns:
            {
                "iso_code": str,
                "nllb_code": str,
                "script": str,
                "confidence": float
            }
        """
        if not text or not text.strip():
            return {
                "iso_code": "unknown",
                "nllb_code": "unknown",
                "script": "unknown",
                "confidence": 0.0,
            }

        cleaned = text.strip()

        # Unicode block heuristics
        bengali_chars = len(re.findall(r'[\u0980-\u09FF]', cleaned))
        devanagari_chars = len(re.findall(r'[\u0900-\u097F]', cleaned))
        tamil_chars = len(re.findall(r'[\u0B80-\u0BFF]', cleaned))
        telugu_chars = len(re.findall(r'[\u0C00-\u0C7F]', cleaned))
        arabic_chars = len(re.findall(r'[\u0600-\u06FF]', cleaned))
        latin_chars = len(re.findall(r'[A-Za-z]', cleaned))

        char_counts = {
            "bn": (bengali_chars, "Bengali"),
            "hi": (devanagari_chars, "Devanagari"),
            "ta": (tamil_chars, "Tamil"),
            "te": (telugu_chars, "Telugu"),
            "ur": (arabic_chars, "Arabic"),
            "en": (latin_chars, "Latin"),
        }

        # Find dominant script
        best_iso, (best_count, best_script) = max(char_counts.items(), key=lambda item: item[1][0])
        total_identifiable = sum(v[0] for v in char_counts.values())

        if total_identifiable == 0 or best_count == 0:
            return {
                "iso_code": "en",
                "nllb_code": "eng_Latn",
                "script": "Latin",
                "confidence": 0.5,
            }

        confidence = round(best_count / total_identifiable, 3)
        nllb_code = self.lang_map.get(best_iso, "eng_Latn")

        return {
            "iso_code": best_iso,
            "nllb_code": nllb_code,
            "script": best_script,
            "confidence": confidence,
        }

    def iso_to_nllb(self, iso_code: str) -> str:
        """Converts ISO-639-1 code to NLLB-200 code."""
        return self.lang_map.get(iso_code.lower().strip(), "eng_Latn")

    def nllb_to_iso(self, nllb_code: str) -> str:
        """Converts NLLB-200 code to ISO-639-1 code."""
        for iso, nllb in self.lang_map.items():
            if nllb.lower() == nllb_code.lower():
                return iso
        return "en"
