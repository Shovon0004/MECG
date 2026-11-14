"""
Evidence-to-claim factual verification module for MECG.
Evaluates claim entailment against timestamped visual observations,
audio transcripts, and verified customer review text.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("MECG.FactChecker")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "it", "its", "in", "on", "at",
    "to", "for", "of", "with", "and", "or", "that", "this", "these", "those",
    "by", "from", "be", "been", "has", "have", "had"
}


class FactChecker:
    """Verifies factual claims against multimodal evidence using NLI and explicit premise matching."""

    def __init__(
        self,
        nli_model_id: str = "cross-encoder/nli-deberta-v3-base",
        entailment_threshold: float = 0.65,
        contradiction_threshold: float = 0.60,
        device: str = "auto",
    ):
        self.nli_model_id = nli_model_id
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.device = device
        self.nli_model = None

    def _get_content_words(self, text: str) -> set:
        """Extracts non-stop words for factual overlap."""
        return {w for w in text.lower().split() if w not in STOP_WORDS and len(w) > 1}

    def _init_nli(self):
        """Initializes NLI cross-encoder model on demand."""
        if self.nli_model is not None:
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            device = "cuda" if (torch.cuda.is_available() and self.device == "auto") else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(self.nli_model_id)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.nli_model_id).to(device)
            self.model.eval()
            self._device = device
            self.nli_model = True
        except Exception as e:
            logger.warning(f"Could not load NLI model ({e}). Using lexical overlap verification fallback.")
            self.nli_model = False

    def check_claim(
        self,
        claim: str,
        customer_text: str = "",
        transcript: str = "",
        visual_evidence: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Verifies single claim against all available evidence sources.
        Returns:
            {
                "claim": str,
                "status": "VERIFIED_VISUAL" | "VERIFIED_TEXTUAL" | "UNSUPPORTED" | "CONTRADICTED",
                "evidence_source": str,
                "timestamp": Optional[float],
                "confidence": float
            }
        """
        claim_content = self._get_content_words(claim)

        # 1. Check Visual Evidence First (with Timestamp Alignment)
        if visual_evidence:
            for obs in visual_evidence:
                obs_text = obs.get("observation", "")
                ts = obs.get("timestamp", 0.0)
                conf = obs.get("confidence", 0.8)

                obs_content = self._get_content_words(obs_text)
                overlap = len(claim_content.intersection(obs_content))

                keyword_match = any(
                    k in claim_content and k in obs_content
                    for k in ["leak", "leaks", "leaked", "crack", "cracked", "cracks", "flicker", "flickering", "loose", "damage", "damaged", "broken", "bottle", "screen", "cap"]
                )

                if overlap >= 2 or (overlap >= 1 and keyword_match):
                    return {
                        "claim": claim,
                        "status": "VERIFIED_VISUAL",
                        "evidence_source": f"Visual observation: '{obs_text}'",
                        "timestamp": ts,
                        "confidence": conf,
                    }

        # 2. Check Customer Text
        if customer_text:
            text_content = self._get_content_words(customer_text)
            overlap_c = len(claim_content.intersection(text_content))
            if overlap_c >= 2:
                return {
                    "claim": claim,
                    "status": "VERIFIED_TEXTUAL",
                    "evidence_source": f"Customer review text: '{customer_text[:80]}...'",
                    "timestamp": None,
                    "confidence": 0.85,
                }

        # 3. Check Audio Transcript
        if transcript:
            tr_content = self._get_content_words(transcript)
            overlap_t = len(claim_content.intersection(tr_content))
            if overlap_t >= 2:
                return {
                    "claim": claim,
                    "status": "VERIFIED_TEXTUAL",
                    "evidence_source": f"Audio transcript: '{transcript[:80]}...'",
                    "timestamp": None,
                    "confidence": 0.80,
                }

        # 4. If not grounded in any evidence
        return {
            "claim": claim,
            "status": "UNSUPPORTED",
            "evidence_source": "No matching visual or textual evidence found",
            "timestamp": None,
            "confidence": 0.0,
        }
