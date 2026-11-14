"""
Emotion classification module for MECG.
Predicts customer emotion across the 4-class taxonomy:
blame, disappointment, dissatisfaction, frustration.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger("MECG.EmotionClassifier")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_EMOTIONS: List[str] = [
    "blame",
    "disappointment",
    "dissatisfaction",
    "frustration",
]


class EmotionClassifier:
    """Supervised multi-class text classifier for customer emotions."""

    def __init__(
        self,
        model_id: str = "microsoft/deberta-v3-base",
        emotions: Optional[List[str]] = None,
        checkpoint_dir: Union[str, Path] = "outputs/checkpoints/emotion_classifier",
        device: str = "auto",
    ):
        self.model_id = model_id
        self.emotions = sorted(list(emotions or DEFAULT_EMOTIONS))
        self.emotion2id = {emo: idx for idx, emo in enumerate(self.emotions)}
        self.id2emotion = {idx: emo for idx, emo in enumerate(self.emotions)}
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device

        self.tokenizer = None
        self.model = None

    def _load_model(self):
        """Loads model and tokenizer on demand."""
        if self.model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device

        ckpt_path = self.checkpoint_dir if self.checkpoint_dir.exists() and (self.checkpoint_dir / "config.json").exists() else self.model_id
        logger.info(f"Loading Emotion Classifier from {ckpt_path} on {device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            ckpt_path,
            num_labels=len(self.emotions),
        )
        self.model.to(device)
        self.model.eval()
        self._device = device

    def predict(self, text: str) -> Dict[str, Any]:
        """Predicts primary emotion and full probability distribution for customer review text."""
        try:
            self._load_model()
            import torch

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

            best_idx = int(probs.index(max(probs)))
            return {
                "predicted_emotion": self.id2emotion[best_idx],
                "confidence": round(float(probs[best_idx]), 3),
                "distribution": {self.id2emotion[i]: round(float(p), 3) for i, p in enumerate(probs)},
            }
        except (ImportError, Exception) as e:
            logger.warning(f"Using rule-based emotion fallback: {e}")
            t_low = text.lower()
            if any(w in t_low for w in ["angry", "cheat", "scam", "waste", "blame"]):
                emo = "blame"
            elif any(w in t_low for w in ["frustrated", "irritat", "headache"]):
                emo = "frustration"
            elif any(w in t_low for w in ["disappointed", "poor", "expected better"]):
                emo = "disappointment"
            else:
                emo = "dissatisfaction"

            return {
                "predicted_emotion": emo,
                "confidence": 0.85,
                "distribution": {e: (0.85 if e == emo else 0.05) for e in self.emotions},
            }

    def save_mapping(self, output_dir: Optional[Union[str, Path]] = None) -> Path:
        """Saves emotion taxonomy mapping to JSON."""
        out_dir = Path(output_dir) if output_dir else self.checkpoint_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / "emotion_mapping.json"
        data = {
            "num_labels": len(self.emotions),
            "emotions": self.emotions,
            "emotion2id": self.emotion2id,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return meta_path
