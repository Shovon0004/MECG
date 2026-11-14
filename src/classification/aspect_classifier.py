"""
Aspect classification module for MECG using DeBERTa-v3.
Implements supervised multi-label classification to identify product complaint aspects.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger("MECG.AspectClassifier")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Default canonical aspect taxonomy derived from Com-VID dataset
DEFAULT_ASPECT_TAXONOMY: List[str] = [
    "Quality",
    "Usability",
    "Damaged Quality",
    "Originality",
    "Functionality",
    "Manufacturing",
    "Size",
    "Defective",
    "Poor Quality",
    "Correctness",
    "Durability",
    "Design",
    "Colour",
    "Performance",
    "Comfortness",
    "Representation",
    "Completeness",
    "Fraud",
    "Finishing",
    "Cleanliness",
    "Packaging",
    "Condition",
    "Cost-effectiveness",
]


class AspectClassifier:
    """Supervised multi-label classifier for complaint aspects."""

    def __init__(
        self,
        model_id: str = "microsoft/deberta-v3-base",
        taxonomy: Optional[List[str]] = None,
        checkpoint_dir: Union[str, Path] = "outputs/checkpoints/aspect_classifier",
        device: str = "auto",
        threshold: float = 0.5,
    ):
        self.model_id = model_id
        self.taxonomy = sorted(list(set(taxonomy or DEFAULT_ASPECT_TAXONOMY)))
        self.aspect2id = {asp: idx for idx, asp in enumerate(self.taxonomy)}
        self.id2aspect = {idx: asp for idx, asp in enumerate(self.taxonomy)}
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self.threshold = threshold

        self.tokenizer = None
        self.model = None

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: Union[str, Path]) -> "AspectClassifier":
        """Loads fine-tuned classifier and taxonomy from checkpoint."""
        ckpt = Path(checkpoint_dir)
        map_file = ckpt / "aspect_mapping.json"
        if not map_file.exists():
            raise FileNotFoundError(f"Aspect mapping not found at: {map_file}")

        with open(map_file, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        taxonomy = mapping.get("taxonomy", DEFAULT_ASPECT_TAXONOMY)
        instance = cls(checkpoint_dir=ckpt, taxonomy=taxonomy)
        instance._load_model()
        return instance

    def _load_model(self):
        """Loads DeBERTa model and tokenizer from checkpoint or pretrained base."""
        if self.model is not None:
            return

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device

        ckpt_path = self.checkpoint_dir if self.checkpoint_dir.exists() and (self.checkpoint_dir / "config.json").exists() else self.model_id
        logger.info(f"Loading Aspect Classifier from {ckpt_path} on {device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            ckpt_path,
            num_labels=len(self.taxonomy),
            problem_type="multi_label_classification",
        )
        self.model.to(device)
        self.model.eval()
        self._device = device

    def encode_labels(self, aspects_list: List[str]) -> List[float]:
        """Converts list of aspect strings into multi-hot binary vector."""
        vec = [0.0] * len(self.taxonomy)
        for asp in aspects_list:
            clean_asp = asp.strip()
            if clean_asp in self.aspect2id:
                vec[self.aspect2id[clean_asp]] = 1.0
        return vec

    def decode_predictions(self, probabilities: List[float], threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Converts sigmoid probabilities to list of detected aspects with confidences."""
        t = threshold if threshold is not None else self.threshold
        detected = []
        for idx, prob in enumerate(probabilities):
            if prob >= t:
                detected.append({
                    "aspect": self.id2aspect[idx],
                    "confidence": round(float(prob), 3),
                })
        # If none exceed threshold, return top-1 aspect
        if not detected and probabilities:
            best_idx = int(probabilities.index(max(probabilities)))
            detected.append({
                "aspect": self.id2aspect[best_idx],
                "confidence": round(float(probabilities[best_idx]), 3),
            })
        return detected

    def predict(self, text: str, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Predicts complaint aspects for input review or transcript."""
        try:
            self._load_model()
            import torch

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

            return self.decode_predictions(probs, threshold=threshold)
        except (ImportError, Exception) as e:
            logger.warning(f"Using rule-based aspect extraction fallback: {e}")
            detected = []
            text_lower = text.lower()
            for asp in self.taxonomy:
                if asp.lower() in text_lower:
                    detected.append({"aspect": asp, "confidence": 0.85})
            if not detected:
                detected.append({"aspect": "Quality", "confidence": 0.5})
            return detected

    def save_mapping(self, output_dir: Optional[Union[str, Path]] = None) -> Path:
        """Saves aspect taxonomy mapping to JSON."""
        out_dir = Path(output_dir) if output_dir else self.checkpoint_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / "aspect_mapping.json"
        data = {
            "num_labels": len(self.taxonomy),
            "taxonomy": self.taxonomy,
            "aspect2id": self.aspect2id,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return meta_path
