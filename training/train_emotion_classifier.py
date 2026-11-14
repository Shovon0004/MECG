"""
Training script for DeBERTa / XLM-R customer emotion classifier.
Preserves the dataset's 4-class emotion taxonomy and logs Macro-F1.
"""

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.classification.emotion_classifier import DEFAULT_EMOTIONS, EmotionClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MECG.TrainEmotion")


class EmotionDataset(Dataset):
    """PyTorch Dataset for multi-class emotion classification."""

    def __init__(self, texts: List[str], label_ids: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.label_ids = label_ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.label_ids[idx], dtype=torch.long),
        }


def compute_multiclass_metrics(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """Computes Macro-F1 and per-class metrics for multi-class classification."""
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    acc = correct / (total + 1e-8)

    f1_list = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * (p * r) / (p + r + 1e-8)
        f1_list.append(f1)

    macro_f1 = sum(f1_list) / len(f1_list)
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
    }


def load_emotion_data(csv_path: str, emotion2id: Dict[str, int]) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Loads text and emotion labels partitioned by split."""
    train_texts, train_labels = [], []
    val_texts, val_labels = [], []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get("split", "train")
            text = row.get("customer_text") or row.get("raw_text", "")
            emotion = row.get("emotion", "").strip()

            if emotion not in emotion2id:
                continue

            lid = emotion2id[emotion]
            if split == "train":
                train_texts.append(text)
                train_labels.append(lid)
            elif split == "val":
                val_texts.append(text)
                val_labels.append(lid)

    return train_texts, train_labels, val_texts, val_labels


def train(
    csv_path: str = "data/processed/ComVID_cleaned.csv",
    output_dir: str = "outputs/checkpoints/emotion_classifier",
    model_id: str = "microsoft/deberta-v3-base",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
    dry_run: bool = False,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    classifier = EmotionClassifier(model_id=model_id, checkpoint_dir=out_path)
    classifier.save_mapping(out_path)

    logger.info("Loading training and validation emotion splits...")
    train_texts, train_labels, val_texts, val_labels = load_emotion_data(csv_path, classifier.emotion2id)
    logger.info(f"Loaded {len(train_texts)} train, {len(val_texts)} val emotion samples")

    if dry_run:
        train_texts, train_labels = train_texts[:16], train_labels[:16]
        val_texts, val_labels = val_texts[:16], val_labels[:16]
        epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(classifier.emotions),
    ).to(device)

    train_ds = EmotionDataset(train_texts, train_labels, tokenizer)
    val_ds = EmotionDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=1e-6, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    best_macro_f1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = criterion(logits, labels)
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits = model(input_ids=ids, attention_mask=mask).logits
                preds = torch.argmax(logits, dim=-1)
                val_preds.append(preds.cpu())
                val_targets.append(labels.cpu())

        val_preds_cat = torch.cat(val_preds, dim=0)
        val_targets_cat = torch.cat(val_targets, dim=0)
        metrics = compute_multiclass_metrics(val_preds_cat, val_targets_cat, len(classifier.emotions))

        logger.info(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Val Accuracy: {metrics['accuracy']:.4f} | Val Macro-F1: {metrics['macro_f1']:.4f}"
        )

        if metrics["macro_f1"] >= best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            model.save_pretrained(out_path)
            tokenizer.save_pretrained(out_path)

    logger.info(f"Emotion classifier training complete. Best Macro-F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="data/processed/ComVID_cleaned.csv")
    parser.add_argument("--output_dir", default="outputs/checkpoints/emotion_classifier")
    parser.add_argument("--model_id", default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    train(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        model_id=args.model_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dry_run=args.dry_run,
    )
