"""
Training script for DeBERTa-v3 multi-label aspect classifier.
Enforces group-aware split isolation (train vs val), positive weight scaling,
and logs Micro-F1 and Macro-F1 metrics.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.classification.aspect_classifier import DEFAULT_ASPECT_TAXONOMY, AspectClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MECG.TrainAspect")


class AspectDataset(Dataset):
    """PyTorch Dataset for multi-label aspect classification."""

    def __init__(self, texts: List[str], labels: List[List[float]], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_vec = self.labels[idx]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_vec, dtype=torch.float),
        }


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """Computes binary multi-label Micro-F1 and Macro-F1."""
    bin_preds = (preds >= threshold).float()

    # Micro F1
    tp = (bin_preds * targets).sum().item()
    fp = (bin_preds * (1 - targets)).sum().item()
    fn = ((1 - bin_preds) * targets).sum().item()

    precision_micro = tp / (tp + fp + 1e-8)
    recall_micro = tp / (tp + fn + 1e-8)
    f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro + 1e-8)

    # Macro F1 (per label)
    f1_list = []
    num_labels = targets.size(1)
    for c in range(num_labels):
        tp_c = (bin_preds[:, c] * targets[:, c]).sum().item()
        fp_c = (bin_preds[:, c] * (1 - targets[:, c])).sum().item()
        fn_c = ((1 - bin_preds[:, c]) * targets[:, c]).sum().item()
        p_c = tp_c / (tp_c + fp_c + 1e-8)
        r_c = tp_c / (tp_c + fn_c + 1e-8)
        f1_c = 2 * (p_c * r_c) / (p_c + r_c + 1e-8)
        f1_list.append(f1_c)

    f1_macro = sum(f1_list) / len(f1_list)

    return {
        "f1_micro": round(f1_micro, 4),
        "f1_macro": round(f1_macro, 4),
        "precision_micro": round(precision_micro, 4),
        "recall_micro": round(recall_micro, 4),
    }


def load_data_from_csv(csv_path: str, aspect2id: Dict[str, int]) -> Tuple[List[str], List[List[float]], List[str], List[List[float]]]:
    """Loads train and val records from cleaned CSV."""
    train_texts, train_labels = [], []
    val_texts, val_labels = [], []

    num_labels = len(aspect2id)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get("split", "train")
            text = row.get("customer_text") or row.get("raw_text", "")
            aspects_raw = row.get("aspects", "")

            # Build multi-hot vector
            vec = [0.0] * num_labels
            if aspects_raw:
                items = [x.strip() for x in aspects_raw.split(",") if x.strip()]
                for it in items:
                    if it in aspect2id:
                        vec[aspect2id[it]] = 1.0

            if split == "train":
                train_texts.append(text)
                train_labels.append(vec)
            elif split == "val":
                val_texts.append(text)
                val_labels.append(vec)

    return train_texts, train_labels, val_texts, val_labels


def train(
    csv_path: str = "data/processed/ComVID_cleaned.csv",
    output_dir: str = "outputs/checkpoints/aspect_classifier",
    model_id: str = "microsoft/deberta-v3-base",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
    dry_run: bool = False,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    classifier = AspectClassifier(model_id=model_id, checkpoint_dir=out_path)
    classifier.save_mapping(out_path)

    logger.info("Loading training and validation splits...")
    train_texts, train_labels, val_texts, val_labels = load_data_from_csv(csv_path, classifier.aspect2id)
    logger.info(f"Loaded {len(train_texts)} train samples, {len(val_texts)} val samples")

    if dry_run:
        train_texts, train_labels = train_texts[:16], train_labels[:16]
        val_texts, val_labels = val_texts[:16], val_labels[:16]
        epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(classifier.taxonomy),
        problem_type="multi_label_classification",
    ).to(device)

    train_ds = AspectDataset(train_texts, train_labels, tokenizer)
    val_ds = AspectDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    # Compute positive weight scaling for BCEWithLogitsLoss
    train_t_labels = torch.tensor(train_labels, dtype=torch.float)
    pos_counts = train_t_labels.sum(dim=0)
    neg_counts = len(train_labels) - pos_counts
    pos_weight = (neg_counts / (pos_counts + 1e-5)).clamp(max=10.0).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_macro_f1 = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=ids, attention_mask=mask).logits
            loss = criterion(logits, labels)
            loss.backward()
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
                labels = batch["labels"].to(device)

                logits = model(input_ids=ids, attention_mask=mask).logits
                probs = torch.sigmoid(logits)
                val_preds.append(probs.cpu())
                val_targets.append(labels.cpu())

        val_preds_cat = torch.cat(val_preds, dim=0)
        val_targets_cat = torch.cat(val_targets, dim=0)
        metrics = compute_metrics(val_preds_cat, val_targets_cat)

        logger.info(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | "
            f"Val Micro-F1: {metrics['f1_micro']:.4f} | Val Macro-F1: {metrics['f1_macro']:.4f}"
        )

        if metrics["f1_macro"] >= best_macro_f1:
            best_macro_f1 = metrics["f1_macro"]
            model.save_pretrained(out_path)
            tokenizer.save_pretrained(out_path)

    logger.info(f"Training completed. Best Val Macro-F1: {best_macro_f1:.4f}. Model saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="data/processed/ComVID_cleaned.csv")
    parser.add_argument("--output_dir", default="outputs/checkpoints/aspect_classifier")
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
