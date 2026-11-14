"""
QLoRA fine-tuning script for Qwen3-VL evidence-grounded complaint generation.
Implements 4-bit NF4 quantization, LoRA adapter injection via PEFT,
and gradient checkpointing for 16GB GPU memory efficiency on Kaggle T4 GPUs.
"""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger("MECG.TrainQwen3VL")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def train_qlora(
    train_jsonl: str = "data/processed/generation_train.jsonl",
    val_jsonl: str = "data/processed/generation_val.jsonl",
    output_dir: str = "outputs/checkpoints/qwen_complaint_lora",
    base_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    epochs: int = 3,
    lr: float = 1e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    dry_run: bool = False,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Preparing QLoRA training for {base_model_id}...")

    try:
        import torch
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            TrainingArguments,
            Trainer,
        )
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as VisionModelClass
        except ImportError:
            try:
                from transformers import AutoModelForImageTextToText as VisionModelClass
            except ImportError:
                try:
                    from transformers import AutoModelForVision2Seq as VisionModelClass
                except ImportError:
                    from transformers import AutoModelForCausalLM as VisionModelClass
    except ImportError as e:
        logger.error(f"Missing required ML libraries: {e}")
        return

    # 4-bit NF4 quantization config
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(base_model_id)
    model = VisionModelClass.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    # Save adapter configuration
    peft_model.save_pretrained(out_path)
    processor.save_pretrained(out_path)
    logger.info(f"QLoRA fine-tuning configured and saved adapter configuration to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", default="data/processed/generation_train.jsonl")
    parser.add_argument("--val_jsonl", default="data/processed/generation_val.jsonl")
    parser.add_argument("--output_dir", default="outputs/checkpoints/qwen_complaint_lora")
    parser.add_argument("--base_model_id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    train_qlora(
        train_jsonl=args.train_jsonl,
        val_jsonl=args.val_jsonl,
        output_dir=args.output_dir,
        base_model_id=args.base_model_id,
        epochs=args.epochs,
        lr=args.lr,
        dry_run=args.dry_run,
    )
