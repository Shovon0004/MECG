"""
Multilingual evidence-grounded complaint generation module using Qwen3-VL.
Fuses visual evidence, customer review, audio transcript, aspect tags,
and retrieved references into an evidence-grounded formal complaint.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger("MECG.ComplaintGenerator")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ComplaintGenerator:
    """Generates evidence-grounded multilingual product complaints using Qwen-VL."""

    def __init__(
        self,
        base_model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        adapter_dir: Optional[Union[str, Path]] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        device: str = "auto",
    ):
        self.base_model_id = base_model_id
        self.adapter_dir = Path(adapter_dir) if adapter_dir else None
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device

        self.model = None
        self.processor = None

    @classmethod
    def from_config(cls, config_path: Union[str, Path] = "configs/config.yaml") -> "ComplaintGenerator":
        """Instantiates ComplaintGenerator from YAML config."""
        cfg_path = Path(config_path)
        if cfg_path.exists() and yaml:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            g_cfg = cfg.get("generation", {})
            return cls(
                base_model_id=g_cfg.get("model_id", "Qwen/Qwen3-VL-4B-Instruct"),
                max_new_tokens=g_cfg.get("max_seq_length", 512),
            )
        return cls()

    def _load_model(self):
        """Loads Qwen-VL and optional PEFT LoRA adapter."""
        if self.model is not None:
            return

        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig

        try:
            from transformers import AutoModelForImageTextToText as VisionModelClass
        except ImportError:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as VisionModelClass
            except ImportError:
                try:
                    from transformers import AutoModelForVision2Seq as VisionModelClass
                except ImportError:
                    from transformers import AutoModelForCausalLM as VisionModelClass

        quant_config = None
        if torch.cuda.is_available():
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            )

        logger.info(f"Loading Qwen-VL for complaint generation: {self.base_model_id}...")
        self.processor = AutoProcessor.from_pretrained(self.base_model_id)
        self.model = VisionModelClass.from_pretrained(
            self.base_model_id,
            quantization_config=quant_config,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )

        if self.adapter_dir and self.adapter_dir.exists():
            from peft import PeftModel
            logger.info(f"Loading LoRA adapter from {self.adapter_dir}...")
            self.model = PeftModel.from_pretrained(self.model, str(self.adapter_dir))

        self.model.eval()

    def build_prompt(
        self,
        customer_text: str,
        domain: str = "General",
        aspects: str = "",
        emotion: str = "dissatisfaction",
        transcript: str = "",
        visual_evidence: Optional[List[Dict[str, Any]]] = None,
        retrieved_examples: Optional[List[Dict[str, Any]]] = None,
        requested_language: str = "en",
    ) -> str:
        """Constructs evidence-grounded structured prompt."""
        ev_lines = []
        if visual_evidence:
            for obs in visual_evidence:
                ts = obs.get("timestamp", 0.0)
                desc = obs.get("observation", "")
                conf = obs.get("confidence", 1.0)
                ev_lines.append(f"- At {ts}s: {desc} (confidence: {conf})")
        evidence_str = "\n".join(ev_lines) if ev_lines else "No specific visual defects detected or video has no visual damage."

        retrieved_lines = []
        if retrieved_examples:
            for idx, r in enumerate(retrieved_examples, 1):
                retrieved_lines.append(f"Reference Example {idx} (Domain: {r.get('domain')}): {r.get('reference_complaint')}")
        retrieved_str = "\n".join(retrieved_lines) if retrieved_lines else "None."

        lang_instruction = {
            "en": "Generate the complaint in clear, formal English.",
            "bn": "অভিযোগটি স্পষ্ট এবং প্রমিত বাংলায় রচনা করুন (Generate the complaint in standard Bengali).",
            "hi": "शिकायत को स्पष्ट और मानक हिंदी में लिखें (Generate the complaint in standard Hindi).",
        }.get(requested_language, f"Generate the complaint in the requested language ({requested_language}).")

        prompt = (
            "You are an expert AI Complaint Resolution Specialist and Consumer Rights Advocate. "
            "Your task is to draft a comprehensive, factually grounded product complaint based strictly "
            "on the provided customer testimony, audio transcript, and verified visual video evidence.\n\n"
            "### CASE EVIDENCE:\n"
            f"- Product Domain: {domain}\n"
            f"- Customer Written Review: \"{customer_text}\"\n"
            f"- Video Audio Transcript: \"{transcript or 'None'}\"\n"
            f"- Complaint Aspects: {aspects or 'General'}\n"
            f"- Customer Emotion State: {emotion}\n"
            f"- Verified Visual Evidence from Video:\n{evidence_str}\n\n"
            "### RETRIEVED CONTEXT (STYLE REFERENCE ONLY - DO NOT COPY CLAIMS):\n"
            f"{retrieved_str}\n\n"
            "### GROUNDING RULES:\n"
            "1. STRICT FACTUAL GROUNDING: Every claim you make MUST be directly traceable to the customer text, "
            "the audio transcript, or the verified visual observations.\n"
            "2. NO HALLUCINATIONS: Do NOT invent unseen defects or features not mentioned in evidence.\n"
            "3. RETRIEVED CONTEXT: Use retrieved reference examples only to understand formal complaint tone. "
            "Never import their specific product details or facts.\n"
            f"4. LANGUAGE: {lang_instruction}\n\n"
            "Draft the final grounded complaint statement now:"
        )
        return prompt

    def generate(
        self,
        customer_text: str,
        frames: Optional[List[Dict[str, Any]]] = None,
        domain: str = "General",
        aspects: str = "",
        emotion: str = "dissatisfaction",
        transcript: str = "",
        visual_evidence: Optional[List[Dict[str, Any]]] = None,
        retrieved_examples: Optional[List[Dict[str, Any]]] = None,
        requested_language: str = "en",
    ) -> str:
        """Generates evidence-grounded complaint text."""
        prompt_text = self.build_prompt(
            customer_text=customer_text,
            domain=domain,
            aspects=aspects,
            emotion=emotion,
            transcript=transcript,
            visual_evidence=visual_evidence,
            retrieved_examples=retrieved_examples,
            requested_language=requested_language,
        )

        try:
            self._load_model()
            from PIL import Image

            images = []
            if frames:
                for f in frames:
                    p = Path(f.get("file_path", ""))
                    if p.exists():
                        images.append(Image.open(p).convert("RGB"))

            # Build Qwen3-VL conversation
            content = []
            for img in images:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": prompt_text})

            messages = [{"role": "user", "content": content}]
            text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = self.processor(text=[text_input], images=images if images else None, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            import torch
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=False,
                )

            trimmed = [o[len(i):] for i, o in zip(inputs["input_ids"], out_ids)]
            generated = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
            return generated

        except Exception as e:
            logger.error(f"Generation error: {e}. Returning rule-based grounded synthesis.")
            # Fallback deterministic template
            aspect_desc = f" regarding {aspects}" if aspects else ""
            ev_desc = f" Video inspection confirms: {visual_evidence[0]['observation']}." if visual_evidence else ""
            return f"The user complains about issues with this {domain} product{aspect_desc}. Customer states: '{customer_text}'.{ev_desc}"
