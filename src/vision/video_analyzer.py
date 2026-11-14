"""
Video understanding and visual defect analysis using Qwen3-VL.
Analyzes sampled timestamped frames, identifies visible product defects,
and outputs structured JSON observations with timestamps and confidence scores.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger("MECG.VideoAnalyzer")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class VideoAnalyzer:
    """
    Multimodal visual evidence analyzer powered by Qwen3-VL.
    Extracts structured, timestamp-aligned observations of physical product defects.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
        quantization_4bit: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        device: str = "auto",
    ):
        self.model_id = model_id
        self.quantization_4bit = quantization_4bit
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device

        self.model = None
        self.processor = None

    @classmethod
    def from_config(cls, config_path: Union[str, Path] = "configs/config.yaml") -> "VideoAnalyzer":
        """Instantiates VideoAnalyzer from YAML configuration."""
        cfg_path = Path(config_path)
        if cfg_path.exists() and yaml:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            v_cfg = cfg.get("vision_analysis", {})
            return cls(
                model_id=v_cfg.get("model_id", "Qwen/Qwen3-VL-4B-Instruct"),
                quantization_4bit=v_cfg.get("quantization_4bit", True),
                max_new_tokens=v_cfg.get("max_new_tokens", 512),
                temperature=v_cfg.get("temperature", 0.2),
            )
        return cls()

    def _init_model(self):
        """Lazily loads Qwen3-VL with 4-bit NF4 quantization to fit within Kaggle 16GB T4 VRAM."""
        if self.model is not None:
            return

        import torch
        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as VisionModelClass
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq as VisionModelClass
            except ImportError:
                from transformers import AutoModelForCausalLM as VisionModelClass

        logger.info(f"Loading Qwen3-VL model: {self.model_id} (4-bit NF4={self.quantization_4bit})...")

        quant_config = None
        if self.quantization_4bit and torch.cuda.is_available():
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        self.processor = AutoProcessor.from_pretrained(self.model_id)

        device_map = "auto" if torch.cuda.is_available() else None
        torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

        self.model = VisionModelClass.from_pretrained(
            self.model_id,
            quantization_config=quant_config,
            torch_dtype=torch_dtype if not quant_config else None,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        logger.info("Qwen3-VL loaded successfully.")

    def release_memory(self):
        """Releases Qwen3-VL weights from GPU memory."""
        if self.model is not None:
            import gc
            import torch
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Qwen3-VL model memory released.")

    def analyze_frames(
        self,
        frames: List[Dict[str, Any]],
        product_domain: str = "",
        customer_review_hint: str = "",
    ) -> Dict[str, Any]:
        """
        Analyzes timestamped frames and returns structured JSON of visible defects.
        Expected JSON Schema:
        {
            "visible_observations": [
                {"timestamp": float, "observation": str, "confidence": float}
            ],
            "possible_aspects": [str, ...],
            "limitations": [str, ...]
        }
        """
        if not frames:
            return {
                "visible_observations": [],
                "possible_aspects": [],
                "limitations": ["No frames provided for visual analysis"],
            }

        prompt = (
            "You are a Senior Multimodal Product Quality Inspector. Analyze the provided sequential timestamped video frames "
            f"for a product in the domain '{product_domain or 'General'}'.\n"
            f"Customer Review Hint: \"{customer_review_hint}\"\n\n"
            "Instructions:\n"
            "1. Identify clear, visible physical defects, damage, leaks, or malfunctions.\n"
            "2. Associate every observation with its specific frame timestamp.\n"
            "3. State your confidence between 0.0 and 1.0.\n"
            "4. NEVER invent invisible internal defects (e.g., do not claim an odor or chemical smell is visually seen).\n"
            "5. If a defect is not visually verifiable, state that under limitations.\n\n"
            "Respond STRICTLY with valid JSON following this format:\n"
            "{\n"
            '  "visible_observations": [\n'
            '    {"timestamp": 2.4, "observation": "The bottle cap does not seal completely and water drips out.", "confidence": 0.92}\n'
            "  ],\n"
            '  "possible_aspects": ["Quality", "Usability"],\n'
            '  "limitations": ["Odor cannot be confirmed from video"]\n'
            "}"
        )

        try:
            self._init_model()
            from PIL import Image

            # Prepare image objects
            images = [Image.open(f["file_path"]).convert("RGB") for f in frames if Path(f["file_path"]).exists()]
            if not images:
                return {
                    "visible_observations": [],
                    "possible_aspects": [],
                    "limitations": ["Frame image files could not be opened"],
                }

            # Build multimodal conversation for Qwen3-VL
            content_list = []
            for idx, f in enumerate(frames):
                content_list.append({"type": "image", "image": images[idx]})
                content_list.append({"type": "text", "text": f" [Frame at {f.get('timestamp', 0.0)}s] "})
            content_list.append({"type": "text", "text": prompt})

            messages = [{"role": "user", "content": content_list}]
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = self.processor(text=[text_prompt], images=images, padding=True, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            import torch
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=False,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            return self._parse_json_response(output_text)

        except Exception as e:
            logger.error(f"Error during Qwen3-VL visual analysis: {str(e)}")
            return {
                "visible_observations": [],
                "possible_aspects": [],
                "limitations": [f"Model execution error: {str(e)}"],
            }

    @staticmethod
    def _parse_json_response(text: str) -> Dict[str, Any]:
        """Extracts and parses JSON object from model output with regex fallback."""
        # Clean markdown codeblocks
        cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()

        # Find first JSON object { ... }
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    # Ensure standard keys
                    data.setdefault("visible_observations", [])
                    data.setdefault("possible_aspects", [])
                    data.setdefault("limitations", [])
                    return data
            except json.JSONDecodeError:
                pass

        # Fallback structured response
        return {
            "visible_observations": [
                {"timestamp": 0.0, "observation": cleaned[:200], "confidence": 0.5}
            ],
            "possible_aspects": [],
            "limitations": ["Could not parse structured JSON from model output"],
        }
