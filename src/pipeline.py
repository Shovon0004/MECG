"""
MECG End-to-End Pipeline Orchestrator.
Executes the full 13-stage multimodal, multilingual, evidence-grounded
complaint generation and verification pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None

from src.audio.audio_extractor import AudioExtractor
from src.audio.transcriber import WhisperTranscriber
from src.classification.aspect_classifier import AspectClassifier
from src.classification.emotion_classifier import EmotionClassifier
from src.generation.complaint_generator import ComplaintGenerator
from src.multilingual.language_detector import LanguageDetector
from src.multilingual.translator import NLLBTranslator
from src.retrieval.visual_retriever import VisualRetriever
from src.verification.grounding_pipeline import GroundingPipeline
from src.vision.frame_extractor import FrameExtractor
from src.vision.video_analyzer import VideoAnalyzer

logger = logging.getLogger("MECG.Pipeline")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class MECGPipeline:
    """End-to-end multimodal, multilingual, evidence-grounded complaint generation pipeline."""

    def __init__(
        self,
        config_path: Union[str, Path] = "configs/config.yaml",
        frame_extractor: Optional[FrameExtractor] = None,
        audio_extractor: Optional[AudioExtractor] = None,
        transcriber: Optional[WhisperTranscriber] = None,
        detector: Optional[LanguageDetector] = None,
        translator: Optional[NLLBTranslator] = None,
        video_analyzer: Optional[VideoAnalyzer] = None,
        aspect_classifier: Optional[AspectClassifier] = None,
        emotion_classifier: Optional[EmotionClassifier] = None,
        visual_retriever: Optional[VisualRetriever] = None,
        generator: Optional[ComplaintGenerator] = None,
        grounding_pipeline: Optional[GroundingPipeline] = None,
    ):
        self.config_path = Path(config_path)

        # Lazy component binding
        self.frame_extractor = frame_extractor or FrameExtractor()
        self.audio_extractor = audio_extractor or AudioExtractor()
        self.transcriber = transcriber or WhisperTranscriber()
        self.detector = detector or LanguageDetector()
        self.translator = translator or NLLBTranslator()
        self.video_analyzer = video_analyzer or VideoAnalyzer()
        self.aspect_classifier = aspect_classifier or AspectClassifier()
        self.emotion_classifier = emotion_classifier or EmotionClassifier()
        self.visual_retriever = visual_retriever or VisualRetriever()
        self.generator = generator or ComplaintGenerator()
        self.grounding_pipeline = grounding_pipeline or GroundingPipeline()

    def process(
        self,
        video_path: Union[str, Path],
        customer_text: str,
        video_id: str = "VID_UNKNOWN",
        product_domain: str = "General",
        requested_output_language: str = "en",
    ) -> Dict[str, Any]:
        """
        Executes the 13 pipeline steps:
        1. Load video and customer text
        2. Extract timestamped frames
        3. Extract audio
        4. Transcribe audio via Whisper
        5. Detect original language
        6. Translate text to English when required
        7. Analyze visual evidence with Qwen3-VL
        8. Predict complaint aspects
        9. Predict emotion
        10. Retrieve visually similar reference complaints
        11. Generate evidence-grounded complaint
        12. Verify claims against evidence
        13. Return structured JSON report
        """
        v_path = Path(video_path)
        logger.info(f"Starting MECG pipeline for {video_id} ({product_domain})...")

        # Step 2: Extract timestamped frames
        frames = []
        if v_path.exists():
            f_res = self.frame_extractor.extract_frames(v_path, video_id=video_id)
            frames = f_res.get("frames", [])

        # Step 3 & 4: Extract audio and transcribe
        transcript_orig = ""
        detected_lang = "en"
        if v_path.exists():
            a_res = self.audio_extractor.extract_audio(v_path, video_id=video_id)
            if a_res.get("has_audio") and a_res.get("audio_path"):
                t_res = self.transcriber.transcribe(a_res["audio_path"], video_id=video_id)
                transcript_orig = t_res.get("transcript", "")
                detected_lang = t_res.get("detected_language", "en")

        # Step 5: Detect language from customer text if no speech
        if detected_lang in ("en", "none", "unknown") and customer_text:
            text_det = self.detector.detect_language(customer_text)
            if text_det["iso_code"] != "en":
                detected_lang = text_det["iso_code"]

        # Step 6: Translate text when required
        transcript_en = transcript_orig
        customer_text_en = customer_text
        if detected_lang not in ("en", "none", "unknown"):
            if transcript_orig:
                transcript_en = self.translator.translate(transcript_orig, src_lang=detected_lang, tgt_lang="en")["translated_text"]
            if customer_text:
                customer_text_en = self.translator.translate(customer_text, src_lang=detected_lang, tgt_lang="en")["translated_text"]

        # Step 7: Analyze visual evidence using Qwen3-VL
        visual_analysis = self.video_analyzer.analyze_frames(
            frames=frames,
            product_domain=product_domain,
            customer_review_hint=customer_text_en,
        )
        visual_evidence = visual_analysis.get("visible_observations", [])

        # Step 8: Predict complaint aspects (Multimodal Fusion: Text + Audio + Video)
        combined_text = f"{customer_text_en} {transcript_en}".strip()
        aspect_preds = self.aspect_classifier.predict(combined_text) if combined_text else []
        text_aspects = [p["aspect"] for p in aspect_preds]
        visual_aspects = visual_analysis.get("possible_aspects", [])

        # Fuse unique aspects preserving order (text + visual evidence)
        seen_aspects = set()
        aspect_names = []
        for asp in text_aspects + visual_aspects:
            clean_asp = asp.strip()
            if clean_asp and clean_asp.lower() not in seen_aspects:
                seen_aspects.add(clean_asp.lower())
                aspect_names.append(clean_asp)

        # Step 9: Predict customer emotion
        emotion_preds = self.emotion_classifier.predict(customer_text_en) if customer_text_en else {"predicted_emotion": "dissatisfaction"}
        emotion_name = emotion_preds.get("predicted_emotion", "dissatisfaction")

        # Step 10: Retrieve visually similar examples from FAISS
        frame_paths = [f["file_path"] for f in frames]
        retrieved_examples = self.visual_retriever.search(
            query_frame_paths=frame_paths,
            top_k=3,
            exclude_video_hash=v_path.stem,
        )

        # Step 11: Generate evidence-grounded complaint using Qwen3-VL
        generated_complaint = self.generator.generate(
            customer_text=customer_text,
            frames=frames,
            domain=product_domain,
            aspects=", ".join(aspect_names),
            emotion=emotion_name,
            transcript=transcript_orig,
            visual_evidence=visual_evidence,
            retrieved_examples=retrieved_examples,
            requested_language=requested_output_language,
        )

        # Step 12: Check generated claims against evidence
        verification = self.grounding_pipeline.verify_complaint(
            generated_complaint=generated_complaint,
            customer_text=customer_text,
            transcript=transcript_orig,
            visual_evidence=visual_evidence,
        )

        # Step 13: Format and return exact output JSON structure
        result: Dict[str, Any] = {
            "video_id": video_id,
            "detected_language": detected_lang,
            "requested_output_language": requested_output_language,
            "transcript_original": transcript_orig,
            "transcript_english": transcript_en,
            "visual_evidence": visual_evidence,
            "aspects": aspect_names,
            "emotion": emotion_name,
            "retrieved_examples": retrieved_examples,
            "generated_complaint": generated_complaint,
            "verified_claims": verification.get("verified_claims", []),
            "unsupported_claims": verification.get("unsupported_claims", []),
            "evidence_grounding_score": verification.get("evidence_grounding_score", None),
        }

        logger.info(f"Pipeline completed for {video_id}. Grounding Score: {result['evidence_grounding_score']}")
        return result
