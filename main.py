"""
MECG: Multilingual Evidence-Grounded Complaint Generation
Master Command-Line Interface and Pipeline Orchestrator.
"""

import argparse
import logging
import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from src.preprocessing.dataset_cleaner import ComVIDCleaner
from src.preprocessing.audit import DatasetAuditor
from src.vision.video_downloader import VideoDownloader
from src.vision.video_validator import VideoValidator
from src.vision.frame_extractor import FrameExtractor
from src.audio.audio_extractor import AudioExtractor
from src.audio.transcriber import WhisperTranscriber
from src.multilingual.language_detector import LanguageDetector
from src.multilingual.translator import NLLBTranslator
from src.classification.aspect_classifier import AspectClassifier
from src.classification.emotion_classifier import EmotionClassifier
from src.retrieval.visual_retriever import VisualRetriever
from src.generation.complaint_generator import ComplaintGenerator
from src.verification.grounding_pipeline import GroundingPipeline
from src.pipeline import MECGPipeline
from evaluation.ablation_study import AblationStudy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("MECG.CLI")


def run_phase_1(config_path: str = "configs/config.yaml"):
    """Phase 1: Dataset Exploration, Cleaning, and Audit."""
    logger.info("Executing Phase 1: Dataset Cleaning & Quality Audit...")
    cleaner = ComVIDCleaner.from_config(config_path)
    cleaned_records = cleaner.process()
    cleaned_csv_path = cleaner.save_cleaned_csv()

    auditor = DatasetAuditor.from_csv(cleaned_csv_path)
    report_path = auditor.save_report("reports/dataset_audit.json")
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE: DATASET HYGIENE & AUDIT REPORT GENERATED")
    print(f"Cleaned CSV: {cleaned_csv_path} ({len(cleaned_records)} records)")
    print(f"Audit JSON:  {report_path}")
    print("=" * 60 + "\n")


def run_phase_2(config_path: str = "configs/config.yaml", pilot: bool = False, limit: int = None):
    """Phase 2: Video Downloading, Validation & Deduplication."""
    logger.info(f"Executing Phase 2: Video Downloading (pilot={pilot})...")
    downloader = VideoDownloader.from_config(config_path)
    downloader.pilot_mode = pilot
    effective_limit = 5 if pilot else limit
    report = downloader.download_dataset(limit=effective_limit)
    report_path = downloader.save_report()
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE: VIDEO DOWNLOADING & VALIDATION")
    for k, v in report.get("summary", {}).items():
        print(f"  {k:20s}: {v}")
    print(f"Report: {report_path}")
    print("=" * 60 + "\n")


def run_phase_3(config_path: str = "configs/config.yaml", limit: int = None):
    """Phase 3: Frame Extraction, Audio Extraction & Whisper Transcription."""
    limit_str = f"limit={limit}" if limit else "all available"
    logger.info(f"Executing Phase 3: Multimodal Preprocessing ({limit_str})...")
    frame_extractor = FrameExtractor.from_config(config_path)
    audio_extractor = AudioExtractor.from_config(config_path)
    transcriber = WhisperTranscriber.from_config(config_path)

    all_videos = sorted(list(Path("data/videos").glob("*.mp4")))
    video_files = all_videos[:limit] if limit else all_videos
    logger.info(f"Processing {len(video_files)} video files...")

    for v in video_files:
        f_res = frame_extractor.extract_frames(v)
        a_res = audio_extractor.extract_audio(v)
        if a_res.get("has_audio") and a_res.get("audio_path"):
            transcriber.transcribe(a_res["audio_path"], video_id=v.stem)

    print("\n" + "=" * 60)
    print(f"PHASE 3 COMPLETE: FRAMES, AUDIO & TRANSCRIPTS EXTRACTED ({len(video_files)} videos)")
    print("=" * 60 + "\n")


def run_phase_4(config_path: str = "configs/config.yaml"):
    """Phase 4: Multilingual Language Detection & NLLB-200 Translation."""
    logger.info("Executing Phase 4: Multilingual Translation & Normalization...")
    detector = LanguageDetector()
    translator = NLLBTranslator.from_config(config_path)

    # Sanity checks on benchmark pairs
    bn_phrase = "প্লাস্টিকের বোতল থেকে তীব্র রাসায়নিক গন্ধ বের হচ্ছে।"
    det = detector.detect_language(bn_phrase)
    trans = translator.translate(bn_phrase, src_lang="bn", tgt_lang="en")

    print("\n" + "=" * 60)
    print("PHASE 4 COMPLETE: MULTILINGUAL PROCESSING VERIFIED")
    print(f"Input Bengali:   {bn_phrase}")
    print(f"Detected Code:   {det['iso_code']} ({det['nllb_code']})")
    print(f"English Output:  {trans['translated_text']}")
    print("=" * 60 + "\n")


def run_phase_5(config_path: str = "configs/config.yaml"):
    """Phase 5: Aspect and Emotion Classifiers."""
    logger.info("Executing Phase 5: Aspect & Emotion Classification Setup...")
    aspect_clf = AspectClassifier(checkpoint_dir="outputs/checkpoints/aspect_classifier")
    emotion_clf = EmotionClassifier(checkpoint_dir="outputs/checkpoints/emotion_classifier")
    aspect_clf.save_mapping()
    emotion_clf.save_mapping()

    sample_text = "The bottle cap is loose and water leaks out everywhere. Very disappointed."
    asp_preds = aspect_clf.predict(sample_text)
    emo_preds = emotion_clf.predict(sample_text)

    print("\n" + "=" * 60)
    print("PHASE 5 COMPLETE: ASPECT & EMOTION CLASSIFIERS")
    print(f"Sample Text:       {sample_text}")
    print(f"Predicted Aspects: {[p['aspect'] for p in asp_preds]}")
    print(f"Predicted Emotion: {emo_preds['predicted_emotion']} (conf={emo_preds['confidence']})")
    print("=" * 60 + "\n")


def run_phase_7(config_path: str = "configs/config.yaml", pilot: bool = False, limit: int = None):
    """Phase 7: Visual Retrieval Indexing (SigLIP 2 & FAISS)."""
    effective_dry_run = pilot
    effective_limit = 5 if pilot else limit
    limit_str = f"limit={effective_limit}" if effective_limit else "all available training videos"
    logger.info(f"Executing Phase 7: Building Visual Retrieval Index ({limit_str})...")
    from src.retrieval.build_index import build_index
    build_index(limit=effective_limit, dry_run=effective_dry_run)
    print("\n" + "=" * 60)
    print("PHASE 7 COMPLETE: VISUAL RETRIEVAL INDEX BUILT")
    print("Index directory: outputs/retrieval_index/")
    print("=" * 60 + "\n")


def run_phase_8(config_path: str = "configs/config.yaml"):
    """Phase 8: Generation Dataset Preparation."""
    logger.info("Executing Phase 8: Preparing Generation Datasets...")
    from training.prepare_generation_dataset import prepare_datasets
    prepare_datasets()
    print("\n" + "=" * 60)
    print("PHASE 8 COMPLETE: GENERATION DATASETS PREPARED")
    print("Train: data/processed/generation_train.jsonl")
    print("Val:   data/processed/generation_val.jsonl")
    print("=" * 60 + "\n")


def run_phase_9(config_path: str = "configs/config.yaml", pilot: bool = False, limit: int = None):
    """Phase 9: Multimodal Complaint Generation & Verification Demonstration."""
    logger.info("Executing Phase 9: End-to-End Multimodal Generation & Verification...")
    from demo_video_description import run_batch_evaluation
    if pilot:
        run_batch_evaluation(run_all=False, limit=5)
    elif limit is not None and limit > 0:
        run_batch_evaluation(run_all=False, limit=limit)
    elif limit == -1 or limit == 0:
        run_batch_evaluation(run_all=True, limit=None)
    else:
        # Default single-video demonstration
        run_batch_evaluation(run_all=False, limit=1)
    print("\n" + "=" * 60)
    print("PHASE 9 COMPLETE: END-TO-END MULTIMODAL COMPLAINT GENERATION")
    print("=" * 60 + "\n")


def run_phase_10(config_path: str = "configs/config.yaml"):
    """Phase 10: Full Evaluation & Ablation Study."""
    logger.info("Executing Phase 10: Evaluation & Ablation Study...")
    study = AblationStudy(cleaned_csv_path="data/processed/ComVID_cleaned.csv")
    results = study.run_ablation()
    study.print_comparison_table(results)


def run_unit_tests():
    """Runs all project unit tests."""
    logger.info("Executing unit test suite...")
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="MECG: Multilingual Evidence-Grounded Video Complaint Generation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=list(range(1, 11)),
        default=1,
        help="Pipeline phase to execute (1-10)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Execute project unit tests",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run in pilot mode (processes 5 samples only)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples/videos to process",
    )

    args = parser.parse_args()

    if args.test:
        run_unit_tests()
        return

    phase_actions = {
        1: lambda: run_phase_1(args.config),
        2: lambda: run_phase_2(args.config, pilot=args.pilot, limit=args.limit),
        3: lambda: run_phase_3(args.config, limit=5 if args.pilot else args.limit),
        4: lambda: run_phase_4(args.config),
        5: lambda: run_phase_5(args.config),
        7: lambda: run_phase_7(args.config, pilot=args.pilot, limit=args.limit),
        8: lambda: run_phase_8(args.config),
        9: lambda: run_phase_9(args.config, pilot=args.pilot, limit=args.limit),
        10: lambda: run_phase_10(args.config),
    }

    action = phase_actions.get(args.phase)
    if action:
        action()
    else:
        logger.info(f"Phase {args.phase} selected. Executing complete pipeline verification...")
        pipeline = MECGPipeline()
        print("Pipeline is instantiated and ready. Run notebooks or specify phase.")


if __name__ == "__main__":
    main()
