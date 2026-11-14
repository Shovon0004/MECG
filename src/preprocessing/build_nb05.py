"""
Script to generate notebooks/05_multilingual_processing.ipynb
"""

import json
from pathlib import Path


def make_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }


def markdown_cell(source):
    if isinstance(source, str):
        source = [line + "\n" for line in source.split("\n")]
        if source:
            source[-1] = source[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code_cell(source):
    if isinstance(source, str):
        source = [line + "\n" for line in source.split("\n")]
        if source:
            source[-1] = source[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }


def create_nb05():
    cells = [
        markdown_cell(
            "# MECG: Multilingual Evidence-Grounded Complaint Generation\n"
            "## Notebook 05: Multilingual Processing (Detection, Translation & Normalization)\n"
            "\n"
            "**Goal**: Process customer complaints and audio transcripts across multiple languages (English, Bengali, Hindi, Tamil, Telugu, etc.) "
            "using `facebook/nllb-200-distilled-600M`:\n"
            "1. **Language Detection**: Automatically recognize language and map between ISO-639-1, Whisper tokens, and NLLB BCP-47 codes.\n"
            "2. **Preservation**: Keep original customer text and transcripts intact.\n"
            "3. **Normalized Translation**: Translate non-English complaint evidence into English for cross-lingual downstream modeling.\n"
            "4. **Persistent Caching**: Cache all translations to eliminate duplicate model inferences.\n"
            "5. **Memory Efficiency**: Lazily load NLLB model and release GPU memory upon completion."
        ),
        code_cell(
            "# Environment & Module Path Setup\n"
            "import os\n"
            "import sys\n"
            "import json\n"
            "from pathlib import Path\n"
            "\n"
            "CURRENT_DIR = Path.cwd()\n"
            "if str(CURRENT_DIR) not in sys.path:\n"
            "    sys.path.append(str(CURRENT_DIR))\n"
            "if str(CURRENT_DIR.parent) not in sys.path:\n"
            "    sys.path.append(str(CURRENT_DIR.parent))\n"
            "\n"
            "from src.multilingual.language_detector import LanguageDetector\n"
            "from src.multilingual.translator import NLLBTranslator\n"
            "\n"
            "detector = LanguageDetector()\n"
            "print('Multilingual modules loaded successfully!')"
        ),
        markdown_cell(
            "### 1. Language Detection Across Diverse Scripts\n"
            "Test the language detector on sample customer complaint texts in English, Bengali, Hindi, Tamil, and Telugu."
        ),
        code_cell(
            "test_phrases = [\n"
            "    ('English', 'The plastic bottle has a pungent chemical smell and leaks water.'),\n"
            "    ('Bengali', 'প্লাস্টিকের বোতল থেকে তীব্র রাসায়নিক গন্ধ বের হচ্ছে এবং পানি পড়ছে।'),\n"
            "    ('Hindi', 'स्क्रीन काम नहीं कर रही है और बार-बार झिलमिला रही है।'),\n"
            "    ('Tamil', 'இந்த தயாரிப்பு மிகவும் சேதமடைந்துள்ளது மற்றும் வேலை செய்யவில்லை.'),\n"
            "    ('Telugu', 'ఈ ఉత్పత్తి సరిగ్గా పని చేయడం లేదు, నాణ్యత చాలా పేలవంగా ఉంది.')\n"
            "]\n"
            "\n"
            "print('Language Detection Results:')\n"
            "for label, phrase in test_phrases:\n"
            "    det = detector.detect_language(phrase)\n"
            "    print(f\"{label:10s} -> ISO: {det['iso_code']:3s} | NLLB: {det['nllb_code']:10s} | Script: {det['script']:10s} | Conf: {det['confidence']}\")"
        ),
        markdown_cell(
            "### 2. Multilingual Translation with NLLB-200 & Caching\n"
            "We initialize `NLLBTranslator`. When the input is already in English, it bypasses model inference automatically."
        ),
        code_cell(
            "translator = NLLBTranslator(\n"
            "    model_id='facebook/nllb-200-distilled-600M',\n"
            "    default_target_lang='eng_Latn',\n"
            "    cache_path='data/processed/translation_cache.json'\n"
            ")\n"
            "\n"
            "# 1. English input (zero-cost passthrough)\n"
            "en_res = translator.translate('The bottle cap is not locking properly.', src_lang='en', tgt_lang='en')\n"
            "print(f\"English Passthrough: {en_res['translated_text']} (Cached: {en_res['cached']})\")\n"
            "\n"
            "# 2. Benchmark Bengali translation check\n"
            "bn_text = 'প্লাস্টিকের বোতল থেকে তীব্র রাসায়নিক গন্ধ বের হচ্ছে।'\n"
            "bn_res = translator.translate(bn_text, src_lang='bn', tgt_lang='en')\n"
            "print(f\"\\nBengali Input : {bn_text}\")\n"
            "print(f\"English Output: {bn_res['translated_text']} (Cached: {bn_res['cached']})\")\n"
            "\n"
            "# 3. Benchmark Hindi translation check\n"
            "hi_text = 'स्क्रीन काम नहीं कर रही है और बार-बार झिलमिला रही है।'\n"
            "hi_res = translator.translate(hi_text, src_lang='hi', tgt_lang='en')\n"
            "print(f\"\\nHindi Input   : {hi_text}\")\n"
            "print(f\"English Output: {hi_res['translated_text']} (Cached: {hi_res['cached']})\")"
        ),
        markdown_cell(
            "### 3. Verify Cache Hit (Sub-Millisecond Retrieval)\n"
            "Re-running the same phrase demonstrates disk-backed cache retrieval without GPU compute."
        ),
        code_cell(
            "import time\n"
            "\n"
            "t0 = time.time()\n"
            "cached_res = translator.translate(bn_text, src_lang='bn', tgt_lang='en')\n"
            "elapsed_ms = (time.time() - t0) * 1000\n"
            "\n"
            "print(f\"Cached Retrieval Result: {cached_res['translated_text']}\")\n"
            "print(f\"Was Cached: {cached_res['cached']} | Elapsed Time: {elapsed_ms:.2f} ms\")\n"
            "assert cached_res['cached'] is True, 'Expected cached result!'"
        ),
        markdown_cell(
            "### 4. GPU Memory Cleanup\n"
            "In a multi-model system on Kaggle's dual T4 GPUs (16 GB per GPU), we must never leave unused models in VRAM."
        ),
        code_cell(
            "translator.release_memory()\n"
            "print('NLLB-200 VRAM cleared. Ready for Aspect & Emotion Classifier Training!')"
        ),
        markdown_cell(
            "### Summary & Next Steps\n"
            "- Multi-script language detection and code mapping are active.\n"
            "- NLLB-200 translation with caching and memory release is verified.\n"
            "- Proceed to **Notebook 06: Aspect & Emotion Classifier Training**."
        )
    ]
    return make_notebook(cells)


def main():
    nb05 = create_nb05()
    with open("notebooks/05_multilingual_processing.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb05, f, indent=2)
    print("Created notebooks/05_multilingual_processing.ipynb")


if __name__ == "__main__":
    main()
