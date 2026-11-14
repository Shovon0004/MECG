"""
Script to generate notebooks/04_multimodal_preprocessing.ipynb
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


def create_nb04():
    cells = [
        markdown_cell(
            "# MECG: Multilingual Evidence-Grounded Complaint Generation\n"
            "## Notebook 04: Multimodal Preprocessing (Frames, Audio & Whisper ASR)\n"
            "\n"
            "**Goal**: Process product videos into multimodal representations for downstream vision-language modeling "
            "and factual evidence grounding:\n"
            "1. Extract $N=8$ uniformly distributed, timestamped keyframes per video (`data/processed/frames/`).\n"
            "2. Extract standardized 16 kHz mono WAV audio tracks (`data/processed/audio/`).\n"
            "3. Perform multilingual speech recognition using Faster-Whisper / Whisper Large-v3-Turbo, "
            "detecting spoken languages, timestamps, and confidence without discarding original transcripts (`data/processed/transcripts.jsonl`)."
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
            "from src.vision.frame_extractor import FrameExtractor\n"
            "from src.audio.audio_extractor import AudioExtractor\n"
            "from src.audio.transcriber import WhisperTranscriber\n"
            "\n"
            "print('Multimodal preprocessing modules imported successfully!')"
        ),
        markdown_cell(
            "### 1. Frame Extraction (Configurable N=8 Frames)\n"
            "Uniform timestamping ensures visual defects appearing at any point in the video "
            "can be identified and timestamp-aligned by Qwen3-VL."
        ),
        code_cell(
            "video_dir = Path('data/videos')\n"
            "frames_dir = Path('data/processed/frames')\n"
            "\n"
            "frame_extractor = FrameExtractor(\n"
            "    frames_dir=frames_dir,\n"
            "    sample_frames=8,\n"
            "    target_resolution=(448, 448)\n"
            ")\n"
            "\n"
            "video_files = sorted(list(video_dir.glob('*.mp4')))\n"
            "print(f'Found {len(video_files)} videos in {video_dir}')\n"
            "\n"
            "if video_files:\n"
            "    sample_vid = video_files[0]\n"
            "    res = frame_extractor.extract_frames(sample_vid)\n"
            "    print(f\"Extracted {res['total_frames_extracted']} frames for {sample_vid.name}\")\n"
            "    for f in res['frames'][:4]:\n"
            "        print(f\"  Frame {f['index']} at timestamp {f['timestamp']}s -> {f['file_path']}\")\n"
            "else:\n"
            "    print('No videos downloaded yet. Run Notebook 03 to download sample videos.')"
        ),
        markdown_cell(
            "### 2. Audio Extraction (16 kHz Mono WAV)\n"
            "Standardizes audio tracks for Whisper ASR. Handles videos without an audio stream gracefully."
        ),
        code_cell(
            "audio_dir = Path('data/processed/audio')\n"
            "audio_extractor = AudioExtractor(\n"
            "    audio_dir=audio_dir,\n"
            "    sample_rate=16000,\n"
            "    channels=1\n"
            ")\n"
            "\n"
            "if video_files:\n"
            "    sample_vid = video_files[0]\n"
            "    a_res = audio_extractor.extract_audio(sample_vid)\n"
            "    print(f\"Audio Extraction for {sample_vid.name}:\")\n"
            "    print(f\"  Has Audio: {a_res['has_audio']}\")\n"
            "    if a_res['has_audio']:\n"
            "        print(f\"  Output Path: {a_res['audio_path']}\")\n"
            "        print(f\"  Duration   : {a_res['duration_seconds']}s\")"
        ),
        markdown_cell(
            "### 3. Whisper Speech Recognition & Language Detection\n"
            "Transcribes audio, detects spoken language (English, Bengali, Hindi, Tamil, etc.), "
            "and generates timestamped segments."
        ),
        code_cell(
            "transcripts_file = Path('data/processed/transcripts.jsonl')\n"
            "\n"
            "transcriber = WhisperTranscriber(\n"
            "    model_id='openai/whisper-large-v3-turbo',\n"
            "    fallback_model_id='openai/whisper-small',\n"
            "    device='auto',\n"
            "    compute_type='auto',\n"
            "    output_jsonl_path=transcripts_file\n"
            ")\n"
            "\n"
            "# Run transcription if audio is present\n"
            "if video_files and a_res.get('has_audio') and a_res.get('audio_path'):\n"
            "    t_res = transcriber.transcribe(a_res['audio_path'], video_id=sample_vid.stem)\n"
            "    print(f\"Transcription Result:\")\n"
            "    print(f\"  Language Detected: {t_res['detected_language']} (p={t_res['language_probability']})\")\n"
            "    print(f\"  Full Transcript  : {t_res['transcript']}\")\n"
            "    print(f\"  Segments Count   : {len(t_res['segments'])}\")\n"
            "else:\n"
            "    print('Sample video has no audio stream or no video present.')"
        ),
        markdown_cell(
            "### Summary & Next Steps\n"
            "- Timestamped frames and standardized audio tracks are extracted and cached.\n"
            "- Original-language speech transcripts with timestamps are logged to `transcripts.jsonl`.\n"
            "- Proceed to **Notebook 05: Multilingual Processing** for NLLB-200 translation and language normalization."
        )
    ]
    return make_notebook(cells)


def main():
    nb04 = create_nb04()
    with open("notebooks/04_multimodal_preprocessing.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb04, f, indent=2)
    print("Created notebooks/04_multimodal_preprocessing.ipynb")


if __name__ == "__main__":
    main()
