# MECG — Multilingual Evidence-Grounded Complaint Generation

**MECG** is a research-oriented, multimodal AI system designed to analyze product complaint videos, understand visual defects, recognize multilingual customer speech, identify complaint aspects and emotions, retrieve relevant references, generate natural-language complaints, and verify that claims are strictly grounded in video and audio evidence.

The project is built upon the **Com-VID dataset** (1,175 records across four product domains: *Electronic Gadgets*, *Fashion*, *Household Item*, *Others*).

---

## 1. Project Architecture

The architecture uses specialized modular components rather than forcing a single model to handle disparate tasks:

```
                                      +------------------------+
                                      |   Customer Review      |
                                      |  (Text + Star Rating)  |
                                      +-----------+------------+
                                                  |
+---------------------+                           v
| Raw Product Video   |                 +-------------------+
|  (HLS M3U8 / MP4)   |                 | DeBERTa-v3 Multi- |
+----------+----------+                 | Label Classifier  |
           |                            +---------+---------+
     +-----+-----+                                |
     |           |                                v
     v           v                       Aspects & Emotion
+---------+ +---------+                           |
| OpenCV/ | | Faster- |                           |
| FFmpeg  | | Whisper |                           |
+----+----+ +----+----+                           |
     |           |                                |
  Sampled     Detected                            |
  Frames     Transcript                           |
     |           |                                |
     |           v                                |
     |     +-----------+                          |
     |     | NLLB-200  |                          |
     |     | Translate |                          |
     |     +-----+-----+                          |
     |           |                                |
     v           v                                v
+---------------------------------------------------+
|            SigLIP 2 + FAISS Index                 |
|   (Contextual Visual & Complaint Retrieval)       |
+-------------------------+-------------------------+
                          |
                          v
+---------------------------------------------------+
|        Qwen3-VL-4B-Instruct (QLoRA / PEFT)        |
|  Evidence-Grounded Complaint Generation (Multi-L) |
+-------------------------+-------------------------+
                          |
                          v
+---------------------------------------------------+
|    Evidence Verification Pipeline (DeBERTa MNLI)  |
| - Claim Decomposition & Timestamp Alignment       |
| - Unsupported Claim Flagging / Revision           |
| - Deterministic Grounding & Verification Score    |
+---------------------------------------------------+
```

---

## 2. Directory Structure

```
MECG/
│
├── data/
│   ├── Com-VID_data.csv                   # Raw dataset (1,175 rows)
│   ├── videos/                            # Downloaded MP4/HLS transcode cache
│   └── processed/
│       ├── ComVID_cleaned.csv             # Cleaned dataset with splits & audit flags
│       ├── transcripts.jsonl              # Audio transcripts with timestamps & lang
│       ├── frames/                        # Extracted video keyframes
│       └── audio/                         # Extracted 16kHz WAV mono audio
│
├── configs/
│   ├── config.yaml                        # Master configuration (paths, models, hyperparameters)
│   └── languages.yaml                     # NLLB-200 and Whisper language mappings
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb       # Forensic dataset audit & EDA
│   ├── 02_dataset_cleaning.ipynb          # Hygiene pipeline, normalization, group-split
│   ├── 03_video_downloading.ipynb         # Resilient HLS/M3U8 download & caching
│   ├── 04_multimodal_preprocessing.ipynb  # Frame & audio extraction
│   ├── 05_multilingual_processing.ipynb   # Whisper ASR & NLLB-200 translation
│   ├── 06_aspect_emotion_training.ipynb   # DeBERTa-v3 aspect & emotion classifiers
│   ├── 07_qwen3vl_finetuning.ipynb        # Qwen3-VL QLoRA fine-tuning
│   ├── 08_full_pipeline_demo.ipynb        # End-to-end multimodal pipeline
│   └── 09_evaluation.ipynb                # Grounding score, CR metric & ablations
│
├── src/
│   ├── preprocessing/                     # Data cleaning, normalization, audit
│   │   ├── normalizers.py                 # Typo maps, HTML parser, hash utilities
│   │   ├── dataset_cleaner.py             # ComVIDCleaner with group-aware splitting
│   │   └── audit.py                       # Quality metrics & JSON report generator
│   ├── multilingual/                      # Whisper ASR & NLLB-200 translation
│   ├── vision/                            # Frame sampling & Qwen3-VL vision analysis
│   ├── audio/                             # Audio extraction & preprocessing
│   ├── classification/                    # Aspect & emotion classification
│   ├── retrieval/                         # SigLIP 2 visual embeddings & FAISS index
│   ├── generation/                        # Multimodal prompt construction & Qwen3-VL
│   ├── verification/                      # Claim decomposition & NLI grounding
│   └── pipeline.py                        # Unified end-to-end inference pipeline
│
├── training/                              # Standalone training & fine-tuning scripts
├── evaluation/                            # Grounding metrics & ablation calculators
├── reports/
│   └── dataset_audit.json                 # Comprehensive dataset quality audit
├── outputs/                               # Checkpoints, FAISS indices, generation logs
├── tests/
│   └── test_preprocessing.py              # Unit tests for preprocessing & leakage
├── requirements.txt                       # Pinned Python package dependencies
├── README.md                              # Project documentation & execution guide
└── main.py                                # CLI orchestrator
```

---

## 3. Forensic Dataset Audit (Phase 1 Findings)

Inspection of `data/Com-VID_data.csv` revealed the following key characteristics:

1. **Total Records**: Exactly **1,175** records across 4 product domains:
   - *Electronic Gadgets*: 655 (55.74%)
   - *Fashion*: 273 (23.23%)
   - *Household Item*: 202 (17.19%)
   - *Others*: 45 (3.83%)
2. **Ratings**: Stored as HTML tags (e.g. `<span class="a-icon-alt">1.0 out of 5 stars</span>`):
   - 1.0 Star: 970 (82.55%)
   - 2.0 Stars: 204 (17.36%)
   - 3.0 Stars: 1 (0.09%)
3. **Identifiers & Duplicate Video URLs**:
   - Unique Review IDs: 1,164 (11 IDs appear twice with different domain or aspect annotations).
   - Unique Video URLs: 1,086 (86 URLs are shared across multiple customer reviews).
   - Missing Video URLs: Exactly 3 records (`R2TD0TOL058O0B`, `RZNHON2W0MNUX`, `T25142`) have empty video URLs.
4. **Data Leakage Prevention**:
   - Random row-level splitting would cause video leakage between training and testing.
   - We enforce **Group-Aware Splitting** keyed on `video_url_hash`. All reviews sharing the same video URL are guaranteed to reside in the same split:
     - **Train**: 940 records (80.0%) across 871 video groups.
     - **Val**: 118 records (10.04%) across 108 video groups.
     - **Test**: 117 records (9.96%) across 110 video groups.
     - **Overlap Count**: Exactly **0%** leakage.
5. **Emotion Taxonomy**:
   - Normalized into 4 canonical classes: `disappointment` (342), `dissatisfaction` (339), `blame` (305), and `frustration` (189).
6. **Aspect Label Hygiene**:
   - Typos like `Qualtiy`, `Usabilty`, `Manufaturing`, `Functionaliy` were normalized to canonical terms without merging semantically distinct concepts (e.g., `Usability` and `Durability` remain independent).

---

## 4. Phase 1 Verification & Execution

### Running Locally

```bash
# 1. Run unit tests (verifies normalizers, row preservation, and zero-leakage splits)
python3 main.py --test

# 2. Execute Phase 1 cleaning and audit generation
python3 main.py --phase 1

# 3. View the generated audit report
cat reports/dataset_audit.json
```

### Running on Kaggle

The notebooks are configured to automatically detect both Kaggle and local environments:
- In Kaggle: Dataset path resolves to `/kaggle/input/comvid-dataset/Com-VID_data.csv`. Output files write to `/kaggle/working/data/processed/`.
- Open and run:
  1. `notebooks/01_dataset_exploration.ipynb`
  2. `notebooks/02_dataset_cleaning.ipynb`
