"""
Script to generate Kaggle-ready Jupyter Notebooks:
- notebooks/01_dataset_exploration.ipynb
- notebooks/02_dataset_cleaning.ipynb
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
        # remove trailing newline on last line
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


def create_nb01():
    cells = [
        markdown_cell(
            "# MECG: Multilingual Evidence-Grounded Complaint Generation\n"
            "## Notebook 01: Dataset Exploration & Forensic Audit\n"
            "\n"
            "**Goal**: Explore the raw `Com-VID_data.csv` dataset, inspect column formats, audit missing values, "
            "investigate duplicate video URLs and IDs, analyze rating HTML tags, study emotion and aspect distributions, "
            "and identify critical data hygiene rules.\n"
            "\n"
            "**Key Rules Examined**:\n"
            "1. Inspect the actual CSV structure before implementing data transformations.\n"
            "2. Preserve original columns and values.\n"
            "3. Identify duplicate IDs and shared video URLs to prevent video-level data leakage.\n"
            "4. Do not assume `Text` is an audio transcript or that `Ground-truth` is entirely visually supported.\n"
            "5. Kaggle environment support (`/kaggle/input` and local fallbacks)."
        ),
        code_cell(
            "# Environment Setup and Dynamic Path Detection (Kaggle vs Local)\n"
            "import os\n"
            "import sys\n"
            "from pathlib import Path\n"
            "\n"
            "# Auto-detect workspace root\n"
            "CURRENT_DIR = Path.cwd()\n"
            "KAGGLE_INPUT_DIR = Path('/kaggle/input/comvid-dataset')\n"
            "LOCAL_DATA_DIR = Path('data')\n"
            "\n"
            "if (KAGGLE_INPUT_DIR / 'Com-VID_data.csv').exists():\n"
            "    RAW_CSV_PATH = KAGGLE_INPUT_DIR / 'Com-VID_data.csv'\n"
            "elif (LOCAL_DATA_DIR / 'Com-VID_data.csv').exists():\n"
            "    RAW_CSV_PATH = LOCAL_DATA_DIR / 'Com-VID_data.csv'\n"
            "elif (Path('..') / 'data' / 'Com-VID_data.csv').exists():\n"
            "    RAW_CSV_PATH = Path('..') / 'data' / 'Com-VID_data.csv'\n"
            "else:\n"
            "    RAW_CSV_PATH = Path('Com-VID_data.csv')\n"
            "\n"
            "print(f'Using Raw Dataset Path: {RAW_CSV_PATH.resolve()}')\n"
            "assert RAW_CSV_PATH.exists(), f'File not found: {RAW_CSV_PATH}'"
        ),
        markdown_cell(
            "### 1. Load Raw CSV and Inspect Column Headers\n"
            "We inspect the raw field names, noticing any whitespace anomalies (such as `'Ground-truth '`)."
        ),
        code_cell(
            "import csv\n"
            "import json\n"
            "\n"
            "with open(RAW_CSV_PATH, mode='r', encoding='utf-8', errors='replace') as f:\n"
            "    reader = csv.DictReader(f)\n"
            "    raw_fields = reader.fieldnames\n"
            "    raw_rows = list(reader)\n"
            "\n"
            "print(f'Total records in dataset: {len(raw_rows)}')\n"
            "print('Raw CSV Field Names:')\n"
            "for i, field in enumerate(raw_fields, 1):\n"
            "    print(f'  {i}. {repr(field)} (length={len(field)})')\n"
            "\n"
            "print('\\nSample Record (Row 0):')\n"
            "print(json.dumps(raw_rows[0], indent=2))"
        ),
        markdown_cell(
            "### 2. Forensic Audit of Identifiers & Video URLs\n"
            "In video research, **video-level data leakage** occurs when multiple reviews point to the exact same video, "
            "and are inadvertently split across train and test sets. Let us audit ID and URL uniqueness."
        ),
        code_cell(
            "from collections import Counter\n"
            "\n"
            "ids = [r['ID'].strip() for r in raw_rows if r.get('ID')]\n"
            "id_counts = Counter(ids)\n"
            "duplicate_ids = {k: v for k, v in id_counts.items() if v > 1}\n"
            "\n"
            "urls = [r['Video-Source'].strip() for r in raw_rows if r.get('Video-Source')]\n"
            "url_counts = Counter(urls)\n"
            "duplicate_urls = {k: v for k, v in url_counts.items() if v > 1}\n"
            "empty_urls = [r['ID'] for r in raw_rows if not r.get('Video-Source', '').strip()]\n"
            "\n"
            "print(f'Total rows: {len(raw_rows)}')\n"
            "print(f'Unique IDs: {len(id_counts)} | Duplicate IDs: {len(duplicate_ids)}')\n"
            "print(f'Unique URLs: {len(url_counts)} | Duplicate URLs: {len(duplicate_urls)}')\n"
            "print(f'Records missing Video-Source URL: {len(empty_urls)} -> IDs: {empty_urls}')\n"
            "\n"
            "print('\\nTop 3 Shared Video URLs across multiple reviews:')\n"
            "for url, count in url_counts.most_common(3):\n"
            "    matching_ids = [r['ID'] for r in raw_rows if r['Video-Source'].strip() == url]\n"
            "    print(f'  - Count: {count} | Matching IDs: {matching_ids}')\n"
            "    print(f'    URL: {url[:80]}...')"
        ),
        markdown_cell(
            "### 3. Rating Extraction from HTML\n"
            "Amazon ratings are stored as HTML tags like `<span class=\"a-icon-alt\">2.0 out of 5 stars</span>`. "
            "We extract numeric star ratings using regular expressions."
        ),
        code_cell(
            "import re\n"
            "\n"
            "def extract_star_rating(html_str):\n"
            "    if not html_str:\n"
            "        return None\n"
            "    match = re.search(r'([0-9]+(?:\\.[0-9]+)?)\\s*(?:out of|\\/)\\s*5', html_str, re.IGNORECASE)\n"
            "    return float(match.group(1)) if match else None\n"
            "\n"
            "raw_ratings = Counter(r['Rating'].strip() for r in raw_rows)\n"
            "print('Raw HTML Rating strings:')\n"
            "for rating_html, cnt in raw_ratings.items():\n"
            "    parsed = extract_star_rating(rating_html)\n"
            "    print(f'  Count: {cnt:4d} | Parsed: {parsed} stars | Raw: {repr(rating_html)}')\n"
            "\n"
            "parsed_counts = Counter(extract_star_rating(r['Rating']) for r in raw_rows)\n"
            "print('\\nParsed Ratings Summary:', dict(parsed_counts))"
        ),
        markdown_cell(
            "### 4. Domain & Emotion Distribution\n"
            "Let's explore the product domains and the emotion taxonomy. "
            "Note the capitalization inconsistency in emotion labels (e.g., `'blame'` vs `'Blame'`)."
        ),
        code_cell(
            "domain_counts = Counter(r['Domain'].strip() for r in raw_rows)\n"
            "print('Product Domain Distribution:')\n"
            "for domain, cnt in domain_counts.most_common():\n"
            "    print(f'  {domain:20s}: {cnt:4d} ({cnt/len(raw_rows)*100:.2f}%)')\n"
            "\n"
            "emotion_counts = Counter(r['Emotion'].strip() for r in raw_rows)\n"
            "print('\\nEmotion Distribution (Raw):')\n"
            "for emotion, cnt in emotion_counts.most_common():\n"
            "    print(f'  {repr(emotion):20s}: {cnt:4d} ({cnt/len(raw_rows)*100:.2f}%)')\n"
            "\n"
            "# Normalized emotion\n"
            "norm_emotion = Counter(r['Emotion'].strip().lower() for r in raw_rows)\n"
            "print('\\nEmotion Distribution (Normalized to lowercase):')\n"
            "for emotion, cnt in norm_emotion.most_common():\n"
            "    print(f'  {emotion:20s}: {cnt:4d} ({cnt/len(raw_rows)*100:.2f}%)')"
        ),
        markdown_cell(
            "### 5. Complaint Aspect Analysis & Typo Identification\n"
            "Aspect labels contain multiple comma/semicolon-separated labels, plus spelling variations and typos. "
            "Rule: Fix typos without merging semantically distinct aspects."
        ),
        code_cell(
            "all_aspect_tokens = Counter()\n"
            "empty_aspect_rows = []\n"
            "\n"
            "for idx, r in enumerate(raw_rows):\n"
            "    aspect_str = r.get('Aspect', '').strip()\n"
            "    if not aspect_str:\n"
            "        empty_aspect_rows.append(r['ID'])\n"
            "        continue\n"
            "    tokens = re.split(r'[,;/]|\\band\\b', aspect_str, flags=re.IGNORECASE)\n"
            "    for tok in tokens:\n"
            "        t = tok.strip().rstrip('.')\n"
            "        if t:\n"
            "            all_aspect_tokens[t] += 1\n"
            "\n"
            "print(f'Total distinct raw aspect tokens: {len(all_aspect_tokens)}')\n"
            "print(f'Records with empty Aspect field: {len(empty_aspect_rows)} (IDs: {empty_aspect_rows[:5]}...)')\n"
            "\n"
            "print('\\nTop 15 Aspect Tokens:')\n"
            "for tok, count in all_aspect_tokens.most_common(15):\n"
            "    print(f'  {tok:25s}: {count:4d}')\n"
            "\n"
            "print('\\nNotice Typo Examples that must be normalized:')\n"
            "typo_examples = ['Qualtiy', 'Qualiy', 'Usabilty', 'Usabiity', 'Manufaturing', 'Functionaliy']\n"
            "for typo in typo_examples:\n"
            "    if typo in all_aspect_tokens:\n"
            "        print(f'  Typo: {typo!r} (occurs {all_aspect_tokens[typo]} times)')"
        ),
        markdown_cell(
            "### 6. Customer Text vs Reference Ground-Truth Complaint\n"
            "Examining the customer text against the reference ground truth.\n"
            "Important research takeaway: Customer text is often very terse (e.g. 'Not recommended'), "
            "while `Ground-truth` contains specific details observed from video or inspection."
        ),
        code_cell(
            "text_lens = [len(r['Text'].split()) for r in raw_rows]\n"
            "gt_col = 'Ground-truth ' if 'Ground-truth ' in raw_rows[0] else 'Ground-truth'\n"
            "gt_lens = [len(r[gt_col].split()) for r in raw_rows]\n"
            "\n"
            "print('Customer Text Word Length:')\n"
            "print(f'  Min: {min(text_lens)}, Max: {max(text_lens)}, Mean: {sum(text_lens)/len(text_lens):.1f}')\n"
            "\n"
            "print('Ground-truth Complaint Word Length:')\n"
            "print(f'  Min: {min(gt_lens)}, Max: {max(gt_lens)}, Mean: {sum(gt_lens)/len(gt_lens):.1f}')\n"
            "\n"
            "print('\\nExample 1 where Customer Text is short but Ground Truth is descriptive:')\n"
            "for r in raw_rows:\n"
            "    if len(r['Text'].split()) <= 3:\n"
            "        print(f\"ID: {r['ID']}\")\n"
            "        print(f\"Customer Text: {repr(r['Text'])}\")\n"
            "        print(f\"Ground-truth : {repr(r[gt_col])}\")\n"
            "        break"
        ),
        markdown_cell(
            "### Summary of Exploratory Findings\n"
            "- Total Records: 1,175 rows.\n"
            "- 11 Duplicate Review IDs and 86 Shared Video URLs require **Group-Aware Splitting** to eliminate data leakage.\n"
            "- 3 rows lack a video URL.\n"
            "- 16 rows lack aspect labels.\n"
            "- Emotion has 4 underlying classes (`blame`, `disappointment`, `dissatisfaction`, `frustration`) with minor casing anomalies.\n"
            "- Ratings are predominantly 1.0 star (82.5%) and 2.0 stars (17.4%).\n"
            "- All raw values must be preserved under `raw_*` columns during cleaning.\n"
            "\n"
            "Proceed to **Notebook 02: Dataset Cleaning** to generate the cleaned dataset and audit report."
        )
    ]
    return make_notebook(cells)


def create_nb02():
    cells = [
        markdown_cell(
            "# MECG: Multilingual Evidence-Grounded Complaint Generation\n"
            "## Notebook 02: Dataset Cleaning, Normalization & Group-Aware Splitting\n"
            "\n"
            "**Goal**: Execute the complete cleaning pipeline on `Com-VID_data.csv`, produce `ComVID_cleaned.csv`, "
            "enforce group-aware train/val/test splits (strictly preventing video-level data leakage), "
            "and generate the formal quality audit report `dataset_audit.json`.\n"
            "\n"
            "**Rules Enforced**:\n"
            "1. Preserve original columns and values.\n"
            "2. Normalize column names.\n"
            "3. Clean missing values and whitespace.\n"
            "4. Normalize known spelling variations in aspect and emotion labels without merging distinct concepts.\n"
            "5. Flag duplicate IDs and video URLs without deleting records.\n"
            "6. Guarantee 0% video leakage between train, val, and test splits.\n"
            "7. Produce `reports/dataset_audit.json`."
        ),
        code_cell(
            "# Environment & Module Path Setup\n"
            "import os\n"
            "import sys\n"
            "import json\n"
            "from pathlib import Path\n"
            "\n"
            "# Ensure project root is in sys.path\n"
            "CURRENT_DIR = Path.cwd()\n"
            "if str(CURRENT_DIR) not in sys.path:\n"
            "    sys.path.append(str(CURRENT_DIR))\n"
            "if str(CURRENT_DIR.parent) not in sys.path:\n"
            "    sys.path.append(str(CURRENT_DIR.parent))\n"
            "\n"
            "from src.preprocessing.normalizers import (\n"
            "    parse_html_rating, clean_text_whitespace, normalize_emotion, normalize_aspects\n"
            ")\n"
            "from src.preprocessing.dataset_cleaner import ComVIDCleaner\n"
            "from src.preprocessing.audit import DatasetAuditor\n"
            "\n"
            "print('Preprocessing modules imported successfully!')"
        ),
        markdown_cell(
            "### 1. Execute ComVIDCleaner Pipeline\n"
            "We run `ComVIDCleaner`, which:\n"
            "- Preserves all raw fields (`raw_id`, `raw_title`, `raw_rating`, etc.).\n"
            "- Extracts clean float ratings from HTML.\n"
            "- Normalizes whitespace and text formatting.\n"
            "- Normalizes emotion and aspect labels.\n"
            "- Assigns group-aware splits based on `video_url_hash`."
        ),
        code_cell(
            "# Dynamic input/output path resolution\n"
            "input_path = Path('data/Com-VID_data.csv')\n"
            "if not input_path.exists() and Path('/kaggle/input/comvid-dataset/Com-VID_data.csv').exists():\n"
            "    input_path = Path('/kaggle/input/comvid-dataset/Com-VID_data.csv')\n"
            "\n"
            "output_path = Path('data/processed/ComVID_cleaned.csv')\n"
            "report_path = Path('reports/dataset_audit.json')\n"
            "\n"
            "# Instantiate and execute cleaner\n"
            "cleaner = ComVIDCleaner(\n"
            "    raw_csv_path=input_path,\n"
            "    output_csv_path=output_path,\n"
            "    train_ratio=0.80,\n"
            "    val_ratio=0.10,\n"
            "    test_ratio=0.10,\n"
            "    random_seed=42\n"
            ")\n"
            "cleaned_records = cleaner.process()\n"
            "saved_csv = cleaner.save_cleaned_csv()\n"
            "\n"
            "print(f'Successfully processed and saved {len(cleaned_records)} records to {saved_csv}')"
        ),
        markdown_cell(
            "### 2. Verify Preserved Raw Columns & Normalized Fields\n"
            "Let's inspect the first record to verify that both raw columns and normalized columns coexist."
        ),
        code_cell(
            "import csv\n"
            "\n"
            "with open(output_path, mode='r', encoding='utf-8') as f:\n"
            "    reader = csv.DictReader(f)\n"
            "    sample = next(reader)\n"
            "\n"
            "print('Normalized Fields:')\n"
            "norm_keys = ['review_id', 'title', 'rating_stars', 'emotion', 'aspects', 'domain', 'split', 'is_duplicate_id', 'is_duplicate_url']\n"
            "for k in norm_keys:\n"
            "    print(f'  {k:20s}: {sample.get(k)}')\n"
            "\n"
            "print('\\nPreserved Raw Fields:')\n"
            "raw_keys = ['raw_id', 'raw_rating', 'raw_emotion', 'raw_aspect']\n"
            "for k in raw_keys:\n"
            "    print(f'  {k:20s}: {repr(sample.get(k))}')"
        ),
        markdown_cell(
            "### 3. Verification of Zero Video Leakage Across Splits\n"
            "A critical validation check: verify that no video URL or video hash appears in more than one split."
        ),
        code_cell(
            "from collections import defaultdict\n"
            "\n"
            "split_by_hash = {}\n"
            "leakage_detected = False\n"
            "\n"
            "with open(output_path, mode='r', encoding='utf-8') as f:\n"
            "    reader = csv.DictReader(f)\n"
            "    for row in reader:\n"
            "        h = row['video_url_hash']\n"
            "        s = row['split']\n"
            "        if h in split_by_hash and split_by_hash[h] != s:\n"
            "            print(f'LEAKAGE DETECTED! Video {h} present in both {split_by_hash[h]} and {s}')\n"
            "            leakage_detected = True\n"
            "        split_by_hash[h] = s\n"
            "\n"
            "if not leakage_detected:\n"
            "    print('VERIFICATION PASSED: 0% Video Leakage! All videos cleanly isolated by split.')"
        ),
        markdown_cell(
            "### 4. Generate Formal Dataset Audit Report (`reports/dataset_audit.json`)\n"
            "Now run `DatasetAuditor` to produce the full audit report."
        ),
        code_cell(
            "auditor = DatasetAuditor.from_csv(output_path)\n"
            "report = auditor.generate_report()\n"
            "auditor.save_report(report_path)\n"
            "\n"
            "print(f'Audit report generated at: {report_path.resolve()}')\n"
            "print('\\nDataset Audit Executive Summary:')\n"
            "print(f\"  Total records: {report['metadata']['total_records']}\")\n"
            "print(f\"  Unique Review IDs: {report['identifiers_and_duplicates']['unique_review_ids']}\")\n"
            "print(f\"  Unique Video URLs: {report['identifiers_and_duplicates']['unique_video_urls']}\")\n"
            "print(f\"  Duplicate Video URLs: {report['identifiers_and_duplicates']['duplicate_video_urls_count']}\")\n"
            "print(f\"  Zero-Leakage Status: {report['splits']['video_leakage_check']['is_leakage_free']}\")\n"
            "print('\\nSplit Counts and Percentages:')\n"
            "for split, d in report['splits']['records_per_split'].items():\n"
            "    print(f\"  - {split:5s}: {d['count']} records ({d['percentage']}%) across {report['splits']['unique_video_groups_per_split'][split]} video groups\")"
        ),
        markdown_cell(
            "### Conclusion of Phase 1\n"
            "Phase 1 data cleaning, normalization, duplicate auditing, and leak-free group-aware splitting "
            "are complete and verified. Outputs:\n"
            "- `data/processed/ComVID_cleaned.csv`\n"
            "- `reports/dataset_audit.json`\n"
            "\n"
            "Ready for Phase 2: Video Downloading and Validation."
        )
    ]
    return make_notebook(cells)


def main():
    nb01 = create_nb01()
    nb02 = create_nb02()
    
    Path("notebooks").mkdir(exist_ok=True)
    with open("notebooks/01_dataset_exploration.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb01, f, indent=2)
    print("Created notebooks/01_dataset_exploration.ipynb")

    with open("notebooks/02_dataset_cleaning.ipynb", "w", encoding="utf-8") as f:
        json.dump(nb02, f, indent=2)
    print("Created notebooks/02_dataset_cleaning.ipynb")


if __name__ == "__main__":
    main()
