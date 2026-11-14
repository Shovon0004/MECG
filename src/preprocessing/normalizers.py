"""
Normalizers and utility functions for the Com-VID dataset.
Handles text cleaning, rating extraction, emotion standardizing,
and aspect typo normalization without merging semantically distinct labels.
"""

import hashlib
import html
import re
from typing import Dict, List, Optional, Set, Tuple


# Standard emotion taxonomy present in the dataset
VALID_EMOTIONS: Set[str] = {
    "blame",
    "disappointment",
    "dissatisfaction",
    "frustration"
}

# Mapping of known emotion spelling/casing variations
EMOTION_MAPPING: Dict[str, str] = {
    "blame": "blame",
    "Blame": "blame",
    "disappointment": "disappointment",
    "dissatisfaction": "dissatisfaction",
    "frustration": "frustration"
}

# Normalization mapping for aspect labels
# Rule: Fix typos, casing, trailing punctuation, and syntactic separators.
# Do NOT arbitrarily merge semantically different aspects (e.g., Usability vs Durability vs Functionality).
ASPECT_NORMALIZATION_MAP: Dict[str, str] = {
    # Quality variations
    "quality": "Quality",
    "Quality": "Quality",
    "Qualtiy": "Quality",
    "Qualiy": "Quality",
    "Qaulity": "Quality",
    "Qualty": "Quality",
    "Poor Quality": "Poor Quality",
    "Damaged quality": "Damaged Quality",
    "Damaged Quality": "Damaged Quality",
    "Manufacturing Quality": "Manufacturing Quality",
    "Packaging Quality": "Packaging Quality",
    "Irreversible Quality": "Irreversible Quality",
    "Pre-owned quality": "Pre-owned quality",

    # Usability variations
    "usability": "Usability",
    "Usability": "Usability",
    "Usabilty": "Usability",
    "Usabiity": "Usability",
    "Usage": "Usage",

    # Functionality & Performance
    "functionality": "Functionality",
    "Functionality": "Functionality",
    "Functionaliy": "Functionality",
    "performance": "Performance",
    "Performance": "Performance",
    "capability": "Capability",
    "Capability": "Capability",

    # Manufacturing variations
    "manufacturing": "Manufacturing",
    "Manufacturing": "Manufacturing",
    "Manufaturing": "Manufacturing",
    "Manfacturing": "Manufacturing",

    # Originality variations
    "originality": "Originality",
    "Originality": "Originality",
    "Originalty": "Originality",
    "Originaltiy": "Originality",
    "originlaity": "Originality",

    # Defectiveness
    "defective": "Defective",
    "Defective": "Defective",
    "Defective Quality": "Defective Quality",

    # Correctness & Discrepancy
    "correctness": "Correctness",
    "Correctness": "Correctness",
    "Correctnes": "Correctness",
    "Disrepancy": "Discrepancy",
    "Discrepancy": "Discrepancy",

    # Condition & Physical state
    "condition": "Condition",
    "Condition": "Condition",
    "Condtion": "Condition",
    "Physical State": "Physical State",
    "Pre-owned condition": "Pre-owned condition",
    "Pre-owned": "Pre-owned",
    "Unworn": "Unworn",

    # Comfort
    "comfortness": "Comfortness",
    "Comfortness": "Comfortness",
    "Confortness": "Comfortness",

    # Connection
    "connection": "Connection",
    "Connection": "Connection",
    "Connetion": "Connection",

    # Cost & Price
    "cost": "Cost",
    "Cost": "Cost",
    "price": "Price",
    "Price": "Price",
    "Cost-effectiveness": "Cost-effectiveness",
    "Cost effectiveness": "Cost-effectiveness",
    "Price unstability": "Price unstability",

    # Colour / Color (standardizing to predominant UK spelling in dataset: Colour)
    "color": "Colour",
    "Color": "Colour",
    "colour": "Colour",
    "Colour": "Colour",

    # Design & Finishing
    "design": "Design",
    "Design": "Design",
    "Designing": "Design",
    "finishing": "Finishing",
    "Finishing": "Finishing",

    # Packaging
    "packaging": "Packaging",
    "Packaging": "Packaging",

    # Size & Shape
    "size": "Size",
    "Size": "Size",
    "shape": "Shape",
    "Shape": "Shape",

    # Representation
    "representation": "Representation",
    "Representation": "Representation",

    # Others
    "fraud": "Fraud",
    "Fraud": "Fraud",
    "service": "Service",
    "Service": "Service",
    "responsibility": "Responsibility",
    "Responsibility": "Responsibility",
    "Reponsibility": "Responsibility",
    "durability": "Durability",
    "Durability": "Durability",
    "cleanliness": "Cleanliness",
    "Cleanliness": "Cleanliness",
    "effectiveness": "Effectiveness",
    "Effectiveness": "Effectiveness",
    "completeness": "Completeness",
    "Completeness": "Completeness",
    "reliability": "Reliability",
    "Reliability": "Reliability",
    "fabric": "Fabric",
    "Fabric": "Fabric",
    "satisfaction": "Satisfaction",
    "Satisfaction": "Satisfaction",
    "Unique inclusion": "Unique inclusion",
    "Brand": "Brand",
    "brand": "Brand",
    "compatibility": "Compatibility",
    "Compatibility": "Compatibility",
    "Flickering illumination": "Flickering illumination",
    "Legibility": "Legibility",
    "Negative feedback": "Negative feedback",
    "Noisy": "Noisy",
    "Suitability": "Suitability",
    "Time": "Time",
    "Weak battery": "Weak battery"
}


def clean_text_whitespace(text: Optional[str]) -> str:
    """Normalizes whitespace and removes excessive control characters."""
    if not text:
        return ""
    text = html.unescape(text)
    # Replace carriage returns and newlines with spaces or uniform newline
    text = re.sub(r'[\r\n]+', ' ', text)
    # Collapse multiple whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def parse_html_rating(raw_rating: Optional[str]) -> Optional[float]:
    """
    Parses Amazon HTML rating string to extract a numeric float.
    Handles:
      '<span class="a-icon-alt">2.0 out of 5 stars</span>' -> 2.0
      '<span class="s-icon-alt">1.0 out of 5 waste</span>' -> 1.0
    """
    if not raw_rating:
        return None
    # Unescape HTML entities
    unescaped = html.unescape(raw_rating)
    match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*(?:out of|\/)\s*5', unescaped, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def normalize_emotion(raw_emotion: Optional[str]) -> str:
    """
    Normalizes emotion labels, trimming whitespace and resolving casing variations.
    """
    if not raw_emotion:
        return "unknown"
    cleaned = raw_emotion.strip()
    normalized = EMOTION_MAPPING.get(cleaned, cleaned.lower())
    return normalized


def normalize_aspects(raw_aspect: Optional[str]) -> List[str]:
    """
    Tokenizes raw aspect string by comma, semicolon, slash, or conjunction 'and',
    cleans trailing dots and whitespace, and maps typos to canonical aspect names.
    Preserves semantic distinctions. Returns a sorted list of unique aspects.
    """
    if not raw_aspect:
        return []
    
    cleaned = raw_aspect.strip()
    if not cleaned:
        return []

    # Split on commas, semicolons, slashes, or word boundary 'and'
    tokens = re.split(r'[,;/]|\band\b', cleaned, flags=re.IGNORECASE)
    normalized_aspects: List[str] = []

    for token in tokens:
        t = token.strip().rstrip('.')
        if not t:
            continue
        
        # Check direct lookup in normalization map
        if t in ASPECT_NORMALIZATION_MAP:
            norm_val = ASPECT_NORMALIZATION_MAP[t]
        elif t.lower() in ASPECT_NORMALIZATION_MAP:
            norm_val = ASPECT_NORMALIZATION_MAP[t.lower()]
        else:
            # Fallback title-casing if unseen
            norm_val = t.strip()
            
        if norm_val and norm_val not in normalized_aspects:
            normalized_aspects.append(norm_val)

    return sorted(normalized_aspects)


def compute_url_hash(url: Optional[str], fallback_id: str = "") -> str:
    """
    Computes a deterministic 16-character SHA-256 hash for a video URL.
    Used for group-aware splitting to strictly prevent video leakage.
    If the URL is missing, falls back to a deterministic hash of the record ID.
    """
    if url and url.strip():
        source = url.strip()
    else:
        source = f"no_url_{fallback_id.strip()}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def detect_language_hint(text: Optional[str]) -> str:
    """
    Lightweight heuristic script detector for Bengali, Devanagari (Hindi/Marathi),
    Tamil, Telugu, or Latin (English/other).
    Does not require heavy dependencies.
    """
    if not text:
        return "unknown"
    
    # Check Unicode ranges
    has_bengali = bool(re.search(r'[\u0980-\u09FF]', text))
    if has_bengali:
        return "bn"
    
    has_devanagari = bool(re.search(r'[\u0900-\u097F]', text))
    if has_devanagari:
        return "hi"
    
    has_tamil = bool(re.search(r'[\u0B80-\u0BFF]', text))
    if has_tamil:
        return "ta"

    has_telugu = bool(re.search(r'[\u0C00-\u0C7F]', text))
    if has_telugu:
        return "te"

    has_latin = bool(re.search(r'[A-Za-z]', text))
    if has_latin:
        return "en"

    return "unknown"
