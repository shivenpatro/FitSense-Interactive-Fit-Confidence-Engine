"""
nlp_engine.py
-------------
Garment attribute mining and sentiment analysis engine targeting 5 core attributes:
`waist`, `bust_chest`, `length`, `hips`, and `fabric_stretch`.

Extracts directional fit skews (-1.0 Runs Small to +1.0 Runs Large) and aggregates
product/SKU-level fit intelligence with review-volume confidence scoring.
Includes fast caching via SQLite and DuckDB.
"""

from __future__ import annotations

import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional DuckDB support
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DB_PATH = PROCESSED_DATA_DIR / "fit_profiles.db"

# Attribute target categories and synonyms
ATTRIBUTE_KEYWORDS: Dict[str, List[str]] = {
    "waist": ["waist", "waistline", "waistband", "midsection", "tummy", "belly", "stomach"],
    "bust_chest": ["bust", "chest", "breast", "shoulders", "bra", "decolletage", "armhole", "cup"],
    "length": ["length", "long", "short", "tall", "petite", "hem", "crop", "cropped", "ankle", "maxi", "midi", "floor"],
    "hips": ["hips", "hip", "thigh", "thighs", "butt", "booty", "rear", "glutes", "seat"],
    "fabric_stretch": ["stretch", "stretchy", "give", "elastic", "spandex", "elastane", "stiff", "rigid", "tightness", "shrink", "wash"],
}

# Directional tokens
RUNS_SMALL_TOKENS = {
    "tight", "small", "snug", "constricting", "suffocating", "choking",
    "dug", "pinching", "tightness", "narrow", "squeezing", "hard to zip",
    "too tight", "runs small", "way small", "very small", "too short", "rides up"
}

RUNS_LARGE_TOKENS = {
    "loose", "large", "big", "baggy", "oversized", "swimming", "gaping",
    "roomy", "falling down", "slouchy", "wide", "too long", "too loose",
    "too big", "runs large", "way big", "drags on floor", "very loose"
}

TRUE_TO_SIZE_TOKENS = {
    "perfect", "flattering", "true to size", "like a glove", "spot on",
    "fits well", "fits nicely", "just right", "accurate", "great fit", "exact"
}

STRETCH_POSITIVE_TOKENS = {
    "stretchy", "stretch", "give", "elastic", "flexible", "forgiving", "super stretchy"
}

STRETCH_NEGATIVE_TOKENS = {
    "stiff", "rigid", "zero stretch", "no stretch", "no give", "inflexible", "shrank", "shrink"
}


def _clean_text(text: str) -> str:
    """Normalize text into lowercase without extra punctuation."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s\-\']", " ", text)
    return " ".join(text.split())


def extract_attribute_sentiment(text: str) -> Dict[str, Optional[float]]:
    """
    Extract directional sentiment for each garment attribute from review text.
    
    Returns:
      Dict mapping attribute name to a float score:
        - For fit dimensions (waist, bust_chest, length, hips):
          - -1.0 to -0.2: Runs small / tight / constricting
          - -0.2 to +0.2: True to size / fits well
          - +0.2 to +1.0: Runs large / loose / long
          - None: Attribute not mentioned in review
        - For fabric_stretch:
          - -1.0 to -0.2: Rigid, stiff, zero give
          - +0.2 to +1.0: High stretch, elastic, forgiving
          - None: Fabric/stretch not mentioned
    """
    cleaned = _clean_text(text)
    clauses = re.split(r"[,;.!?\n]|(\bbut\b)|(\bhowever\b)|(\band\b)", text.lower())
    clauses = [c.strip() for c in clauses if c and c.strip() not in ["but", "however", "and"]]

    results: Dict[str, Optional[float]] = {
        "waist": None,
        "bust_chest": None,
        "length": None,
        "hips": None,
        "fabric_stretch": None,
    }

    for attr, keywords in ATTRIBUTE_KEYWORDS.items():
        # Check if attribute keywords exist in the text
        pattern = r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b"
        if not re.search(pattern, cleaned):
            continue

        # Score based on relevant clauses
        attr_scores = []
        for clause in clauses:
            if re.search(pattern, clause):
                score = _score_clause_for_attribute(clause, attr)
                if score is not None:
                    attr_scores.append(score)

        if attr_scores:
            results[attr] = float(np.clip(np.mean(attr_scores), -1.0, 1.0))
        else:
            # Fallback: check whole sentence if clause splitting missed context
            fallback_score = _score_clause_for_attribute(cleaned, attr)
            results[attr] = fallback_score

    return results


def _score_clause_for_attribute(clause: str, attr: str) -> Optional[float]:
    """Score a single clause targeting a specific garment attribute."""
    # Check for negations like 'not too tight', 'doesn't feel loose'
    has_negation = bool(re.search(r"\b(not|n't|never|no)\b", clause))

    if attr == "fabric_stretch":
        # Check positive stretch
        has_pos = any(t in clause for t in STRETCH_POSITIVE_TOKENS)
        has_neg = any(t in clause for t in STRETCH_NEGATIVE_TOKENS)

        if "zero stretch" in clause or "no stretch" in clause or "no give" in clause:
            return -0.85
        if has_pos and not has_negation:
            return 0.85
        if has_neg and not has_negation:
            return -0.80
        if has_negation and has_pos:
            return -0.70
        return 0.0

    # For fit attributes: waist, bust_chest, length, hips
    has_small = any(re.search(r"\b" + re.escape(t) + r"\b", clause) for t in RUNS_SMALL_TOKENS)
    has_large = any(re.search(r"\b" + re.escape(t) + r"\b", clause) for t in RUNS_LARGE_TOKENS)
    has_true = any(re.search(r"\b" + re.escape(t) + r"\b", clause) for t in TRUE_TO_SIZE_TOKENS)

    # Specific length adjustments
    if attr == "length":
        if "short" in clause and not has_negation:
            return -0.75
        if "long" in clause and not has_negation:
            return 0.75

    if has_negation:
        # e.g., 'not too tight' -> true to size (0.0)
        if has_small:
            return 0.1
        if has_large:
            return -0.1

    if has_small and not has_large:
        # Check degree words (way too tight, extremely small)
        if any(w in clause for w in ["way", "extremely", "super", "very", "much too"]):
            return -0.90
        return -0.70

    if has_large and not has_small:
        if any(w in clause for w in ["way", "extremely", "super", "very", "much too"]):
            return 0.90
        return 0.70

    if has_true:
        return 0.0

    return None


def calculate_sample_confidence(n_reviews: int) -> float:
    """
    Calculate sample confidence score based on the volume of reviews for that SKU/Brand.
    Formula: min(1.0, log10(n_reviews + 1) / 2.0)
    """
    if n_reviews <= 0:
        return 0.0
    return float(min(1.0, math.log10(n_reviews + 1.0) / 2.0))


def compute_sku_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate review-level NLP sentiment into product-level fit indices:
    - `fit_tendency`: -1.0 Runs Small to +1.0 Runs Large
    - `confidence_score`: review volume confidence
    - `waist_fit`, `chest_fit`, `length_fit`, `hips_fit`, `stretch_score`
    - `return_rate`, `total_reviews`
    """
    # Extract attribute sentiment for every review
    sentiment_records = []
    for _, row in df.iterrows():
        review_text = row.get("review_text", "")
        sentiments = extract_attribute_sentiment(review_text)
        sentiment_records.append(sentiments)

    sent_df = pd.DataFrame(sentiment_records)
    combined = pd.concat([df.reset_index(drop=True), sent_df.reset_index(drop=True)], axis=1)

    # Numerical fit feedback mapping for fit_tendency
    feedback_num = combined["fit_feedback"].map({"Small": -1.0, "Fit": 0.0, "Large": 1.0}).fillna(0.0)
    combined["fit_num"] = feedback_num

    profiles = []
    for item_id, group in combined.groupby("item_id"):
        n_reviews = len(group)
        brand = group["brand"].iloc[0]
        category = group["category"].iloc[0]

        # Calculate fit tendency (blend of feedback categorical and NLP review text)
        fit_tendency = float(group["fit_num"].mean())

        # Attribute skews
        waist_fit = float(group["waist"].dropna().mean()) if not group["waist"].dropna().empty else 0.0
        chest_fit = float(group["bust_chest"].dropna().mean()) if not group["bust_chest"].dropna().empty else 0.0
        length_fit = float(group["length"].dropna().mean()) if not group["length"].dropna().empty else 0.0
        hips_fit = float(group["hips"].dropna().mean()) if not group["hips"].dropna().empty else 0.0

        # Stretch score normalized between 0.0 (rigid) and 1.0 (very stretchy)
        raw_stretch = float(group["fabric_stretch"].dropna().mean()) if not group["fabric_stretch"].dropna().empty else 0.0
        stretch_score = float(np.clip((raw_stretch + 1.0) / 2.0, 0.0, 1.0))

        # Confidence and return rate
        confidence = calculate_sample_confidence(n_reviews)
        return_rate = float(group["returned"].mean())
        avg_rating = float(group["rating"].mean())

        # Fit feedback distribution
        counts = group["fit_feedback"].value_counts(normalize=True).to_dict()
        pct_small = round(counts.get("Small", 0.0) * 100, 1)
        pct_fit = round(counts.get("Fit", 0.0) * 100, 1)
        pct_large = round(counts.get("Large", 0.0) * 100, 1)

        profiles.append({
            "item_id": item_id,
            "brand": brand,
            "category": category,
            "total_reviews": n_reviews,
            "fit_tendency": round(fit_tendency, 3),
            "confidence_score": round(confidence, 3),
            "waist_fit": round(waist_fit, 3),
            "chest_fit": round(chest_fit, 3),
            "length_fit": round(length_fit, 3),
            "hips_fit": round(hips_fit, 3),
            "stretch_score": round(stretch_score, 3),
            "return_rate": round(return_rate, 4),
            "avg_rating": round(avg_rating, 2),
            "pct_small": pct_small,
            "pct_fit": pct_fit,
            "pct_large": pct_large,
        })

    profile_df = pd.DataFrame(profiles)
    return profile_df


def cache_sku_profiles(profiles_df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    """Save aggregated SKU fit profiles into SQLite / DuckDB cache for sub-millisecond retrieval."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Cache into SQLite
    conn = sqlite3.connect(str(db_path))
    profiles_df.to_sql("sku_fit_profiles", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    # Cache into DuckDB if available
    if HAS_DUCKDB:
        duck_path = db_path.with_suffix(".duckdb")
        duck_conn = duckdb.connect(str(duck_path))
        duck_conn.execute("CREATE OR REPLACE TABLE sku_fit_profiles AS SELECT * FROM profiles_df")
        duck_conn.close()

    print(f"Cached {len(profiles_df)} SKU profiles to {db_path}")


def get_cached_sku_profile(item_id: str, db_path: Path = DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve SKU fit profile from cache by item_id."""
    if not db_path.exists():
        return None

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM sku_fit_profiles WHERE item_id = ?", (item_id,))
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None
    return dict(row)


if __name__ == "__main__":
    from data_pipeline import load_or_generate_dataset
    df = load_or_generate_dataset()
    profiles = compute_sku_profiles(df)
    cache_sku_profiles(profiles)
    print("Sample SKU profile:")
    print(profiles.head())
