"""
data_pipeline.py
----------------
Data loader and synthetic generator modeled after Kaggle ModCloth/RentTheRunway
fashion e-commerce fit datasets. Generates statistically sound distributions
reflecting real-world apparel fit feedback, anthropometric attributes, returns,
and customer review sentiment.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# Define base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Constants
SIZES = ["XS", "S", "M", "L", "XL"]
BODY_TYPES = ["Hourglass", "Pear", "Petite", "Athletic", "Regular"]
CATEGORIES = ["Dress", "Jeans", "Top", "Ethnic"]
RETURN_REASONS = ["Fit", "Quality", "Not as pictured", "None"]

# Catalog catalog definition (50 SKUs across leading e-commerce brands)
CATALOG_ITEMS = [
    # Dresses
    {"item_id": "DRS-101", "name": "A-Line Floral Midi Dress", "category": "Dress", "brand": "Mango", "base_price": 2490, "fit_bias": -0.4, "waist_bias": -0.3, "bust_bias": -0.4, "length_bias": 0.2, "stretch_level": 0.2},
    {"item_id": "DRS-102", "name": "Wrap Style Cocktail Dress", "category": "Dress", "brand": "Dressberry", "base_price": 1799, "fit_bias": 0.0, "waist_bias": 0.1, "bust_bias": 0.0, "length_bias": 0.0, "stretch_level": 0.6},
    {"item_id": "DRS-103", "name": "Bodycon Ribbed Knit Dress", "category": "Dress", "brand": "Roadster", "base_price": 1299, "fit_bias": -0.5, "waist_bias": -0.5, "bust_bias": -0.4, "length_bias": -0.1, "stretch_level": 0.8},
    {"item_id": "DRS-104", "name": "Tiered Maxi Sundress", "category": "Dress", "brand": "Taavi", "base_price": 1999, "fit_bias": 0.3, "waist_bias": 0.3, "bust_bias": 0.1, "length_bias": 0.5, "stretch_level": 0.1},
    {"item_id": "DRS-105", "name": "Sheath Workwear Dress", "category": "Dress", "brand": "Mango", "base_price": 2999, "fit_bias": -0.4, "waist_bias": -0.4, "bust_bias": -0.3, "length_bias": 0.0, "stretch_level": 0.2},
    {"item_id": "DRS-106", "name": "Fit & Flare Skater Dress", "category": "Dress", "brand": "Dressberry", "base_price": 1499, "fit_bias": 0.0, "waist_bias": 0.0, "bust_bias": 0.1, "length_bias": -0.1, "stretch_level": 0.5},
    {"item_id": "DRS-107", "name": "Pleated Shirt Dress", "category": "Dress", "brand": "Mast & Harbour", "base_price": 2199, "fit_bias": 0.2, "waist_bias": 0.2, "bust_bias": 0.2, "length_bias": 0.2, "stretch_level": 0.1},
    {"item_id": "DRS-108", "name": "Ruched Velvet Evening Gown", "category": "Dress", "brand": "Mango", "base_price": 3999, "fit_bias": -0.3, "waist_bias": -0.3, "bust_bias": -0.2, "length_bias": 0.4, "stretch_level": 0.5},
    {"item_id": "DRS-109", "name": "Off-Shoulder Smocked Dress", "category": "Dress", "brand": "Roadster", "base_price": 1599, "fit_bias": 0.1, "waist_bias": 0.1, "bust_bias": 0.3, "length_bias": 0.0, "stretch_level": 0.7},
    {"item_id": "DRS-110", "name": "Sleeveless Shift Mini Dress", "category": "Dress", "brand": "Dressberry", "base_price": 1399, "fit_bias": -0.2, "waist_bias": -0.1, "bust_bias": -0.2, "length_bias": -0.4, "stretch_level": 0.2},
    {"item_id": "DRS-111", "name": "Embroidered Georgette Gown", "category": "Dress", "brand": "Anouk", "base_price": 3299, "fit_bias": 0.1, "waist_bias": 0.0, "bust_bias": 0.1, "length_bias": 0.5, "stretch_level": 0.1},
    {"item_id": "DRS-112", "name": "Printed Slip Dress", "category": "Dress", "brand": "Mast & Harbour", "base_price": 1699, "fit_bias": -0.2, "waist_bias": -0.2, "bust_bias": -0.3, "length_bias": 0.1, "stretch_level": 0.2},

    # Jeans & Bottoms
    {"item_id": "JNS-201", "name": "High-Rise Super Skinny Jeans", "category": "Jeans", "brand": "Roadster", "base_price": 1899, "fit_bias": -0.6, "waist_bias": -0.6, "bust_bias": 0.0, "length_bias": 0.1, "stretch_level": 0.4},
    {"item_id": "JNS-202", "name": "Wide-Leg Ankle Length Denim", "category": "Jeans", "brand": "Mango", "base_price": 3490, "fit_bias": 0.2, "waist_bias": -0.2, "bust_bias": 0.0, "length_bias": 0.4, "stretch_level": 0.1},
    {"item_id": "JNS-203", "name": "Mid-Rise Distressed Boyfriend Jeans", "category": "Jeans", "brand": "Roadster", "base_price": 2099, "fit_bias": 0.4, "waist_bias": 0.3, "bust_bias": 0.0, "length_bias": 0.0, "stretch_level": 0.2},
    {"item_id": "JNS-204", "name": "Straight Leg Power Stretch Denim", "category": "Jeans", "brand": "HRX", "base_price": 2299, "fit_bias": 0.0, "waist_bias": 0.0, "bust_bias": 0.0, "length_bias": 0.0, "stretch_level": 0.9},
    {"item_id": "JNS-205", "name": "Mom Fit Tapered Washed Jeans", "category": "Jeans", "brand": "Dressberry", "base_price": 1799, "fit_bias": -0.2, "waist_bias": -0.4, "bust_bias": 0.0, "length_bias": -0.2, "stretch_level": 0.3},
    {"item_id": "JNS-206", "name": "Bootcut High-Waist Trousers", "category": "Jeans", "brand": "Mast & Harbour", "base_price": 2399, "fit_bias": 0.1, "waist_bias": -0.1, "bust_bias": 0.0, "length_bias": 0.3, "stretch_level": 0.5},
    {"item_id": "JNS-207", "name": "Raw-Hem Cropped Flare Jeans", "category": "Jeans", "brand": "Mango", "base_price": 3190, "fit_bias": -0.3, "waist_bias": -0.3, "bust_bias": 0.0, "length_bias": -0.3, "stretch_level": 0.2},
    {"item_id": "JNS-208", "name": "Relaxed Cargo Jogger Pants", "category": "Jeans", "brand": "Wrogn", "base_price": 2599, "fit_bias": 0.3, "waist_bias": 0.2, "bust_bias": 0.0, "length_bias": 0.1, "stretch_level": 0.6},
    {"item_id": "JNS-209", "name": "Clean Look Slim Fit Chinos", "category": "Jeans", "brand": "Mast & Harbour", "base_price": 1899, "fit_bias": -0.1, "waist_bias": -0.2, "bust_bias": 0.0, "length_bias": 0.0, "stretch_level": 0.4},
    {"item_id": "JNS-210", "name": "Super Comfy Treggings", "category": "Jeans", "brand": "Dressberry", "base_price": 1199, "fit_bias": 0.1, "waist_bias": 0.2, "bust_bias": 0.0, "length_bias": 0.0, "stretch_level": 0.95},
    {"item_id": "JNS-211", "name": "Paperbag Waist Denim Shorts", "category": "Jeans", "brand": "Roadster", "base_price": 1299, "fit_bias": 0.2, "waist_bias": 0.3, "bust_bias": 0.0, "length_bias": -0.5, "stretch_level": 0.3},
    {"item_id": "JNS-212", "name": "Flared Split-Hem Casual Jeans", "category": "Jeans", "brand": "Mango", "base_price": 3590, "fit_bias": -0.2, "waist_bias": -0.2, "bust_bias": 0.0, "length_bias": 0.5, "stretch_level": 0.3},

    # Tops & Shirts
    {"item_id": "TOP-301", "name": "Ribbed Fitted Crewneck Top", "category": "Top", "brand": "Roadster", "base_price": 799, "fit_bias": -0.4, "waist_bias": -0.2, "bust_bias": -0.5, "length_bias": -0.1, "stretch_level": 0.7},
    {"item_id": "TOP-302", "name": "Oversized Graphic Cotton Tee", "category": "Top", "brand": "Wrogn", "base_price": 1199, "fit_bias": 0.6, "waist_bias": 0.5, "bust_bias": 0.5, "length_bias": 0.3, "stretch_level": 0.3},
    {"item_id": "TOP-303", "name": "Satin Peplum Blouse", "category": "Top", "brand": "Dressberry", "base_price": 1499, "fit_bias": -0.3, "waist_bias": -0.3, "bust_bias": -0.4, "length_bias": 0.0, "stretch_level": 0.1},
    {"item_id": "TOP-304", "name": "Active Dry-Fit Training Tank", "category": "Top", "brand": "HRX", "base_price": 999, "fit_bias": 0.0, "waist_bias": 0.0, "bust_bias": -0.1, "length_bias": 0.0, "stretch_level": 0.85},
    {"item_id": "TOP-305", "name": "Puff Sleeve Sweetheart Top", "category": "Top", "brand": "Mango", "base_price": 2290, "fit_bias": -0.3, "waist_bias": -0.2, "bust_bias": -0.5, "length_bias": -0.2, "stretch_level": 0.2},
    {"item_id": "TOP-306", "name": "Linen Blend Button-Down Shirt", "category": "Top", "brand": "Mast & Harbour", "base_price": 1899, "fit_bias": 0.2, "waist_bias": 0.2, "bust_bias": 0.1, "length_bias": 0.2, "stretch_level": 0.05},
    {"item_id": "TOP-307", "name": "Seamless Crop Workout Top", "category": "Top", "brand": "HRX", "base_price": 1299, "fit_bias": -0.5, "waist_bias": -0.2, "bust_bias": -0.6, "length_bias": -0.4, "stretch_level": 0.75},
    {"item_id": "TOP-308", "name": "Floral Chiffon Ruffle Top", "category": "Top", "brand": "Dressberry", "base_price": 1299, "fit_bias": 0.1, "waist_bias": 0.1, "bust_bias": 0.1, "length_bias": 0.0, "stretch_level": 0.15},
    {"item_id": "TOP-309", "name": "Classic Oxford Tailored Shirt", "category": "Top", "brand": "Mango", "base_price": 2790, "fit_bias": -0.2, "waist_bias": -0.1, "bust_bias": -0.3, "length_bias": 0.1, "stretch_level": 0.1},
    {"item_id": "TOP-310", "name": "Boxy Cropped Everyday Tee", "category": "Top", "brand": "Roadster", "base_price": 699, "fit_bias": 0.3, "waist_bias": 0.4, "bust_bias": 0.3, "length_bias": -0.3, "stretch_level": 0.4},
    {"item_id": "TOP-311", "name": "Mock Neck Long Sleeve Top", "category": "Top", "brand": "Mast & Harbour", "base_price": 1199, "fit_bias": -0.2, "waist_bias": -0.1, "bust_bias": -0.3, "length_bias": 0.0, "stretch_level": 0.6},
    {"item_id": "TOP-312", "name": "Drop-Shoulder Hooded Sweatshirt", "category": "Top", "brand": "Wrogn", "base_price": 2199, "fit_bias": 0.5, "waist_bias": 0.4, "bust_bias": 0.4, "length_bias": 0.2, "stretch_level": 0.5},
    {"item_id": "TOP-313", "name": "Tie-Front Bohemian Blouse", "category": "Top", "brand": "Taavi", "base_price": 1499, "fit_bias": 0.2, "waist_bias": 0.2, "bust_bias": 0.2, "length_bias": 0.1, "stretch_level": 0.1},

    # Ethnic Wear
    {"item_id": "ETH-401", "name": "Anarkali Pure Cotton Kurta Set", "category": "Ethnic", "brand": "Libas", "base_price": 2699, "fit_bias": 0.1, "waist_bias": 0.1, "bust_bias": 0.0, "length_bias": 0.3, "stretch_level": 0.1},
    {"item_id": "ETH-402", "name": "Straight Hem Embroidered Kurti", "category": "Ethnic", "brand": "Anouk", "base_price": 1499, "fit_bias": -0.2, "waist_bias": -0.2, "bust_bias": -0.3, "length_bias": 0.1, "stretch_level": 0.05},
    {"item_id": "ETH-403", "name": "A-Line Block Print Festive Kurta", "category": "Ethnic", "brand": "W for Woman", "base_price": 2999, "fit_bias": 0.0, "waist_bias": 0.1, "bust_bias": -0.1, "length_bias": 0.2, "stretch_level": 0.1},
    {"item_id": "ETH-404", "name": "Chikankari Handloom Tunic", "category": "Ethnic", "brand": "Taavi", "base_price": 1899, "fit_bias": 0.2, "waist_bias": 0.3, "bust_bias": 0.1, "length_bias": 0.1, "stretch_level": 0.05},
    {"item_id": "ETH-405", "name": "Palazzo Pants & Flared Kurta Set", "category": "Ethnic", "brand": "Libas", "base_price": 3199, "fit_bias": 0.1, "waist_bias": 0.2, "bust_bias": 0.0, "length_bias": 0.4, "stretch_level": 0.2},
    {"item_id": "ETH-406", "name": "Angrakha Style Designer Kurta", "category": "Ethnic", "brand": "Anouk", "base_price": 2199, "fit_bias": -0.3, "waist_bias": -0.3, "bust_bias": -0.4, "length_bias": 0.1, "stretch_level": 0.1},
    {"item_id": "ETH-407", "name": "Contemporary Fusion Co-ord Set", "category": "Ethnic", "brand": "W for Woman", "base_price": 3499, "fit_bias": 0.0, "waist_bias": -0.1, "bust_bias": 0.0, "length_bias": 0.0, "stretch_level": 0.3},
    {"item_id": "ETH-408", "name": "Silk Blend Straight Fit Kurta", "category": "Ethnic", "brand": "Libas", "base_price": 1999, "fit_bias": -0.1, "waist_bias": -0.2, "bust_bias": -0.2, "length_bias": 0.2, "stretch_level": 0.05},
    {"item_id": "ETH-409", "name": "Kalamkari Printed Festive Gown", "category": "Ethnic", "brand": "Taavi", "base_price": 2899, "fit_bias": 0.2, "waist_bias": 0.2, "bust_bias": 0.1, "length_bias": 0.5, "stretch_level": 0.1},
    {"item_id": "ETH-410", "name": "Geometric Print Everyday Kurti", "category": "Ethnic", "brand": "Anouk", "base_price": 1199, "fit_bias": -0.1, "waist_bias": -0.1, "bust_bias": -0.2, "length_bias": 0.0, "stretch_level": 0.1},
    {"item_id": "ETH-411", "name": "Layered Asymmetrical Kurta", "category": "Ethnic", "brand": "W for Woman", "base_price": 2799, "fit_bias": 0.1, "waist_bias": 0.2, "bust_bias": 0.0, "length_bias": 0.2, "stretch_level": 0.15},
    {"item_id": "ETH-412", "name": "Zari Work Georgette Kurta Set", "category": "Ethnic", "brand": "Libas", "base_price": 3799, "fit_bias": 0.0, "waist_bias": 0.0, "bust_bias": -0.1, "length_bias": 0.3, "stretch_level": 0.1},
    {"item_id": "ETH-413", "name": "Ikat Woven Tunic Kurta", "category": "Ethnic", "brand": "Taavi", "base_price": 1699, "fit_bias": 0.2, "waist_bias": 0.2, "bust_bias": 0.1, "length_bias": 0.1, "stretch_level": 0.05},
]


def determine_ideal_size(height_cm: float, weight_kg: float, body_type: str) -> str:
    """Calculate the anthropometric baseline size from height, weight, and body profile."""
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m * height_m)

    # Body type adjustment factor
    bmi_adj = 0.0
    if body_type == "Hourglass":
        bmi_adj += 0.3
    elif body_type == "Pear":
        bmi_adj += 0.5
    elif body_type == "Petite":
        bmi_adj -= 0.5
    elif body_type == "Athletic":
        bmi_adj += 0.4

    effective_bmi = bmi + bmi_adj

    if effective_bmi < 19.5:
        return "XS"
    elif effective_bmi < 22.5:
        return "S"
    elif effective_bmi < 25.5:
        return "M"
    elif effective_bmi < 29.0:
        return "L"
    else:
        return "XL"


def generate_review_snippet(
    fit_feedback: str,
    item_meta: dict,
    user_body_type: str,
    user_height_cm: float,
    size_ordered: str,
) -> str:
    """Generate realistic e-commerce customer review text targeting specific garment dimensions."""
    category = item_meta["category"]
    waist_bias = item_meta.get("waist_bias", 0.0)
    bust_bias = item_meta.get("bust_bias", 0.0)
    length_bias = item_meta.get("length_bias", 0.0)
    stretch = item_meta.get("stretch_level", 0.3)

    sentences = []

    # 1. Overall impression sentence
    if fit_feedback == "Small":
        lead_options = [
            f"Ordered size {size_ordered} but this definitely runs small.",
            f"I usually wear {size_ordered}, but this was way too snug.",
            f"Beautiful piece, but sizing is off and it runs tight.",
            f"Felt very suffocating in size {size_ordered}, recommend sizing up.",
        ]
    elif fit_feedback == "Large":
        lead_options = [
            f"Ordered my usual size {size_ordered} and it was swimming on me.",
            f"Runs on the larger side, felt baggy and loose.",
            f"Had to return size {size_ordered} because it was much too big.",
            f"Definitely oversized cut, consider sizing down.",
        ]
    else:
        lead_options = [
            f"Size {size_ordered} fits like a glove! True to size.",
            f"Absolutely love the fit of size {size_ordered}.",
            f"Perfect fit right out of the package.",
            f"Matches the size chart accurately, so comfortable.",
        ]
    sentences.append(random.choice(lead_options))

    # 2. Attribute-specific mentions (Waist, Bust/Chest, Length, Hips, Fabric)
    attr_mentions = []

    # Waist & Hips
    if waist_bias < -0.2 or (fit_feedback == "Small" and random.random() < 0.6):
        attr_mentions.append(random.choice([
            "Extremely tight around waist with almost no give.",
            "The waistband dug right into my sides.",
            "Waist was noticeably smaller than standard sizing.",
            "Very constricting waistline.",
        ]))
    elif waist_bias > 0.2 or (fit_feedback == "Large" and random.random() < 0.5):
        attr_mentions.append(random.choice([
            "Gaped at the waist quite a bit.",
            "Waist is roomy and relaxed.",
            "Plenty of room around the waist.",
        ]))
    else:
        if random.random() < 0.35:
            attr_mentions.append("Fits very comfortably around the waist.")

    # Bust/Chest
    if category in ["Top", "Dress", "Ethnic"]:
        if bust_bias < -0.2 or (fit_feedback == "Small" and user_body_type == "Hourglass"):
            attr_mentions.append(random.choice([
                "Way too tight across the bust and chest.",
                "Buttons pulled open at the chest.",
                "Tight across the shoulders and chest area.",
                "Runs small at bust for anyone with curves.",
            ]))
        elif bust_bias > 0.2 or (fit_feedback == "Large" and random.random() < 0.4):
            attr_mentions.append("Loose at the bust and chest armholes.")
        elif random.random() < 0.3:
            attr_mentions.append("Chest and shoulder seams align nicely.")

    # Length
    if length_bias > 0.2 or user_height_cm < 156:
        attr_mentions.append(random.choice([
            "Too long for my height, dragged on the floor without heels.",
            "Length is quite long, might need tailoring.",
            "A bit longer than pictured on the model.",
        ]))
    elif length_bias < -0.2 or user_height_cm > 174:
        attr_mentions.append(random.choice([
            "A little too short on me.",
            "Cropped shorter than expected for tall frames.",
            "Length was somewhat brief.",
        ]))
    elif random.random() < 0.35:
        attr_mentions.append("Great length, falls just right.")

    # Fabric / Stretch
    if stretch >= 0.7:
        attr_mentions.append(random.choice([
            "The fabric has amazing stretch and recovery.",
            "Super stretchy and forgiving material.",
            "Love the comfortable stretch in this fabric.",
        ]))
    elif stretch <= 0.15:
        attr_mentions.append(random.choice([
            "Fabric has zero stretch and is quite stiff.",
            "Rigid material with no give whatsoever.",
            "Watch out after washing, might shrink slightly.",
        ]))

    # Pick 1 or 2 attribute mentions
    if attr_mentions:
        chosen_mentions = random.sample(attr_mentions, k=min(2, len(attr_mentions)))
        sentences.extend(chosen_mentions)

    return " ".join(sentences)


def generate_synthetic_fit_dataset(
    n_samples: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a statistically sound synthetic dataset of 10,000 fashion transactions.
    
    Fields:
      `review_id`, `item_id`, `category`, `brand`, `user_id`, `size_ordered`,
      `user_height_cm`, `user_weight_kg`, `user_body_type`, `fit_feedback`,
      `review_text`, `rating`, `returned`, `return_reason`
    """
    random.seed(seed)
    np.random.seed(seed)

    size_map = {0: "XS", 1: "S", 2: "M", 3: "L", 4: "XL"}
    size_to_idx = {v: k for k, v in size_map.items()}

    records = []

    # SKU item lookup
    sku_lookup = {item["item_id"]: item for item in CATALOG_ITEMS}
    item_ids = list(sku_lookup.keys())

    for idx in range(1, n_samples + 1):
        review_id = f"REV-{idx:06d}"
        user_id = f"USR-{random.randint(1000, 9999)}"

        # Pick item
        item_id = random.choice(item_ids)
        item_meta = sku_lookup[item_id]
        category = item_meta["category"]
        brand = item_meta["brand"]

        # Anthropometrics
        # Heights: Normal distribution centered at 162.5 cm with std 6.5 cm
        height_cm = float(np.clip(np.random.normal(162.5, 6.8), 145.0, 188.0))

        # Weight: Correlated with height (BMI centered at 23.0, std 3.8)
        sim_bmi = float(np.clip(np.random.normal(23.2, 3.8), 16.5, 38.0))
        weight_kg = float(np.clip(sim_bmi * ((height_cm / 100.0) ** 2), 40.0, 102.0))

        # Body type distribution
        if height_cm < 155:
            body_weights = [0.15, 0.20, 0.40, 0.10, 0.15]  # Petite elevated
        else:
            body_weights = [0.22, 0.25, 0.08, 0.20, 0.25]
        body_type = random.choices(BODY_TYPES, weights=body_weights, k=1)[0]

        # Ideal anthropometric size
        ideal_size = determine_ideal_size(height_cm, weight_kg, body_type)
        ideal_idx = size_to_idx[ideal_size]

        # Brand / SKU fit bias
        sku_fit_bias = item_meta.get("fit_bias", 0.0)  # negative means runs small

        # Customer ordering behavior:
        # 66% order their ideal size, 17% order 1 size smaller, 17% order 1 size larger
        order_shift = random.choices([-1, 0, 1], weights=[0.17, 0.66, 0.17], k=1)[0]
        ordered_idx = int(np.clip(ideal_idx + order_shift, 0, 4))
        size_ordered = size_map[ordered_idx]

        # Net fit deviation:
        # If user ordered smaller than ideal -> positive size deficit (item feels Small)
        # If item has negative fit_bias (runs small) -> increases chance of feeling Small
        net_mismatch = (ideal_idx - ordered_idx) - (sku_fit_bias * 1.5)

        # Fit feedback probabilistic determination
        if net_mismatch >= 0.75:
            fit_feedback = random.choices(["Small", "Fit", "Large"], weights=[0.82, 0.15, 0.03], k=1)[0]
        elif net_mismatch <= -0.75:
            fit_feedback = random.choices(["Small", "Fit", "Large"], weights=[0.03, 0.15, 0.82], k=1)[0]
        else:
            fit_feedback = random.choices(["Small", "Fit", "Large"], weights=[0.12, 0.76, 0.12], k=1)[0]

        # Return status and reason modeling
        if fit_feedback == "Small":
            returned = 1 if random.random() < 0.74 else 0
            if returned:
                return_reason = random.choices(
                    ["Fit", "Quality", "Not as pictured"],
                    weights=[0.88, 0.08, 0.04],
                    k=1
                )[0]
            else:
                return_reason = "None"
            rating = random.choices([1, 2, 3, 4, 5], weights=[0.35, 0.30, 0.20, 0.10, 0.05], k=1)[0]

        elif fit_feedback == "Large":
            returned = 1 if random.random() < 0.72 else 0
            if returned:
                return_reason = random.choices(
                    ["Fit", "Quality", "Not as pictured"],
                    weights=[0.86, 0.09, 0.05],
                    k=1
                )[0]
            else:
                return_reason = "None"
            rating = random.choices([1, 2, 3, 4, 5], weights=[0.30, 0.32, 0.22, 0.11, 0.05], k=1)[0]

        else:  # "Fit"
            # Fits well -> very low return rate (~8.5%)
            returned = 1 if random.random() < 0.085 else 0
            if returned:
                return_reason = random.choices(
                    ["Quality", "Not as pictured", "Fit"],
                    weights=[0.55, 0.35, 0.10],
                    k=1
                )[0]
            else:
                return_reason = "None"
            rating = random.choices([1, 2, 3, 4, 5], weights=[0.02, 0.03, 0.10, 0.40, 0.45], k=1)[0]

        # Review text generation
        review_text = generate_review_snippet(
            fit_feedback=fit_feedback,
            item_meta=item_meta,
            user_body_type=body_type,
            user_height_cm=height_cm,
            size_ordered=size_ordered,
        )

        records.append({
            "review_id": review_id,
            "item_id": item_id,
            "category": category,
            "brand": brand,
            "user_id": user_id,
            "size_ordered": size_ordered,
            "user_height_cm": round(height_cm, 1),
            "user_weight_kg": round(weight_kg, 1),
            "user_body_type": body_type,
            "fit_feedback": fit_feedback,
            "review_text": review_text,
            "rating": int(rating),
            "returned": int(returned),
            "return_reason": return_reason,
        })

    df = pd.DataFrame(records)
    return df


def load_or_generate_dataset(
    force_regenerate: bool = False,
    n_samples: int = 10000,
) -> pd.DataFrame:
    """
    Load clean dataset from processed parquet/csv file, or auto-generate if missing.
    Saves to `data/processed/clean_fit_data.parquet` and `data/processed/clean_fit_data.csv`.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    parquet_path = PROCESSED_DATA_DIR / "clean_fit_data.parquet"
    csv_path = PROCESSED_DATA_DIR / "clean_fit_data.csv"

    if not force_regenerate and parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass

    if not force_regenerate and csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            pass

    # Generate synthetic dataset
    print(f"Generating statistically sound {n_samples}-row fashion fit dataset...")
    df = generate_synthetic_fit_dataset(n_samples=n_samples)

    # Save to both formats for universal accessibility
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"Saved processed dataset to {parquet_path}")
    except Exception as e:
        print(f"Warning: could not save parquet: {e}")

    df.to_csv(csv_path, index=False)
    print(f"Saved processed dataset to {csv_path}")

    return df


if __name__ == "__main__":
    df = load_or_generate_dataset(force_regenerate=True)
    print(f"Dataset generated with shape: {df.shape}")
    print(df.head())
    print("\nFit feedback distribution:\n", df["fit_feedback"].value_counts(normalize=True))
    print("\nReturn reasons distribution:\n", df["return_reason"].value_counts())
    print("\nReturn rate by fit feedback:\n", df.groupby("fit_feedback")["returned"].mean())
