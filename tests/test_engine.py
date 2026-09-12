"""
test_engine.py
--------------
Comprehensive unit test suite for the FitSense engine across all modules:
  - Phase 1: Data pipeline schema, distributions, anthropometrics, return dynamics
  - Phase 2: NLP attribute-level sentiment, negations, edge cases, SKU caching
  - Phase 3: Personalized fit recommendations, size shift logic, confidence scoring
  - Phase 4: Business reverse logistics model, A/B simulation, Z-test hypothesis testing
"""

import math
import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Ensure fit_engine root is on sys.path
ENGINE_ROOT = Path(__file__).resolve().parent.parent
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from src.data_pipeline import (
    generate_synthetic_fit_dataset,
    load_or_generate_dataset,
    determine_ideal_size,
    SIZES,
    CATEGORIES,
    BODY_TYPES,
    RETURN_REASONS,
    CATALOG_ITEMS,
)
from src.nlp_engine import (
    extract_attribute_sentiment,
    calculate_sample_confidence,
    compute_sku_profiles,
    cache_sku_profiles,
    get_cached_sku_profile,
    DB_PATH,
)
from src.recommendation import (
    FitRecommendationEngine,
    RecommendationResult,
    SIZE_MAP,
    SIZE_TO_IDX,
)
from src.metrics import (
    compute_reverse_logistics_impact,
    two_sample_z_test,
    simulate_ab_test,
    BusinessImpactSummary,
)


# ==============================================================================
# PHASE 1 TESTS: DATA PIPELINE & SYNTHETIC DATA GENERATOR
# ==============================================================================
class TestPhase1DataPipeline:
    """Test suite for Phase 1 data generation and schema integrity."""

    def test_schema_and_column_integrity(self):
        """Verify all mandatory fields are present and correctly typed."""
        df = generate_synthetic_fit_dataset(n_samples=500, seed=123)

        expected_cols = [
            "review_id", "item_id", "category", "brand", "user_id",
            "size_ordered", "user_height_cm", "user_weight_kg",
            "user_body_type", "fit_feedback", "review_text",
            "rating", "returned", "return_reason",
        ]

        for col in expected_cols:
            assert col in df.columns, f"Missing required column: {col}"

        assert df[expected_cols].isnull().sum().sum() == 0, "Found unexpected null values in dataset"
        assert len(df) == 500

    def test_categorical_validations(self):
        """Check that categorical attributes match expected domain values."""
        df = generate_synthetic_fit_dataset(n_samples=300, seed=456)

        assert set(df["category"].unique()).issubset(set(CATEGORIES))
        assert set(df["size_ordered"].unique()).issubset(set(SIZES))
        assert set(df["user_body_type"].unique()).issubset(set(BODY_TYPES))
        assert set(df["fit_feedback"].unique()).issubset({"Small", "Fit", "Large"})
        assert set(df["return_reason"].unique()).issubset(set(RETURN_REASONS))

    def test_anthropometric_and_rating_bounds(self):
        """Verify heights, weights, and ratings fall within realistic bounds."""
        df = generate_synthetic_fit_dataset(n_samples=400, seed=789)

        assert df["user_height_cm"].between(140, 200).all()
        assert df["user_weight_kg"].between(35, 120).all()
        assert df["rating"].between(1, 5).all()
        assert df["returned"].isin([0, 1]).all()

    def test_return_reason_consistency(self):
        """Check that unreturned items have return_reason 'None' and returned items have valid causes."""
        df = generate_synthetic_fit_dataset(n_samples=500, seed=999)

        unreturned = df[df["returned"] == 0]
        assert (unreturned["return_reason"] == "None").all()

        returned = df[df["returned"] == 1]
        assert (returned["return_reason"].isin(["Fit", "Quality", "Not as pictured"])).all()

    def test_fit_return_correlation_dynamics(self):
        """Confirm that fit dissatisfaction directly drives high return rates."""
        df = generate_synthetic_fit_dataset(n_samples=1000, seed=101)

        fit_return_rate = df[df["fit_feedback"] == "Fit"]["returned"].mean()
        misfit_return_rate = df[df["fit_feedback"].isin(["Small", "Large"])]["returned"].mean()

        assert fit_return_rate < 0.20, f"Fit return rate unexpectedly high: {fit_return_rate}"
        assert misfit_return_rate > 0.50, f"Misfit return rate unexpectedly low: {misfit_return_rate}"

    def test_ideal_size_heuristics(self):
        """Verify baseline size logic adjusts predictably with BMI."""
        assert determine_ideal_size(165, 45, "Petite") == "XS"
        assert determine_ideal_size(165, 55, "Regular") == "S"
        assert determine_ideal_size(165, 65, "Regular") == "M"
        assert determine_ideal_size(165, 76, "Regular") == "L"
        assert determine_ideal_size(165, 90, "Regular") == "XL"


# ==============================================================================
# PHASE 2 TESTS: NLP SENTIMENT & ATTRIBUTE EXTRACTION ENGINE
# ==============================================================================
class TestPhase2NLPEngine:
    """Test suite for Phase 2 NLP extraction, polarity parsing, and caching."""

    def test_attribute_polarity_tight_waist(self):
        """Negative polarity for tight waist."""
        text = "This dress is beautiful, but it was way too tight around waist and constricting."
        res = extract_attribute_sentiment(text)
        assert res["waist"] is not None
        assert res["waist"] < -0.4, f"Expected negative waist score, got {res['waist']}"

    def test_attribute_polarity_loose_bust(self):
        """Positive polarity for loose / large chest."""
        text = "Great material, however the chest and bust area was loose and gaping on me."
        res = extract_attribute_sentiment(text)
        assert res["bust_chest"] is not None
        assert res["bust_chest"] > 0.4, f"Expected positive bust score, got {res['bust_chest']}"

    def test_attribute_polarity_length_tall_petite(self):
        """Directional score for length."""
        text_short = "Way too short in length for anyone tall."
        res_short = extract_attribute_sentiment(text_short)
        assert res_short["length"] is not None
        assert res_short["length"] < -0.4

        text_long = "The hem drags on the floor, much too long for my height."
        res_long = extract_attribute_sentiment(text_long)
        assert res_long["length"] is not None
        assert res_long["length"] > 0.4

    def test_fabric_stretch_detection(self):
        """Detect rigid vs stretchy fabric."""
        stretchy_text = "Love this denim, super stretchy and forgiving fabric."
        res_stretchy = extract_attribute_sentiment(stretchy_text)
        assert res_stretchy["fabric_stretch"] is not None
        assert res_stretchy["fabric_stretch"] > 0.5

        rigid_text = "Zero stretch in this fabric, very stiff and rigid material."
        res_rigid = extract_attribute_sentiment(rigid_text)
        assert res_rigid["fabric_stretch"] is not None
        assert res_rigid["fabric_stretch"] < -0.5

    def test_negation_handling(self):
        """Negations like 'not too tight' should not register as tight."""
        text = "Comfortable fit, not too tight around waist."
        res = extract_attribute_sentiment(text)
        assert res["waist"] is not None
        assert res["waist"] >= 0.0, f"Negated tight should not be negative: {res['waist']}"

    def test_sample_confidence_formula(self):
        """Verify log10 sample confidence scaling."""
        assert calculate_sample_confidence(0) == 0.0
        assert round(calculate_sample_confidence(9), 2) == 0.50
        assert round(calculate_sample_confidence(99), 2) == 1.00
        assert calculate_sample_confidence(1000) == 1.00  # Capped at 1.0

    def test_sku_profile_aggregation_and_cache(self, tmp_path):
        """Verify aggregating dataframe into SKU profiles and caching to DB."""
        sample_df = generate_synthetic_fit_dataset(n_samples=250, seed=42)
        profiles = compute_sku_profiles(sample_df)

        assert not profiles.empty
        assert "fit_tendency" in profiles.columns
        assert "confidence_score" in profiles.columns
        assert "waist_fit" in profiles.columns

        # Test caching
        test_db = tmp_path / "test_fit.db"
        cache_sku_profiles(profiles, db_path=test_db)
        assert test_db.exists()

        sample_item = profiles["item_id"].iloc[0]
        cached_item = get_cached_sku_profile(sample_item, db_path=test_db)
        assert cached_item is not None
        assert cached_item["item_id"] == sample_item


# ==============================================================================
# PHASE 3 TESTS: RECOMMENDATION ALGORITHM & CONFIDENCE SCORING
# ==============================================================================
class TestPhase3RecommendationEngine:
    """Test suite for personalized sizing logic, bounds, and reasoning."""

    @pytest.fixture
    def rec_engine(self):
        df = generate_synthetic_fit_dataset(n_samples=500, seed=77)
        return FitRecommendationEngine(historical_df=df)

    def test_safe_bounded_recommendations(self, rec_engine):
        """Ensure all outputs are valid sizes and confidence scores are within 0-100%."""
        test_cases = [
            ("DRS-101", 145, 40, "Petite", "Snug"),
            ("DRS-103", 188, 105, "Athletic", "Relaxed"),
            ("JNS-201", 162, 58, "Hourglass", "Regular"),
            ("TOP-302", 170, 70, "Regular", "Relaxed"),
        ]

        for item_id, h, w, bt, pref in test_cases:
            res = rec_engine.recommend(
                item_id=item_id,
                user_height_cm=h,
                user_weight_kg=w,
                user_body_type=bt,
                user_fit_preference=pref,
            )
            assert isinstance(res, RecommendationResult)
            assert res.recommended_size in SIZES
            assert 0.0 <= res.confidence_score <= 100.0
            assert len(res.reasoning) > 20
            assert isinstance(res.attribute_breakdown, dict)

    def test_preference_adjustment_relaxed_vs_snug(self, rec_engine):
        """Relaxed preference should recommend >= size than snug preference."""
        res_relaxed = rec_engine.recommend(
            item_id="DRS-102",
            user_height_cm=165,
            user_weight_kg=62,
            user_body_type="Regular",
            user_fit_preference="Relaxed",
        )
        res_snug = rec_engine.recommend(
            item_id="DRS-102",
            user_height_cm=165,
            user_weight_kg=62,
            user_body_type="Regular",
            user_fit_preference="Snug",
        )

        idx_relaxed = SIZE_TO_IDX[res_relaxed.recommended_size]
        idx_snug = SIZE_TO_IDX[res_snug.recommended_size]
        assert idx_relaxed >= idx_snug, "Relaxed size should be >= snug size"

    def test_runs_small_sku_triggers_size_up(self, rec_engine):
        """SKU with runs-small bias (e.g. DRS-103) should trigger size-up or alert."""
        res = rec_engine.recommend(
            item_id="DRS-103",
            user_height_cm=165,
            user_weight_kg=56,
            user_body_type="Regular",
            user_fit_preference="Regular",
        )
        assert res.fit_tendency_label == "Runs Small"
        assert SIZE_TO_IDX[res.recommended_size] >= SIZE_TO_IDX[res.base_size]

    def test_extreme_anthropometrics_edge_cases(self, rec_engine):
        """Extreme values must never throw errors or produce out-of-bounds sizes."""
        # Very low weight
        res_tiny = rec_engine.recommend("DRS-101", 130, 30, "Petite", "Regular")
        assert res_tiny.recommended_size in SIZES

        # Very high weight
        res_plus = rec_engine.recommend("DRS-101", 195, 140, "Regular", "Regular")
        assert res_plus.recommended_size in SIZES


# ==============================================================================
# PHASE 4 TESTS: BUSINESS METRICS & A/B TEST SIMULATOR
# ==============================================================================
class TestPhase4MetricsAndSimulation:
    """Test suite for reverse logistics unit economics and A/B test statistics."""

    def test_reverse_logistics_impact_math(self):
        """Verify math integrity for returns saved, logistics savings, and GMV."""
        impact = compute_reverse_logistics_impact(
            monthly_orders=10000,
            avg_order_value_inr=2000.0,
            baseline_return_rate=0.30,
            projected_return_rate=0.20,
            fitsense_adoption_rate=1.0,  # 100% adoption
            cost_per_return_inr=120.0,
        )

        # Baseline returns: 10,000 * 0.30 = 3,000
        # Projected returns: 10,000 * 0.20 = 2,000
        # Returns saved: 1,000
        assert impact.monthly_returns_saved == 1000
        assert impact.annual_returns_saved == 12000
        assert impact.monthly_logistics_savings_inr == 120000.0  # 1,000 * 120
        assert impact.annual_logistics_savings_inr == 1440000.0
        assert impact.monthly_gmv_preserved_inr == 2000000.0  # 1,000 * 2,000

    def test_two_sample_z_test_known_values(self):
        """Verify two-sample Z-test calculation against known statistical values."""
        # Significant difference
        # n1 = 1000, x1 = 100 (10%)
        # n2 = 1000, x2 = 150 (15%)
        z, p, (ci_low, ci_high) = two_sample_z_test(100, 1000, 150, 1000)

        assert z > 3.0, f"Expected z > 3.0, got {z}"
        assert p < 0.005, f"Expected p < 0.005, got {p}"
        assert ci_low > 0.0, f"Expected CI lower bound > 0, got {ci_low}"

    def test_ab_simulation_30_days_structure(self):
        """Verify 30-day simulated timeseries shape and columns."""
        df = simulate_ab_test(n_days=30, daily_traffic_per_variant=1000, seed=42)

        assert len(df) == 30
        assert "cum_ctrl_visitors" in df.columns
        assert "cum_var_visitors" in df.columns
        assert "p_val_return_rate" in df.columns
        assert "significant_95" in df.columns

        # Cumulative visitors must strictly increase
        assert df["cum_ctrl_visitors"].is_monotonic_increasing
        assert df["cum_var_visitors"].is_monotonic_increasing

        # Final day should have higher variant conversion and lower variant return rate
        final_day = df.iloc[-1]
        assert final_day["cum_var_cvr"] > final_day["cum_ctrl_cvr"]
        assert final_day["cum_var_return_rate"] < final_day["cum_ctrl_return_rate"]
