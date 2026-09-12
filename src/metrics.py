"""
metrics.py
----------
Business impact calculator, reverse logistics unit economics, and 30-day
A/B testing simulation framework for fashion e-commerce platforms.

Features:
  - Reverse logistics cost modeling (courier, QC, restocking, GMV preserved)
  - Statistical A/B test simulation with Two-sample Z-test for proportions
  - 95% Confidence Intervals & daily p-value convergence tracking
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class BusinessImpactSummary:
    """Executive KPI summary of reverse logistics and revenue metrics."""
    monthly_orders: int
    avg_order_value_inr: float
    baseline_return_rate: float
    projected_return_rate: float
    fitsense_adoption_rate: float
    effective_return_rate: float
    cost_per_return_inr: float
    monthly_returns_saved: int
    annual_returns_saved: int
    monthly_logistics_savings_inr: float
    annual_logistics_savings_inr: float
    monthly_gmv_preserved_inr: float
    annual_gmv_preserved_inr: float
    relative_return_reduction_pct: float
    margin_expansion_bps: float


def compute_reverse_logistics_impact(
    monthly_orders: int = 50000,
    avg_order_value_inr: float = 1850.0,
    baseline_return_rate: float = 0.30,
    projected_return_rate: float = 0.215,
    fitsense_adoption_rate: float = 0.65,
    cost_per_return_inr: float = 120.0,
    ebitda_margin_pct: float = 0.08,
) -> BusinessImpactSummary:
    """
    Calculate baseline vs projected supply chain reverse logistics metrics.
    
    Reverse Logistics Cost Components:
      - Reverse pickup courier charges (~₹65)
      - QC, authentication & inspection (~₹25)
      - Re-tagging, dry cleaning & re-shelving (~₹30)
      - Total cost per returned unit ~₹120
    """
    # Effective blended return rate across adopters and non-adopters
    effective_return_rate = (
        (baseline_return_rate * (1.0 - fitsense_adoption_rate)) +
        (projected_return_rate * fitsense_adoption_rate)
    )

    baseline_returns = int(monthly_orders * baseline_return_rate)
    effective_returns = int(monthly_orders * effective_return_rate)
    monthly_returns_saved = max(0, baseline_returns - effective_returns)
    annual_returns_saved = monthly_returns_saved * 12

    monthly_logistics_savings = monthly_returns_saved * cost_per_return_inr
    annual_logistics_savings = monthly_logistics_savings * 12

    monthly_gmv_preserved = monthly_returns_saved * avg_order_value_inr
    annual_gmv_preserved = monthly_gmv_preserved * 12

    relative_reduction_pct = (
        ((baseline_return_rate - effective_return_rate) / baseline_return_rate) * 100.0
    )

    # Margin expansion in basis points (1 bps = 0.01%)
    total_gmv = monthly_orders * avg_order_value_inr
    margin_expansion_bps = (monthly_logistics_savings / total_gmv) * 10000.0 if total_gmv > 0 else 0.0

    return BusinessImpactSummary(
        monthly_orders=monthly_orders,
        avg_order_value_inr=avg_order_value_inr,
        baseline_return_rate=round(baseline_return_rate, 4),
        projected_return_rate=round(projected_return_rate, 4),
        fitsense_adoption_rate=round(fitsense_adoption_rate, 4),
        effective_return_rate=round(effective_return_rate, 4),
        cost_per_return_inr=cost_per_return_inr,
        monthly_returns_saved=monthly_returns_saved,
        annual_returns_saved=annual_returns_saved,
        monthly_logistics_savings_inr=round(monthly_logistics_savings, 2),
        annual_logistics_savings_inr=round(annual_logistics_savings, 2),
        monthly_gmv_preserved_inr=round(monthly_gmv_preserved, 2),
        annual_gmv_preserved_inr=round(annual_gmv_preserved, 2),
        relative_return_reduction_pct=round(relative_reduction_pct, 2),
        margin_expansion_bps=round(margin_expansion_bps, 1),
    )


def two_sample_z_test(
    count_1: int,
    n_1: int,
    count_2: int,
    n_2: int,
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Compute two-sample Z-test for proportions with two-sided p-value and 95% Confidence Interval.
    
    Returns:
      (z_score, p_value, (ci_lower, ci_upper))
    """
    if n_1 <= 0 or n_2 <= 0:
        return 0.0, 1.0, (0.0, 0.0)

    p1 = count_1 / n_1
    p2 = count_2 / n_2
    diff = p2 - p1

    # Pooled proportion for hypothesis testing (H0: p1 == p2)
    p_pool = (count_1 + count_2) / (n_1 + n_2)
    se_pool = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n_1 + 1.0 / n_2))

    if se_pool <= 1e-12:
        z_score = 0.0
        p_val = 1.0
    else:
        z_score = diff / se_pool
        # Two-sided p-value using error function
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_score) / math.sqrt(2.0))))
        p_val = max(0.0, min(1.0, p_val))

    # Standard error for unpooled 95% Confidence Interval
    se_unpooled = math.sqrt((p1 * (1.0 - p1) / n_1) + (p2 * (1.0 - p2) / n_2))
    ci_margin = 1.95996 * se_unpooled
    ci_lower = diff - ci_margin
    ci_upper = diff + ci_margin

    return round(z_score, 4), round(p_val, 6), (round(ci_lower, 5), round(ci_upper, 5))


def simulate_ab_test(
    n_days: int = 30,
    daily_traffic_per_variant: int = 2500,
    control_cvr: float = 0.032,
    variant_cvr: float = 0.0385,
    control_return_rate: float = 0.298,
    variant_return_rate: float = 0.214,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate a 30-day live A/B experiment between Control (standard PDP)
    and Variant (FitSense interactive widget).
    
    Generates daily traffic, orders, returns, conversion rates, return rates,
    cumulative counts, two-sample Z-test statistics, and p-value convergence.
    """
    np.random.seed(seed)
    records = []

    cum_ctrl_visitors = 0
    cum_ctrl_orders = 0
    cum_ctrl_returns = 0

    cum_var_visitors = 0
    cum_var_orders = 0
    cum_var_returns = 0

    for day in range(1, n_days + 1):
        # Daily visitor noise (+/- 8%)
        traffic_noise = np.random.normal(1.0, 0.05)
        ctrl_visitors = int(daily_traffic_per_variant * traffic_noise)
        var_visitors = int(daily_traffic_per_variant * traffic_noise)

        # Conversions (Binomial process)
        ctrl_orders = int(np.random.binomial(ctrl_visitors, control_cvr))
        var_orders = int(np.random.binomial(var_visitors, variant_cvr))

        # Returns (Binomial process based on orders)
        ctrl_returns = int(np.random.binomial(ctrl_orders, control_return_rate))
        var_returns = int(np.random.binomial(var_orders, variant_return_rate))

        # Accumulate
        cum_ctrl_visitors += ctrl_visitors
        cum_ctrl_orders += ctrl_orders
        cum_ctrl_returns += ctrl_returns

        cum_var_visitors += var_visitors
        cum_var_orders += var_orders
        cum_var_returns += var_returns

        # Cumulative rates
        c_cvr = cum_ctrl_orders / cum_ctrl_visitors if cum_ctrl_visitors > 0 else 0.0
        v_cvr = cum_var_orders / cum_var_visitors if cum_var_visitors > 0 else 0.0
        cvr_uplift_pct = ((v_cvr - c_cvr) / c_cvr * 100.0) if c_cvr > 0 else 0.0

        c_ret = cum_ctrl_returns / cum_ctrl_orders if cum_ctrl_orders > 0 else 0.0
        v_ret = cum_var_returns / cum_var_orders if cum_var_orders > 0 else 0.0
        ret_reduction_pct = ((c_ret - v_ret) / c_ret * 100.0) if c_ret > 0 else 0.0

        # Z-test for conversion rate
        z_cvr, p_cvr, (ci_cvr_low, ci_cvr_high) = two_sample_z_test(
            count_1=cum_ctrl_orders,
            n_1=cum_ctrl_visitors,
            count_2=cum_var_orders,
            n_2=cum_var_visitors,
        )

        # Z-test for return rate reduction
        z_ret, p_ret, (ci_ret_low, ci_ret_high) = two_sample_z_test(
            count_1=cum_ctrl_returns,
            n_1=cum_ctrl_orders,
            count_2=cum_var_returns,
            n_2=cum_var_orders,
        )

        records.append({
            "day": day,
            "ctrl_visitors": ctrl_visitors,
            "var_visitors": var_visitors,
            "cum_ctrl_visitors": cum_ctrl_visitors,
            "cum_var_visitors": cum_var_visitors,
            "ctrl_orders": ctrl_orders,
            "var_orders": var_orders,
            "cum_ctrl_orders": cum_ctrl_orders,
            "cum_var_orders": cum_var_orders,
            "ctrl_returns": ctrl_returns,
            "var_returns": var_returns,
            "cum_ctrl_returns": cum_ctrl_returns,
            "cum_var_returns": cum_var_returns,
            "cum_ctrl_cvr": round(c_cvr, 4),
            "cum_var_cvr": round(v_cvr, 4),
            "cvr_uplift_pct": round(cvr_uplift_pct, 2),
            "p_val_cvr": round(p_cvr, 5),
            "cum_ctrl_return_rate": round(c_ret, 4),
            "cum_var_return_rate": round(v_ret, 4),
            "return_reduction_pct": round(ret_reduction_pct, 2),
            "p_val_return_rate": round(p_ret, 5),
            "significant_95": bool(p_ret < 0.05),
            "significant_99": bool(p_ret < 0.01),
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    impact = compute_reverse_logistics_impact()
    print("Reverse Logistics Impact:")
    print(f"Monthly returns saved: {impact.monthly_returns_saved:,}")
    print(f"Annual logistics cost saved: ₹{impact.annual_logistics_savings_inr:,.2f}")
    print(f"Annual GMV preserved: ₹{impact.annual_gmv_preserved_inr:,.2f}")

    ab_df = simulate_ab_test()
    print("\nA/B Test Final Day Metrics (Day 30):")
    final_day = ab_df.iloc[-1]
    print(f"Control CVR: {final_day['cum_ctrl_cvr']*100:.2f}%, Variant CVR: {final_day['cum_var_cvr']*100:.2f}% (p={final_day['p_val_cvr']})")
    print(f"Control Return Rate: {final_day['cum_ctrl_return_rate']*100:.2f}%, Variant Return Rate: {final_day['cum_var_return_rate']*100:.2f}% (p={final_day['p_val_return_rate']})")
    print(f"Statistically Significant @ 95% CI: {final_day['significant_95']}")
