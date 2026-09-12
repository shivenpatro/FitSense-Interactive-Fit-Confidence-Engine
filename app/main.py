"""
main.py
-------
FitSense: Interactive Fit Confidence Engine
Production-grade Streamlit application showcasing:
  - Tab 1: Myntra-style Customer PDP & FitSense™ Interactive Sizing Widget
  - Tab 2: Product Manager & Reverse Logistics Supply Chain Dashboard
  - Tab 3: Live PRD & Strategic Architectural Documentation
"""

import math
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Ensure fit_engine root is on sys.path
ENGINE_ROOT = Path(__file__).resolve().parent.parent
if str(ENGINE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT))

from src.data_pipeline import CATALOG_ITEMS, SIZES, load_or_generate_dataset
from src.nlp_engine import compute_sku_profiles, cache_sku_profiles, DB_PATH
from src.recommendation import FitRecommendationEngine
from src.metrics import compute_reverse_logistics_impact, simulate_ab_test

# Set page configuration
st.set_page_config(
    page_title="FitSense™ | Interactive Fit Confidence Engine",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Myntra Visual Identity
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Assistant:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Assistant', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Myntra Brand Colors */
    :root {
        --myntra-pink: #ff3f6c;
        --myntra-orange: #f26a10;
        --myntra-dark: #282c3f;
        --myntra-gray: #535766;
        --myntra-light: #f5f5f6;
        --myntra-border: #eaeaec;
        --myntra-green: #03a685;
    }
    
    /* Header Brand Banner */
    .myntra-header {
        background: linear-gradient(135deg, #ffffff 0%, #fff5f7 100%);
        padding: 1.2rem 1.8rem;
        border-radius: 12px;
        border-bottom: 3px solid #ff3f6c;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 4px 14px rgba(255, 63, 108, 0.08);
    }
    
    .brand-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: #282c3f;
        margin: 0;
    }
    
    .brand-title span {
        color: #ff3f6c;
    }
    
    .brand-tagline {
        font-size: 0.95rem;
        font-weight: 600;
        color: #535766;
        margin: 0;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #eaeaec;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 10px rgba(40, 44, 63, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(40, 44, 63, 0.08);
    }
    
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #535766;
        margin-bottom: 0.4rem;
    }
    
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #282c3f;
        margin: 0;
    }
    
    .kpi-subtext {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.4rem;
    }
    
    .kpi-green { color: #03a685; }
    .kpi-pink { color: #ff3f6c; }

    /* PDP Product Card */
    .pdp-card {
        background: #ffffff;
        border: 1px solid #eaeaec;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(40, 44, 63, 0.06);
    }
    
    .brand-badge {
        background: #f5f5f6;
        color: #282c3f;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    .price-tag {
        font-size: 1.5rem;
        font-weight: 800;
        color: #282c3f;
    }
    
    .original-price {
        font-size: 1.05rem;
        text-decoration: line-through;
        color: #94969f;
        margin-left: 0.5rem;
    }
    
    .discount-pct {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ff905a;
        margin-left: 0.5rem;
    }

    /* FitSense Widget Card */
    .fitsense-widget-box {
        background: #ffffff;
        border: 1.5px solid #ff3f6c;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(255, 63, 108, 0.09);
    }
    
    .recommendation-badge {
        background: linear-gradient(135deg, #ff3f6c 0%, #ff527b 100%);
        color: #ffffff;
        padding: 0.85rem 1.4rem;
        border-radius: 8px;
        font-size: 1.35rem;
        font-weight: 800;
        text-align: center;
        letter-spacing: 0.5px;
        box-shadow: 0 4px 12px rgba(255, 63, 108, 0.25);
        margin: 1rem 0;
    }
    
    .reasoning-box {
        background: #fff8f9;
        border-left: 4px solid #ff3f6c;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.95rem;
        color: #282c3f;
        line-height: 1.45;
        margin-top: 1rem;
    }
    
    .peer-stat-pill {
        background: #f5f5f6;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 700;
        color: #535766;
        display: inline-block;
        margin-right: 6px;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_dataset_and_engine():
    """Load or generate the fashion fit dataset and initialize the recommendation engine."""
    df = load_or_generate_dataset()
    if not DB_PATH.exists():
        profiles = compute_sku_profiles(df)
        cache_sku_profiles(profiles)
    engine = FitRecommendationEngine(historical_df=df)
    return df, engine


df, engine = get_dataset_and_engine()

# App Header
st.markdown("""
<div class="myntra-header">
    <div>
        <h1 class="brand-title">MYNTRA <span>• FITSENSE™</span></h1>
        <p class="brand-tagline">Interactive Fit Confidence & Sizing Intelligence Engine</p>
    </div>
    <div style="text-align: right;">
        <span style="background: #ff3f6c; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 0.85rem;">
            PM & CX SHOWCASE
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_cx, tab_pm, tab_prd = st.tabs([
    "🛍️ Customer PDP Experience (CX Demo)",
    "📊 Product Manager & Supply Chain Dashboard",
    "📄 Live PRD & System Architecture",
])


# ==============================================================================
# TAB 1: CUSTOMER PDP EXPERIENCE (STOREFRONT DEMO)
# ==============================================================================
with tab_cx:
    st.markdown("### 👗 Product Detail Page (PDP) Storefront Experience")
    st.caption("Interact with the live FitSense™ widget to receive real-time personalized size guidance and explainable reasoning.")

    # Product Selector Bar
    col_cat, col_item = st.columns([1, 2])
    with col_cat:
        selected_category = st.selectbox(
            "Filter Category",
            options=["Dress", "Jeans", "Top", "Ethnic"],
            index=0,
        )
    
    category_skus = [item for item in CATALOG_ITEMS if item["category"] == selected_category]
    sku_options = {f"{item['brand']} - {item['name']} ({item['item_id']})": item["item_id"] for item in category_skus}
    
    with col_item:
        selected_sku_label = st.selectbox(
            "Select Garment SKU to Test",
            options=list(sku_options.keys()),
            index=0,
        )
    
    selected_item_id = sku_options[selected_sku_label]
    sku_meta = next(item for item in CATALOG_ITEMS if item["item_id"] == selected_item_id)

    # Two-Column PDP Layout
    col_pdp, col_widget = st.columns([1.1, 1.2], gap="large")

    with col_pdp:
        st.markdown(f"""
        <div class="pdp-card">
            <span class="brand-badge">{sku_meta['brand'].upper()}</span>
            <h2 style="margin: 0.2rem 0; font-weight: 800; font-size: 1.5rem; color: #282c3f;">{sku_meta['name']}</h2>
            <p style="color: #535766; font-size: 0.9rem; margin-bottom: 0.8rem;">SKU: {sku_meta['item_id']} | Category: {sku_meta['category']}</p>
            
            <div style="display: flex; align-items: baseline; margin-bottom: 1rem;">
                <span class="price-tag">₹{int(sku_meta['base_price'] * 0.65)}</span>
                <span class="original-price">₹{sku_meta['base_price']}</span>
                <span class="discount-pct">(35% OFF)</span>
            </div>
            
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1.2rem;">
                <span style="background: #03a685; color: white; padding: 2px 8px; border-radius: 4px; font-weight: 700; font-size: 0.85rem;">4.2 ★</span>
                <span style="color: #535766; font-size: 0.85rem; font-weight: 600;">1,420 Ratings & 380 Verified Reviews</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # SVG Stylized Fashion Garment Card
        if sku_meta["category"] == "Dress":
            garment_icon = "👗"
            garment_desc = "Flattering feminine drape with elegant tailored silhouette. Perfect for evening outings or casual daywear."
        elif sku_meta["category"] == "Jeans":
            garment_icon = "👖"
            garment_desc = "Premium denim engineered for comfort and retention. Structured waistband with precision seam stitching."
        elif sku_meta["category"] == "Top":
            garment_icon = "👚"
            garment_desc = "Breathable lightweight fabric cut with modern aesthetics. Versatile pairing across casual and work settings."
        else:
            garment_icon = "🥻"
            garment_desc = "Authentic festive craftsmanship with intricate motifs and regal silhouette."

        st.markdown(f"""
        <div style="background: linear-gradient(180deg, #fbfbfb 0%, #f4f4f6 100%); border-radius: 12px; border: 1px dashed #d4d5d9; padding: 2rem; text-align: center; margin-top: 1rem;">
            <div style="font-size: 5rem;">{garment_icon}</div>
            <p style="font-weight: 700; color: #282c3f; margin-top: 0.5rem; font-size: 1.05rem;">{sku_meta['name']}</p>
            <p style="color: #535766; font-size: 0.88rem; max-width: 400px; margin: 0 auto;">{garment_desc}</p>
        </div>
        """, unsafe_allow_html=True)

        # Recent Verified Customer Reviews
        st.markdown("#### 💬 Verified Customer Fit Mentions")
        sku_reviews = df[df["item_id"] == selected_item_id].head(3)
        if not sku_reviews.empty:
            for _, rev in sku_reviews.iterrows():
                badge_color = "#03a685" if rev["fit_feedback"] == "Fit" else ("#ff3f6c" if rev["fit_feedback"] == "Small" else "#f26a10")
                st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #eaeaec; border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.6rem;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 0.3rem;">
                        <span style="font-weight: 700; font-size: 0.85rem; color: #282c3f;">Customer ({rev['user_body_type']}, {int(rev['user_height_cm'])}cm, {int(rev['user_weight_kg'])}kg)</span>
                        <span style="background: {badge_color}22; color: {badge_color}; padding: 2px 8px; border-radius: 12px; font-weight: 700; font-size: 0.75rem;">Felt: {rev['fit_feedback']} (Ordered {rev['size_ordered']})</span>
                    </div>
                    <p style="font-size: 0.85rem; color: #535766; margin: 0; font-style: italic;">"{rev['review_text']}"</p>
                </div>
                """, unsafe_allow_html=True)

    with col_widget:
        st.markdown("""
        <div class="fitsense-widget-box">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
                <h3 style="margin: 0; color: #ff3f6c; font-weight: 800; font-size: 1.25rem;">
                    ✨ FitSense™ Personalized Sizing
                </h3>
                <span style="background: #ff3f6c15; color: #ff3f6c; font-weight: 700; font-size: 0.75rem; padding: 3px 8px; border-radius: 4px;">
                    AI ACTIVE
                </span>
            </div>
            <p style="color: #535766; font-size: 0.88rem; margin-bottom: 1.2rem;">
                Tired of returns? Input your measurements to calculate your highest-confidence size.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Interactive Inputs
        col_h, col_w = st.columns(2)
        with col_h:
            height_val = st.slider(
                "Your Height (cm)",
                min_value=145,
                max_value=195,
                value=165,
                step=1,
                help="We match your height against garment inseams and drape lengths.",
            )
            feet = int(height_val / 30.48)
            inches = int(round((height_val / 30.48 - feet) * 12))
            st.caption(f"Equivalent: **{feet} ft {inches} in**")

        with col_w:
            weight_val = st.slider(
                "Your Weight (kg)",
                min_value=40,
                max_value=115,
                value=62,
                step=1,
                help="Calculates baseline volumetric sizing combined with height.",
            )
            bmi_val = round(weight_val / ((height_val / 100) ** 2), 1)
            st.caption(f"Baseline BMI: **{bmi_val}**")

        col_bt, col_pref = st.columns(2)
        with col_bt:
            body_type_val = st.selectbox(
                "Your Body Shape",
                options=["Hourglass", "Pear", "Petite", "Athletic", "Regular"],
                index=0,
                help="Accounts for regional weight distribution across hips, shoulders, and waist.",
            )

        with col_pref:
            preference_val = st.selectbox(
                "Fit Preference",
                options=["Regular", "Snug / Form-Fitting", "Relaxed / Loose"],
                index=0,
                help="Overrides baseline sizing to match your personal styling comfort.",
            )

        # Run Recommendation Engine
        rec = engine.recommend(
            item_id=selected_item_id,
            user_height_cm=float(height_val),
            user_weight_kg=float(weight_val),
            user_body_type=body_type_val,
            user_fit_preference=preference_val,
        )

        # Recommendation Result Presentation
        st.markdown(f"""
        <div class="recommendation-badge">
            RECOMMENDED SIZE: {rec.recommended_size}
        </div>
        """, unsafe_allow_html=True)

        # Confidence Gauge & Stats
        col_gauge, col_conf_stats = st.columns([1.2, 1])

        with col_gauge:
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=rec.confidence_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Fit Confidence", 'font': {'size': 14, 'color': '#282c3f', 'family': 'Assistant'}},
                number={'suffix': "%", 'font': {'size': 26, 'color': '#282c3f', 'family': 'Assistant'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#535766"},
                    'bar': {'color': "#ff3f6c"},
                    'bgcolor': "#f5f5f6",
                    'borderwidth': 1,
                    'bordercolor': "#eaeaec",
                    'steps': [
                        {'range': [0, 60], 'color': '#ffe0e6'},
                        {'range': [60, 80], 'color': '#ffb3c1'},
                        {'range': [80, 100], 'color': '#e8f5e9'},
                    ],
                }
            ))
            gauge_fig.update_layout(height=160, margin=dict(l=10, r=10, t=25, b=10))
            st.plotly_chart(gauge_fig, use_container_width=True)

        with col_conf_stats:
            st.markdown(f"""
            <div style="padding: 0.5rem 0;">
                <div class="peer-stat-pill">SKU Tendency: <b>{rec.fit_tendency_label}</b></div>
                <div class="peer-stat-pill">Peers Evaluated: <b>{rec.peer_stats.get('peer_count', 24)}</b></div>
                <div class="peer-stat-pill">Peer Fit Rate: <b>{rec.peer_stats.get('peer_fit_satisfaction_pct', 80)}%</b></div>
                <div class="peer-stat-pill">Fabric Stretch: <b>{int(rec.stretch_score * 100)}%</b></div>
            </div>
            """, unsafe_allow_html=True)

        # Explainable CX Reasoning
        st.markdown(f"""
        <div class="reasoning-box">
            <b>💡 Why FitSense recommends Size {rec.recommended_size}:</b><br>
            {rec.reasoning}
        </div>
        """, unsafe_allow_html=True)

        # Garment Attribute Breakdown Plotly Chart
        st.markdown("##### 📐 Garment Attribute Fit Skew")
        attr_data = pd.DataFrame([
            {"Attribute": k, "Skew": v}
            for k, v in rec.attribute_breakdown.items()
        ])
        
        # Color bar green for near zero, red/pink for small, orange for large
        colors = []
        for v in attr_data["Skew"]:
            if abs(v) < 0.2:
                colors.append("#03a685")
            elif v < -0.2:
                colors.append("#ff3f6c")
            else:
                colors.append("#f26a10")
        
        fig_attr = go.Figure(go.Bar(
            x=attr_data["Skew"],
            y=attr_data["Attribute"],
            orientation='h',
            marker_color=colors,
            text=[f"{v:+.2f}" for v in attr_data["Skew"]],
            textposition="auto",
        ))
        fig_attr.update_layout(
            xaxis=dict(
                title="Fit Skew (-1.0 Runs Small | +1.0 Runs Large)",
                range=[-1.0, 1.0],
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor="#282c3f",
            ),
            yaxis=dict(autorange="reversed"),
            height=180,
            margin=dict(l=10, r=10, t=10, b=25),
        )
        st.plotly_chart(fig_attr, use_container_width=True)

        # Interactive "Add to Bag" Action
        st.markdown("<br>", unsafe_allow_html=True)
        col_bag_btn, col_bag_info = st.columns([1.5, 1])
        with col_bag_btn:
            if st.button(f"🛍️ ADD SIZE {rec.recommended_size} TO BAG", type="primary", use_container_width=True):
                st.balloons()
                st.success(f"Added Size {rec.recommended_size} to Bag with FitSense™ 95% Sizing Protection Guarantee!")
        with col_bag_info:
            st.caption("🔒 Guaranteed Easy Exchanges & 100% Fit Protection")


# ==============================================================================
# TAB 2: PRODUCT MANAGER & SUPPLY CHAIN DASHBOARD
# ==============================================================================
with tab_pm:
    st.markdown("### 📊 Product Analytics, Reverse Logistics & A/B Experimentation")
    st.caption("Executive overview of reverse logistics cost preservation, customer retention lift, and live A/B testing statistical convergence.")

    # Executive Parameter Controls
    with st.expander("⚙️ Scenario Simulation Parameters & Unit Economics", expanded=False):
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            pm_orders = st.number_input("Monthly Fitted Apparel Orders", min_value=5000, max_value=500000, value=50000, step=5000)
            pm_aov = st.number_input("Average Order Value (₹)", min_value=500.0, max_value=10000.0, value=1850.0, step=50.0)
        with col_p2:
            pm_baseline_ret = st.slider("Baseline Return Rate (%)", min_value=15.0, max_value=45.0, value=30.0, step=0.5) / 100.0
            pm_projected_ret = st.slider("FitSense Adopter Return Rate (%)", min_value=10.0, max_value=30.0, value=21.5, step=0.5) / 100.0
        with col_p3:
            pm_cost_per_ret = st.number_input("Direct Reverse Logistics Cost / Return (₹)", min_value=50.0, max_value=300.0, value=120.0, step=5.0)
            pm_adoption = st.slider("FitSense Widget Engagement Rate (%)", min_value=20.0, max_value=95.0, value=65.0, step=5.0) / 100.0

    # Calculate Business Impact
    impact = compute_reverse_logistics_impact(
        monthly_orders=int(pm_orders),
        avg_order_value_inr=float(pm_aov),
        baseline_return_rate=float(pm_baseline_ret),
        projected_return_rate=float(pm_projected_ret),
        fitsense_adoption_rate=float(pm_adoption),
        cost_per_return_inr=float(pm_cost_per_ret),
    )

    # Executive Scorecards
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    with col_k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Monthly Returns Prevented</div>
            <div class="kpi-value kpi-green">{impact.monthly_returns_saved:,}</div>
            <div class="kpi-subtext">{impact.annual_returns_saved:,} Units / Year</div>
        </div>
        """, unsafe_allow_html=True)

    with col_k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Annual Logistics Cost Preserved</div>
            <div class="kpi-value kpi-pink">₹{impact.annual_logistics_savings_inr / 100000:.1f} L</div>
            <div class="kpi-subtext">₹{impact.monthly_logistics_savings_inr:,.0f} / Month Saved</div>
        </div>
        """, unsafe_allow_html=True)

    with col_k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Annual GMV Preserved</div>
            <div class="kpi-value kpi-green">₹{impact.annual_gmv_preserved_inr / 10000000:.2f} Cr</div>
            <div class="kpi-subtext">Reduced Churn & Bracket Returns</div>
        </div>
        """, unsafe_allow_html=True)

    with col_k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Net Return Rate Drop</div>
            <div class="kpi-value">{impact.baseline_return_rate*100:.1f}% → {impact.effective_return_rate*100:.1f}%</div>
            <div class="kpi-subtext kpi-green">↓ {impact.relative_return_reduction_pct:.1f}% Relative Reduction</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Analytics Charts Row 1
    col_ch1, col_ch2 = st.columns(2)

    with col_ch1:
        st.markdown("#### 🎯 Apparel Return Reasons Distribution")
        # Return reasons pie / donut
        reasons_counts = df[df["returned"] == 1]["return_reason"].value_counts().reset_index()
        reasons_counts.columns = ["Reason", "Count"]

        fig_reasons = px.pie(
            reasons_counts,
            values="Count",
            names="Reason",
            hole=0.55,
            color="Reason",
            color_discrete_map={
                "Fit": "#ff3f6c",
                "Quality": "#535766",
                "Not as pictured": "#ff905a",
                "None": "#eaeaec",
            }
        )
        fig_reasons.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_reasons, use_container_width=True)

    with col_ch2:
        st.markdown("#### 📈 SKU Return Rate vs. Fit Confidence")
        # SKU scatter plot
        sku_summary = df.groupby("item_id").agg({
            "returned": "mean",
            "rating": "mean",
            "brand": "first",
            "category": "first",
            "review_id": "count",
        }).reset_index().rename(columns={"review_id": "review_count", "returned": "return_rate"})

        # Compute synthetic confidence score for SKU
        sku_summary["fit_confidence"] = np.clip(
            (np.log10(sku_summary["review_count"] + 1) / 2.0 * 60) + (sku_summary["rating"] * 8),
            50, 95
        )

        fig_scatter = px.scatter(
            sku_summary,
            x="fit_confidence",
            y="return_rate",
            size="review_count",
            color="category",
            hover_name="item_id",
            hover_data={"brand": True, "return_rate": ":.1%"},
            labels={
                "fit_confidence": "FitSense Confidence Score (%)",
                "return_rate": "Actual SKU Return Rate",
            },
            color_discrete_sequence=["#ff3f6c", "#282c3f", "#f26a10", "#03a685"],
        )
        # Add regression trendline cleanly with numpy without requiring extra dependencies
        if len(sku_summary) > 1:
            z_fit = np.polyfit(sku_summary["fit_confidence"], sku_summary["return_rate"], 1)
            p_fit = np.poly1d(z_fit)
            x_line = np.linspace(sku_summary["fit_confidence"].min(), sku_summary["fit_confidence"].max(), 50)
            fig_scatter.add_trace(go.Scatter(
                x=x_line,
                y=p_fit(x_line),
                mode="lines",
                name="Trendline (Linear Fit)",
                line=dict(color="#535766", dash="dash", width=2),
            ))
        fig_scatter.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=20, b=20),
            yaxis_tickformat='.0%',
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Analytics Charts Row 2: 30-Day Simulated A/B Testing
    st.markdown("---")
    st.markdown("#### 🧪 30-Day Simulated A/B Experimentation: Control vs. FitSense Variant")
    st.caption("Live statistical hypothesis testing (Two-sample Z-test) demonstrating return rate suppression and conversion rate uplift over 30 days.")

    ab_df = simulate_ab_test(n_days=30, seed=42)

    col_ab1, col_ab2 = st.columns(2)

    with col_ab1:
        st.markdown("##### Cumulative Return Rate Convergence")
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Scatter(
            x=ab_df["day"],
            y=ab_df["cum_ctrl_return_rate"] * 100,
            mode="lines+markers",
            name="Control (Standard Size Chart)",
            line=dict(color="#535766", width=2, dash="dash"),
        ))
        fig_ret.add_trace(go.Scatter(
            x=ab_df["day"],
            y=ab_df["cum_var_return_rate"] * 100,
            mode="lines+markers",
            name="Variant (FitSense™ Widget)",
            line=dict(color="#ff3f6c", width=3),
        ))
        fig_ret.update_layout(
            xaxis_title="Experiment Day",
            yaxis_title="Return Rate (%)",
            height=320,
            margin=dict(l=10, r=10, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    with col_ab2:
        st.markdown("##### Daily Statistical Significance (P-Value Curve)")
        fig_pval = go.Figure()
        fig_pval.add_trace(go.Scatter(
            x=ab_df["day"],
            y=ab_df["p_val_return_rate"],
            mode="lines+markers",
            name="Two-Sample Z-Test p-Value",
            line=dict(color="#03a685", width=2.5),
        ))
        # Threshold line p = 0.05
        fig_pval.add_hline(
            y=0.05,
            line_dash="dot",
            line_color="#ff3f6c",
            annotation_text="Significance Threshold (p = 0.05)",
            annotation_position="top right",
        )
        fig_pval.update_layout(
            xaxis_title="Experiment Day",
            yaxis_title="P-Value (Log Scale)",
            yaxis_type="log",
            height=320,
            margin=dict(l=10, r=10, t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        )
        st.plotly_chart(fig_pval, use_container_width=True)


# ==============================================================================
# TAB 3: LIVE PRD & STRATEGIC DOCUMENTATION
# ==============================================================================
with tab_prd:
    st.markdown("### 📄 Product Requirement Document & Technical Architecture")
    prd_path = ENGINE_ROOT / "docs" / "PRD.md"
    if prd_path.exists():
        with open(prd_path, "r", encoding="utf-8") as f:
            prd_content = f.read()
        st.markdown(prd_content, unsafe_allow_html=False)
    else:
        st.info("PRD.md documentation is loading...")
