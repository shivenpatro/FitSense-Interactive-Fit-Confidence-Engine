---
title: FitSense Interactive Fit Confidence Engine
emoji: 👗
colorFrom: pink
colorTo: red
sdk: streamlit
sdk_version: "1.35.0"
app_file: app/main.py
pinned: false
license: mit
---

# FitSense: Interactive Fit Confidence Engine

> **A Production-Grade Sizing Recommendation & Reverse Logistics Optimization Showcase for Fashion E-Commerce (Myntra PM / CX & Supply Chain Showcase)**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Analytics](https://img.shields.io/badge/Engine-DuckDB%20%7C%20Pandas%20%7C%20Scikit--Learn-orange.svg)](https://duckdb.org/)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-brightgreen.svg)](https://pytest.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Executive Summary & Problem Statement

In online apparel e-commerce (platforms like **Myntra**, **Ajio**, **ASOS**, and **Zalando**), return rates regularly reach **28% to 35%**, with fitted categories (denim, structured dresses, tailored tops) experiencing returns as high as **38%**.

Audits show that **over 64% of apparel returns are caused entirely by sizing and fit mismatches**. When customers are uncertain about sizing, they either:
1. **Abandon their carts** due to decision paralysis (lowering conversion rates).
2. **"Bracket order"** (ordering two adjacent sizes, e.g., M and L, with the premeditated intent of returning one).

Each return incurs **₹120 to ₹145** in direct, non-recoverable reverse logistics operational costs (doorstep reverse courier pickup, quality inspection, re-tagging, dry cleaning, and re-warehousing) plus days of inventory lock-up.

**FitSense** solves this at the point of decision on the Product Detail Page (PDP). By fusing **customer anthropometrics**, **attribute-level review NLP sentiment**, and **peer cohort similarity**, FitSense delivers a high-confidence personalized size recommendation, an explainable reasoning card, and a visual garment fit skew breakdown.

---

## 2. System Architecture & Tech Stack

```
                               ┌────────────────────────────────────────────────────────┐
                               │             CLIENT STOREFRONT INTERFACE                │
                               │          (Streamlit • Myntra Visual Identity)          │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │                 FITSENSE INFERENCE CORE                │
                               │                                                        │
 ┌──────────────────────────┐  │   ┌────────────────────────┐  ┌──────────────────────┐ │
 │ Anthropometric Modeler   │──┼──>│ Base Size Determinism  │  │ SKU NLP Profile DB   │ │
 │ (Height, Weight, Body)   │  │   └───────────┬────────────┘  │ (Waist, Bust, Length)│ │
 └──────────────────────────┘  │               │               └──────────┬───────────┘ │
                               │               ▼                          ▼             │
 ┌──────────────────────────┐  │   ┌──────────────────────────────────────────────────┐ │
 │ Customer Preference      │──┼──>│ FitRecommendationEngine                          │ │
 │ (Snug, Regular, Relaxed) │  │   │ • Attribute & Fabric Stretch Moderation          │ │
 └──────────────────────────┘  │   │ • Confidence Scoring (Review Volume, Similarity) │ │
                               │   │ • CX Explainability & Natural Language Generator │ │
                               │   └──────────────────────────┬───────────────────────┘ │
                               └──────────────────────────────┼─────────────────────────┘
                                                              ▼
                               ┌────────────────────────────────────────────────────────┐
                               │           ANALYTICS & SUPPLY CHAIN SIMULATOR           │
                               │  • Reverse Logistics Cost Model (Courier, QC, Restock) │
                               │  • 30-Day A/B Test Simulator (Two-Sample Z-Test, CVR)  │
                               │  • DuckDB / SQLite Cache & Parquet Data Lake           │
                               └────────────────────────────────────────────────────────┘
```

### Core Technologies
- **Language**: Python 3.10+
- **Data Ingestion & Lake**: Pandas, NumPy, DuckDB, SQLite, PyArrow
- **NLP & Sentiment Mining**: Rule-based Directional Parsing, Regular Expressions, TextBlob Polarity
- **Recommendation & ML**: Scikit-Learn, Anthropometric BMICalibrator, Cohort Distance Kernels
- **Application & UX**: Streamlit (with custom CSS injection matching Myntra's design tokens)
- **Visualization**: Plotly Express & Graph Objects
- **Automated Testing**: Pytest

---

## 3. How the Recommendation Algorithm Works

FitSense computes recommendations through a 4-stage pipeline:

### Stage 1: Anthropometric Baseline Size
Calculates the customer's base sizing index from height, weight, and regional morphology:
$$\text{BMI}_{\text{effective}} = \frac{\text{Weight (kg)}}{(\text{Height (m)})^2} + \Delta_{\text{BodyType}}$$
Where $\Delta_{\text{BodyType}}$ applies calibrated shifts for *Hourglass* (+0.3), *Pear* (+0.5), *Petite* (-0.5), or *Athletic* (+0.4) silhouettes.

### Stage 2: SKU Attribute NLP Profile & Preference Calibration
In tandem, the engine evaluates product-level attributes mined from historical customer reviews:
- **Fit Tendency** $\in [-1.0 \text{ (Runs Small)}, +1.0 \text{ (Runs Large)}]$
- **Specific Dimensions**: `Waist`, `Bust/Chest`, `Length`, `Hips`
- **Fabric Stretch Factor** $\in [0.0 \text{ (Rigid/No Give)}, 1.0 \text{ (High Stretch)}]$

*Adjustment Matrix*:
- If SKU runs small ($\text{Tendency} \le -0.28$) and user does not prefer snug $\rightarrow$ **+1 Size Shift**.
- If SKU runs large ($\text{Tendency} \ge +0.28$) and user does not prefer relaxed $\rightarrow$ **-1 Size Shift**.
- If fabric is rigid ($\text{Stretch} \le 0.15$) and body type is Pear/Hourglass in fitted denim/dresses $\rightarrow$ Sizing protection applied.
- User fit preference ("Snug", "Relaxed") acts as an intentional silhouette override.

### Stage 3: Multi-Factor Fit Confidence Score (0% – 100%)
$$\text{Confidence Score} = \left( W_{\text{review}} \times 0.40 + S_{\text{body}} \times 0.40 + V_{\text{consensus}} \times 0.20 \right) \times 100$$
1. **Review Count Weight ($W_{\text{review}}$)**: $\min\left(1.0, \frac{\log_{10}(N_{\text{reviews}} + 1)}{2.0}\right)$ (rewards statistically robust review volumes).
2. **Body Match Similarity ($S_{\text{body}}$)**: Empirical fit satisfaction rate among past shoppers within $\pm 6\text{ cm}$ height and $\pm 6\text{ kg}$ weight with matching body type.
3. **Variance Consensus Penalty ($V_{\text{consensus}}$)**: Quantifies customer feedback agreement; penalizes items with polarized opinions.

### Stage 4: Explainable CX Reasoning
Produces customer-centric explanations:
> *"82% of shoppers with an Athletic build (168 cm, 64 kg) reported that Mango's tailored cut runs snug across the chest. We sized you up to **Size L** to guarantee a flattering, comfortable drape."*

---

## 4. Key Metrics & Reverse Logistics Unit Economics

### Reverse Logistics Cost Equation
$$\text{Monthly Operational Savings (₹)} = N_{\text{orders}} \times (\text{ReturnRate}_{\text{baseline}} - \text{ReturnRate}_{\text{effective}}) \times C_{\text{return}}$$

At **50,000 monthly orders** in fitted apparel:
- **Baseline Return Rate**: 30.0% (15,000 returned units)
- **FitSense Projected Return Rate**: 21.5% (Adoption: 65% $\rightarrow$ Effective Blended: 24.5%)
- **Monthly Returns Prevented**: **2,762 orders**
- **Direct Reverse Logistics Cost per Return**: **₹120** (Courier ₹65 + QC ₹25 + Restock ₹30)
- **Net Monthly Logistics Savings**: **₹3,31,440** (₹3.31 Lakhs / month)
- **Annual Operational Cost Preserved**: **₹39,77,280** (~**₹40 Lakhs**)
- **Annual GMV Preserved**: **₹6.13 Crore** (based on ₹1,850 AOV)

---

## 5. Repository Structure

```
fit_engine/
├── .gitignore               # Clean git exclusions (pycache, env, raw data)
├── requirements.txt         # Production dependencies
├── README.md                # Comprehensive documentation
├── docs/
│   └── PRD.md               # Full Product Requirement Document
├── data/
│   ├── raw/                 # Raw source data
│   └── processed/           # Processed parquet data lake & SQLite DB
│       ├── clean_fit_data.parquet
│       ├── clean_fit_data.csv
│       └── fit_profiles.db
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py     # 10,000-row synthetic e-commerce fit generator
│   ├── nlp_engine.py        # Attribute sentiment & fit tendency mining
│   ├── recommendation.py    # Multi-factor sizing & confidence engine
│   └── metrics.py           # Supply chain economics & A/B test simulator
├── app/
│   ├── __init__.py
│   └── main.py              # Myntra-styled Streamlit multi-tab application
└── tests/
    ├── __init__.py
    └── test_engine.py       # Full Pytest test suite (Phases 1-4)
```

---

## 6. Local Setup & Execution

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed.

### Step 1: Clone and Navigate
```bash
git clone <repo-url>
cd fit_engine
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Test Suite
Verify that all unit tests across data generation, NLP, recommendation logic, and metrics pass:
```bash
pytest tests/ -v
```

### Step 4: Launch the Streamlit Application
```bash
streamlit run app/main.py
```
Open your browser at `http://localhost:8501`.

---

## 7. Interactive App Features

1. **Tab 1: Customer PDP Experience (CX Storefront Demo)**
   - Interactive Myntra PDP preview for Dresses, Jeans, Tops, and Ethnic Wear.
   - FitSense widget: Height & Weight sliders, Body Type selector, Preferred Fit toggle.
   - Recommended Size badge, Fit Confidence meter, Attribute skew bar chart, and AI reasoning.
   - Interactive "Add to Bag" interaction.
2. **Tab 2: Product Manager & Supply Chain Dashboard**
   - Real-time KPI scorecards: Returns saved, Cost preserved, Conversion uplift.
   - Return reason distribution donut chart.
   - SKU Return Rate vs. Fit Confidence scatter plot.
   - Simulated 30-Day A/B Test conversion and p-value significance convergence chart.
   - Configurable business parameters (order volume, return handling cost, adoption rate).
3. **Tab 3: Live PRD & Strategic Documentation**
   - Complete embedded PRD with OKRs, system architecture, and unit economics formulas.

---

## License
MIT License. Created by Shiven Patro.
