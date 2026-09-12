# Product Requirement Document (PRD)

## FitSense: Interactive Fit Confidence & Sizing Engine
**Product Tier**: Core CX & Supply Chain Profitability  
**Target Platform**: Myntra Web & Mobile App (iOS / Android)  
**Target Category**: Women’s & Unisex Apparel (Dresses, Denim, Tops, Ethnic Wear)  
**Author**: Shiven Patro (Senior Full-Stack & Product Analytics)  
**Status**: Ready for Production Deployment  

---

## 1. Executive Summary & Problem Framing

### 1.1 The Industry Sizing Crisis
In Indian fashion e-commerce, **apparel return rates range between 25% and 35%**, with category peaks in Western Wear and Fitted Denim reaching **38%**. Internal and industry audits confirm that **62% to 70% of apparel returns are triggered strictly by fit and sizing discrepancies** rather than product defects.

```
Total E-Commerce GMV
        │
        ├── Delivered Orders (68%)
        └── Returns / RTOs (32%)
                 ├── Sizing & Fit Mismatch (64%)  <-- TARGET PROBLEM
                 ├── Quality / Material (18%)
                 ├── Not as Pictured (11%)
                 └── Customer Regret / Impulsive (7%)
```

### 1.2 The Supply Chain Reverse Logistics Drain
Each returned apparel shipment incurs substantial non-recoverable operational friction:
- **Reverse Pickup Courier Charges**: ₹65 – ₹80
- **Quality Inspection & Authentication**: ₹20 – ₹28
- **Refurbishment, Re-Tagging, Steaming & Re-Warehousing**: ₹25 – ₹35
- **Inventory Depreciation & Markdown Risk**: 15% – 25% reduction in full-price sell-through due to transit lag (returned items average 9.4 days in transit before re-shelving).

**Total direct cost per return**: **₹120 – ₹145 per unit**. At a scale of 50,000 monthly orders in fitted categories, sizing returns destroy **₹1.8 Crore to ₹2.4 Crore annually** in direct logistics margin, excluding GMV churn.

### 1.3 Strategic Solution: FitSense
FitSense is a client-facing interactive fit assistant embedded into the Product Detail Page (PDP). By coupling:
1. **Anthropometric morphology modeling** (Height, Weight, Body Shape)
2. **Review-level attribute NLP sentiment extraction** (Waist, Bust/Chest, Length, Hips, Stretch)
3. **Statistical consensus & peer similarity confidence scoring**

FitSense replaces ambiguous static 2D size charts with actionable, high-confidence personalized size guidance and transparent reasoning.

---

## 2. Business Objectives & OKRs

### 2.1 Primary OKRs
| Objective | Key Result | Target |
| :--- | :--- | :--- |
| **Drastically Curtail Reverse Logistics Waste** | Reduce overall apparel return rate across pilot categories | **-26% Relative Drop** (30.0% -> 22.2%) |
| **Preserve Gross Margin** | Direct reverse logistics operational savings | **> ₹7.5 Lakh / Month** per 50k orders |
| **Accelerate Funnel Conversion** | Lift PDP-to-Checkout conversion by removing sizing hesitation | **+15% to +20% Relative CVR Uplift** (3.2% -> 3.8%) |
| **Elevate Customer Trust** | FitSense widget adoption rate among PDP visitors | **> 60% Engagement Rate** |

### 2.2 Secondary Product Metrics
- **Bracket Ordering Reduction**: Decrease instances of customers ordering multiple sizes of the same SKU (e.g., ordering both S and M) by **45%**.
- **Customer Lifetime Value (LTV)**: Customers who experience 0 fit returns in their first 2 purchases exhibit **2.4x higher 6-month repeat purchase rates**.

---

## 3. User Personas & Journey

### Persona 1: Ananya (The Discerning Shopper)
- **Profile**: 26-year-old software professional, 168 cm, 64 kg, Athletic build.
- **Pain Point**: Often caught between Size M and L. Shirts that fit her waist are tight across the shoulders.
- **FitSense Experience**: FitSense detects from 180+ verified reviews that the SKU runs narrow across the shoulders, proactively recommending Size L with an 88% confidence score and explanation: *"78% of athletic shoppers found this cut snug at the shoulders. We suggest Size L for comfort."*

### Persona 2: Rohit (Supply Chain & Merchandising Director)
- **Profile**: VP of Logistics & Merchandising.
- **Pain Point**: High RTO and reverse logistics expenses eating into category EBITDA margins.
- **FitSense Experience**: Real-time PM dashboard monitoring SKU return rate vs. Fit Confidence, automated brand skew alerts, and A/B test statistical convergence.

---

## 4. System Architecture & Technical Specifications

```
┌─────────────────────────────────────────────────────────────────┐
│                      CLIENT / PDP STOREFRONT                    │
│   • Myntra Mobile App / Web PDP                                 │
│   • Interactive FitSense Widget (Height, Weight, Body, Pref)   │
│   • Confidence Meter & Garment Attribute Skew Bar Chart         │
└───────────────────────────────┬─────────────────────────────────┘
                                │ JSON API
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 FITSENSE INFERENCE ENGINE (src/)                │
│                                                                 │
│   ┌────────────────────────┐      ┌─────────────────────────┐   │
│   │ Anthropometric Base    │      │ SKU Attribute Profile   │   │
│   │ Size Estimator         │      │ (NLP Extracted Skews)   │   │
│   └───────────┬────────────┘      └───────────┬─────────────┘   │
│               │                               │                 │
│               └───────────────┬───────────────┘                 │
│                               ▼                                 │
│               ┌───────────────────────────────┐                 │
│               │ FitRecommendationEngine       │                 │
│               │ • Preference Adjustment       │                 │
│               │ • Fabric Stretch Moderation   │                 │
│               │ • Confidence Scoring (0-100%) │                 │
│               │ • Explainable Reasoning Gen   │                 │
│               └───────────────────────────────┘                 │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA & ANALYTICS STORAGE                     │
│   • DuckDB / SQLite Cache (`fit_profiles.db`)                   │
│   • 10,000 Transaction Parquet Lake (`clean_fit_data.parquet`) │
│   • Reverse Logistics Cost & A/B Testing Engine (`metrics.py`)  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.1 Recommendation Algorithm Logic
1. **Base Size Calculation**:
   $$\text{BMI} = \frac{\text{Weight (kg)}}{(\text{Height (m)})^2} + \Delta_{\text{BodyType}}$$
   Where $\Delta_{\text{BodyType}} \in [-0.5, +0.5]$ calibrates for regional mass distribution.
2. **SKU NLP Profile Ingestion**:
   Attributes evaluated: $\text{Waist}, \text{Bust/Chest}, \text{Length}, \text{Hips}, \text{Fabric Stretch}$.
   SKU Fit Tendency $\in [-1.0 \text{ (Runs Small)}, +1.0 \text{ (Runs Large)}]$.
3. **Multi-Factor Confidence Score Equation**:
   $$\text{Confidence} = \left( W_{\text{reviews}} \times 0.40 + S_{\text{peer\_match}} \times 0.40 + V_{\text{consensus}} \times 0.20 \right) \times 100$$
   - $W_{\text{reviews}} = \min\left(1.0, \frac{\log_{10}(n + 1)}{2}\right)$
   - $S_{\text{peer\_match}}$ = Empirical satisfaction rate of cohort peers ($\pm 6\text{ cm}, \pm 6\text{ kg}$).
   - $V_{\text{consensus}}$ = Inverse entropy of feedback dispersion across sizes.

---

## 5. Reverse Logistics Unit Economics

### 5.1 Financial Formula
$$\text{Net Monthly Savings (₹)} = N_{\text{orders}} \times \Delta_{\text{return\_rate}} \times \alpha_{\text{adoption}} \times C_{\text{unit\_return}}$$
Where:
- $N_{\text{orders}} = 50,000$ monthly volume
- $\Delta_{\text{return\_rate}} = 30.0\% - 21.5\% = 8.5\%$ absolute reduction
- $\alpha_{\text{adoption}} = 65\%$ client engagement
- $C_{\text{unit\_return}} = ₹120$ per return operational handling

$$\text{Monthly Operational Savings} = 50,000 \times 0.085 \times 0.65 \times 120 = \mathbf{₹3,31,500 \text{ per month}}$$
$$\text{Annual Operational Savings} = \mathbf{₹39,78,000} \approx \mathbf{₹40 \text{ Lakhs}}$$
$$\text{Annual GMV Preserved} = 50,000 \times 0.085 \times 0.65 \times 1,850 \times 12 = \mathbf{₹6.13 \text{ Crore GMV}}$$

---

## 6. A/B Testing & Rollout Strategy

### 6.1 Experimentation Setup
- **Traffic Allocation**: 50% Control (Standard PDP Size Chart) vs. 50% Variant (FitSense Interactive Widget).
- **Duration**: 30 Days.
- **Statistical Power**: $1 - \beta = 0.80$, Significance Level $\alpha = 0.05$.
- **Hypothesis**: The Variant will demonstrate statistically significant ($p < 0.05$) drop in 30-day return rate with non-inferior (positive) conversion rate.

### 6.2 Rollout Phases
- **Phase A (Internal Dogfooding)**: 100 internal Myntra employees testing 50 SKUs.
- **Phase B (Category Pilot - 10% Traffic)**: Western Wear & Women's Denim.
- **Phase C (Full Rollout - 100% Traffic)**: All fitted apparel categories.
