# 🏛️ Illinois County Economic Intelligence Dashboard

An interactive spatial data analysis dashboard built with Python and Streamlit. This project combines the **high predictive accuracy of Machine Learning (Random Forest)** with the **interpretability of traditional Econometrics (OLS Linear Regression)** to diagnose, visualize, and simulate unemployment rates across all 102 Illinois counties using 7 core socioeconomic and demographic features.

---

## ✨ Key Features

- 🧠 **Dual-Model Architecture**:
  - **Random Forest**: Captures complex, non-linear relationships for high-accuracy unemployment predictions and feature importance rankings.
  - **OLS (Ordinary Least Squares)**: Provides rigorous econometric inference (coefficients and p-values), explaining the directional impact of each variable.
- 🔮 **AI Scenario Simulator (What-If Analysis)**: Dynamically adjust 7 socioeconomic levers (e.g., increasing Bachelor's degree rates, lowering poverty) and observe real-time model re-predictions via a gauge chart.
- 🗺️ **Spatial Intelligence Maps**: Interactive Plotly Mapbox choropleth maps with statewide and Chicago metro zoom views, visualizing economic metrics and **model residuals** across geographic space.
- 📊 **Dimensional Metric Splitting**: Separates percentage metrics (0–100%) from absolute values (USD/Years) in county profiles to avoid scale imbalance visualization pitfalls.
- 📈 **Extended Diagnostic Analysis** (in QMD notebooks):
  - K-Means clustering for economic typology mapping
  - Temporal recovery analysis (2020 → 2023)
  - Single-variable scatter subplots and quantile analysis
  - OLS summary table with Q-Q residual diagnostics
  - Outlier county deep dive (Top 5 / Bottom 5 residuals)

---

## 📂 Project Structure

```text
├── raw-data/                        # Raw, unprocessed CSV datasets
│   ├── UnemploymentReport.csv       # Main table: 2023 unemployment by county
│   ├── bachelor.csv                 # Educational attainment data
│   ├── PovertyReport.csv            # Poverty rate data
│   ├── DECENNIALDHC2020...csv       # Census demographic / race data
│   ├── ACSDP5Y2023.DP03-...csv      # ACS DP03: income, labor force, manufacturing
│   └── ACSDP5Y2023.DP05-...csv      # ACS DP05: median age
│
├── derived-data/                    # Cleaned datasets produced by preprocessing.qmd
│   ├── Cleaned_DP03_Econ.csv        # Economic features (Income, Labor Force, Manufacturing)
│   └── Cleaned_DP05_Age.csv         # Demographic features (Median Age)
│
├── preprocessing.qmd                # Step 1 — Data cleaning & feature engineering
├── vis_and_forecast.qmd             # Step 2 — Full analysis: ML, diagnostics, all visualizations
├── writeup_final.qmd                # Step 3 — 3-page project writeup (renders to HTML & PDF)
│
├── app.py                           # Streamlit dashboard application
└── README.md                        # Project documentation
```

---

## 🗂️ QMD Notebooks Guide

This project uses three Quarto (`.qmd`) notebooks that should be run **in order**:

### 1. `preprocessing.qmd` — Data Cleaning & Feature Engineering
Loads and cleans all raw Census datasets. Key tasks:
- Cleans ACS DP05 (Median Age) and DP03 (Median Income, Labor Force %, Manufacturing %) from their transposed wide-format CSV structure.
- Standardizes FIPS codes to the 5-digit Illinois format (`17XXX`).
- Outputs `Cleaned_DP03_Econ.csv` and `Cleaned_DP05_Age.csv` to `derived-data/`.

**Must be run before** `vis_and_forecast.qmd` and `app.py`.

### 2. `vis_and_forecast.qmd` — Analysis & Visualizations
The main analysis notebook. Covers:
- **Part 1**: Data merging across all sources
- **Part 2**: Linear Regression + residual computation
- **Part 3**: Core visualizations (boxplot, scatter, heatmap, leaderboard, choropleth maps)
- **Part 4**: Spatial residual maps (statewide + Chicago zoom)
- **Part 5**: OLS statistical inference table + Q-Q plot
- **Part 6**: Temporal recovery analysis (2020 → 2023)
- **Part 7**: Single-variable scatter subplots + poverty quantile analysis
- **Part 8**: K-Means economic typology map (k=4)
- **Part 9**: Outlier county deep dive — Top 5 / Bottom 5 residuals with radar profile chart

### 3. `writeup_final.qmd` — Project Writeup
A concise 3-page writeup covering research question, data & approach, 3 static plots (matplotlib), and Streamlit app description. Renders to both HTML and PDF.

To render:
```bash
quarto render writeup_final.qmd --to html
quarto render writeup_final.qmd --to pdf
```
> PDF rendering requires LaTeX. If not installed: `quarto install tinytex`

---

## 🧬 Data Dictionary

The final model uses the following 7 features to predict **2023 County-Level Unemployment Rate**:

| Feature | Description | Unit | Source |
|---|---|---|---|
| `Bach_Pct` | Population with Bachelor's degree or higher | % | ACS DP02 |
| `Poverty_Pct` | Population below poverty line | % | SAIPE |
| `Black_Pct` | African American population share | % | Decennial Census 2020 |
| `Median_Income` | Median household income | USD | ACS DP03 |
| `Labor_Force_Pct` | Labor force participation rate | % | ACS DP03 |
| `Manufacturing_Pct` | Employed population in manufacturing | % | ACS DP03 |
| `Median_Age` | Median age of population | Years | ACS DP05 |

---

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Install Python dependencies
```bash
pip install pandas numpy streamlit plotly scikit-learn statsmodels scipy matplotlib requests
```

### 3. Run preprocessing (required first)
```bash
quarto render preprocessing.qmd
```

### 4. Run the Streamlit dashboard
```bash
streamlit run app.py
```
The dashboard will open automatically at `http://localhost:8501`.

**Live demo**: [https://winter-2026-data-visualization-final-project-kyzeqkck3cwlbiztg.streamlit.app](https://winter-2026-data-visualization-final-project-kyzeqkck3cwlbiztg.streamlit.app)

---

## 🖥️ Dashboard Tabs

| Tab | Description |
|---|---|
| 📍 **County Profile** | Select any county and compare its 7 indicators against the Illinois state average. Metrics are split by scale (% vs. absolute) to avoid distortion. |
| 📉 **Regional Trends & ML** | OLS coefficient table (color-coded by direction), Random Forest feature importance chart, and Top 10 over/under-predicted county residual leaderboard. |
| 🔮 **AI Scenario Simulator** | 7 policy-lever sliders with a real-time gauge chart. Test interventions like "If median income increases by $10,000, how much does unemployment drop?" |
| 🗺️ **Spatial Intelligence Map** | Choropleth layers (Actual Unemployment, Residuals, Income, Manufacturing) with statewide and Chicago metro dual-view. Selected county highlighted dynamically. |

---

## ✍️ Author & Acknowledgments

**Mingyang Li** | Tuesday/Thursday Sec 4 | GitHub: `MingyangL28`

Data Sources: U.S. Census Bureau (ACS 5-Year Estimates, Decennial Census), Illinois Department of Labor.
Built with Streamlit; interactive mapping powered by Plotly Mapbox.
