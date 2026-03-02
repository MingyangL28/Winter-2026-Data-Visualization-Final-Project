# 📈 Macroeconomic Unemployment Prediction Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]
## 📌 Project Overview
This project explores the impact of macroeconomic indicators on the U.S. Unemployment Rate (UNRATE). By leveraging a **Gamma Generalized Linear Model (GLM)** and **SHAP (Shapley Additive exPlanations)** attribution analysis, this interactive dashboard predicts future unemployment trends and dynamically explains the economic intuition behind each predictor.

**👉 [Click here to view the Live Dashboard!]**

## ✨ Key Features
* **Interactive SHAP Explainer:** Select different economic indicators (e.g., Term Spread, CPI, Industrial Production) to see how they positively or negatively drive the unemployment rate.
* **Global Feature Importance:** A dynamic visualization of which macroeconomic variables contribute the most to the model's predictions.
* **Real-time Prediction:** Forecasts the next period's unemployment rate based on the latest available macroeconomic data.
* **Economic Intuition:** Built-in explanations mapping model coefficients to real-world economic theories (like the Phillips Curve and Yield Curve Inversions).

## 🛠️ Methodology & Data
* **Data Source:** Federal Reserve Economic Data (FRED).
* **Predictors:** * Term Spread (10Y - 3M Treasury Yields) & its lags.
  * Industrial Production Growth & Momentum.
  * CPI Year-over-Year (Inflation).
  * Lagged Short-term and Long-term Interest Rates.
* **Model:** Gamma GLM (Generalized Linear Model), chosen for its suitability with strictly positive, right-skewed continuous variables.

## 🚀 How to Run Locally

If you'd like to run this project on your local machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MingyangL28/Winter-2026-Data-Visualization-Final-Project.git
