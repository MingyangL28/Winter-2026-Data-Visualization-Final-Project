# 🏛️ Illinois County Economic Intelligence Dashboard

This is an interactive spatial data analysis dashboard built with Python and Streamlit. This project innovatively combines the **high predictive accuracy of Machine Learning (Random Forest)** with the **high interpretability of traditional Econometrics (OLS Linear Regression)**. It provides an in-depth diagnostic, spatial analysis, and predictive sandbox simulation of the unemployment rates across all 102 counties in Illinois based on 7 core socioeconomic and demographic features.

---

## ✨ Key Features

- 🧠 **Dual-Model Architecture**:
  - **Random Forest**: Handles complex, non-linear relationships to provide highly accurate unemployment predictions and feature importance rankings.
  - **OLS (Ordinary Least Squares)**: Breaks the "black box" of machine learning by providing rigorous econometric inference (Coefficients and P-Values), explaining the specific positive or negative impact of each variable.
- 🔮 **AI Scenario Simulator (What-If Analysis)**: Allows users to dynamically adjust 7 socioeconomic levers (e.g., increasing Bachelor's degree rates, lowering poverty) and observe in real-time how the model re-predicts the county's unemployment rate.
- 🗺️ **Spatial Intelligence Maps**: Features interactive Choropleth maps built with Plotly Mapbox, displaying both statewide and Chicago metropolitan views. It visually maps out economic metrics and **model residuals (prediction biases)** across geographic space.
- 📊 **Dimensional Metric Splitting**: Intelligently separates percentage metrics (0-100%) from absolute values (USD/Years) in the county profiles, avoiding common data visualization pitfalls associated with scale imbalances.

---

## 📂 Project Structure

```text
├── raw-data/                  # Raw, unprocessed CSV datasets
│   ├── UnemploymentReport.csv # Main table: 2023 Unemployment by county
│   ├── bachelor.csv           # Educational attainment data
│   ├── PovertyReport.csv      # Poverty rate data
│   └── DECENNIALDHC2020...csv # Census demographic/race data
├── derived-data/              # Cleaned datasets with extracted advanced features
│   ├── Cleaned_DP03_Econ.csv  # Economic features (Income, Labor Force, Manufacturing)
│   └── Cleaned_DP05_Age.csv   # Demographic features (Median Age)
├── app.py                     # Main Streamlit application (UI, data merging, ML logic)
└── README.md                  # Project documentation
```
🧬 Data Dictionary
The final model utilizes the following 7 core features to predict the 2023 Unemployment Rate (Target):

Bach_Pct: Percentage of population with a Bachelor's degree or higher (%)

Poverty_Pct: Percentage of population living below the poverty line (%)

Black_Pct: Percentage of the African American population (%)

Median_Income: Median household income (USD)

Labor_Force_Pct: Labor force participation rate (%)

Manufacturing_Pct: Percentage of the employed population in manufacturing (%)

Median_Age: Median age of the population (Years)

🚀 Installation & Setup
1. Clone or download this repository to your local machine.

2. Install Dependencies: Ensure you have the required Python libraries installed in your environment:

```{python}
pip install pandas numpy streamlit plotly scikit-learn statsmodels requests
```
3. Run the Dashboard: Navigate to the project root directory in your terminal/command prompt and run:

```{python}
streamlit run app.py
```
(The dashboard will automatically open in your default web browser at http://localhost:8501/)

🖥️ Dashboard Tabs Breakdown
📍 County Profile

Select a specific county to visualize its 7 economic features against the "Illinois State Average" using properly scaled bar charts.

📉 Regional Trends & ML

Displays the OLS econometric interpretation table (coefficients & significance).

Shows the AI feature importance ranking via a horizontal bar chart.

Analyzes model residual distributions and lists the Top 10 over-predicted and under-predicted counties.

🔮 AI Scenario Simulator

Features 7 dynamic sliders and a real-time gauge chart to test policy interventions (e.g., "If median income increases by $10,000, how much will unemployment drop?").

🗺️ Spatial Intelligence Map

Map different data layers (Actual Unemployment, Model Residuals, Income, Manufacturing) onto the geographic space. Supports full statewide overview and deep zoom into the Chicago metro area.

✍️ Author & Acknowledgments
Data Sources: U.S. Census Bureau (ACS 5-Year Data), Illinois Department of Labor, and open government datasets.

Built with the Streamlit framework; interactive mapping powered by Plotly.
