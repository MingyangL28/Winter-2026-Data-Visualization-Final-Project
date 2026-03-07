import os
import json
import requests
import copy
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# =========================
# 1. Page Configuration
# =========================
st.set_page_config(page_title="Illinois Economic Intelligence", layout="wide")
st.title("🏛️ Illinois County Economic Intelligence Dashboard")
st.caption("Powered by Machine Learning (Random Forest) and Econometric Inference (OLS) across 7 Socioeconomic Features.")
st.markdown("---")

# =========================
# 2. Helper Functions
# =========================
def clean_fips(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.replace(",", "", regex=False).str.replace(r"\.0$", "", regex=True)
    return s.str.zfill(5)

def make_merge_name(name_series: pd.Series) -> pd.Series:
    s = name_series.astype(str)
    s = s.str.replace(" County, IL", "", regex=False).str.replace(" County, Illinois", "", regex=False).str.replace(" County", "", regex=False)
    s = s.str.lower().str.replace(" ", "", regex=False).str.replace(".", "", regex=False).str.replace("'", "", regex=False).str.replace("-", "", regex=False)
    return s

def ensure_il_fips5(df: pd.DataFrame, col="FIPS") -> pd.DataFrame:
    s = df[col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    lens = s.str.len().value_counts().to_dict()
    threeish = sum(v for k, v in lens.items() if k <= 3)
    fiveish = lens.get(5, 0)

    if fiveish == 0 and threeish > 0:
        df[col] = "17" + s.str.zfill(3)
        return df
    df[col] = s.str.zfill(5)
    return df

# =========================
# 3. GeoJSON Loader
# =========================
@st.cache_data
def load_il_county_geojson():
    url = 'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json'
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        counties = r.json()
        il_features = [f for f in counties['features'] if f['id'].startswith('17')]
        return {'type': 'FeatureCollection', 'features': il_features}
    except Exception as e:
        st.warning(f"⚠️ GeoJSON download failed (Check internet connection): {str(e)}")
        return None

# =========================
# 4. Data Loading & Merging
# =========================
@st.cache_data
def load_all_data(debug=False):
    try:
        # Load Raw Data
        df_main = pd.read_csv("raw-data/UnemploymentReport.csv")
        df_main.columns = df_main.columns.str.strip()
        fips_col = [c for c in df_main.columns if "FIPS" in c.upper()][0]
        df_main = df_main.rename(columns={fips_col: "FIPS"})
        df_main["FIPS"] = clean_fips(df_main["FIPS"])
        df_main = ensure_il_fips5(df_main, "FIPS")
        df_main["Merge_Name"] = make_merge_name(df_main["Name"])

        df_bach = pd.read_csv("raw-data/bachelor.csv")
        fips_col = [c for c in df_bach.columns if "FIPS" in c.upper()][0]
        df_bach = df_bach[[fips_col, "2019-2023"]].rename(columns={fips_col: "FIPS", "2019-2023": "Bach_Pct"})
        df_bach["FIPS"] = clean_fips(df_bach["FIPS"])
        df_bach = ensure_il_fips5(df_bach, "FIPS")
        df_bach["Bach_Pct"] = pd.to_numeric(df_bach["Bach_Pct"].astype(str).str.replace("%", ""), errors="coerce")

        df_pov_raw = pd.read_csv("raw-data/PovertyReport.csv", skiprows=2)
        fips_col = [c for c in df_pov_raw.columns if "FIPS" in c.upper() or "FIP" in c.upper()][0]
        pov_col = [c for c in df_pov_raw.columns if ("POVERTY" in c.upper()) and (("PCT" in c.upper()) or ("PERCENT" in c.upper()))]
        if not pov_col: pov_col = [c for c in df_pov_raw.columns if ("PCT" in c.upper()) or ("PERCENT" in c.upper())][:1]
        df_pov = df_pov_raw[[fips_col, pov_col[0]]].rename(columns={fips_col: "FIPS", pov_col[0]: "Poverty_Pct"})
        df_pov["FIPS"] = clean_fips(df_pov["FIPS"])
        df_pov = ensure_il_fips5(df_pov, "FIPS")
        df_pov["Poverty_Pct"] = pd.to_numeric(df_pov["Poverty_Pct"], errors="coerce")

        df_race_raw = pd.read_csv("raw-data/DECENNIALDHC2020.P9-2026-03-04T220749.csv")
        df_race_t = df_race_raw.transpose()
        df_race_t.columns = df_race_t.iloc[0]
        df_race_t = df_race_t.drop(df_race_t.index[0])
        total_col = [c for c in df_race_t.columns if "Total:" in str(c)][0]
        black_col = [c for c in df_race_t.columns if "Black or African American alone" in str(c)][0]
        df_race_t["Black_Pct"] = (pd.to_numeric(df_race_t[black_col].astype(str).str.replace(",", ""), errors="coerce") / 
                                  pd.to_numeric(df_race_t[total_col].astype(str).str.replace(",", ""), errors="coerce")) * 100
        df_race_t["Merge_Name"] = make_merge_name(pd.Series(df_race_t.index)).values

        # Load Derived Data (Census DP03 & DP05)
        df_econ = pd.read_csv("derived-data/Cleaned_DP03_Econ.csv")
        df_age = pd.read_csv("derived-data/Cleaned_DP05_Age.csv")
        df_census = pd.merge(df_econ, df_age, on="Merge_Name", how="inner")

        # Execute Grand Merge
        m1 = pd.merge(df_main, df_bach, on="FIPS", how="left")
        m2 = pd.merge(m1, df_pov, on="FIPS", how="left")
        m3 = pd.merge(m2, df_race_t[["Merge_Name", "Black_Pct"]], on="Merge_Name", how="left")
        m4 = pd.merge(m3, df_census, on="Merge_Name", how="inner")

        # Impute missing values with column mean to prevent model crashes
        feature_cols_all = ["2020", "Bach_Pct", "Poverty_Pct", "Black_Pct", "Median_Income", "Labor_Force_Pct", "Manufacturing_Pct", "Median_Age", "2023"]
        for c in feature_cols_all:
            m4[c] = pd.to_numeric(m4[c], errors="coerce")
            m4[c] = m4[c].fillna(m4[c].mean())

        return m4.dropna(subset=["2023"])
    except Exception as e:
        st.error(f"❌ Data loading failed: {str(e)}")
        st.stop()

# =========================
# 5. Sidebar & ML Setup
# =========================
st.sidebar.header("⚙️ Controls")
debug = st.sidebar.checkbox("🛠️ Debug Mode", value=False)
st.sidebar.markdown("---")

df = load_all_data(debug=debug)

# Define 7 Core Features
features = ["Bach_Pct", "Poverty_Pct", "Black_Pct", "Median_Income", "Labor_Force_Pct", "Manufacturing_Pct", "Median_Age"]
feature_names = ["Bachelor's Degree %", "Poverty Rate %", "Black Population %", "Median Income ($)", "Labor Force Participation %", "Manufacturing %", "Median Age"]

X = df[features]
y = df["2023"]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
rf_model.fit(X_scaled, y)

# Predictions & Diagnostics
df["Predicted_2023"] = rf_model.predict(X_scaled)
df["Residual"] = df["Predicted_2023"] - df["2023"]
df["Abs_Error"] = df["Residual"].abs()

r2_score = rf_model.score(X_scaled, y)

# Sidebar Overview
st.sidebar.header("📊 Model Overview")
st.sidebar.metric("AI Predictive Power (R²)", f"{r2_score:.1%}")
st.sidebar.write(f"Counties Analyzed: **{len(df)}**")
st.sidebar.markdown("---")
selected_county = st.sidebar.selectbox("🏷️ Select a County to Focus", df["Name"].sort_values().unique())
c_data = df.loc[df["Name"] == selected_county].iloc[0]

def add_selected_outline_mapbox(fig, geojson_obj, featureidkey, df_map, selected_name):
    sel = df_map[df_map["Name"] == selected_name]
    if len(sel) != 1: return fig
    hi = px.choropleth_mapbox(sel, geojson=geojson_obj, locations="FIPS", featureidkey=featureidkey, color_discrete_sequence=["rgba(0,0,0,0)"], center={"lat": 40.0, "lon": -89.0}, zoom=5)
    hi.update_traces(marker_line_width=3.0, marker_line_color="#00FF00") 
    fig.add_trace(hi.data[0])
    return fig

# =========================
# 6. Main Dashboard (Tabs)
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["📍 County Profile", "📉 Regional Trends & ML", "🔮 AI Scenario Simulator", "🗺️ Spatial Intelligence Map"])

with tab1:
    st.subheader(f"📍 Comprehensive Profile: {selected_county}")
    
    avg_data = df[features + ["2023", "Predicted_2023", "Residual"]].mean(numeric_only=True)
    
    # --- 1. Percentage Metrics Chart ---
    pct_features = ["Bach_Pct", "Poverty_Pct", "Black_Pct", "Labor_Force_Pct", "Manufacturing_Pct", "2023", "Predicted_2023"]
    pct_names = ["Bachelor's Degree %", "Poverty Rate %", "Black Population %", "Labor Force %", "Manufacturing %", "Actual Unemployment %", "AI Predicted Unemployment %"]
    
    pct_df = pd.DataFrame({
        "Metric": pct_names,
        "County Value": [c_data[f] for f in pct_features],
        "Illinois Average": [avg_data[f] for f in pct_features]
    })
    
    fig_pct = go.Figure()
    fig_pct.add_trace(go.Bar(name=selected_county, x=pct_df["Metric"], y=pct_df["County Value"], marker_color="#1f77b4"))
    fig_pct.add_trace(go.Bar(name="State Average", x=pct_df["Metric"], y=pct_df["Illinois Average"], marker_color="#d62728"))
    fig_pct.update_layout(barmode="group", height=450, plot_bgcolor="rgba(0,0,0,0)", 
                          title="📊 Core Percentage Metrics Comparison", yaxis_title="Percentage (%)")
    st.plotly_chart(fig_pct, use_container_width=True)

    st.divider()

    # --- 2. Absolute Values (Income & Age) ---
    col_abs1, col_abs2 = st.columns(2)
    
    with col_abs1:
        fig_inc = go.Figure()
        fig_inc.add_trace(go.Bar(name=selected_county, x=["Median Income"], y=[c_data["Median_Income"]], marker_color="#2ca02c", text=[f"${c_data['Median_Income']:,.0f}"], textposition='auto'))
        fig_inc.add_trace(go.Bar(name="State Average", x=["Median Income"], y=[avg_data["Median_Income"]], marker_color="#7f7f7f", text=[f"${avg_data['Median_Income']:,.0f}"], textposition='auto'))
        fig_inc.update_layout(barmode="group", height=350, plot_bgcolor="rgba(0,0,0,0)", 
                              title="💰 Economic Wealth (Median Income)", yaxis_title="USD ($)")
        st.plotly_chart(fig_inc, use_container_width=True)
        
    with col_abs2:
        fig_age = go.Figure()
        fig_age.add_trace(go.Bar(name=selected_county, x=["Median Age"], y=[c_data["Median_Age"]], marker_color="#ff7f0e", text=[f"{c_data['Median_Age']:.1f}"], textposition='auto'))
        fig_age.add_trace(go.Bar(name="State Average", x=["Median Age"], y=[avg_data["Median_Age"]], marker_color="#7f7f7f", text=[f"{avg_data['Median_Age']:.1f}"], textposition='auto'))
        fig_age.update_layout(barmode="group", height=350, plot_bgcolor="rgba(0,0,0,0)", 
                              title="👥 Demographics (Median Age)", yaxis_title="Age (Years)")
        st.plotly_chart(fig_age, use_container_width=True)

with tab2:
    st.markdown("### 📈 ML Diagnostics & Econometric Inference")
    
    # --- OLS Inference ---
    X_unscaled = sm.add_constant(df[features]) 
    ols_model = sm.OLS(df["2023"], X_unscaled).fit()
    
    st.subheader("1. Specific Variable Impacts (via OLS)")
    st.markdown("This shows the specific impact (direction and magnitude) on the unemployment rate when a given variable **increases by 1 unit**, holding all other variables constant.")
    
    coef_df = pd.DataFrame({
        "Feature Variable": ["Intercept (Baseline)"] + feature_names,
        "Coefficient (Impact Size)": ols_model.params.values,
        "P-Value (Significance)": ols_model.pvalues.values
    })
    def interpret_coef(row):
        if row["Feature Variable"] == "Intercept (Baseline)": return "Base starting point for unemployment"
        if row["P-Value (Significance)"] > 0.1: return "⚪ Insignificant (Statistically irrelevant)"
        if row["Coefficient (Impact Size)"] > 0: return "🔴 Increases Unemployment (Worsens)"
        return "🟢 Decreases Unemployment (Improves)"
        
    coef_df["Business Interpretation"] = coef_df.apply(interpret_coef, axis=1)
    
    # BUG FIX: Removed .background_gradient() to fix the Matplotlib ImportError
    st.dataframe(coef_df.style.format({
        "Coefficient (Impact Size)": "{:.4f}",
        "P-Value (Significance)": "{:.4f}"
    }), use_container_width=True)

    st.divider()

    # --- Feature Importance & Scatter ---
    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.subheader("2. AI Feature Importance Ranking")
        importances = rf_model.feature_importances_
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=True)
        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation='h', color="Importance", color_continuous_scale="magma")
        fig_imp.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_imp, use_container_width=True)

    with row1_col2:
        st.subheader("3. Actual vs AI Predicted Unemployment")
        fig_scatter = px.scatter(df, x="2023", y="Predicted_2023", hover_name="Name", color="Abs_Error", color_continuous_scale="Reds", labels={"2023": "Actual 2023 (%)", "Predicted_2023": "AI Predicted (%)"})
        lo, hi = float(df["2023"].min()), float(df["2023"].max())
        fig_scatter.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color="blue", dash="dot"))
        fig_scatter.update_layout(height=350, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # --- Residual Diagnostics ---
    st.subheader("4. Model Prediction Bias Analysis (Residual Diagnostics)")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        top_over = df.sort_values("Residual", ascending=False).head(10)
        fig_over = px.bar(top_over, x="Residual", y="Name", orientation="h", title="🔴 Top 10 Over-predicted (Model guessed higher than actual)")
        fig_over.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_over, use_container_width=True)

    with col_b2:
        top_under = df.sort_values("Residual", ascending=True).head(10)
        fig_under = px.bar(top_under, x="Residual", y="Name", orientation="h", title="🟢 Top 10 Under-predicted (Model guessed lower than actual)")
        fig_under.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_under, use_container_width=True)
        
    fig_hist = px.histogram(df, x="Residual", nbins=30, title="Statewide Residual Distribution")
    fig_hist.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_hist, use_container_width=True)

with tab3:
    st.header("🔮 AI Scenario Simulator")
    st.caption("Adjust the socioeconomic levers below to see how the Random Forest model re-predicts the unemployment rate for the selected county.")
    
    col_sliders, col_gauge = st.columns([2, 1])
    with col_sliders:
        c1, c2, c3 = st.columns(3)
        v_bach = c1.slider(feature_names[0], 0.0, 80.0, float(c_data["Bach_Pct"]), step=1.0)
        v_pov = c2.slider(feature_names[1], 0.0, 40.0, float(c_data["Poverty_Pct"]), step=1.0)
        v_black = c3.slider(feature_names[2], 0.0, 100.0, float(c_data["Black_Pct"]), step=1.0)
        
        v_inc = c1.slider(feature_names[3], 20000.0, 150000.0, float(c_data["Median_Income"]), step=1000.0)
        v_lfpr = c2.slider(feature_names[4], 20.0, 90.0, float(c_data["Labor_Force_Pct"]), step=1.0)
        v_mfg = c3.slider(feature_names[5], 0.0, 40.0, float(c_data["Manufacturing_Pct"]), step=1.0)
        
        v_age = c1.slider(feature_names[6], 20.0, 60.0, float(c_data["Median_Age"]), step=0.5)
        
        input_df = pd.DataFrame([[v_bach, v_pov, v_black, v_inc, v_lfpr, v_mfg, v_age]], columns=features)
        pred = float(rf_model.predict(scaler.transform(input_df))[0])

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta", value = pred, domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Simulated Unemployment<br><span style='font-size:0.8em;color:gray'>vs Actual for {selected_county} ({float(c_data['2023']):.2f}%)</span>"},
            delta = {'reference': float(c_data['2023']), 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, max(15, pred+2)], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white", 'borderwidth': 2, 'bordercolor': "gray",
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': df['2023'].mean()} 
            }
        ))
        fig_gauge.update_layout(height=400, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

with tab4:
    st.subheader("🗺️ Spatial Intelligence Map (Mapbox Edition)")
    
    raw_geojson = load_il_county_geojson()
    if raw_geojson is None:
        st.warning("Missing GeoJSON boundary file. This map requires an active internet connection.")
        st.stop()
    il_geojson = copy.deepcopy(raw_geojson)

    df_map = df.copy()
    featureidkey = "id" 
    hover_cols = {"FIPS": False, "2023": ":.2f", "Predicted_2023": ":.2f", "Residual": ":.2f", "Median_Income": ":.0f", "Manufacturing_Pct": ":.1f"}

    with st.container():
        colA, colB, colC = st.columns(3)
        with colA:
            metric_mode = st.selectbox("Map Data Layer", ["Model Residual (Predicted - Actual)", "2023 Actual Unemployment", "Median Income", "Manufacturing %"], index=0)
        with colB:
            map_style = st.selectbox("Map Theme", ["carto-darkmatter", "carto-positron", "open-street-map"], index=0)

    if metric_mode == "Model Residual (Predicted - Actual)":
        color_col = "Residual"
        color_scale = "RdBu_r"
    elif metric_mode == "2023 Actual Unemployment":
        color_col = "2023"
        color_scale = "Reds"
    elif metric_mode == "Median Income":
        color_col = "Median_Income"
        color_scale = "Greens"
    else:
        color_col = "Manufacturing_Pct"
        color_scale = "Purples"

    map_col1, map_col2 = st.columns(2)
    with map_col1:
        fig_state = px.choropleth_mapbox(
            df_map, geojson=il_geojson, locations="FIPS", featureidkey=featureidkey,
            color=color_col, hover_name="Name", hover_data=hover_cols,
            color_continuous_scale=color_scale, mapbox_style=map_style, zoom=5.4, center={"lat": 40.0, "lon": -89.2}, opacity=0.8
        )
        if color_scale == "RdBu_r": fig_state.update_layout(coloraxis=dict(cmid=0))
        fig_state.update_layout(height=650, margin=dict(l=0, r=0, t=30, b=0), title="Illinois Statewide Overview")
        fig_state = add_selected_outline_mapbox(fig_state, il_geojson, featureidkey, df_map, selected_county)
        st.plotly_chart(fig_state, use_container_width=True)

    with map_col2:
        zoom_df = df_map[df_map["FIPS"].isin(["17031","17043","17089","17093","17097","17111","17197"])]
        if len(zoom_df) == 0: zoom_df = df_map.copy()
        fig_zoom = px.choropleth_mapbox(
            zoom_df, geojson=il_geojson, locations="FIPS", featureidkey=featureidkey,
            color=color_col, hover_name="Name", hover_data=hover_cols,
            color_continuous_scale=color_scale, mapbox_style=map_style, zoom=8.0, center={"lat": 41.8, "lon": -87.8}, opacity=0.8
        )
        if color_scale == "RdBu_r": fig_zoom.update_layout(coloraxis=dict(cmid=0))
        fig_zoom.update_layout(height=650, margin=dict(l=0, r=0, t=30, b=0), title="Chicago Metro Zoomed View")
        fig_zoom = add_selected_outline_mapbox(fig_zoom, il_geojson, featureidkey, zoom_df, selected_county)
        st.plotly_chart(fig_zoom, use_container_width=True)