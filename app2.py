import os
import re
import glob
import pandas as pd
import plotly.express as px
import streamlit as st

# =========================
# App config
# =========================
st.set_page_config(page_title="State Unemployment Dashboard", layout="wide")

st.title("US State Unemployment Rate Dashboard")
st.caption("Monthly, seasonally adjusted. Data source: Downloaded FRED state series (e.g., CAUR, TXUR).")

# =========================
# Settings (editable)
# =========================
DEFAULT_DATA_DIR = "state_ur"
DEFAULT_COLOR_SCALE = "Turbo"  # High-contrast
COLOR_SCALE_OPTIONS = ["Turbo", "Plasma", "Viridis", "Inferno", "Cividis", "RdBu_r"]

# =========================
# Helpers: load & reshape
# =========================
def extract_series_code_from_filename(fp: str) -> str | None:
    base = os.path.basename(fp).upper()
    m = re.search(r"\b([A-Z]{2}UR)\b", base)
    return m.group(1) if m else None

def read_one_state_csv(fp: str) -> pd.DataFrame:
    df = pd.read_csv(fp)

    if "observation_date" not in df.columns:
        for cand in ["DATE", "date", "Date"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "observation_date"})
                break
    if "observation_date" not in df.columns:
        raise ValueError(f"Missing date column in {fp} (expected 'observation_date' or 'DATE').")

    series = extract_series_code_from_filename(fp)
    if series is None:
        for c in df.columns:
            cu = str(c).upper()
            if re.fullmatch(r"[A-Z]{2}UR", cu):
                series = cu
                break

    df.columns = [c.upper() for c in df.columns]
    if series is None:
        raise ValueError(f"Cannot identify series code (e.g., CAUR) from {fp}.")
    if series not in df.columns:
        raise ValueError(f"Column '{series}' not found in {fp}.")

    out = df[["OBSERVATION_DATE", series]].copy()
    out["OBSERVATION_DATE"] = pd.to_datetime(out["OBSERVATION_DATE"])
    out = out.rename(columns={series: "unrate"})
    out["state"] = series[:2]
    return out

@st.cache_data(show_spinner=False)
def load_all_states(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}. Check the folder/path.")

    dfs = []
    skipped = []
    for fp in files:
        try:
            dfs.append(read_one_state_csv(fp))
        except Exception as e:
            skipped.append((os.path.basename(fp), str(e)))

    if not dfs:
        msg = "\n".join([f"- {f}: {err}" for f, err in skipped])
        raise RuntimeError(f"All files failed to load.\n{msg}")

    df_long = pd.concat(dfs, ignore_index=True)
    df_long["unrate"] = pd.to_numeric(df_long["unrate"], errors="coerce")
    df_long = df_long.dropna(subset=["unrate"])

    df_long["month"] = df_long["OBSERVATION_DATE"].dt.to_period("M").astype(str)
    return df_long

def compute_fixed_range(df_long: pd.DataFrame, q_low=0.02, q_high=0.98):
    return float(df_long["unrate"].quantile(q_low)), float(df_long["unrate"].quantile(q_high))

# =========================
# Sidebar controls
# =========================
st.sidebar.header("Controls")

data_dir = st.sidebar.text_input("Data folder", value=DEFAULT_DATA_DIR)
color_scale = st.sidebar.selectbox("Color scale", COLOR_SCALE_OPTIONS, index=COLOR_SCALE_OPTIONS.index(DEFAULT_COLOR_SCALE))
use_fixed_range = st.sidebar.checkbox("Use fixed color range (recommended)", value=True)

mode = st.sidebar.radio("View", ["Single-month map", "Animated map", "State time series"], index=0)

# =========================
# Load data
# =========================
try:
    df_long = load_all_states(data_dir)
except Exception as e:
    st.error(str(e))
    st.stop()

states = sorted(df_long["state"].unique())
months = sorted(df_long["month"].unique())

st.sidebar.markdown("---")
st.sidebar.write(f"**States loaded:** {len(states)}")
st.sidebar.write(f"**Months:** {months[0]} → {months[-1]}")

if use_fixed_range:
    cmin, cmax = compute_fixed_range(df_long)
else:
    cmin, cmax = None, None

# =========================
# Main layouts
# =========================
colA, colB = st.columns([2, 1], gap="large")

with colB:
    st.subheader("Quick stats")
    st.metric("States", len(states))
    st.metric("Date range", f"{months[0]} to {months[-1]}")
    st.write("Loaded states:", ", ".join(states))

# =========================
# Mode: Single-month map
# =========================
if mode == "Single-month map":
    with colA:
        st.subheader("Single-month choropleth")

        month_sel = st.select_slider("Select month", options=months, value=months[-1])

        d = df_long[df_long["month"] == month_sel].copy()
        if d.empty:
            st.warning(f"No data for {month_sel}")
            st.stop()

        fig = px.choropleth(
            d,
            locations="state",
            locationmode="USA-states",
            color="unrate",
            scope="usa",
            hover_name="state",
            hover_data={"unrate": ":.1f", "month": True},
            color_continuous_scale=color_scale,
            range_color=(cmin, cmax) if use_fixed_range else None,
            title=f"State Unemployment Rate ({month_sel})"
        )
        fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show data table"):
            st.dataframe(d.sort_values("unrate", ascending=False), use_container_width=True)

# =========================
# Mode: Animated map (Optimized for performance)
# =========================
elif mode == "Animated map":
    with colA:
        st.subheader("Animated choropleth (Annual Average)")

        # 核心优化点：提取年份，并计算每个州每年的平均失业率
        d_anim = df_long.copy()
        d_anim["year"] = d_anim["OBSERVATION_DATE"].dt.year
        d_anim = d_anim.groupby(["state", "year"], as_index=False)["unrate"].mean()

        fig = px.choropleth(
            d_anim,
            locations="state",
            locationmode="USA-states",
            color="unrate",
            animation_frame="year", # 改为按年播放动画
            scope="usa",
            hover_name="state",
            hover_data={"unrate": ":.1f"},
            color_continuous_scale=color_scale,
            range_color=(cmin, cmax) if use_fixed_range else None,
            title="State Unemployment Rate (Annual Average Animation)"
        )
        fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.success("✅ Performance Optimized: The animation now displays the **annual average** instead of monthly data to ensure smooth performance across 50 years of data.")

# =========================
# Mode: Time series
# =========================
else:
    with colA:
        st.subheader("State time series")

        sel_states = st.multiselect("Select states", options=states, default=states[:3] if len(states) >= 3 else states)
        if not sel_states:
            st.warning("Please select at least one state.")
            st.stop()

        # Optional time filter
        start_month, end_month = st.select_slider(
            "Select month range",
            options=months,
            value=(months[0], months[-1])
        )

        d = df_long[df_long["state"].isin(sel_states)].copy()
        d = d[(d["month"] >= start_month) & (d["month"] <= end_month)]
        d = d.sort_values("OBSERVATION_DATE")

        fig = px.line(
            d,
            x="OBSERVATION_DATE",
            y="unrate",
            color="state",
            title="Unemployment Rate Over Time",
            labels={"OBSERVATION_DATE": "Date", "unrate": "Unemployment rate (%)", "state": "State"}
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Download filtered data as CSV"):
            csv_bytes = d.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv_bytes,
                file_name="state_unemployment_filtered.csv",
                mime="text/csv"
            )