import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Rychfire | Fire Activity Calendar", page_icon="🔥", layout="wide")
st.title("🔥 Rychfire")
st.caption("Satellite Fire Activity Calendar · MVP v0.1")
st.warning("DEMO DATA ONLY — all detections and indicators below are synthetic. No NASA observations or sensor harmonization are used yet.")

REGIONS = {
    "Argentina · Córdoba (demo)": (-31.4, -64.2),
    "Australia · New South Wales (demo)": (-33.9, 151.2),
    "USA · California (demo)": (36.8, -119.4),
}
region = st.sidebar.selectbox("Demo region", list(REGIONS))
sensor = st.sidebar.selectbox("Satellite product (simulated)", ["MODIS", "VIIRS", "Combined (not harmonized)"])
year_range = st.sidebar.slider("Years", 2012, 2025, (2016, 2025))
st.sidebar.info("The satellite selector changes simulated data only. Combined counts are NOT harmonized.")

lat, lon = REGIONS[region]
seed = list(REGIONS).index(region) * 17 + [0, 1, 2][["MODIS", "VIIRS", "Combined (not harmonized)"].index(sensor)]
rng = np.random.default_rng(2026 + seed)
rows = []
for year in range(2012, 2026):
    for month in range(1, 13):
        season = 1.2 + 0.8 * np.cos(2 * np.pi * (month - 8) / 12)
        if "Australia" in region:
            season = 1.2 + 0.8 * np.cos(2 * np.pi * (month - 1) / 12)
        factor = {"MODIS": 1, "VIIRS": 1.6, "Combined (not harmonized)": 2.6}[sensor]
        count = int(rng.poisson(max(2, 40 * season * factor * (1 + 0.02 * (year - 2012)))))
        rows.append({"year": year, "month": month, "detections": count})
df = pd.DataFrame(rows)
df = df[df.year.between(*year_range)].copy()
df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))

monthly_baseline = df.groupby("month")["detections"].median()
latest = df.sort_values("date").iloc[-1]
base = float(monthly_baseline.loc[int(latest.month)])
anomaly = (float(latest.detections) / base - 1) * 100 if base else float("nan")
peak = int(df.groupby("month")["detections"].median().idxmax())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Simulated detections", f"{df.detections.sum():,}")
m2.metric("Latest month", f"{int(latest.detections):,}")
m3.metric("vs. monthly median", f"{anomaly:+.1f}%")
m4.metric("Peak month (median)", pd.Timestamp(2025, peak, 1).strftime("%B"))

left, right = st.columns([1, 1.25])
with left:
    st.subheader("Demo hotspot map")
    map_rng = np.random.default_rng(77 + seed)
    points = pd.DataFrame({
        "lat": lat + map_rng.normal(0, 0.7, 180),
        "lon": lon + map_rng.normal(0, 0.7, 180),
    })
    st.map(points, latitude="lat", longitude="lon", zoom=6)
    st.caption("Illustrative points; not actual fire detections. Region selection is preset in v0.1.")
with right:
    st.subheader("Burning Activity Calendar")
    grid = df.pivot(index="year", columns="month", values="detections")
    grid = grid.reindex(columns=range(1, 13))
    fig = px.imshow(grid, labels=dict(x="Month", y="Year", color="Detections"),
                    x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
                    aspect="auto", color_continuous_scale="YlOrRd")
    fig.update_layout(height=410, margin=dict(l=10, r=10, t=15, b=10))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Historical Activity")
trend = px.line(df, x="date", y="detections", markers=True,
                labels={"date": "Month", "detections": "Simulated hotspot detections"})
trend.update_layout(height=300, margin=dict(l=10, r=10, t=15, b=10))
st.plotly_chart(trend, use_container_width=True)

st.download_button("⬇️ Download demo monthly data (CSV)", data=df.to_csv(index=False).encode("utf-8"),
                   file_name="rychfire_demo_monthly.csv", mime="text/csv")
st.caption("Roadmap: real NASA FIRMS ingestion → spatial AOI selection → validated MODIS/VIIRS harmonization → anomaly confidence indicators.")
