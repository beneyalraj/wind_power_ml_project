"""
src/serving/streamlit_app.py — Wind Power Prediction Dashboard

HOW IT WORKS:
  - Reads API_URL from environment (defaults to localhost for local dev)
  - In Docker, API_URL = http://fastapi:8000 via docker-compose
  - Never touches the model directly — all calls go through FastAPI

RUN LOCALLY:
  streamlit run src/serving/streamlit_app.py

RUN IN DOCKER:
  Handled by docker-compose — API_URL injected automatically
"""

import os
import requests
import pandas as pd
import numpy as np
import streamlit as st

# -----------------------------
# Configuration
# -----------------------------
API_URL = os.getenv("API_URL", "http://localhost:8000")

# -----------------------------
# Page Config — must be first Streamlit call
# -----------------------------
st.set_page_config(
    page_title="Wind Power Predictor",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# CSS
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #f4f6f9;
    color: #1a2332;
}
.stApp { background-color: #f4f6f9; }
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 1.8rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #0f4c75 0%, #1b6ca8 60%, #2980b9 100%);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1rem;
    color: white;
}
.app-title {
    font-size: 1.9rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.02em;
}
.app-subtitle {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.7);
    margin-top: 0.3rem;
    font-weight: 300;
}

/* ── Status Strip ── */
.status-strip {
    display: flex;
    align-items: center;
    gap: 2rem;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.75rem 1.25rem;
    margin-bottom: 1.5rem;
    font-size: 0.82rem;
    color: #4a5568;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.status-online  { color: #16a34a; font-weight: 600; }
.status-offline { color: #dc2626; font-weight: 600; }
.status-label   { color: #94a3b8; margin-right: 4px; }
.status-val     {
    color: #1a2332; font-weight: 500;
    font-family: 'DM Mono', monospace; font-size: 0.78rem;
}
.status-version {
    background: #eff6ff;
    color: #1d4ed8;
    border: 1px solid #bfdbfe;
    border-radius: 4px;
    padding: 2px 10px;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
}

/* ── Cards ── */
.card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.card-title {
    font-size: 0.7rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1.2rem;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid #f1f5f9;
}

/* ── Metric Cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 1rem;
}
.metric-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 1.1rem;
    text-align: center;
}
.metric-box-accent {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 1.1rem;
    text-align: center;
}
.metric-val {
    font-family: 'DM Mono', monospace;
    font-size: 1.5rem;
    font-weight: 500;
    color: #1d4ed8;
    line-height: 1.2;
}
.metric-val-main {
    font-family: 'DM Mono', monospace;
    font-size: 1.8rem;
    font-weight: 500;
    color: #0f4c75;
    line-height: 1.2;
}
.metric-unit { font-size: 0.78rem; color: #94a3b8; margin-left: 3px; }
.metric-lbl  {
    font-size: 0.7rem; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 0.08em;
    margin-top: 0.25rem; font-weight: 500;
}

/* ── Request ID trace bar ── */
.trace-bar {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    margin-top: 0.8rem;
    display: flex;
    gap: 2rem;
    align-items: center;
}
.trace-label {
    font-size: 0.65rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.trace-value {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #475569;
}

/* ── Awaiting placeholder ── */
.awaiting {
    height: 160px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f8fafc;
    border: 1.5px dashed #cbd5e1;
    border-radius: 8px;
    color: #94a3b8;
    font-size: 0.85rem;
}

/* ── Stat row ── */
.stat-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-top: 1rem;
}
.stat-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.7rem 0.8rem;
    text-align: center;
}
.stat-val {
    font-family: 'DM Mono', monospace;
    font-size: 0.95rem;
    color: #1d4ed8;
    font-weight: 500;
}
.stat-lbl {
    font-size: 0.65rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 3px;
}

/* ── System info table ── */
.sysinfo-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.83rem;
}
.sysinfo-table th {
    text-align: left;
    font-size: 0.65rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 0 0 0.5rem 0;
    border-bottom: 1px solid #f1f5f9;
}
.sysinfo-table td {
    padding: 0.45rem 0;
    color: #475569;
    border-bottom: 1px solid #f8fafc;
    vertical-align: top;
}
.sysinfo-table td:first-child {
    color: #94a3b8;
    font-size: 0.78rem;
    width: 45%;
    padding-right: 1rem;
}
.sysinfo-table td:last-child {
    font-weight: 500;
    color: #1a2332;
}
.perf-pass { color: #16a34a; font-weight: 600; }
.perf-val  {
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    color: #1d4ed8;
}

/* ── Slider labels ── */
.stSlider label, .stSlider p,
.stSlider [data-testid="stMarkdownContainer"] p {
    color: #1a2332 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}
.stNumberInput label, .stNumberInput p {
    color: #1a2332 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
}
.stSlider > div > div > div > div { background: #1d4ed8 !important; }

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0f4c75, #1b6ca8) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1b6ca8, #2980b9) !important;
    box-shadow: 0 4px 12px rgba(15,76,117,0.3) !important;
}
.stButton > button[kind="secondary"] {
    background: #ffffff !important;
    color: #1d4ed8 !important;
    border: 1.5px solid #bfdbfe !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #eff6ff !important;
    border-color: #93c5fd !important;
}

hr { border-color: #e2e8f0 !important; margin: 1.2rem 0 !important; }
.stAlert { border-radius: 6px !important; font-size: 0.82rem !important; }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Helper Functions
# -----------------------------
def check_api_health() -> dict | None:
    try:
        res = requests.get(f"{API_URL}/health", timeout=5)
        if res.status_code == 200:
            return res.json()
    except requests.exceptions.ConnectionError:
        pass
    return None


def predict_single(payload: dict) -> dict | None:
    try:
        res = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if res.status_code == 200:
            return res.json()
        st.error(f"API {res.status_code}: {res.json().get('detail', 'Unknown error')}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Is FastAPI running?")
    except requests.exceptions.Timeout:
        st.error("Request timed out.")
    return None


def predict_batch(payloads: list[dict]) -> list[float] | None:
    try:
        res = requests.post(f"{API_URL}/predict_batch", json=payloads, timeout=20)
        if res.status_code == 200:
            return res.json().get("predictions_kw", [])
        st.error(f"Batch API {res.status_code}: {res.json().get('detail', '')}")
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API.")
    except requests.exceptions.Timeout:
        st.error("Batch request timed out.")
    return None


# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="app-header">
    <div class="app-title">🌬️ Wind Power Prediction</div>
    <div class="app-subtitle">ML system for predicting wind farm power output</div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Status Strip + Refresh Button
# -----------------------------
health = check_api_health()

scol1, scol2 = st.columns([5, 1])

with scol1:
    if health:
        version      = health.get("model_version", "unknown")
        model_loaded = health.get("model_loaded", False)
        st.markdown(f"""
        <div class="status-strip">
            <span class="status-online">● Online</span>
            <span>
                <span class="status-label">Model</span>
                <span class="status-version">{version}</span>
            </span>
            <span>
                <span class="status-label">Status</span>
                <span class="status-val">{'Loaded' if model_loaded else 'Not Loaded'}</span>
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-strip">
            <span class="status-offline">● Offline</span>
            <span style="color:#94a3b8; font-size:0.82rem;">
                Cannot reach API — start FastAPI first
            </span>
        </div>
        """, unsafe_allow_html=True)

with scol2:
    # Refresh button sits next to the status strip — visible and functional
    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    if st.button("↻  Refresh", use_container_width=True, type="secondary"):
        st.rerun()

if not health:
    st.error("Run: `uvicorn src.serving.app:app --reload`")
    st.stop()

# -----------------------------
# Main Layout
# -----------------------------
left, right = st.columns([1, 1.5], gap="large")

# LEFT — Input
with left:
    st.markdown('<div class="card"><div class="card-title">Input Parameters</div>',
                unsafe_allow_html=True)

    wind_speed = st.slider(
        "Wind Speed (m/s)",
        min_value=0.0, max_value=25.0, value=12.5, step=0.5,
        help="Wind speed at hub height. Cut-in ~3 m/s, rated ~12–15 m/s"
    )
    wind_direction = st.slider(
        "Wind Direction (°)",
        min_value=0.0, max_value=360.0, value=180.0, step=5.0,
        help="0° = North · 90° = East · 180° = South · 270° = West"
    )
    turbulence_intensity = st.slider(
        "Turbulence Intensity",
        min_value=0.0, max_value=1.0, value=0.08, step=0.01,
        help="σ(wind) / mean(wind). Low: 0.05 · Typical: 0.08–0.12 · High: >0.15"
    )
    num_turbines = st.number_input(
        "Number of Turbines",
        min_value=1, max_value=500, value=50, step=1,
        help="Total active turbines in the wind farm"
    )

    st.markdown('</div>', unsafe_allow_html=True)
    predict_clicked = st.button(
        "⚡  Predict Power Output",
        use_container_width=True,
        type="primary"
    )

# RIGHT — Output
with right:
    st.markdown('<div class="card"><div class="card-title">Prediction Result</div>',
                unsafe_allow_html=True)

    if predict_clicked:
        payload = {
            "wind_speed":           wind_speed,
            "wind_direction":       wind_direction,
            "turbulence_intensity": turbulence_intensity,
            "num_turbines":         num_turbines
        }
        with st.spinner("Running inference..."):
            result = predict_single(payload)

        if result:
            kw  = result["prediction_kw"]
            mw  = kw / 1000
            per = kw / num_turbines

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-box-accent">
                    <div class="metric-val-main">{kw:,.0f}<span class="metric-unit">kW</span></div>
                    <div class="metric-lbl">Total Output</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{mw:,.1f}<span class="metric-unit">MW</span></div>
                    <div class="metric-lbl">Megawatts</div>
                </div>
                <div class="metric-box">
                    <div class="metric-val">{per:,.0f}<span class="metric-unit">kW</span></div>
                    <div class="metric-lbl">Per Turbine</div>
                </div>
            </div>
            <div class="trace-bar">
                <div>
                    <div class="trace-label">Request ID</div>
                    <div class="trace-value">{result.get('request_id', '—')}</div>
                </div>
                <div>
                    <div class="trace-label">Model</div>
                    <div class="trace-value">{result.get('model_version', '—')}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="awaiting">Set parameters and click Predict</div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Power Curve
# -----------------------------
st.markdown(
    '<div class="card"><div class="card-title">Power Curve Analysis — Batch Inference</div>',
    unsafe_allow_html=True
)
st.markdown(
    "<p style='font-size:0.83rem; color:#64748b; margin-bottom:1rem;'>"
    "Simulates predicted output across the full wind speed envelope (0–25 m/s) "
    "using a single vectorised batch call across 51 data points."
    "</p>",
    unsafe_allow_html=True
)

if st.button("📈  Generate Power Curve", type="secondary"):
    wind_speeds    = np.arange(0.0, 25.5, 0.5).tolist()
    batch_payloads = [
        {
            "wind_speed":           float(ws),
            "wind_direction":       wind_direction,
            "turbulence_intensity": turbulence_intensity,
            "num_turbines":         num_turbines
        }
        for ws in wind_speeds
    ]
    with st.spinner(f"Batch inference — {len(batch_payloads)} records..."):
        predictions = predict_batch(batch_payloads)

    if predictions:
        chart_df = pd.DataFrame({
            "Wind Speed (m/s)":     wind_speeds[:len(predictions)],
            "Predicted Power (kW)": predictions
        }).set_index("Wind Speed (m/s)")

        st.line_chart(chart_df, use_container_width=True, color="#1d4ed8")

        peak    = max(predictions)
        cur_idx = int(wind_speed * 2)
        cur_val = predictions[cur_idx] if cur_idx < len(predictions) else 0
        cut_in  = next(
            (wind_speeds[i] for i, p in enumerate(predictions) if p > 0), "N/A"
        )
        cap_fac = (
            sum(predictions) / (peak * len(predictions)) * 100
        ) if peak > 0 else 0

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-box">
                <div class="stat-val">{peak:,.0f} kW</div>
                <div class="stat-lbl">Peak Output</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{cur_val:,.0f} kW</div>
                <div class="stat-lbl">At {wind_speed} m/s</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{cut_in} m/s</div>
                <div class="stat-lbl">Cut-in Speed</div>
            </div>
            <div class="stat-box">
                <div class="stat-val">{cap_fac:.1f}%</div>
                <div class="stat-lbl">Capacity Factor</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# System Information Expander
# -----------------------------
st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)

with st.expander("ℹ️  System Information"):

    c1, c2, c3 = st.columns(3)

    # --- Column 1: ML Stack ---
    with c1:
        st.markdown("**ML Stack**")
        st.markdown("""
<table class="sysinfo-table">
<tr><th>Component</th><th>Detail</th></tr>
<tr><td>Serving</td><td>FastAPI + Pydantic v2</td></tr>
<tr><td>Registry</td><td>DagHub MLflow</td></tr>
<tr><td>Containerisation</td><td>Docker</td></tr>
<tr><td>Deployment</td><td>AWS</td></tr>
<tr><td>Monitoring</td><td>Evidently AI</td></tr>
<tr><td>CI/CD</td><td>GitHub Actions</td></tr>
</table>
""", unsafe_allow_html=True)

    # --- Column 2: Training Pipeline ---
    with c2:
        st.markdown("**Training Pipeline**")
        st.markdown("""
<table class="sysinfo-table">
<tr><th>Stage</th><th>Description</th></tr>
<tr><td>Ingestion</td><td>Raw HDF5 from AWS S3</td></tr>
<tr><td>Validation</td><td>Schema + quality checks</td></tr>
<tr><td>Extraction</td><td>Scenario-level Parquet</td></tr>
<tr><td>Features</td><td>Physics-informed engineering</td></tr>
<tr><td>Training</td><td>4 models, best selected</td></tr>
<tr><td>Gate</td><td>R² ≥ 0.85 · SMAPE ≤ 20%</td></tr>
</table>
""", unsafe_allow_html=True)

    # --- Column 3: Model Performance ---
    with c3:
        st.markdown("**Model Performance**")
        st.markdown("""
<table class="sysinfo-table">
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Features</td><td>9 (4 raw + 5 engineered)</td></tr>
<tr><td>Training samples</td><td>3,500</td></tr>
<tr><td>Validation R²</td>
    <td><span class="perf-pass">✓</span>
        <span class="perf-val"> 0.9282</span></td></tr>
<tr><td>Test R²</td>
    <td><span class="perf-pass">✓</span>
        <span class="perf-val"> 0.9714</span></td></tr>
<tr><td>Test RMSE</td>
    <td><span class="perf-val">43,870 kW</span></td></tr>
<tr><td>Test SMAPE</td>
    <td><span class="perf-val">29.90%</span></td></tr>
</table>
""", unsafe_allow_html=True)

    st.divider()

    # --- Bottom row: live status + API ---
    b1, b2 = st.columns([1, 2])

    with b1:
        st.markdown("**Live System Status**")
        if health:
            st.json({
                "api_status":   health.get("status"),
                "model_loaded": health.get("model_loaded"),
                "model":        health.get("model_version")
            })

    with b2:
        st.markdown("**API Endpoints**")
        st.markdown(f"""
<table class="sysinfo-table">
<tr><th>Endpoint</th><th>Description</th></tr>
<tr><td><code>GET /health</code></td>
    <td>Liveness check — model loaded status and version</td></tr>
<tr><td><code>POST /predict</code></td>
    <td>Single record inference — returns kW prediction with trace ID</td></tr>
<tr><td><code>POST /predict_batch</code></td>
    <td>Vectorised batch inference — up to 500 records per call</td></tr>
<tr><td><code>GET /docs</code></td>
    <td>Interactive Swagger UI — full API contract and testing</td></tr>
</table>
<p style='font-size:0.75rem; color:#94a3b8; margin-top:0.6rem;'>
    Base URL: <code>{API_URL}</code>
</p>
""", unsafe_allow_html=True)