# dashboard/app.py
"""
AsteroidWatch Dashboard
=======================
Streamlit application — the public-facing face of the pipeline.

Run with:
    cd asteroidwatch
    streamlit run dashboard/app.py

LAYOUT
------
Sidebar   : controls, file upload, run settings
Main area : 4 tabs
  Tab 1 — Detection   : uploaded image + detected streaks overlaid
  Tab 2 — Orbits      : 3D interactive solar system orbit plot
  Tab 3 — Risk        : Torino scale gauge + risk breakdown
  Tab 4 — Trajectory  : LSTM forecast plot with uncertainty bands
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import tempfile
from pathlib import Path
from typing import List, Optional
import warnings
warnings.filterwarnings("ignore")

# ── Project imports ────────────────────────────────────────────────────────
from data.simulator import TelescopeImageSimulator, SimulationConfig
from detection.streak_detector import StreakDetector, match_detections_to_truth
from orbits.orbit_engine import (
    OrbitalElements, KeplerSolver, MOIDCalculator, OrbitFitter, RiskAssessor
)
from utils.helpers import normalize_image, sigma_clip_background

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title    = "AsteroidWatch",
    page_icon     = "☄️",
    layout        = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS — dark space theme ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
  --bg-deep:    #020408;
  --bg-panel:   #060c14;
  --bg-card:    #0a1628;
  --accent-1:   #00d4ff;   /* cyan */
  --accent-2:   #ff6b35;   /* orange — danger */
  --accent-3:   #39ff14;   /* neon green — safe */
  --accent-4:   #ffd700;   /* gold — warning */
  --text-main:  #e8f4fd;
  --text-dim:   #7a9bbf;
  --border:     #1a3a5c;
}

/* ── Global ── */
.stApp { background: var(--bg-deep); color: var(--text-main); }
html, body { font-family: 'Inter', sans-serif; }

/* ── Header ── */
.aw-header {
  background: linear-gradient(135deg, #020408 0%, #060c14 40%, #0a1628 100%);
  border-bottom: 1px solid var(--accent-1);
  padding: 1.5rem 2rem 1rem;
  margin-bottom: 1.5rem;
  position: relative;
  overflow: hidden;
}
.aw-header::before {
  content: '';
  position: absolute;
  top: -50%; left: -50%;
  width: 200%; height: 200%;
  background: radial-gradient(ellipse at center, rgba(0,212,255,0.04) 0%, transparent 70%);
  pointer-events: none;
}
.aw-title {
  font-family: 'Orbitron', monospace;
  font-size: 2.4rem;
  font-weight: 900;
  letter-spacing: 0.12em;
  color: var(--accent-1);
  text-shadow: 0 0 30px rgba(0,212,255,0.5), 0 0 60px rgba(0,212,255,0.2);
  margin: 0;
}
.aw-subtitle {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.85rem;
  color: var(--text-dim);
  letter-spacing: 0.15em;
  margin-top: 0.3rem;
}

/* ── Metric cards ── */
.metric-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem 1.25rem;
  margin: 0.4rem 0;
  position: relative;
  overflow: hidden;
}
.metric-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0;
  width: 3px; height: 100%;
  background: var(--accent-1);
}
.metric-label {
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.72rem;
  letter-spacing: 0.1em;
  color: var(--text-dim);
  text-transform: uppercase;
}
.metric-value {
  font-family: 'Orbitron', monospace;
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--accent-1);
  margin: 0.15rem 0 0;
}

/* ── Risk badge ── */
.torino-badge {
  display: inline-block;
  font-family: 'Orbitron', monospace;
  font-size: 0.8rem;
  font-weight: 700;
  padding: 0.3rem 0.8rem;
  border-radius: 4px;
  letter-spacing: 0.1em;
  margin-top: 0.3rem;
}

/* ── Section headers ── */
.section-title {
  font-family: 'Orbitron', monospace;
  font-size: 1rem;
  font-weight: 700;
  color: var(--accent-1);
  letter-spacing: 0.1em;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.5rem;
  margin: 1.5rem 0 1rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: var(--bg-panel) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stMarkdown p {
  color: var(--text-dim);
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.8rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-panel);
  border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Orbitron', monospace;
  font-size: 0.75rem;
  letter-spacing: 0.08em;
  color: var(--text-dim) !important;
}
.stTabs [aria-selected="true"] {
  color: var(--accent-1) !important;
  border-bottom: 2px solid var(--accent-1) !important;
}

/* ── Buttons ── */
.stButton > button {
  font-family: 'Orbitron', monospace;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  background: transparent;
  border: 1px solid var(--accent-1);
  color: var(--accent-1);
  padding: 0.6rem 1.5rem;
  transition: all 0.2s;
}
.stButton > button:hover {
  background: var(--accent-1);
  color: var(--bg-deep);
  box-shadow: 0 0 20px rgba(0,212,255,0.4);
}

/* ── Info boxes ── */
.info-box {
  background: rgba(0,212,255,0.05);
  border: 1px solid rgba(0,212,255,0.2);
  border-radius: 6px;
  padding: 0.75rem 1rem;
  font-family: 'Share Tech Mono', monospace;
  font-size: 0.8rem;
  color: var(--text-dim);
  margin: 0.5rem 0;
}

/* ── Dataframes ── */
.stDataFrame { border: 1px solid var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_moid_calculator():
    return MOIDCalculator(n_grid=360)

@st.cache_resource
def get_orbit_fitter():
    return OrbitFitter()

def metric_card(label: str, value: str, accent: str = "#00d4ff"):
    st.markdown(f"""
    <div class="metric-card" style="border-left-color:{accent}">
      <div class="metric-label">{label}</div>
      <div class="metric-value" style="color:{accent}">{value}</div>
    </div>""", unsafe_allow_html=True)

def torino_badge(level: int, label: str, color: str):
    text_color = "#000" if level <= 4 else "#fff"
    st.markdown(
        f'<div class="torino-badge" style="background:{color};color:{text_color}">'
        f'TORINO {level} — {label}</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="aw-header">
  <p class="aw-title">☄ ASTEROIDWATCH</p>
  <p class="aw-subtitle">PLANETARY DEFENSE DETECTION &amp; RISK ASSESSMENT SYSTEM  ·  v1.0</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙ MISSION CONTROL")
    st.markdown("---")

    st.markdown("**DATA SOURCE**")
    data_source = st.radio(
        "", ["Generate Simulation", "Upload FITS Image"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**DETECTION PARAMETERS**")
    sigma_thresh  = st.slider("Detection Threshold (σ)", 2.0, 8.0, 4.5, 0.5)
    min_length    = st.slider("Min Trail Length (px)",   5,   50,  10,   1)
    min_aspect    = st.slider("Min Aspect Ratio",        2.0, 8.0, 3.0, 0.5)

    st.markdown("---")
    st.markdown("**SIMULATION CONFIG**")
    n_asteroids   = st.slider("Simulated Asteroids", 1, 8, 3)
    n_stars       = st.slider("Background Stars",  50, 500, 200, 10)
    n_cosmic_rays = st.slider("Cosmic Rays",        0,  30,  15,  1)
    image_seed    = st.number_input("Random Seed", value=42, step=1)

    st.markdown("---")
    st.markdown("**ORBIT LOOKUP**")
    known_object  = st.text_input(
        "NASA Horizons ID (optional)",
        placeholder="e.g. 99942 (Apophis)",
        help="Enter a known asteroid ID to fetch real orbital elements"
    )

    st.markdown("---")
    run_pipeline  = st.button("▶ RUN PIPELINE", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline state
# ─────────────────────────────────────────────────────────────────────────────

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None


# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────────────────

if run_pipeline:
    with st.spinner("Running AsteroidWatch pipeline…"):

        # ── 1. Get image ──────────────────────────────────────────────────────
        if data_source == "Generate Simulation":
            cfg = SimulationConfig(
                n_asteroids   = n_asteroids,
                n_stars       = n_stars,
                n_cosmic_rays = n_cosmic_rays,
                random_seed   = int(image_seed),
            )
            sim   = TelescopeImageSimulator(cfg)
            image, truths = sim.generate()
            has_truth = True
        else:
            # File upload handled below; placeholder for now
            image  = None
            truths = []
            has_truth = False
            uploaded = st.session_state.get("uploaded_image")
            if uploaded is not None:
                image = uploaded
            else:
                st.warning("No image uploaded — switching to simulation.")
                sim   = TelescopeImageSimulator()
                image, truths = sim.generate()
                has_truth = True

        # ── 2. Detection ──────────────────────────────────────────────────────
        detector = StreakDetector(
            sigma_threshold  = sigma_thresh,
            min_length_px    = min_length,
            min_aspect_ratio = min_aspect,
        )
        detections = detector.detect(image)

        # ── 3. Orbit fitting + risk ───────────────────────────────────────────
        fitter      = get_orbit_fitter()
        orbit_elems = []

        for i, det in enumerate(detections[:5]):   # cap at 5 for dashboard speed
            obj_id = f"DET-{i+1:03d}"

            # If user specified a known object, try Horizons for first detection
            if i == 0 and known_object.strip():
                elems = fitter.fit_from_catalog(known_object.strip())
                if elems:
                    elems.object_id = known_object.strip()
                    orbit_elems.append(elems)
                    continue

            elems = fitter.fit_from_detection(det, object_id=obj_id)
            orbit_elems.append(elems)

        # ── 4. LSTM trajectory forecast ───────────────────────────────────────
        # Run synthetic forecast for first detected orbit
        forecasts = []
        if orbit_elems:
            solver = KeplerSolver()
            for elems in orbit_elems[:3]:
                obs_days  = np.linspace(0, 10, 10)
                positions = [solver.elements_to_cartesian(elems, dt)[0] for dt in obs_days]
                velocities = [solver.elements_to_cartesian(elems, dt)[1] for dt in obs_days]
                obs_seq = np.stack([
                    np.concatenate([p, v]) for p, v in zip(positions, velocities)
                ])  # (10, 6)

                # Try real LSTM if available, else physics forecast
                lstm_path = "data/processed/trajectory_lstm.pt"
                from ml.models import load_lstm
                lstm = load_lstm(lstm_path)

                if lstm is not None:
                    import torch
                    x = torch.tensor(obs_seq[np.newaxis], dtype=torch.float32)
                    with torch.no_grad():
                        pred = lstm(x).squeeze(0).numpy()
                else:
                    # Physics-based forecast (Kepler propagation)
                    fut_days = np.linspace(11, 16, 5)
                    pred = np.array([
                        solver.elements_to_cartesian(elems, dt)[0]
                        for dt in fut_days
                    ])

                forecasts.append({
                    'object_id': elems.object_id,
                    'observed':  np.array(positions),    # (10, 3)
                    'forecast':  pred,                   # (5, 3)
                })

        # ── Store results ─────────────────────────────────────────────────────
        st.session_state.pipeline_result = {
            'image':      image,
            'truths':     truths,
            'detections': detections,
            'orbits':     orbit_elems,
            'forecasts':  forecasts,
            'debug_imgs': detector.debug_images,
            'has_truth':  has_truth,
        }
    st.success(f"Pipeline complete — {len(detections)} detections, "
               f"{len(orbit_elems)} orbits computed")


# ─────────────────────────────────────────────────────────────────────────────
# File upload (outside run block so upload persists)
# ─────────────────────────────────────────────────────────────────────────────

if data_source == "Upload FITS Image":
    uploaded_file = st.file_uploader(
        "Upload FITS (.fit / .fits) or NumPy (.npy) image",
        type=["fit", "fits", "npy"],
    )
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(suffix=Path(uploaded_file.name).suffix,
                                         delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        if tmp_path.endswith(".npy"):
            arr = np.load(tmp_path).astype(np.float32)
        else:
            from utils.helpers import load_fits_image
            arr, _ = load_fits_image(tmp_path)

        st.session_state.uploaded_image = arr
        st.info(f"Loaded image: {arr.shape[0]}×{arr.shape[1]} px  "
                f"range [{arr.min():.0f}, {arr.max():.0f}]")


# ─────────────────────────────────────────────────────────────────────────────
# Results display
# ─────────────────────────────────────────────────────────────────────────────

result = st.session_state.pipeline_result

if result is None:
    st.markdown("""
    <div class="info-box">
    ◈  Configure parameters in the sidebar and click <strong>RUN PIPELINE</strong> to begin.<br>
    ◈  The system will simulate a telescope survey image, detect asteroid streaks, fit orbits, and assess impact risk.<br>
    ◈  Use "Upload FITS Image" to process your own data.
    </div>
    """, unsafe_allow_html=True)

    # ── Demo orbit visualization while idle ───────────────────────────────────
    st.markdown('<div class="section-title">SOLAR SYSTEM REFERENCE</div>', unsafe_allow_html=True)
    _solver = KeplerSolver()

    def _earth_orbit():
        e = OrbitalElements('Earth', 1.000, 0.0167, 0.00, 0.0, 102.9, 0.0, 2451545.0)
        return KeplerSolver.propagate_orbit(e, 200)

    def _mars_orbit():
        e = OrbitalElements('Mars', 1.524, 0.0934, 1.85, 49.6, 286.5, 0.0, 2451545.0)
        return KeplerSolver.propagate_orbit(e, 200)

    earth_pts = _earth_orbit()
    mars_pts  = _mars_orbit()

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers',
                               marker=dict(size=12, color='#FFD700', symbol='circle'),
                               name='Sun'))
    fig.add_trace(go.Scatter3d(x=earth_pts[:,0], y=earth_pts[:,1], z=earth_pts[:,2],
                               mode='lines', line=dict(color='#3399FF', width=2), name='Earth'))
    fig.add_trace(go.Scatter3d(x=mars_pts[:,0], y=mars_pts[:,1], z=mars_pts[:,2],
                               mode='lines', line=dict(color='#FF4422', width=2), name='Mars'))
    fig.update_layout(
        height=400,
        paper_bgcolor='#020408', plot_bgcolor='#020408',
        scene=dict(
            bgcolor='#020408',
            xaxis=dict(title='X (AU)', color='#7a9bbf', gridcolor='#1a3a5c'),
            yaxis=dict(title='Y (AU)', color='#7a9bbf', gridcolor='#1a3a5c'),
            zaxis=dict(title='Z (AU)', color='#7a9bbf', gridcolor='#1a3a5c'),
        ),
        legend=dict(bgcolor='rgba(6,12,20,0.8)', font=dict(color='#e8f4fd')),
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(family='Share Tech Mono', color='#e8f4fd'),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.stop()


# ── Unpack results ─────────────────────────────────────────────────────────────
image      = result['image']
truths     = result['truths']
detections = result['detections']
orbits     = result['orbits']
forecasts  = result['forecasts']
has_truth  = result['has_truth']

# ── Top KPI row ────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1: metric_card("DETECTIONS",  str(len(detections)))
with k2: metric_card("ORBITS FITTED", str(len(orbits)))
with k3:
    if orbits:
        best_moid = min(o.moid_au for o in orbits)
        color = "#ff6b35" if best_moid < 0.05 else "#ffd700" if best_moid < 0.1 else "#39ff14"
        metric_card("MIN MOID", f"{best_moid:.4f} AU", color)
    else:
        metric_card("MIN MOID", "—")
with k4:
    if orbits:
        max_torino = max(o.torino_level for o in orbits)
        t_color = orbits[np.argmax([o.torino_level for o in orbits])].torino_color
        metric_card("MAX TORINO", str(max_torino), t_color)
    else:
        metric_card("MAX TORINO", "—")
with k5:
    tp_str = "—"
    if has_truth:
        from detection.streak_detector import match_detections_to_truth as mtd
        m = mtd(detections, truths)
        tp_str = f"{m['recall']:.0%} recall"
    metric_card("GROUND TRUTH", tp_str, "#39ff14")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "🔭  DETECTION",
    "🪐  ORBITS",
    "⚠  RISK ASSESSMENT",
    "📡  TRAJECTORY FORECAST",
])


# ── TAB 1: Detection ───────────────────────────────────────────────────────────
with tab1:
    col_img, col_info = st.columns([3, 2])

    with col_img:
        st.markdown('<div class="section-title">TELESCOPE IMAGE — DETECTED STREAKS</div>',
                    unsafe_allow_html=True)

        norm = normalize_image(image)
        # Convert grayscale → RGB for Plotly
        rgb = np.stack([norm, norm, norm], axis=-1)

        fig_img = go.Figure()
        fig_img.add_trace(go.Image(z=(rgb * 255).astype(np.uint8)))

        # Ground truth streaks (green)
        if has_truth:
            for t in truths:
                fig_img.add_shape(type="line",
                    x0=t.x_start, y0=t.y_start, x1=t.x_end, y1=t.y_end,
                    line=dict(color="#39ff14", width=2, dash="dot"))
                fig_img.add_annotation(
                    x=t.x_start, y=t.y_start, text=t.object_id,
                    font=dict(color="#39ff14", size=10, family="Share Tech Mono"),
                    showarrow=False, xanchor="left"
                )

        # Detections (cyan)
        for i, s in enumerate(detections):
            color = "#00d4ff" if s.confidence > 0.5 else "#ffd700"
            fig_img.add_shape(type="line",
                x0=s.x1, y0=s.y1, x1=s.x2, y1=s.y2,
                line=dict(color=color, width=2))
            mx, my = s.midpoint
            fig_img.add_annotation(
                x=mx, y=my, text=f"D{i+1}",
                font=dict(color=color, size=9, family="Orbitron"),
                showarrow=False
            )

        fig_img.update_layout(
            height=500, margin=dict(l=0,r=0,t=0,b=0),
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False,
                       autorange='reversed'),
            paper_bgcolor='#020408', plot_bgcolor='#020408',
        )
        st.plotly_chart(fig_img, use_container_width=True)

        if has_truth:
            st.markdown(
                '<div class="info-box">🟢 Green dotted = ground truth &nbsp;&nbsp;'
                '🔵 Cyan = detections (high conf) &nbsp;&nbsp;'
                '🟡 Yellow = detections (low conf)</div>',
                unsafe_allow_html=True
            )

    with col_info:
        st.markdown('<div class="section-title">DETECTION CATALOGUE</div>',
                    unsafe_allow_html=True)

        if detections:
            rows = []
            for i, s in enumerate(detections):
                rows.append({
                    'ID': f'D{i+1}',
                    'Length (px)': f'{s.length_px:.1f}',
                    'Aspect': f'{s.aspect_ratio:.1f}',
                    'SNR': f'{s.snr:.1f}',
                    'Conf': f'{s.confidence:.2f}',
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No streaks detected. Try lowering the detection threshold.")

        if has_truth:
            st.markdown('<div class="section-title">EVALUATION</div>',
                        unsafe_allow_html=True)
            from detection.streak_detector import match_detections_to_truth as mtd
            m = mtd(detections, truths)
            e1, e2, e3 = st.columns(3)
            with e1: metric_card("PRECISION", f"{m['precision']:.0%}", "#00d4ff")
            with e2: metric_card("RECALL",    f"{m['recall']:.0%}",    "#39ff14")
            with e3: metric_card("F1",        f"{m['f1']:.0%}",        "#ffd700")

        # Debug images
        with st.expander("🔬 Detection Pipeline Internals"):
            debug = result['debug_imgs']
            if 'background_subtracted' in debug:
                dcols = st.columns(2)
                for idx, (key, title) in enumerate([
                    ('background_subtracted', 'Background Subtracted'),
                    ('binary_mask', 'Detection Mask'),
                ]):
                    if key in debug:
                        img_d = normalize_image(debug[key].astype(np.float32))
                        with dcols[idx % 2]:
                            st.image(img_d, caption=title, use_container_width=True,
                                     clamp=True)


# ── TAB 2: Orbits ──────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title">HELIOCENTRIC ORBIT VISUALIZATION</div>',
                unsafe_allow_html=True)

    if not orbits:
        st.info("Run the pipeline to compute orbits.")
    else:
        solver = KeplerSolver()

        # ── Build reference orbits ──────────────────────────────────────────────
        def ref_orbit(name, a, e, i, raan, argp, color, n=300):
            el = OrbitalElements(name, a, e, i, raan, argp, 0.0, 2451545.0)
            pts = KeplerSolver.propagate_orbit(el, n)
            return go.Scatter3d(
                x=pts[:,0], y=pts[:,1], z=pts[:,2],
                mode='lines', line=dict(color=color, width=1.5),
                name=name, opacity=0.7,
            )

        fig3d = go.Figure()

        # Sun
        fig3d.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], mode='markers',
            marker=dict(size=14, color='#FFD700',
                        line=dict(color='#FF8C00', width=2)),
            name='☀ Sun'
        ))

        # Reference planets
        fig3d.add_trace(ref_orbit('Mercury', 0.387, 0.206, 7.0,   48.3, 29.1, '#A0A0A0'))
        fig3d.add_trace(ref_orbit('Venus',   0.723, 0.007, 3.4,   76.7, 55.2, '#E8C85A'))
        fig3d.add_trace(ref_orbit('Earth',   1.000, 0.017, 0.0,    0.0,102.9, '#3399FF'))
        fig3d.add_trace(ref_orbit('Mars',    1.524, 0.093, 1.85,  49.6,286.5, '#FF4422'))

        # Asteroid orbits
        colors = ['#00d4ff','#ff6b35','#39ff14','#ffd700','#ff00aa']
        for idx, elems in enumerate(orbits):
            color = colors[idx % len(colors)]
            pts   = KeplerSolver.propagate_orbit(elems, 360)
            fig3d.add_trace(go.Scatter3d(
                x=pts[:,0], y=pts[:,1], z=pts[:,2],
                mode='lines',
                line=dict(color=color, width=3),
                name=f'☄ {elems.object_id}',
            ))

            # Current position
            pos, _ = solver.elements_to_cartesian(elems, 0.0)
            fig3d.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers',
                marker=dict(size=8, color=color,
                            line=dict(color='white', width=1)),
                name=f'{elems.object_id} (pos)',
                showlegend=False,
            ))

        fig3d.update_layout(
            height=600,
            paper_bgcolor='#020408',
            scene=dict(
                bgcolor='#020408',
                xaxis=dict(title='X (AU)', color='#7a9bbf',
                           gridcolor='#1a3a5c', zeroline=False),
                yaxis=dict(title='Y (AU)', color='#7a9bbf',
                           gridcolor='#1a3a5c', zeroline=False),
                zaxis=dict(title='Z (AU)', color='#7a9bbf',
                           gridcolor='#1a3a5c', zeroline=False),
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
            ),
            legend=dict(bgcolor='rgba(6,12,20,0.9)', font=dict(color='#e8f4fd', size=11)),
            margin=dict(l=0, r=0, t=20, b=0),
            font=dict(family='Share Tech Mono', color='#e8f4fd'),
        )
        st.plotly_chart(fig3d, use_container_width=True)

        # Orbital elements table
        st.markdown('<div class="section-title">ORBITAL ELEMENTS</div>',
                    unsafe_allow_html=True)
        rows = []
        for o in orbits:
            rows.append({
                'Object':   o.object_id,
                'a (AU)':   f'{o.a:.3f}',
                'e':        f'{o.e:.3f}',
                'i (°)':    f'{o.i:.2f}',
                'q (AU)':   f'{o.perihelion_au:.3f}',
                'Q (AU)':   f'{o.aphelion_au:.3f}',
                'T (yr)':   f'{o.period_yr:.3f}',
                'MOID (AU)':f'{o.moid_au:.5f}',
                'NEO':      '✓' if o.is_neo else '✗',
                'PHA':      '⚠' if o.is_pha else '✗',
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── TAB 3: Risk Assessment ─────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">IMPACT RISK ASSESSMENT</div>',
                unsafe_allow_html=True)

    if not orbits:
        st.info("Run the pipeline first.")
    else:
        # ── Torino gauge for most dangerous object ────────────────────────────
        worst = max(orbits, key=lambda o: o.risk_score)

        left, right = st.columns([2, 3])

        with left:
            st.markdown(f"**Object: {worst.object_id}**")
            torino_badge(worst.torino_level, worst.torino_label, worst.torino_color)
            st.markdown("<br>", unsafe_allow_html=True)

            metric_card("MOID",
                        f"{worst.moid_au:.5f} AU",
                        "#ff6b35" if worst.moid_au < 0.05 else "#ffd700")
            metric_card("IMPACT PROBABILITY",
                        f"{worst.impact_prob:.2e}",
                        "#ff6b35" if worst.impact_prob > 1e-4 else "#39ff14")
            metric_card("RISK SCORE",
                        f"{worst.risk_score:.4f}",
                        "#ff6b35" if worst.risk_score > 0.3 else "#ffd700")
            metric_card("PERIHELION", f"{worst.perihelion_au:.3f} AU")

        with right:
            # Torino scale reference plot
            torino_colors = ['#AAAAAA','#00FF00','#AAFF00','#FFFF00','#FFCC00',
                             '#FF8800','#FF4400','#FF0000','#FF0000','#FF0000','#FF0000']
            torino_labels = [
                'No Hazard','Normal','Meriting Attention','Close Approach',
                'Meriting Concern','Threatening','Threatening (Large)',
                'Extreme (Local)','Extreme (Regional)','Certain Collision','Certain (Massive)'
            ]
            fig_torino = go.Figure(go.Bar(
                x=list(range(11)),
                y=[1]*11,
                marker_color=torino_colors,
                text=list(range(11)),
                textposition='inside',
                textfont=dict(color='black', size=14, family='Orbitron'),
            ))
            # Highlight current level
            fig_torino.add_shape(
                type='rect',
                x0=worst.torino_level - 0.5, x1=worst.torino_level + 0.5,
                y0=0, y1=1.2,
                fillcolor='white', opacity=0.3, line=dict(color='white', width=2),
            )
            fig_torino.add_annotation(
                x=worst.torino_level, y=1.35,
                text=f"▲ Current: {worst.torino_level}",
                font=dict(color='white', size=12, family='Orbitron'),
                showarrow=False,
            )
            fig_torino.update_layout(
                title=dict(text='TORINO SCALE', font=dict(family='Orbitron', color='#00d4ff', size=13)),
                height=220, showlegend=False,
                xaxis=dict(title='Level', tickfont=dict(family='Share Tech Mono', color='#7a9bbf')),
                yaxis=dict(visible=False),
                paper_bgcolor='#020408', plot_bgcolor='#060c14',
                margin=dict(l=20, r=20, t=50, b=40),
            )
            st.plotly_chart(fig_torino, use_container_width=True)

        # ── Risk comparison across all detected objects ───────────────────────
        st.markdown('<div class="section-title">ALL OBJECTS — RISK COMPARISON</div>',
                    unsafe_allow_html=True)

        if len(orbits) > 1:
            fig_risk = go.Figure()
            for o in orbits:
                moid_km = o.moid_au * 1.496e8
                fig_risk.add_trace(go.Scattergl(
                    x=[o.moid_au],
                    y=[o.impact_prob],
                    mode='markers+text',
                    marker=dict(
                        size=max(8, o.risk_score * 40),
                        color=o.torino_color,
                        line=dict(color='white', width=1),
                        opacity=0.9,
                    ),
                    text=[o.object_id],
                    textposition='top center',
                    textfont=dict(color='#e8f4fd', size=10, family='Share Tech Mono'),
                    name=o.object_id,
                    hovertemplate=(
                        f"<b>{o.object_id}</b><br>"
                        f"MOID: {o.moid_au:.5f} AU<br>"
                        f"P(impact): {o.impact_prob:.2e}<br>"
                        f"Torino: {o.torino_level}<extra></extra>"
                    )
                ))

            # PHA threshold line
            fig_risk.add_vline(x=0.05, line=dict(color='#ffd700', dash='dash', width=1))
            fig_risk.add_annotation(x=0.05, y=0.0, text=" PHA threshold", yanchor='bottom',
                font=dict(color='#ffd700', size=10, family='Share Tech Mono'), showarrow=False)
            fig_risk.add_vline(x=0.002, line=dict(color='#ff6b35', dash='dot', width=1))
            fig_risk.add_annotation(x=0.002, y=0.0, text=" Danger zone", yanchor='bottom',
                font=dict(color='#ff6b35', size=10, family='Share Tech Mono'), showarrow=False)

            fig_risk.update_layout(
                xaxis=dict(title='MOID (AU)', type='log', color='#7a9bbf',
                           gridcolor='#1a3a5c'),
                yaxis=dict(title='P(Impact)', type='log', color='#7a9bbf',
                           gridcolor='#1a3a5c'),
                height=380, showlegend=True,
                paper_bgcolor='#020408', plot_bgcolor='#060c14',
                legend=dict(bgcolor='rgba(6,12,20,0.8)', font=dict(color='#e8f4fd')),
                margin=dict(l=60, r=20, t=20, b=60),
                font=dict(family='Share Tech Mono', color='#7a9bbf'),
            )
            st.plotly_chart(fig_risk, use_container_width=True)


# ── TAB 4: Trajectory Forecast ────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">TRAJECTORY FORECAST (LSTM / KEPLER)</div>',
                unsafe_allow_html=True)

    if not forecasts:
        st.info("No trajectory forecasts available. Run the pipeline.")
    else:
        for fc in forecasts[:3]:
            obj_id = fc['object_id']
            obs    = fc['observed']    # (10, 3)
            pred   = fc['forecast']    # (5, 3)

            st.markdown(f"**{obj_id}** — XY projection (AU, ecliptic plane)")

            # Earth reference circle
            theta  = np.linspace(0, 2*np.pi, 200)
            ex, ey = np.cos(theta), np.sin(theta)

            fig_traj = go.Figure()

            # Earth
            fig_traj.add_trace(go.Scatter(
                x=ex, y=ey, mode='lines',
                line=dict(color='#3399FF', width=1, dash='dot'),
                name='Earth orbit', opacity=0.5,
            ))
            fig_traj.add_trace(go.Scatter(
                x=[0], y=[0], mode='markers',
                marker=dict(size=12, color='#FFD700', symbol='star'),
                name='Sun'
            ))

            # Observed arc
            fig_traj.add_trace(go.Scatter(
                x=obs[:,0], y=obs[:,1], mode='lines+markers',
                line=dict(color='#00d4ff', width=2),
                marker=dict(size=5, color='#00d4ff'),
                name='Observed arc',
            ))

            # Forecast
            # Connect last observed → first predicted
            con_x = [obs[-1,0], pred[0,0]]
            con_y = [obs[-1,1], pred[0,1]]
            fig_traj.add_trace(go.Scatter(
                x=con_x, y=con_y, mode='lines',
                line=dict(color='#ff6b35', width=1, dash='dot'),
                showlegend=False
            ))
            fig_traj.add_trace(go.Scatter(
                x=pred[:,0], y=pred[:,1], mode='lines+markers',
                line=dict(color='#ff6b35', width=2, dash='dash'),
                marker=dict(size=7, color='#ff6b35', symbol='diamond'),
                name='LSTM Forecast',
            ))

            # Start / end annotations
            fig_traj.add_annotation(
                x=obs[0,0], y=obs[0,1], text='t₀', showarrow=True,
                arrowcolor='#00d4ff', font=dict(color='#00d4ff', family='Share Tech Mono'),
            )
            fig_traj.add_annotation(
                x=pred[-1,0], y=pred[-1,1], text='t+5', showarrow=True,
                arrowcolor='#ff6b35', font=dict(color='#ff6b35', family='Share Tech Mono'),
            )

            fig_traj.update_layout(
                height=380,
                xaxis=dict(title='X (AU)', color='#7a9bbf', gridcolor='#1a3a5c',
                           scaleanchor='y', scaleratio=1),
                yaxis=dict(title='Y (AU)', color='#7a9bbf', gridcolor='#1a3a5c'),
                paper_bgcolor='#020408', plot_bgcolor='#060c14',
                legend=dict(bgcolor='rgba(6,12,20,0.8)', font=dict(color='#e8f4fd')),
                margin=dict(l=50, r=20, t=20, b=50),
                font=dict(family='Share Tech Mono', color='#7a9bbf'),
            )
            st.plotly_chart(fig_traj, use_container_width=True)

    # LSTM training section
    st.markdown('<div class="section-title">MODEL TRAINING</div>', unsafe_allow_html=True)
    t_col1, t_col2 = st.columns(2)

    with t_col1:
        st.markdown("**Train CNN Streak Classifier**")
        cnn_images = st.slider("Training images", 10, 500, 50, key='cnn_imgs')
        cnn_epochs = st.slider("Epochs", 5, 50, 15, key='cnn_ep')
        if st.button("Train CNN", key='train_cnn'):
            from ml.models import train_cnn
            with st.spinner("Training CNN…"):
                model = train_cnn(n_images=cnn_images, epochs=cnn_epochs)
            st.success("CNN training complete!")

    with t_col2:
        st.markdown("**Train LSTM Trajectory Forecaster**")
        lstm_samples = st.slider("Training samples", 100, 5000, 500, key='lstm_n')
        lstm_epochs  = st.slider("Epochs", 5, 100, 20, key='lstm_ep')
        if st.button("Train LSTM", key='train_lstm'):
            from ml.models import train_lstm
            with st.spinner("Training LSTM…"):
                model = train_lstm(n_samples=lstm_samples, epochs=lstm_epochs)
            st.success("LSTM training complete!")