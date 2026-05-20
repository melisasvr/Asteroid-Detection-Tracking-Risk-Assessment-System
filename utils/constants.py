# utils/constants.py
"""
Global constants for AsteroidWatch.
All physical units are SI or as noted.
"""

# ── Orbital / Physical ──────────────────────────────────────────────────────
AU_TO_KM          = 1.495978707e8   # 1 AU in km
EARTH_RADIUS_KM   = 6371.0          # mean Earth radius in km
EARTH_MOID_AU     = 0.05            # NEO threshold: Earth MOID < 0.05 AU
MOID_DANGER_AU    = 0.002           # "Potentially Hazardous": MOID < 0.002 AU (NASA PHA definition)
MOID_WARNING_AU   = 0.01            # Yellow-flag zone

# ── Torino Scale ─────────────────────────────────────────────────────────────
# Maps (MOID_AU, probability_of_impact) → Torino level (0-10)
# Simplified thresholds used for portfolio scoring — real Torino uses a 2D table
TORINO_SCALE_THRESHOLDS = [
    # (max_moid_au, min_probability, torino_level, label, color)
    (0.0001, 0.99, 10, "Certain Collision",      "#FF0000"),
    (0.0005, 0.70, 8,  "Certain Regional Impact", "#FF4400"),
    (0.001,  0.30, 6,  "Threatening",             "#FF8800"),
    (0.002,  0.05, 4,  "Meriting Concern",         "#FFCC00"),
    (0.002,  0.01, 3,  "Close Approach",           "#FFFF00"),
    (0.01,   0.001,2,  "Meriting Attention",       "#AAFF00"),
    (0.05,   0.0,  1,  "Normal",                   "#00FF00"),
    (99.0,   0.0,  0,  "No Hazard",                "#AAAAAA"),
]

# ── Data sources ─────────────────────────────────────────────────────────────
MPC_CATALOG_URL   = "https://www.minorplanetcenter.net/iau/MPCORB/MPCORB.DAT"
HORIZONS_BASE_URL = "https://ssd.jpl.nasa.gov/horizons_batch.cgi"

# ── Detection ────────────────────────────────────────────────────────────────
# Typical FITS survey image parameters (adjustable per instrument)
DEFAULT_PIXEL_SCALE    = 1.0      # arcsec per pixel (TESS-like)
MIN_STREAK_LENGTH_PX   = 10      # ignore blobs shorter than this
MAX_STREAK_WIDTH_PX    = 8       # anything wider is likely cosmic ray cluster
MIN_STREAK_ASPECT      = 3.0     # length/width ratio floor for a real streak
SIGMA_CLIP_THRESHOLD   = 3.0     # background σ-clipping level
DETECTION_THRESHOLD    = 5.0     # source must be > 5σ above background

# ── ML ───────────────────────────────────────────────────────────────────────
LSTM_SEQUENCE_LEN   = 10   # number of past observations fed to LSTM
LSTM_PREDICT_STEPS  = 5    # how many future steps to predict
LSTM_HIDDEN_DIM     = 64
LSTM_NUM_LAYERS     = 2