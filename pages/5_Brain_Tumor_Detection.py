"""
pages/5_Brain_Tumor_Detection.py
Brain Tumor MRI Classification 
Model downloads from Google Drive on first run (not stored in GitHub)
"""

import streamlit as st
import numpy as np
import json
import os
import gdown
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    import keras
    import tensorflow as tf

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI · Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,700;1,9..40,300&family=JetBrains+Mono:wght@400;700&display=swap');

/* ════════════════════════════════════════
   ROOT TOKENS
════════════════════════════════════════ */
:root {
    --bg-void:      #050507;
    --bg-surface:   #0d0d12;
    --bg-card:      #12121a;
    --bg-elevated:  #1a1a26;
    --border-dim:   rgba(255,255,255,0.06);
    --border-glow:  rgba(229,9,20,0.4);

    --red-primary:  #e50914;
    --red-hot:      #ff2030;
    --red-dim:      rgba(229,9,20,0.15);
    --red-glow:     rgba(229,9,20,0.08);

    --cyan-accent:  #00d4ff;
    --gold-accent:  #f5c518;
    --green-safe:   #00e676;
    --purple-med:   #b388ff;
    --amber-warn:   #ffab40;

    --text-primary: #f5f5f1;
    --text-secondary: rgba(245,245,241,0.6);
    --text-muted:   rgba(245,245,241,0.35);

    --font-display: 'Bebas Neue', sans-serif;
    --font-body:    'DM Sans', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;

    --radius-sm: 6px;
    --radius-md: 12px;
    --radius-lg: 20px;
    --radius-xl: 28px;

    --transition: 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ════════════════════════════════════════
   GLOBAL RESET
════════════════════════════════════════ */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
    background-color: var(--bg-void) !important;
}

.stApp {
    background: var(--bg-void) !important;
}

/* Animated grain overlay */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 9999;
    opacity: 0.4;
}

/* ════════════════════════════════════════
   SIDEBAR
════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a0f 0%, #0d0d14 100%) !important;
    border-right: 1px solid var(--border-dim) !important;
    box-shadow: 4px 0 40px rgba(0,0,0,0.6) !important;
}

section[data-testid="stSidebar"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--red-primary), var(--cyan-accent), var(--red-primary));
    background-size: 200%;
    animation: shimmer 3s infinite linear;
}

@keyframes shimmer {
    0%   { background-position: 0% 0%; }
    100% { background-position: 200% 0%; }
}

section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: var(--font-display) !important;
    letter-spacing: 0.1em !important;
    color: var(--text-primary) !important;
}

/* ════════════════════════════════════════
   HERO HEADER
════════════════════════════════════════ */
.ns-hero {
    position: relative;
    text-align: center;
    padding: 3.5rem 2rem 2rem;
    margin-bottom: 0;
    overflow: hidden;
}

.ns-hero::before {
    content: '';
    position: absolute;
    top: -60%; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 600px;
    background: radial-gradient(circle, rgba(229,9,20,0.12) 0%, transparent 65%);
    pointer-events: none;
}

.ns-eyebrow {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.35em;
    color: var(--red-primary);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
    opacity: 0.9;
}

.ns-title {
    font-family: var(--font-display);
    font-size: clamp(3rem, 7vw, 5.5rem);
    letter-spacing: 0.04em;
    line-height: 0.95;
    background: linear-gradient(135deg, #fff 30%, rgba(255,255,255,0.55) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem;
}

.ns-subtitle {
    font-family: var(--font-body);
    font-size: 0.88rem;
    color: var(--text-secondary);
    letter-spacing: 0.05em;
    font-weight: 300;
    margin-bottom: 0.3rem;
}

.ns-badge-row {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 1.2rem;
}

.ns-badge {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    padding: 0.25rem 0.7rem;
    border-radius: 4px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 700;
}

.ns-badge-red   { background: rgba(229,9,20,0.15); color: #ff6b75; border: 1px solid rgba(229,9,20,0.3); }
.ns-badge-cyan  { background: rgba(0,212,255,0.1);  color: var(--cyan-accent); border: 1px solid rgba(0,212,255,0.25); }
.ns-badge-gold  { background: rgba(245,197,24,0.1); color: var(--gold-accent); border: 1px solid rgba(245,197,24,0.25); }

/* ════════════════════════════════════════
   DIVIDER
════════════════════════════════════════ */
.ns-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(229,9,20,0.4), rgba(255,255,255,0.08), transparent);
    margin: 1.5rem 0;
}

/* ════════════════════════════════════════
   UPLOAD ZONE
════════════════════════════════════════ */
section[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed rgba(229,9,20,0.35) !important;
    border-radius: var(--radius-lg) !important;
    padding: 2rem 1.5rem !important;
    transition: var(--transition) !important;
    position: relative !important;
    overflow: hidden !important;
}

section[data-testid="stFileUploader"]::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(ellipse at center, rgba(229,9,20,0.04) 0%, transparent 70%);
    pointer-events: none;
}

section[data-testid="stFileUploader"]:hover {
    border-color: rgba(229,9,20,0.7) !important;
    background: rgba(229,9,20,0.04) !important;
    box-shadow: 0 0 40px rgba(229,9,20,0.1), inset 0 0 40px rgba(229,9,20,0.03) !important;
}

section[data-testid="stFileUploader"] label,
section[data-testid="stFileUploader"] span,
section[data-testid="stFileUploader"] p {
    color: var(--text-secondary) !important;
    font-family: var(--font-body) !important;
}

section[data-testid="stFileUploader"] small {
    color: var(--text-muted) !important;
}

/* Upload button override */
section[data-testid="stFileUploader"] button {
    background: transparent !important;
    border: 1px solid rgba(229,9,20,0.4) !important;
    color: var(--red-hot) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    transition: var(--transition) !important;
}

section[data-testid="stFileUploader"] button:hover {
    background: var(--red-dim) !important;
    box-shadow: 0 0 20px rgba(229,9,20,0.2) !important;
}

/* ════════════════════════════════════════
   STEP CARDS (pre-upload)
════════════════════════════════════════ */
.ns-step-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}

.ns-step {
    background: var(--bg-card);
    border: 1px solid var(--border-dim);
    border-radius: var(--radius-md);
    padding: 1.5rem 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: var(--transition);
    cursor: default;
}

.ns-step::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--red-primary), transparent);
    transform: scaleX(0);
    transition: transform 0.4s ease;
}

.ns-step:hover::before { transform: scaleX(1); }

.ns-step:hover {
    border-color: rgba(229,9,20,0.3);
    background: var(--bg-elevated);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(229,9,20,0.1);
    transform: translateY(-3px);
}

.ns-step-num {
    font-family: var(--font-display);
    font-size: 3rem;
    line-height: 1;
    color: rgba(229,9,20,0.2);
    margin-bottom: 0.5rem;
}

.ns-step-icon { font-size: 1.6rem; margin-bottom: 0.5rem; }

.ns-step-title {
    font-family: var(--font-body);
    font-weight: 700;
    font-size: 0.85rem;
    color: var(--text-primary);
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.ns-step-desc {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
    line-height: 1.5;
}

/* ════════════════════════════════════════
   PREDICTION CARD
════════════════════════════════════════ */
.ns-result-wrapper {
    position: relative;
    margin: 1.5rem 0;
}

.ns-result-card {
    background: var(--bg-card);
    border-radius: var(--radius-xl);
    padding: 2rem 2.5rem;
    border: 1px solid var(--border-dim);
    display: flex;
    align-items: center;
    gap: 2rem;
    position: relative;
    overflow: hidden;
}

.ns-result-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: var(--result-glow, rgba(229,9,20,0.05));
    pointer-events: none;
}

.ns-result-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--result-color, var(--red-primary));
    box-shadow: 0 0 20px var(--result-color, var(--red-primary));
}

.ns-result-icon-wrap {
    width: 80px; height: 80px;
    min-width: 80px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 2.5rem;
    background: rgba(255,255,255,0.04);
    border: 2px solid var(--result-color, var(--red-primary));
    box-shadow: 0 0 30px var(--result-color, rgba(229,9,20,0.3));
}

.ns-result-body { flex: 1; }

.ns-result-label {
    font-family: var(--font-display);
    font-size: 2.8rem;
    letter-spacing: 0.05em;
    line-height: 1;
    color: var(--text-primary);
    margin-bottom: 0.3rem;
}

.ns-result-desc {
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.5;
    max-width: 500px;
}

.ns-conf-block {
    text-align: right;
    min-width: 140px;
}

.ns-conf-value {
    font-family: var(--font-display);
    font-size: 3.5rem;
    line-height: 1;
    color: var(--result-color, var(--red-primary));
}

.ns-conf-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* ════════════════════════════════════════
   URGENCY BANNER
════════════════════════════════════════ */
.ns-urgency {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.75rem 1.25rem;
    border-radius: var(--radius-md);
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    border: 1px solid;
    margin: 0.75rem 0;
}

.urg-high   { background: rgba(229,9,20,0.08);  border-color: rgba(229,9,20,0.3);  color: #ff6b75; }
.urg-medium { background: rgba(0,212,255,0.06); border-color: rgba(0,212,255,0.2); color: var(--cyan-accent); }
.urg-low    { background: rgba(0,230,118,0.06); border-color: rgba(0,230,118,0.2); color: var(--green-safe); }

/* ════════════════════════════════════════
   WARNING BOX
════════════════════════════════════════ */
div[data-testid="stAlert"] {
    background: rgba(255, 171, 64, 0.06) !important;
    border: 1px solid rgba(255, 171, 64, 0.25) !important;
    border-radius: var(--radius-md) !important;
    color: var(--amber-warn) !important;
}

/* ════════════════════════════════════════
   EXPANDER
════════════════════════════════════════ */
details[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: var(--radius-md) !important;
}

details[data-testid="stExpander"] summary {
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    color: var(--text-primary) !important;
    padding: 0.75rem 1rem !important;
}

/* ════════════════════════════════════════
   IMAGE SECTION LABELS
════════════════════════════════════════ */
.ns-img-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--text-muted);
    text-align: center;
    margin-bottom: 0.5rem;
    padding: 0.3rem 0.6rem;
    background: var(--bg-elevated);
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-dim);
    display: inline-block;
}

.ns-img-wrap {
    background: var(--bg-card);
    border-radius: var(--radius-md);
    border: 1px solid var(--border-dim);
    overflow: hidden;
    transition: var(--transition);
    text-align: center;
    padding: 0.75rem;
}

.ns-img-wrap:hover {
    border-color: rgba(229,9,20,0.3);
    box-shadow: 0 0 30px rgba(229,9,20,0.08);
}

/* ════════════════════════════════════════
   PROBABILITIES SECTION
════════════════════════════════════════ */
.ns-section-head {
    font-family: var(--font-display);
    font-size: 1.5rem;
    letter-spacing: 0.06em;
    color: var(--text-primary);
    margin: 1.5rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.ns-section-head::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(255,255,255,0.08), transparent);
}

.ns-prob-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.55rem 0.75rem;
    border-radius: var(--radius-sm);
    margin-bottom: 0.3rem;
    border: 1px solid transparent;
    transition: var(--transition);
}

.ns-prob-row:hover {
    background: rgba(255,255,255,0.025);
    border-color: var(--border-dim);
}

.ns-prob-row.is-top {
    background: rgba(229,9,20,0.06);
    border-color: rgba(229,9,20,0.2);
}

.ns-prob-name {
    font-family: var(--font-body);
    font-size: 0.82rem;
    font-weight: 500;
    flex: 1;
    text-transform: capitalize;
}

.ns-prob-val {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    font-weight: 700;
    min-width: 54px;
    text-align: right;
}

/* ════════════════════════════════════════
   METRIC BOXES (sidebar)
════════════════════════════════════════ */
.metric-box {
    background: var(--bg-card);
    border-radius: var(--radius-sm);
    padding: 0.6rem 0.8rem;
    margin: 0.3rem 0;
    border: 1px solid var(--border-dim);
    border-left: 3px solid var(--red-primary);
}

.metric-label {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
}

.metric-value {
    font-family: var(--font-display);
    font-size: 1.5rem;
    color: var(--text-primary);
    line-height: 1.2;
}

/* ════════════════════════════════════════
   DISCLAIMER
════════════════════════════════════════ */
.ns-disclaimer {
    background: rgba(245,197,24,0.05);
    border: 1px solid rgba(245,197,24,0.2);
    border-radius: var(--radius-md);
    padding: 0.9rem 1.2rem;
    font-size: 0.8rem;
    color: rgba(245,197,24,0.75);
    line-height: 1.6;
    display: flex;
    gap: 0.75rem;
    align-items: flex-start;
}

/* ════════════════════════════════════════
   STREAMLIT OVERRIDES
════════════════════════════════════════ */
.stButton > button {
    background: var(--red-primary) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    padding: 0.55rem 1.2rem !important;
    transition: var(--transition) !important;
    box-shadow: 0 4px 20px rgba(229,9,20,0.3) !important;
}

.stButton > button:hover {
    background: var(--red-hot) !important;
    box-shadow: 0 6px 30px rgba(229,9,20,0.5) !important;
    transform: translateY(-1px) !important;
}

.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    transition: var(--transition) !important;
}

.stDownloadButton > button:hover {
    border-color: rgba(229,9,20,0.4) !important;
    color: var(--text-primary) !important;
    background: var(--red-dim) !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] {
    padding-top: 0.5rem !important;
}

/* Checkbox */
.stCheckbox label {
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
}

/* Success box override */
div[data-testid="stAlert"][data-alert-type="success"] {
    background: rgba(0,230,118,0.06) !important;
    border: 1px solid rgba(0,230,118,0.2) !important;
    border-radius: var(--radius-md) !important;
    color: var(--green-safe) !important;
}

/* Spinner text */
.stSpinner > div {
    color: var(--red-primary) !important;
}

/* Caption text */
.stImage figcaption {
    font-family: var(--font-mono) !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    color: var(--text-muted) !important;
    text-align: center !important;
    text-transform: uppercase !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; background: var(--bg-void); }
::-webkit-scrollbar-thumb { background: rgba(229,9,20,0.3); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(229,9,20,0.6); }

/* HR */
hr { border-color: var(--border-dim) !important; }

/* matplotlib figure background */
.stPlotlyChart, [data-testid="stImage"] { border-radius: var(--radius-md); overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS  (unchanged)
# ─────────────────────────────────────────────
IMAGE_SIZE      = (224, 224)
MODEL_DRIVE_ID = "182-lKxcwQUDxs4so2jcql5N8yy6JUsjO"
MODEL_PATH      = "best_model.h5"

DEFAULT_LABELS = {
    "0": "glioma", "1": "meningioma",
    "2": "notumor", "3": "other", "4": "pituitary"
}

DEFAULT_MODEL_INFO = {
    "overall_accuracy": 0.98, "macro_auc": 0.99, "kappa": 0.97,
    "per_class_accuracy": {
        "glioma": 0.953, "meningioma": 0.957,
        "notumor": 0.995, "other": 0.995, "pituitary": 0.996
    }
}

CLASS_COLORS = {
    "glioma": "#e50914", "meningioma": "#00d4ff",
    "notumor": "#00e676", "other": "#ffab40", "pituitary": "#b388ff"
}

CLASS_INFO = {
    "glioma":     {"icon": "🔴", "urgency": "high",
                   "desc": "Arises from glial cells. Ranges from slow-growing (grade I) to aggressive (grade IV / Glioblastoma)."},
    "meningioma": {"icon": "🔵", "urgency": "medium",
                   "desc": "Forms on the meninges. Usually benign and slow-growing."},
    "notumor":    {"icon": "🟢", "urgency": "low",
                   "desc": "No tumor detected. MRI scan appears normal."},
    "other":      {"icon": "🟡", "urgency": "medium",
                   "desc": "Abnormality detected outside primary categories. Specialist evaluation recommended."},
    "pituitary":  {"icon": "🟣", "urgency": "medium",
                   "desc": "Tumor in the pituitary gland. Most are benign adenomas."}
}

URGENCY = {
    "high":   ("⚠️ Requires urgent medical attention", "urg-high"),
    "medium": ("ℹ️ Consult a specialist soon", "urg-medium"),
    "low":    ("✅ No immediate concern detected", "urg-low")
}

# ─────────────────────────────────────────────
# MODEL LOADING  
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model from Drive (first run only)..."):
            try:
                url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
                gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
            except Exception as e:
                st.error(f"Download failed: {e}")
                return None

    if not os.path.exists(MODEL_PATH):
        st.error("Model download failed.")
        return None

    try:
        try:
            from tensorflow.keras.layers import Dense as _Dense
        except ImportError:
            from keras.layers import Dense as _Dense

        class PatchedDense(_Dense):
            @classmethod
            def from_config(cls, config):
                config.pop("quantization_config", None)
                return super().from_config(config)

        try:
            model = tf.keras.models.load_model(
                MODEL_PATH, compile=False,
                custom_objects={"Dense": PatchedDense}
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-5),
                loss="categorical_crossentropy", metrics=["accuracy"]
            )
        except Exception:
            import keras as keras_standalone
            model = keras_standalone.models.load_model(
                MODEL_PATH, compile=False,
                custom_objects={"Dense": PatchedDense}
            )
            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy", metrics=["accuracy"]
            )
        return model
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

def load_metadata():
    labels = DEFAULT_LABELS.copy()
    info   = DEFAULT_MODEL_INFO.copy()
    for path in ["class_labels.json", os.path.join("models", "class_labels.json")]:
        if os.path.exists(path):
            with open(path) as f: labels = json.load(f)
            break
    for path in ["model_info.json", os.path.join("models", "model_info.json")]:
        if os.path.exists(path):
            with open(path) as f: info = json.load(f)
            break
    return labels, info

# ─────────────────────────────────────────────
# PREPROCESSING  
# ─────────────────────────────────────────────
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = arr[..., ::-1]
    arr[..., 0] -= 103.939
    arr[..., 1] -= 116.779
    arr[..., 2] -= 123.68
    return np.expand_dims(arr, 0)

# ─────────────────────────────────────────────
# GRADCAM  
# ─────────────────────────────────────────────
def gradcam(model, arr, layer="block5_conv3"):
    try:
        try:
            Model = tf.keras.Model
        except AttributeError:
            from keras import Model
        gm = Model(inputs=model.inputs,
                   outputs=[model.get_layer(layer).output, model.output])
        with tf.GradientTape() as tape:
            conv_out, preds = gm(arr)
            idx   = tf.argmax(preds[0])
            score = preds[:, idx]
        grads  = tape.gradient(score, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        hm     = conv_out[0] @ pooled[..., tf.newaxis]
        hm     = tf.squeeze(hm)
        hm     = tf.maximum(hm, 0) / (tf.math.reduce_max(hm) + 1e-8)
        return hm.numpy()
    except:
        return None

def overlay(img: Image.Image, hm: np.ndarray, alpha=0.4):
    hm_r = np.array(Image.fromarray(np.uint8(255 * hm)).resize(IMAGE_SIZE))
    try:
        cmap = plt.colormaps["jet"]
    except AttributeError:
        cmap = mpl_cm.get_cmap("jet")
    colored = np.uint8(cmap(hm_r)[:, :, :3] * 255)
    orig    = np.array(img.convert("RGB").resize(IMAGE_SIZE), dtype=np.float32)
    ov      = np.uint8(orig * (1 - alpha) + colored * alpha)
    return Image.fromarray(ov), Image.fromarray(colored)

# ─────────────────────────────────────────────
# CHART  — dark theme
# ─────────────────────────────────────────────
def prob_chart(probs, labels):
    names  = [labels[str(i)] for i in range(len(probs))]
    colors = [CLASS_COLORS.get(n, "#888") for n in names]
    pairs  = sorted(zip(probs, names, colors), reverse=True)
    ps, ns, cs = zip(*pairs)

    fig, ax = plt.subplots(figsize=(5.5, 3), facecolor="#12121a")
    ax.set_facecolor("#12121a")
    bars = ax.barh(ns, [p * 100 for p in ps], color=cs,
                   edgecolor="none", linewidth=0, height=0.55)
    ax.set_xlabel("Confidence (%)", fontsize=8, color="#666", fontfamily="monospace")
    ax.set_xlim([0, 120])
    ax.tick_params(axis="y", labelsize=8.5, colors="#aaa", length=0)
    ax.tick_params(axis="x", labelsize=7, colors="#555", length=0)
    ax.spines[:].set_visible(False)
    ax.grid(axis="x", color=(1, 1, 1, 0.04), linewidth=0.5, linestyle="-")
    for bar, p, c in zip(bars, ps, cs):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", fontsize=8, fontweight="bold",
                color=c, fontfamily="monospace")
    plt.tight_layout(pad=0.5)
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <div style='font-family:"Bebas Neue",sans-serif; font-size:1.8rem;
                    letter-spacing:0.1em; color:#fff;'>NEUROSCAN</div>
        <div style='font-family:"JetBrains Mono",monospace; font-size:0.6rem;
                    letter-spacing:0.25em; color:#e50914; text-transform:uppercase;'>
            AI · Settings
        </div>
    </div>
    <hr style='border-color:rgba(229,9,20,0.2); margin: 0.5rem 0 1rem;'>
    """, unsafe_allow_html=True)

    conf_thresh  = st.slider("Confidence Threshold", 0.30, 0.99, 0.50, 0.01)
    show_gradcam = st.checkbox("Show GradCAM Heatmap", value=True)
    show_probs   = st.checkbox("Show All Probabilities", value=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:"JetBrains Mono",monospace; font-size:0.6rem;
                letter-spacing:0.2em; color:rgba(229,9,20,0.8); text-transform:uppercase;
                margin-bottom:0.75rem;'>Model Performance</div>
    """, unsafe_allow_html=True)

    _, mi = load_metadata()
    st.markdown(f"""
    <div class='metric-box'>
        <div class='metric-label'>Overall Accuracy</div>
        <div class='metric-value'>{mi.get('overall_accuracy',0)*100:.1f}%</div>
    </div>
    <div class='metric-box'>
        <div class='metric-label'>Macro AUC-ROC</div>
        <div class='metric-value'>{mi.get('macro_auc',0):.4f}</div>
    </div>
    <div class='metric-box'>
        <div class='metric-label'>Cohen's Kappa</div>
        <div class='metric-value'>{mi.get('kappa',0):.4f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-family:"JetBrains Mono",monospace; font-size:0.6rem;
                letter-spacing:0.2em; color:rgba(255,255,255,0.3); text-transform:uppercase;
                margin: 1rem 0 0.5rem;'>Per-Class Accuracy</div>
    """, unsafe_allow_html=True)

    for cls, acc in mi.get("per_class_accuracy", {}).items():
        col  = CLASS_COLORS.get(cls, "#888")
        icon = CLASS_INFO.get(cls, {}).get("icon", "")
        pct  = acc * 100
        st.markdown(f"""
        <div style='margin-bottom:0.5rem;'>
            <div style='display:flex;justify-content:space-between;align-items:center;
                        font-size:0.78rem; margin-bottom:3px;'>
                <span style='color:{col}; font-weight:600;'>{icon} {cls}</span>
                <span style='font-family:"JetBrains Mono",monospace; font-size:0.7rem;
                             color:{col};'>{pct:.1f}%</span>
            </div>
            <div style='background:rgba(255,255,255,0.06); border-radius:3px; height:3px;'>
                <div style='background:{col}; width:{pct}%; height:100%;
                            border-radius:3px; box-shadow:0 0 8px {col}55;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:"JetBrains Mono",monospace; font-size:0.6rem;
                letter-spacing:0.2em; color:rgba(255,255,255,0.3); text-transform:uppercase;
                margin-bottom:0.5rem;'>Classification Labels</div>
    """, unsafe_allow_html=True)
    for cls, ci in CLASS_INFO.items():
        st.markdown(
            f"<div style='font-size:0.8rem; padding:0.25rem 0; color:rgba(255,255,255,0.6);'>"
            f"{ci['icon']} <strong style='color:rgba(255,255,255,0.85);'>{cls.capitalize()}</strong>"
            f" — {ci['urgency']}</div>",
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────
# MAIN 
# ─────────────────────────────────────────────
st.markdown("""
<div class='ns-hero'>
    <div class='ns-eyebrow'>Neuro Fusion · Diagnostic AI</div>
    <div class='ns-title'>Brain Tumor<br>Detection</div>
    <div class='ns-subtitle'>VGG16 + Squeeze-and-Excitation Attention · Two-Phase Fine-Tuning</div>
    <div class='ns-badge-row'>
        <span class='ns-badge ns-badge-red'>VGG-16</span>
        <span class='ns-badge ns-badge-cyan'>GradCAM</span>
        <span class='ns-badge ns-badge-gold'>98% Accuracy</span>
        <span class='ns-badge ns-badge-red'>5-Class</span>
        <span class='ns-badge ns-badge-cyan'>MRI Analysis</span>
    </div>
</div>
<div class='ns-divider'></div>
""", unsafe_allow_html=True)

# Load model
with st.spinner("Initializing neural network..."):
    model = load_model()

if model is None:
    st.stop()

st.success("✅  Model ready — upload a scan to begin analysis")
st.markdown("<div class='ns-divider'></div>", unsafe_allow_html=True)

labels, model_info = load_metadata()

# ─────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────
uploaded = st.file_uploader(
    "📤  Upload Brain MRI Scan  ·  JPG / PNG",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible"
)

# ─────────────────────────────────────────────
# PRE-UPLOAD STATE
# ─────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div class='ns-step-grid'>
        <div class='ns-step'>
            <div class='ns-step-num'>01</div>
            <div class='ns-step-icon'>📤</div>
            <div class='ns-step-title'>Upload MRI</div>
            <div class='ns-step-desc'>JPG or PNG brain MRI scan from any modality</div>
        </div>
        <div class='ns-step'>
            <div class='ns-step-num'>02</div>
            <div class='ns-step-icon'>⚡</div>
            <div class='ns-step-title'>AI Analysis</div>
            <div class='ns-step-desc'>VGG16 deep network classifies in milliseconds</div>
        </div>
        <div class='ns-step'>
            <div class='ns-step-num'>03</div>
            <div class='ns-step-icon'>📊</div>
            <div class='ns-step-title'>Results + GradCAM</div>
            <div class='ns-step-desc'>Visual heatmap highlights focus regions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='ns-disclaimer'>
        <span style='font-size:1.1rem;'>⚕️</span>
        <span><strong style='color:rgba(245,197,24,0.9);'>Medical Disclaimer:</strong>
        For research and educational purposes only.
        Not a substitute for professional medical diagnosis.</span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# POST-UPLOAD — INFERENCE + RESULTS
# ─────────────────────────────────────────────
else:
    orig_img = Image.open(uploaded)

    with st.spinner("🔬  Running inference..."):
        arr        = preprocess(orig_img)
        probs      = model.predict(arr, verbose=0)[0]
        pred_idx   = int(np.argmax(probs))
        pred_class = labels[str(pred_idx)]
        confidence = float(probs[pred_idx])

    ci    = CLASS_INFO[pred_class]
    color = CLASS_COLORS[pred_class]
    urg   = URGENCY[ci["urgency"]]

    # ── Result card ──
    glow_map = {
        "high": "rgba(229,9,20,0.08)", "medium": "rgba(0,212,255,0.06)", "low": "rgba(0,230,118,0.06)"
    }
    st.markdown(f"""
    <div class='ns-result-card'
         style='--result-color:{color}; --result-glow:{glow_map[ci["urgency"]]};'>
        <div class='ns-result-icon-wrap'>{ci['icon']}</div>
        <div class='ns-result-body'>
            <div class='ns-result-label'>{pred_class.upper()}</div>
            <div class='ns-result-desc'>{ci['desc']}</div>
        </div>
        <div class='ns-conf-block'>
            <div class='ns-conf-value'>{confidence*100:.0f}<span style='font-size:1.5rem;'>%</span></div>
            <div class='ns-conf-label'>Confidence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Urgency banner ──
    st.markdown(f"""
    <div class='ns-urgency {urg[1]}'>
        {urg[0]}
    </div>
    """, unsafe_allow_html=True)

    if confidence < conf_thresh:
        st.warning(
            f"⚠️  Low confidence ({confidence*100:.1f}%) — below threshold "
            f"{conf_thresh*100:.0f}%. Consider a clearer scan or specialist review."
        )

    with st.expander("📖  About this prediction", expanded=True):
        st.markdown(
            f"**{ci['icon']} {pred_class.capitalize()}:** {ci['desc']}"
        )

    st.markdown("<div class='ns-divider'></div>", unsafe_allow_html=True)

    # ── GradCAM images ──
    if show_gradcam:
        with st.spinner("🎨  Generating GradCAM visualization..."):
            hm = gradcam(model, arr)

        if hm is not None:
            ov_img, heat_img = overlay(orig_img, hm)
            c1, c2, c3 = st.columns(3, gap="medium")
            for col_obj, img_obj, lbl in [
                (c1, orig_img.resize(IMAGE_SIZE), "Original MRI"),
                (c2, heat_img, "GradCAM Heatmap"),
                (c3, ov_img, "Overlay")
            ]:
                with col_obj:
                    st.markdown(f"""
                    <div style='text-align:center; margin-bottom:0.5rem;'>
                        <span class='ns-img-label'>{lbl}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(img_obj, use_container_width=True)

            st.markdown("""
            <div style='text-align:center; margin-top:0.5rem;'>
                <span style='font-family:"JetBrains Mono",monospace; font-size:0.65rem;
                             letter-spacing:0.1em; color:rgba(255,255,255,0.3);'>
                🔴 Red/yellow zones = regions the model weighted most heavily
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.image(orig_img.resize(IMAGE_SIZE), width=300)
    else:
        st.image(orig_img.resize(IMAGE_SIZE), caption="Uploaded MRI", width=300)

    # ── Probabilities ──
    if show_probs:
        st.markdown("<div class='ns-divider'></div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='ns-section-head'>📊 Class Probabilities</div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([1.4, 1], gap="large")
        with c1:
            fig = prob_chart(probs, labels)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with c2:
            sorted_probs = sorted(
                [(labels[str(i)], float(probs[i])) for i in range(len(probs))],
                key=lambda x: x[1], reverse=True
            )
            for cls, prob in sorted_probs:
                is_top  = cls == pred_class
                c       = CLASS_COLORS.get(cls, "#888")
                icon    = CLASS_INFO[cls]['icon']
                top_cls = "is-top" if is_top else ""
                st.markdown(f"""
                <div class='ns-prob-row {top_cls}'>
                    <span style='font-size:1rem;'>{icon}</span>
                    <span class='ns-prob-name'
                          style='color:{"var(--text-primary)" if is_top else "var(--text-secondary)"};
                                 font-weight:{"700" if is_top else "400"};'>
                        {cls}
                    </span>
                    <span class='ns-prob-val' style='color:{c};'>{prob*100:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)

    # ── Download report ──
    st.markdown("<div class='ns-divider'></div>", unsafe_allow_html=True)
    report = f"""BRAIN TUMOR MRI CLASSIFICATION REPORT
======================================
Predicted Class  : {pred_class.upper()}
Confidence       : {confidence*100:.2f}%
Urgency Level    : {ci['urgency'].upper()}

All Class Probabilities:
{chr(10).join(f"  {labels[str(i)]:<14}: {probs[i]*100:.2f}%" for i in range(len(probs)))}

Model Performance:
  Overall Accuracy : {model_info.get('overall_accuracy',0)*100:.2f}%
  Macro AUC-ROC    : {model_info.get('macro_auc',0):.4f}
  Cohen's Kappa    : {model_info.get('kappa',0):.4f}

DISCLAIMER: For research and educational use only.
Not a substitute for professional medical diagnosis.
"""
    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        st.download_button(
            "📥  Download Report",
            data=report,
            file_name=f"neuroscan_{pred_class}_report.txt",
            mime="text/plain"
        )

    st.markdown("""
    <div class='ns-disclaimer' style='margin-top:1rem;'>
        <span style='font-size:1.1rem;'>⚕️</span>
        <span><strong style='color:rgba(245,197,24,0.9);'>Medical Disclaimer:</strong>
        For research and educational purposes only. Not a substitute for professional medical diagnosis.
        Always consult a qualified radiologist or neurologist.</span>
    </div>
    """, unsafe_allow_html=True)
