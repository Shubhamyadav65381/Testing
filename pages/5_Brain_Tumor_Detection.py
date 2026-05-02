"""
pages/5_Brain_Tumor_Detection.py
Brain Tumor MRI Classification — integrated into Neuro Fusion system
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

# TF 2.16+ moved Keras to a standalone package; handle both cases
try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    import keras
    import tensorflow as tf

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

# ─────────────────────────────────────────────
# LOAD SHARED CSS  (same style_v1.css your other pages use)
# ─────────────────────────────────────────────
css_path = None
for candidate in [
    os.path.join("utils", "style_v1.css"),
    os.path.join(os.path.dirname(__file__), '..', 'utils', 'style_v1.css'),
    os.path.join(os.path.dirname(__file__), '..', 'style_v1.css'),
]:
    if os.path.exists(candidate):
        css_path = candidate
        break

if css_path:
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Extra CSS for this page
st.markdown("""
<style>

/* ===== GLOBAL ===== */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ===== HEADER ===== */
.bt-header {
    font-size: 2.6rem;
    font-weight: 800;
    text-align: center;
    padding: 1rem 0;

    background: linear-gradient(90deg, #00c6ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;

    letter-spacing: 1px;
}

.bt-sub {
    font-size: 1rem;
    text-align: center;
    color: #bbbbbb;
    margin-bottom: 1.8rem;
}

/* ===== PREDICTION CARD ===== */
.pred-card {
    border-radius: 16px;
    padding: 1.6rem;
    text-align: center;
    margin: 1rem 0;

    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);

    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}

.pred-label {
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: 2px;
    margin-top: 0.5rem;
}

.pred-conf {
    font-size: 1rem;
    opacity: 0.9;
}

/* ===== METRIC BOX ===== */
.metric-box {
    background: rgba(255,255,255,0.05);
    border-left: 4px solid #00c6ff;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;

    backdrop-filter: blur(10px);
}

.metric-label {
    font-size: 0.75rem;
    color: #aaa;
}

.metric-value {
    font-size: 1.3rem;
    font-weight: 700;
    color: #ffffff;
}

/* ===== DISCLAIMER ===== */
.disclaimer {
    background: rgba(255, 193, 7, 0.1);
    border: 1px solid #ffc107;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #ffd54f;
}

/* ===== SUCCESS BOX ===== */
div[data-testid="stAlert"] {
    background: linear-gradient(135deg, #00c853, #64dd17);
    color: white !important;
    border-radius: 12px;
    padding: 12px;
    font-weight: 600;
}

/* ===== BUTTON ===== */
.stButton>button {
    background: linear-gradient(90deg, #0072ff, #00c6ff);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: 600;
}

.stButton>button:hover {
    opacity: 0.9;
}

/* ===== FILE UPLOADER ===== */
section[data-testid="stFileUploader"] {
    border: 2px dashed rgba(255,255,255,0.2);
    border-radius: 12px;
    padding: 1rem;
}

/* ===== SIDEBAR ===== */
section[data-testid="stSidebar"] {
    background: rgba(20,20,30,0.9);
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-thumb {
    background: #444;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)
# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
IMAGE_SIZE      = (224, 224)
MODEL_DRIVE_ID  = "1XQOEyJXJ62QYdvK5noT4w8RTXNwF5tsb"   # your Drive file ID
MODEL_PATH      = "best_model.h5"                          # cached locally at runtime

DEFAULT_LABELS = {
    "0": "glioma",
    "1": "meningioma",
    "2": "notumor",
    "3": "other",
    "4": "pituitary"
}

DEFAULT_MODEL_INFO = {
    "overall_accuracy": 0.98,
    "macro_auc":        0.99,
    "kappa":            0.97,
    "per_class_accuracy": {
        "glioma": 0.953, "meningioma": 0.957,
        "notumor": 0.995, "other": 0.995, "pituitary": 0.996
    }
}

CLASS_COLORS = {
    "glioma": "#E74C3C", "meningioma": "#3498DB",
    "notumor": "#2ECC71", "other": "#F39C12", "pituitary": "#9B59B6"
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
    "high":   ("⚠️ Requires urgent medical attention", "#ffebee", "#c62828"),
    "medium": ("ℹ️ Consult a specialist soon",          "#e8f4f8", "#1565c0"),
    "low":    ("✅ No immediate concern detected",       "#e8f5e9", "#2e7d32")
}

# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading Brain Tumor model from Drive (first run only)..."):
            try:
                # gdown 5.x: use fuzzy=True to handle all Drive URL formats
                url = f"https://drive.google.com/uc?id={MODEL_DRIVE_ID}"
                gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)
            except Exception as e:
                st.error(f"Download failed: {e}")
                return None

    if not os.path.exists(MODEL_PATH):
        st.error("Model download failed.")
        return None

    try:
        # PatchedDense strips quantization_config added by Keras 3.4+
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
                MODEL_PATH,
                compile=False,
                custom_objects={"Dense": PatchedDense}
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(1e-5),
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
        except Exception:
            # Fallback for keras standalone package
            import keras as keras_standalone
            model = keras_standalone.models.load_model(
                MODEL_PATH,
                compile=False,
                custom_objects={"Dense": PatchedDense}
            )
            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"]
            )
        return model
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

def load_metadata():
    labels = DEFAULT_LABELS.copy()
    info   = DEFAULT_MODEL_INFO.copy()

    # Check if class_labels.json exists anywhere in repo
    for path in ["class_labels.json",
                 os.path.join("models", "class_labels.json")]:
        if os.path.exists(path):
            with open(path) as f:
                labels = json.load(f)
            break

    for path in ["model_info.json",
                 os.path.join("models", "model_info.json")]:
        if os.path.exists(path):
            with open(path) as f:
                info = json.load(f)
            break

    return labels, info

# ─────────────────────────────────────────────
# PREPROCESSING  (must match training: vgg_preprocess)
# ─────────────────────────────────────────────
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = arr[..., ::-1]       # RGB → BGR
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
        gm = Model(
            inputs=model.inputs,
            outputs=[model.get_layer(layer).output, model.output]
        )
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
    hm_r    = np.array(Image.fromarray(np.uint8(255 * hm)).resize(IMAGE_SIZE))
    try:
        cmap = plt.colormaps["jet"]
    except AttributeError:
        cmap = mpl_cm.get_cmap("jet")  # fallback for older matplotlib
    colored = np.uint8(cmap(hm_r)[:, :, :3] * 255)
    orig    = np.array(img.convert("RGB").resize(IMAGE_SIZE), dtype=np.float32)
    ov      = np.uint8(orig * (1 - alpha) + colored * alpha)
    return Image.fromarray(ov), Image.fromarray(colored)

# ─────────────────────────────────────────────
# CONFIDENCE BAR CHART
# ─────────────────────────────────────────────
def prob_chart(probs, labels):
    names  = [labels[str(i)] for i in range(len(probs))]
    colors = [CLASS_COLORS.get(n, "#888") for n in names]
    pairs  = sorted(zip(probs, names, colors), reverse=True)
    ps, ns, cs = zip(*pairs)

    fig, ax = plt.subplots(figsize=(5, 2.8))
    bars = ax.barh(ns, [p * 100 for p in ps], color=cs, edgecolor="white", linewidth=0.8)
    ax.set_xlabel("Confidence (%)", fontsize=8)
    ax.set_xlim([0, 115])
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, p in zip(bars, ps):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{p*100:.1f}%", va="center", fontsize=7.5, fontweight="bold")
    plt.tight_layout()
    return fig

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Brain Tumor Settings")
    conf_thresh    = st.slider("Confidence Threshold", 0.30, 0.99, 0.50, 0.01)
    show_gradcam   = st.checkbox("Show GradCAM Heatmap", value=True)
    show_probs     = st.checkbox("Show All Probabilities", value=True)

    st.markdown("---")
    st.markdown("### 📊 Model Stats")
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

    st.markdown("**Per-Class Accuracy**")
    for cls, acc in mi.get("per_class_accuracy", {}).items():
        col = CLASS_COLORS.get(cls, "#888")
        icon = CLASS_INFO.get(cls, {}).get("icon", "")
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;"
            f"padding:2px 0;font-size:0.82rem;'>"
            f"<span style='color:{col};font-weight:600;'>{icon} {cls}</span>"
            f"<span style='font-weight:700;'>{acc*100:.1f}%</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### 🏷️ Classes")
    for cls, ci in CLASS_INFO.items():
        st.markdown(f"{ci['icon']} **{cls.capitalize()}**")

# ─────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────
st.markdown(
    "<div class='bt-header'>🧠 Brain Tumor MRI Detection</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='bt-sub'>VGG16 + Squeeze-and-Excitation Attention · Two-Phase Fine-Tuning</div>",
    unsafe_allow_html=True
)

# Load model
with st.spinner("Loading model..."):
    model = load_model()

if model is None:
    st.stop()

st.success("✅ Model ready!")
st.markdown("---")

labels, model_info = load_metadata()

uploaded = st.file_uploader(
    "📤 Upload Brain MRI Scan (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded is None:
    c1, c2, c3 = st.columns(3)
    c1.info("**📤 Step 1**\nUpload a brain MRI image")
    c2.info("**⚡ Step 2**\nModel classifies in seconds")
    c3.info("**📋 Step 3**\nReview prediction + GradCAM")
    st.markdown("""
    <div class='disclaimer'>
    ⚕️ <strong>Medical Disclaimer:</strong> For research and educational purposes only.
    Not a substitute for professional medical diagnosis.
    </div>
    """, unsafe_allow_html=True)

else:
    orig_img = Image.open(uploaded)

    with st.spinner("🔍 Analysing..."):
        arr        = preprocess(orig_img)
        probs      = model.predict(arr, verbose=0)[0]
        pred_idx   = int(np.argmax(probs))
        pred_class = labels[str(pred_idx)]
        confidence = float(probs[pred_idx])

    ci      = CLASS_INFO[pred_class]
    urg     = URGENCY[ci["urgency"]]
    color   = CLASS_COLORS[pred_class]

    # Prediction card
    st.markdown(f"""
    <div class='pred-card' style='background:linear-gradient(135deg,{color}dd,{color}66);'>
        <div style='font-size:2rem;'>{ci['icon']}</div>
        <div class='pred-label'>{pred_class}</div>
        <div class='pred-conf'>Confidence: {confidence*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Urgency badge
    st.markdown(
        f"<div style='background:{urg[1]};color:{urg[2]};padding:0.5rem 1rem;"
        f"border-radius:8px;font-size:0.88rem;font-weight:500;margin-bottom:0.8rem;'>"
        f"{urg[0]}</div>", unsafe_allow_html=True
    )

    if confidence < conf_thresh:
        st.warning(
            f"⚠️ Low confidence ({confidence*100:.1f}%) — below threshold "
            f"{conf_thresh*100:.0f}%. Consider clearer scan or specialist review."
        )

    with st.expander("📖 About this prediction", expanded=True):
        st.markdown(f"**{ci['icon']} {pred_class.capitalize()}:** {ci['desc']}")

    st.markdown("---")

    # Images
    if show_gradcam:
        with st.spinner("🎨 Generating GradCAM..."):
            hm = gradcam(model, arr)

        if hm is not None:
            ov_img, heat_img = overlay(orig_img, hm)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("**Original MRI**")
                st.image(orig_img.resize(IMAGE_SIZE), use_container_width=True)
            with c2:
                st.markdown("**GradCAM Heatmap**")
                st.image(heat_img, use_container_width=True)
            with c3:
                st.markdown("**Overlay**")
                st.image(ov_img, use_container_width=True)
            st.caption("🔴 Red/yellow = regions the model focused on most.")
        else:
            st.image(orig_img.resize(IMAGE_SIZE), width=300)
    else:
        st.image(orig_img.resize(IMAGE_SIZE), caption="Uploaded MRI", width=300)

    # Probabilities
    if show_probs:
        st.markdown("---")
        st.markdown("**📊 All Class Probabilities**")
        c1, c2 = st.columns([1.3, 1])
        with c1:
            fig = prob_chart(probs, labels)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with c2:
            for cls, prob in sorted(
                [(labels[str(i)], float(probs[i])) for i in range(len(probs))],
                key=lambda x: x[1], reverse=True
            ):
                is_top = cls == pred_class
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;padding:4px 0;'>"
                    f"<span>{CLASS_INFO[cls]['icon']}</span>"
                    f"<span style='flex:1;font-weight:{'700' if is_top else '400'};"
                    f"color:{'#1a1a2e' if is_top else '#555'};'>{cls}</span>"
                    f"<span style='font-weight:700;color:{CLASS_COLORS[cls]};'>"
                    f"{prob*100:.2f}%</span></div>",
                    unsafe_allow_html=True
                )

    # Download report
    st.markdown("---")
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
    st.download_button(
        "📥 Download Report (.txt)",
        data=report,
        file_name=f"brain_tumor_{pred_class}_report.txt",
        mime="text/plain"
    )
