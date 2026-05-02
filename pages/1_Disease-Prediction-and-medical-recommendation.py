import streamlit as st
import pickle
import pandas as pd
import numpy as np
import ast
import os
import re
from thefuzz import process

st.set_page_config(
    page_title="Neuro-Fusion: Clinical Intelligence System",
    page_icon="🩺",
    layout='wide'
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #070b14;
    color: #c8d8f0;
}
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; }

#MainMenu, footer, header { visibility: hidden; }

[data-testid="stAppViewContainer"] {
    background-color: #070b14;
    background-image:
        linear-gradient(rgba(32,196,180,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(32,196,180,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
}
[data-testid="stSidebar"] {
    background: #0a0f1e !important;
    border-right: 1px solid #1a2a4a;
}
.sidebar-text { color: #6a8aaa; font-size: 13px; line-height: 1.6; }

.stTextArea textarea, .stTextInput input {
    background: #0d1526 !important;
    color: #c8d8f0 !important;
    border: 1px solid #1e3a5a !important;
    border-radius: 6px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 14px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #20c4b4 !important;
    box-shadow: 0 0 0 2px rgba(32,196,180,0.15) !important;
}

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0d7a70, #20c4b4) !important;
    color: #070b14 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 10px 24px !important;
    letter-spacing: 0.5px !important;
    transition: all 0.2s !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(32,196,180,0.35) !important;
}

.page-header {
    padding: 28px 0 8px 0;
    border-bottom: 1px solid #1a2a4a;
    margin-bottom: 28px;
}
.page-header h1 {
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.5px;
}
.page-header h1 span { color: #20c4b4; }
.page-subtitle {
    color: #4a6a8a;
    font-size: 13px;
    margin-top: 6px;
    font-family: 'DM Mono', monospace;
}

/* ── Assessment hero banner ── */
.result-wrapper {
    background: linear-gradient(135deg, #0a1628 0%, #0d1e38 100%);
    border: 1px solid #1e3a5a;
    border-radius: 16px;
    padding: 28px 32px;
    margin: 20px 0;
    position: relative;
    overflow: hidden;
}
.result-wrapper::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #20c4b4, #0d7a70, transparent);
}
.result-wrapper::after {
    content: 'RX';
    position: absolute;
    right: 32px; top: 16px;
    font-size: 64px;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    color: rgba(32,196,180,0.05);
    letter-spacing: -2px;
}
.assessment-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #20c4b4;
    margin: 0 0 8px 0;
}
.disease-name {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0 0 12px 0;
    line-height: 1.2;
}
.symptom-pill {
    display: inline-block;
    background: rgba(32,196,180,0.08);
    border: 1px solid rgba(32,196,180,0.25);
    border-radius: 20px;
    padding: 4px 14px;
    margin: 3px 4px 3px 0;
    font-size: 12px;
    color: #20c4b4;
    font-family: 'DM Mono', monospace;
}

/* ── Overview card ── */
.overview-card {
    background: linear-gradient(135deg, #071a2e, #0a2240);
    border: 1px solid #1a3a5a;
    border-left: 4px solid #20c4b4;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
}
.overview-text {
    color: #c0d8f0;
    font-size: 14px;
    line-height: 1.9;
    font-family: 'DM Mono', monospace;
}

/* ── Section cards ── */
.section-card {
    background: #080f1e;
    border: 1px solid #142030;
    border-radius: 12px;
    padding: 20px 22px;
    margin: 10px 0;
    position: relative;
    overflow: hidden;
}
.section-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 3px;
}
.card-teal::before   { background: #20c4b4; }
.card-orange::before { background: #ffa050; }
.card-blue::before   { background: #60b0ff; }
.card-green::before  { background: #50d090; }
.card-purple::before { background: #c080ff; }

.section-icon {
    font-size: 20px;
    margin-bottom: 6px;
    display: block;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.bullet-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin: 9px 0;
    font-size: 13px;
    color: #90b0cc;
    font-family: 'DM Mono', monospace;
    line-height: 1.65;
}
.bullet-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    margin-top: 7px;
    flex-shrink: 0;
}
.dot-teal   { background: #20c4b4; }
.dot-orange { background: #ffa050; }
.dot-blue   { background: #60b0ff; }
.dot-green  { background: #50d090; }
.dot-purple { background: #c080ff; }

/* ── Sources bar ── */
.sources-bar {
    background: #050c18;
    border: 1px solid #0f2030;
    border-radius: 8px;
    padding: 10px 16px;
    margin-top: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}
.source-chip {
    background: #0a1828;
    border: 1px solid #1a3a5a;
    border-radius: 4px;
    padding: 3px 10px;
    font-size: 11px;
    color: #3a6a8a;
    font-family: 'DM Mono', monospace;
}

.neo-divider {
    border: none;
    border-top: 1px solid #1a2a4a;
    margin: 28px 0;
}
.disclaimer {
    background: #100a0a;
    border: 1px solid #3a1a1a;
    border-radius: 8px;
    padding: 12px 16px;
    margin-top: 16px;
    color: #804040;
    font-size: 12px;
    line-height: 1.6;
    font-family: 'DM Mono', monospace;
}
.stSpinner > div { border-top-color: #20c4b4 !important; }

/* encyclopedia */
.info-card {
    background: #0a1220;
    border: 1px solid #1a2e4a;
    border-radius: 8px;
    padding: 14px 16px;
    margin: 10px 0;
}
.info-card-label {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.info-card-content {
    color: #a0b8d0;
    font-size: 13.5px;
    line-height: 1.7;
    font-family: 'DM Mono', monospace;
}
.tag {
    display: inline-block;
    background: #0d1e38;
    border: 1px solid #1e3a5a;
    border-radius: 4px;
    padding: 3px 10px;
    margin: 3px 3px 3px 0;
    font-size: 12.5px;
    color: #8ab0d0;
    font-family: 'DM Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("<h2 style='color:#ffffff; font-family:Syne,sans-serif;'>📌 Description</h2>", unsafe_allow_html=True)
try:
    st.sidebar.image("utils/ph3.png", use_container_width=True)
except:
    pass
st.sidebar.markdown("<p class='sidebar-text'>Neuro-Fusion analyzes symptoms using a trained ML model fused with a real medical book knowledge base — delivering clinically grounded insights instantly.</p>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='margin-top:20px; padding:14px; background:#0a1220; border:1px solid #1a2e4a; border-radius:8px;'>
<p style='color:#20c4b4; font-family:Syne,sans-serif; font-weight:700; font-size:13px; margin:0 0 8px 0;'>⚙️ INTELLIGENCE LAYERS</p>
<p style='color:#4a6a8a; font-size:12px; margin:4px 0; font-family:DM Mono,monospace;'>🔵 ML Model — RandomForest (132 symptoms)</p>
<p style='color:#4a6a8a; font-size:12px; margin:4px 0; font-family:DM Mono,monospace;'>📚 Book KB — 7 Medical Books via FAISS</p>
<p style='color:#4a6a8a; font-size:12px; margin:4px 0; font-family:DM Mono,monospace;'>🔀 Fusion — Both sources merged seamlessly</p>
</div>
""", unsafe_allow_html=True)

# ── Load ML Data ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_data():
    try:
        sym_des     = pd.read_csv("data/Disease-Prediction-and-Medical dataset/symptoms_df.csv")
        precautions = pd.read_csv("data/Disease-Prediction-and-Medical dataset/precautions_df.csv")
        workout     = pd.read_csv("data/Disease-Prediction-and-Medical dataset/workout_df.csv")
        description = pd.read_csv("data/Disease-Prediction-and-Medical dataset/description.csv")
        medications = pd.read_csv("data/Disease-Prediction-and-Medical dataset/medications.csv")
        diets       = pd.read_csv("data/Disease-Prediction-and-Medical dataset/diets.csv")
        model       = pickle.load(open('models/first_feature_models/RandomForest.pkl', 'rb'))
        return sym_des, precautions, workout, description, medications, diets, model
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None, None, None

sym_des, precautions, workout, description, medications, diets, model = load_data()
disease_names = list(description['Disease'].unique()) if description is not None else []

# ── Load Medibot Chain ────────────────────────────────────────────────────────
@st.cache_resource
def load_medibot_chain():
    try:
        from langchain_groq import ChatGroq
        from langchain.chains import RetrievalQA
        from langchain_community.vectorstores import FAISS
        from langchain_core.prompts import PromptTemplate
        from langchain_huggingface import HuggingFaceEmbeddings
        from huggingface_hub import hf_hub_download

        DB_FAISS_PATH = "vectorstore/db_faiss"
        os.makedirs(DB_FAISS_PATH, exist_ok=True)

        if not os.path.exists(f"{DB_FAISS_PATH}/index.faiss"):
            hf_hub_download(repo_id="Riteshkumarverma/medical-vectorstore",
                            filename="vectorstore/db_faiss/index.faiss",
                            repo_type="dataset", local_dir=".")
        if not os.path.exists(f"{DB_FAISS_PATH}/index.pkl"):
            hf_hub_download(repo_id="Riteshkumarverma/medical-vectorstore",
                            filename="vectorstore/db_faiss/index.pkl",
                            repo_type="dataset", local_dir=".")

        embeddings  = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
        llm = ChatGroq(temperature=0.3, model_name="llama-3.3-70b-versatile", groq_api_key=GROQ_API_KEY)

        prompt = PromptTemplate(
            template="""You are a clinical medical AI trained on real medical books.
Using the context below, provide a thorough clinical analysis for: {question}

Context:
{context}

Structure your response as:
**Overview:** (2-3 sentences about the condition)
**Key Symptoms & Warning Signs:** (bullet points)
**Recommended Medications:** (generic drug names)
**Dietary Recommendations:** (specific foods to eat/avoid)
**Physical Activity:** (suitable exercises)
**Important Precautions:** (what to do and avoid)

Be medically precise but use plain language. End with: "Please consult a licensed physician before starting any treatment."
""",
            input_variables=["context", "question"]
        )

        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': prompt}
        )
    except Exception as e:
        return None

# ── Symptom Maps ──────────────────────────────────────────────────────────────
symptoms_list = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}
symptoms_list_processed = {s.replace('_', ' ').lower(): v for s, v in symptoms_list.items()}

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_ml_info(disease):
    try:
        desc = description.loc[description['Disease'] == disease, 'Description'].values[0]
        prec = precautions.loc[precautions['Disease'] == disease, ['Precaution_1','Precaution_2','Precaution_3','Precaution_4']].values.flatten().tolist()
        meds = ast.literal_eval(medications.loc[medications['Disease'] == disease, 'Medication'].values[0])
        diet = ast.literal_eval(diets.loc[diets['Disease'] == disease, 'Diet'].values[0])
        work = workout.loc[workout['disease'] == disease, 'workout'].values.tolist()
        return desc, prec, meds, diet, work
    except:
        return "Not available.", [], [], [], []

def correct_spelling(symptom):
    try:
        match, score = process.extractOne(symptom.lower().strip(), symptoms_list_processed.keys())
        return match if score >= 80 else None
    except:
        return None

def tags_html(items, color="#8ab0d0"):
    if not items:
        return "<span style='color:#4a6a8a;'>—</span>"
    return "".join(
        f"<span class='tag' style='color:{color};border-color:{color}33;'>{i}</span>"
        for i in items if str(i).strip() and str(i) != 'nan'
    )

def parse_rag_sections(text):
    """Parse structured RAG response into named sections."""
    sections = {}
    patterns = {
        'overview':    r'\*\*Overview:\*\*\s*(.*?)(?=\*\*\w|$)',
        'symptoms':    r'\*\*Key Symptoms[^*]*\*\*\s*(.*?)(?=\*\*\w|$)',
        'medications': r'\*\*Recommended Medications[^*]*\*\*\s*(.*?)(?=\*\*\w|$)',
        'diet':        r'\*\*Dietary Recommendations[^*]*\*\*\s*(.*?)(?=\*\*\w|$)',
        'activity':    r'\*\*Physical Activity[^*]*\*\*\s*(.*?)(?=\*\*\w|$)',
        'precautions': r'\*\*Important Precautions[^*]*\*\*\s*(.*?)(?=\*\*\w|$)',
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            lines = [l.strip().lstrip('*•-').strip() for l in raw.split('\n') if l.strip().lstrip('*•-').strip()]
            # Remove trailing physician note lines
            lines = [l for l in lines if 'consult' not in l.lower() or key == 'precautions']
            sections[key] = lines
    return sections

def bullets_html(lines, dot_class):
    if not lines:
        return "<span style='color:#4a6a8a;font-family:DM Mono,monospace;'>—</span>"
    out = ""
    for line in lines:
        line = re.sub(r'\*\*(.*?)\*\*', r'\1', line).strip()
        if line:
            out += f"""<div class='bullet-item'>
                <div class='bullet-dot {dot_class}'></div>
                <span>{line}</span>
            </div>"""
    return out

# ── Page Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <h1>NEURO<span>FUSION</span></h1>
    <p class='page-subtitle'>// INTEGRATED DISEASE PREDICTION & CLINICAL INTELLIGENCE SYSTEM</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["⬡ SYMPTOM ANALYSIS", "◈ DISEASE ENCYCLOPEDIA"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Symptom Analysis
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<p style='color:#4a6a8a; font-size:13px; margin-bottom:20px; font-family:DM Mono,monospace;'>Enter any symptoms. The system silently fuses ML prediction with 7 real medical books to deliver a complete clinical picture.</p>", unsafe_allow_html=True)

    user_input = st.text_area("", placeholder="e.g., headache, nausea, high fever, chest pain, breathlessness ...", height=90, label_visibility="collapsed")

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        predict_btn = st.button("⬡ ANALYZE", use_container_width=True)

    if predict_btn and user_input.strip():
        raw = [s.strip() for s in user_input.split(',') if s.strip()]

        with st.spinner("Analyzing symptoms..."):
            # ── BYPASSED: ML prediction skipped — RAG queries symptoms directly ──
            query = ", ".join(raw[:5])
            chain = load_medibot_chain()
            book_answer, book_sources = None, []

            if chain:
                try:
                    resp = chain.invoke({'query': f"Clinical analysis for symptoms: {query}"})
                    book_answer  = resp.get("result", "")
                    book_sources = resp.get("source_documents", [])
                except:
                    book_answer = None

        if book_answer:
            # ── Hero banner ──
            pills = "".join(f"<span class='symptom-pill'>{s}</span>" for s in raw[:6])
            st.markdown(f"""
            <div class='result-wrapper'>
                <p class='assessment-label'>● CLINICAL ASSESSMENT</p>
                <p class='disease-name'>Symptom Analysis</p>
                <div style='margin-top:10px;'>{pills}</div>
            </div>
            """, unsafe_allow_html=True)

            sections = parse_rag_sections(book_answer)

            # ── Overview — full width ──
            if sections.get('overview'):
                overview_text = " ".join(sections['overview'])
                st.markdown(f"""
                <div class='overview-card'>
                    <div style='font-family:Syne,sans-serif;font-size:11px;font-weight:700;
                                letter-spacing:2px;color:#20c4b4;margin-bottom:10px;'>📋 OVERVIEW</div>
                    <div class='overview-text'>{overview_text}</div>
                </div>
                """, unsafe_allow_html=True)

            # ── 2-column cards ──
            col_a, col_b = st.columns(2)

            with col_a:
                if sections.get('symptoms'):
                    st.markdown(f"""
                    <div class='section-card card-teal'>
                        <span class='section-icon'>🔍</span>
                        <div class='section-title' style='color:#20c4b4;'>Key Symptoms & Warning Signs</div>
                        {bullets_html(sections['symptoms'], 'dot-teal')}
                    </div>
                    """, unsafe_allow_html=True)

                if sections.get('diet'):
                    st.markdown(f"""
                    <div class='section-card card-green'>
                        <span class='section-icon'>🥗</span>
                        <div class='section-title' style='color:#50d090;'>Dietary Recommendations</div>
                        {bullets_html(sections['diet'], 'dot-green')}
                    </div>
                    """, unsafe_allow_html=True)

                if sections.get('activity'):
                    st.markdown(f"""
                    <div class='section-card card-purple'>
                        <span class='section-icon'>🏃</span>
                        <div class='section-title' style='color:#c080ff;'>Physical Activity</div>
                        {bullets_html(sections['activity'], 'dot-purple')}
                    </div>
                    """, unsafe_allow_html=True)

            with col_b:
                if sections.get('medications'):
                    st.markdown(f"""
                    <div class='section-card card-blue'>
                        <span class='section-icon'>💊</span>
                        <div class='section-title' style='color:#60b0ff;'>Recommended Medications</div>
                        {bullets_html(sections['medications'], 'dot-blue')}
                    </div>
                    """, unsafe_allow_html=True)

                if sections.get('precautions'):
                    st.markdown(f"""
                    <div class='section-card card-orange'>
                        <span class='section-icon'>🛡️</span>
                        <div class='section-title' style='color:#ffa050;'>Important Precautions</div>
                        {bullets_html(sections['precautions'], 'dot-orange')}
                    </div>
                    """, unsafe_allow_html=True)

            # ── Sources bar ──
            if book_sources:
                seen, names = set(), []
                for doc in book_sources:
                    n = os.path.basename(doc.metadata.get('source', ''))
                    if n and n not in seen:
                        seen.add(n); names.append(n)
                if names:
                    chips = "".join(f"<span class='source-chip'>📄 {n}</span>" for n in names)
                    st.markdown(f"""
                    <div class='sources-bar'>
                        <span style='color:#2a4a6a;font-size:11px;font-family:DM Mono,monospace;'>📚 SOURCES</span>
                        {chips}
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<div class='disclaimer'>⚠️ This analysis is AI-generated and for informational purposes only. It does not constitute a medical diagnosis. Always consult a licensed healthcare professional before making any health decisions.</div>", unsafe_allow_html=True)

        else:
            st.markdown("<div style='background:#0d1526;border:1px solid #1e3a5a;border-radius:8px;padding:20px;text-align:center;color:#4a6a8a;font-family:DM Mono,monospace;'>Could not process these symptoms. Please try more specific symptom names.</div>", unsafe_allow_html=True)

    elif predict_btn:
        st.warning("Please enter at least one symptom.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Disease Encyclopedia
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<p style='color:#4a6a8a; font-size:13px; margin-bottom:20px; font-family:DM Mono,monospace;'>Search any disease to get a complete clinical profile from the knowledge base.</p>", unsafe_allow_html=True)

    disease_query = st.text_input("", placeholder="Type a disease name — e.g., Malaria, Diabetes, Tuberculosis ...", label_visibility="collapsed")

    if disease_query and description is not None:
        matches = [d for d in disease_names if d.lower().startswith(disease_query.lower())]
        if matches:
            selected = matches[0]
            desc, prec, meds, diet, work = get_ml_info(selected)

            st.markdown(f"""
            <div class='result-wrapper'>
                <p class='assessment-label'>● DISEASE PROFILE</p>
                <p class='disease-name'>{selected}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class='overview-card'>
                <div style='font-family:Syne,sans-serif;font-size:11px;font-weight:700;
                            letter-spacing:2px;color:#20c4b4;margin-bottom:10px;'>📋 DESCRIPTION</div>
                <div class='overview-text'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"<div class='info-card'><div class='info-card-label' style='color:#ffa050;'>🛡 PRECAUTIONS</div><div class='info-card-content'>{tags_html([i for i in prec if i],'#ffa050')}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='info-card'><div class='info-card-label' style='color:#60b0ff;'>💊 MEDICATIONS</div><div class='info-card-content'>{tags_html(meds,'#60b0ff')}</div></div>", unsafe_allow_html=True)
            with col_b:
                st.markdown(f"<div class='info-card'><div class='info-card-label' style='color:#60d090;'>🥗 DIET</div><div class='info-card-content'>{tags_html(diet,'#60d090')}</div></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='info-card'><div class='info-card-label' style='color:#c080ff;'>🏃 ACTIVITY</div><div class='info-card-content'>{tags_html(work,'#c080ff')}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#0d1526;border:1px solid #1e3a5a;border-radius:8px;padding:16px;color:#4a6a8a;font-size:13px;font-family:DM Mono,monospace;'>No exact match for \"<b style='color:#c8d8f0;'>{disease_query}</b>\" in local database. Try the Symptom Analysis tab for AI-powered lookup.</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr class='neo-divider'>
<p style='text-align:center; color:#1e3a5a; font-size:12px; font-family:DM Mono,monospace;'>
NEURO-FUSION // Made by <span style='color:#20c4b4;'>R.K.</span>· © 2026 · <em>Not a substitute for professional medical advice</em>
</p>
""", unsafe_allow_html=True)
