import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ┌────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# └────────────────────────────────────────
st.set_page_config(
    page_title="FlightSense · Predicción de Retrasos",
    page_icon=None, # Removed emoji
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ┌────────────────────────
#  LOAD MODEL & ENCODERS
# └─────────────────────────
try:
    best_gradient_boosting_model   = joblib.load("best_gradient_boosting_model.joblib")
    one_hot_encoder_Airline        = joblib.load("one_hot_encoder_Airline.joblib")
    one_hot_encoder_DepTime_label  = joblib.load("one_hot_encoder_DepTime_label.joblib")
    one_hot_encoder_Model          = joblib.load("one_hot_encoder_Model.joblib")
except Exception as e:
    st.error(f"Error al cargar el modelo o encoders: {e}")
    st.stop()

# ┌───────────
#  GLOBAL CSS
# └───────────
st.markdown("""
<style>
/* ── Google Fonts ─────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500&family=Plus+Jakarta+Sans:wght@300;400;500;600&display=swap');

/* ── Design tokens ───────────────────── */
:root {
    --bg:        #0b0e17;
    --surface:   #111520;
    --card:      #161c2e;
    --border:    rgba(255,185,50,0.18);
    --amber:     #f5a623;
    --amber-dim: #c47d0e;
    --sky:       #3b82f6;
    --green:     #22c55e;
    --red:       #ef4444;
    --text:      #e8eaf2;
    --muted:     #6b7280;
    --font-head: 'Syne', sans-serif;
    --font-body: 'Plus Jakarta Sans', sans-serif;
    --font-mono: 'IBM Plex Mono', monospace;
}

/* ── Reset / base ─────────────────── */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Streamlit wrappers */
.stApp { background: var(--bg) !important; }
.block-container { max-width: 740px !important; padding: 2rem 1.5rem 4rem !important; }

/* ── Animated grid background ─────────── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(245,166,35,.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(245,166,35,.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Hero header ────────────────── */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    position: relative;
}
.hero-eyebrow {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    color: var(--amber);
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.hero-title {
    font-family: var(--font-head);
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 800;
    line-height: 1.1;
    color: var(--text);
    margin: 0 0 1rem;
    letter-spacing: -0.02em;
}
.hero-title span {
    color: var(--amber);
}
.hero-sub {
    font-size: 0.95rem;
    color: var(--muted);
    font-weight: 300;
    letter-spacing: 0.01em;
}

/* ── Divider ────────────────────── */
.divider {
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--amber), transparent);
    margin: 1.5rem 0 2rem;
    opacity: 0.4;
}

/* ── Section label ───────────────── */
.section-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    color: var(--amber);
    text-transform: uppercase;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Card ─────────────────────────── */
.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--amber), transparent);
}

/* ── Corner decoration ──────────── */
.corner-tl, .corner-br {
    position: absolute;
    width: 18px; height: 18px;
    border-color: var(--amber);
    border-style: solid;
    opacity: 0.5;
}
.corner-tl { top: 10px; left: 10px; border-width: 1.5px 0 0 1.5px; }
.corner-br { bottom: 10px; right: 10px; border-width: 0 1.5px 1.5px 0; }

/* ── Form card title ────────────── */
.card-title {
    font-family: var(--font-head);
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 1.5rem;
    letter-spacing: -0.01em;
}

/* ── Streamlit form elements ──────── */
/* Labels */
label, .stSelectbox label, .stSlider label, .stForm label {
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.15em !important;
    color: var(--muted) !important;
    text-transform: uppercase !important;
    font-weight: 500 !important;
}

/* Select boxes */
.stSelectbox > div > div {
    background: #0f1523 !important;
    border: 1px solid rgba(255,185,50,0.2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s ease !important;
}
.stSelectbox > div > div:focus-within,
.stSelectbox > div > div:hover {
    border-color: rgba(245,166,35,0.6) !important;
    box-shadow: 0 0 0 3px rgba(245,166,35,0.08) !important;
}
.stSelectbox svg { color: var(--amber) !important; }

/* Dropdown options */
[data-baseweb="popover"] { background: #0f1523 !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }
[data-baseweb="option"] { background: transparent !important; color: var(--text) !important; font-family: var(--font-body) !important; }
[data-baseweb="option"]:hover { background: rgba(245,166,35,0.1) !important; }

/* Slider */
.stSlider > div > div > div {
    background: linear-gradient(90deg, var(--amber), var(--amber-dim)) !important;
}
[data-testid="stThumbValue"] {
    background: var(--amber) !important;
    color: var(--bg) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    border-radius: 4px !important;
    font-weight: 600 !important;
}
[data-baseweb="slider"] > div { background: rgba(255,255,255,0.08) !important; }

/* Thumb dot */
[role="slider"] {
    background: var(--amber) !important;
    border: 2px solid var(--bg) !important;
    box-shadow: 0 0 12px rgba(245,166,35,0.5) !important;
}

/* ── Submit button ────────────── */
.stFormSubmitButton > button,
.stButton > button {
    font-family: var(--font-head) !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.05em !important;
    background: linear-gradient(135deg, var(--amber), var(--amber-dim)) !important;
    color: #0b0e17 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.65rem 2rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(245,166,35,0.3) !important;
    width: 100% !important;
    margin-top: 0.5rem !important;
    text-transform: uppercase !important;
}
.stFormSubmitButton > button:hover,
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(245,166,35,0.45) !important;
    filter: brightness(1.05) !important;
}
.stFormSubmitButton > button:active,
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── Result cards ─────────────────── */
.result-delay, .result-ok {
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin-top: 1rem;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.result-delay {
    background: rgba(239,68,68,0.08);
    border-color: rgba(239,68,68,0.3);
}
.result-ok {
    background: rgba(34,197,94,0.08);
    border-color: rgba(34,197,94,0.3);
}
.result-icon { font-size: 3rem; margin-bottom: 0.5rem; display: block; }
.result-verdict {
    font-family: var(--font-head);
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0.25rem 0;
}
.result-delay .result-verdict { color: var(--red); }
.result-ok   .result-verdict { color: var(--green); }
.result-prob {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    color: var(--muted);
    margin-top: 0.5rem;
}
.result-advice {
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 0.75rem;
    font-weight: 300;
    line-height: 1.6;
}

/* Gauge bar */
.gauge-wrap { margin: 1rem auto 0; max-width: 300px; }
.gauge-label {
    display: flex;
    justify-content: space-between;
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    color: var(--muted);
    margin-bottom: 0.35rem;
}
.gauge-track {
    width: 100%;
    height: 6px;
    background: rgba(255,255,255,0.08);
    border-radius: 99px;
    overflow: hidden;
}
.gauge-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
}

/* ── Stat row ─────────────────── */
.stat-row {
    display: flex;
    gap: 1rem;
    margin-top: 1.25rem;
}
.stat {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    text-align: center;
}
.stat-val {
    font-family: var(--font-mono);
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--amber);
}
.stat-key {
    font-family: var(--font-mono);
    font-size: 0.58rem;
    letter-spacing: 0.15em;
    color: var(--muted);
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* ── Alert/success boxes ───────── */
.stAlert { border-radius: 10px !important; font-family: var(--font-body) !important; font-size: 0.88rem !important; }

/* ── Hide default Streamlit chrome ───────── */
#MainMenu, footer, header { display: none !important; }
.stDeployButton { display: none !important; }

/* ── Tooltip text in slider ─────────── */
[data-testid="stTickBar"] span {
    font-family: var(--font-mono) !important;
    font-size: 0.6rem !important;
    color: var(--muted) !important;
}
</style>
""", unsafe_allow_html=True)

# ┌──────────────
#  HERO HEADER
# └──────────────
st.markdown("""
<div class="hero">
    <p class="hero-eyebrow">Sistema de análisis predictivo</p>
    <h1 class="hero-title">Flight<span>Sense</span></h1>
    <p class="hero-sub">Introduce los datos de tu vuelo y el modelo evaluará<br>la probabilidad de que experimente un retraso.</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# ┌────
#  FORM
# └────
st.markdown("""
<p class="section-label">Parámetros de vuelo</p>
<div class="card">
    <div class="corner-tl"></div>
    <div class="corner-br"></div>
    <p class="card-title">Información del Vuelo</p>
</div>
""", unsafe_allow_html=True)

airline_options       = one_hot_encoder_Airline.categories_[0].tolist()
dept_time_label_opts  = one_hot_encoder_DepTime_label.categories_[0].tolist()
model_options         = one_hot_encoder_Model.categories_[0].tolist()

DAY_NAMES = {1:"Lunes", 2:"Martes", 3:"Miércoles",
             4:"Jueves", 5:"Viernes", 6:"Sábado", 7:"Domingo"}

with st.form("flight_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        airline = st.selectbox("Aerolínea", options=airline_options, index=0)
        dep_time_label = st.selectbox("Franja horaria de salida", options=dept_time_label_opts, index=0)

    with col2:
        model = st.selectbox("Modelo de aeronave", options=model_options, index=0)
        day_of_week = st.slider(
            "Día de la semana",
            min_value=1, max_value=7, value=4,
            help="1 = Lunes · 7 = Domingo"
        )
        flight_duration = st.number_input(
            "Duración del Vuelo (minutos)",
            min_value=30, max_value=1000, value=120,
            help="Duración programada del vuelo en minutos"
        )

    # Removed st.caption for selected day

    submitted = st.form_submit_button("Analizar vuelo")

# ┌──────────────
#  PREDICTION
# └──────────────
if submitted:
    input_data = pd.DataFrame({
        "Day_Of_Week":      [day_of_week], # Changed DayOfWeek to Day_Of_Week
        "Airline":        [airline],
        "DepTime_label":  [dep_time_label],
        "Model":          [model],
        "Flight_Duration":[flight_duration]
    })

    airline_encoded  = one_hot_encoder_Airline.transform(input_data[["Airline"]])
    airline_df       = pd.DataFrame(airline_encoded,
                         columns=one_hot_encoder_Airline.get_feature_names_out(["Airline"]))

    dept_encoded     = one_hot_encoder_DepTime_label.transform(input_data[["DepTime_label"]])
    dept_df          = pd.DataFrame(dept_encoded,
                         columns=one_hot_encoder_DepTime_label.get_feature_names_out(["DepTime_label"]))

    model_encoded    = one_hot_encoder_Model.transform(input_data[["Model"]])
    model_df         = pd.DataFrame(model_encoded,
                         columns=one_hot_encoder_Model.get_feature_names_out(["Model"]))

    processed_input  = pd.concat(
        [input_data[["Day_Of_Week", "Flight_Duration"]].reset_index(drop=True), airline_df, dept_df, model_df],
        axis=1
    )

    try:
        prediction       = best_gradient_boosting_model.predict(processed_input)
        prediction_proba = best_gradient_boosting_model.predict_proba(processed_input)[:, 1]
        prob             = float(prediction_proba[0])
        pct              = int(prob * 100)
        delayed          = prediction[0] == 1

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="section-label">Resultado del análisis</p>', unsafe_allow_html=True)

        # Stat row
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat">
                <div class="stat-val">{pct}%</div>
                <div class="stat-key">Prob. retraso</div>
            </div>
            <div class="stat">
                <div class="stat-val">{airline[:12]}</div>
                <div class="stat-key">Aerolínea</div>
            </div>
            <div class="stat">
                <div class="stat-val">{DAY_NAMES[day_of_week]}</div>
                <div class="stat-key">Día de vuelo</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Result card
        gauge_color = "#ef4444" if delayed else "#22c55e"
        if delayed:
            st.markdown(f"""
            <div class="result-delay">
                <span class="result-icon"></span>
                <div class="result-verdict">Retraso Probable</div>
                <div class="gauge-wrap">
                    <div class="gauge-label"><span>0%</span><span>Riesgo de retraso</span><span>100%</span></div>
                    <div class="gauge-track">
                        <div class="gauge-fill" style="width:{pct}%; background:linear-gradient(90deg,#ef4444,#dc2626);"></div>
                    </div>
                </div>
                <p class="result-advice">
                    Con una probabilidad de <strong style="color:#ef4444">{prob:.0%}</strong>, el modelo anticipa
                    un retraso en este vuelo. Recomendamos monitorear el estado con la aerolínea y llegar
                    con tiempo adicional al aeropuerto.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-ok">
                <span class="result-icon"></span>
                <div class="result-verdict">Vuelo Puntual</div>
                <div class="gauge-wrap">
                    <div class="gauge-label"><span>0%</span><span>Riesgo de retraso</span><span>100%</span></div>
                    <div class="gauge-track">
                        <div class="gauge-fill" style="width:{pct}%; background:linear-gradient(90deg,#22c55e,#16a34a);"></div>
                    </div>
                </div>
                <p class="result-advice">
                    El modelo estima una probabilidad de retraso de solo <strong style="color:#22c55e">{prob:.0%}</strong>.
                    ¡Todo apunta a una salida a tiempo! Igualmente, siempre verifica el estado del vuelo
                    antes de dirigirte al aeropuerto.
                </p>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error al realizar la predicción: {e}")
        st.warning("Verifica que los encoders y el modelo coincidan con los datos de entrada.")