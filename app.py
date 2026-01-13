import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sys
import subprocess
import time
import requests

# --- 🛠 AUTO-INSTALLER ---
try:
    import sklearn
    import plotly.express as px
    import plotly.graph_objects as go
    from streamlit_lottie import st_lottie
except ImportError:
    st.warning("⚠️ Installing visualization packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn", "pandas", "numpy", "joblib", "plotly", "streamlit-lottie", "requests"])
    st.success("✅ Ready! Restarting...")
    time.sleep(2)
    st.rerun()

# --- MODEL LOADING ---
try:
    # Loading from 'models' folder
    model = joblib.load('models/personality_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
except FileNotFoundError:
    st.error("ERROR: Model files not found! Please check 'models/' directory.")
    st.stop()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Personality Analysis System", page_icon="🧬", layout="wide")

# --- 🌑 PROFESSIONAL DARK MODE CSS ---
st.markdown("""
<style>
    /* Main Background */
    .stApp { background-color: #0E1117; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    
    /* Text Colors */
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown { color: #E6EDF3 !important; }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #21262D;
        border: 1px solid #30363D;
        padding: 20px;
        border-radius: 8px;
        color: white;
        text-align: center;
    }
    
    /* Inputs */
    .stNumberInput > div > div > input { color: white; background-color: #0D1117; }
    
    /* Button Design - Professional Green */
    div.stButton > button {
        background: #238636;
        color: white;
        border: none;
        padding: 12px 24px;
        font-weight: bold;
        border-radius: 6px;
        transition: 0.3s;
        width: 100%;
    }
    div.stButton > button:hover {
        background: #2EA043;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR (LEFT) ---
with st.sidebar:
    st.header("🎛️ Parameters")
    st.markdown("Enter data for analysis.")
    st.markdown("---")
    
    with st.expander("👤 Demographics", expanded=True):
        age = st.slider("Age", 18, 70, 25)
        # Model logic: Female=0, Male=1
        gender = 0 if st.selectbox("Gender", ["Female", "Male"]) == "Female" else 1
        
        # Education Mapping
        edu_map = {"Basic Level": 0, "High Level": 1}
        education = edu_map[st.selectbox("Education", list(edu_map.keys()))]
        
        # Interest Mapping
        interest_map = {"Arts": 0, "Others": 1, "Sports": 2, "Technology": 3, "Unknown": 4}
        interest = interest_map[st.selectbox("Interest", list(interest_map.keys()))]

    st.warning("👇 Enter scores between 0-10.")
    
    with st.expander("🧠 MBTI Scores", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            introversion = st.number_input("Introversion (I)", 0.0, 10.0, 3.00, 0.1)
            thinking = st.number_input("Thinking (T)", 0.0, 10.0, 7.00, 0.1)
        with col2:
            sensing = st.number_input("Sensing (S)", 0.0, 10.0, 3.00, 0.1)
            judging = st.number_input("Judging (J)", 0.0, 10.0, 3.00, 0.1)
            
    st.markdown("---")
    analyze_btn = st.button("RUN ANALYSIS")

# --- MAIN SCREEN (RIGHT) ---
st.title("🧬 AI Personality Analysis System")
st.markdown("#### CENG313 Term Project | Model Accuracy: **%90.51**")

if analyze_btn:
    # Data Processing
    input_data = np.array([[age, gender, education, introversion, sensing, thinking, judging, interest]])
    input_scaled = scaler.transform(input_data)
    
    # Prediction
    prediction_index = int(model.predict(input_scaled)[0])
    personality_types = [
        'ENFJ', 'ENFP', 'ENTJ', 'ENTP', 'ESFJ', 'ESFP', 'ESTJ', 'ESTP',
        'INFJ', 'INFP', 'INTJ', 'INTP', 'ISFJ', 'ISFP', 'ISTJ', 'ISTP'
    ]
    predicted_type = personality_types[prediction_index]
    
    # --- RESULT DASHBOARD ---
    st.markdown("---")
    
    m1, m2, m3 = st.columns(3)
    
    # Box 1: Result
    m1.metric("Predicted Type", predicted_type, "Model Output")
    
    # Box 2: Static Success
    m2.metric("System Accuracy", "%90.51", "Test Score") 
    
    # Box 3: Dominant Trait
    energy_dir = "Introvert" if 'I' in predicted_type else "Extravert"
    m3.metric("Dominant Trait", energy_dir, delta=None)

    st.markdown("---")

    # --- RADAR CHART ---
    col_spacer_left, col_chart, col_spacer_right = st.columns([1, 4, 1])

    with col_chart:
        # Chart Data (Reversing I for visual accuracy)
        r_intro = 10 - introversion
        r_extra = introversion
        
        categories = ['Introversion', 'Sensing', 'Thinking', 'Judging', 'Extraversion', 'Intuition', 'Feeling', 'Perceiving']
        values = [r_intro, sensing, thinking, judging, r_extra, 10-sensing, 10-thinking, 10-judging]
        
        # Plotly Dark Template
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories, fill='toself', name=predicted_type,
            line=dict(color='#00FF9D', width=3),
            fillcolor='rgba(0, 255, 157, 0.2)'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10], color='gray', gridcolor='#30363D'),
                angularaxis=dict(color='white'),
                bgcolor='#161B22'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=14),
            margin=dict(t=40, b=40, l=40, r=40),
            showlegend=False,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Enter parameters in the sidebar to start the analysis.")