import streamlit as st, pandas as pd, numpy as np, plotly.express as px, google.generativeai as genai, os, time

def init_session_state():
    defaults = {
        'chat_history': [],
        'data_quality_score': 0,
        'last_analysis': None,
        'industry_template': 'General',
        'auto_refresh': False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def setup_gemini_api():
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("🚨 GEMINI_API_KEY not found.")
        st.stop()
    genai.configure(api_key=gemini_api_key)

def assess_data_quality(df):
    completeness = 1 - (df.isnull().sum().sum() / df.size)
    uniqueness  = len(df.drop_duplicates()) / len(df)
    consistency = 1
    accuracy    = 1
    overall     = (completeness + uniqueness + consistency + accuracy) / 4
    return {
        'completeness': completeness,
        'uniqueness': uniqueness,
        'consistency': consistency,
        'accuracy': accuracy
    }, overall

def get_quality_color(score):
    return "🟢" if score >= .8 else "🟡" if score >= .6 else "🔴"

def generate_enhanced_ai_insights(df, overview, template, score):
    prompt = f"Act as a {template} analyst. Summarize:\n{overview}"
    model  = genai.GenerativeModel("gemini-pro")
    resp   = model.generate_content(prompt)
    return resp.text + f"\n\nConfidence: {get_quality_color(score)} {score:.1%}", "gemini-pro"

def add_chat_interface():
    st.subheader("💬 Ask Follow-up")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    if prompt := st.chat_input("Ask about your data..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            reply = f"Echo: {prompt}"
            st.write(reply)
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

def create_sidebar():
    # placeholder – whatever your sidebar code needs
    pass
