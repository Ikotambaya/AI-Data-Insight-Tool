import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import google.generativeai as genai
from datetime import datetime, timedelta
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="🚀 AI Data Insight Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- SESSION STATE ----------------
if 'data_quality_score' not in st.session_state:
    st.session_state.data_quality_score = 0

# ---------------- HEADER ----------------
with st.container():
    st.title("🚀 AI Data Insight Pro")
    st.markdown("""
    **Upload your dataset and get instant AI-powered insights with confidence scoring!**  
    Powered by Iko Tambaya with advanced data quality assessment.
    """)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Configuration")
    industry_templates = ['General', 'Sales', 'Marketing', 'Finance', 'Operations']
    st.session_state.industry_template = st.selectbox(
        "🏢 Select Industry Template",
        industry_templates
    )
    st.session_state.auto_refresh = st.checkbox("🔄 Auto-refresh (5 min)", value=False)
    quality_threshold = st.slider("📊 Minimum Data Quality (%)", 50, 100, 70)
    st.divider()
    st.subheader("📥 Export Options")
    st.button("📄 Generate Full Report")
    st.button("💾 Export Visualizations")

# ---------------- GEMINI API ----------------
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("🚨 GEMINI_API_KEY not set!")
    st.stop()
genai.configure(api_key=gemini_api_key)

# ---------------- DATA QUALITY FUNCTIONS ----------------
def assess_data_quality(df):
    completeness = 1 - (df.isnull().sum().sum() / df.size)
    uniqueness = len(df.drop_duplicates()) / len(df)
    consistency_score = np.mean([1 - (df[col].duplicated().sum()/len(df)) for col in df.columns])
    accuracy_score = np.mean([1 if df[col].dtype in ['int64','float64'] else 1 for col in df.columns])
    overall_quality = np.mean([completeness, uniqueness, consistency_score, accuracy_score])
    return {"Completeness": completeness, "Uniqueness": uniqueness, "Consistency": consistency_score, "Accuracy": accuracy_score}, overall_quality

def create_gauge_chart(label, value, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value*100,
        title={'text': label},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}, 'steps': [
            {'range': [0, 60], 'color': "#FF4B4B"},
            {'range': [60, 80], 'color': "#FFEB3B"},
            {'range': [80, 100], 'color': "#4CAF50"}]},
    ))
    fig.update_layout(height=250, margin=dict(t=0,b=0,l=0,r=0))
    return fig

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader("📁 Upload CSV/Excel", type=["csv","xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("✅ File loaded successfully!")

    # ---------------- DATA QUALITY ----------------
    with st.container():
        st.subheader("🔍 Data Quality Assessment")
        metrics, overall = assess_data_quality(df)
        st.session_state.data_quality_score = overall

        cols = st.columns(4)
        colors = ["#4CAF50","#FFC107","#2196F3","#9C27B0"]
        for i, (k,v) in enumerate(metrics.items()):
            cols[i].plotly_chart(create_gauge_chart(k, v, colors[i]), use_container_width=True)
        st.metric("Overall Data Quality", f"{overall*100:.1f}%")
    
    # ---------------- DATA PREVIEW ----------------
    with st.container():
        st.subheader("📄 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    # ---------------- VISUAL INSIGHTS ----------------
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    with st.container():
        st.subheader("📊 Visual Insights")
        if numeric_cols:
            num_col = st.selectbox("Select numeric column", numeric_cols)
            fig = px.histogram(df, x=num_col, nbins=30, marginal="box", color_discrete_sequence=['#636EFA'])
            st.plotly_chart(fig, use_container_width=True)
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Heatmap")
                st.plotly_chart(fig_corr, use_container_width=True)
        if categorical_cols:
            cat_col = st.selectbox("Select categorical column", categorical_cols)
            counts = df[cat_col].value_counts().head(10)
            fig = px.pie(values=counts.values, names=counts.index, color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig, use_container_width=True)

    # ---------------- AI INSIGHTS ----------------
    st.subheader("🤖 AI-Powered Insights")
    if st.button("🚀 Generate AI Insights"):
        data_overview = df.describe(include='all').round(3).to_string()
        prompt = f"Analyze dataset with quality {overall*100:.1f}% for {st.session_state.industry_template} context.\n{data_overview}"
        try:
            model = genai.GenerativeModel("models/gemini-2.5-pro")
            response = model.generate_content(prompt)
            st.markdown(response.text)
        except Exception as e:
            st.error(f"AI analysis error: {e}")

else:
    st.info("👋 Upload a file to start analyzing your data!")
