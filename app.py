# streamlit_app.py – SIMPLIFIED, NO MACHINE-LEARNING, NO access_control/main_functions
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import os
import time
from datetime import datetime

# ---------- page ----------
st.set_page_config(
    page_title="🚀 AI Data Insight Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- css ----------
st.markdown("""
<style>
    .stApp{background:#0e1117;color:#fafafa}
    .metric-card{background:linear-gradient(135deg,#6366f120,#8b5cf620);border-radius:16px;padding:20px;border:1px solid #6366f130}
    div[data-testid="stMetric"]{text-align:center}
</style>
""", unsafe_allow_html=True)

# ---------- session ----------
for k, v in {
    "chat_history": [],
    "data_quality_score": 0.0,
    "last_analysis": None,
    "industry_template": "General",
    "df": None,
    "file_name": "",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------- helpers ----------
@st.cache_data(show_spinner=False)
def read_file(uploaded_file):
    """Return DataFrame from uploaded CSV/Excel."""
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)

def ai_insights(df: pd.DataFrame, template: str) -> str:
    """Lightweight Gemini call – no ML models."""
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"You are a {template} analyst. Summarise this dataset in 5 bullet points:\n{df.describe(include='all').round(2)}"
    return model.generate_content(prompt).text

# ---------- header ----------
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    st.title("🚀 AI Data Insight Pro")
    st.caption("Next-generation AI-powered data intelligence platform")
with c3:
    st.metric("⏱ Live", datetime.now().strftime("%H:%M:%S"))

# ---------- sidebar ----------
with st.sidebar:
    st.session_state.industry_template = st.selectbox(
        "Industry", ["General", "Sales", "Marketing", "Finance", "Operations"]
    )
    st.slider("Quality threshold (%)", 50, 100, 75, key="qt")

# ---------- uploader ----------
uploaded = st.file_uploader("📁 Upload CSV or Excel", type=["csv", "xlsx"])
if uploaded:
    try:
        with st.status("Reading file…"):
            df = read_file(uploaded)
            st.session_state.df = df
            st.session_state.file_name = uploaded.name
    except Exception as e:
        st.error(f"❌ {e}")
        st.stop()

# ---------- dashboard ----------
if st.session_state.df is not None:
    df = st.session_state.df
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    cat = df.select_dtypes(exclude=np.number).columns.tolist()

    # KPI row
    col = st.columns(6)
    stats = {
        "Records": len(df),
        "Columns": df.shape[1],
        "Numeric": len(numeric),
        "Categories": len(cat),
        "Missing %": round(df.isnull.sum().sum() / df.size * 100, 1),
        "Duplicates": df.duplicated().sum(),
    }
    for c, (k, v) in zip(col, stats.items()):
        c.metric(k, v)

    # Quality gauge
    completeness = 1 - df.isnull().sum().sum() / df.size
    uniqueness = len(df.drop_duplicates()) / len(df)
    score = (completeness + uniqueness) / 2 * 100
    st.session_state.data_quality_score = score
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Data Quality"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#10b981" if score > 75 else "#f59e0b"}},
        )
    )
    fig.update_layout(height=200, margin=dict(l=30, r=30, t=30, b=30))
    st.plotly_chart(fig, use_container_width=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Visuals", "🤖 AI Insights", "💬 Chat"])
    with tab1:
        if numeric:
            col1, col2 = st.columns(2)
            col = col1.selectbox("Numeric column", numeric)
            col1.plotly_chart(px.histogram(df, x=col, marginal="box"), use_container_width=True)
            if len(numeric) > 1:
                col2.plotly_chart(px.imshow(df[numeric].corr(), text_auto=True, aspect="auto", color_continuous_scale="viridis"), use_container_width=True)
        if cat:
            pick = st.selectbox("Categorical column", cat)
            st.plotly_chart(px.pie(df, names=pick), use_container_width=True)

    with tab2:
        if st.button("Generate AI Insights", type="primary"):
            with st.spinner("Thinking…"):
                insight = ai_insights(df, st.session_state.industry_template)
            st.markdown(insight)

    with tab3:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        if prompt := st.chat_input("Ask something"):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("assistant"):
                reply = f"Echo: {prompt}"
                st.write(reply)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})

# ---------- footer ----------
st.divider()
st.caption("Built with Streamlit & Google Gemini | Made with ❤️ by Iko Tambaya")
