import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import google.generativeai as genai

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Data Insight Tool",
    page_icon="📊",
    layout="wide"
)

# -------------------------------
# HEADER
# -------------------------------
st.title("📊 AI Data Insight Tool")
st.markdown("""
Upload a dataset to explore, visualize, and generate AI-driven insights automatically.
""")

# -------------------------------
# API KEY
# -------------------------------
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("🚨 Gemini API key not found. Set GEMINI_API_KEY as an environment variable.")
    st.stop()

genai.configure(api_key=gemini_api_key)

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # -------------------------------
    # DATA PREVIEW & BASIC STATS
    # -------------------------------
    st.subheader("📄 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("🧮 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric("Numeric Columns", len(df.select_dtypes(include=np.number).columns))

    with st.expander("Show Detailed Data Types"):
        st.write(df.dtypes)

    # Summary statistics card-style
    with st.expander("Summary Statistics"):
        st.dataframe(df.describe(include='all').round(3), use_container_width=True)

    # -------------------------------
    # VISUAL INSIGHTS
    # -------------------------------
    st.subheader("📊 Visual Insights")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Distribution and correlation charts
    if numeric_cols:
        with st.expander("Numeric Column Analysis"):
            selected_num = st.selectbox("Select a numeric column", numeric_cols)
            fig_dist = px.histogram(df, x=selected_num, nbins=30, marginal="box", 
                                    color_discrete_sequence=['#636EFA'], title=f"Distribution of {selected_num}")
            st.plotly_chart(fig_dist, use_container_width=True)

            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                                     title="Correlation Heatmap (Numeric Columns)")
                st.plotly_chart(fig_corr, use_container_width=True)

    # Categorical distribution
    if categorical_cols:
        with st.expander("Categorical Column Analysis"):
            selected_cat = st.selectbox("Select a categorical column", categorical_cols)
            cat_counts = df[selected_cat].value_counts()
            fig_cat = px.pie(values=cat_counts.values, names=cat_counts.index,
                             color_discrete_sequence=px.colors.qualitative.Pastel,
                             title=f"Category Distribution of {selected_cat}")
            st.plotly_chart(fig_cat, use_container_width=True)

    # -------------------------------
    # AI-DRIVEN INSIGHTS
    # -------------------------------
    st.subheader("🤖 AI Insights")

    # Automatically pick the best model
    available_models = [
        m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods
    ]
    preferred_models = ["models/gemini-2.5-pro", "models/gemini-2.5-flash", "models/gemini-2.0-flash"]
    model_name = next((m for m in preferred_models if m in available_models), available_models[0])
    st.info(f"Using model: {model_name}")

    # Prepare prompt for AI
    data_overview = df.describe(include='all').to_string()
    prompt = f"""
    You are an expert data analyst.
    Analyze the dataset described below and provide clear, actionable insights.
    Identify trends, correlations, anomalies, and business-relevant implications.

    Dataset summary:
    {data_overview}
    """

    try:
        model = genai.GenerativeModel(model_name)
        with st.spinner("Generating AI summary..."):
            response = model.generate_content(prompt)
        ai_summary = response.text
        with st.expander("AI Summary"):
            st.write(ai_summary)
    except Exception as e:
        st.error(f"Error generating AI summary: {e}")

else:
    st.info("Upload a dataset to begin analysis.")

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Built with ❤️ by Iko Tambaya")
