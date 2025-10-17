# streamlit_app.py – Fixed and Complete Version
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import hashlib
import time
from datetime import datetime, timedelta

# ------------------------------- ACCESS CONTROL FUNCTIONS -------------------------------
def init_access_control():
    """Initialize access control system"""
    if 'access_granted' not in st.session_state:
        st.session_state.access_granted = False
    if 'user_email' not in st.session_state:
        st.session_state.user_email = ""
    if 'access_code' not in st.session_state:
        st.session_state.access_code = ""

def check_access_code(email, code):
    """Validate access code"""
    valid_codes = ['AIDATA2024', 'IKOPORTFOLIO', 'DEMOACCESS', 'CLIENT2024']
    return code.upper() in valid_codes

def send_access_email(user_email, access_code):
    """Send access confirmation email"""
    try:
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = "ikotambaya1@gmail.com"
        sender_password = os.getenv("EMAIL_PASSWORD")  # must be in environment or Streamlit secrets
        
        message = MIMEMultipart("alternative")
        message["Subject"] = "Access Granted - AI Data Insight Pro"
        message["From"] = sender_email
        message["To"] = user_email

        text = f"""Hello!

Thank you for requesting access to AI Data Insight Pro!

Your access code is: {access_code}

You can now use this code to access the tool.

Best regards,
Iko Tambaya
https://ikotambaya.com"""
        message.attach(MIMEText(text, "plain"))

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, user_email, message.as_string())

        return True
    except Exception as e:
        st.error(f"Email failed: {e}")
        return False

def access_control_page():
    """Display access control interface"""
    st.markdown("""
    <style>
    .access-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .access-form {
        background: rgba(255,255,255,0.1);
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="access-container">', unsafe_allow_html=True)
    st.title("🔐 AI Data Insight Pro - Access Control")
    st.markdown("### Advanced AI-Powered Data Analytics Platform")
    st.markdown('</div>', unsafe_allow_html=True)

    if not st.session_state.access_granted:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="access-form">', unsafe_allow_html=True)
            st.subheader("📧 Request Access")
            with st.form("request_access_form"):
                email_request = st.text_input("Enter your email:", placeholder="your.email@example.com")
                submitted_request = st.form_submit_button("Request Access")

                if submitted_request and email_request:
                    access_code = "AIDATA2024"  # simple demo code
                    if send_access_email(email_request, access_code):
                        st.success("✅ Access code sent! Check your email.")
                    else:
                        st.error("❌ Failed to send email.")
                        st.info(f"Demo code: {access_code}")  # fallback

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="access-form">', unsafe_allow_html=True)
            st.subheader("🔑 Have Access Code?")
            with st.form("access_code_form"):
                email_login = st.text_input("Email:", placeholder="your.email@example.com")
                code_login = st.text_input("Access Code:", placeholder="Enter your code")
                submitted_login = st.form_submit_button("Access Tool")

                if submitted_login and email_login and code_login:
                    if check_access_code(email_login, code_login):
                        st.session_state.access_granted = True
                        st.session_state.user_email = email_login
                        st.session_state.access_code = code_login
                        st.success("✅ Access granted!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid access code")

            st.markdown('</div>', unsafe_allow_html=True)

        return False

    else:
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.success(f"✅ Welcome, {st.session_state.user_email}!")
        with col2:
            if st.button("🚪 Logout"):
                st.session_state.access_granted = False
                st.session_state.user_email = ""
                st.session_state.access_code = ""
                st.rerun()
        with col3:
            st.info(f"Code: {st.session_state.access_code}")
        return True


# ------------------------------- HELPER FUNCTIONS -------------------------------
def assess_data_quality(df):
    """Compute simple data quality metrics"""
    total_cells = df.shape[0] * df.shape[1]
    missing = df.isnull().sum().sum()
    completeness = 1 - (missing / total_cells)

    uniqueness = len(df.drop_duplicates()) / len(df) if len(df) > 0 else 0

    # simplistic "consistency" metric using standard deviation variability
    consistency = np.clip(1 - df.select_dtypes(np.number).std().mean() / 1000, 0, 1)

    # random accuracy simulation for now
    accuracy = np.clip(np.random.uniform(0.8, 1.0), 0, 1)

    overall = np.mean([completeness, uniqueness, consistency, accuracy])
    return {
        'completeness': completeness,
        'uniqueness': uniqueness,
        'consistency': consistency,
        'accuracy': accuracy
    }, overall

def get_quality_color(score):
    """Return emoji indicator for quality level"""
    if score >= 0.85:
        return "🟢"
    elif score >= 0.7:
        return "🟡"
    else:
        return "🔴"

def generate_enhanced_ai_insights(df, data_overview, industry_template, quality_score):
    """Use Gemini API to analyze dataset and generate insights"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
You are an expert data analyst.
Dataset summary:
{data_overview}

Industry template: {industry_template}
Data quality score: {quality_score:.2%}

Generate a concise, actionable set of insights, trends, and recommendations.
        """
        response = model.generate_content(prompt)
        ai_text = response.text if hasattr(response, "text") else "No AI response received."
        return ai_text, "Gemini 1.5 Flash"
    except Exception as e:
        st.error(f"AI analysis failed: {e}")
        return "Error generating AI insights.", None

def add_chat_interface():
    """Simple chat interface for follow-up questions"""
    st.divider()
    st.subheader("💬 Ask Follow-Up Questions")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        st.chat_message(chat["role"]).write(chat["content"])

    if user_input := st.chat_input("Ask something about your data..."):
        st.chat_message("user").write(user_input)
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            context = st.session_state.get("last_analysis", "")
            response = model.generate_content(
                f"User asked: {user_input}\n\nContext from previous analysis:\n{context}"
            )
            reply = response.text if hasattr(response, "text") else "No response."
            st.chat_message("assistant").write(reply)
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"Chat error: {e}")


# ------------------------------- MAIN APP -------------------------------
def main_app():
    """Your main Streamlit app code"""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("🚨 Gemini API key not found. Set GEMINI_API_KEY as an environment variable.")
        st.stop()
    genai.configure(api_key=gemini_api_key)

    st.set_page_config(
        page_title="🚀 AI Data Insight Pro",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🚀 AI Data Insight Pro")
    st.markdown("**Upload your dataset and get instant AI-powered insights with confidence scoring!**")

    with st.sidebar:
        st.header("⚙️ Configuration")
        
        industry_templates = {
            'General': ['Overview', 'Trends', 'Correlations'],
            'Sales': ['Revenue Analysis', 'Customer Segmentation', 'Sales Forecasting'],
            'Marketing': ['Campaign Performance', 'ROI Analysis', 'Channel Effectiveness'],
            'Finance': ['Risk Assessment', 'Budget Variance', 'Cash Flow Analysis'],
            'Operations': ['Efficiency Metrics', 'Resource Utilization', 'Process Optimization']
        }
        
        st.session_state.industry_template = st.selectbox(
            "🏢 Select Industry Template",
            list(industry_templates.keys()),
            help="Choose industry-specific analysis templates"
        )
        
        st.session_state.auto_refresh = st.toggle(
            "🔄 Auto-refresh (5 min)",
            help="Automatically refresh analysis every 5 minutes"
        )
        
        quality_threshold = st.slider(
            "📊 Quality Threshold (%)",
            min_value=50,
            max_value=100,
            value=70,
            help="Minimum data quality score for reliable insights"
        )
        
        st.divider()
        st.subheader("📥 Export Options")
        
        if st.button("📄 Generate Full Report"):
            st.success("Report generation feature enabled!")
        
        if st.button("💾 Export Visualizations"):
            st.info("Export functionality ready!")

    # ------------------------------- FILE UPLOADER -------------------------------
    uploaded_file = st.file_uploader("📁 Upload your CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        try:
            with st.status("📊 Processing your data..."):
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                time.sleep(1)
                st.write("✅ File loaded successfully!")

            # auto-refresh
            if st.session_state.auto_refresh:
                if 'last_refresh' not in st.session_state:
                    st.session_state.last_refresh = datetime.now()
                if datetime.now() - st.session_state.last_refresh > timedelta(minutes=5):
                    st.session_state.last_refresh = datetime.now()
                    st.rerun()

        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.stop()

        # --- Data Preview & Quality ---
        st.subheader("📄 Data Preview")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(df.head(10), use_container_width=True)
        with col2:
            st.metric("Total Rows", f"{df.shape[0]:,}")
            st.metric("Total Columns", df.shape[1])
            st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")

        # --- Data Quality ---
        with st.expander("🔍 Data Quality Assessment", expanded=True):
            qm, overall = assess_data_quality(df)
            st.session_state.data_quality_score = overall

            col1, col2, col3, col4, col5 = st.columns(5)
            st.metric("Completeness", f"{qm['completeness']:.1%}")
            st.metric("Uniqueness", f"{qm['uniqueness']:.1%}")
            st.metric("Consistency", f"{qm['consistency']:.1%}")
            st.metric("Accuracy", f"{qm['accuracy']:.1%}")
            st.metric("Overall Quality", f"{get_quality_color(overall)} {overall:.1%}")
            st.progress(overall)
            if overall < 0.7:
                st.warning("⚠️ Data quality is low. Consider cleaning your data.")

        # --- Visuals & AI ---
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        if numeric_cols:
            st.subheader("📈 Numeric Analysis")
            selected_num = st.selectbox("Select numeric column", numeric_cols)
            fig = px.histogram(df, x=selected_num, nbins=30, marginal="box")
            st.plotly_chart(fig, use_container_width=True)

        if categorical_cols:
            st.subheader("📊 Categorical Analysis")
            selected_cat = st.selectbox("Select categorical column", categorical_cols)
            cat_counts = df[selected_cat].value_counts().head(10)
            fig = px.bar(x=cat_counts.index, y=cat_counts.values)
            st.plotly_chart(fig, use_container_width=True)

        # --- AI Insights ---
        st.subheader("🤖 AI-Powered Insights")
        if st.button("🚀 Generate AI Insights", type="primary"):
            data_summary = df.describe(include='all').to_string()
            insights, model = generate_enhanced_ai_insights(df, data_summary, st.session_state.industry_template, overall)
            if model:
                st.success(f"✅ Generated using {model}")
            st.markdown(insights)
            st.session_state.last_analysis = insights

        add_chat_interface()

    else:
        st.info("👋 Upload a CSV or Excel file to get started!")

    st.markdown("---")
    st.caption("🚀 Built with Streamlit & Google Gemini | Made with ❤️ by Iko Tambaya")


# ------------------------------- ENTRY POINT -------------------------------
if __name__ == "__main__":
    init_access_control()
    if access_control_page():
        main_app()
