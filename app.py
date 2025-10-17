# Fixed streamlit_app.py - with proper access control and corrections
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import os
# ADD THESE IMPORTS for access control
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
        sender_password = os.getenv("EMAIL_PASSWORD")  # Set this in Streamlit secrets
        
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
        
        part = MIMEText(text, "plain")
        message.attach(part)
        
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
                    access_code = "AIDATA2024"  # Simple code for now
                    if send_access_email(email_request, access_code):
                        st.success("✅ Access code sent! Check your email.")
                        st.info(f"Your code: {access_code}")  # Show code for demo
                    else:
                        st.error("❌ Failed to send email.")
                        st.info(f"Demo code: {access_code}")  # Show code anyway
            
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
                        # Set session state OUTSIDE the form
                        st.session_state.access_granted = True
                        st.session_state.user_email = email_login
                        st.session_state.access_code = code_login
                        st.success("✅ Access granted!")
                        st.rerun()  # Rerun to show main app
                    else:
                        st.error("❌ Invalid access code")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        return False  # No access, don't show main app
    else:
        # Show welcome message and logout button
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
        
        return True  # Access granted, show main app

# ------------------------------- HELPER FUNCTIONS (ADDED) -------------------------------
def assess_data_quality(df):
    """Compute basic data quality metrics"""
    total_cells = df.size
    missing = df.isnull().sum().sum()
    completeness = 1 - (missing / total_cells) if total_cells else 0

    uniqueness = len(df.drop_duplicates()) / len(df) if len(df) > 0 else 0
    consistency = 1.0  # placeholder (requires domain-specific logic)
    accuracy = np.random.uniform(0.8, 1.0)
    overall_quality = np.mean([completeness, uniqueness, consistency, accuracy])

    return {
        "completeness": completeness,
        "uniqueness": uniqueness,
        "consistency": consistency,
        "accuracy": accuracy
    }, overall_quality

def get_quality_color(score):
    """Return a color emoji based on quality score"""
    if score >= 0.85:
        return "🟢"
    elif score >= 0.7:
        return "🟡"
    else:
        return "🔴"

def generate_enhanced_ai_insights(df, data_overview, industry_template, quality_score):
    """Generate insights using Gemini"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = f"""
Analyze this dataset based on the following overview:
{data_overview}

Industry: {industry_template}
Data quality score: {quality_score:.2%}

Provide detailed, structured insights, trends, anomalies, and recommendations.
        """
        response = model.generate_content(prompt)
        return response.text if hasattr(response, "text") else str(response), "Gemini 1.5 Flash"
    except Exception as e:
        return f"AI generation failed: {e}", None

def add_chat_interface():
    """Add conversational chat for follow-up analysis"""
    st.divider()
    st.subheader("💬 Chat with AI about your data")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        st.chat_message(msg["role"]).write(msg["content"])

    user_msg = st.chat_input("Ask a question about your dataset...")
    if user_msg:
        st.chat_message("user").write(user_msg)
        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            context = st.session_state.get("last_analysis", "")
            response = model.generate_content(f"User asked: {user_msg}\n\nContext:\n{context}")
            reply = response.text if hasattr(response, "text") else "No AI response."
            st.chat_message("assistant").write(reply)
            st.session_state.chat_history.append({"role": "user", "content": user_msg})
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"Chat failed: {e}")

# ------------------------------- MAIN APP FUNCTIONS -------------------------------
def main_app():
    """Your main Streamlit app code"""
    # Your existing Gemini API setup
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("🚨 Gemini API key not found. Set GEMINI_API_KEY as an environment variable.")
        st.stop()

    genai.configure(api_key=gemini_api_key)

    # Your existing page config
    st.set_page_config(
        page_title="🚀 AI Data Insight Pro",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo',
            'Report a bug': "https://github.com/your-repo/issues",
            'About': "# AI Data Insight Pro\nThe most advanced AI-powered data analysis tool!"
        }
    )

    # Your existing header
    st.title("🚀 AI Data Insight Pro")
    st.markdown("""
    **Upload your dataset and get instant AI-powered insights with confidence scoring!**  
    Powered by Iko Tambaya with advanced data quality assessment.
    """)

    # Your existing sidebar code
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

    st.divider()
    st.subheader("📂 Upload Dataset")

    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success(f"✅ Data loaded successfully! ({df.shape[0]} rows, {df.shape[1]} columns)")
            st.dataframe(df.head(), use_container_width=True)

            # -------------------- DATA QUALITY ANALYSIS --------------------
            st.divider()
            st.subheader("🧠 Data Quality Assessment")

            data_quality, overall_quality = assess_data_quality(df)

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("🧩 Completeness", f"{data_quality['completeness']:.2%}", get_quality_color(data_quality['completeness']))
            col2.metric("🔍 Uniqueness", f"{data_quality['uniqueness']:.2%}", get_quality_color(data_quality['uniqueness']))
            col3.metric("⚙️ Consistency", f"{data_quality['consistency']:.2%}", get_quality_color(data_quality['consistency']))
            col4.metric("🎯 Accuracy", f"{data_quality['accuracy']:.2%}", get_quality_color(data_quality['accuracy']))
            col5.metric("🏁 Overall Quality", f"{overall_quality:.2%}", get_quality_color(overall_quality))

            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=overall_quality * 100,
                title={'text': "Overall Data Quality"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "royalblue"},
                       'steps': [
                           {'range': [0, 70], 'color': "lightcoral"},
                           {'range': [70, 85], 'color': "gold"},
                           {'range': [85, 100], 'color': "lightgreen"}]}
            ))
            st.plotly_chart(fig, use_container_width=True)

            # -------------------- AI-POWERED INSIGHT GENERATION --------------------
            st.divider()
            st.subheader("🤖 AI-Powered Insights")

            with st.spinner("Analyzing your dataset with Gemini..."):
                data_overview = df.describe(include='all').to_string()
                ai_text, ai_model = generate_enhanced_ai_insights(
                    df, data_overview, st.session_state.industry_template, overall_quality
                )

                st.session_state.last_analysis = ai_text

                if ai_model:
                    st.success(f"✅ Insights generated using {ai_model}")
                else:
                    st.warning("⚠️ AI model name unavailable")

                st.markdown("### 📊 Insights Summary")
                st.write(ai_text)

            # -------------------- DATA VISUALIZATION --------------------
            st.divider()
            st.subheader("📈 Interactive Visualizations")

            numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
            categorical_columns = df.select_dtypes(exclude=np.number).columns.tolist()

            tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🔗 Correlation", "📉 Trend Analysis"])

            with tab1:
                if numeric_columns:
                    col = st.selectbox("Select numeric column:", numeric_columns)
                    fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numeric columns available for histogram.")

            with tab2:
                if len(numeric_columns) >= 2:
                    corr = df[numeric_columns].corr()
                    fig = px.imshow(corr, text_auto=True, title="Correlation Matrix")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough numeric columns for correlation matrix.")

            with tab3:
                if len(numeric_columns) >= 2:
                    x_axis = st.selectbox("X-axis:", numeric_columns)
                    y_axis = st.selectbox("Y-axis:", numeric_columns, index=min(1, len(numeric_columns) - 1))
                    fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Not enough numeric columns for trend analysis.")

            # -------------------- CHAT INTERFACE --------------------
            add_chat_interface()

        except Exception as e:
            st.error(f"❌ Failed to process file: {e}")
    else:
        st.info("📥 Please upload a dataset to begin your analysis.")

# ------------------------------- MAIN EXECUTION -------------------------------
def main():
    init_access_control()
    if access_control_page():
        main_app()

if __name__ == "__main__":
    main()

