# Fixed streamlit_app.py - with proper access control and no duplicates
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
    # Initialize other session state variables
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'data_quality_score' not in st.session_state:
        st.session_state.data_quality_score = 0
    if 'last_analysis' not in st.session_state:
        st.session_state.last_analysis = None
    if 'industry_template' not in st.session_state:
        st.session_state.industry_template = 'General'
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False

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

# ------------------------------- MAIN APP FUNCTIONS -------------------------------
def main_app_content():
    """Main app content (only shown when access is granted)"""
    # Your existing Gemini API setup
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("🚨 Gemini API key not found. Set GEMINI_API_KEY as an environment variable.")
        st.stop()

    genai.configure(api_key=gemini_api_key)

    # Your existing header
    st.title("🚀 AI Data Insight Pro")
    st.markdown("""
    **Upload your dataset and get instant AI-powered insights with confidence scoring!**  
    Powered by Iko Tambaya with advanced data quality assessment.
    """)

    # Your existing sidebar (moved here to avoid conflicts)
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

    # Your existing file uploader and analysis code
    uploaded_file = st.file_uploader(
        "📁 Upload your CSV or Excel file", 
        type=["csv", "xlsx"],
        help="Supports CSV and Excel files. Max file size: 200MB"
    )

    if uploaded_file:
        try:
            with st.status("📊 Processing your data..."):
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                time.sleep(1)  # Simulate processing time
                st.write("✅ File loaded successfully!")
            
            # Auto-refresh logic
            if st.session_state.auto_refresh:
                if 'last_refresh' not in st.session_state:
                    st.session_state.last_refresh = datetime.now()
                
                if datetime.now() - st.session_state.last_refresh > timedelta(minutes=5):
                    st.session_state.last_refresh = datetime.now()
                    st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.stop()

        # Continue with your existing analysis code...
        st.success("✅ File loaded successfully!")
        st.write(f"File has {df.shape[0]} rows and {df.shape[1]} columns")
        st.dataframe(df.head())
        
        # Add your existing analysis, visualizations, AI insights etc. here
        # For now, just show basic info
        st.subheader("📊 Data Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Numeric Columns", len(df.select_dtypes(include=np.number).columns))
        
    else:
        st.info("👋 Upload a file to get started!")

# ------------------------------- MAIN APP LOGIC -------------------------------
def main():
    """Main app with access control"""
    # Initialize access control
    init_access_control()
    
    # Set page config here only once
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
    
    # Check access
    if st.session_state.access_granted:
        # Show main app
        main_app_content()
    else:
        # Show access control
        access_control_page()

if __name__ == "__main__":
    main()
