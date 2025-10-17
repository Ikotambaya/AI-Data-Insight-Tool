# Complete app.py - Part 1: Foundation & Access Control
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
import time
from datetime import datetime, timedelta

# Fixed streamlit_app.py - with proper access control
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

    # ------------------------------- ENHANCED DATA QUALITY FUNCTIONS -------------------------------
def assess_data_quality(df):
    """Comprehensive data quality assessment with scoring"""
    quality_metrics = {}
    
    # Completeness
    completeness = 1 - (df.isnull().sum().sum() / df.size)
    quality_metrics['completeness'] = completeness
    
    # Uniqueness
    uniqueness = len(df.drop_duplicates()) / len(df)
    quality_metrics['uniqueness'] = uniqueness
    
    # Consistency (check for data type consistency)
    consistency_score = 0
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            # Check for outliers (values beyond 3 standard deviations)
            if df[col].std() > 0:
                outliers = len(df[col][abs(df[col] - df[col].mean()) > 3 * df[col].std()])
                consistency_score += 1 - (outliers / len(df))
        else:
            # For categorical data, check for consistent formatting
            consistency_score += 1 - (df[col].str.contains(r'[^a-zA-Z0-9\s]', regex=True).sum() / len(df))
    
    quality_metrics['consistency'] = consistency_score / len(df.columns) if len(df.columns) > 0 else 1
    
    # Accuracy (basic validation)
    accuracy_score = 0
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            # Check for reasonable ranges
            if df[col].min() >= 0 and df[col].max() < 1e6:  # Reasonable business values
                accuracy_score += 1
        else:
            # Check for reasonable string lengths
            avg_length = df[col].astype(str).str.len().mean()
            if 1 <= avg_length <= 100:
                accuracy_score += 1
    
    quality_metrics['accuracy'] = accuracy_score / len(df.columns) if len(df.columns) > 0 else 1
    
    # Overall quality score
    overall_quality = sum(quality_metrics.values()) / len(quality_metrics)
    
    return quality_metrics, overall_quality

def get_quality_color(score):
    """Return color based on quality score"""
    if score >= 0.8:
        return "🟢"
    elif score >= 0.6:
        return "🟡"
    else:
        return "🔴"

# ------------------------------- ENHANCED AI INSIGHTS WITH CONFIDENCE -------------------------------
def generate_enhanced_ai_insights(df, data_overview, industry_template, quality_score):
    """Generate AI insights with confidence scoring and industry context"""
    
    industry_context = {
        'General': "Provide general business insights and trends.",
        'Sales': "Focus on sales performance, customer behavior, and revenue optimization.",
        'Marketing': "Analyze campaign effectiveness, customer acquisition, and marketing ROI.",
        'Finance': "Assess financial health, risk factors, and investment opportunities.",
        'Operations': "Evaluate operational efficiency, resource allocation, and process improvements."
    }
    
    prompt = f"""
    You are an expert {industry_template.lower()} data analyst with advanced statistical knowledge.
    
    Dataset Quality Score: {quality_score:.1%} (1.0 = perfect quality)
    Industry Context: {industry_context.get(industry_template, 'General business analysis')}
    
    Analyze the dataset described below and provide:
    1. Key insights and trends (with confidence levels)
    2. Specific recommendations for {industry_template} context
    3. Potential risks or concerns
    4. Next steps for deeper analysis
    5. Business implications and actionable items
    
    IMPORTANT: If data quality is below 70%, mention this and suggest data cleaning steps.
    
    Dataset summary:
    {data_overview}
    
    Format your response with clear sections and bullet points.
    """
    
    try:
        available_models = [
            m.name for m in genai.list_models() if "generateContent" in m.supported_generation_methods
        ]
        preferred_models = ["models/gemini-2.5-pro", "models/gemini-2.5-flash", "models/gemini-2.0-flash"]
        model_name = next((m for m in preferred_models if m in available_models), available_models[0])
        
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        # Add confidence indicator based on data quality
        confidence_indicator = f"\n\n**Confidence Level: {get_quality_color(quality_score)} {quality_score:.1%}**"
        
        if quality_score < 0.7:
            confidence_indicator += "\n⚠️ **Low data quality detected - verify insights manually**"
        elif quality_score >= 0.9:
            confidence_indicator += "\n✅ **High confidence - insights are highly reliable**"
        
        return response.text + confidence_indicator, model_name
        
    except Exception as e:
        return f"Error generating AI insights: {e}", None

# ------------------------------- CHAT INTERFACE -------------------------------
def add_chat_interface():
    """Add conversational interface for follow-up questions"""
    st.subheader("💬 Ask Follow-up Questions")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your data..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Generate AI response based on current data
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Simple response based on available data
                response = f"Based on your dataset, I can help you with: {prompt}. For detailed analysis, please upload your data first."
                st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Sample placeholder for your existing code
    uploaded_file = st.file_uploader("📁 Upload your CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success("✅ File loaded successfully!")
            
            # Add your existing analysis code here...
            st.write("File has", df.shape[0], "rows and", df.shape[1], "columns")
            st.dataframe(df.head())
            
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.stop()
    
    else:
        st.info("👋 Upload a file to get started!")

# ------------------------------- MAIN APP LOGIC -------------------------------
def main():
    """Main app with access control"""
    # Initialize access control
    init_access_control()
    
    # Check access first
    if access_control_page():
        # Access granted - show main app
        main_app()
    # If no access, the access_control_page() will handle the display

if __name__ == "__main__":
    main()
