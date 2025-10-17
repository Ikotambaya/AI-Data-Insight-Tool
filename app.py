# Your existing streamlit_app.py - with access control added
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

# Replace your access_control_page() function with this:
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
            
            with st.form("request_access_form"):  # CHANGED: Unique form key
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
            
            with st.form("access_code_form"):  # CHANGED: Unique form key
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
# ------------------------------- YOUR EXISTING APP CODE -------------------------------
# Put all your existing code here, but wrap it in a function
def main_app():
    """Your existing Streamlit app code"""
    # Your existing imports and setup
    st.set_page_config(
        page_title="🚀 AI Data Insight Pro",
        page_icon="📊",
        layout="wide"
    )
    
    # Your existing Gemini API setup
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        st.error("🚨 Gemini API key not found")
        st.stop()
    
    genai.configure(api_key=gemini_api_key)
    
    # Your existing app code continues here...
    st.title("🚀 AI Data Insight Pro")
    st.markdown("**Advanced AI-Powered Data Analysis Tool**")
    
    uploaded_file = st.file_uploader("📁 Upload your CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file:
        # Your existing file processing code...
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Your existing analysis code...
            st.success("✅ File loaded successfully!")
            
            # Continue with your existing visualization code...
            
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.stop()

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
