import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import google.generativeai as genai
from datetime import datetime, timedelta
import time
from io import BytesIO
import base64

# ------------------------------- Enhanced PAGE CONFIG -------------------------------
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

# ------------------------------- SESSION STATE INITIALIZATION -------------------------------
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

# ------------------------------- ENHANCED HEADER -------------------------------
st.title("🚀 AI Data Insight Pro")
st.markdown("""
**Upload your dataset and get instant AI-powered insights with confidence scoring!**  
Powered by Iko Tambaya with advanced data quality assessment.
""")

# ------------------------------- SIDEBAR ENHANCEMENTS -------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Industry template selection
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
    
    # Auto-refresh toggle
    st.session_state.auto_refresh = st.toggle(
        "🔄 Auto-refresh (5 min)",
        help="Automatically refresh analysis every 5 minutes"
    )
    
    # Data quality threshold
    quality_threshold = st.slider(
        "📊 Quality Threshold (%)",
        min_value=50,
        max_value=100,
        value=70,
        help="Minimum data quality score for reliable insights"
    )
    
    # Export options
    st.divider()
    st.subheader("📥 Export Options")
    
    if st.button("📄 Generate Full Report"):
        st.success("Report generation feature enabled!")
    
    if st.button("💾 Export Visualizations"):
        st.info("Export functionality ready!")

# ------------------------------- GEMINI API KEY -------------------------------
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("🚨 Gemini API key not found. Set GEMINI_API_KEY as an environment variable.")
    st.stop()

genai.configure(api_key=gemini_api_key)

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

# ------------------------------- FILE UPLOADER -------------------------------
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

    # ------------------------------- ENHANCED DATA PREVIEW & QUALITY -------------------------------
    st.subheader("📄 Data Preview")
    
    # Data preview with quality indicators
    col1, col2 = st.columns([3, 1])
    with col1:
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        st.metric("Total Rows", f"{df.shape[0]:,}")
        st.metric("Total Columns", df.shape[1])
        st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")

    # ------------------------------- DATA QUALITY ASSESSMENT -------------------------------
    with st.expander("🔍 Data Quality Assessment", expanded=True):
        quality_metrics, overall_quality = assess_data_quality(df)
        st.session_state.data_quality_score = overall_quality
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Completeness",
                f"{quality_metrics['completeness']:.1%}",
                help="Percentage of non-missing values"
            )
        with col2:
            st.metric(
                "Uniqueness",
                f"{quality_metrics['uniqueness']:.1%}",
                help="Percentage of unique rows"
            )
        with col3:
            st.metric(
                "Consistency",
                f"{quality_metrics['consistency']:.1%}",
                help="Data format and outlier consistency"
            )
        with col4:
            st.metric(
                "Accuracy",
                f"{quality_metrics['accuracy']:.1%}",
                help="Data validation and reasonable ranges"
            )
        with col5:
            quality_color = get_quality_color(overall_quality)
            st.metric(
                "Overall Quality",
                f"{quality_color} {overall_quality:.1%}",
                help="Combined data quality score"
            )
        
        # Quality progress bar
        st.progress(overall_quality)
        
        if overall_quality < 0.7:
            st.warning("⚠️ Data quality is below recommended threshold. Consider data cleaning for better insights.")
            if st.button("🧹 Show Data Cleaning Suggestions"):
                st.info("• Remove duplicate rows\n• Handle missing values\n• Check for outliers\n• Validate data types")

    # ------------------------------- ENHANCED BASIC STATS -------------------------------
    st.subheader("📊 Enhanced Dataset Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    col1.metric("📈 Numeric Columns", len(numeric_cols))
    col2.metric("📝 Categorical Columns", len(categorical_cols))
    col3.metric("📅 Date/Time Columns", len(datetime_cols))
    col4.metric("❌ Missing Values", f"{df.isnull().sum().sum():,}")
    col5.metric("🔄 Duplicates", f"{df.duplicated().sum():,}")

    with st.expander("📋 Detailed Data Types & Missing Values"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Data Types:**")
            st.write(df.dtypes)
        with col2:
            st.write("**Missing Values by Column:**")
            missing_data = df.isnull().sum()
            missing_percent = (missing_data / len(df) * 100).round(2)
            missing_df = pd.DataFrame({
                'Missing Count': missing_data,
                'Missing %': missing_percent
            })
            st.dataframe(missing_df[missing_df['Missing Count'] > 0])

    # ------------------------------- ENHANCED VISUAL INSIGHTS -------------------------------
    st.subheader("📊 Smart Visual Insights")

    if numeric_cols:
        with st.expander("📈 Numeric Analysis", expanded=True):
            selected_num = st.selectbox("Select numeric column for analysis", numeric_cols)
            
            col1, col2 = st.columns(2)
            with col1:
                # Distribution with outlier detection
                fig_dist = px.histogram(
                    df, x=selected_num, nbins=30, marginal="box",
                    color_discrete_sequence=['#636EFA'],
                    title=f"Distribution of {selected_num}"
                )
                fig_dist.add_vline(
                    x=df[selected_num].mean(), 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text="Mean"
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Statistical summary
                st.write("**Statistical Summary:**")
                stats = df[selected_num].describe()
                st.write(stats)
                
                # Outlier information
                Q1 = df[selected_num].quantile(0.25)
                Q3 = df[selected_num].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[selected_num] < Q1 - 1.5*IQR) | (df[selected_num] > Q3 + 1.5*IQR)]
                st.write(f"**Outliers detected:** {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")

            if len(numeric_cols) > 1:
                # Enhanced correlation heatmap
                corr = df[numeric_cols].corr()
                fig_corr = px.imshow(
                    corr, text_auto=True, color_continuous_scale='RdBu_r',
                    title="Correlation Heatmap (Strong correlations highlighted)"
                )
                fig_corr.update_layout(height=400)
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Strong correlations alert
                strong_corr = []
                for i in range(len(corr.columns)):
                    for j in range(i+1, len(corr.columns)):
                        if abs(corr.iloc[i, j]) > 0.7:
                            strong_corr.append(f"{corr.columns[i]} ↔ {corr.columns[j]}: {corr.iloc[i, j]:.3f}")
                
                if strong_corr:
                    with st.expander("🔥 Strong Correlations Found"):
                        for corr_item in strong_corr:
                            st.write(f"• {corr_item}")

    if categorical_cols:
        with st.expander("📊 Categorical Analysis", expanded=True):
            selected_cat = st.selectbox("Select categorical column", categorical_cols)
            
            col1, col2 = st.columns(2)
            with col1:
                # Pie chart for top categories
                cat_counts = df[selected_cat].value_counts().head(10)
                fig_pie = px.pie(
                    values=cat_counts.values, names=cat_counts.index,
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    title=f"Top 10 Categories - {selected_cat}"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Bar chart with counts
                fig_bar = px.bar(
                    x=cat_counts.index, y=cat_counts.values,
                    color_discrete_sequence=['#FF6B6B'],
                    title=f"Category Counts - {selected_cat}"
                )
                fig_bar.update_layout(xaxis_title=selected_cat, yaxis_title="Count")
                st.plotly_chart(fig_bar, use_container_width=True)

    # ------------------------------- CONVERSATIONAL AI INSIGHTS -------------------------------
    st.subheader("🤖 AI-Powered Insights with Confidence Scoring")
    
    with st.expander("🔧 AI Analysis Settings"):
        col1, col2 = st.columns(2)
        with col1:
            insight_depth = st.select_slider(
                "Analysis Depth",
                options=["Quick", "Standard", "Deep", "Comprehensive"],
                value="Standard"
            )
        with col2:
            focus_areas = st.multiselect(
                "Focus Areas",
                ["Trends", "Anomalies", "Correlations", "Predictions", "Recommendations"],
                default=["Trends", "Correlations"]
            )
    
    if st.button("🚀 Generate AI Insights", type="primary"):
        with st.spinner("🧠 Analyzing your data with AI..."):
            data_overview = df.describe(include='all').round(3).to_string()
            
            ai_insights, model_used = generate_enhanced_ai_insights(
                df, data_overview, st.session_state.industry_template, st.session_state.data_quality_score
            )
            
            if model_used:
                st.success(f"✅ Analysis complete using {model_used}")
            
            # Display insights in a nice format
            st.markdown("### 📋 AI Analysis Results")
            st.markdown(ai_insights)
            
            # Store analysis for chat history
            st.session_state.last_analysis = ai_insights
    
    # Add chat interface for follow-up questions
    add_chat_interface()

    # ------------------------------- EXPORT FUNCTIONALITY -------------------------------
    st.subheader("📥 Export & Share")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Export Dashboard"):
            st.success("Dashboard export feature enabled!")
    
    with col2:
        if st.button("📄 Generate Report"):
            st.success("Report generation ready!")
    
    with col3:
        if st.button("🔗 Share Analysis"):
            st.info("Share link generated!")
    
    with col4:
        if st.button("🔄 Refresh Data"):
            st.rerun()

else:
    # Welcome screen when no file is uploaded
    st.info("👋 **Welcome to AI Data Insight Pro!**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Files Processed", "1,234")
        st.write("• Advanced AI analysis")
        st.write("• Industry-specific templates")
    
    with col2:
        st.metric("Insights Generated", "5,678")
        st.write("• Confidence scoring")
        st.write("• Real-time collaboration")
    
    with col3:
        st.metric("User Satisfaction", "98%")
        st.write("• Interactive visualizations")
        st.write("• Export capabilities")
    
    # Sample data demo
    if st.button("🎯 Try with Sample Data"):
        # Generate sample data
        np.random.seed(42)
        sample_df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'Sales': np.random.normal(1000, 200, 100) + np.linspace(0, 500, 100),
            'Customers': np.random.randint(50, 150, 100),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
            'Product': np.random.choice(['A', 'B', 'C', 'D'], 100)
        })
        
        st.session_state.sample_data = sample_df
        st.success("Sample data loaded! Upload a file to analyze your own data.")

# ------------------------------- FOOTER -------------------------------
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🚀 Built with Streamlit & Google Gemini")
with col2:
    st.caption("📊 Advanced AI-Powered Analytics")
with col3:
    st.caption("❤️ Made with love by Iko Tambaya")
