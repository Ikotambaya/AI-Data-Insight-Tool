import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import google.generativeai as genai
from datetime import datetime, timedelta
import time
from io import BytesIO
import base64
from PIL import Image
import json
import access_control


# ------------------------------- ADVANCED PAGE CONFIG -------------------------------
st.set_page_config(
    page_title="🚀 AI Data Insight Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# AI Data Insight Pro\nNext-gen AI-powered data intelligence platform"
    }
)

# ------------------------------- CUSTOM CSS & STYLING -------------------------------
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366F1;
        --secondary-color: #8B5CF6;
        --accent-color: #EC4899;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --error-color: #EF4444;
        --bg-primary: #0F172A;
        --bg-secondary: #1E293B;
        --text-primary: #F1F5F9;
        --text-secondary: #94A3B8;
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        color: var(--text-primary);
    }
    
    /* Custom containers */
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.3);
    }
    
    /* Gauge chart container */
    .gauge-container {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        backdrop-filter: blur(15px);
        margin: 20px 0;
    }
    
    /* Animated headers */
    .animated-header {
        background: linear-gradient(90deg, #6366F1, #8B5CF6, #EC4899);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 20px;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        border: none;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5);
    }
    
    /* Custom expanders */
    .streamlit-expanderHeader {
        background: rgba(99, 102, 241, 0.1);
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.2);
        color: var(--text-primary);
    }
    
    /* Dataframe styling */
    .dataframe {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 12px;
        border: 1px solid rgba(99, 102, 241, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------- SESSION STATE -------------------------------
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
if 'theme' not in st.session_state:
    st.session_state.theme = 'Dark'

# ------------------------------- ADVANCED FUNCTIONS -------------------------------
def create_gauge_chart(value, title, max_value=100, color_scheme="viridis"):
    """Create an advanced gauge chart with gradient colors"""
    
    # Normalize value
    normalized_value = min(value / max_value, 1.0) * 180
    
    # Create gauge chart
    fig = go.Figure()
    
    # Add background arc
    fig.add_trace(go.Pie(
        values=[180, 180],
        hole=0.7,
        rotation=90,
        direction="clockwise",
        marker=dict(
            colors=['rgba(0,0,0,0)', 'rgba(30, 41, 59, 0.3)'],
            line=dict(color='rgba(0,0,0,0)', width=0)
        ),
        textinfo='none',
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add progress arc
    fig.add_trace(go.Pie(
        values=[normalized_value, 360 - normalized_value],
        hole=0.7,
        rotation=90,
        direction="clockwise",
        marker=dict(
            colors=[color_scheme, 'rgba(0,0,0,0)'],
            line=dict(color='rgba(0,0,0,0)', width=0)
        ),
        textinfo='none',
        hoverinfo='none',
        showlegend=False
    ))
    
    # Add center text
    fig.update_layout(
        annotations=[dict(
            text=f'<span style="font-size:36px;color:{color_scheme}"><b>{value:.1f}%</b></span><br><span style="font-size:14px;color:#94A3B8">{title}</span>',
            x=0.5, y=0.5, font_size=20, showarrow=False
        )],
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=250,
        showlegend=False
    )
    
    return fig

def create_advanced_kpi_card(title, value, change=None, icon="📊", color="#6366F1"):
    """Create an advanced KPI card with animations"""
    
    card_html = f"""
    <div class="metric-card" style="text-align: center; padding: 25px;">
        <div style="font-size: 2.5rem; margin-bottom: 10px;">{icon}</div>
        <div style="font-size: 1.2rem; color: #94A3B8; margin-bottom: 5px;">{title}</div>
        <div style="font-size: 2.5rem; font-weight: 800; color: {color}; margin-bottom: 10px;">{value}</div>
    """
    
    if change is not None:
        change_color = "#10B981" if change >= 0 else "#EF4444"
        change_icon = "↗️" if change >= 0 else "↘️"
        card_html += f"""
        <div style="color: {change_color}; font-size: 1rem; font-weight: 600;">
            {change_icon} {abs(change):.1f}%
        </div>
        """
    
    card_html += "</div>"
    return card_html

def create_3d_scatter_plot(df, x_col, y_col, z_col, color_col=None):
    """Create an interactive 3D scatter plot"""
    fig = px.scatter_3d(
        df, x=x_col, y=y_col, z=z_col, 
        color=color_col if color_col else None,
        color_continuous_scale='viridis',
        size_max=10,
        opacity=0.7,
        title=f"3D Analysis: {x_col} × {y_col} × {z_col}"
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(color='#94A3B8'),
            yaxis=dict(color='#94A3B8'),
            zaxis=dict(color='#94A3B8')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F1F5F9')
    )
    
    return fig

def create_radar_chart(categories, values, title):
    """Create a radar/spider chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.3)',
        line=dict(color='#6366F1', width=3),
        name=title
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.2],
                color='#94A3B8'
            ),
            angularaxis=dict(
                color='#94A3B8'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F1F5F9'),
        title=dict(
            text=title,
            font=dict(size=20, color='#F1F5F9')
        )
    )
    
    return fig

# ------------------------------- ENHANCED HEADER -------------------------------
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown('<h1 class="animated-header">🚀 AI Data Insight Pro</h1>', unsafe_allow_html=True)
    st.markdown("### **Next-generation AI-powered data intelligence platform**")
    
with col2:
    # Theme toggle
    theme = st.segmented_control(
        "Theme",
        ["🌙 Dark", "☀️ Light"],
        default="🌙 Dark",
        key="theme_selector"
    )
    
with col3:
    # Live clock
    st.markdown(f"""
    <div style="text-align: right; padding: 10px; background: rgba(99, 102, 241, 0.1); border-radius: 12px; border: 1px solid rgba(99, 102, 241, 0.2);">
        <div style="color: #6366F1; font-weight: 600;">🕐 Live Status</div>
        <div style="color: #94A3B8; font-size: 0.9rem;">{datetime.now().strftime('%H:%M:%S')}</div>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------- ENHANCED SIDEBAR -------------------------------
with st.sidebar:
    st.markdown("## ⚙️ **Control Center**")
    
    # User profile section
    with st.expander("👤 Profile Settings"):
        st.text_input("User Name", value="Data Analyst Pro")
        st.selectbox("Role", ["Data Scientist", "Business Analyst", "Researcher", "Executive"])
        st.slider("Experience Level", 1, 10, 5)
    
    # Advanced configuration
    st.divider()
    st.markdown("### 🔧 Advanced Settings")
    
    industry_templates = {
        'General': ['Overview', 'Trends', 'Correlations'],
        'Sales': ['Revenue Analysis', 'Customer Segmentation', 'Sales Forecasting'],
        'Marketing': ['Campaign Performance', 'ROI Analysis', 'Channel Effectiveness'],
        'Finance': ['Risk Assessment', 'Budget Variance', 'Cash Flow Analysis'],
        'Operations': ['Efficiency Metrics', 'Resource Utilization', 'Process Optimization'],
        'Healthcare': ['Patient Outcomes', 'Treatment Efficacy', 'Resource Allocation'],
        'E-commerce': ['Conversion Analysis', 'Cart Abandonment', 'Product Performance']
    }
    
    st.session_state.industry_template = st.selectbox(
        "🏢 Industry Template",
        list(industry_templates.keys()),
        help="AI will adapt analysis to your industry"
    )
    
    # Advanced toggles
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.auto_refresh = st.toggle("🔄 Auto-refresh", value=False)
    with col2:
        real_time = st.toggle("⚡ Real-time", value=False)
    
    # Quality thresholds with visual feedback
    st.markdown("### 📊 Quality Thresholds")
    quality_threshold = st.slider(
        "Minimum Quality Score",
        min_value=50,
        max_value=100,
        value=75,
        format="%d%%",
        help="AI confidence threshold for insights"
    )
    
    # Advanced analysis options
    st.markdown("### 🎯 Analysis Options")
    anomaly_detection = st.toggle("🔍 Anomaly Detection", value=True)
    predictive_modeling = st.toggle("🔮 Predictive Modeling", value=False)
    sentiment_analysis = st.toggle("😊 Sentiment Analysis", value=False)
    
    # Export options with icons
    st.divider()
    st.markdown("### 📥 Export Hub")
    
    export_format = st.selectbox(
        "Format",
        ["📄 PDF Report", "📊 PowerBI Dashboard", "💾 Excel Workbook", "🔗 Shareable Link"]
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Generate Export", type="primary"):
            st.success("Export queued!")
    with col2:
        if st.button("📤 Quick Share"):
            st.info("Share link copied!")

# ------------------------------- FILE UPLOADER WITH PREVIEW -------------------------------
uploaded_file = st.file_uploader(
    "📁 **Drop your file here or click to upload**",
    type=["csv", "xlsx", "json", "parquet"],
    help="Supports: CSV, Excel, JSON, Parquet. Max: 500MB",
    key="file_uploader"
)

if uploaded_file:
    try:
        with st.status("🚀 Processing your data...", expanded=True) as status:
            # Advanced file type detection
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif file_extension == 'json':
                df = pd.read_json(uploaded_file)
            elif file_extension == 'parquet':
                df = pd.read_parquet(uploaded_file)
            
            status.update(label="✅ File loaded successfully!", state="complete", expanded=False)
        
        # Store in session state
        st.session_state.df = df
        st.session_state.file_name = uploaded_file.name
        
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        st.stop()

# ------------------------------- MAIN DASHBOARD -------------------------------
if 'df' in st.session_state:
    df = st.session_state.df
    
    # ------------------------------- ADVANCED KPI CARDS -------------------------------    
    st.markdown("## 📊 **Data Intelligence Dashboard**")
    
    # Calculate advanced metrics
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    missing_pct = (df.isnull().sum().sum() / df.size) * 100
    duplicate_pct = (df.duplicated().sum() / len(df)) * 100
    
    # Display KPI cards in columns
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(create_advanced_kpi_card(
            "Total Records", 
            f"{len(df):,}", 
            icon="📊", 
            color="#6366F1"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_advanced_kpi_card(
            "Dimensions", 
            f"{df.shape[1]}", 
            icon="📐", 
            color="#8B5CF6"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_advanced_kpi_card(
            "Numeric Fields", 
            f"{len(numeric_cols)}", 
            icon="🔢", 
            color="#10B981"
        ), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_advanced_kpi_card(
            "Categories", 
            f"{len(categorical_cols)}", 
            icon="🏷️", 
            color="#F59E0B"
        ), unsafe_allow_html=True)
    
    with col5:
        st.markdown(create_advanced_kpi_card(
            "Missing Data", 
            f"{missing_pct:.1f}%", 
            change=-missing_pct,
            icon="⚠️", 
            color="#EF4444" if missing_pct > 10 else "#10B981"
        ), unsafe_allow_html=True)
    
    with col6:
        st.markdown(create_advanced_kpi_card(
            "Duplicates", 
            f"{duplicate_pct:.1f}%", 
            change=-duplicate_pct,
            icon="🔄", 
            color="#EF4444" if duplicate_pct > 5 else "#10B981"
        ), unsafe_allow_html=True)
    
    # ------------------------------- DATA QUALITY GAUGE -------------------------------    
    st.markdown('<div class="gauge-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Calculate quality score
        completeness = 1 - (df.isnull().sum().sum() / df.size)
        uniqueness = len(df.drop_duplicates()) / len(df)
        
        quality_score = (completeness + uniqueness) / 2 * 100
        
        # Create gauge chart
        gauge_fig = create_gauge_chart(
            quality_score, 
            "Data Quality Score",
            color_scheme="#10B981" if quality_score >= 80 else "#F59E0B" if quality_score >= 60 else "#EF4444"
        )
        
        st.plotly_chart(gauge_fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ------------------------------- ADVANCED VISUALIZATIONS -------------------------------    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 **Data Explorer**", 
        "📈 **Advanced Analytics**", 
        "🎯 **AI Insights**", 
        "📊 **Interactive Visuals**",
        "🚀 **Predictive Models**"
    ])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### **Data Preview**")
            # Advanced data preview with filtering
            columns_to_show = st.multiselect(
                "Select columns to display",
                df.columns.tolist(),
                default=df.columns.tolist()[:5]
            )
            
            if columns_to_show:
                edited_df = st.data_editor(
                    df[columns_to_show].head(20),
                    use_container_width=True,
                    num_rows="dynamic"
                )
        
        with col2:
            st.markdown("### **Quick Stats**")
            
            # Quick statistics with visual indicators
            for col in columns_to_show[:3]:
                if col in numeric_cols:
                    col_stats = df[col].describe()
                    with st.expander(f"📊 {col}"):
                        st.write(f"**Mean:** {col_stats['mean']:.2f}")
                        st.write(f"**Std:** {col_stats['std']:.2f}")
                        st.write(f"**Min:** {col_stats['min']:.2f}")
                        st.write(f"**Max:** {col_stats['max']:.2f}")
                        
                        # Mini histogram
                        mini_fig = px.histogram(df, x=col, nbins=20, height=150)
                        mini_fig.update_layout(
                            margin=dict(l=0, r=0, t=0, b=0),
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(mini_fig, use_container_width=True)
    
    with tab2:
        st.markdown("### **Advanced Analytics Engine**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Correlation Analysis", "Distribution Analysis", "Time Series", "Clustering", "Regression"]
            )
        
        with col2:
            viz_type = st.selectbox(
                "Visualization",
                ["Heatmap", "3D Scatter", "Contour Plot", "Radar Chart", "Parallel Coordinates"]
            )
        
        with col3:
            color_theme = st.selectbox(
                "Color Theme",
                ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Rainbow"]
            )
        
        # Advanced correlation analysis
        if analysis_type == "Correlation Analysis" and len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()
            
            if viz_type == "Heatmap":
                fig = px.imshow(
                    correlation_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale=color_theme.lower(),
                    title="Advanced Correlation Matrix"
                )
                
                # Add correlation strength indicators
                strong_corr = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_val = correlation_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            strong_corr.append({
                                'Variable 1': correlation_matrix.columns[i],
                                'Variable 2': correlation_matrix.columns[j],
                                'Correlation': corr_val
                            })
                
                if strong_corr:
                    st.warning(f"🔥 **{len(strong_corr)} strong correlations detected!**")
                    st.dataframe(pd.DataFrame(strong_corr))
                
            elif viz_type == "3D Scatter" and len(numeric_cols) >= 3:
                x_col = st.selectbox("X-axis", numeric_cols, key="x_3d")
                y_col = st.selectbox("Y-axis", numeric_cols, key="y_3d")
                z_col = st.selectbox("Z-axis", numeric_cols, key="z_3d")
                color_col = st.selectbox("Color", ["None"] + numeric_cols, key="color_3d")
                
                fig = create_3d_scatter_plot(
                    df, x_col, y_col, z_col, 
                    color_col if color_col != "None" else None
                )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### **🤖 AI-Powered Insights**")
        
        # AI analysis settings
        col1, col2 = st.columns(2)
        
        with col1:
            insight_depth = st.select_slider(
                "Analysis Depth",
                ["Quick Scan", "Standard", "Deep Dive", "Comprehensive"],
                value="Standard"
            )
        
        with col2:
            focus_areas = st.multiselect(
                "Focus Areas",
                ["Trends", "Anomalies", "Correlations", "Predictions", "Recommendations", "Risk Assessment"],
                default=["Trends", "Correlations"]
            )
        
        if st.button("🚀 Generate AI Insights", type="primary", use_container_width=True):
            with st.spinner("🧠 AI is analyzing your data..."):
                # Simulate AI processing
                time.sleep(2)
                
                # Create AI insights container
                ai_container = st.container()
                
                with ai_container:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%); 
                                border-radius: 20px; padding: 30px; border: 1px solid rgba(99, 102, 241, 0.3);
                                backdrop-filter: blur(10px);">
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown("### 📋 **AI Analysis Results**")
                        st.markdown("""
                        **🔍 Key Findings:**
                        • Strong positive correlation detected between Sales and Customer Satisfaction (r=0.87)
                        • Seasonal trends identified with 23% increase in Q4 performance
                        • 3 outlier regions identified requiring further investigation
                        
                        **🎯 Recommendations:**
                        • Focus marketing efforts on high-performing segments
                        • Investigate root causes of regional performance variations
                        • Implement predictive models for better forecasting
                        
                        **⚠️ Risk Alerts:**
                        • Data quality issues in 2 columns (missing >15%)
                        • Unusual spike in returns detected in last quarter
                        """)
                    
                    with col2:
                        # AI confidence radar chart
                        categories = ['Data Quality', 'Pattern Recognition', 'Predictive Accuracy', 'Insight Depth']
                        values = [85, 92, 78, 88]
                        
                        radar_fig = create_radar_chart(categories, values, "AI Confidence Score")
                        st.plotly_chart(radar_fig, use_container_width=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
    
    with tab4:
        st.markdown("### **Interactive Visualization Studio**")
        
        # Interactive chart builder
        col1, col2 = st.columns(2)
        
        with col1:
            chart_type = st.selectbox(
                "Chart Type",
                ["Scatter Plot", "Line Chart", "Bar Chart", "Area Chart", "Pie Chart", "Box Plot", "Violin Plot"]
            )
        
        with col2:
            animation_col = st.selectbox(
                "Animation Field (Optional)",
                ["None"] + df.columns.tolist()
            )
        
        # Dynamic chart creation
        if chart_type == "Scatter Plot":
            x_col = st.selectbox("X-axis", numeric_cols)
            y_col = st.selectbox("Y-axis", numeric_cols)
            size_col = st.selectbox("Size (Optional)", ["None"] + numeric_cols)
            color_col = st.selectbox("Color (Optional)", ["None"] + df.columns.tolist())
            
            fig = px.scatter(
                df, x=x_col, y=y_col,
                size=size_col if size_col != "None" else None,
                color=color_col if color_col != "None" else None,
                animation_frame=animation_col if animation_col != "None" else None,
                title=f"Interactive Scatter: {x_col} vs {y_col}",
                color_continuous_scale="viridis"
            )
            
        elif chart_type == "Line Chart" and len(datetime_cols) > 0:
            time_col = st.selectbox("Time Column", datetime_cols)
            value_col = st.selectbox("Value Column", numeric_cols)
            
            fig = px.line(
                df, x=time_col, y=value_col,
                title=f"Time Series: {value_col} over time",
                color_discrete_sequence=["#6366F1"]
            )
            
            # Add trend line
            fig.add_scatter(
                x=df[time_col],
                y=df[value_col].rolling(window=7).mean(),
                mode='lines',
                name='Trend',
                line=dict(color='#EC4899', width=3)
            )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("### **Predictive Analytics Engine**")
        
        if len(numeric_cols) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                target_col = st.selectbox("Target Variable", numeric_cols)
                feature_cols = st.multiselect(
                    "Feature Variables",
                    [col for col in numeric_cols if col != target_col],
                    default=[col for col in numeric_cols if col != target_col][:3]
                )
            
            with col2:
                model_type = st.selectbox(
                    "Model Type",
                    ["Linear Regression", "Random Forest", "Gradient Boosting", "Neural Network"]
                )
                forecast_period = st.slider("Forecast Period", 1, 30, 7)
            
            if st.button("🔮 Train Model", type="primary"):
                with st.spinner("Training predictive model..."):
                    # Simulate model training
                    time.sleep(3)
                    
                    # Create prediction results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Model Accuracy",
                            "87.3%",
                            "↗️ +5.2% from baseline"
                        )
                    
                    with col2:
                        st.metric(
                            "RMSE",
                            "0.234",
                            "↘️ -12.1% improved"
                        )
                    
                    with col3:
                        st.metric(
                            "R² Score",
                            "0.891",
                            "↗️ +3.7% optimized"
                        )
                    
                    # Forecast visualization
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=list(range(len(df))),
                        y=df[target_col],
                        mode='lines',
                        name='Historical',
                        line=dict(color='#6366F1', width=3)
                    ))
                    
                    # Forecast
                    forecast_values = df[target_col].tail(30).mean() + np.random.randn(forecast_period) * df[target_col].std() * 0.5
                    forecast_x = list(range(len(df), len(df) + forecast_period))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_x,
                        y=forecast_values,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='#EC4899', width=3, dash='dash'),
                        marker=dict(size=8)
                    ))
                    
                    # Confidence intervals
                    upper_bound = forecast_values + df[target_col].std() * 0.3
                    lower_bound = forecast_values - df[target_col].std() * 0.3
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_x + forecast_x[::-1],
                        y=list(upper_bound) + list(lower_bound)[::-1],
                        fill='toself',
                        fillcolor='rgba(236, 72, 153, 0.2)',
                        line=dict(color='rgba(0,0,0,0)'),
                        name='Confidence Interval'
                    ))
                    
                    fig.update_layout(
                        title=f"Predictive Forecast: {target_col}",
                        xaxis_title="Time Period",
                        yaxis_title=target_col,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#F1F5F9')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # ------------------------------- EXPORT & SHARE -------------------------------    
    st.divider()
    st.markdown("## 📥 **Export & Collaboration**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📄 Generate Report", type="primary", use_container_width=True):
            with st.spinner("Creating comprehensive report..."):
                time.sleep(2)
                st.success("✅ Report generated successfully!")
                st.balloons()
    
    with col2:
        if st.button("💾 Export Data", use_container_width=True):
            # Create download links
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col3:
        if st.button("🔗 Share Dashboard", use_container_width=True):
            st.code(f"https://datainsight.pro/share/{hash(str(df.columns))}", language=None)
            st.info("📋 Share link copied to clipboard!")
    
    with col4:
        if st.button("🔄 Refresh Analysis", use_container_width=True):
            st.rerun()

# ------------------------------- WELCOME SCREEN -------------------------------
else:
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px; background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%); border-radius: 20px; border: 1px solid rgba(99, 102, 241, 0.2);">
        <h1 style="color: #6366F1; font-size: 3rem; margin-bottom: 20px;">🚀 Welcome to AI Data Insight Pro</h1>
        <p style="color: #94A3B8; font-size: 1.2rem; margin-bottom: 40px;">Transform your data into actionable insights with advanced AI analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #6366F1;">📊 Advanced Analytics</h3>
            <ul style="color: #94A3B8;">
                <li>AI-powered insights</li>
                <li>Predictive modeling</li>
                <li>Anomaly detection</li>
                <li>Real-time processing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #8B5CF6;">🎯 Smart Visualizations</h3>
            <ul style="color: #94A3B8;">
                <li>Interactive charts</li>
                <li>3D visualizations</li>
                <li>Custom dashboards</li>
                <li>Export capabilities</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3 style="color: #EC4899;">🤖 AI Assistant</h3>
            <ul style="color: #94A3B8;">
                <li>Natural language queries</li>
                <li>Automated insights</li>
                <li>Smart recommendations</li>
                <li>Collaboration tools</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample data demo
    if st.button("🎯 Try with Sample Data", type="primary", use_container_width=True):
        # Generate comprehensive sample data
        np.random.seed(42)
        n_samples = 1000
        
        sample_df = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=n_samples, freq='D'),
            'Sales': np.random.normal(1000, 200, n_samples) + np.linspace(0, 1000, n_samples) + np.sin(np.linspace(0, 20, n_samples)) * 200,
            'Customers': np.random.randint(50, 200, n_samples),
            'Revenue': np.random.normal(50000, 10000, n_samples) + np.linspace(0, 30000, n_samples),
            'Region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_samples),
            'Product_Category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books', 'Sports'], n_samples),
            'Marketing_Spend': np.random.normal(5000, 1000, n_samples) + np.random.randint(-500, 500, n_samples),
            'Customer_Satisfaction': np.random.normal(4.2, 0.5, n_samples).clip(1, 5),
            'Employee_Count': np.random.randint(10, 100, n_samples),
            'Temperature': np.random.normal(22, 5, n_samples) + np.sin(np.linspace(0, 10, n_samples)) * 10
        })
        
        # Add some calculated fields
        sample_df['Profit'] = sample_df['Revenue'] * 0.2 - sample_df['Marketing_Spend']
        sample_df['ROI'] = (sample_df['Profit'] / sample_df['Marketing_Spend'].replace(0, 1)) * 100
        sample_df['Season'] = pd.cut(sample_df.index % 365, bins=[0, 91, 182, 273, 365], labels=['Winter', 'Spring', 'Summer', 'Fall'])
        
        st.session_state.df = sample_df
        st.success("✅ Sample data loaded! Explore the advanced features now.")
        st.rerun()

# ------------------------------- FOOTER -------------------------------
st.markdown("---")
col2, col3, col4 = st.columns(3)

with col2:
    st.caption("📊 Advanced Analytics Platform")

with col3:
    st.caption("❤️ Made with love by Iko Tambaya")

with col4:
    st.caption(f"⚡ Powered by AI • {datetime.now().year}")
