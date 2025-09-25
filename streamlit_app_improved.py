import streamlit as st
import pandas as pd
import json
import os
from app import DataPipelineOrchestrator
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import io

# Set page config
st.set_page_config(
    page_title="🤖 ML Pipeline Orchestrator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better readability
st.markdown("""
    <style>
    /* Import better fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* CSS Variables for enhanced theme adaptation */
    :root {
        --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-tertiary: #f1f5f9;
        --bg-info: #eff6ff;
        --bg-success: #f0fdf4;
        --bg-warning: #fffbeb;
        --bg-error: #fef2f2;
        --text-primary: #1e293b;
        --text-secondary: #64748b;
        --text-muted: #94a3b8;
        --border-primary: #3b82f6;
        --border-secondary: #e2e8f0;
        --border-hover: #2563eb;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --radius-sm: 0.375rem;
        --radius-md: 0.5rem;
        --radius-lg: 0.75rem;
    }
    
    /* Dark mode variables */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-tertiary: #334155;
            --bg-info: #1e3a8a;
            --bg-success: #14532d;
            --bg-warning: #92400e;
            --bg-error: #991b1b;
            --text-primary: #f8fafc;
            --text-secondary: #cbd5e1;
            --text-muted: #94a3b8;
            --border-primary: #60a5fa;
            --border-secondary: #475569;
            --border-hover: #93c5fd;
            --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.3);
            --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.3), 0 2px 4px -2px rgb(0 0 0 / 0.3);
            --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.3), 0 4px 6px -4px rgb(0 0 0 / 0.3);
        }
    }
    
    /* Base typography and layout */
    .stApp {
        font-family: var(--font-family) !important;
    }
    
    .main {
        padding: 1rem 2rem;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Enhanced headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-family) !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        line-height: 1.2 !important;
    }
    
    h1 { font-size: 2.5rem !important; margin-bottom: 1rem !important; }
    h2 { font-size: 2rem !important; margin-bottom: 0.75rem !important; }
    h3 { font-size: 1.5rem !important; margin-bottom: 0.5rem !important; }
    
    /* Enhanced paragraph text */
    p, .stMarkdown, .stText {
        font-family: var(--font-family) !important;
        color: var(--text-secondary) !important;
        line-height: 1.6 !important;
        font-size: 1rem !important;
    }
    
    /* Card-like containers */
    .card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-secondary);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }
    
    .card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid var(--border-secondary);
    }
    
    .card-icon {
        font-size: 1.5rem;
        width: 2.5rem;
        height: 2.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--bg-info);
        border-radius: var(--radius-md);
    }
    
    /* Enhanced buttons */
    .stButton > button {
        font-family: var(--font-family) !important;
        font-weight: 500 !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border-primary) !important;
        background: var(--border-primary) !important;
        color: white !important;
        transition: all 0.2s ease !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    .stButton > button:hover {
        background: var(--border-hover) !important;
        border-color: var(--border-hover) !important;
        box-shadow: var(--shadow-md) !important;
        transform: translateY(-1px) !important;
    }
    
    /* Primary button variant */
    .primary-button button {
        background: linear-gradient(135deg, var(--border-primary), var(--border-hover)) !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        padding: 1rem 2rem !important;
    }
    
    /* Enhanced file uploader */
    .upload-area {
        background: var(--bg-tertiary);
        border: 2px dashed var(--border-secondary);
        border-radius: var(--radius-lg);
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
        margin: 1rem 0;
    }
    
    .upload-area:hover {
        border-color: var(--border-primary);
        background: var(--bg-info);
    }
    
    .upload-icon {
        font-size: 3rem;
        color: var(--border-primary);
        margin-bottom: 1rem;
    }
    
    /* Enhanced info boxes */
    .info-box {
        background: var(--bg-info);
        border: 1px solid var(--border-primary);
        border-radius: var(--radius-md);
        padding: 1rem;
        margin: 0.75rem 0;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    .success-box {
        background: var(--bg-success);
        border-color: #22c55e;
        color: #15803d;
    }
    
    .warning-box {
        background: var(--bg-warning);
        border-color: #f59e0b;
        color: #d97706;
    }
    
    .error-box {
        background: var(--bg-error);
        border-color: #ef4444;
        color: #dc2626;
    }
    
    /* Enhanced metrics */
    .metric-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-secondary);
        border-radius: var(--radius-md);
        padding: 1.25rem;
        text-align: center;
        box-shadow: var(--shadow-sm);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--border-primary);
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: var(--text-muted);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Enhanced progress indicators */
    .progress-container {
        background: var(--bg-secondary);
        border-radius: var(--radius-md);
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid var(--border-secondary);
    }
    
    .progress-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .progress-stage {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: var(--bg-info);
        border-radius: var(--radius-sm);
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--bg-tertiary);
        padding: 0.25rem;
        border-radius: var(--radius-md);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border-radius: var(--radius-sm) !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--bg-secondary) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--bg-primary) !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-secondary) !important;
    }
    
    /* Enhanced expanders */
    .stExpander {
        border: 1px solid var(--border-secondary) !important;
        border-radius: var(--radius-md) !important;
        overflow: hidden !important;
        margin: 0.5rem 0 !important;
    }
    
    .stExpander > div:first-child {
        background: var(--bg-tertiary) !important;
        padding: 1rem !important;
        font-weight: 500 !important;
    }
    
    .stExpander > div:first-child:hover {
        background: var(--bg-info) !important;
    }
    
    /* Enhanced form elements */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-secondary) !important;
        border-radius: var(--radius-sm) !important;
        font-family: var(--font-family) !important;
        transition: all 0.2s ease !important;
    }
    
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--border-primary) !important;
        box-shadow: 0 0 0 3px rgb(59 130 246 / 0.1) !important;
    }
    
    /* Enhanced alerts */
    .stAlert {
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border-secondary) !important;
        font-family: var(--font-family) !important;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main {
            padding: 0.5rem 1rem;
        }
        
        .card {
            padding: 1rem;
        }
        
        h1 { font-size: 2rem !important; }
        h2 { font-size: 1.5rem !important; }
        h3 { font-size: 1.25rem !important; }
    }
    
    /* Loading states */
    .loading-container {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        padding: 2rem;
        background: var(--bg-secondary);
        border-radius: var(--radius-md);
        margin: 1rem 0;
    }
    
    .loading-spinner {
        width: 2rem;
        height: 2rem;
        border: 3px solid var(--border-secondary);
        border-top: 3px solid var(--border-primary);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Enhanced data display */
    .stDataFrame {
        border-radius: var(--radius-md) !important;
        border: 1px solid var(--border-secondary) !important;
        overflow: hidden !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-tertiary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-secondary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }
    </style>
    """, unsafe_allow_html=True)

# Enhanced UI Components
def create_card(title, content, icon="📊"):
    """Create a styled card component"""
    return f"""
    <div class="card">
        <div class="card-header">
            <div class="card-icon">{icon}</div>
            <h3 style="margin: 0; color: var(--text-primary);">{title}</h3>
        </div>
        <div style="color: var(--text-secondary);">
            {content}
        </div>
    </div>
    """

def create_metric_card(value, label, icon="📈"):
    """Create a styled metric card"""
    return f"""
    <div class="metric-card">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def create_info_box(content, box_type="info"):
    """Create a styled info box"""
    return f"""
    <div class="{box_type}-box">
        {content}
    </div>
    """

def create_upload_area():
    """Create a styled upload area"""
    return """
    <div class="upload-area">
        <div class="upload-icon">📤</div>
        <h4 style="margin: 0.5rem 0; color: var(--text-primary);">Upload Your Dataset</h4>
        <p style="margin: 0; color: var(--text-secondary);">Drag and drop your CSV file here or click to browse</p>
    </div>
    """

def create_progress_indicator(stage, progress, message):
    """Create a styled progress indicator"""
    return f"""
    <div class="progress-container">
        <div class="progress-header">
            <h4 style="margin: 0; color: var(--text-primary);">Pipeline Progress</h4>
            <div class="progress-stage">{stage}</div>
        </div>
        <div style="margin-bottom: 1rem;">
            <div style="background: var(--border-secondary); height: 8px; border-radius: 4px; overflow: hidden;">
                <div style="background: var(--border-primary); height: 100%; width: {progress}%; transition: width 0.3s ease;"></div>
            </div>
        </div>
        <p style="margin: 0; color: var(--text-secondary); font-size: 0.9rem;">{message}</p>
    </div>
    """

# Import the rest of the functions from the original file
from streamlit_app import (
    load_sample_datasets, analyze_uploaded_dataset, get_dataset_info,
    create_ml_glossary, display_interactive_configuration,
    display_problem_detection_results, display_contextual_learning_popup,
    display_educational_sidebar, display_cleaning_results,
    display_eda_results, display_model_results, display_evaluation_results
)

def main():
    # Enhanced header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, var(--bg-info), var(--bg-secondary)); border-radius: var(--radius-lg); margin-bottom: 2rem;">
        <h1 style="margin: 0; background: linear-gradient(135deg, var(--border-primary), var(--border-hover)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
            🤖 ML Pipeline Orchestrator
        </h1>
        <p style="font-size: 1.2rem; margin: 0.5rem 0 0 0; color: var(--text-secondary);">
            AI-Powered Machine Learning Pipeline with Educational Transparency
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced description
    st.markdown(create_info_box("""
    <strong>🎯 What makes this special:</strong><br>
    • <strong>5 AI Agents</strong> working together with CrewAI orchestration<br>
    • <strong>Educational Transparency</strong> - Every decision explained with ML concepts<br>
    • <strong>Interactive Learning</strong> - Hands-on experimentation with real-time feedback<br>
    • <strong>Target-Aware Processing</strong> - Intelligent handling to prevent data leakage
    """), unsafe_allow_html=True)
    
    # Educational sidebar
    display_educational_sidebar()
    
    # Enhanced data source selection
    st.sidebar.markdown("## 📁 Data Source")
    data_source = st.sidebar.radio(
        "Choose your data source",
        ["📤 Upload CSV File", "📊 Use Sample Dataset"],
        help="Upload your own CSV file or explore with our curated sample datasets"
    )
    
    uploaded_data = None
    uploaded_analysis = None
    selected_dataset = None
    
    if data_source == "📤 Upload CSV File":
        st.markdown("### 📤 Upload Your Dataset")
        
        # Enhanced upload section
        st.markdown(create_upload_area(), unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with your data",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                uploaded_data = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Shape: {uploaded_data.shape}")
                
                # Enhanced analysis display
                with st.spinner("🔍 Analyzing your dataset..."):
                    uploaded_analysis = analyze_uploaded_dataset(uploaded_data)
                
                # Display analysis in enhanced cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(create_metric_card(
                        f"{uploaded_data.shape[0]} × {uploaded_data.shape[1]}", 
                        "Dataset Shape", 
                        "📊"
                    ), unsafe_allow_html=True)
                
                with col2:
                    st.markdown(create_metric_card(
                        uploaded_analysis.get('problem_type', 'Unknown'), 
                        "Problem Type", 
                        "🎯"
                    ), unsafe_allow_html=True)
                
                with col3:
                    st.markdown(create_metric_card(
                        uploaded_analysis.get('target_variable', 'Not Found'), 
                        "Target Variable", 
                        "🏹"
                    ), unsafe_allow_html=True)
                
                # Enhanced target analysis
                if uploaded_analysis['target_variable']:
                    st.markdown(create_info_box(f"""
                    <strong>🎯 Target Analysis:</strong><br>
                    <strong>Variable:</strong> {uploaded_analysis['target_variable']}<br>
                    <strong>Type:</strong> {uploaded_analysis['problem_type']}<br>
                    <strong>Difficulty:</strong> {uploaded_analysis.get('difficulty', '🟡 Intermediate')}
                    """, "success"), unsafe_allow_html=True)
                else:
                    st.markdown(create_info_box("""
                    <strong>⚠️ No Target Variable Identified</strong><br>
                    Please manually select a target variable from your dataset columns.
                    """, "warning"), unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(create_info_box(f"""
                <strong>❌ Error reading file:</strong><br>
                {str(e)}<br><br>
                Please ensure your file is a valid CSV format.
                """, "error"), unsafe_allow_html=True)
    
    else:
        # Enhanced sample dataset selection
        st.sidebar.markdown("## 📊 Dataset Selection")
        available_datasets = load_sample_datasets()
        selected_dataset = st.sidebar.selectbox(
            "Choose a sample dataset",
            available_datasets,
            index=0,
            help="Select from our curated collection of educational datasets"
        )
    
    # Get dataset information
    if uploaded_data is not None:
        dataset_info = get_dataset_info("uploaded", uploaded_data, uploaded_analysis)
    else:
        dataset_info = get_dataset_info(selected_dataset)
    
    # Enhanced dataset information display
    with st.expander("📚 About This Dataset & Learning Opportunities", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(create_card("Dataset Overview", f"""
                <strong>Difficulty Level:</strong> {dataset_info['difficulty']}<br>
                <strong>Problem Type:</strong> {dataset_info['problem_type']}<br>
                <strong>Description:</strong> {dataset_info['description']}
            """, "📋"), unsafe_allow_html=True)
            
            if dataset_info['challenges']:
                challenges_list = "<br>".join([f"• {challenge}" for challenge in dataset_info['challenges']])
                st.markdown(create_card("Key Challenges", challenges_list, "⚡"), unsafe_allow_html=True)
        
        with col2:
            if dataset_info['learning_objectives']:
                objectives_list = "<br>".join([f"• {obj}" for obj in dataset_info['learning_objectives']])
                st.markdown(create_card("Learning Objectives", objectives_list, "🎓"), unsafe_allow_html=True)
            
            if dataset_info['recommended_config']:
                config = dataset_info['recommended_config']
                config_items = []
                for key, value in config.items():
                    if isinstance(value, list):
                        config_items.append(f"<strong>{key.title()}:</strong> {', '.join(value)}")
                    else:
                        config_items.append(f"<strong>{key.title()}:</strong> {value}")
                
                config_text = "<br>".join(config_items)
                st.markdown(create_card("Recommended Settings", config_text, "⚙️"), unsafe_allow_html=True)
                
                col_btn1, col_btn2 = st.columns([1, 2])
                with col_btn1:
                    if st.button("⚡ Apply Settings", key="apply_recommended"):
                        st.session_state.recommended_config = config
                        st.success("✅ Settings applied!")
    
    # Interactive Configuration with enhanced UI
    st.markdown("---")
    st.markdown("## ⚙️ Pipeline Configuration")
    st.markdown(create_info_box("""
    <strong>🎛️ Customize Your ML Pipeline</strong><br>
    Configure each stage to understand the impact of different choices on your model's performance.
    """), unsafe_allow_html=True)
    
    config = display_interactive_configuration()
    
    # Initialize session state for pipeline results
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = None
    
    # Enhanced Pipeline Execution Section
    st.markdown("---")
    st.markdown("## 🚀 Execute ML Pipeline")
    
    # Pipeline progress tracking with enhanced UI
    if 'pipeline_progress' not in st.session_state:
        st.session_state.pipeline_progress = {
            'stage': 'ready',
            'progress': 0,
            'current_step': 'Ready to start pipeline execution',
            'logs': []
        }
    
    # Enhanced progress display
    if st.session_state.pipeline_progress['stage'] != 'ready':
        st.markdown(create_progress_indicator(
            st.session_state.pipeline_progress['stage'].replace('_', ' ').title(),
            st.session_state.pipeline_progress['progress'],
            st.session_state.pipeline_progress['current_step']
        ), unsafe_allow_html=True)
    
    # Enhanced run button
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<div class="primary-button">', unsafe_allow_html=True)
        run_pipeline = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if run_pipeline:
            if uploaded_data is not None:
                st.markdown(create_info_box("""
                <strong>💡 Pipeline Starting!</strong><br>
                Running on your uploaded dataset with selected configuration options.
                """, "info"), unsafe_allow_html=True)
            else:
                st.markdown(create_info_box("""
                <strong>💡 Pipeline Starting!</strong><br>
                Using sample dataset with your configuration. Compare results with different settings!
                """, "info"), unsafe_allow_html=True)
    
    # Pipeline execution with enhanced error handling
    if run_pipeline:
        # Initialize progress
        st.session_state.pipeline_progress = {
            'stage': 'starting',
            'progress': 0,
            'current_step': 'Initializing AI agents...',
            'logs': []
        }
        
        # Create enhanced log container
        with st.container():
            st.markdown("### 📋 Pipeline Execution Log")
            log_placeholder = st.empty()
        
        # Get OpenAI API key
        try:
            openai_api_key = st.secrets["OPENAI_API_KEY"]
        except:
            try:
                # Try to read from .env file
                with open('.env', 'r') as f:
                    for line in f:
                        if line.startswith('OPENAI_API_KEY'):
                            openai_api_key = line.split('=')[1].strip().strip('"')
                            break
                    else:
                        raise ValueError("API key not found")
            except:
                st.markdown(create_info_box("""
                <strong>❌ OpenAI API Key Not Found</strong><br>
                Please add your OpenAI API key to the .env file or Streamlit secrets.
                """, "error"), unsafe_allow_html=True)
                return
        
        # Run pipeline with enhanced error handling
        try:
            if uploaded_data is not None:
                temp_path = "temp_uploaded_data.csv"
                uploaded_data.to_csv(temp_path, index=False)
                orchestrator = DataPipelineOrchestrator(temp_path, openai_api_key=openai_api_key)
            else:
                orchestrator = DataPipelineOrchestrator(f"sample_data/{selected_dataset}", openai_api_key=openai_api_key)
            
            # Progress callback function with enhanced logging
            def update_progress(stage, progress, message):
                st.session_state.pipeline_progress['stage'] = stage
                st.session_state.pipeline_progress['progress'] = progress
                st.session_state.pipeline_progress['current_step'] = message
                st.session_state.pipeline_progress['logs'].append({
                    'stage': stage.upper(),
                    'message': message,
                    'timestamp': pd.Timestamp.now().strftime('%H:%M:%S')
                })
                
                # Update log display
                log_entries = []
                for log in st.session_state.pipeline_progress['logs'][-5:]:  # Show last 5 logs
                    log_entries.append(f"[{log['timestamp']}] **{log['stage']}**: {log['message']}")
                
                log_placeholder.markdown("\n\n".join(log_entries))
            
            # Run pipeline
            with st.spinner("🤖 AI Agents are working..."):
                orchestrator.run_pipeline(
                    cleaning_config=config["cleaning"],
                    eda_config=config["eda"],
                    training_config=config["training"],
                    evaluation_config=config["evaluation"],
                    progress_callback=update_progress
                )
            
            # Load results
            with open("pipeline_results.json", "r") as f:
                st.session_state.pipeline_results = json.load(f)
            
            # Clean up
            if uploaded_data is not None and os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Update progress to completion
            st.session_state.pipeline_progress['stage'] = 'completed'
            st.session_state.pipeline_progress['progress'] = 100
            st.session_state.pipeline_progress['current_step'] = '🎉 Pipeline completed successfully!'
            
            st.markdown(create_info_box("""
            <strong>✅ Pipeline Completed Successfully!</strong><br>
            All AI agents have finished their tasks. Review the results below.
            """, "success"), unsafe_allow_html=True)
            
            st.rerun()
            
        except Exception as e:
            st.session_state.pipeline_progress['stage'] = 'failed'
            st.markdown(create_info_box(f"""
            <strong>❌ Pipeline Failed</strong><br>
            Error: {str(e)}<br>
            Please check your configuration and try again.
            """, "error"), unsafe_allow_html=True)
    
    # Enhanced results display
    if st.session_state.pipeline_results:
        st.markdown("---")
        st.markdown("## 📊 Pipeline Results")
        
        # Create enhanced tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Problem Detection", "🧹 Data Cleaning", "📊 EDA", "🤖 Model Training", "📈 Evaluation"
        ])
        
        with tab1:
            if "problem_analysis" in st.session_state.pipeline_results:
                display_problem_detection_results(st.session_state.pipeline_results["problem_analysis"])
        
        with tab2:
            if "cleaning_summary_results" in st.session_state.pipeline_results:
                display_cleaning_results(st.session_state.pipeline_results["cleaning_summary_results"])
        
        with tab3:
            if "eda_results" in st.session_state.pipeline_results:
                display_eda_results(st.session_state.pipeline_results["eda_results"])
        
        with tab4:
            if "model_results" in st.session_state.pipeline_results:
                display_model_results(st.session_state.pipeline_results["model_results"])
        
        with tab5:
            if "evaluation_results" in st.session_state.pipeline_results:
                display_evaluation_results(st.session_state.pipeline_results["evaluation_results"])
    
    # Enhanced footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: var(--bg-secondary); border-radius: var(--radius-lg); margin-top: 2rem;">
        <p style="margin: 0; color: var(--text-secondary);">
            Built with ❤️ using <strong>CrewAI</strong>, <strong>Streamlit</strong>, and <strong>OpenAI GPT-4</strong>
        </p>
        <p style="margin: 0.5rem 0 0 0; color: var(--text-muted); font-size: 0.875rem;">
            🎓 Educational ML Pipeline • 🤖 Multi-Agent AI • 🔍 Transparent Decisions
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
