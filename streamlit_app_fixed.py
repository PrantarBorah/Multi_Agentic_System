import streamlit as st
import pandas as pd
import json
import os
from app import DataPipelineOrchestrator
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import io

# Set page config - MUST be first Streamlit command
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
    </style>
    """, unsafe_allow_html=True)

def load_sample_datasets():
    """Load available sample datasets"""
    sample_data_dir = Path("sample_data")
    return [f.name for f in sample_data_dir.glob("*.csv")]

def analyze_uploaded_dataset(data: pd.DataFrame) -> dict:
    """Analyze uploaded dataset to identify problem type and target variable"""
    analysis = {
        "target_variable": None,
        "problem_type": "Unknown",
        "target_selection_reasoning": [],
        "data_characteristics": {},
        "recommendations": {}
    }
    
    # Basic data characteristics
    analysis["data_characteristics"] = {
        "shape": data.shape,
        "columns": list(data.columns),
        "dtypes": data.dtypes.to_dict(),
        "missing_values": data.isnull().sum().to_dict(),
        "numeric_columns": data.select_dtypes(include=['number']).columns.tolist(),
        "categorical_columns": data.select_dtypes(include=['object', 'category']).columns.tolist()
    }
    
    # Simple target identification
    target_candidates = []
    
    # Look for common target indicators
    target_indicators = ['target', 'label', 'class', 'outcome', 'result', 'disease', 'survived', 'churn', 'price']
    
    for col in data.columns:
        col_lower = col.lower()
        score = 0
        reasons = []
        
        # Check for target indicators
        for indicator in target_indicators:
            if indicator in col_lower:
                score += 10
                reasons.append(f"Column name contains '{indicator}' indicator")
                break
        
        # Check data characteristics
        unique_vals = data[col].nunique()
        unique_ratio = unique_vals / len(data)
        
        # Binary classification patterns
        if unique_vals == 2:
            score += 8
            reasons.append(f"Binary values suggest classification target")
        elif unique_ratio < 0.1:
            score += 5
            reasons.append(f"Low cardinality suggests categorical target")
        
        if score > 0:
            target_candidates.append({
                'column': col,
                'score': score,
                'reasons': reasons,
                'unique_values': unique_vals,
                'dtype': str(data[col].dtype)
            })
    
    # Sort by score
    target_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    if target_candidates:
        best_candidate = target_candidates[0]
        analysis["target_variable"] = best_candidate['column']
        analysis["target_selection_reasoning"] = best_candidate['reasons']
        
        # Determine problem type
        target_data = data[best_candidate['column']]
        if target_data.nunique() == 2:
            analysis["problem_type"] = "Binary Classification"
        elif target_data.nunique() < 20 and target_data.dtype in ['object', 'category']:
            analysis["problem_type"] = "Multi-class Classification"
        else:
            analysis["problem_type"] = "Regression"
            
        analysis["difficulty"] = "🟡 Intermediate"
    
    return analysis

def get_dataset_info(dataset_name, uploaded_data=None, uploaded_analysis=None):
    """Get educational information about the selected dataset"""
    
    if uploaded_data is not None and uploaded_analysis is not None:
        return {
            "difficulty": uploaded_analysis.get("difficulty", "🟡 Intermediate"),
            "problem_type": uploaded_analysis.get("problem_type", "Unknown"),
            "description": f"Uploaded dataset with {uploaded_data.shape[0]} rows and {uploaded_data.shape[1]} columns",
            "challenges": ["Dynamic target identification", "Custom data characteristics"],
            "learning_objectives": ["Understanding target variable selection", "Working with custom datasets"],
            "recommended_config": {}
        }
    
    # Sample dataset info
    dataset_info = {
        "1_survey_lung_cancer.csv": {
            "difficulty": "🟢 Beginner",
            "problem_type": "Binary Classification",
            "description": "Predict lung cancer based on survey responses",
            "challenges": ["Imbalanced classes", "Categorical features"],
            "learning_objectives": ["Binary classification", "Handling imbalanced data"],
            "recommended_config": {}
        },
        "2_Iris.csv": {
            "difficulty": "🟢 Beginner",
            "problem_type": "Multi-class Classification",
            "description": "Classic iris flower classification",
            "challenges": ["Feature scaling", "Model selection"],
            "learning_objectives": ["Multi-class classification", "Feature importance"],
            "recommended_config": {}
        }
    }
    
    return dataset_info.get(dataset_name, {
        "difficulty": "🟡 Intermediate",
        "problem_type": "Unknown",
        "description": "Sample dataset for ML learning",
        "challenges": [],
        "learning_objectives": [],
        "recommended_config": {}
    })

def create_info_box(content, box_type="info"):
    """Create a styled info box"""
    return f"""
    <div class="{box_type}-box">
        {content}
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

def simple_configuration():
    """Simple configuration interface"""
    st.subheader("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Data Cleaning**")
        imputation = st.selectbox("Missing Value Strategy", ["auto", "mean", "median", "mode"])
        outliers = st.selectbox("Outlier Handling", ["auto", "remove", "cap"])
    
    with col2:
        st.markdown("**Model Training**")
        algorithms = st.multiselect("Algorithms", ["RandomForest", "LogisticRegression", "SVM"], default=["RandomForest"])
        cv_strategy = st.selectbox("Cross Validation", ["auto", "stratified", "kfold"])
    
    return {
        "cleaning": {"imputation_method": imputation, "outlier_method": outliers},
        "training": {"algorithms": algorithms, "cv_strategy": cv_strategy},
        "eda": {},
        "evaluation": {}
    }

def display_results(results):
    """Display pipeline results"""
    if not results:
        return
    
    st.subheader("📊 Pipeline Results")
    
    # Create tabs for results
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Problem Detection", "🧹 Data Cleaning", "📊 EDA", "🤖 Model Training", "📈 Evaluation"
    ])
    
    with tab1:
        if "problem_analysis" in results:
            st.json(results["problem_analysis"])
    
    with tab2:
        if "cleaning_summary_results" in results:
            st.write("Data cleaning completed successfully")
            if "original_shape" in results["cleaning_summary_results"]:
                st.write(f"Original shape: {results['cleaning_summary_results']['original_shape']}")
    
    with tab3:
        if "eda_results" in results:
            st.write("EDA completed successfully")
            if "summary_stats" in results["eda_results"]:
                st.write("Summary statistics generated")
    
    with tab4:
        if "model_results" in results:
            st.write("Model training completed successfully")
            if "selected_model" in results["model_results"]:
                st.write(f"Selected model: {results['model_results']['selected_model']}")
    
    with tab5:
        if "evaluation_results" in results:
            st.write("Model evaluation completed successfully")
            if "performance_metrics" in results["evaluation_results"]:
                st.json(results["evaluation_results"]["performance_metrics"])

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
    
    # Description
    st.markdown(create_info_box("""
    <strong>🎯 What makes this special:</strong><br>
    • <strong>5 AI Agents</strong> working together with CrewAI orchestration<br>
    • <strong>Educational Transparency</strong> - Every decision explained<br>
    • <strong>Interactive Learning</strong> - Hands-on experimentation<br>
    • <strong>Target-Aware Processing</strong> - Intelligent data handling
    """), unsafe_allow_html=True)
    
    # Data source selection
    st.sidebar.markdown("## 📁 Data Source")
    data_source = st.sidebar.radio(
        "Choose your data source",
        ["📤 Upload CSV File", "📊 Use Sample Dataset"]
    )
    
    uploaded_data = None
    uploaded_analysis = None
    selected_dataset = None
    
    if data_source == "📤 Upload CSV File":
        st.markdown("### 📤 Upload Your Dataset")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                uploaded_data = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Shape: {uploaded_data.shape}")
                
                # Analyze dataset
                with st.spinner("🔍 Analyzing dataset..."):
                    uploaded_analysis = analyze_uploaded_dataset(uploaded_data)
                
                # Display metrics
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
                
            except Exception as e:
                st.markdown(create_info_box(f"""
                <strong>❌ Error reading file:</strong><br>
                {str(e)}
                """, "error"), unsafe_allow_html=True)
    
    else:
        # Sample dataset selection
        st.sidebar.markdown("## 📊 Dataset Selection")
        available_datasets = load_sample_datasets()
        selected_dataset = st.sidebar.selectbox("Choose a dataset", available_datasets)
    
    # Dataset information
    if uploaded_data is not None:
        dataset_info = get_dataset_info("uploaded", uploaded_data, uploaded_analysis)
    else:
        dataset_info = get_dataset_info(selected_dataset)
    
    # Display dataset info
    with st.expander("📚 About This Dataset", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Difficulty:** {dataset_info['difficulty']}")
            st.markdown(f"**Problem Type:** {dataset_info['problem_type']}")
            st.markdown(f"**Description:** {dataset_info['description']}")
        
        with col2:
            if dataset_info['challenges']:
                st.markdown("**Challenges:**")
                for challenge in dataset_info['challenges']:
                    st.markdown(f"• {challenge}")
            
            if dataset_info['learning_objectives']:
                st.markdown("**Learning Objectives:**")
                for obj in dataset_info['learning_objectives']:
                    st.markdown(f"• {obj}")
    
    # Configuration
    st.markdown("---")
    config = simple_configuration()
    
    # Initialize session state
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = None
    
    # Pipeline execution
    st.markdown("---")
    st.markdown("## 🚀 Execute ML Pipeline")
    
    if st.button("🚀 Run Pipeline", type="primary", use_container_width=True):
        # Get API key
        try:
            openai_api_key = st.secrets["OPENAI_API_KEY"]
        except:
            try:
                with open('.env', 'r') as f:
                    for line in f:
                        if line.startswith('OPENAI_API_KEY'):
                            openai_api_key = line.split('=')[1].strip().strip('"')
                            break
                    else:
                        raise ValueError("API key not found")
            except:
                st.error("❌ OpenAI API key not found. Please add it to .env file or Streamlit secrets.")
                return
        
        # Run pipeline
        try:
            with st.spinner("🤖 AI Agents are working..."):
                if uploaded_data is not None:
                    temp_path = "temp_uploaded_data.csv"
                    uploaded_data.to_csv(temp_path, index=False)
                    orchestrator = DataPipelineOrchestrator(temp_path, openai_api_key=openai_api_key)
                else:
                    orchestrator = DataPipelineOrchestrator(f"sample_data/{selected_dataset}", openai_api_key=openai_api_key)
                
                orchestrator.run_pipeline(
                    cleaning_config=config["cleaning"],
                    eda_config=config["eda"],
                    training_config=config["training"],
                    evaluation_config=config["evaluation"]
                )
                
                # Load results
                with open("pipeline_results.json", "r") as f:
                    st.session_state.pipeline_results = json.load(f)
                
                # Clean up
                if uploaded_data is not None and os.path.exists(temp_path):
                    os.remove(temp_path)
                
                st.success("✅ Pipeline completed successfully!")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Pipeline failed: {str(e)}")
    
    # Display results
    display_results(st.session_state.pipeline_results)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: var(--bg-secondary); border-radius: var(--radius-lg); margin-top: 2rem;">
        <p style="margin: 0; color: var(--text-secondary);">
            Built with ❤️ using <strong>CrewAI</strong>, <strong>Streamlit</strong>, and <strong>OpenAI GPT-4</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
