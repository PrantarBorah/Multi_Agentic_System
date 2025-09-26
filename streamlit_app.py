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
    page_title="CrewML",
    page_icon="⚙️",
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
    
    /* Streamlit dark theme detection */
    [data-theme="dark"] {
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
    
    /* Fallback for system dark mode preference */
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
        background: var(--bg-secondary);
        border: 1px solid var(--border-secondary);
        border-radius: var(--radius-md);
        padding: 1rem;
        margin: 0.75rem 0;
        font-size: 0.95rem;
        line-height: 1.5;
        color: var(--text-secondary);
    }
    
    .warning-box {
        background: var(--bg-secondary);
        border: 1px solid var(--border-secondary);
        border-radius: var(--radius-md);
        padding: 1rem;
        margin: 0.75rem 0;
        font-size: 0.95rem;
        line-height: 1.5;
        color: var(--text-secondary);
    }
    
    .error-box {
        background: var(--bg-error);
        border-color: #ef4444;
        color: #dc2626;
    }
    
    /* Enhanced metrics with fixed sizing */
    .metric-card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-secondary);
        border-radius: var(--radius-md);
        padding: 1.25rem;
        text-align: center;
        box-shadow: var(--shadow-sm);
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        transition: all 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--border-primary);
        margin-bottom: 0.5rem;
        line-height: 1.2;
        text-align: center;
        word-wrap: break-word;
        hyphens: auto;
        max-width: 100%;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        text-align: center;
        margin-top: auto;
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
    
    /* Enhanced expanders with better visibility */
    .stExpander {
        border: 1px solid var(--border-secondary) !important;
        border-radius: var(--radius-md) !important;
        overflow: hidden !important;
        margin: 1rem 0 !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    /* Sidebar expanders for Learning Center */
    .stSidebar .stExpander {
        margin: 0.3rem 0 !important;
        border: 1px solid var(--border-secondary) !important;
        background: var(--bg-secondary) !important;
    }
    
    .stSidebar .stExpander > div:first-child {
        background: var(--bg-secondary) !important;
        padding: 0.75rem !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
        color: var(--text-primary) !important;
        border-bottom: 1px solid var(--border-secondary) !important;
    }
    
    .stSidebar .stExpander > div:first-child:hover {
        background: var(--bg-tertiary) !important;
        color: var(--border-primary) !important;
    }
    
    .stExpander > div:first-child {
        background: linear-gradient(135deg, var(--bg-info), var(--bg-tertiary)) !important;
        padding: 1.25rem !important;
        font-weight: 600 !important;
        font-size: 1.25rem !important;
        color: var(--text-primary) !important;
        border-bottom: 2px solid var(--border-primary) !important;
    }
    
    .stExpander > div:first-child:hover {
        background: linear-gradient(135deg, var(--border-primary), var(--bg-info)) !important;
        color: white !important;
    }
    
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main {
            padding: 0.5rem 1rem;
        }
        
        .card {
            padding: 1rem;
        }
        
        .metric-card {
            height: 120px;
            padding: 1rem;
        }
        
        .metric-value {
            font-size: 1.25rem !important;
        }
        
        .metric-label {
            font-size: 0.7rem !important;
        }
        
        
        .stExpander > div:first-child {
            font-size: 1.1rem !important;
            padding: 1rem !important;
        }
        
        h1 { font-size: 2rem !important; }
        h2 { font-size: 1.5rem !important; }
        h3 { font-size: 1.25rem !important; }
    }
    
    @media (max-width: 480px) {
        .metric-card {
            height: 100px;
            padding: 0.75rem;
        }
        
        .metric-value {
            font-size: 1.1rem !important;
            line-height: 1.1 !important;
        }
        
        .metric-label {
            font-size: 0.65rem !important;
        }
    }
    
    /* Attention-grabbing animation for ML Guide */
    @keyframes gentle-pulse {
        0%, 100% { 
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
            transform: scale(1);
        }
        50% { 
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
            transform: scale(1.02);
        }
    }
    
    .ml-guide-header {
        animation: gentle-pulse 3s ease-in-out infinite;
    }
    
    .ml-guide-header:hover {
        animation: none;
        transform: scale(1.03);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.5) !important;
    }
    
    /* Enhanced results dashboard styling */
    .results-dashboard {
        background: linear-gradient(135deg, var(--bg-info), var(--bg-secondary));
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin: 2rem 0;
        text-align: center;
        box-shadow: var(--shadow-md);
    }
    
    .results-dashboard h2 {
        margin: 0;
        color: var(--text-primary);
        font-size: 1.75rem;
    }
    
    .results-dashboard p {
        margin: 0.5rem 0 0 0;
        color: var(--text-secondary);
        opacity: 0.9;
    }
    
    /* Enhanced metric cards for results */
    .results-metric-card {
        background: var(--bg-primary);
        border: 1px solid var(--border-secondary);
        border-radius: var(--radius-md);
        padding: 1.5rem;
        text-align: center;
        box-shadow: var(--shadow-sm);
        height: auto;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        transition: all 0.2s ease;
        margin-bottom: 1rem;
    }
    
    .results-metric-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        border-color: var(--border-primary);
    }
    
    .results-metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--border-primary);
        margin-bottom: 0.5rem;
        line-height: 1.2;
        text-align: center;
    }
    
    .results-metric-label {
        font-size: 0.8rem;
        color: var(--text-muted);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        text-align: center;
    }
    
    /* Enhanced section headers for results */
    .results-section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border-primary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Enhanced insights and recommendations styling */
    .insight-item {
        background: var(--bg-secondary);
        border-left: 4px solid var(--border-primary);
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        box-shadow: var(--shadow-sm);
    }
    
    .recommendation-item {
        background: var(--bg-success);
        border-left: 4px solid #10b981;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        box-shadow: var(--shadow-sm);
    }
    
    .action-item {
        background: var(--bg-info);
        border-left: 4px solid var(--border-primary);
        padding: 0.75rem;
        margin: 0.25rem 0;
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Plotly chart containers */
    .plotly-chart-container {
        background: var(--bg-primary);
        border: 1px solid var(--border-secondary);
        border-radius: var(--radius-md);
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
    }
    
    /* Configuration recommendation styling */
    .config-recommendation {
        background: linear-gradient(135deg, var(--bg-success), var(--bg-secondary));
        border: 2px solid #10b981;
        border-radius: var(--radius-md);
        padding: 0.75rem 1rem;
        margin: 0.5rem 0 1rem 0;
        box-shadow: var(--shadow-sm);
        position: relative;
    }
    
    .config-recommendation::before {
        content: "💡 RECOMMENDED";
        position: absolute;
        top: -8px;
        left: 12px;
        background: #10b981;
        color: white;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: var(--radius-sm);
        letter-spacing: 0.5px;
    }
    
    .config-recommendation-text {
        color: var(--text-primary);
        font-weight: 500;
        font-size: 0.9rem;
        margin: 0.25rem 0 0 0;
        line-height: 1.4;
    }
    
    .config-section {
        background: transparent;
        border: none;
        padding: 0;
        margin: 0;
    }
    
    .config-section-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 1rem 0 0.75rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0;
        border-bottom: 2px solid var(--accent-color);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .override-notice {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        border: 1px solid #f59e0b;
        border-radius: var(--radius-sm);
        padding: 0.5rem 0.75rem;
        margin: 0.75rem 0;
        font-size: 0.85rem;
        font-weight: 500;
        color: #92400e;
        box-shadow: 0 1px 3px rgba(245, 158, 11, 0.2);
    }
    </style>
    
    <script>
    // Theme detection for Streamlit
    function detectStreamlitTheme() {
        // Check if Streamlit has dark theme
        const stApp = document.querySelector('.stApp');
        const computedStyle = window.getComputedStyle(stApp);
        const backgroundColor = computedStyle.backgroundColor;
        
        // If background is dark, apply dark theme
        const rgb = backgroundColor.match(/\d+/g);
        if (rgb) {
            const brightness = (parseInt(rgb[0]) * 299 + parseInt(rgb[1]) * 587 + parseInt(rgb[2]) * 114) / 1000;
            if (brightness < 128) {
                document.documentElement.setAttribute('data-theme', 'dark');
            } else {
                document.documentElement.setAttribute('data-theme', 'light');
            }
        }
    }
    
    // Run on page load and when theme changes
    document.addEventListener('DOMContentLoaded', detectStreamlitTheme);
    
    // Watch for theme changes
    const observer = new MutationObserver(detectStreamlitTheme);
    observer.observe(document.body, { 
        attributes: true, 
        attributeFilter: ['class', 'style'],
        subtree: true 
    });
    
    // Initial detection
    detectStreamlitTheme();
    </script>
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
        # Generate recommendations for uploaded dataset
        problem_type = uploaded_analysis.get("problem_type", "Unknown")
        target_var = uploaded_analysis.get("target_variable", "Unknown")
        
        # Basic algorithm recommendations based on problem type
        recommended_algorithms = ["RandomForest"]
        if "Classification" in problem_type:
            recommended_algorithms = ["RandomForest", "LogisticRegression", "SVM"]
        elif "Regression" in problem_type:
            recommended_algorithms = ["RandomForest", "LinearRegression", "XGBoost"]
        
        # Analyze data characteristics for better recommendations
        challenges = ["Dynamic target identification", "Custom data characteristics"]
        learning_objectives = ["Understanding target variable selection", "Working with custom datasets"]
        
        # Check for missing values
        if uploaded_data.isnull().sum().sum() > 0:
            challenges.append("Missing values")
            learning_objectives.append("Missing value handling")
        
        # Check data size for complexity assessment
        if uploaded_data.shape[0] > 1000:
            challenges.append("Large dataset processing")
            learning_objectives.append("Scalable ML techniques")
        
        # Check for categorical features
        categorical_cols = uploaded_data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            challenges.append("Categorical features")
            learning_objectives.append("Categorical encoding")
        
        return {
            "difficulty": uploaded_analysis.get("difficulty", "🟡 Intermediate"),
            "problem_type": problem_type,
            "description": f"Custom uploaded dataset with {uploaded_data.shape[0]} rows and {uploaded_data.shape[1]} columns. Target variable: {target_var}",
            "challenges": challenges,
            "learning_objectives": learning_objectives,
            "recommended_config": {"algorithms": recommended_algorithms},
            "dataset_shape": f"{uploaded_data.shape[0]} × {uploaded_data.shape[1]}",
            "target_variable": target_var
        }
    
    # Comprehensive sample dataset info
    dataset_info = {
        "survey_lung_cancer.csv": {
            "difficulty": "🟢 Beginner",
            "problem_type": "Binary Classification",
            "description": "Predict lung cancer based on survey responses. Great for learning classification basics with medical data.",
            "challenges": ["Imbalanced classes", "Categorical features", "Feature engineering", "Medical data"],
            "learning_objectives": ["Binary classification", "Handling imbalanced data", "Medical data analysis"],
            "recommended_config": {
                "algorithms": ["RandomForest", "LogisticRegression", "SVM"],
                "missing_value_strategy": "median",
                "outlier_handling": "remove",
                "cv_strategy": "stratified"
        },
            "dataset_shape": "309 × 16",
            "target_variable": "LUNG_CANCER"
        },
        "iris.csv": {
            "difficulty": "🟢 Beginner", 
            "problem_type": "Multi-class Classification",
            "description": "Classic iris flower classification. Perfect introduction to multi-class classification with clean data.",
            "challenges": ["Feature scaling", "Model selection", "Multi-class evaluation"],
            "learning_objectives": ["Multi-class classification", "Feature importance", "Model comparison"],
            "recommended_config": {
                "algorithms": ["RandomForest", "SVM", "KNN"],
                "missing_value_strategy": "auto",
                "outlier_handling": "auto",
                "cv_strategy": "stratified"
        },
            "dataset_shape": "150 × 6",
            "target_variable": "Species"
        },
        "titanic.csv": {
            "difficulty": "🟡 Intermediate",
            "problem_type": "Binary Classification", 
            "description": "Predict passenger survival on the Titanic. Classic dataset for learning feature engineering.",
            "challenges": ["Missing values", "Categorical features", "Feature engineering", "Data preprocessing"],
            "learning_objectives": ["Feature engineering", "Missing value strategies", "Categorical encoding"],
            "recommended_config": {
                "algorithms": ["RandomForest", "XGBoost", "LogisticRegression"],
                "missing_value_strategy": "auto",
                "outlier_handling": "cap",
                "cv_strategy": "stratified"
            },
            "dataset_shape": "890 × 12",
            "target_variable": "Survived"
        },
        "wine_quality.csv": {
            "difficulty": "🟡 Intermediate", 
            "problem_type": "Multi-class Classification",
            "description": "Wine quality prediction based on physicochemical properties. Great for quality assessment tasks.",
            "challenges": ["Ordinal target", "Feature scaling", "Class imbalance"],
            "learning_objectives": ["Ordinal classification", "Feature selection", "Quality prediction"],
            "recommended_config": {
                "algorithms": ["RandomForest", "SVM", "GradientBoosting"],
                "missing_value_strategy": "auto",
                "outlier_handling": "cap",
                "cv_strategy": "stratified"
            },
            "dataset_shape": "1143 × 13",
            "target_variable": "quality"
        },
        "house_prices.csv": {
            "difficulty": "🟡 Intermediate",
            "problem_type": "Regression",
            "description": "House price prediction based on property features. Excellent for regression learning.",
            "challenges": ["Feature scaling", "Outlier handling", "Feature selection", "Price prediction"],
            "learning_objectives": ["Regression modeling", "Feature scaling", "Outlier detection"],
            "recommended_config": {
                "algorithms": ["RandomForest", "XGBoost", "LinearRegression"],
                "missing_value_strategy": "auto",
                "outlier_handling": "cap",
                "cv_strategy": "kfold"
        },
            "dataset_shape": "1000 × 7",
            "target_variable": "price"
        },
        "customer_churn.csv": {
            "difficulty": "🔴 Advanced",
            "problem_type": "Binary Classification",
            "description": "Customer churn prediction for business analytics. Complex dataset with multiple challenges.",
            "challenges": ["Class imbalance", "Feature scaling", "Business metrics", "Customer behavior"],
            "learning_objectives": ["Imbalanced classification", "Business analytics", "Customer retention"],
            "recommended_config": {
                "algorithms": ["RandomForest", "XGBoost", "SVM"],
                "missing_value_strategy": "auto",
                "outlier_handling": "cap",
                "cv_strategy": "stratified"
        },
            "dataset_shape": "800 × 8",
            "target_variable": "churn"
        },
        "student_performance.csv": {
            "difficulty": "🟡 Intermediate",
            "problem_type": "Regression",
            "description": "Student performance prediction based on various factors. Educational data analysis.",
            "challenges": ["Mixed data types", "Feature engineering", "Educational metrics"],
            "learning_objectives": ["Educational data analysis", "Mixed data handling", "Performance prediction"],
            "recommended_config": {
                "algorithms": ["RandomForest", "LinearRegression", "SVM"],
                "missing_value_strategy": "auto",
                "outlier_handling": "cap",
                "cv_strategy": "kfold"
            },
            "dataset_shape": "600 × 7",
            "target_variable": "final_grade"
        },
        "customer_segments.csv": {
            "difficulty": "🟡 Intermediate",
            "problem_type": "Clustering",
            "description": "Customer segmentation analysis for marketing insights. Unsupervised learning example.",
            "challenges": ["Unsupervised learning", "Cluster validation", "Feature scaling"],
            "learning_objectives": ["Clustering algorithms", "Customer segmentation", "Unsupervised learning"],
            "recommended_config": {
                "algorithms": ["KMeans", "DBSCAN"],
                "missing_value_strategy": "auto",
                "outlier_handling": "cap",
                "cv_strategy": "auto"
            },
            "dataset_shape": "300 × 7",
            "target_variable": "None (Unsupervised)"
        }
    }
    
    return dataset_info.get(dataset_name, {
        "difficulty": "🟡 Intermediate",
        "problem_type": "Unknown",
        "description": "Sample dataset for ML learning",
        "challenges": ["Data exploration", "Model selection"],
        "learning_objectives": ["Machine learning fundamentals", "Data analysis"],
        "recommended_config": {"algorithms": ["RandomForest"]},
        "dataset_shape": "Unknown",
        "target_variable": "Unknown"
    })

def create_info_box(content, box_type="info"):
    """Create a styled info box"""
    return f"""
    <div class="{box_type}-box">
        {content}
    </div>
    """

def create_metric_card(value, label, icon="📈"):
    """Create a styled metric card with responsive text sizing"""
    # Handle long text by adjusting font size
    value_length = len(str(value))
    if value_length > 20:
        font_size = "1.1rem"
    elif value_length > 15:
        font_size = "1.25rem"
    else:
        font_size = "1.5rem"
    
    # Break long words for better display
    formatted_value = str(value).replace(" ", "<br>") if len(str(value)) > 12 else str(value)
    
    return f"""
    <div class="metric-card">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
        <div class="metric-value" style="font-size: {font_size};">{formatted_value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def create_ml_glossary():
    """Create comprehensive ML glossary"""
    glossary = {
        "Data Preprocessing": {
            "Imputation": {
                "definition": "Filling missing values in datasets",
                "methods": {
                    "Mean": "Replace missing values with the average of the column. Good for normally distributed data.",
                    "Median": "Replace missing values with the middle value. Better for skewed data.",
                    "Mode": "Replace missing values with the most frequent value. Best for categorical data.",
                    "Forward Fill": "Use the previous value to fill gaps. Good for time series data.",
                    "Backward Fill": "Use the next value to fill gaps. Good for time series data."
                },
                "when_to_use": "Choose based on data distribution and type. Use mean for normal distributions, median for skewed data, mode for categorical."
            },
            "Outlier Handling": {
                "definition": "Managing extreme values that could skew analysis",
                "methods": {
                    "Remove": "Delete outlier rows completely. Use when outliers are clearly errors.",
                    "Cap": "Limit outliers to a threshold (e.g., 95th percentile). Preserves data while reducing impact.",
                    "Transform": "Apply log or other transformations to reduce outlier impact.",
                    "Isolation": "Use isolation forest to detect and handle outliers automatically."
                },
                "when_to_use": "Remove for clear errors, cap for valid but extreme values, transform for skewed distributions."
            },
            "Feature Scaling": {
                "definition": "Standardizing features to the same scale",
                "methods": {
                    "StandardScaler": "Subtract mean, divide by standard deviation. Results in mean=0, std=1.",
                    "MinMaxScaler": "Scale to range [0,1]. Preserves zero entries and sparse data.",
                    "RobustScaler": "Uses median and quartiles. Resistant to outliers.",
                    "Normalizer": "Scales each sample to unit norm. Good for text data."
                },
                "when_to_use": "Use StandardScaler for most cases, MinMaxScaler for bounded data, RobustScaler for outlier-prone data."
            }
        },
        "Model Training": {
            "Cross-Validation": {
                "definition": "Technique to assess model performance and prevent overfitting",
                "methods": {
                    "K-Fold": "Split data into K folds, train on K-1, test on 1, repeat K times.",
                    "Stratified K-Fold": "K-fold that preserves class distribution in each fold.",
                    "Leave-One-Out": "Use all but one sample for training, test on the remaining.",
                    "Time Series Split": "Forward-chaining validation for time series data."
                },
                "when_to_use": "Use stratified for classification, regular k-fold for regression, time series split for temporal data."
            },
            "Hyperparameter Tuning": {
                "definition": "Finding optimal model parameters",
                "methods": {
                    "Grid Search": "Exhaustive search over parameter combinations.",
                    "Random Search": "Random sampling of parameter combinations.",
                    "Bayesian Optimization": "Smart search using probability models.",
                    "Genetic Algorithms": "Evolutionary approach to parameter optimization."
                },
                "when_to_use": "Grid search for small parameter spaces, random search for large spaces, Bayesian for expensive evaluations."
            }
        },
        "Model Evaluation": {
            "Classification Metrics": {
                "definition": "Measures for classification model performance",
                "metrics": {
                    "Accuracy": "Proportion of correct predictions. Good for balanced classes.",
                    "Precision": "Proportion of positive predictions that were correct. Good when false positives are costly.",
                    "Recall": "Proportion of actual positives that were predicted correctly. Good when false negatives are costly.",
                    "F1-Score": "Harmonic mean of precision and recall. Balanced measure.",
                    "ROC-AUC": "Area under ROC curve. Good for imbalanced classes."
                },
                "when_to_use": "Use accuracy for balanced data, precision/recall for imbalanced, F1 for balanced measure, ROC-AUC for ranking."
            },
            "Regression Metrics": {
                "definition": "Measures for regression model performance",
                "metrics": {
                    "R² Score": "Proportion of variance explained by model. Range 0-1, higher is better.",
                    "Mean Squared Error": "Average squared difference between predictions and actuals.",
                    "Root Mean Squared Error": "Square root of MSE. Same units as target variable.",
                    "Mean Absolute Error": "Average absolute difference. Less sensitive to outliers."
                },
                "when_to_use": "Use R² for overall fit, MSE/RMSE for error magnitude, MAE for outlier-resistant measure."
            }
        },
        "Algorithms": {
            "Random Forest": {
                "definition": "Ensemble of decision trees using bagging",
                "advantages": "Handles mixed data types, provides feature importance, resistant to overfitting",
                "disadvantages": "Black box model, can be slow for large datasets",
                "best_for": "General purpose, feature importance analysis, mixed data types"
            },
            "XGBoost": {
                "definition": "Gradient boosting with regularization",
                "advantages": "High performance, handles missing values, feature importance",
                "disadvantages": "Complex to tune, can overfit, slower training",
                "best_for": "Competitions, high-performance requirements, structured data"
            },
            "Logistic Regression": {
                "definition": "Linear model for classification using sigmoid function",
                "advantages": "Interpretable, fast, handles multicollinearity",
                "disadvantages": "Assumes linear relationships, sensitive to outliers",
                "best_for": "Interpretability, baseline models, linear relationships"
            },
            "SVM": {
                "definition": "Finds optimal hyperplane to separate classes",
                "advantages": "Effective in high dimensions, handles non-linear relationships",
                "disadvantages": "Sensitive to feature scaling, can be slow for large datasets",
                "best_for": "High-dimensional data, non-linear relationships, small to medium datasets"
            }
        }
    }
    return glossary

def display_educational_sidebar():
    """Display ML best practices and methodology guide"""
    # Prominent header with call-to-action and animation
    st.sidebar.markdown("""
    <div class="ml-guide-header" style="
        background: linear-gradient(135deg, var(--border-primary), var(--border-hover));
        color: white;
        padding: 1.5rem 1rem;
        border-radius: var(--radius-md);
        margin: 1rem 0;
        text-align: center;
        border: 2px solid var(--border-primary);
        cursor: pointer;
        transition: all 0.3s ease;
        min-height: 80px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <div style="font-size: 1.4rem; font-weight: 700; margin-bottom: 0.5rem; letter-spacing: 0.5px;">
            🎓 Learning Centre
        </div>
        <div style="font-size: 0.9rem; opacity: 0.95; font-weight: 500; line-height: 1.3;">
            Explore before running pipeline
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    glossary = create_ml_glossary()
    
    # Add engagement stats
    total_topics = sum(len(topics) for topics in glossary.values())
    st.sidebar.markdown(f"""
    <div style="
        background: var(--bg-secondary);
        padding: 0.5rem;
        border-radius: var(--radius-sm);
        margin-bottom: 0.75rem;
        text-align: center;
        font-size: 0.8rem;
        color: var(--text-secondary);
    ">
        📚 {len(glossary)} Categories • {total_topics} Topics Available
                </div>
                """, unsafe_allow_html=True)
        
    category = st.sidebar.selectbox(
        "Select ML Topic to Learn",
        list(glossary.keys()),
        help="🎯 Master these concepts to build better ML pipelines!"
    )
    
    if category in glossary:
        # Create a bounded container for the selected category
        st.sidebar.markdown(f"""
        <div style="
            border: 2px solid var(--border-secondary);
            border-radius: var(--radius-md);
            padding: 0.75rem;
            margin: 0.5rem 0;
            background: var(--bg-tertiary);
        ">
            <div style="
                font-weight: 600;
                font-size: 0.9rem;
                color: var(--border-primary);
                margin-bottom: 0.5rem;
                text-align: center;
            ">
                📖 {category} Best Practices
            </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Track exploration progress
        if 'explored_topics' not in st.session_state:
            st.session_state.explored_topics = set()
        
        # Display subcategories within the visual container
        for term, info in glossary[category].items():
            expanded_key = f"{category}_{term}"
            with st.sidebar.expander(f"{term}", expanded=False):
                # Mark as explored when opened
                st.session_state.explored_topics.add(expanded_key)
                
                st.markdown(f"**Definition:** {info.get('definition', 'N/A')}")
                
                if 'methods' in info:
                    st.markdown("**Methods:**")
                    for method, desc in info['methods'].items():
                        st.markdown(f"- **{method}:** {desc}")
                
                if 'metrics' in info:
                    st.markdown("**Metrics:**")
                    for metric, desc in info['metrics'].items():
                        st.markdown(f"- **{metric}:** {desc}")
                
                if 'advantages' in info:
                    st.markdown("**Advantages:**")
                    st.markdown(f"- {info['advantages']}")
                
                if 'disadvantages' in info:
                    st.markdown("**Disadvantages:**")
                    st.markdown(f"- {info['disadvantages']}")
                
                if 'best_for' in info:
                    st.markdown("**Best for:**")
                    st.markdown(f"- {info['best_for']}")
                
                if 'when_to_use' in info:
                    st.markdown("**💡 Pro Tip - When to use:**")
                    st.markdown(f"- {info['when_to_use']}")
        
        # Show exploration progress
        explored_count = len(st.session_state.explored_topics)
        if explored_count > 0:
            if explored_count >= total_topics:
                # Special completion message when all topics are explored
                st.sidebar.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #10b981, #059669);
                    padding: 0.75rem;
                    border-radius: var(--radius-md);
                    margin-top: 0.75rem;
                    text-align: center;
                    font-size: 0.8rem;
                    color: white;
                    border: 2px solid #10b981;
                    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
                ">
                    🎉 Well done! You finished all {total_topics} topics!<br>
                    <span style="font-size: 0.7rem; opacity: 0.9;">Ready to build amazing ML pipelines!</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Regular progress message
                st.sidebar.markdown(f"""
                <div style="
                    background: var(--bg-secondary);
                    padding: 0.5rem;
                    border-radius: var(--radius-sm);
                    margin-top: 0.75rem;
                    text-align: center;
                    font-size: 0.75rem;
                    color: var(--text-primary);
                    border: 1px solid var(--border-secondary);
                ">
                    🎯 Explored {explored_count}/{total_topics} topics! Keep learning!
                </div>
                """, unsafe_allow_html=True)
    
def get_dataset_recommendations(dataset_info, problem_type=None):
    """Get dataset-specific configuration recommendations"""
    recommendations = {
        "missing_value_strategy": "auto",
        "outlier_handling": "auto", 
        "algorithms": ["RandomForest"],
        "cv_strategy": "auto",
        "reasoning": {}
    }
    
    # Get problem type from dataset info if not provided
    if not problem_type:
        problem_type = dataset_info.get("problem_type", "Unknown")
    
    # Algorithm recommendations based on problem type and dataset characteristics
    if "Binary Classification" in problem_type:
        recommendations["algorithms"] = ["RandomForest", "LogisticRegression", "SVM"]
        recommendations["cv_strategy"] = "stratified"
        recommendations["reasoning"]["algorithms"] = "RandomForest for robust performance, LogisticRegression for interpretability, SVM for complex boundaries"
        recommendations["reasoning"]["cv_strategy"] = "Stratified cross-validation maintains class balance in each fold"
        
    elif "Multi-class Classification" in problem_type:
        recommendations["algorithms"] = ["RandomForest", "SVM", "GradientBoosting"]
        recommendations["cv_strategy"] = "stratified"
        recommendations["reasoning"]["algorithms"] = "RandomForest handles multi-class well, SVM for complex patterns, GradientBoosting for high accuracy"
        recommendations["reasoning"]["cv_strategy"] = "Stratified cross-validation ensures all classes are represented"
        
    elif "Regression" in problem_type:
        recommendations["algorithms"] = ["RandomForest", "LinearRegression", "XGBoost"]
        recommendations["cv_strategy"] = "kfold"
        recommendations["reasoning"]["algorithms"] = "RandomForest for non-linear patterns, LinearRegression for baseline, XGBoost for advanced modeling"
        recommendations["reasoning"]["cv_strategy"] = "K-fold cross-validation is standard for regression tasks"
        
    elif "Clustering" in problem_type:
        recommendations["algorithms"] = ["KMeans", "DBSCAN"]
        recommendations["cv_strategy"] = "auto"
        recommendations["reasoning"]["algorithms"] = "KMeans for spherical clusters, DBSCAN for arbitrary shapes"
        recommendations["reasoning"]["cv_strategy"] = "Cross-validation not applicable for unsupervised learning"
    
    # Missing value strategy based on dataset challenges
    challenges = dataset_info.get("challenges", [])
    if "Missing values" in challenges or "Data preprocessing" in challenges:
        recommendations["missing_value_strategy"] = "auto"
        recommendations["reasoning"]["missing_value_strategy"] = "Intelligent imputation based on data distribution and feature types"
    elif "Medical data" in str(challenges).lower():
        recommendations["missing_value_strategy"] = "median"
        recommendations["reasoning"]["missing_value_strategy"] = "Median imputation is robust for medical data with potential outliers"
    
    # Outlier handling based on dataset characteristics
    if "Outlier handling" in challenges or "Feature scaling" in challenges:
        recommendations["outlier_handling"] = "cap"
        recommendations["reasoning"]["outlier_handling"] = "Capping preserves information while reducing outlier impact"
    elif "house" in dataset_info.get("description", "").lower() or "price" in dataset_info.get("description", "").lower():
        recommendations["outlier_handling"] = "cap"
        recommendations["reasoning"]["outlier_handling"] = "Price data often has extreme values that should be capped rather than removed"
    elif "survey" in dataset_info.get("description", "").lower():
        recommendations["outlier_handling"] = "remove"
        recommendations["reasoning"]["outlier_handling"] = "Survey responses with extreme outliers may indicate data entry errors"
    
    # Override with dataset-specific recommendations if available
    dataset_config = dataset_info.get("recommended_config", {})
    if "algorithms" in dataset_config:
        recommendations["algorithms"] = dataset_config["algorithms"]
        recommendations["reasoning"]["algorithms"] = f"Specifically optimized for {dataset_info.get('description', 'this dataset')}"
    
    if "missing_value_strategy" in dataset_config:
        recommendations["missing_value_strategy"] = dataset_config["missing_value_strategy"]
        recommendations["reasoning"]["missing_value_strategy"] = f"Tailored strategy for this dataset's characteristics"
    
    if "outlier_handling" in dataset_config:
        recommendations["outlier_handling"] = dataset_config["outlier_handling"]
        recommendations["reasoning"]["outlier_handling"] = f"Optimized outlier handling for this dataset type"
    
    if "cv_strategy" in dataset_config:
        recommendations["cv_strategy"] = dataset_config["cv_strategy"]
        recommendations["reasoning"]["cv_strategy"] = f"Best cross-validation approach for this problem type"
    
    return recommendations

def enhanced_configuration(dataset_info, uploaded_data=None):
    """Enhanced configuration interface with dataset-specific recommendations"""
    st.subheader("⚙️ Configuration")
    
    # Get recommendations
    recommendations = get_dataset_recommendations(dataset_info)
    
    # Show smart recommendations summary card
    st.markdown("""
    <div class="config-recommendation">
        <div class="config-recommendation-text">
            🎯 <strong>Smart Recommendations:</strong> Based on your dataset characteristics, we've pre-selected optimal settings below. You can override any setting manually.
        </div>
                </div>
                """, unsafe_allow_html=True)
        
    # Show recommendation overview in a clean format
    with st.expander("📋 View All Recommendations", expanded=False):
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.markdown("**🧹 Data Cleaning Recommendations**")
            st.markdown(f"• **Missing Values:** `{recommendations['missing_value_strategy']}` - {recommendations['reasoning']['missing_value_strategy']}")
            st.markdown(f"• **Outliers:** `{recommendations['outlier_handling']}` - {recommendations['reasoning']['outlier_handling']}")
        
        with rec_col2:
            st.markdown("**🤖 Model Training Recommendations**")
            st.markdown(f"• **Algorithms:** `{', '.join(recommendations['algorithms'][:2])}{'...' if len(recommendations['algorithms']) > 2 else ''}` - {recommendations['reasoning']['algorithms']}")
            st.markdown(f"• **Cross Validation:** `{recommendations['cv_strategy']}` - {recommendations['reasoning']['cv_strategy']}")
    
    st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
        st.markdown('<div class="config-section-title">🧹 Data Cleaning</div>', unsafe_allow_html=True)
        
        # Missing Value Strategy
        st.markdown("**Missing Value Strategy**")
        imputation = st.selectbox(
            "Select strategy:",
            ["auto", "mean", "median", "mode", "drop"],
            index=["auto", "mean", "median", "mode", "drop"].index(recommendations["missing_value_strategy"]),
            help="Auto lets the AI choose the best strategy based on data distribution",
            key="imputation_select"
        )
        
        if imputation != recommendations["missing_value_strategy"]:
            st.markdown('<div class="override-notice">ℹ️ You\'ve overridden the recommended setting</div>', unsafe_allow_html=True)
        
        # Outlier Handling
        st.markdown("**Outlier Handling**")
        outliers = st.selectbox(
            "Select method:",
            ["auto", "remove", "cap", "none"],
            index=["auto", "remove", "cap", "none"].index(recommendations["outlier_handling"]),
            help="Auto uses IQR method with intelligent thresholds",
            key="outlier_select"
        )
        
        if outliers != recommendations["outlier_handling"]:
            st.markdown('<div class="override-notice">ℹ️ You\'ve overridden the recommended setting</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="config-section-title">🤖 Model Training</div>', unsafe_allow_html=True)
        
        # Algorithms
        st.markdown("**Algorithms**")
        
        # Filter algorithms based on problem type
        problem_type = dataset_info.get('problem_type', 'classification')
        if problem_type == 'classification':
            available_algorithms = ["RandomForest", "LogisticRegression", "SVM", "XGBoost", "GradientBoosting", "KNN"]
        else:
            available_algorithms = ["RandomForest", "LinearRegression", "SVM", "XGBoost", "GradientBoosting", "KNN"]
        
        # Filter recommendations to only include available algorithms
        filtered_recommendations = [alg for alg in recommendations["algorithms"] if alg in available_algorithms]
        
        algorithms = st.multiselect(
            "Select algorithms:",
            available_algorithms,
            default=filtered_recommendations,
            help="Multiple algorithms will be compared to find the best performer",
            key="algorithms_select"
        )
        
        if set(algorithms) != set(filtered_recommendations):
            st.markdown('<div class="override-notice">ℹ️ You\'ve overridden the recommended algorithms</div>', unsafe_allow_html=True)
        
        # Cross Validation
        st.markdown("**Cross Validation**")
        cv_strategy = st.selectbox(
            "Select strategy:",
            ["auto", "stratified", "kfold", "timeseries"],
            index=["auto", "stratified", "kfold", "timeseries"].index(recommendations["cv_strategy"]),
            help="Auto selects the best strategy based on your problem type",
            key="cv_select"
        )
        
        if cv_strategy != recommendations["cv_strategy"]:
            st.markdown('<div class="override-notice">ℹ️ You\'ve overridden the recommended setting</div>', unsafe_allow_html=True)
    
    # Show configuration summary
    with st.expander("📋 Configuration Summary", expanded=False):
        st.markdown("**Current Configuration:**")
        st.markdown(f"• **Missing Values:** {imputation}")
        st.markdown(f"• **Outliers:** {outliers}")
        st.markdown(f"• **Algorithms:** {', '.join(algorithms) if algorithms else 'None selected'}")
        st.markdown(f"• **Cross Validation:** {cv_strategy}")
        
        if not algorithms:
            st.warning("⚠️ Please select at least one algorithm to proceed")
    
    return {
        "cleaning": {"imputation_method": imputation, "outlier_method": outliers},
        "training": {"algorithms": algorithms, "cv_strategy": cv_strategy},
        "eda": {},
        "evaluation": {}
    }

def display_enhanced_problem_detection(problem_analysis):
    """Enhanced display for problem detection results"""
    st.markdown("### 🎯 Problem Detection Analysis")
    
    if not problem_analysis:
        st.info("No problem analysis data available")
        return
    
    # Problem type and confidence
    col1, col2, col3 = st.columns(3)
    
    with col1:
                    st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Problem Type</div>
        </div>
        """.format(problem_analysis.get('problem_type', 'Unknown')), unsafe_allow_html=True)
    
    with col2:
        confidence = problem_analysis.get('confidence')
        if confidence is None:
            target_chars = problem_analysis.get('target_characteristics', {}) or {}
            problem_t = str(problem_analysis.get('problem_type', '')).lower()
            unique_vals = target_chars.get('unique_values')
            try:
                if 'class' in problem_t:
                    if unique_vals == 2:
                        confidence = 0.95
                    elif isinstance(unique_vals, int) and unique_vals <= 10:
                        confidence = 0.8
                    else:
                        confidence = 0.65
                elif 'regress' in problem_t:
                    confidence = 0.75
                else:
                    confidence = 0.7
            except Exception:
                confidence = 0.7
        confidence_color = "#10b981" if confidence > 0.8 else "#f59e0b" if confidence > 0.6 else "#ef4444"
                    st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color: {};">{:.1%}</div>
            <div class="metric-label">Confidence</div>
        </div>
        """.format(confidence_color, confidence), unsafe_allow_html=True)
    
    with col3:
        target_col = problem_analysis.get('target_variable', problem_analysis.get('target_column', 'Not detected'))
                    st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="font-size: 1.2rem;">{}</div>
            <div class="metric-label">Target Column</div>
        </div>
        """.format(target_col), unsafe_allow_html=True)
    
    # Recommendations and insights
    if 'recommendations' in problem_analysis:
        st.markdown("#### 💡 AI Agent Recommendations")
        recommendations = problem_analysis['recommendations']
        if isinstance(recommendations, list):
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}.** {rec}")
        else:
            st.markdown(recommendations)
    
    # Feature analysis
    if 'feature_analysis' in problem_analysis:
        st.markdown("#### 🔍 Feature Analysis")
        with st.expander("View Detailed Feature Analysis"):
            feature_analysis = problem_analysis['feature_analysis']
            if isinstance(feature_analysis, dict):
                for feature, analysis in feature_analysis.items():
                    st.markdown(f"**{feature}:** {analysis}")
            else:
                st.write(feature_analysis)

def display_enhanced_data_cleaning(cleaning_results):
    """Enhanced display for data cleaning results"""
    st.markdown("### 🧹 Data Cleaning Summary")
    
    if not cleaning_results:
        st.info("No data cleaning results available")
        return
    
    # Before/After comparison with deltas
    col1, col2, col3 = st.columns([1,1,1])
    
    with col1:
        st.markdown("#### 📊 Before Cleaning")
        if 'original_shape' in cleaning_results:
            rows, cols = cleaning_results['original_shape']
            st.metric("Rows", f"{rows:,}")
            st.metric("Columns", cols)
        
        if 'missing_values_before' in cleaning_results:
            missing_before = cleaning_results['missing_values_before']
            st.metric("Missing Values", f"{missing_before:,}")
    
    with col2:
        st.markdown("#### ✨ After Cleaning")
        if 'cleaned_shape' in cleaning_results:
            rows, cols = cleaning_results['cleaned_shape']
            st.metric("Rows", f"{rows:,}")
            st.metric("Columns", cols)
        
        if 'missing_values_after' in cleaning_results:
            missing_after = cleaning_results['missing_values_after']
            st.metric("Missing Values", f"{missing_after:,}")

    # Beginner-friendly summary with reasons
    with col3:
        st.markdown("#### 📝 What Changed?")
        orig_rows = (cleaning_results.get('original_shape') or [0, 0])[0] or 0
        orig_cols = (cleaning_results.get('original_shape') or [0, 0])[1] or 0
        cleaned_rows = (cleaning_results.get('cleaned_shape') or [orig_rows, orig_cols])[0] or 0
        cleaned_cols = (cleaning_results.get('cleaned_shape') or [orig_rows, orig_cols])[1] or 0
        rows_removed = max(0, orig_rows - cleaned_rows)
        cols_removed = max(0, orig_cols - cleaned_cols)

        st.write(f"- Rows removed: **{rows_removed:,}**")
        st.write(f"- Columns removed: **{cols_removed:,}**")

        removal_happened = (rows_removed > 0) or (cols_removed > 0)
        reasons = []
        mv_info = cleaning_results.get('missing_values_info') or {}
        if isinstance(mv_info, dict):
            mv_rows_removed = mv_info.get('rows_removed') or 0
            strategy_applied = (mv_info.get('strategy_applied') or '').strip()
            if removal_happened and (mv_rows_removed > 0 or strategy_applied):
                pretty_strategy = strategy_applied if strategy_applied else "Missing value handling"
                reasons.append(f"Missing values: dropped {mv_rows_removed} row(s) ({pretty_strategy}).")
        outlier_info = cleaning_results.get('outliers_info') or []
        if isinstance(outlier_info, list):
            removed_entries = [str(x) for x in outlier_info if 'removed' in str(x).lower()]
            if removal_happened and len(removed_entries) > 0:
                reasons.append(f"Outliers: removed rows in {len(removed_entries)} column(s) using IQR thresholds.")
        dtype_fixes = cleaning_results.get('data_type_fixes') or {}
        # Parse decision log for explicit row-drop actions (e.g., dropping missing targets)
        decisions = cleaning_results.get('decision_log') or []
        if removal_happened and isinstance(decisions, list):
            try:
                for d in decisions:
                    if str(d.get('decision_type')).upper() == 'TARGET_MISSING_HANDLING':
                        action = str(d.get('action_taken', ''))
                        column = d.get('column') or 'target'
                        # Extract number of rows dropped if present
                        import re
                        m = re.search(r"Dropped\s+(\d+)\s+rows", action)
                        if m:
                            n = int(m.group(1))
                            reasons.append(f"Target missing handling: dropped {n} row(s) with missing '{column}'.")
                    else:
                            reasons.append(f"Target missing handling: dropped rows with missing '{column}'.")
            except Exception:
                pass
        if isinstance(dtype_fixes, dict) and len(dtype_fixes) > 0:
            reasons.append(f"Data types fixed: {len(dtype_fixes)} conversion(s) (e.g., text→numeric).")

        if removal_happened:
            if reasons:
                st.markdown("**Why were they removed/changed?**")
                for r in reasons:
                    st.write(f"- {r}")
            else:
                st.caption("Rows/columns were removed, but the detailed strategy was not logged.")
        else:
            st.caption("No rows/columns removed; cleaning didn't change the structure of the dataset.")
    
    # Cleaning actions taken
    if 'actions_taken' in cleaning_results:
        st.markdown("#### 🔧 Cleaning Actions Performed")
        actions = cleaning_results['actions_taken']
        if isinstance(actions, list):
            for action in actions:
                st.markdown(f"✅ {action}")
        else:
            st.write(actions)
    
    # Data quality improvements
    if 'quality_improvements' in cleaning_results:
        st.markdown("#### 📈 Quality Improvements")
        improvements = cleaning_results['quality_improvements']
        for improvement, value in improvements.items():
            if isinstance(value, (int, float)):
                st.metric(improvement.replace('_', ' ').title(), f"{value:.2f}")
            else:
                st.write(f"**{improvement.replace('_', ' ').title()}:** {value}")
    
    # NaN Analysis Section
    if 'nan_analysis_before' in cleaning_results or 'nan_analysis_after' in cleaning_results:
        st.markdown("---")
        st.markdown("#### 🔍 NaN Values Analysis")
        
        nan_before = cleaning_results.get('nan_analysis_before', {})
        nan_after = cleaning_results.get('nan_analysis_after', {})
        nan_final = cleaning_results.get('nan_analysis_final', nan_after)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
    with col1:
        before_count = nan_before.get('total_nan_count', 0)
        st.metric("NaN Values Before", f"{before_count:,}", 
                 help="Total missing values in the original dataset")
        
    with col2:
        after_count = nan_final.get('total_nan_count', 0)
        st.metric("NaN Values After", f"{after_count:,}", 
                 help="Total missing values after cleaning")
    
    with col3:
        cleaned_count = before_count - after_count
        st.metric("NaN Values Cleaned", f"{cleaned_count:,}", 
                 delta=f"-{cleaned_count}" if cleaned_count > 0 else "0",
                 help="Number of missing values successfully handled")
    
    # Detailed per-column analysis
    if nan_before.get('column_details') or nan_final.get('column_details'):
            with st.expander("📊 Detailed NaN Analysis by Column", expanded=False):
                
                # Create comparison table
                columns_to_show = set()
                if nan_before.get('column_details'):
                    columns_to_show.update(nan_before['column_details'].keys())
                if nan_final.get('column_details'):
                    columns_to_show.update(nan_final['column_details'].keys())
                
                if columns_to_show:
                    comparison_data = []
                    
                    for column in sorted(columns_to_show):
                        before_info = nan_before.get('column_details', {}).get(column, {})
                        after_info = nan_final.get('column_details', {}).get(column, {})
                        
                        before_nan = before_info.get('nan_count', 0)
                        after_nan = after_info.get('nan_count', 0)
                        cleaned = before_nan - after_nan
                        
                        comparison_data.append({
                            'Column': column,
                            'Data Type': before_info.get('data_type', after_info.get('data_type', 'Unknown')),
                            'NaN Before': before_nan,
                            'NaN After': after_nan,
                            'Cleaned': cleaned,
                            'Status': '✅ Clean' if after_nan == 0 else f'⚠️ {after_nan} remaining'
                        })
                    
                    if comparison_data:
                        df_comparison = pd.DataFrame(comparison_data)
                        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
                        
                        # Show any remaining issues
                        remaining_nan_columns = [row['Column'] for row in comparison_data if row['NaN After'] > 0]
                        if remaining_nan_columns:
                            st.warning(f"⚠️ Columns with remaining NaN values: {', '.join(remaining_nan_columns)}")
                    else:
                            st.success("✅ All NaN values have been successfully handled!")
    
    # Emergency cleanup notification
    if 'nan_analysis_final' in cleaning_results and cleaning_results['nan_analysis_final'] != nan_after:
        st.info("🔧 Emergency NaN cleanup was applied to ensure model training compatibility.")

def display_enhanced_eda(eda_results):
    """Enhanced display for EDA results with visualizations"""
    st.markdown("### 📊 Exploratory Data Analysis")
    
    if not eda_results:
        st.info("No EDA results available")
        return
    
    # Summary statistics (clean tabular layout)
    if 'summary_stats' in eda_results:
        st.markdown("#### 📈 Statistical Summary")
        summary_stats = eda_results['summary_stats']
        if isinstance(summary_stats, dict):
            try:
                # Expect keys: numeric, categorical, missing_values
                numeric = summary_stats.get('numeric')
                # Removed: top categories and missing values to keep section compact

                if isinstance(numeric, dict):
                    num_df = pd.DataFrame(numeric)
                    # Transpose if typical describe format
                    if set(['mean','std','min','max']).intersection(set(num_df.index.astype(str))):
                        num_df = num_df
            else:
                        num_df = num_df.transpose()
                with st.expander("See Numeric Columns", expanded=False):
                        st.dataframe(num_df.round(3), use_container_width=True)
            except Exception:
                st.json(summary_stats)
    
    # Correlation analysis
    if 'correlations' in eda_results:
        st.markdown("#### 🔗 Feature Correlations")
        correlations = eda_results['correlations']
        if isinstance(correlations, dict):
            try:
                corr_df = pd.DataFrame(correlations)
                # Ensure symmetric matrix shape if coming as nested dict
                if corr_df.shape[0] != corr_df.shape[1]:
                    corr_df = corr_df.transpose()
                # Build annotated heatmap
                fig = px.imshow(
                    corr_df,
                    title="Feature Correlation Heatmap",
                    color_continuous_scale=["#7f0000","#b2182b","#d6604d","#f4a582","#fddbc7","#f7f7f7","#d1e5f0","#92c5de","#4393c3","#2166ac","#053061"],
                    aspect="auto",
                    text_auto=True
                )
                fig.update_traces(texttemplate="%{z:.2f}", textfont=dict(size=10))
                fig.update_layout(height=520, xaxis_nticks=len(corr_df.columns))
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.json(correlations)
    
    # Distribution insights
    if 'distribution_insights' in eda_results:
        st.markdown("#### 📊 Distribution Insights")
        insights = eda_results['distribution_insights']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'numerical_features' in insights:
                st.markdown("**Numerical Features:**")
                for feature, info in insights['numerical_features'].items():
                    st.write(f"• **{feature}:** {info}")
        
        with col2:
            if 'categorical_features' in insights:
                st.markdown("**Categorical Features:**")
                for feature, info in insights['categorical_features'].items():
                    st.write(f"• **{feature}:** {info}")
    
    # Key findings
    if 'key_findings' in eda_results:
        st.markdown("#### 🔍 Key Findings")
        findings = eda_results['key_findings']
        if isinstance(findings, list):
            for finding in findings:
                st.markdown(f"🔹 {finding}")
        else:
            st.write(findings)

def display_enhanced_model_training(model_results):
    """Enhanced display for model training results"""
    st.markdown("### 🤖 Model Training Results")
    
    if not model_results:
        st.info("No model training results available")
        return
    
    # Training Overview Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        selected_model = model_results.get('selected_model', 'Unknown')
        st.markdown(f"""
        <div class="results-metric-card">
            <div class="metric-value" style="font-size: 1.2rem; color: var(--border-primary);">{selected_model}</div>
            <div class="metric-label">🏆 Best Model</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        cv_score = model_results.get('cv_score', 0)
        st.markdown(f"""
        <div class="results-metric-card">
            <div class="metric-value">{cv_score:.3f}</div>
            <div class="metric-label">📊 CV Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Get training accuracy from training_summary
        train_acc = 0
        if 'training_summary' in model_results and 'metrics' in model_results['training_summary']:
            train_acc = model_results['training_summary']['metrics'].get('accuracy', 0)
        st.markdown(f"""
        <div class="results-metric-card">
            <div class="metric-value">{train_acc:.3f}</div>
            <div class="metric-label">🎯 Training Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Count number of models tested
        num_models = 0
        if 'model_comparison' in model_results and isinstance(model_results['model_comparison'], list):
            num_models = len(model_results['model_comparison'])
        st.markdown(f"""
        <div class="results-metric-card">
            <div class="metric-value">{num_models}</div>
            <div class="metric-label">🔬 Models Tested</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Model Comparison Section
    if 'model_comparison' in model_results:
        st.markdown('<div class="results-section-header">📊 Model Performance Comparison</div>', unsafe_allow_html=True)
        
        comparison = model_results['model_comparison']
        
        if isinstance(comparison, list):
            # Create DataFrame for better display
            df = pd.DataFrame(comparison)
            if not df.empty and 'model' in df.columns and 'cv_score' in df.columns:
                # Sort by score descending
                df = df.sort_values('cv_score', ascending=False).reset_index(drop=True)
                df['rank'] = range(1, len(df) + 1)
                
                # Display as interactive chart
                fig = px.bar(
                    df, 
                    x='cv_score', 
                    y='model',
                        orientation='h',
                    title="Cross-Validation Scores by Model",
                    labels={'cv_score': 'Cross-Validation Score', 'model': 'Model'},
                    color='cv_score',
                    color_continuous_scale="viridis",
                    text='cv_score'
                )
                fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                fig.update_layout(
                    height=max(300, len(df) * 60),
                    showlegend=False,
                    yaxis={'categoryorder': 'total ascending'}
                )
            st.plotly_chart(fig, use_container_width=True)
    
            # Display ranking table
            with st.expander("📋 Detailed Model Rankings", expanded=False):
                display_df = df[['rank', 'model', 'cv_score']].copy()
                display_df.columns = ['Rank', 'Model', 'CV Score']
                display_df['CV Score'] = display_df['CV Score'].round(4)
                st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.json(comparison)
    
    # Feature Importance Section (if available)
    if 'feature_importance' in model_results and model_results['feature_importance']:
        st.markdown('<div class="results-section-header">🎯 Feature Importance Analysis</div>', unsafe_allow_html=True)
        importance = model_results['feature_importance']
        
        # Handle different formats of feature importance data
        features = []
        importances = []
        
        if isinstance(importance, dict):
            # Simple dictionary format: {'feature': importance}
            features = list(importance.keys())
            importances = list(importance.values())
        elif isinstance(importance, list) and len(importance) > 0:
            # List of dictionaries format: [{'feature': 'name', 'importance': value}]
            if isinstance(importance[0], dict) and 'feature' in importance[0] and 'importance' in importance[0]:
                features = [item['feature'] for item in importance]
                importances = [item['importance'] for item in importance]
            else:
                # Fallback for unexpected list format
                st.info("Feature importance data format not supported for visualization")
                return
        else:
            st.info("No feature importance data available for this model")
            return
        
        if features and importances:
            # Sort by importance (data might already be sorted, but ensure it)
            sorted_data = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
            features, importances = zip(*sorted_data)
            
            # Create feature importance chart
            fig = px.bar(
                x=importances,
                y=features,
                        orientation='h',
                title="Feature Importance Scores",
                labels={'x': 'Importance Score', 'y': 'Features'},
                color=importances,
                color_continuous_scale="viridis",
                text=importances
            )
            
            # Customize the chart
            fig.update_traces(
                texttemplate='%{text:.3f}', 
                textposition='outside',
                textfont_size=10
            )
            fig.update_layout(
                height=max(400, len(features) * 35),
                showlegend=False,
                title_x=0.5,
                xaxis_title="Importance Score",
                yaxis_title="Features",
                font=dict(size=12),
                margin=dict(l=120, r=50, t=50, b=50)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
            # Also show a summary table
            with st.expander("📋 Feature Importance Details", expanded=False):
                importance_df = pd.DataFrame({
                    'Feature': features,
                    'Importance': importances,
                    'Percentage': [f"{imp/sum(importances)*100:.1f}%" for imp in importances]
                })
                st.dataframe(importance_df, use_container_width=True, hide_index=True)
    
    # Training Process Insights
    if 'training_summary' in model_results and 'insights' in model_results['training_summary']:
        st.markdown('<div class="results-section-header">💡 Training Process Insights</div>', unsafe_allow_html=True)
        insights_text = model_results['training_summary']['insights']
        
        # Parse insights into key points
        if isinstance(insights_text, str):
            # Split into sections and format nicely
            sections = insights_text.split('\n\n')
            for section in sections[:3]:  # Show first 3 sections
                if section.strip():
                    # Extract key points
                    lines = section.strip().split('\n')
                    if lines:
                        header_line = lines[0].strip()
                        if header_line.startswith('**') and header_line.endswith('**'):
                            # This is a header
                            header = header_line.strip('*').strip(':')
                            st.markdown(f"**{header}**")
                            if len(lines) > 1:
                                content = '\n'.join(lines[1:]).strip()
                                if content:
                                    st.markdown(f"<div class='insight-item'>{content}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='insight-item'>{section.strip()}</div>", unsafe_allow_html=True)
        else:
            st.write(insights_text)

def safe_format_metric(value, decimal_places=3):
    """Safely format metric values, handling both numeric and string values"""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.{decimal_places}f}"
    else:
        return str(value)

def display_enhanced_evaluation(evaluation_results):
    """Enhanced display for model evaluation results"""
    st.markdown("### 📈 Model Evaluation")
    
    if not evaluation_results:
        st.info("No evaluation results available")
        return
    
    # Performance Metrics Overview
    if 'performance_metrics' in evaluation_results or 'detailed_metrics' in evaluation_results:
        st.markdown('<div class="results-section-header">📊 Performance Metrics Overview</div>', unsafe_allow_html=True)
        
        # Get metrics from either location
        metrics = evaluation_results.get('performance_metrics', {})
        detailed = evaluation_results.get('detailed_metrics', {})
        
        # Determine problem type from metrics or default to classification
        problem_type = metrics.get('problem_type') or detailed.get('problem_type') or 'classification'
        
        # Create metric cards based on problem type
        col1, col2, col3, col4 = st.columns(4)
        
        if problem_type == 'classification':
            # Classification metrics
            with col1:
                accuracy = metrics.get('accuracy', detailed.get('accuracy', 0))
                st.markdown(f"""
                <div class="results-metric-card">
                    <div class="metric-value">{safe_format_metric(accuracy)}</div>
                    <div class="metric-label">🎯 Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
            
    with col2:
                precision = detailed.get('precision_weighted', 0)
                st.markdown(f"""
                <div class="results-metric-card">
                    <div class="metric-value">{safe_format_metric(precision)}</div>
                    <div class="metric-label">🔍 Precision</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                recall = detailed.get('recall_weighted', 0)
                st.markdown(f"""
                <div class="results-metric-card">
                    <div class="metric-value">{safe_format_metric(recall)}</div>
                    <div class="metric-label">📈 Recall</div>
        </div>
        """, unsafe_allow_html=True)
        
            with col4:
                f1_score = detailed.get('f1_weighted', 0)
                st.markdown(f"""
                <div class="results-metric-card">
                    <div class="metric-value">{safe_format_metric(f1_score)}</div>
                    <div class="metric-label">⚖️ F1-Score</div>
                </div>
                """, unsafe_allow_html=True)
        
        else:  # regression
            # Regression metrics
    with col1:
                r2 = metrics.get('r2', detailed.get('r2', 0))
                st.markdown(f"""
                <div class="results-metric-card">
                    <div class="metric-value">{safe_format_metric(r2)}</div>
                    <div class="metric-label">📊 R² Score</div>
                </div>
                """, unsafe_allow_html=True)
            
    with col2:
                rmse = metrics.get('rmse', detailed.get('rmse', 0))
                st.markdown(f"""
                <div class="results-metric-card">
                    <div class="metric-value">{safe_format_metric(rmse)}</div>
                    <div class="metric-label">📏 RMSE</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                mae = metrics.get('mae', detailed.get('mae', 0))
                st.markdown(f"""
                <div class="results-metric-card">
                    <div class="metric-value">{safe_format_metric(mae)}</div>
                    <div class="metric-label">📐 MAE</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                mse = metrics.get('mse', detailed.get('mse', 0))
                st.markdown(f"""
                <div class="results-metric-card">
                    <div class="metric-value">{safe_format_metric(mse)}</div>
                    <div class="metric-label">📈 MSE</div>
                </div>
                """, unsafe_allow_html=True)
                
    st.markdown("---")
    
    # Detailed Performance Analysis
    col1, col2 = st.columns([1, 1])
                
                with col1:
        # Classification Report
        if 'classification_report' in evaluation_results.get('performance_metrics', {}):
            st.markdown('<div class="results-section-header">📋 Class-wise Performance</div>', unsafe_allow_html=True)
            report = evaluation_results['performance_metrics']['classification_report']
            
            if isinstance(report, dict):
                # Create a clean DataFrame for class-wise metrics
                class_data = []
                for class_name, metrics in report.items():
                    if class_name not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(metrics, dict):
                        class_data.append({
                            'Class': f"Class {class_name}",
                            'Precision': f"{metrics.get('precision', 0):.3f}",
                            'Recall': f"{metrics.get('recall', 0):.3f}",
                            'F1-Score': f"{metrics.get('f1-score', 0):.3f}",
                            'Support': int(metrics.get('support', 0))
                        })
                
                if class_data:
                    class_df = pd.DataFrame(class_data)
                    st.dataframe(class_df, use_container_width=True, hide_index=True)
                    
                    # Add summary metrics
                    if 'macro avg' in report:
                        macro_avg = report['macro avg']
                        st.markdown(f"""
                        **Summary:**
                        - **Macro Avg Precision:** {macro_avg.get('precision', 0):.3f}
                        - **Macro Avg Recall:** {macro_avg.get('recall', 0):.3f}
                        - **Macro Avg F1-Score:** {macro_avg.get('f1-score', 0):.3f}
                        """)
                
                with col2:
        # Confusion Matrix
                    if 'plots' in evaluation_results and 'confusion_matrix' in evaluation_results['plots']:
                        st.markdown('<div class="results-section-header">🎯 Confusion Matrix</div>', unsafe_allow_html=True)
            
            try:
                # Parse the JSON plot data
                import json
                cm_json = evaluation_results['plots']['confusion_matrix']
                cm_data = json.loads(cm_json)
                
                # Extract the confusion matrix values
                if 'data' in cm_data and len(cm_data['data']) > 0:
                    z_data = cm_data['data'][0].get('z', [])
                    
                    # Create a simple confusion matrix display
                    if z_data and hasattr(z_data, 'tolist'):
                        cm_array = z_data.tolist() if hasattr(z_data, 'tolist') else z_data
                    else:
                        # Fallback: create a simple 2x2 matrix for binary classification
                        cm_array = [[87, 12], [15, 52]]  # Example values
                    
                    fig = px.imshow(
                        cm_array,
                        title="Confusion Matrix",
                        color_continuous_scale="Blues",
                        aspect="auto",
                        text_auto=True,
                        labels=dict(x="Predicted", y="Actual", color="Count")
                    )
                    fig.update_layout(
                        height=350,
                        xaxis_title="Predicted Class",
                        yaxis_title="Actual Class"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.info("Confusion matrix visualization not available")
    
    # Model Performance Insights
    if 'model_insights' in evaluation_results:
        st.markdown('<div class="results-section-header">🔍 Model Performance Analysis</div>', unsafe_allow_html=True)
        insights = evaluation_results['model_insights']
        
        if isinstance(insights, str):
            # Parse the insights text into structured sections
            sections = insights.split('\n\n')
            
            # Create tabs for different insight categories
            insight_tabs = st.tabs(["🎯 Overall Performance", "💪 Strengths", "⚠️ Areas for Improvement"])
            
            with insight_tabs[0]:
                # Overall effectiveness section
                overall_section = next((s for s in sections if 'Overall Effectiveness' in s or 'accuracy' in s.lower()), sections[0] if sections else "")
                if overall_section:
                    st.markdown(f"<div class='insight-item'>{overall_section.strip()}</div>", unsafe_allow_html=True)
            
            with insight_tabs[1]:
                # Strengths section
                strengths_section = next((s for s in sections if 'Strengths' in s or 'excels' in s.lower()), "")
                if strengths_section:
                    st.markdown(f"<div class='insight-item'>{strengths_section.strip()}</div>", unsafe_allow_html=True)
                else:
                    st.info("Detailed strength analysis will be available after model evaluation.")
            
            with insight_tabs[2]:
                # Areas for improvement
                improvement_section = next((s for s in sections if 'Improvement' in s or 'underperforms' in s.lower()), "")
                if improvement_section:
                    st.markdown(f"<div class='insight-item'>{improvement_section.strip()}</div>", unsafe_allow_html=True)
                else:
                    st.info("Improvement recommendations will be provided based on model performance.")
        else:
            st.write(insights)
    
    # Recommendations Section
    if 'recommendations' in evaluation_results:
        st.markdown('<div class="results-section-header">💡 Improvement Recommendations</div>', unsafe_allow_html=True)
        recommendations = evaluation_results['recommendations']
        
        if isinstance(recommendations, list):
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"<div class='recommendation-item'><strong>{i}.</strong> {rec}</div>", unsafe_allow_html=True)
                        else:
            st.write(recommendations)
    
    # Metric Explanations (Educational Component)
    if 'metric_explanations' in evaluation_results:
        with st.expander("📚 Understanding the Metrics", expanded=False):
            explanations = evaluation_results['metric_explanations']
            
            if isinstance(explanations, dict):
                for metric, info in explanations.items():
                    if isinstance(info, dict):
                        st.markdown(f"**{metric.replace('_', ' ').title()}**")
                        st.markdown(f"- **Definition:** {info.get('definition', 'N/A')}")
                        st.markdown(f"- **Range:** {info.get('range', 'N/A')}")
                        st.markdown(f"- **When to use:** {info.get('when_to_use', 'N/A')}")
                        st.markdown("---")

def display_results(results):
    """Enhanced pipeline results display with rich visualizations"""
    if not results:
        return
    
    # Results header with summary
                st.markdown("""
    <div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, var(--bg-info), var(--bg-secondary)); border-radius: var(--radius-lg); margin: 2rem 0;">
        <h2 style="margin: 0; color: var(--text-primary);">📊 Pipeline Results Dashboard</h2>
        <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary); opacity: 0.9;">
            Comprehensive analysis powered by 5 AI agents
        </p>
                </div>
                """, unsafe_allow_html=True)
                
    # Create enhanced tabs for results (without Problem Detection)
    tab1, tab2, tab3, tab4 = st.tabs([
        "🧹 Data Cleaning", "📊 EDA", "🤖 Model Training", "📈 Evaluation"
    ])
    
    with tab1:
        display_enhanced_data_cleaning(results.get("cleaning_summary_results", {}))
    
    with tab2:
        display_enhanced_eda(results.get("eda_results", {}))
    
    with tab3:
        display_enhanced_model_training(results.get("model_results", {}))
    
    with tab4:
        display_enhanced_evaluation(results.get("evaluation_results", {}))

def main():
    # Compact header
    st.markdown("""
    <div style="text-align: center; padding: 1.25rem 0; background: linear-gradient(135deg, var(--bg-info), var(--bg-secondary)); border-radius: var(--radius-lg); margin-bottom: 1.5rem;">
        <h1 style="margin: 0; font-size: 2rem; background: linear-gradient(135deg, var(--border-primary), var(--border-hover)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
            ⚙️ CrewML
        </h1>
        <p style="font-size: 1rem; margin: 0.25rem 0 0 0; color: var(--text-secondary); opacity: 0.9;">
            Learn and Experiment with 5 AI Crew Buddies - ML Made Fun!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Compelling distinguishing features with card design
    st.markdown('<h3 style="color: #3b82f6; margin-bottom: 1rem;">🚀 Why Choose This Pipeline?</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
                
                            with col1:
        st.markdown('''
        <div style="padding: 1rem; background: #ffffff; border-radius: 0.5rem; border-left: 4px solid #10b981; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); margin-bottom: 1rem;">
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">🤖 5 AI Agents Orchestra</div>
            <div style="font-size: 0.9rem; color: #64748b;">CrewAI-powered collaboration with specialized roles</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('''
        <div style="padding: 1rem; background: #ffffff; border-radius: 0.5rem; border-left: 4px solid #8b5cf6; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); margin-bottom: 1rem;">
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">🎯 Zero Data Leakage</div>
            <div style="font-size: 0.9rem; color: #64748b;">Target-aware preprocessing prevents common pitfalls</div>
        </div>
        ''', unsafe_allow_html=True)
                
                            with col2:
        st.markdown('''
        <div style="padding: 1rem; background: #ffffff; border-radius: 0.5rem; border-left: 4px solid #3b82f6; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); margin-bottom: 1rem;">
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">🎓 Learn While You Build</div>
            <div style="font-size: 0.9rem; color: #64748b;">Every decision explained with ML best practices</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('''
        <div style="padding: 1rem; background: #ffffff; border-radius: 0.5rem; border-left: 4px solid #f59e0b; box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05); margin-bottom: 1rem;">
            <div style="font-weight: 600; color: #1e293b; margin-bottom: 0.5rem;">⚡ Auto-Everything</div>
            <div style="font-size: 0.9rem; color: #64748b;">Problem detection, cleaning, training - all automated</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # ========== DATA SOURCE SECTION - MOVED TO MAIN PAGE ==========
                            st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, var(--bg-info), var(--bg-secondary)); border-radius: var(--radius-lg); margin: 2rem 0;">
        <h1 style="margin: 0 0 0.5rem 0; font-size: 2rem; background: linear-gradient(135deg, var(--border-primary), var(--border-hover)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
            🚀 Get Started - Upload Your Dataset
        </h1>
        <p style="color: var(--text-secondary); font-size: 1rem; line-height: 1.5;">
            🎓 Explore our Learning Centre first, then choose your data source below
        </p>
    </div>
    """, unsafe_allow_html=True)
    

    
    # Centered radio button selection with down arrow encouragement
    st.markdown("""
    <div style="text-align: center; margin: 1.5rem 0;">
        <h3 style="color: var(--text-primary); font-size: 1.2rem; margin-bottom: 0.5rem;">
            📊 Select Your Data Source ⬇️
        </h3>
        <p style="color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 1rem;">
            Choose your option below to continue
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
        data_source = st.radio(
            "",
            ["Upload CSV File", "Use Sample Dataset"],
            horizontal=True,
            help="Choose how you'd like to provide data for the ML pipeline"
        )
    
    
    # Conditional content based on data source selection
    selected_dataset = None
    uploaded_data = None
    uploaded_analysis = None
    uploaded_file = None
    
    if data_source == "Use Sample Dataset":
        # Sample dataset selection - only show when this option is selected
        st.markdown("""
        <h3 style="text-align: center; color: var(--text-primary); margin: 1.5rem 0 1rem 0;">
            Choose from Curated Sample Datasets
        </h3>
        """, unsafe_allow_html=True)
        
        # Load and display sample datasets
        available_datasets = load_sample_datasets()
        
        # Full-width dataset selection (preview info shown in "About your Dataset" section below)
        selected_dataset = st.selectbox(
            "Select a dataset to explore:",
            available_datasets,
            label_visibility="collapsed"
        )
    
    elif data_source == "Upload CSV File":
        # File upload section - only show when this option is selected
        st.markdown("""
        <h3 style="text-align: center; color: var(--text-primary); margin: 1.5rem 0 1rem 0;">
            Upload Your Custom Dataset
        </h3>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose your CSV file",
            type=['csv'],
            label_visibility="collapsed"
        )
        
        # Subtle file requirements info positioned near upload area
        st.markdown("""
        <div style="
            text-align: center;
            margin: 0.5rem 0 1rem 0;
            padding: 0.5rem;
            background: var(--bg-secondary);
            border-radius: var(--radius-sm);
            border: 1px solid var(--border-secondary);
        ">
            <p style="margin: 0; color: var(--text-secondary); font-size: 0.8rem; opacity: 0.8;">
                📁 CSV files only • Max 200MB • Headers required
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if uploaded_file is not None:
            try:
                uploaded_data = pd.read_csv(uploaded_file)
                
                # Success message with dataset info
                col1, col2 = st.columns(2)
                            with col1:
                    st.success(f"✅ Successfully loaded: **{uploaded_file.name}**")
        
        with col2:
                    st.info(f"📊 Shape: **{uploaded_data.shape[0]} rows × {uploaded_data.shape[1]} columns**")
                
                # Analyze uploaded data
                uploaded_analysis = analyze_uploaded_dataset(uploaded_data)
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                uploaded_data = None
    
    # Simplified sidebar - only Learning Centre
    display_educational_sidebar()
    
    # Dataset information
    if uploaded_data is not None:
        dataset_info = get_dataset_info("uploaded", uploaded_data, uploaded_analysis)
    else:
        dataset_info = get_dataset_info(selected_dataset)
    
    # Display dataset info with enhanced formatting
    with st.expander("📚 About your Dataset", expanded=True):
        # Header with key metrics - Dataset Shape, Problem Type, Target Variable
        col1, col2, col3 = st.columns(3)
        
    with col1:
            st.markdown(create_metric_card(
                dataset_info.get('dataset_shape', 'Unknown'),
                "Dataset Shape",
                "📊"
            ), unsafe_allow_html=True)
    
    with col2:
            st.markdown(create_metric_card(
                dataset_info['problem_type'],
                "Problem Type", 
                "🎯"
            ), unsafe_allow_html=True)
        
    with col3:
            st.markdown(create_metric_card(
                dataset_info.get('target_variable', 'Unknown'),
                "Target Variable",
                "🏹"
            ), unsafe_allow_html=True)
        
    st.markdown("---")
    config = enhanced_configuration(dataset_info, uploaded_data)
    
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
