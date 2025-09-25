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
    page_title="ML Pipeline Orchestrator",
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
    
    /* Dataset section headers */
    .dataset-section-header {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin-bottom: 0.75rem !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 2px solid var(--border-primary) !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
    }
    
    .dataset-section-header .icon {
        font-size: 1.25rem !important;
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
        
        .dataset-section-header {
            font-size: 1rem !important;
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
        return {
            "difficulty": uploaded_analysis.get("difficulty", "🟡 Intermediate"),
            "problem_type": uploaded_analysis.get("problem_type", "Unknown"),
            "description": f"Uploaded dataset with {uploaded_data.shape[0]} rows and {uploaded_data.shape[1]} columns",
            "challenges": ["Dynamic target identification", "Custom data characteristics"],
            "learning_objectives": ["Understanding target variable selection", "Working with custom datasets"],
            "recommended_config": {},
            "dataset_shape": f"{uploaded_data.shape[0]} × {uploaded_data.shape[1]}",
            "target_variable": uploaded_analysis.get("target_variable", "Unknown")
        }
    
    # Comprehensive sample dataset info
    dataset_info = {
        "survey_lung_cancer.csv": {
            "difficulty": "🟢 Beginner",
            "problem_type": "Binary Classification",
            "description": "Predict lung cancer based on survey responses. Great for learning classification basics with medical data.",
            "challenges": ["Imbalanced classes", "Categorical features", "Feature engineering"],
            "learning_objectives": ["Binary classification", "Handling imbalanced data", "Medical data analysis"],
            "recommended_config": {"algorithms": ["RandomForest", "LogisticRegression"]},
            "dataset_shape": "309 × 16",
            "target_variable": "LUNG_CANCER"
        },
        "iris.csv": {
            "difficulty": "🟢 Beginner",
            "problem_type": "Multi-class Classification",
            "description": "Classic iris flower classification. Perfect introduction to multi-class classification with clean data.",
            "challenges": ["Feature scaling", "Model selection", "Multi-class evaluation"],
            "learning_objectives": ["Multi-class classification", "Feature importance", "Model comparison"],
            "recommended_config": {"algorithms": ["RandomForest", "SVM", "KNN"]},
            "dataset_shape": "150 × 6",
            "target_variable": "Species"
        },
        "titanic.csv": {
            "difficulty": "🟡 Intermediate",
            "problem_type": "Binary Classification",
            "description": "Predict passenger survival on the Titanic. Classic dataset for learning feature engineering.",
            "challenges": ["Missing values", "Categorical features", "Feature engineering", "Data preprocessing"],
            "learning_objectives": ["Feature engineering", "Missing value strategies", "Categorical encoding"],
            "recommended_config": {"algorithms": ["RandomForest", "XGBoost", "LogisticRegression"]},
            "dataset_shape": "890 × 12",
            "target_variable": "Survived"
        },
        "wine_quality.csv": {
            "difficulty": "🟡 Intermediate", 
            "problem_type": "Multi-class Classification",
            "description": "Wine quality prediction based on physicochemical properties. Great for quality assessment tasks.",
            "challenges": ["Ordinal target", "Feature scaling", "Class imbalance"],
            "learning_objectives": ["Ordinal classification", "Feature selection", "Quality prediction"],
            "recommended_config": {"algorithms": ["RandomForest", "SVM", "GradientBoosting"]},
            "dataset_shape": "1143 × 13",
            "target_variable": "quality"
        },
        "house_prices.csv": {
            "difficulty": "🟡 Intermediate",
            "problem_type": "Regression",
            "description": "House price prediction based on property features. Excellent for regression learning.",
            "challenges": ["Feature scaling", "Outlier handling", "Feature selection", "Price prediction"],
            "learning_objectives": ["Regression modeling", "Feature scaling", "Outlier detection"],
            "recommended_config": {"algorithms": ["RandomForest", "XGBoost", "LinearRegression"]},
            "dataset_shape": "1000 × 7",
            "target_variable": "price"
        },
        "customer_churn.csv": {
            "difficulty": "🔴 Advanced",
            "problem_type": "Binary Classification",
            "description": "Customer churn prediction for business analytics. Complex dataset with multiple challenges.",
            "challenges": ["Class imbalance", "Feature scaling", "Business metrics", "Customer behavior"],
            "learning_objectives": ["Imbalanced classification", "Business analytics", "Customer retention"],
            "recommended_config": {"algorithms": ["RandomForest", "XGBoost", "SVM"]},
            "dataset_shape": "800 × 8",
            "target_variable": "churn"
        },
        "student_performance.csv": {
            "difficulty": "🟡 Intermediate",
            "problem_type": "Regression",
            "description": "Student performance prediction based on various factors. Educational data analysis.",
            "challenges": ["Mixed data types", "Feature engineering", "Educational metrics"],
            "learning_objectives": ["Educational data analysis", "Mixed data handling", "Performance prediction"],
            "recommended_config": {"algorithms": ["RandomForest", "LinearRegression", "SVM"]},
            "dataset_shape": "600 × 7",
            "target_variable": "final_grade"
        },
        "customer_segments.csv": {
            "difficulty": "🟡 Intermediate",
            "problem_type": "Clustering",
            "description": "Customer segmentation analysis for marketing insights. Unsupervised learning example.",
            "challenges": ["Unsupervised learning", "Cluster validation", "Feature scaling"],
            "learning_objectives": ["Clustering algorithms", "Customer segmentation", "Unsupervised learning"],
            "recommended_config": {"algorithms": ["KMeans", "DBSCAN", "Hierarchical"]},
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
            🧠 ML Guide
        </div>
        <div style="font-size: 0.9rem; opacity: 0.95; font-weight: 500; line-height: 1.3;">
            ✨ Explore before running pipeline
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
        "💡 Select ML Topic to Learn",
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
            with st.sidebar.expander(f"📖 {term}", expanded=False):
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
    
    # Data source selection - positioned at top for workflow
    st.sidebar.markdown("## 📁 Data Source")
    data_source = st.sidebar.radio(
        "Choose your data source",
        ["📤 Upload CSV File", "📊 Use Sample Dataset"]
    )
    
    # Sample dataset selection (appears right below data source)
    selected_dataset = None
    if data_source == "📊 Use Sample Dataset":
        available_datasets = load_sample_datasets()
        selected_dataset = st.sidebar.selectbox("Choose a dataset", available_datasets)
    
    # Educational sidebar - positioned below data source
    display_educational_sidebar()
    
    uploaded_data = None
    uploaded_analysis = None
    
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
                
            except Exception as e:
                st.markdown(create_info_box(f"""
                <strong>❌ Error reading file:</strong><br>
                {str(e)}
                """, "error"), unsafe_allow_html=True)
    
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
        
        # Detailed information
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="dataset-section-header"><span class="icon">📖</span>Description</div>', unsafe_allow_html=True)
            st.markdown(create_info_box(dataset_info['description']), unsafe_allow_html=True)
            
            if dataset_info.get('recommended_config', {}).get('algorithms'):
                st.markdown('<div class="dataset-section-header"><span class="icon">🔧</span>Recommended Algorithms</div>', unsafe_allow_html=True)
                algorithms = dataset_info['recommended_config']['algorithms']
                algorithm_text = ", ".join([f"**{alg}**" for alg in algorithms])
                st.markdown(create_info_box(algorithm_text), unsafe_allow_html=True)
        
        with col2:
            if dataset_info['challenges']:
                st.markdown('<div class="dataset-section-header"><span class="icon">⚡</span>Key Challenges</div>', unsafe_allow_html=True)
                challenges_html = "<br>".join([f"• {challenge}" for challenge in dataset_info['challenges']])
                st.markdown(create_info_box(challenges_html, "info"), unsafe_allow_html=True)
            
            if dataset_info['learning_objectives']:
                st.markdown('<div class="dataset-section-header"><span class="icon">🎓</span>Learning Objectives</div>', unsafe_allow_html=True)
                objectives_html = "<br>".join([f"• {obj}" for obj in dataset_info['learning_objectives']])
                st.markdown(create_info_box(objectives_html, "info"), unsafe_allow_html=True)
    
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
