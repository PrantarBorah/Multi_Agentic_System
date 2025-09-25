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
    page_title="ML Pipeline Orchestrator",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    /* CSS Variables for theme adaptation */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f0f2f6;
        --bg-tertiary: #f8f9fa;
        --bg-info: #e7f3ff;
        --bg-warning: #fff3cd;
        --text-primary: #1f2937;
        --text-secondary: #6c757d;
        --border-primary: #007bff;
        --border-secondary: #ffc107;
        --border-hover: #0056b3;
        --shadow: rgba(0, 0, 0, 0.1);
    }
    
    /* Dark mode variables */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #1a1a1a;
            --bg-secondary: #2d3748;
            --bg-tertiary: #4a5568;
            --bg-info: #2c5282;
            --bg-warning: #744210;
            --text-primary: #f7fafc;
            --text-secondary: #a0aec0;
            --border-primary: #63b3ed;
            --border-secondary: #f6ad55;
            --border-hover: #90cdf4;
            --shadow: rgba(0, 0, 0, 0.3);
        }
    }
    
    /* Streamlit dark mode detection */
    .stApp[data-testid="stAppViewContainer"] {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }
    
    .main {
        padding: 2rem;
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }
    
    .stButton>button {
        width: 100%;
        background-color: var(--border-primary);
        color: var(--bg-primary);
        border: 1px solid var(--border-primary);
    }
    
    .stButton>button:hover {
        background-color: var(--border-hover);
        border-color: var(--border-hover);
    }
    
    .step-container {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-primary);
        box-shadow: 0 2px 4px var(--shadow);
    }
    
    .config-section {
        background-color: var(--bg-tertiary);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid var(--border-primary);
        box-shadow: 0 2px 4px var(--shadow);
    }
    
    .info-box {
        background-color: var(--bg-info);
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: var(--text-primary);
        border: 1px solid var(--border-primary);
    }
    
    .upload-section {
        background-color: var(--bg-tertiary);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 2px dashed var(--border-primary);
        text-align: center;
        color: var(--text-primary);
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        background-color: var(--bg-info);
        border-color: var(--border-hover);
    }
    
    .target-analysis {
        background-color: var(--bg-warning);
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid var(--border-secondary);
        color: var(--text-primary);
        box-shadow: 0 2px 4px var(--shadow);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border: 2px dashed var(--border-primary) !important;
        background-color: var(--bg-tertiary) !important;
        border-radius: 0.5rem !important;
        padding: 1rem !important;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--border-hover) !important;
        background-color: var(--bg-info) !important;
    }
    
    .stFileUploader > div > div {
        color: var(--border-primary) !important;
    }
    
    .stFileUploader > div > div > div {
        color: var(--text-secondary) !important;
    }
    
    /* Expander styling */
    .stExpander > div > div {
        background-color: var(--bg-warning) !important;
        border-left: 4px solid var(--border-secondary) !important;
        color: var(--text-primary) !important;
    }
    
    .stExpander > div > div > div {
        background-color: var(--bg-warning) !important;
        color: var(--text-primary) !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }
    
    /* Metric styling */
    .stMetric {
        background-color: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
    }
    
    /* Checkbox styling */
    .stCheckbox > div > div {
        color: var(--text-primary) !important;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-primary) !important;
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        background-color: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-primary) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }
    
    /* Tab styling */
    .stTabs > div > div > div > div {
        background-color: var(--bg-tertiary) !important;
        color: var(--text-primary) !important;
    }
    
    .stTabs > div > div > div > div[data-baseweb="tab"] {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }
    
    /* Plot styling */
    .js-plotly-plot {
        background-color: var(--bg-primary) !important;
    }
    
    /* Markdown text color */
    .markdown-text-container {
        color: var(--text-primary) !important;
    }
    
    /* Code block styling */
    .stCodeBlock {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }
    
    /* Success/Error message styling */
    .stAlert {
        background-color: var(--bg-info) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-primary) !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: var(--border-primary) !important;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        color: var(--border-primary) !important;
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
    
    # Target variable identification logic - IMPROVED VERSION
    target_candidates = []
    target_selection_reasoning = []
    
    # HIGH PRIORITY: Binary classification targets
    binary_target_indicators = [
        'disease', 'heart', 'cancer', 'diabetes', 'stroke', 'attack',
        'survived', 'died', 'alive', 'dead', 'success', 'failure',
        'churn', 'retention', 'conversion', 'purchase', 'buy',
        'fraud', 'spam', 'malicious', 'legitimate',
        'positive', 'negative', 'yes', 'no', 'true', 'false',
        'target', 'label', 'class', 'outcome', 'result'
    ]
    
    # MEDIUM PRIORITY: Multi-class classification targets
    multiclass_target_indicators = [
        'species', 'type', 'category', 'grade', 'level', 'class',
        'quality', 'rating', 'score', 'performance', 'status'
    ]
    
    # LOW PRIORITY: Regression targets (avoid common features)
    regression_target_indicators = [
        'price', 'cost', 'salary', 'income', 'revenue', 'sales',
        'amount', 'value', 'score', 'rating', 'prediction'
    ]
    
    # EXCLUDE: Common feature columns that should never be targets
    exclude_from_targets = [
        'id', 'index', 'row', 'name', 'first', 'last', 'full',
        'age', 'gender', 'sex', 'male', 'female', 'height', 'weight',
        'date', 'time', 'year', 'month', 'day', 'hour', 'minute',
        'email', 'phone', 'address', 'city', 'state', 'country',
        'zip', 'postal', 'latitude', 'longitude', 'location',
        'user', 'customer', 'client', 'patient', 'person', 'individual'
    ]
    
    for col in data.columns:
        col_lower = col.lower()
        score = 0
        reasons = []
        
        # Check if column should be excluded
        should_exclude = False
        for exclude_term in exclude_from_targets:
            if exclude_term in col_lower:
                should_exclude = True
                reasons.append(f"Column '{col}' contains common feature indicator '{exclude_term}' - likely not a target")
                break
        
        if should_exclude:
            continue
        
        # HIGH PRIORITY: Check for binary classification indicators
        for indicator in binary_target_indicators:
            if indicator in col_lower:
                score += 10  # High priority
                reasons.append(f"Column name '{col}' contains binary target indicator '{indicator}'")
                break
        
        # MEDIUM PRIORITY: Check for multi-class indicators
        for indicator in multiclass_target_indicators:
            if indicator in col_lower:
                score += 5
                reasons.append(f"Column name '{col}' contains multi-class indicator '{indicator}'")
                break
        
        # LOW PRIORITY: Check for regression indicators
        for indicator in regression_target_indicators:
            if indicator in col_lower:
                score += 2
                reasons.append(f"Column name '{col}' contains regression indicator '{indicator}'")
                break
        
        # Analyze data characteristics
        unique_vals = data[col].nunique()
        total_rows = len(data[col])
        unique_ratio = unique_vals / total_rows
        
        # Check for binary patterns (highest priority)
        if data[col].dtype in ['int64', 'bool'] or (data[col].dtype == 'object' and unique_vals == 2):
            unique_values = data[col].unique()
            if len(unique_values) == 2:
                # Check for common binary patterns
                binary_patterns = [
                    {0, 1}, {True, False}, {'0', '1'}, {'yes', 'no'},
                    {'true', 'false'}, {'positive', 'negative'}, {'Y', 'N'}
                ]
                if set(str(v).lower() for v in unique_values) in [set(str(v).lower() for v in pattern) for pattern in binary_patterns]:
                    score += 15  # Very high priority for binary
                    reasons.append(f"Binary values {unique_values} strongly suggest binary classification target")
                else:
                    score += 8
                    reasons.append(f"Binary values {unique_values} suggest classification target")
        
        # Check cardinality for classification vs regression
        if data[col].dtype in ['object', 'category']:
            if unique_ratio < 0.1:  # Very low cardinality
                score += 6
                reasons.append(f"Low cardinality ({unique_vals} unique values) suggests categorical target")
            elif unique_ratio < 0.3:  # Low cardinality
                score += 3
                reasons.append(f"Moderate cardinality ({unique_vals} unique values) suggests categorical target")
        elif data[col].dtype in ['int64', 'float64']:
            if unique_ratio < 0.05:  # Very low cardinality for numeric
                score += 8
                reasons.append(f"Very low cardinality ({unique_vals} unique values) suggests categorical target")
            elif unique_ratio < 0.1:  # Low cardinality for numeric
                score += 5
                reasons.append(f"Low cardinality ({unique_vals} unique values) suggests categorical target")
            elif unique_ratio > 0.8:  # High cardinality suggests continuous
                score += 1
                reasons.append(f"High cardinality ({unique_vals} unique values) suggests continuous target")
        
        # Check for class imbalance in categorical data
        if data[col].dtype in ['object', 'category'] or unique_vals < 20:
            value_counts = data[col].value_counts()
            if len(value_counts) == 2:
                ratio = min(value_counts) / max(value_counts)
                if ratio < 0.3:
                    reasons.append(f"Imbalanced classes detected (ratio: {ratio:.2f})")
                else:
                    reasons.append(f"Relatively balanced classes (ratio: {ratio:.2f})")
        
        # Additional checks for specific patterns
        if 'disease' in col_lower or 'health' in col_lower:
            score += 5
            reasons.append(f"Health-related column '{col}' likely indicates medical outcome")
        
        if 'survived' in col_lower or 'died' in col_lower:
            score += 8
            reasons.append(f"Survival-related column '{col}' strongly suggests binary outcome")
        
        if 'churn' in col_lower or 'retention' in col_lower:
            score += 6
            reasons.append(f"Customer behavior column '{col}' suggests business outcome")
        
        # Penalize columns that look like features
        if any(feature_term in col_lower for feature_term in ['age', 'gender', 'height', 'weight', 'income']):
            score -= 5
            reasons.append(f"Column '{col}' appears to be a demographic/feature variable")
        
        if score > 0:
            target_candidates.append({
                'column': col,
                'score': score,
                'reasons': reasons,
                'unique_values': unique_vals,
                'dtype': str(data[col].dtype),
                'unique_ratio': unique_ratio
            })
    
    # Sort candidates by score (highest first)
    target_candidates.sort(key=lambda x: x['score'], reverse=True)
    
    if target_candidates:
        best_candidate = target_candidates[0]
        analysis["target_variable"] = best_candidate['column']
        analysis["target_selection_reasoning"] = best_candidate['reasons']
        
        # Determine problem type based on target characteristics
        target_col = best_candidate['column']
        target_data = data[target_col]
        
        if target_data.dtype in ['object', 'category'] or target_data.nunique() < 20:
            if target_data.nunique() == 2:
                analysis["problem_type"] = "Binary Classification"
                analysis["recommendations"]["algorithms"] = ["RandomForest", "LogisticRegression", "SVM"]
                analysis["recommendations"]["cv_strategy"] = "stratified"
            else:
                analysis["problem_type"] = "Multi-class Classification"
                analysis["recommendations"]["algorithms"] = ["RandomForest", "SVM", "KNN"]
                analysis["recommendations"]["cv_strategy"] = "stratified"
        else:
            analysis["problem_type"] = "Regression"
            analysis["recommendations"]["algorithms"] = ["RandomForest", "XGBoost", "LinearRegression"]
            analysis["recommendations"]["cv_strategy"] = "kfold"
        
        # Add difficulty assessment
        if analysis["problem_type"] in ["Binary Classification", "Multi-class Classification"]:
            if target_data.nunique() == 2:
                # Check for class imbalance
                value_counts = target_data.value_counts()
                ratio = min(value_counts) / max(value_counts)
                if ratio < 0.3:
                    analysis["difficulty"] = "🔴 Advanced"
                    analysis["recommendations"]["imputation_method"] = "median"
                    analysis["recommendations"]["outlier_handling"] = "cap"
                else:
                    analysis["difficulty"] = "🟡 Intermediate"
                    analysis["recommendations"]["imputation_method"] = "mean"
                    analysis["recommendations"]["outlier_handling"] = "remove"
            else:
                analysis["difficulty"] = "🟡 Intermediate"
                analysis["recommendations"]["imputation_method"] = "mean"
                analysis["recommendations"]["outlier_handling"] = "remove"
        else:  # Regression
            analysis["difficulty"] = "🟡 Intermediate"
            analysis["recommendations"]["imputation_method"] = "median"
            analysis["recommendations"]["outlier_handling"] = "cap"
        
        # Add all candidates for transparency
        analysis["all_candidates"] = target_candidates[:5]  # Top 5 candidates
    else:
        analysis["target_selection_reasoning"] = [
            "No clear target variable identified. Please manually select a target column.",
            "Consider columns with:",
            "- Binary values (0/1, True/False, Yes/No)",
            "- Low cardinality categorical values",
            "- Names suggesting outcomes (disease, survived, churn, etc.)",
            "- Avoid common feature columns (age, gender, income, etc.)"
        ]
    
    return analysis

def get_dataset_info(dataset_name, uploaded_data=None, uploaded_analysis=None):
    """Get educational information about the selected dataset"""
    
    # If it's an uploaded dataset, use the analysis
    if uploaded_data is not None and uploaded_analysis is not None:
        return {
            "difficulty": uploaded_analysis.get("difficulty", "🟡 Intermediate"),
            "problem_type": uploaded_analysis.get("problem_type", "Unknown"),
            "description": f"Uploaded dataset with {uploaded_data.shape[0]} rows and {uploaded_data.shape[1]} columns. Target variable: {uploaded_analysis.get('target_variable', 'Not identified')}",
            "challenges": [
                "Dynamic target identification",
                "Custom data characteristics",
                "Unknown data quality issues"
            ],
            "learning_objectives": [
                "Understanding target variable selection",
                "Working with custom datasets",
                "Adapting to different data types"
            ],
            "recommended_config": uploaded_analysis.get("recommendations", {})
        }
    
    # Otherwise, use predefined dataset information
    dataset_info = {
        "1_survey_lung_cancer.csv": {
            "difficulty": "🟢 Beginner",
            "problem_type": "Binary Classification",
            "description": "Predict lung cancer based on survey responses. Great for learning classification basics.",
            "challenges": ["Imbalanced classes", "Categorical features", "Feature engineering"],
            "learning_objectives": ["Binary classification", "Handling imbalanced data", "Categorical encoding"],
            "recommended_config": {
                "imputation_method": "mode",
                "outlier_handling": "cap",
                "algorithms": ["RandomForest", "LogisticRegression"],
                "cv_strategy": "stratified"
            }
        },
        "2_Iris.csv": {
            "difficulty": "🟢 Beginner", 
            "problem_type": "Multi-class Classification",
            "description": "Classic iris flower classification. Perfect for learning multi-class classification.",
            "challenges": ["Feature scaling", "Model selection", "Multi-class evaluation"],
            "learning_objectives": ["Multi-class classification", "Feature importance", "Model comparison"],
            "recommended_config": {
                "imputation_method": "mean",
                "outlier_handling": "remove",
                "algorithms": ["RandomForest", "SVM", "KNN"],
                "cv_strategy": "stratified"
            }
        },
        "3_titanic.csv": {
            "difficulty": "🟡 Intermediate",
            "problem_type": "Binary Classification", 
            "description": "Predict survival on Titanic. Good for learning feature engineering and missing value handling.",
            "challenges": ["Missing values", "Categorical features", "Feature engineering"],
            "learning_objectives": ["Feature engineering", "Missing value strategies", "Categorical encoding"],
            "recommended_config": {
                "imputation_method": "median",
                "outlier_handling": "cap",
                "algorithms": ["RandomForest", "XGBoost", "LogisticRegression"],
                "cv_strategy": "stratified"
            }
        },
        "4_house_prices.csv": {
            "difficulty": "🟡 Intermediate",
            "problem_type": "Regression",
            "description": "Predict house prices. Excellent for learning regression and feature scaling.",
            "challenges": ["Feature scaling", "Outlier handling", "Feature selection"],
            "learning_objectives": ["Regression modeling", "Feature scaling", "Outlier detection"],
            "recommended_config": {
                "imputation_method": "median",
                "outlier_handling": "cap",
                "algorithms": ["RandomForest", "XGBoost", "LinearRegression"],
                "cv_strategy": "kfold"
            }
        },
        "5_customer_churn.csv": {
            "difficulty": "🔴 Advanced",
            "problem_type": "Binary Classification",
            "description": "Predict customer churn. Complex dataset with multiple challenges.",
            "challenges": ["Class imbalance", "Feature scaling", "Model tuning"],
            "learning_objectives": ["Imbalanced classification", "Advanced feature engineering", "Model optimization"],
            "recommended_config": {
                "imputation_method": "median",
                "outlier_handling": "cap",
                "algorithms": ["RandomForest", "XGBoost", "SVM"],
                "cv_strategy": "stratified"
            }
        },
        "6_student_performance.csv": {
            "difficulty": "🟡 Intermediate",
            "problem_type": "Multi-class Classification",
            "description": "Predict student performance levels. Good for learning multi-class with mixed data types.",
            "challenges": ["Mixed data types", "Feature engineering", "Multi-class evaluation"],
            "learning_objectives": ["Multi-class classification", "Mixed data handling", "Feature importance"],
            "recommended_config": {
                "imputation_method": "mean",
                "outlier_handling": "remove",
                "algorithms": ["RandomForest", "SVM", "KNN"],
                "cv_strategy": "stratified"
            }
        }
    }
    return dataset_info.get(dataset_name, {
        "difficulty": "🟡 Intermediate",
        "problem_type": "Unknown",
        "description": "Custom dataset",
        "challenges": [],
        "learning_objectives": [],
        "recommended_config": {}
    })

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

def display_interactive_configuration():
    """Display interactive configuration options for each agent"""
    st.subheader("⚙️ Interactive Configuration")
    st.markdown("Customize each step of the ML pipeline to understand the impact of different choices.")
    
    # Create tabs for each agent configuration
    config_tab1, config_tab2, config_tab3, config_tab4 = st.tabs([
        "🧹 Data Cleaning", "📊 EDA", "🤖 Model Training", "📈 Evaluation"
    ])
    
    with config_tab1:
        st.markdown("### Data Cleaning Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Missing Value Handling**")
            imputation_method = st.selectbox(
                "Imputation Method",
                ["auto", "mean", "median", "mode", "forward_fill", "backward_fill"],
                help="Choose how to handle missing values. Auto lets the agent decide based on data characteristics."
            )
            
            if imputation_method != "auto":
                st.markdown(f"""
                <div class="info-box">
                <strong>Selected Method:</strong> {imputation_method.title()}<br>
                <strong>Best for:</strong> {
                    "Mean: Normal distributions" if imputation_method == "mean" else
                    "Median: Skewed data" if imputation_method == "median" else
                    "Mode: Categorical data" if imputation_method == "mode" else
                    "Forward/Backward: Time series data" if imputation_method in ["forward_fill", "backward_fill"] else ""
                }
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Outlier Handling**")
            outlier_method = st.selectbox(
                "Outlier Method",
                ["auto", "remove", "cap", "transform", "isolation"],
                help="Choose how to handle outliers. Auto lets the agent decide based on data characteristics."
            )
            
            if outlier_method != "auto":
                st.markdown(f"""
                <div class="info-box">
                <strong>Selected Method:</strong> {outlier_method.title()}<br>
                <strong>Best for:</strong> {
                    "Remove: Clear data errors" if outlier_method == "remove" else
                    "Cap: Valid but extreme values" if outlier_method == "cap" else
                    "Transform: Skewed distributions" if outlier_method == "transform" else
                    "Isolation: Automatic detection" if outlier_method == "isolation" else ""
                }
                </div>
                """, unsafe_allow_html=True)
    
    with config_tab2:
        st.markdown("### EDA Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Visualization Options**")
            include_correlations = st.checkbox("Correlation Analysis", value=True, 
                help="Show correlation heatmap for numeric features")
            include_distributions = st.checkbox("Distribution Plots", value=True,
                help="Show distribution plots for each feature")
            include_boxplots = st.checkbox("Box Plots", value=True,
                help="Show box plots for outlier detection")
            include_pairplots = st.checkbox("Pair Plots", value=False,
                help="Show pairwise relationships (can be slow for large datasets)")
        
        with col2:
            st.markdown("**Analysis Options**")
            include_insights = st.checkbox("AI-Generated Insights", value=True,
                help="Generate AI-powered insights about the data")
            include_recommendations = st.checkbox("ML Recommendations", value=True,
                help="Provide recommendations for model selection and preprocessing")
            custom_questions = st.text_area("Custom Questions (one per line)",
                placeholder="What is the correlation between age and income?\nWhich features have the most outliers?",
                help="Ask specific questions about your data")
    
    with config_tab3:
        st.markdown("### Model Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Algorithm Selection**")
            available_algorithms = {
                "RandomForest": "Ensemble of decision trees - good for most problems",
                "XGBoost": "Gradient boosting - high performance, good for competitions",
                "LogisticRegression": "Linear model - interpretable, good baseline",
                "SVM": "Support Vector Machine - good for high-dimensional data",
                "KNN": "K-Nearest Neighbors - simple, good for small datasets",
                "DecisionTree": "Single tree - interpretable, can overfit",
                "LinearRegression": "Linear regression for continuous targets"
            }
            
            selected_algorithms = st.multiselect(
                "Choose Algorithms to Compare",
                list(available_algorithms.keys()),
                default=["RandomForest", "LogisticRegression"],
                help="Select multiple algorithms to compare their performance"
            )
            
            for algo in selected_algorithms:
                st.markdown(f"""
                <div class="info-box">
                <strong>{algo}:</strong> {available_algorithms[algo]}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**Training Configuration**")
            cv_strategy = st.selectbox(
                "Cross-Validation Strategy",
                ["auto", "stratified", "kfold", "holdout"],
                help="Choose validation strategy. Auto adapts based on data characteristics."
            )
            
            hyperparameter_tuning = st.checkbox("Enable Hyperparameter Tuning", value=True,
                help="Automatically tune model parameters for better performance")
            
            feature_selection = st.checkbox("Enable Feature Selection", value=False,
                help="Automatically select the most important features")
    
    with config_tab4:
        st.markdown("### Evaluation Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Classification Metrics**")
            classification_metrics = st.multiselect(
                "Select Classification Metrics",
                ["accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"],
                default=["accuracy", "precision", "recall", "f1"],
                help="Choose which metrics to calculate for classification problems"
            )
        
        with col2:
            st.markdown("**Regression Metrics**")
            regression_metrics = st.multiselect(
                "Select Regression Metrics", 
                ["r2", "mse", "rmse", "mae"],
                default=["r2", "rmse"],
                help="Choose which metrics to calculate for regression problems"
            )
        
        st.markdown("**Visualization Options**")
        col3, col4 = st.columns(2)
        
        with col3:
            include_confusion_matrix = st.checkbox("Confusion Matrix", value=True)
            include_roc_curve = st.checkbox("ROC Curve", value=True)
            include_precision_recall = st.checkbox("Precision-Recall Curve", value=False)
        
        with col4:
            include_feature_importance = st.checkbox("Feature Importance", value=True)
            include_predictions_plot = st.checkbox("Predictions vs Actual", value=True)
            include_model_comparison = st.checkbox("Model Comparison Table", value=True)
    
    # Return configuration
    config = {
        "cleaning": {
            "imputation_method": imputation_method,
            "outlier_method": outlier_method
        },
        "eda": {
            "include_correlations": include_correlations,
            "include_distributions": include_distributions,
            "include_boxplots": include_boxplots,
            "include_pairplots": include_pairplots,
            "include_insights": include_insights,
            "include_recommendations": include_recommendations,
            "custom_questions": custom_questions.split('\n') if custom_questions else []
        },
        "training": {
            "algorithms": selected_algorithms,
            "cv_strategy": cv_strategy,
            "hyperparameter_tuning": hyperparameter_tuning,
            "feature_selection": feature_selection
        },
        "evaluation": {
            "classification_metrics": classification_metrics,
            "regression_metrics": regression_metrics,
            "include_confusion_matrix": include_confusion_matrix,
            "include_roc_curve": include_roc_curve,
            "include_precision_recall": include_precision_recall,
            "include_feature_importance": include_feature_importance,
            "include_predictions_plot": include_predictions_plot,
            "include_model_comparison": include_model_comparison
        }
    }
    
    return config

def display_problem_detection_results(results):
    """Display problem type detection results"""
    st.subheader("🎯 Problem Type Detection Results")
    
    # Add contextual learning popup for problem detection stage
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 🎯 Problem Type Analysis")
    with col2:
        display_contextual_learning_popup("problem_detection", "Learn Detection Concepts")
    
    # Display problem type information
    if "problem_type" in results:
        st.markdown(f"**🔍 Problem Type:** {results['problem_type']}")
    
    if "target_variable" in results:
        st.markdown(f"**🎯 Target Variable:** {results['target_variable']}")
    
    if "target_characteristics" in results:
        st.markdown("**📊 Target Characteristics:**")
        target_chars = results["target_characteristics"]
        col1, col2 = st.columns(2)
        
        with col1:
            if "data_type" in target_chars:
                st.write(f"**Data Type:** {target_chars['data_type']}")
            if "unique_values" in target_chars:
                st.write(f"**Unique Values:** {target_chars['unique_values']}")
        
        with col2:
            if "missing_values" in target_chars:
                st.write(f"**Missing Values:** {target_chars['missing_values']}")
            if "distribution" in target_chars:
                st.write(f"**Distribution:** {target_chars['distribution']}")
    
    if "reasoning" in results:
        st.markdown("**🧠 Detection Reasoning:**")
        for reason in results["reasoning"]:
            st.markdown(f"- {reason}")
    
    if "recommendations" in results:
        st.markdown("**💡 Recommendations:**")
        for rec in results["recommendations"]:
            st.markdown(f"- {rec}")

def display_contextual_learning_popup(stage: str, trigger_text: str = "Click for ML concepts"):
    """Display contextual learning popup for specific pipeline stages"""
    
    # Stage-specific ML concepts
    stage_concepts = {
        "problem_detection": {
            "title": "🎯 Problem Type Detection Concepts",
            "concepts": {
                "Supervised Learning": "Learning from labeled data to predict outcomes",
                "Classification": "Predicting categorical outcomes (e.g., yes/no, categories)",
                "Regression": "Predicting continuous numerical values",
                "Unsupervised Learning": "Finding patterns in unlabeled data",
                "Target Variable": "The variable we want to predict (dependent variable)",
                "Feature Variables": "Variables used to make predictions (independent variables)"
            }
        },
        "cleaning": {
            "title": "🧹 Data Cleaning Concepts",
            "concepts": {
                "Missing Values": "Data points that are not available (NaN, null)",
                "Imputation": "Filling missing values using statistical methods",
                "Outlier Detection": "Identifying data points that deviate significantly",
                "Data Type Conversion": "Converting data to appropriate formats",
                "Feature Scaling": "Normalizing features to same range",
                "Data Validation": "Checking data quality and consistency"
            }
        },
        "eda": {
            "title": "📊 Exploratory Data Analysis Concepts",
            "concepts": {
                "Descriptive Statistics": "Summary statistics (mean, median, std, etc.)",
                "Correlation Analysis": "Understanding relationships between variables",
                "Data Distribution": "How data is spread across different values",
                "Visualization": "Graphical representation of data patterns",
                "Feature Relationships": "How features interact with each other",
                "Data Insights": "Key findings and patterns in the data"
            }
        },
        "training": {
            "title": "🤖 Model Training Concepts",
            "concepts": {
                "Algorithm Selection": "Choosing the right ML algorithm for the task",
                "Cross-Validation": "Assessing model performance robustly",
                "Hyperparameter Tuning": "Optimizing model parameters",
                "Training/Test Split": "Dividing data for training and evaluation",
                "Overfitting": "Model memorizes training data instead of learning patterns",
                "Model Performance": "How well the model generalizes to new data"
            }
        },
        "evaluation": {
            "title": "📈 Model Evaluation Concepts",
            "concepts": {
                "Accuracy": "Proportion of correct predictions",
                "Precision": "Accuracy of positive predictions",
                "Recall": "Ability to find all positive cases",
                "F1-Score": "Balanced measure of precision and recall",
                "ROC Curve": "Graphical plot of true positive vs false positive rates",
                "Confusion Matrix": "Table showing prediction vs actual results"
            }
        }
    }
    
    if stage in stage_concepts:
        # Create a button that triggers the popup
        if st.button(f"📚 {trigger_text}", key=f"learn_{stage}"):
            # Display concepts in an expander (simulating popup behavior)
            with st.expander(stage_concepts[stage]["title"], expanded=True):
                for concept, explanation in stage_concepts[stage]["concepts"].items():
                    st.markdown(f"**{concept}**: {explanation}")
                
                # Add practical tips
                st.markdown("---")
                st.markdown("**💡 Practical Tips:**")
                if stage == "cleaning":
                    st.markdown("""
                    - Always check for missing values first
                    - Use domain knowledge to handle outliers
                    - Ensure data types are appropriate
                    - Document all cleaning decisions
                    """)
                elif stage == "training":
                    st.markdown("""
                    - Start with simple models first
                    - Use cross-validation for reliable estimates
                    - Monitor for overfitting
                    - Try multiple algorithms
                    """)
                elif stage == "evaluation":
                    st.markdown("""
                    - Don't rely on just one metric
                    - Consider business context
                    - Validate on unseen data
                    - Document model limitations
                    """)

def display_educational_sidebar():
    """Display educational sidebar with glossary and learning resources"""
    st.sidebar.markdown("## 🎓 Learning Center")
    
    # ML Glossary
    if st.sidebar.button("📚 Open ML Glossary"):
        st.session_state.show_glossary = True
    
    if st.session_state.get('show_glossary', False):
        glossary = create_ml_glossary()
        
        st.sidebar.markdown("### 📚 ML Glossary")
        category = st.sidebar.selectbox(
            "Choose Category",
            list(glossary.keys())
        )
        
        if category in glossary:
            for term, info in glossary[category].items():
                with st.sidebar.expander(f"📖 {term}"):
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
                        st.markdown("**When to use:**")
                        st.markdown(f"- {info['when_to_use']}")
        
        if st.sidebar.button("❌ Close Glossary"):
            st.session_state.show_glossary = False
    
    # Quick Concepts
    st.sidebar.markdown("### 🔍 Quick Concepts")
    
    with st.sidebar.expander("📊 Data Preprocessing"):
        st.markdown("""
        **Key Steps:**
        - **Cleaning:** Handle missing values and outliers
        - **Scaling:** Normalize features to same scale
        - **Encoding:** Convert categorical to numeric
        - **Feature Engineering:** Create new features
        """)
    
    with st.sidebar.expander("🤖 Model Selection"):
        st.markdown("""
        **Choose based on:**
        - **Data size:** Small → KNN, Large → Random Forest
        - **Interpretability:** Need explanation → Decision Tree
        - **Performance:** High accuracy → XGBoost
        - **Speed:** Fast training → Linear models
        """)
    
    with st.sidebar.expander("📈 Evaluation"):
        st.markdown("""
        **Classification:**
        - Accuracy: Overall correctness
        - Precision: How many positives were correct
        - Recall: How many actual positives found
        - F1: Balanced precision/recall
        
        **Regression:**
        - R²: How well model explains variance
        - RMSE: Average prediction error
        """)

def display_cleaning_results(results):
    """Display enhanced data cleaning results with transparency"""
    st.subheader("📊 Data Cleaning Results")
    
    # Add contextual learning popup for cleaning stage
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 🧹 Data Cleaning Process")
    with col2:
        display_contextual_learning_popup("cleaning", "Learn Cleaning Concepts")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Decision Log", "👣 Step-by-Step", "🎓 Learning", "📝 Summary"])
    
    with tab1:
        st.markdown("### 🤔 Decision Rationale")
        if "decision_log" in results:
            for decision in results["decision_log"]:
                with st.expander(f"🔍 {decision.get('column', 'Unknown')} - {decision.get('action_taken', 'Unknown Action')}"):
                    st.markdown(f"**Decision Type:** {decision.get('decision_type', 'N/A')}")
                    st.markdown(f"**Action Taken:** {decision.get('action_taken', 'N/A')}")
                    st.markdown(f"**Reasoning:** {decision.get('reasoning', 'N/A')}")
                    if 'data_characteristics' in decision:
                        st.markdown(f"**Data Characteristics:** {decision['data_characteristics']}")
                    if 'alternatives_considered' in decision and decision['alternatives_considered']:
                        st.markdown("**Alternatives Considered:**")
                        for alt in decision['alternatives_considered']:
                            st.markdown(f"- {alt}")
                    if 'ml_concept' in decision:
                        st.markdown(f"**ML Concept:** {decision['ml_concept']}")
    
    with tab2:
        st.markdown("### 👣 Process Timeline")
        if "step_log" in results:
            for i, step in enumerate(results["step_log"], 1):
                st.markdown(f"**{i}. {step.get('description', 'Unknown Step')}**")
                st.markdown(f"*{step.get('timestamp', 'Unknown Time')}*")
                if 'details' in step and step['details']:
                    st.markdown(f"Details: {step['details']}")
                st.markdown("---")
    
    with tab3:
        st.markdown("### 🎓 Educational Insights")
        if "educational_insights" in results:
            insights = results["educational_insights"]
            if isinstance(insights, dict):
                for concept, info in insights.items():
                    if isinstance(info, dict):
                        with st.expander(f"📚 {concept}"):
                            st.markdown(f"**Definition:** {info.get('definition', 'N/A')}")
                            if 'why_important' in info:
                                st.markdown(f"**Why Important:** {info['why_important']}")
                            if 'common_methods' in info:
                                st.markdown("**Common Methods:**")
                                for method in info['common_methods']:
                                    st.markdown(f"- {method}")
                            if 'best_practices' in info:
                                st.markdown(f"**Best Practices:** {info['best_practices']}")
                    else:
                        # Handle case where info is a string
                        with st.expander(f"📚 {concept}"):
                            st.markdown(f"**Information:** {info}")
            else:
                # Handle case where educational_insights is a list or other type
                st.markdown(f"**Educational Insights:** {insights}")
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            if "original_shape" in results:
                st.write("Original Data Shape:", results["original_shape"])
            if "cleaned_shape" in results:
                st.write("Cleaned Data Shape:", results["cleaned_shape"])
            
            if "missing_values_info" in results:
                st.write("Missing Values Handling:")
                st.json(results["missing_values_info"])
        
        with col2:
            if "data_type_fixes" in results:
                st.write("Data Type Fixes:")
                st.json(results["data_type_fixes"])
            
            if "cleaning_summary" in results:
                st.write("Cleaning Summary:")
                st.write(results["cleaning_summary"])

def display_eda_results(results):
    """Display EDA results"""
    st.subheader("📈 Exploratory Data Analysis")
    
    # Add contextual learning popup for EDA stage
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 📊 Data Exploration Process")
    with col2:
        display_contextual_learning_popup("eda", "Learn EDA Concepts")
    
    # Display summary statistics
    if "summary_stats" in results:
        st.write("Summary Statistics:")
        
        st.subheader("Numeric Summary")
        numeric_summary = pd.DataFrame(results["summary_stats"]["numeric"])
        st.dataframe(numeric_summary.transpose())
        
        st.subheader("Categorical Value Counts")
        if results["summary_stats"]["categorical"]:
            for col, counts in results["summary_stats"]["categorical"].items():
                st.write(f"**{col}**: {counts}")
        else:
            st.write("No categorical columns to display value counts for.")

        st.subheader("Missing Values")
        missing_values = pd.DataFrame.from_dict(results["summary_stats"]["missing_values"], orient='index', columns=['Count'])
        st.dataframe(missing_values)
    
    # Display correlations
    if "correlations" in results:
        st.write("Feature Correlations:")
        corr_df = pd.DataFrame(results["correlations"])
        fig = px.imshow(corr_df,
                       title="Correlation Heatmap",
                       color_continuous_scale="RdBu",
                       text_auto=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # Display distributions
    if "distributions" in results:
        st.write("Feature Distributions:")
        for col, dist_data in results["distributions"].items():
            fig = go.Figure(dist_data)
            st.plotly_chart(fig, use_container_width=True)
    
    # Display insights
    if "eda_insights" in results:
        st.write("EDA Insights:")
        st.write(results["eda_insights"])

def display_model_results(results):
    """Display model training results"""
    st.subheader("🤖 Model Training Results")
    
    # Add contextual learning popup for training stage
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 🤖 Model Training Process")
    with col2:
        display_contextual_learning_popup("training", "Learn Training Concepts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "selected_model" in results:
            st.write("Selected Model:", results["selected_model"])
        if "cv_score" in results:
            st.write("Cross-validation Score:", results["cv_score"])
        
        if results.get("feature_importance") and len(results["feature_importance"]) > 0:
            st.write("Feature Importance:")
            importance_df = pd.DataFrame(results["feature_importance"])
            fig = px.bar(importance_df,
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if "model_comparison" in results:
            st.write("Model Comparison:")
            st.dataframe(pd.DataFrame(results["model_comparison"]))
        
        if "training_summary" in results:
            st.write("Training Summary:")
            st.write(results["training_summary"])

def display_evaluation_results(results):
    """Display model evaluation results"""
    st.subheader("📊 Model Evaluation")
    
    # Add contextual learning popup for evaluation stage
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 📊 Model Evaluation Process")
    with col2:
        display_contextual_learning_popup("evaluation", "Learn Evaluation Concepts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if "performance_metrics" in results:
            st.write("Performance Metrics:")
            st.json(results["performance_metrics"])
        
        if "plots" in results and "confusion_matrix" in results["plots"]:
            st.write("Confusion Matrix:")
            fig = go.Figure(json.loads(results["plots"]["confusion_matrix"]))
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if "plots" in results and "actual_vs_predicted" in results["plots"]:
            st.write("Predictions vs Actual:")
            fig = go.Figure(json.loads(results["plots"]["actual_vs_predicted"]))
            st.plotly_chart(fig, use_container_width=True)
        
        if "recommendations" in results:
            st.write("Recommendations:")
            st.write(results["recommendations"])

def main():
    st.title("🤖 ML Pipeline Orchestrator")
    st.markdown("""
    This application demonstrates an end-to-end machine learning pipeline using AI agents.
    The pipeline includes data cleaning, exploratory data analysis, model training, and evaluation.
    """)
    
    # Educational sidebar
    display_educational_sidebar()
    
    # Data Source Selection
    st.sidebar.title("Data Source")
    data_source = st.sidebar.radio(
        "Choose data source",
        ["📁 Upload CSV File", "📊 Use Sample Dataset"],
        help="Upload your own CSV file or use one of our sample datasets"
    )
    
    uploaded_data = None
    uploaded_analysis = None
    selected_dataset = None
    
    if data_source == "📁 Upload CSV File":
        st.markdown("### 📁 Upload Your Dataset")
        
        # Upload section with proper styling
        st.markdown("""
        <div class="upload-section">
        <h4>📤 Upload CSV File</h4>
        <p>Upload your CSV file to analyze and process with our ML pipeline.</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with your data"
        )
        
        if uploaded_file is not None:
            try:
                # Read the uploaded file
                uploaded_data = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Shape: {uploaded_data.shape}")
                
                # Analyze the uploaded dataset
                with st.spinner("🔍 Analyzing your dataset..."):
                    uploaded_analysis = analyze_uploaded_dataset(uploaded_data)
                
                # Display target analysis with transparency
                st.markdown("### 🎯 Target Variable Analysis")
                
                # Target analysis section with proper styling
                st.markdown("""
                <div class="target-analysis">
                <h4>🎯 Automatic Target Identification</h4>
                <p>Our system automatically analyzed your dataset to identify the most likely target variable.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Compact layout: Dataset overview and target variable in one row
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    st.markdown("**📊 Dataset Overview**")
                    st.write(f"**Shape:** {uploaded_data.shape[0]} rows × {uploaded_data.shape[1]} columns")
                    st.write(f"**Numeric:** {len(uploaded_analysis['data_characteristics']['numeric_columns'])}")
                    st.write(f"**Categorical:** {len(uploaded_analysis['data_characteristics']['categorical_columns'])}")
                    
                    if uploaded_analysis['data_characteristics']['missing_values']:
                        missing_cols = [col for col, count in uploaded_analysis['data_characteristics']['missing_values'].items() if count > 0]
                        if missing_cols:
                            st.write(f"**Missing values:** {len(missing_cols)} cols")
                
                with col2:
                    st.markdown("**🎯 Target Variable**")
                    if uploaded_analysis['target_variable']:
                        st.success(f"**Target:** {uploaded_analysis['target_variable']}")
                        st.write(f"**Type:** {uploaded_analysis['problem_type']}")
                        st.write(f"**Difficulty:** {uploaded_analysis.get('difficulty', '🟡 Intermediate')}")
                        
                        # Compact target statistics
                        target_col = uploaded_analysis['target_variable']
                        target_data = uploaded_data[target_col]
                        st.write(f"**Data type:** {target_data.dtype}")
                        st.write(f"**Unique values:** {target_data.nunique()}")
                        
                        if target_data.dtype in ['int64', 'float64']:
                            st.write(f"**Range:** {target_data.min():.1f} - {target_data.max():.1f}")
                    else:
                        st.warning("**No target identified**")
                
                with col3:
                    st.markdown("**🔍 Selection Reasoning**")
                    # Show top 3 reasons in compact format
                    for i, reason in enumerate(uploaded_analysis['target_selection_reasoning'][:3], 1):
                        st.markdown(f"{i}. {reason}")
                    
                    if len(uploaded_analysis['target_selection_reasoning']) > 3:
                        with st.expander(f"View all {len(uploaded_analysis['target_selection_reasoning'])} reasons"):
                            for reason in uploaded_analysis['target_selection_reasoning']:
                                st.markdown(f"- {reason}")
                
                # Display Problem Type Analysis Results (NEW)
                if 'problem_analysis' in st.session_state.pipeline_results:
                    st.markdown("### 🔍 Problem Type Analysis Results")
                    
                    problem_analysis = st.session_state.pipeline_results['problem_analysis']
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("**📋 Problem Type Detection**")
                        st.success(f"**Problem Type:** {problem_analysis['problem_type']}")
                        
                        if problem_analysis.get('target_variable'):
                            st.write(f"**Target Variable:** {problem_analysis['target_variable']}")
                            
                            # Target characteristics
                            target_chars = problem_analysis['target_characteristics']
                            st.write(f"**Data Type:** {target_chars['data_type']}")
                            st.write(f"**Unique Values:** {target_chars['unique_values']}")
                            st.write(f"**Missing Values:** {target_chars['missing_count']} ({target_chars['missing_percentage']:.1%})")
                            
                            if target_chars.get('imbalance_warning'):
                                st.warning(f"⚠️ Class Imbalance Detected (Ratio: {target_chars['class_balance_ratio']:.2f})")
                        else:
                            st.info("🔍 Unsupervised Learning Problem")
                    
                    with col2:
                        st.markdown("**⚙️ Cleaning Recommendations**")
                        recommendations = problem_analysis.get('cleaning_recommendations', {})
                        
                        # Target handling
                        target_handling = recommendations.get('target_handling', {})
                        if target_handling:
                            st.write("**Target Handling:**")
                            for key, value in target_handling.items():
                                st.write(f"- {key}: {value}")
                        
                        # Validation strategy
                        validation_strategy = recommendations.get('validation_strategy', {})
                        if validation_strategy:
                            st.write("**Validation Strategy:**")
                            for key, value in validation_strategy.items():
                                st.write(f"- {key}: {value}")
                        
                        # Algorithm recommendations
                        algorithms = recommendations.get('algorithm_recommendations', [])
                        if algorithms:
                            st.write("**Recommended Algorithms:**")
                            for algo in algorithms:
                                st.write(f"- {algo}")
                    
                    # Show decision log
                    if problem_analysis.get('decision_log'):
                        with st.expander("📝 Problem Type Detection Decision Log"):
                            for decision in problem_analysis['decision_log']:
                                st.markdown(f"**{decision['decision_type']}:** {decision['reasoning']}")
                                st.markdown(f"*Action:* {decision['action_taken']}")
                                st.markdown("---")
                
                # Manual target selection in a compact row
                st.markdown("---")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**🔧 Manual Target Selection**")
                    manual_target = st.selectbox(
                        "Select target variable",
                        ["Auto-detected"] + list(uploaded_data.columns),
                        index=0,
                        help="Choose the column that represents your target variable"
                    )
                
                with col2:
                    if manual_target != "Auto-detected":
                        uploaded_analysis['target_variable'] = manual_target
                        st.success(f"✅ Set to: {manual_target}")
                        
                        # Re-analyze with manual selection
                        target_data = uploaded_data[manual_target]
                        if target_data.dtype in ['object', 'category'] or target_data.nunique() < 20:
                            if target_data.nunique() == 2:
                                uploaded_analysis['problem_type'] = "Binary Classification"
                            else:
                                uploaded_analysis['problem_type'] = "Multi-class Classification"
                        else:
                            uploaded_analysis['problem_type'] = "Regression"
                        
                        st.info(f"Type: {uploaded_analysis['problem_type']}")
                
                # Show top candidates in a compact expander
                if 'all_candidates' in uploaded_analysis and len(uploaded_analysis['all_candidates']) > 1:
                    with st.expander("🏆 View All Target Candidates"):
                        for i, candidate in enumerate(uploaded_analysis['all_candidates'], 1):
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.markdown(f"**#{i}: {candidate['column']}**")
                            with col2:
                                st.markdown(f"Score: {candidate['score']}")
                            with col3:
                                st.markdown(f"Type: {candidate['dtype']}")
                            
                            # Show top reason only
                            if candidate['reasons']:
                                st.markdown(f"*{candidate['reasons'][0]}*")
                            st.markdown("---")
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.info("Please make sure your file is a valid CSV format")
    
    else:
        # Sample dataset selection
        st.sidebar.title("Dataset Selection")
    available_datasets = load_sample_datasets()
    selected_dataset = st.sidebar.selectbox(
        "Choose a dataset",
        available_datasets,
        index=0
    )
    
    # Get dataset information
    if uploaded_data is not None:
        dataset_info = get_dataset_info("uploaded", uploaded_data, uploaded_analysis)
    else:
        dataset_info = get_dataset_info(selected_dataset)
    
    # Display dataset information
    with st.expander("📊 About This Dataset & Learning Opportunities"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Difficulty Level:** {dataset_info['difficulty']}")
            st.markdown(f"**Problem Type:** {dataset_info['problem_type']}")
            st.markdown(f"**Description:** {dataset_info['description']}")
            
            st.markdown("**Key Challenges:**")
            for challenge in dataset_info['challenges']:
                st.markdown(f"- {challenge}")
        
        with col2:
            st.markdown("**Learning Objectives:**")
            for objective in dataset_info['learning_objectives']:
                st.markdown(f"- {objective}")
            
            if dataset_info['recommended_config']:
                st.markdown("**Recommended Configuration:**")
                config = dataset_info['recommended_config']
                if 'imputation_method' in config:
                    st.markdown(f"- Imputation: {config['imputation_method']}")
                if 'outlier_handling' in config:
                    st.markdown(f"- Outliers: {config['outlier_handling']}")
                if 'algorithms' in config:
                    st.markdown(f"- Algorithms: {', '.join(config['algorithms'])}")
                if 'cv_strategy' in config:
                    st.markdown(f"- CV Strategy: {config['cv_strategy']}")
                
                if st.button("⚡ Apply Recommended Settings", key="apply_recommended"):
                    st.session_state.recommended_config = config
                    st.success("Recommended settings applied! Check the configuration tabs below.")
    
    # Interactive Configuration
    st.markdown("---")
    config = display_interactive_configuration()
    
    # Initialize session state for pipeline results
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = None
    
    # Interactive Pipeline Execution
    st.markdown("---")
    st.subheader("🚀 Execute ML Pipeline")
    
    # Pipeline progress tracking
    if 'pipeline_progress' not in st.session_state:
        st.session_state.pipeline_progress = {
            'stage': 'ready',
            'progress': 0,
            'current_step': '',
            'logs': []
        }
    
    # Progress bar and status
    progress_col1, progress_col2 = st.columns([3, 1])
    with progress_col1:
        progress_bar = st.progress(0)
        status_text = st.empty()
    with progress_col2:
        stage_badge = st.empty()
    
    # Run pipeline button
    col1, col2 = st.columns([1, 3])
    with col1:
        run_pipeline = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)
    
    with col2:
        if run_pipeline:
            if uploaded_data is not None:
                st.info("💡 Running pipeline on your uploaded dataset with your selected configuration!")
            else:
                st.info("💡 The pipeline will use your selected configuration options. You can compare results with different settings!")
    
    # Pipeline execution with real-time updates
    if run_pipeline:
        # Initialize progress
        st.session_state.pipeline_progress = {
            'stage': 'starting',
            'progress': 0,
            'current_step': 'Initializing pipeline...',
            'logs': []
        }
        
        # Create a container for real-time logs
        log_container = st.container()
        
        with log_container:
            st.subheader("📋 Pipeline Execution Log")
            log_display = st.empty()
        
        # Get OpenAI API key from Streamlit secrets
        try:
            openai_api_key = st.secrets["OPENAI_API_KEY"]
        except:
            st.error("❌ OpenAI API key not found in Streamlit secrets. Please add it in the app settings.")
            return
        
        # Run pipeline in background with progress updates
        try:
            if uploaded_data is not None:
                temp_path = "temp_uploaded_data.csv"
                uploaded_data.to_csv(temp_path, index=False)
                orchestrator = DataPipelineOrchestrator(temp_path, openai_api_key=openai_api_key)
            else:
                orchestrator = DataPipelineOrchestrator(f"sample_data/{selected_dataset}", openai_api_key=openai_api_key)
            
            # Progress callback function
            def update_progress(stage, progress, message):
                st.session_state.pipeline_progress['stage'] = stage
                st.session_state.pipeline_progress['progress'] = progress
                st.session_state.pipeline_progress['current_step'] = message
                st.session_state.pipeline_progress['logs'].append(f"[{stage.upper()}] {message}")
                
                # Note: We can't update Streamlit elements from within the callback
                # The updates will be reflected when the app reruns
            
            # Run pipeline with progress tracking
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
            
            # Clean up temporary file
            if uploaded_data is not None and os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Update progress to completion
            st.session_state.pipeline_progress['stage'] = 'completed'
            st.session_state.pipeline_progress['progress'] = 100
            st.session_state.pipeline_progress['current_step'] = 'Pipeline completed successfully!'
            
            st.success("✅ Pipeline completed successfully!")
            
            # Trigger a rerun to update the display
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Pipeline failed: {str(e)}")
            st.session_state.pipeline_progress['stage'] = 'failed'
        
        # Update progress display from session state
        progress_bar.progress(st.session_state.pipeline_progress['progress'] / 100)
        status_text.text(st.session_state.pipeline_progress['current_step'])
        
        # Update stage badge
        stage_colors = {
            'ready': '🔵',
            'starting': '🟡',
            'problem_detection': '🟡',
            'cleaning': '🟡',
            'eda': '🟡',
            'training': '🟡',
            'evaluation': '🟡',
            'completed': '🟟',
            'failed': '🔴'
        }
        stage_badge.markdown(f"**{stage_colors.get(st.session_state.pipeline_progress['stage'], '⚪')} {st.session_state.pipeline_progress['stage'].replace('_', ' ').title()}**")
        
        # Update log display
        if st.session_state.pipeline_progress['logs']:
            log_display.markdown("\n".join(st.session_state.pipeline_progress['logs'][-10:]))  # Show last 10 logs
    
    # Display results if available
    if st.session_state.pipeline_results:
        st.markdown("---")
        st.subheader("📊 Pipeline Results")
        
        # Create tabs for different pipeline stages
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
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit and CrewAI")

if __name__ == "__main__":
    main() 