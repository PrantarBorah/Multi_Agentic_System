import streamlit as st
import pandas as pd
import json
import os
from app import DataPipelineOrchestrator
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="ML Pipeline Orchestrator",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .step-container {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

def load_sample_datasets():
    """Load available sample datasets"""
    sample_data_dir = Path("sample_data")
    return [f.name for f in sample_data_dir.glob("*.csv")]

def display_cleaning_results(results):
    """Display data cleaning results"""
    st.subheader("📊 Data Cleaning Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Display shapes
        if "original_shape" in results:
            st.write("Original Data Shape:", results["original_shape"])
        if "cleaned_shape" in results:
            st.write("Cleaned Data Shape:", results["cleaned_shape"])
        
        # Display missing values info
        if "missing_values_info" in results:
            st.write("Missing Values Handling:")
            st.json(results["missing_values_info"])
    
    with col2:
        # Display data type fixes
        if "data_type_fixes" in results:
            st.write("Data Type Fixes:")
            st.json(results["data_type_fixes"])
        
        # Display cleaning summary
        if "cleaning_summary" in results:
            st.write("Cleaning Summary:")
            st.write(results["cleaning_summary"])

def display_eda_results(results):
    """Display EDA results"""
    st.subheader("📈 Exploratory Data Analysis")
    
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
    
    # Sidebar for dataset selection
    st.sidebar.title("Dataset Selection")
    available_datasets = load_sample_datasets()
    selected_dataset = st.sidebar.selectbox(
        "Choose a dataset",
        available_datasets,
        index=0
    )
    
    # Initialize session state for pipeline results
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = None
    
    # Run pipeline button
    if st.sidebar.button("Run Pipeline", type="primary"):
        with st.spinner("Running pipeline... This may take a few minutes."):
            # Initialize and run pipeline
            orchestrator = DataPipelineOrchestrator(f"sample_data/{selected_dataset}")
            orchestrator.run_pipeline()
            
            # Load results
            with open("pipeline_results.json", "r") as f:
                st.session_state.pipeline_results = json.load(f)
    
    # Display results if available
    if st.session_state.pipeline_results:
        # Create tabs for different pipeline stages
        tab1, tab2, tab3, tab4 = st.tabs([
            "Data Cleaning", "EDA", "Model Training", "Evaluation"
        ])
        
        with tab1:
            if "cleaning_summary_results" in st.session_state.pipeline_results:
                display_cleaning_results(st.session_state.pipeline_results["cleaning_summary_results"])
        
        with tab2:
            if "eda_results" in st.session_state.pipeline_results:
                display_eda_results(st.session_state.pipeline_results["eda_results"])
        
        with tab3:
            if "model_results" in st.session_state.pipeline_results:
                display_model_results(st.session_state.pipeline_results["model_results"])
        
        with tab4:
            if "evaluation_results" in st.session_state.pipeline_results:
                display_evaluation_results(st.session_state.pipeline_results["evaluation_results"])
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit and CrewAI")

if __name__ == "__main__":
    main() 