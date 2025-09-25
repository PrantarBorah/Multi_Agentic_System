import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scipy.stats
import openai
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

class EDAAgent:
    def __init__(self, openai_api_key: str = None):
        # Use provided API key, or try Streamlit secrets, or fallback to environment variable
        api_key = openai_api_key
        if not api_key:
            try:
                api_key = st.secrets["OPENAI_API_KEY"]
            except:
                api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set it in Streamlit secrets or environment variables.")
        
        self.openai_client = openai.OpenAI(api_key=api_key)
    
    def perform_eda(self, data: pd.DataFrame, eda_options: dict = None) -> dict:
        """Perform comprehensive exploratory data analysis"""
        eda_results = {
            "summary_stats": None,
            "correlations": None,
            "distributions": {},
            "custom_visualizations": {},
            "custom_insights": {},
            "eda_insights": None,
            "target_variable": None,
            "problem_type": None
        }

        if eda_options is None:
            eda_options = {
                "enable_summary_stats": True,
                "enable_correlations": True,
                "enable_distributions": True,
                "enable_eda_insights": True,
                "custom_visualizations": {
                    "selected_charts": {},
                    "custom_scatter_plots": [],
                    "custom_questions": []
                }
        }
        
        try:
            
            print("📊 Generating summary statistics...")
            if eda_options.get("enable_summary_stats", True):
                eda_results["summary_stats"] = self._generate_summary_stats(data)
            
            print("🔍 Analyzing data types...")
            data_types = self._analyze_data_types(data) # This is always needed for other steps
            
            print("🔗 Computing correlations...")
            if eda_options.get("enable_correlations", True):
                eda_results["correlations"] = self._compute_correlations(data)
            
            print("📈 Creating visualizations...")
            if eda_options.get("enable_distributions", True):
                eda_results["distributions"] = self._create_distributions(data)
            
            # Target variable detection is often needed for model training later
            print("🎯 Detecting target variable...")
            eda_results["target_variable"], eda_results["problem_type"] = self._detect_target_and_problem_type(data)
            
            print("🧠 Generating AI insights...")
            if eda_options.get("enable_eda_insights", True):
                eda_results["eda_insights"] = self._generate_ai_insights(data, eda_results)
            
            # Generate custom visualizations
            print("🎨 Creating custom visualizations...")
            custom_viz_config = eda_options.get("custom_visualizations", {})
            if custom_viz_config.get("selected_charts") or custom_viz_config.get("custom_scatter_plots"):
                eda_results["custom_visualizations"] = self._create_custom_visualizations(data, custom_viz_config)
            
            # Answer custom questions
            print("💬 Answering custom questions...")
            if custom_viz_config.get("custom_questions"):
                eda_results["custom_insights"] = self._answer_custom_questions(data, eda_results, custom_viz_config["custom_questions"])
            
            return eda_results
            
        except Exception as e:
            print(f"❌ EDA failed: {str(e)}")
            raise
    
    def _generate_summary_stats(self, data: pd.DataFrame) -> dict:
        """Generate summary statistics"""
        numeric_data = data.select_dtypes(include=[np.number])
        categorical_data = data.select_dtypes(include=['object', 'category'])
        
        stats = {}
        
        print(f"Debug: numeric_data.empty: {numeric_data.empty}, len(numeric_data.columns): {len(numeric_data.columns)}")

        if not numeric_data.empty and len(numeric_data.columns) > 0:
            stats["numeric"] = numeric_data.describe().to_dict()
        else:
            stats["numeric"] = {"message": "No numeric columns to display summary statistics for."}

        stats["categorical"] = {col: data[col].value_counts().head().to_dict() 
                                 for col in categorical_data.columns}
        stats["missing_values"] = data.isnull().sum().to_dict()
        
        return stats
    
    def _analyze_data_types(self, data: pd.DataFrame) -> dict:
        """Analyze data types and suggest improvements"""
        return {
            "dtypes": {col: str(dtype) for col, dtype in data.dtypes.items()},
            "numeric_columns": data.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": data.select_dtypes(include=['object', 'category']).columns.tolist()
        }
    
    def _compute_correlations(self, data: pd.DataFrame) -> dict:
        """Compute correlation matrix for numeric variables"""
        numeric_data = data.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) > 1:
            corr_matrix = numeric_data.corr()
            return corr_matrix.to_dict()
        else:
            return {}
    
    def _create_distributions(self, data: pd.DataFrame) -> dict:
        """Create distribution plots for numeric variables"""
        distributions = {}
        numeric_data_to_plot = data.select_dtypes(include=[np.number]).drop(columns=['PassengerId'], errors='ignore')

        # Identify columns that are numeric but represent categories (heuristic)
        categorical_numeric_cols = [col for col in numeric_data_to_plot.columns if data[col].nunique() < 10 and data[col].dtype != float]

        for col in numeric_data_to_plot.columns:
            if col in categorical_numeric_cols:
                # Treat as categorical for histogram to ensure discrete bins
                fig = px.histogram(data, x=col, title=f'Distribution of {col}', category_orders={col: sorted(data[col].unique().tolist())})
                fig.update_layout(xaxis=dict(type='category')) # Explicitly set x-axis type to category
            else:
                # For truly continuous numeric data, use default histogram
                fig = px.histogram(data, x=col, title=f'Distribution of {col}', nbins=20)
            distributions[col] = fig.to_dict()
        
        return distributions
    
    def _detect_target_and_problem_type(self, data: pd.DataFrame) -> tuple:
        """Detect likely target variable and problem type"""
        # Simple heuristic: last column is often the target
        target_column = data.columns[-1]
        
        # Determine problem type based on target variable
        if data[target_column].dtype in ['object', 'category']:
            problem_type = "classification"
        elif len(data[target_column].unique()) < 10:
            problem_type = "classification"
        else:
            problem_type = "regression"
        
        return target_column, problem_type
    
    def _generate_ai_insights(self, data: pd.DataFrame, eda_results: dict) -> dict:
        """Generate AI-powered insights about the dataset"""
        try:
            prompt = f"""
            Analyze this dataset and provide key insights:
            - Shape: {data.shape}
            - Columns: {list(data.columns)}
            - Problem type: {eda_results['problem_type']}
            - Target variable: {eda_results['target_variable']}
            
            Provide 3-5 key insights about this dataset for machine learning.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            
            return {
                "insights": response.choices[0].message.content,
                "data_quality_score": self._calculate_data_quality_score(data),
                "recommendations": self._generate_recommendations(eda_results)
            }
            
        except Exception as e:
            return {
                "insights": f"Error generating insights: {str(e)}",
                "data_quality_score": 0.5,
                "recommendations": ["Review data quality", "Check for missing values"]
            }
    
    def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate a simple data quality score"""
        completeness = 1 - (data.isnull().sum().sum() / (data.shape[0] * data.shape[1]))
        return round(completeness, 2)
    
    def _generate_recommendations(self, eda_results: dict) -> list:
        """Generate recommendations based on EDA"""
        recommendations = []
        
        if eda_results['problem_type'] == "classification":
            recommendations.append("Consider using classification algorithms (Random Forest, XGBoost)")
        else:
            recommendations.append("Consider using regression algorithms (Linear Regression, Random Forest)")
        
        return recommendations
    
    def _create_custom_visualizations(self, data: pd.DataFrame, viz_config: dict) -> dict:
        """Create custom visualizations based on user selections"""
        custom_plots = {}
        
        try:
            # Column-specific charts
            selected_charts = viz_config.get("selected_charts", {})
            for column, chart_type in selected_charts.items():
                if column not in data.columns:
                    continue
                
                try:
                    if chart_type == "Auto":
                        # Use automatic chart selection based on data type
                        if pd.api.types.is_numeric_dtype(data[column]):
                            fig = px.histogram(data, x=column, title=f'Distribution of {column}')
                        else:
                            value_counts = data[column].value_counts().head(10)
                            fig = px.bar(x=value_counts.index.tolist(), y=value_counts.values.tolist(),
                                       title=f'Top 10 Values in {column}')
                    
                    elif chart_type == "Histogram":
                        fig = px.histogram(data, x=column, title=f'Histogram of {column}')
                    
                    elif chart_type == "Box Plot":
                        fig = px.box(data, y=column, title=f'Box Plot of {column}')
                    
                    elif chart_type == "Line Plot":
                        fig = px.line(data.reset_index(), x='index', y=column, 
                                    title=f'Line Plot of {column}')
                    
                    elif chart_type == "Density Plot":
                        from scipy.stats import gaussian_kde
                        density = gaussian_kde(data[column].dropna())
                        xs = np.linspace(data[column].min(), data[column].max(), 200)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=xs, y=density(xs), mode='lines', 
                                               name=f'Density of {column}'))
                        fig.update_layout(title=f'Density Plot of {column}')
                    
                    elif chart_type == "Bar Chart":
                        value_counts = data[column].value_counts().head(15)
                        fig = px.bar(x=value_counts.index.tolist(), y=value_counts.values.tolist(),
                                   title=f'Bar Chart of {column}')
                    
                    elif chart_type == "Pie Chart":
                        value_counts = data[column].value_counts().head(10)
                        fig = px.pie(values=value_counts.values.tolist(), names=value_counts.index.tolist(),
                                   title=f'Pie Chart of {column}')
                    
                    elif chart_type == "Count Plot":
                        value_counts = data[column].value_counts()
                        fig = px.bar(x=value_counts.index.tolist(), y=value_counts.values.tolist(),
                                   title=f'Count Plot of {column}')
                    
                    # Convert figure to dict and ensure data types are JSON serializable
                    plot_dict = fig.to_dict()
                    
                    # Debug: Check if there are any problematic data types
                    for trace in plot_dict.get('data', []):
                        if 'x' in trace and isinstance(trace['x'], str):
                            print(f"⚠️ Warning: Found string x data for {column}_{chart_type}: {trace['x']}")
                        if 'y' in trace and isinstance(trace['y'], str):
                            print(f"⚠️ Warning: Found string y data for {column}_{chart_type}: {trace['y']}")
                    
                    custom_plots[f"{column}_{chart_type}"] = plot_dict
                
                except Exception as e:
                    print(f"⚠️ Error creating {chart_type} for {column}: {e}")
            
            # Custom scatter plots
            scatter_plots = viz_config.get("custom_scatter_plots", [])
            for i, scatter_config in enumerate(scatter_plots):
                try:
                    x_col = scatter_config["x"]
                    y_col = scatter_config["y"]
                    
                    if x_col in data.columns and y_col in data.columns:
                        fig = px.scatter(data, x=x_col, y=y_col, 
                                       title=scatter_config["title"])
                        custom_plots[f"scatter_{i}_{x_col}_vs_{y_col}"] = fig.to_dict()
                
                except Exception as e:
                    print(f"⚠️ Error creating scatter plot {i}: {e}")
            
            return custom_plots
            
        except Exception as e:
            print(f"❌ Custom visualization creation failed: {str(e)}")
            return {}
    
    def _answer_custom_questions(self, data: pd.DataFrame, eda_results: dict, questions: list) -> dict:
        """Answer custom questions about the data using AI"""
        insights = {}
        
        try:
            for i, question in enumerate(questions):
                if not question.strip():
                    continue
                
                # Prepare context for the LLM
                context = f"""
                Dataset Information:
                - Shape: {data.shape}
                - Columns: {list(data.columns)}
                - Numeric columns: {data.select_dtypes(include=[np.number]).columns.tolist()}
                - Categorical columns: {data.select_dtypes(include=['object', 'category']).columns.tolist()}
                - Missing values: {data.isnull().sum().to_dict()}
                
                Statistical Summary:
                {data.describe().to_string() if not data.select_dtypes(include=[np.number]).empty else "No numeric data"}
                
                Question: {question}
                
                Please provide a detailed, data-driven answer to this question based on the dataset information provided.
                """
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": context}],
                    max_tokens=400
                )
                
                insights[f"question_{i+1}"] = {
                    "question": question,
                    "answer": response.choices[0].message.content
                }
                
        except Exception as e:
            insights["error"] = f"Error answering custom questions: {str(e)}"
        
        return insights