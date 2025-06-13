import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import openai
import os
import plotly.express as px
import plotly.graph_objects as go

class EDAAgent:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def perform_eda(self, data: pd.DataFrame) -> dict:
        """Perform comprehensive exploratory data analysis"""
        eda_results = {
            "summary_stats": None,
            "correlations": None,
            "distributions": {},
            "eda_insights": None,
            "target_variable": None,
            "problem_type": None
        }
        
        try:
            print("📊 Generating summary statistics...")
            eda_results["summary_stats"] = self._generate_summary_stats(data)
            
            print("🔍 Analyzing data types...")
            data_types = self._analyze_data_types(data)
            
            print("🔗 Computing correlations...")
            eda_results["correlations"] = self._compute_correlations(data)
            
            print("📈 Creating visualizations...")
            eda_results["distributions"] = self._create_distributions(data)
            
            print("🎯 Detecting target variable...")
            eda_results["target_variable"], eda_results["problem_type"] = self._detect_target_and_problem_type(data)
            
            print("🧠 Generating AI insights...")
            eda_results["eda_insights"] = self._generate_ai_insights(data, eda_results)
            
            return eda_results
            
        except Exception as e:
            print(f"❌ EDA failed: {str(e)}")
            raise
    
    def _generate_summary_stats(self, data: pd.DataFrame) -> dict:
        """Generate summary statistics"""
        numeric_data = data.select_dtypes(include=[np.number])
        categorical_data = data.select_dtypes(include=['object', 'category'])
        
        stats = {
            "numeric": numeric_data.describe().to_dict(),
            "categorical": {col: data[col].value_counts().head().to_dict() 
                          for col in categorical_data.columns},
            "missing_values": data.isnull().sum().to_dict()
        }
        
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