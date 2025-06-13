import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import openai
import os

class EvaluatorAgent:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def evaluate_model(self, model_results: dict, data: pd.DataFrame) -> dict:
        """Comprehensive model evaluation and reporting"""
        evaluation_results = {
            "performance_metrics": {},
            "plots": {},
            "model_insights": None,
            "recommendations": []
        }
        
        try:
            print("📊 Computing performance metrics...")
            evaluation_results["performance_metrics"] = self._compute_performance_metrics(model_results)
            
            print("📈 Creating evaluation plots...")
            evaluation_results["plots"] = self._create_evaluation_plots(model_results, data)
            
            print("💡 Generating model insights...")
            evaluation_results["model_insights"] = self._generate_model_insights(model_results, evaluation_results)
            
            print("🎯 Creating recommendations...")
            evaluation_results["recommendations"] = self._generate_recommendations(model_results, evaluation_results)
            
            return evaluation_results
            
        except Exception as e:
            print(f"❌ Model evaluation failed: {str(e)}")
            raise
    
    def _compute_performance_metrics(self, model_results: dict) -> dict:
        """Compute comprehensive performance metrics"""
        metrics = model_results["training_summary"]["metrics"]
        problem_type = model_results["problem_type"]
        
        if problem_type == "classification":
            return {
                "accuracy": metrics["accuracy"],
                "classification_report": metrics["classification_report"],
                "cv_score": model_results["cv_score"],
                "model_type": model_results["selected_model"]
            }
        else:
            return {
                "r2": metrics["r2"],
                "mse": metrics["mse"],
                "rmse": metrics["rmse"],
                "cv_score": model_results["cv_score"],
                "model_type": model_results["selected_model"]
            }
    
    def _create_evaluation_plots(self, model_results: dict, data: pd.DataFrame) -> dict:
        """Create evaluation visualizations using Plotly"""
        plots = {}
        
        try:
            # Feature importance plot
            if model_results["feature_importance"]:
                importance_df = pd.DataFrame(model_results["feature_importance"])
                fig = px.bar(
                    importance_df,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Feature Importance',
                    labels={'importance': 'Importance Score', 'feature': 'Features'}
                )
                plots["feature_importance"] = fig.to_json()
            
            # Model comparison plot
            if model_results["model_comparison"]:
                comparison_df = pd.DataFrame(model_results["model_comparison"])
                fig = px.bar(
                    comparison_df,
                    x='model',
                    y='cv_score',
                    title='Model Comparison (CV Scores)',
                    labels={'model': 'Model Type', 'cv_score': 'Cross-Validation Score'}
                )
                plots["model_comparison"] = fig.to_json()
            
            # Performance metrics plot
            metrics = model_results["training_summary"]["metrics"]
            if model_results["problem_type"] == "classification":
                # Create confusion matrix plot
                cm = confusion_matrix(model_results["y_test"], model_results["predictions"])
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=[str(label) for label in np.unique(model_results["y_test"])],
                    y=[str(label) for label in np.unique(model_results["y_test"])],
                    title='Confusion Matrix',
                    text_auto=True
                )
                plots["confusion_matrix"] = fig.to_json()
            else:
                # Create actual vs predicted plot
                fig = px.scatter(
                    x=model_results["y_test"],
                    y=model_results["predictions"],
                    title='Actual vs Predicted Values',
                    labels={'x': 'Actual', 'y': 'Predicted'}
                )
                fig.add_trace(go.Scatter(x=[min(model_results["y_test"]), max(model_results["y_test"])], y=[min(model_results["y_test"]), max(model_results["y_test"])], mode='lines', name='Ideal'))
                plots["actual_vs_predicted"] = fig.to_json()
            
            return plots
            
        except Exception as e:
            print(f"⚠️ Warning: Evaluation plot creation failed: {str(e)}")
            return {}
    
    def _generate_model_insights(self, model_results: dict, evaluation_results: dict) -> str:
        """Generate AI-powered model insights"""
        try:
            metrics = evaluation_results["performance_metrics"]
            problem_type = model_results["problem_type"]
            
            prompt = f"""
            Analyze this {problem_type} model performance:
            
            Model Details:
            - Type: {metrics['model_type']}
            - CV Score: {metrics['cv_score']:.4f}
            
            Performance Metrics:
            {metrics}
            
            Provide a brief analysis of the model's performance, highlighting:
            1. Overall effectiveness
            2. Key strengths
            3. Potential areas for improvement
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating insights: {str(e)}"
    
    def _generate_recommendations(self, model_results: dict, evaluation_results: dict) -> list:
        """Generate actionable recommendations"""
        recommendations = []
        metrics = evaluation_results["performance_metrics"]
        
        # General recommendations based on CV score
        cv_score = metrics["cv_score"]
        if cv_score < 0.6:
            recommendations.extend([
                "🔄 Consider collecting more training data",
                "🛠️ Try different algorithms (XGBoost, Neural Networks)",
                "🔧 Perform extensive feature engineering",
                "📊 Review data quality and preprocessing steps"
            ])
        elif cv_score < 0.8:
            recommendations.extend([
                "🔍 Analyze feature importance for feature selection",
                "⚙️ Perform hyperparameter tuning",
                "🔄 Try ensemble methods",
                "📈 Consider feature interactions"
            ])
        else:
            recommendations.extend([
                "📊 Monitor model performance over time",
                "🔍 Analyze prediction errors",
                "📈 Consider model interpretability",
                "🔄 Plan for model retraining"
            ])
        
        return recommendations