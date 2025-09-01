import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report, accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
import openai
import os
import json
from datetime import datetime

class EvaluatorAgent:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        # Initialize model storage for comparisons
        if not hasattr(self, 'stored_models'):
            self.stored_models = []
    
    def evaluate_model(self, model_results: dict, data: pd.DataFrame, evaluation_options: dict = None) -> dict:
        """Comprehensive model evaluation and reporting with enhanced metric selection"""
        if evaluation_options is None:
            evaluation_options = {
                "enable_confusion_matrix": True,
                "enable_actual_vs_predicted_plot": True,
                "enable_evaluation_recommendations": True,
                "selected_metrics": [],  # New: specific metrics to focus on
                "store_model_for_comparison": False,  # New: store model for comparison
                "comparison_models": []  # New: models to compare against
            }
        
        evaluation_results = {
            "performance_metrics": {},
            "detailed_metrics": {},  # New: comprehensive metrics breakdown
            "plots": {},
            "model_insights": None,
            "recommendations": [],
            "metric_explanations": {},  # New: educational explanations
            "model_comparison_results": None  # New: comparison results
        }
        
        try:
            print("📊 Computing performance metrics...")
            evaluation_results["performance_metrics"] = self._compute_performance_metrics(model_results)
            print("✅ Performance metrics computed successfully")
            
            print("📈 Computing detailed metrics...")
            evaluation_results["detailed_metrics"] = self._compute_detailed_metrics(
                model_results, evaluation_options.get("selected_metrics", [])
            )
            print("✅ Detailed metrics computed successfully")
            
            print("📚 Generating metric explanations...")
            evaluation_results["metric_explanations"] = self._get_metric_explanations(
                model_results["problem_type"], evaluation_options.get("selected_metrics", [])
            )
            print("✅ Metric explanations generated successfully")
            
            print("📈 Creating evaluation plots...")
            evaluation_results["plots"] = self._create_evaluation_plots(model_results, data, evaluation_options)
            print("✅ Evaluation plots created successfully")
            
            print("💡 Generating model insights...")
            evaluation_results["model_insights"] = self._generate_model_insights(model_results, evaluation_results)
            print("✅ Model insights generated successfully")
            
            print("🎯 Creating recommendations...")
            evaluation_results["recommendations"] = self._generate_recommendations(model_results, evaluation_results, evaluation_options)
            print("✅ Recommendations generated successfully")
            
            # Store model for comparison if requested
            if evaluation_options.get("store_model_for_comparison", False):
                self._store_model_for_comparison(model_results, evaluation_results)
            
            # Perform model comparison if requested
            if evaluation_options.get("comparison_models"):
                print("🔍 Performing model comparison...")
                evaluation_results["model_comparison_results"] = self._compare_models(
                    model_results, evaluation_results, evaluation_options.get("comparison_models", [])
                )
            
            return evaluation_results
            
        except Exception as e:
            print(f"❌ Model evaluation failed: {str(e)}")
            raise
    
    def _compute_performance_metrics(self, model_results: dict) -> dict:
        """Compute comprehensive performance metrics"""
        problem_type = model_results["problem_type"]
        cv_score = model_results["cv_score"]
        model_type = model_results["selected_model"]
        
        # Check if we have training_summary with metrics
        if "training_summary" in model_results and model_results["training_summary"] and "metrics" in model_results["training_summary"]:
            metrics = model_results["training_summary"]["metrics"]
            
            if problem_type == "classification":
                return {
                    "accuracy": metrics.get("accuracy", "N/A"),
                    "classification_report": metrics.get("classification_report", "N/A"),
                    "cv_score": cv_score,
                    "model_type": model_type
                }
            else:
                return {
                    "r2": metrics.get("r2", "N/A"),
                    "mse": metrics.get("mse", "N/A"),
                    "rmse": metrics.get("rmse", "N/A"),
                    "cv_score": cv_score,
                    "model_type": model_type
                }
        else:
            # Fallback for cases without training_summary or metrics
            return {
                "cv_score": cv_score,
                "model_type": model_type,
                "problem_type": problem_type
            }
    
    def _compute_detailed_metrics(self, model_results: dict, selected_metrics: list) -> dict:
        """Compute comprehensive performance metrics with user selection"""
        problem_type = model_results["problem_type"]
        y_true = np.array(model_results["y_test"])
        y_pred = np.array(model_results["predictions"])
        
        detailed_metrics = {}
        
        if problem_type == "classification":
            # Get probabilities if available
            y_prob = model_results.get("prediction_probabilities")
            
            # Core classification metrics
            available_metrics = {
                "accuracy": lambda: accuracy_score(y_true, y_pred),
                "precision_macro": lambda: precision_score(y_true, y_pred, average='macro', zero_division=0),
                "precision_micro": lambda: precision_score(y_true, y_pred, average='micro', zero_division=0),
                "precision_weighted": lambda: precision_score(y_true, y_pred, average='weighted', zero_division=0),
                "recall_macro": lambda: recall_score(y_true, y_pred, average='macro', zero_division=0),
                "recall_micro": lambda: recall_score(y_true, y_pred, average='micro', zero_division=0),
                "recall_weighted": lambda: recall_score(y_true, y_pred, average='weighted', zero_division=0),
                "f1_macro": lambda: f1_score(y_true, y_pred, average='macro', zero_division=0),
                "f1_micro": lambda: f1_score(y_true, y_pred, average='micro', zero_division=0),
                "f1_weighted": lambda: f1_score(y_true, y_pred, average='weighted', zero_division=0),
            }
            
            # Add ROC AUC if probabilities available and binary classification
            if y_prob is not None and len(np.unique(y_true)) == 2:
                available_metrics["roc_auc"] = lambda: roc_auc_score(y_true, y_prob)
            
        else:  # regression
            available_metrics = {
                "r2": lambda: r2_score(y_true, y_pred),
                "mse": lambda: mean_squared_error(y_true, y_pred),
                "rmse": lambda: np.sqrt(mean_squared_error(y_true, y_pred)),
                "mae": lambda: mean_absolute_error(y_true, y_pred),
                "mape": lambda: np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.nan,
                "explained_variance": lambda: 1 - np.var(y_true - y_pred) / np.var(y_true)
            }
        
        # Compute selected metrics or all if none selected
        metrics_to_compute = selected_metrics if selected_metrics else list(available_metrics.keys())
        
        for metric_name in metrics_to_compute:
            if metric_name in available_metrics:
                try:
                    detailed_metrics[metric_name] = float(available_metrics[metric_name]())
                except Exception as e:
                    detailed_metrics[metric_name] = f"Error: {str(e)}"
        
        # Add cross-validation score
        detailed_metrics["cv_score"] = model_results["cv_score"]
        
        return detailed_metrics
    
    def _get_metric_explanations(self, problem_type: str, selected_metrics: list) -> dict:
        """Provide educational explanations for metrics"""
        explanations = {}
        
        if problem_type == "classification":
            metric_explanations = {
                "accuracy": {
                    "definition": "Proportion of correct predictions among total predictions",
                    "formula": "TP + TN / (TP + TN + FP + FN)",
                    "when_to_use": "Good for balanced datasets, can be misleading with imbalanced classes",
                    "range": "0 to 1 (higher is better)"
                },
                "precision_macro": {
                    "definition": "Average precision across all classes (unweighted)",
                    "formula": "Average of precision for each class",
                    "when_to_use": "When all classes are equally important",
                    "range": "0 to 1 (higher is better)"
                },
                "precision_weighted": {
                    "definition": "Weighted average precision by class frequency",
                    "formula": "Weighted average of precision for each class",
                    "when_to_use": "When classes have different importance based on frequency",
                    "range": "0 to 1 (higher is better)"
                },
                "recall_macro": {
                    "definition": "Average recall across all classes (unweighted)",
                    "formula": "Average of recall for each class",
                    "when_to_use": "When detecting all positive cases is important",
                    "range": "0 to 1 (higher is better)"
                },
                "f1_macro": {
                    "definition": "Harmonic mean of precision and recall (macro average)",
                    "formula": "2 * (precision * recall) / (precision + recall)",
                    "when_to_use": "Balance between precision and recall",
                    "range": "0 to 1 (higher is better)"
                },
                "roc_auc": {
                    "definition": "Area Under the ROC Curve - measures classifier performance",
                    "formula": "Area under the curve plotting TPR vs FPR",
                    "when_to_use": "Binary classification, especially with balanced classes",
                    "range": "0 to 1 (0.5 = random, 1 = perfect)"
                }
            }
        else:  # regression
            metric_explanations = {
                "r2": {
                    "definition": "Coefficient of determination - proportion of variance explained",
                    "formula": "1 - (SS_res / SS_tot)",
                    "when_to_use": "Understanding how well model explains variance",
                    "range": "0 to 1 (higher is better, can be negative)"
                },
                "mse": {
                    "definition": "Mean Squared Error - average of squared differences",
                    "formula": "Average of (y_true - y_pred)²",
                    "when_to_use": "Penalizes large errors more heavily",
                    "range": "0 to ∞ (lower is better)"
                },
                "rmse": {
                    "definition": "Root Mean Squared Error - square root of MSE",
                    "formula": "√(MSE)",
                    "when_to_use": "Same units as target variable",
                    "range": "0 to ∞ (lower is better)"
                },
                "mae": {
                    "definition": "Mean Absolute Error - average of absolute differences",
                    "formula": "Average of |y_true - y_pred|",
                    "when_to_use": "Less sensitive to outliers than MSE",
                    "range": "0 to ∞ (lower is better)"
                },
                "mape": {
                    "definition": "Mean Absolute Percentage Error - percentage error",
                    "formula": "Average of |y_true - y_pred| / |y_true| * 100",
                    "when_to_use": "Relative error measurement, interpretable as percentage",
                    "range": "0 to ∞ (lower is better)"
                }
            }
        
        # Return explanations for selected metrics or all if none selected
        metrics_to_explain = selected_metrics if selected_metrics else list(metric_explanations.keys())
        
        for metric in metrics_to_explain:
            if metric in metric_explanations:
                explanations[metric] = metric_explanations[metric]
        
        return explanations
    
    def _store_model_for_comparison(self, model_results: dict, evaluation_results: dict):
        """Store model results for future comparison"""
        model_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": f"{model_results['selected_model']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "model_type": model_results["selected_model"],
            "problem_type": model_results["problem_type"],
            "cv_score": model_results["cv_score"],
            "detailed_metrics": evaluation_results["detailed_metrics"],
            "hyperparameters": model_results.get("hyperparameters_used", {}),
            "algorithm_explanation": model_results.get("algorithm_explanation", ""),
            "feature_importance": model_results.get("feature_importance", [])
        }
        
        self.stored_models.append(model_entry)
        print(f"📦 Model stored for comparison: {model_entry['model_name']}")
    
    def _compare_models(self, current_model: dict, current_evaluation: dict, comparison_model_indices: list) -> dict:
        """Compare current model with stored models"""
        if not self.stored_models:
            return {"error": "No stored models available for comparison"}
        
        comparison_results = {
            "current_model": {
                "name": f"{current_model['selected_model']}_current",
                "type": current_model["selected_model"],
                "cv_score": current_model["cv_score"],
                "metrics": current_evaluation["detailed_metrics"]
            },
            "compared_models": [],
            "best_model": None,
            "comparison_insights": None,
            "metric_comparison_plot": None
        }
        
        # Add comparison models
        for idx in comparison_model_indices:
            if 0 <= idx < len(self.stored_models):
                model = self.stored_models[idx]
                comparison_results["compared_models"].append({
                    "name": model["model_name"],
                    "type": model["model_type"],
                    "cv_score": model["cv_score"],
                    "metrics": model["detailed_metrics"],
                    "timestamp": model["timestamp"]
                })
        
        # Determine best model based on CV score
        all_models = [comparison_results["current_model"]] + comparison_results["compared_models"]
        comparison_results["best_model"] = max(all_models, key=lambda x: x["cv_score"])
        
        # Generate comparison insights
        comparison_results["comparison_insights"] = self._generate_comparison_insights(all_models, current_model["problem_type"])
        
        # Create comparison plot
        comparison_results["metric_comparison_plot"] = self._create_model_comparison_plot(all_models)
        
        return comparison_results
    
    def _generate_comparison_insights(self, models: list, problem_type: str) -> str:
        """Generate AI insights about model comparison"""
        try:
            models_summary = []
            for model in models:
                model_info = f"- {model['name']} ({model['type']}): CV Score = {model['cv_score']:.4f}"
                models_summary.append(model_info)
            
            prompt = f"""
            Compare these {problem_type} models and provide insights:
            
            Models compared:
            {chr(10).join(models_summary)}
            
            Provide a brief analysis covering:
            1. Which model performs best and why
            2. Key performance differences between models
            3. Recommendations for model selection
            4. When to use each model type
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating comparison insights: {str(e)}"
    
    def _create_model_comparison_plot(self, models: list) -> str:
        """Create a comparison plot for multiple models"""
        try:
            model_names = [model["name"] for model in models]
            cv_scores = [model["cv_score"] for model in models]
            model_types = [model["type"] for model in models]
            
            fig = go.Figure()
            
            # Create bar plot
            fig.add_trace(go.Bar(
                x=model_names,
                y=cv_scores,
                text=[f"{score:.4f}" for score in cv_scores],
                textposition='auto',
                name='CV Score',
                hovertemplate='<b>%{x}</b><br>CV Score: %{y:.4f}<br>Type: %{customdata}<extra></extra>',
                customdata=model_types
            ))
            
            fig.update_layout(
                title='Model Performance Comparison',
                xaxis_title='Models',
                yaxis_title='Cross-Validation Score',
                showlegend=False,
                height=500
            )
            
            return fig.to_json()
            
        except Exception as e:
            print(f"⚠️ Warning: Model comparison plot creation failed: {str(e)}")
            return None
    
    def _create_evaluation_plots(self, model_results: dict, data: pd.DataFrame, evaluation_options: dict = None) -> dict:
        """Create evaluation visualizations using Plotly"""
        plots = {}
        
        if evaluation_options is None:
            evaluation_options = {
                "enable_confusion_matrix": True,
                "enable_actual_vs_predicted_plot": True,
                "enable_evaluation_recommendations": True
            }

        try:
            # Feature importance plot (always generated if available)
            if model_results.get("feature_importance") and len(model_results["feature_importance"]) > 0:
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
            
            # Model comparison plot (always generated if available)
            if model_results.get("model_comparison") and len(model_results["model_comparison"]) > 0:
                comparison_df = pd.DataFrame(model_results["model_comparison"])
                fig = px.bar(
                    comparison_df,
                    x='model',
                    y='cv_score',
                    title='Model Comparison (CV Scores)',
                    labels={'model': 'Model Type', 'cv_score': 'Cross-Validation Score'}
                )
                plots["model_comparison"] = fig.to_json()
            
            # Check if we have the required data for plots
            if "training_summary" in model_results and model_results["training_summary"] and "metrics" in model_results["training_summary"]:
                metrics = model_results["training_summary"]["metrics"]
                if model_results["problem_type"] == "classification":
                    # Create confusion matrix plot if enabled
                    if evaluation_options.get("enable_confusion_matrix", True):
                        try:
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
                        except Exception as e:
                            print(f"⚠️ Confusion matrix creation failed: {str(e)}")
                else:
                    # Create actual vs predicted plot if enabled (for regression)
                    if evaluation_options.get("enable_actual_vs_predicted_plot", True):
                        try:
                            fig = px.scatter(
                                x=model_results["y_test"],
                                y=model_results["predictions"],
                                title='Actual vs Predicted Values',
                                labels={'x': 'Actual', 'y': 'Predicted'}
                            )
                            fig.add_trace(go.Scatter(x=[min(model_results["y_test"]), max(model_results["y_test"])], y=[min(model_results["y_test"]), max(model_results["y_test"])], mode='lines', name='Ideal'))
                            plots["actual_vs_predicted"] = fig.to_json()
                        except Exception as e:
                            print(f"⚠️ Actual vs predicted plot creation failed: {str(e)}")
            
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
    
    def _generate_recommendations(self, model_results: dict, evaluation_results: dict, evaluation_options: dict = None) -> list:
        """Generate actionable recommendations"""
        if evaluation_options is None:
            evaluation_options = {
                "enable_confusion_matrix": True,
                "enable_actual_vs_predicted_plot": True,
                "enable_evaluation_recommendations": True
            }
        
        if not evaluation_options.get("enable_evaluation_recommendations", True):
            return ["Recommendations disabled by user."]

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
    
    def get_available_metrics(self, problem_type: str) -> list:
        """Get list of available metrics for the problem type"""
        if problem_type == "classification":
            return [
                "accuracy", "precision_macro", "precision_micro", "precision_weighted",
                "recall_macro", "recall_micro", "recall_weighted",
                "f1_macro", "f1_micro", "f1_weighted", "roc_auc"
            ]
        else:  # regression
            return ["r2", "mse", "rmse", "mae", "mape", "explained_variance"]
    
    def get_stored_models_summary(self) -> list:
        """Get summary of stored models for UI display"""
        return [
            {
                "index": idx,
                "name": model["model_name"],
                "type": model["model_type"],
                "cv_score": model["cv_score"],
                "timestamp": model["timestamp"],
                "problem_type": model["problem_type"]
            }
            for idx, model in enumerate(self.stored_models)
        ]