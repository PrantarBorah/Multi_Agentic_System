import os
import dotenv
import logging
import pandas as pd
from agents.eda_agent import EDAAgent
from agents.model_trainer_agent import ModelTrainerAgent
from agents.evaluator_agent import EvaluatorAgent

def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("evaluator_agent_test.log"),
            logging.StreamHandler()
        ]
    )

def main():
    # Load environment and cleaned data
    dotenv.load_dotenv()
    data_path = "intermediate_data/cleaned_titanic.csv"
    df = pd.read_csv(data_path)
    logging.info(f"✅ Loaded cleaned data with shape: {df.shape}")

    # Prepare cleaned_state
    cleaned_state = {"data": df}

    # 1. EDA step
    logging.info("🔍 Running EDAAgent.perform_eda()")
    eda_agent = EDAAgent()
    eda_results = eda_agent.perform_eda(cleaned_state)
    logging.info(f"   • EDA completed with target: {eda_results['target_variable']} and problem type: {eda_results['problem_type']}")

    # 2. Model training step
    logging.info("🎓 Running ModelTrainerAgent.train_model()")
    trainer = ModelTrainerAgent()
    model_results = trainer.train_model(cleaned_state, eda_results)
    logging.info(f"   • Trained {model_results['model_type']} model. Metrics: {model_results['training_metrics']}")

    # 3. Evaluation step
    logging.info("📊 Running EvaluatorAgent.evaluate_model()")
    evaluator = EvaluatorAgent()
    eval_results = evaluator.evaluate_model(model_results, cleaned_state)
    logging.info(f"   • Performance Summary: {eval_results['performance_summary']}")
    logging.info("   • Final AI Report:\n" + eval_results['final_report']['ai_generated_report'])
    logging.info(f"   • Recommendations: {eval_results['recommendations']}")

    logging.info("✅ EvaluatorAgent test completed successfully")

if __name__ == "__main__":
    configure_logging()
    main()













# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.linear_model import LogisticRegression, LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import make_classification, make_regression
# from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
# import openai
# import os
# from typing import Dict, List, Any
# import warnings
# warnings.filterwarnings('ignore')

# class EvaluatorAgent:
#     def __init__(self):
#         # Make OpenAI client optional for testing
#         self.openai_client = None
#         try:
#             if os.getenv('OPENAI_API_KEY'):
#                 self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
#         except Exception:
#             print("⚠️ OpenAI client not available - using mock responses")
    
#     def evaluate_model(self, model_results: dict, cleaned_state: dict) -> dict:
#         """Comprehensive model evaluation and reporting"""
#         evaluation_results = {
#             "performance_summary": {},
#             "visualizations": [],
#             "detailed_metrics": {},
#             "final_report": {},
#             "recommendations": []
#         }
        
#         try:
#             print("📊 Generating performance summary...")
#             evaluation_results["performance_summary"] = self._create_performance_summary(model_results)
            
#             print("📈 Creating evaluation visualizations...")
#             evaluation_results["visualizations"] = self._create_evaluation_plots(model_results, cleaned_state)
            
#             print("🔍 Computing detailed metrics...")
#             evaluation_results["detailed_metrics"] = self._compute_detailed_metrics(model_results)
            
#             print("📋 Generating final report...")
#             evaluation_results["final_report"] = self._generate_final_report(model_results, evaluation_results)
            
#             print("💡 Creating recommendations...")
#             evaluation_results["recommendations"] = self._generate_recommendations(model_results, evaluation_results)
            
#             return evaluation_results
            
#         except Exception as e:
#             print(f"❌ Model evaluation failed: {str(e)}")
#             raise
    
#     def _create_performance_summary(self, model_results: dict) -> dict:
#         """Create a summary of model performance"""
#         metrics = model_results["training_metrics"]
#         problem_type = model_results["problem_type"]
        
#         if problem_type == "classification":
#             return {
#                 "model_type": model_results["model_type"],
#                 "problem_type": problem_type,
#                 "accuracy": metrics["test_accuracy"],
#                 "performance_tier": model_results.get("ai_insights", {}).get("performance_tier", "Unknown"),
#                 "features_count": len(model_results["features"]),
#                 "target_variable": model_results["target"]
#             }
#         else:
#             return {
#                 "model_type": model_results["model_type"],
#                 "problem_type": problem_type,
#                 "r2_score": metrics["test_r2"],
#                 "rmse": metrics["test_rmse"],
#                 "performance_tier": model_results.get("ai_insights", {}).get("performance_tier", "Unknown"),
#                 "features_count": len(model_results["features"]),
#                 "target_variable": model_results["target"]
#             }
    
#     def _create_evaluation_plots(self, model_results: dict, cleaned_state: dict) -> list:
#         """Create evaluation visualizations"""
#         visualizations = []
        
#         try:
#             plt.style.use('seaborn-v0_8')
            
#             # Performance metrics visualization
#             metrics = model_results["training_metrics"]
#             problem_type = model_results["problem_type"]
            
#             if problem_type == "classification":
#                 # Classification metrics plot
#                 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
#                 # Accuracy comparison
#                 train_acc = metrics["train_accuracy"]
#                 test_acc = metrics["test_accuracy"]
                
#                 ax1.bar(['Training', 'Testing'], [train_acc, test_acc], 
#                        color=['skyblue', 'lightcoral'])
#                 ax1.set_title('Model Accuracy Comparison')
#                 ax1.set_ylabel('Accuracy')
#                 ax1.set_ylim(0, 1)
                
#                 # Add value labels on bars
#                 for i, v in enumerate([train_acc, test_acc]):
#                     ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
                
#                 # Feature importance (if available)
#                 if hasattr(model_results["model"], 'feature_importances_'):
#                     importances = model_results["model"].feature_importances_
#                     features = model_results["features"][:10]  # Top 10 features
#                     importances = importances[:10]
                    
#                     ax2.barh(features, importances, color='lightgreen')
#                     ax2.set_title('Top 10 Feature Importances')
#                     ax2.set_xlabel('Importance')
#                 else:
#                     ax2.text(0.5, 0.5, 'Feature importance\nnot available', 
#                             ha='center', va='center', transform=ax2.transAxes)
#                     ax2.set_title('Feature Importance')
                
#                 plt.tight_layout()
#                 plt.savefig('classification_evaluation.png', dpi=300, bbox_inches='tight')
#                 plt.close()
#                 visualizations.append("classification_evaluation.png")
                
#             else:
#                 # Regression metrics plot
#                 fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
#                 # R² and RMSE comparison
#                 train_r2 = metrics["train_r2"]
#                 test_r2 = metrics["test_r2"]
#                 train_rmse = metrics["train_rmse"]
#                 test_rmse = metrics["test_rmse"]
                
#                 # R² scores
#                 ax1.bar(['Training', 'Testing'], [train_r2, test_r2], 
#                        color=['skyblue', 'lightcoral'])
#                 ax1.set_title('R² Score Comparison')
#                 ax1.set_ylabel('R² Score')
#                 ax1.set_ylim(0, 1)
                
#                 for i, v in enumerate([train_r2, test_r2]):
#                     ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
                
#                 # RMSE scores
#                 ax2.bar(['Training', 'Testing'], [train_rmse, test_rmse], 
#                        color=['lightgreen', 'orange'])
#                 ax2.set_title('RMSE Comparison')
#                 ax2.set_ylabel('RMSE')
                
#                 for i, v in enumerate([train_rmse, test_rmse]):
#                     ax2.text(i, v + (max(train_rmse, test_rmse) * 0.01), f'{v:.3f}', 
#                             ha='center', va='bottom')
                
#                 plt.tight_layout()
#                 plt.savefig('regression_evaluation.png', dpi=300, bbox_inches='tight')
#                 plt.close()
#                 visualizations.append("regression_evaluation.png")
            
#             return visualizations
            
#         except Exception as e:
#             print(f"⚠️ Warning: Evaluation plot creation failed: {str(e)}")
#             return []
    
#     def _compute_detailed_metrics(self, model_results: dict) -> dict:
#         """Compute additional detailed metrics"""
#         metrics = model_results["training_metrics"]
#         problem_type = model_results["problem_type"]
        
#         detailed_metrics = {
#             "basic_metrics": metrics,
#             "model_complexity": {
#                 "features_used": len(model_results["features"]),
#                 "model_type": model_results["model_type"]
#             }
#         }
        
#         if problem_type == "classification":
#             # Add classification-specific metrics
#             detailed_metrics["classification_details"] = {
#                 "accuracy_difference": metrics["train_accuracy"] - metrics["test_accuracy"],
#                 "overfitting_indicator": "High" if (metrics["train_accuracy"] - metrics["test_accuracy"]) > 0.1 else "Low"
#             }
#         else:
#             # Add regression-specific metrics
#             detailed_metrics["regression_details"] = {
#                 "r2_difference": metrics["train_r2"] - metrics["test_r2"],
#                 "rmse_ratio": metrics["test_rmse"] / metrics["train_rmse"] if metrics["train_rmse"] > 0 else 1,
#                 "overfitting_indicator": "High" if (metrics["train_r2"] - metrics["test_r2"]) > 0.1 else "Low"
#             }
        
#         return detailed_metrics
    
#     def _generate_final_report(self, model_results: dict, evaluation_results: dict) -> dict:
#         """Generate comprehensive final report using AI or mock response"""
#         try:
#             performance_summary = evaluation_results["performance_summary"]
#             detailed_metrics = evaluation_results["detailed_metrics"]
            
#             if self.openai_client:
#                 prompt = f"""
#                 Generate a comprehensive machine learning model evaluation report:
                
#                 Model Details:
#                 - Type: {performance_summary['model_type']}
#                 - Problem: {performance_summary['problem_type']}
#                 - Features: {performance_summary['features_count']}
#                 - Target: {performance_summary['target_variable']}
                
#                 Performance:
#                 - Performance Tier: {performance_summary['performance_tier']}
#                 - Overfitting: {detailed_metrics.get('classification_details', detailed_metrics.get('regression_details', {})).get('overfitting_indicator', 'Unknown')}
                
#                 Create a professional summary with:
#                 1. Executive Summary
#                 2. Model Performance Analysis
#                 3. Key Findings
#                 4. Recommendations
#                 """
                
#                 response = self.openai_client.chat.completions.create(
#                     model="gpt-4o",
#                     messages=[{"role": "user", "content": prompt}],
#                     max_tokens=500
#                 )
                
#                 report_content = response.choices[0].message.content
#             else:
#                 # Mock response when OpenAI is not available
#                 problem_type = performance_summary['problem_type']
#                 model_type = performance_summary['model_type']
                
#                 if problem_type == "classification":
#                     accuracy = performance_summary['accuracy']
#                     report_content = f"""
# **EXECUTIVE SUMMARY**
# The {model_type} classification model achieved {accuracy:.3f} accuracy on test data with {performance_summary['features_count']} features.

# **MODEL PERFORMANCE ANALYSIS**
# - Model Type: {model_type}
# - Problem Type: Classification
# - Test Accuracy: {accuracy:.3f}
# - Performance Tier: {performance_summary['performance_tier']}
# - Overfitting Level: {detailed_metrics.get('classification_details', {}).get('overfitting_indicator', 'Unknown')}

# **KEY FINDINGS**
# - The model shows {'good generalization' if detailed_metrics.get('classification_details', {}).get('overfitting_indicator') == 'Low' else 'potential overfitting'}
# - Feature set includes {performance_summary['features_count']} variables
# - Performance tier indicates {performance_summary['performance_tier'].lower()} model quality

# **RECOMMENDATIONS**
# - {'Continue with deployment preparation' if performance_summary['performance_tier'] in ['Good', 'Excellent'] else 'Consider model improvements'}
# - Monitor model performance in production
# - Implement regular model retraining schedule
#                     """
#                 else:
#                     r2 = performance_summary['r2_score']
#                     rmse = performance_summary['rmse']
#                     report_content = f"""
# **EXECUTIVE SUMMARY**
# The {model_type} regression model achieved R² score of {r2:.3f} and RMSE of {rmse:.3f} on test data.

# **MODEL PERFORMANCE ANALYSIS**
# - Model Type: {model_type}
# - Problem Type: Regression
# - Test R² Score: {r2:.3f}
# - Test RMSE: {rmse:.3f}
# - Performance Tier: {performance_summary['performance_tier']}

# **KEY FINDINGS**
# - The model explains {r2*100:.1f}% of variance in the target variable
# - RMSE indicates {'good' if r2 > 0.7 else 'moderate' if r2 > 0.5 else 'poor'} prediction accuracy
# - {'Low' if detailed_metrics.get('regression_details', {}).get('overfitting_indicator') == 'Low' else 'High'} overfitting detected

# **RECOMMENDATIONS**
# - {'Model ready for production' if r2 > 0.7 else 'Consider model improvements'}
# - Implement prediction interval estimation
# - Set up model monitoring pipeline
#                     """
            
#             return {
#                 "ai_generated_report": report_content,
#                 "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 "model_summary": performance_summary,
#                 "evaluation_status": "Completed Successfully"
#             }
            
#         except Exception as e:
#             return {
#                 "ai_generated_report": f"Error generating report: {str(e)}",
#                 "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
#                 "model_summary": evaluation_results["performance_summary"],
#                 "evaluation_status": "Completed with Errors"
#             }
    
#     def _generate_recommendations(self, model_results: dict, evaluation_results: dict) -> list:
#         """Generate actionable recommendations"""
#         recommendations = []
        
#         performance_tier = evaluation_results["performance_summary"]["performance_tier"]
#         detailed_metrics = evaluation_results["detailed_metrics"]
        
#         # General recommendations based on performance
#         if performance_tier == "Poor":
#             recommendations.extend([
#                 "🔄 Consider collecting more training data",
#                 "🛠️ Try different algorithms (XGBoost, Neural Networks)",
#                 "🔧 Perform extensive feature engineering",
#                 "📊 Review data quality and preprocessing steps"
#             ])
#         elif performance_tier == "Fair":
#             recommendations.extend([
#                 "⚙️ Tune hyperparameters using GridSearch or RandomSearch",
#                 "🎯 Try ensemble methods to improve performance",
#                 "📈 Consider feature selection techniques",
#                 "🔍 Analyze prediction errors for insights"
#             ])
#         elif performance_tier == "Good":
#             recommendations.extend([
#                 "✨ Fine-tune hyperparameters for optimal performance",
#                 "🚀 Consider model deployment preparation",
#                 "📝 Document model for production use",
#                 "🔄 Set up model monitoring pipeline"
#             ])
#         else:  # Excellent
#             recommendations.extend([
#                 "🎉 Model is ready for production deployment",
#                 "📊 Implement model monitoring and drift detection",
#                 "🔄 Set up automated retraining pipeline",
#                 "📋 Create comprehensive model documentation"
#             ])
        
#         # Overfitting recommendations
#         overfitting = detailed_metrics.get('classification_details', detailed_metrics.get('regression_details', {})).get('overfitting_indicator', 'Unknown')
#         if overfitting == "High":
#             recommendations.extend([
#                 "⚠️ Address overfitting with regularization techniques",
#                 "📉 Reduce model complexity or feature count",
#                 "🔄 Increase training data if possible",
#                 "✂️ Apply cross-validation more rigorously"
#             ])
        
#         return recommendations[:8]  # Limit to top 8 recommendations


# def create_mock_classification_data():
#     """Create mock classification dataset and model results"""
#     print("🔧 Creating mock classification data...")
    
#     # Generate synthetic classification data
#     X, y = make_classification(
#         n_samples=1000, 
#         n_features=10, 
#         n_informative=8, 
#         n_redundant=2, 
#         random_state=42
#     )
    
#     # Create feature names
#     feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
    
#     # Train model
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
    
#     # Get predictions
#     train_pred = model.predict(X_train)
#     test_pred = model.predict(X_test)
    
#     # Calculate metrics
#     train_accuracy = accuracy_score(y_train, train_pred)
#     test_accuracy = accuracy_score(y_test, test_pred)
    
#     # Determine performance tier
#     if test_accuracy >= 0.9:
#         performance_tier = "Excellent"
#     elif test_accuracy >= 0.8:
#         performance_tier = "Good"
#     elif test_accuracy >= 0.7:
#         performance_tier = "Fair"
#     else:
#         performance_tier = "Poor"
    
#     # Create model results structure
#     model_results = {
#         "model": model,
#         "model_type": "RandomForestClassifier",
#         "problem_type": "classification",
#         "features": feature_names,
#         "target": "target_class",
#         "training_metrics": {
#             "train_accuracy": train_accuracy,
#             "test_accuracy": test_accuracy
#         },
#         "ai_insights": {
#             "performance_tier": performance_tier
#         }
#     }
    
#     # Create cleaned state (mock)
#     cleaned_state = {
#         "data_shape": X.shape,
#         "preprocessing_steps": ["StandardScaler", "train_test_split"],
#         "target_distribution": np.bincount(y).tolist()
#     }
    
#     return model_results, cleaned_state


# def create_mock_regression_data():
#     """Create mock regression dataset and model results"""
#     print("🔧 Creating mock regression data...")
    
#     # Generate synthetic regression data
#     X, y = make_regression(
#         n_samples=1000, 
#         n_features=8, 
#         noise=0.1, 
#         random_state=42
#     )
    
#     # Create feature names
#     feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )
    
#     # Train model
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
    
#     # Get predictions
#     train_pred = model.predict(X_train)
#     test_pred = model.predict(X_test)
    
#     # Calculate metrics
#     train_r2 = r2_score(y_train, train_pred)
#     test_r2 = r2_score(y_test, test_pred)
#     train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
#     test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
#     # Determine performance tier
#     if test_r2 >= 0.9:
#         performance_tier = "Excellent"
#     elif test_r2 >= 0.8:
#         performance_tier = "Good"
#     elif test_r2 >= 0.6:
#         performance_tier = "Fair"
#     else:
#         performance_tier = "Poor"
    
#     # Create model results structure
#     model_results = {
#         "model": model,
#         "model_type": "RandomForestRegressor",
#         "problem_type": "regression",
#         "features": feature_names,
#         "target": "target_value",
#         "training_metrics": {
#             "train_r2": train_r2,
#             "test_r2": test_r2,
#             "train_rmse": train_rmse,
#             "test_rmse": test_rmse
#         },
#         "ai_insights": {
#             "performance_tier": performance_tier
#         }
#     }
    
#     # Create cleaned state (mock)
#     cleaned_state = {
#         "data_shape": X.shape,
#         "preprocessing_steps": ["StandardScaler", "train_test_split"],
#         "target_stats": {
#             "mean": float(np.mean(y)),
#             "std": float(np.std(y)),
#             "min": float(np.min(y)),
#             "max": float(np.max(y))
#         }
#     }
    
#     return model_results, cleaned_state


# def print_evaluation_results(results: Dict[str, Any]):
#     """Pretty print evaluation results"""
#     print("\n" + "="*60)
#     print("📊 EVALUATION RESULTS SUMMARY")
#     print("="*60)
    
#     # Performance Summary
#     summary = results["performance_summary"]
#     print(f"\n🎯 MODEL OVERVIEW:")
#     print(f"   • Type: {summary['model_type']}")
#     print(f"   • Problem: {summary['problem_type'].title()}")
#     print(f"   • Features: {summary['features_count']}")
#     print(f"   • Target: {summary['target_variable']}")
#     print(f"   • Performance Tier: {summary['performance_tier']}")
    
#     if summary['problem_type'] == 'classification':
#         print(f"   • Test Accuracy: {summary['accuracy']:.3f}")
#     else:
#         print(f"   • Test R² Score: {summary['r2_score']:.3f}")
#         print(f"   • Test RMSE: {summary['rmse']:.3f}")
    
#     # Detailed Metrics
#     print(f"\n🔍 DETAILED ANALYSIS:")
#     detailed = results["detailed_metrics"]
#     if 'classification_details' in detailed:
#         print(f"   • Accuracy Gap: {detailed['classification_details']['accuracy_difference']:.3f}")
#         print(f"   • Overfitting: {detailed['classification_details']['overfitting_indicator']}")
#     elif 'regression_details' in detailed:
#         print(f"   • R² Gap: {detailed['regression_details']['r2_difference']:.3f}")
#         print(f"   • RMSE Ratio: {detailed['regression_details']['rmse_ratio']:.3f}")
#         print(f"   • Overfitting: {detailed['regression_details']['overfitting_indicator']}")
    
#     # Visualizations
#     if results["visualizations"]:
#         print(f"\n📈 VISUALIZATIONS CREATED:")
#         for viz in results["visualizations"]:
#             print(f"   • {viz}")
    
#     # Recommendations
#     print(f"\n💡 TOP RECOMMENDATIONS:")
#     for i, rec in enumerate(results["recommendations"][:5], 1):
#         print(f"   {i}. {rec}")
    
#     # Final Report
#     print(f"\n📋 AI GENERATED REPORT:")
#     print("-" * 50)
#     report = results["final_report"]["ai_generated_report"]
#     print(report)
#     print("-" * 50)
    
#     print(f"\n✅ Evaluation completed at: {results['final_report']['timestamp']}")
#     print(f"📊 Status: {results['final_report']['evaluation_status']}")


# def test_classification_workflow():
#     """Test the evaluator agent with classification data"""
#     print("\n🧪 TESTING CLASSIFICATION WORKFLOW")
#     print("="*50)
    
#     # Create mock data and results
#     model_results, cleaned_state = create_mock_classification_data()
    
#     # Initialize evaluator
#     evaluator = EvaluatorAgent()
    
#     # Run evaluation
#     results = evaluator.evaluate_model(model_results, cleaned_state)
    
#     # Print results
#     print_evaluation_results(results)
    
#     return results


# def test_regression_workflow():
#     """Test the evaluator agent with regression data"""
#     print("\n🧪 TESTING REGRESSION WORKFLOW")
#     print("="*50)
    
#     # Create mock data and results
#     model_results, cleaned_state = create_mock_regression_data()
    
#     # Initialize evaluator
#     evaluator = EvaluatorAgent()
    
#     # Run evaluation
#     results = evaluator.evaluate_model(model_results, cleaned_state)
    
#     # Print results
#     print_evaluation_results(results)
    
#     return results


# def main():
#     """Main test function"""
#     print("🚀 EVALUATOR AGENT TEST SUITE")
#     print("="*60)
#     print("This script tests the EvaluatorAgent with mock data")
#     print("No external dependencies required!")
#     print("="*60)
    
#     try:
#         # Test classification workflow
#         classification_results = test_classification_workflow()
        
#         # Test regression workflow  
#         regression_results = test_regression_workflow()
        
#         print("\n" + "="*60)
#         print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
#         print("="*60)
#         print("\n📊 Summary:")
#         print(f"   • Classification model evaluated: ✅")
#         print(f"   • Regression model evaluated: ✅")
#         print(f"   • Visualizations created: ✅")
#         print(f"   • Reports generated: ✅")
#         print(f"   • Recommendations provided: ✅")
        
#         # Check if visualizations were saved
#         import os
#         viz_files = [f for f in os.listdir('.') if f.endswith('_evaluation.png')]
#         if viz_files:
#             print(f"\n🎨 Generated visualization files:")
#             for file in viz_files:
#                 print(f"   • {file}")
        
#     except Exception as e:
#         print(f"\n❌ Test failed with error: {str(e)}")
#         import traceback
#         traceback.print_exc()


# if __name__ == "__main__":
#     main()