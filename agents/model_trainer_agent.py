import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
import openai
import os
import plotly.express as px
import plotly.graph_objects as go

class ModelTrainerAgent:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def train_model(self, data: pd.DataFrame, eda_results: dict) -> dict:
        """Train machine learning model based on problem type"""
        problem_type = eda_results["problem_type"]
        target_column = eda_results["target_variable"]
        
        model_results = {
            "selected_model": None,
            "cv_score": None,
            "feature_importance": None,
            "model_comparison": None,
            "training_summary": None,
            "y_test": None,
            "predictions": None
        }
        
        try:
            print(f"🎯 Training {problem_type} model...")
            
            # Prepare features and target
            X, y = self._prepare_features_target(data, target_column)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if problem_type == "classification" else None
            )
            
            # Preprocess features
            X_train_scaled, X_test_scaled = self._preprocess_features(X_train, X_test)
            
            # Select and train model
            model, cv_score, model_comparison = self._select_and_train_model(X_train_scaled, y_train, problem_type)
            model_results["selected_model"] = type(model).__name__
            model_results["cv_score"] = float(cv_score)
            model_results["model_comparison"] = model_comparison
            model_results["problem_type"] = problem_type
            model_results["target_column"] = target_column
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                model_results["feature_importance"] = importance_df.to_dict('records')
            
            # Generate training summary
            model_results["training_summary"] = self._generate_training_summary(
                model, X_test_scaled, y_test, problem_type
            )
            model_results["y_test"] = y_test.tolist()
            model_results["predictions"] = model.predict(X_test_scaled).tolist()
            
            # Save model
            model_path = f"trained_model_{problem_type}.joblib"
            joblib.dump(model, model_path)
            
            print(f"✅ Model training completed: {model_results['selected_model']}")
            return model_results
            
        except Exception as e:
            print(f"❌ Model training failed: {str(e)}")
            raise
    
    def _prepare_features_target(self, data: pd.DataFrame, target_column: str) -> tuple:
        """Prepare features and target variables"""
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle categorical features
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            # Simple label encoding for categorical variables
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Handle categorical target for classification
        if y.dtype in ['object', 'category']:
            y = self.label_encoder.fit_transform(y)
        
        return X, y
    
    def _preprocess_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        """Preprocess features using scaling"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def _select_and_train_model(self, X_train: np.ndarray, y_train: np.ndarray, problem_type: str) -> tuple:
        """Select and train appropriate model"""
        if problem_type == "classification":
            # Try multiple models and select best one
            models = {
                "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
                "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000)
            }
        else:
            models = {
                "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
                "LinearRegression": LinearRegression()
            }
        
        best_model = None
        best_score = -np.inf
        model_comparison = []
        
        for name, model in models.items():
            # Use cross-validation to select best model
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                      scoring='accuracy' if problem_type == "classification" else 'r2')
            avg_score = cv_scores.mean()
            
            print(f"📊 {name} CV Score: {avg_score:.4f}")
            
            model_comparison.append({
                "model": name,
                "cv_score": float(avg_score)
            })
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
        
        # Train the best model
        best_model.fit(X_train, y_train)
        return best_model, best_score, model_comparison
    
    def _generate_training_summary(self, model, X_test: np.ndarray, y_test: np.ndarray, problem_type: str) -> dict:
        """Generate training summary with metrics and insights"""
        try:
            y_pred = model.predict(X_test)
            
            if problem_type == "classification":
                metrics = {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "classification_report": classification_report(y_test, y_pred, output_dict=True)
                }
            else:
                metrics = {
                    "r2": float(r2_score(y_test, y_pred)),
                    "mse": float(mean_squared_error(y_test, y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred)))
                }
            
            prompt = f"""
            Analyze this {problem_type} model performance:
            - Model: {type(model).__name__}
            - Metrics: {metrics}
            
            Provide a brief summary of the model's performance and key insights.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            
            return {
                "metrics": metrics,
                "insights": response.choices[0].message.content
            }
            
        except Exception as e:
            return {
                "metrics": {},
                "insights": f"Error generating summary: {str(e)}"
            }