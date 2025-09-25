import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib
import openai
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

class ModelTrainerAgent:
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
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def train_model(self, data: pd.DataFrame, eda_results: dict, model_training_options: dict = None) -> dict:
        """Train machine learning model based on problem type and user options"""
        problem_type = eda_results["problem_type"]
        target_column = eda_results["target_variable"]
        
        model_results = {
            "selected_model": None,
            "cv_score": None,
            "feature_importance": None,
            "model_comparison": None,
            "training_summary": None,
            "y_test": None,
            "predictions": None,
            "hyperparameters_used": None,
            "cv_strategy_used": None,
            "algorithm_explanation": None,
            "training_insights": None
        }
        
        if model_training_options is None:
            model_training_options = {
                "algorithm": "Auto-Select Best",
                "cv_strategy": "stratified_kfold",
                "cv_folds": 5,
                "hyperparameters": {}
        }
        
        try:
            print(f"🎯 Training {problem_type} model...")
            print(f"📊 Algorithm: {model_training_options.get('algorithm', 'Auto-Select Best')}")
            print(f"🔄 CV Strategy: {model_training_options.get('cv_strategy', 'stratified_kfold')}")
            
            # Prepare features and target
            X, y = self._prepare_features_target(data, target_column)
            
            # Handle extreme class imbalance for classification
            if problem_type == "classification":
                X, y = self._handle_class_imbalance(X, y, model_training_options)
            
            # Validate class distribution for classification problems
            use_stratification = False
            if problem_type == "classification":
                use_stratification = self._can_use_stratification(y, min_samples_per_class=2)
                if not use_stratification:
                    print("⚠️ Warning: Some classes have too few samples for stratification. Using random split instead.")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if use_stratification else None
            )
            
            # Preprocess features
            X_train_scaled, X_test_scaled = self._preprocess_features(X_train, X_test)
            
            # Get cross-validation strategy
            cv_strategy = self._get_cv_strategy(
                model_training_options.get("cv_strategy", "stratified_kfold"),
                model_training_options.get("cv_folds", 5),
                problem_type,
                y_train  # Pass y_train to check class distribution
            )
            
            # Select and train model
            model, cv_score, model_comparison, hyperparams = self._select_and_train_model(
                X_train_scaled, y_train, problem_type, model_training_options, cv_strategy
            )
            
            model_results["selected_model"] = type(model).__name__
            model_results["cv_score"] = float(cv_score)
            model_results["model_comparison"] = model_comparison
            model_results["problem_type"] = problem_type
            model_results["target_column"] = target_column
            model_results["hyperparameters_used"] = hyperparams
            model_results["cv_strategy_used"] = model_training_options.get("cv_strategy", "stratified_kfold")
            
            # Get algorithm explanation
            model_results["algorithm_explanation"] = self._get_algorithm_explanation(type(model).__name__)
            
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
            
            # Generate educational insights
            model_results["training_insights"] = self._generate_training_insights(
                model, cv_score, model_comparison, problem_type, len(X.columns)
            )
            
            model_results["y_test"] = y_test.tolist()
            model_results["predictions"] = model.predict(X_test_scaled).tolist()
            
            # Add prediction probabilities for classification models (needed for ROC AUC)
            if problem_type == "classification" and hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(X_test_scaled)
                    # For binary classification, use probabilities of positive class
                    if probabilities.shape[1] == 2:
                        model_results["prediction_probabilities"] = probabilities[:, 1].tolist()
                    else:
                        # For multiclass, store all probabilities
                        model_results["prediction_probabilities"] = probabilities.tolist()
                except Exception as e:
                    print(f"⚠️ Warning: Could not get prediction probabilities: {str(e)}")
                    model_results["prediction_probabilities"] = None
            
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
            
        # Drop datetime columns as they cannot be directly scaled
        datetime_columns = X.select_dtypes(include=[np.datetime64]).columns
        if not datetime_columns.empty:
            print(f"⚠️ Dropping datetime columns from features: {datetime_columns.tolist()} for scaling.")
            X = X.drop(columns=datetime_columns)

        # Check if any features remain
        if X.shape[1] == 0:
            raise ValueError("No suitable numeric features remain after preprocessing. Please check your dataset and cleaning/type conversion settings.")
        
        # Handle target encoding robustly
        unique_classes = y.nunique()
        # If numeric but low cardinality, treat as classification labels
        if pd.api.types.is_numeric_dtype(y) and unique_classes <= 10 and set(pd.Series(y.dropna().unique()).astype(float).astype(int)) == set(pd.Series(y.dropna().unique()).astype(float)):
            # Coerce to int labels (e.g., 0.0/1.0 -> 0/1)
            try:
                y = y.astype(float).round().astype(int)
                print(f"🎯 Target variable coerced to integer classes (unique={unique_classes})")
            except Exception:
                pass
        if y.dtype in ['object', 'category']:
            unique_classes = y.nunique()
            print(f"🎯 Target variable info: {unique_classes} unique classes detected")
            y = self.label_encoder.fit_transform(y)
        else:
            unique_classes = y.nunique()
            print(f"🎯 Target variable info: {unique_classes} unique values detected")
        
        # Additional data quality checks
        total_samples = len(y)
        print(f"📊 Dataset size: {total_samples} samples, {X.shape[1]} features")
        
        if hasattr(y, 'value_counts'):
            print(f"🔍 Target distribution preview:")
            target_counts = y.value_counts().head(10)  # Show top 10 most frequent values
            for value, count in target_counts.items():
                percentage = (count / total_samples) * 100
                print(f"   Value {value}: {count} samples ({percentage:.1f}%)")
            if len(target_counts) < y.nunique():
                print(f"   ... and {y.nunique() - len(target_counts)} more classes")
        
        return X, y
    
    def _handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series, options: dict) -> tuple:
        """Handle extreme class imbalance by removing singleton classes or providing alternatives"""
        from collections import Counter
        
        class_counts = Counter(y)
        original_classes = len(class_counts)
        original_samples = len(y)
        
        # Find classes with very few samples (singleton or near-singleton)
        min_samples_threshold = 3  # Minimum samples per class for robust training
        problematic_classes = {cls: count for cls, count in class_counts.items() 
                             if count < min_samples_threshold}
        
        if problematic_classes:
            print(f"🔍 Detected {len(problematic_classes)} problematic classes with < {min_samples_threshold} samples:")
            for cls, count in problematic_classes.items():
                print(f"   Class {cls}: {count} samples")
            
            # Calculate impact of removing these classes
            samples_to_remove = sum(problematic_classes.values())
            remaining_samples = original_samples - samples_to_remove
            removal_percentage = (samples_to_remove / original_samples) * 100
            
            print(f"📊 Impact analysis:")
            print(f"   Samples to remove: {samples_to_remove} ({removal_percentage:.1f}%)")
            print(f"   Remaining samples: {remaining_samples}")
            print(f"   Remaining classes: {original_classes - len(problematic_classes)}")
            
            # Decide on strategy based on impact
            if removal_percentage <= 5.0 and remaining_samples >= 50:
                # Safe to remove problematic classes
                print("🔧 Strategy: Removing problematic classes (low impact)")
                
                # Create mask for rows to keep
                keep_mask = ~y.isin(problematic_classes.keys())
                X_filtered = X[keep_mask].reset_index(drop=True)
                y_filtered = y[keep_mask].reset_index(drop=True)
                
                print(f"✅ Dataset filtered: {len(y_filtered)} samples, {len(Counter(y_filtered))} classes")
                return X_filtered, y_filtered
                
            elif remaining_samples >= 30:
                # Moderate impact but still workable
                print("⚠️ Strategy: Removing problematic classes (moderate impact - be cautious of results)")
                
                keep_mask = ~y.isin(problematic_classes.keys())
                X_filtered = X[keep_mask].reset_index(drop=True)
                y_filtered = y[keep_mask].reset_index(drop=True)
                
                print(f"✅ Dataset filtered: {len(y_filtered)} samples, {len(Counter(y_filtered))} classes")
                return X_filtered, y_filtered
                
            else:
                # Too much impact - keep original data but warn user
                print("❌ Cannot safely remove problematic classes (would remove too much data)")
                print("🔧 Strategy: Keeping original data - will use robust validation methods")
                print("⚠️ Warning: Model performance may be unreliable due to class imbalance")
                
                return X, y
        
        # No problematic classes found
        return X, y
    
    def _can_use_stratification(self, y, min_samples_per_class=2):
        """Check if the target variable supports stratification"""
        try:
            from collections import Counter
            class_counts = Counter(y)
            
            # Check if all classes have at least min_samples_per_class samples
            min_count = min(class_counts.values())
            
            if min_count < min_samples_per_class:
                print(f"🔍 Class distribution analysis:")
                for class_label, count in sorted(class_counts.items()):
                    print(f"   Class {class_label}: {count} samples")
                print(f"   Minimum samples per class: {min_count} (required: {min_samples_per_class})")
                return False
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error checking stratification compatibility: {str(e)}")
            return False
    
    def _preprocess_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
        """Preprocess features using scaling"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def _select_and_train_model(self, X_train: np.ndarray, y_train: np.ndarray, problem_type: str, options: dict, cv_strategy) -> tuple:
        """Select and train the best model based on options"""
        
        # Handle both 'algorithm' (legacy) and 'algorithms' (new UI format)
        algorithm = options.get("algorithm", "Auto-Select Best")
        algorithms_list = options.get("algorithms", [])
        hyperparameters = options.get("hyperparameters", {})
        
        # Debug: Print what we received
        print(f"🔍 DEBUG: Received algorithms_list: {algorithms_list}")
        print(f"🔍 DEBUG: Received algorithm: {algorithm}")
        print(f"🔍 DEBUG: Full options: {options}")
        
        # If algorithms list is provided from UI, use it instead of auto-select
        if algorithms_list and len(algorithms_list) > 0:
            algorithm = "Custom-List"
            print(f"🔍 DEBUG: Switching to Custom-List mode with {len(algorithms_list)} algorithms")
        
        if algorithm == "Auto-Select Best" or algorithm == "Custom-List":
            # Define all available models
            if problem_type == "classification":
                all_models = {
                    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
                    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
                    "SVM": SVC(random_state=42),
                    "XGBoost": GradientBoostingClassifier(n_estimators=100, random_state=42),  # Using GradientBoosting as XGBoost substitute
                    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                    "KNN": KNeighborsClassifier(n_neighbors=5)
                }
            else:
                all_models = {
                    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
                    "LinearRegression": LinearRegression(),
                    "SVM": SVR(),
                    "XGBoost": GradientBoostingRegressor(n_estimators=100, random_state=42),  # Using GradientBoosting as XGBoost substitute
                    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
                    "KNN": KNeighborsRegressor(n_neighbors=5)
                }
            
            # Filter models based on user selection
            if algorithm == "Custom-List" and algorithms_list:
                print(f"🔍 DEBUG: Available models: {list(all_models.keys())}")
                models = {}
                for alg_name in algorithms_list:
                    if alg_name in all_models:
                        models[alg_name] = all_models[alg_name]
                        print(f"✅ DEBUG: Added algorithm '{alg_name}' to training list")
                    else:
                        print(f"⚠️ Algorithm '{alg_name}' not recognized, skipping...")
                        print(f"🔍 DEBUG: Available options are: {list(all_models.keys())}")
                
                print(f"🔍 DEBUG: Final models to train: {list(models.keys())}")
                
                if not models:
                    print("⚠️ No valid algorithms selected, using RandomForest as fallback")
                    models = {"RandomForest": all_models.get("RandomForest", list(all_models.values())[0])}
            else:
                # Use all models for auto-select
                models = all_models
                print(f"🔍 DEBUG: Using auto-select with all models: {list(models.keys())}")
            
            best_model = None
            best_score = -np.inf
            model_comparison = []
            
            for name, model in models.items():
                try:
                    # Use cross-validation to select best model
                    cv_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, 
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
                            
                except Exception as e:
                    print(f"⚠️ {name} failed in CV: {str(e)[:100]}... Using holdout validation.")
                    
                    # Fallback to simple holdout validation
                    try:
                        holdout_score = self._holdout_validation(model, X_train, y_train, problem_type)
                        print(f"📊 {name} Holdout Score: {holdout_score:.4f}")
                        
                        model_comparison.append({
                            "model": name + " (holdout)",
                            "cv_score": float(holdout_score)
                        })
                        
                        if holdout_score > best_score:
                            best_score = holdout_score
                            best_model = model
                            
                    except Exception as e2:
                        print(f"❌ {name} also failed holdout validation: {str(e2)[:100]}...")
                        # Skip this model entirely
            
            # If no model worked, use a safe fallback
            if best_model is None:
                print("⚠️ All models failed. Using RandomForest as fallback.")
                if problem_type == "classification":
                    best_model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    best_model = RandomForestRegressor(n_estimators=100, random_state=42)
                best_score = 0.0
                model_comparison = [{"model": f"{type(best_model).__name__} (fallback)", "cv_score": 0.0}]
            
            hyperparameters = {}
            
        else:
            # Handle specific algorithm selection with custom hyperparameters
            default_hyperparams = self._get_default_hyperparameters(algorithm, problem_type)
            
            # Merge user hyperparameters with defaults
            final_hyperparams = {**default_hyperparams, **hyperparameters}
            
            try:
                best_model = self._create_model_instance(algorithm, final_hyperparams)
                model_comparison = []
                scoring = 'accuracy' if problem_type == "classification" else 'r2'
                try:
                    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_strategy, scoring=scoring)
                    best_score = cv_scores.mean()
                    model_comparison.append({"model": algorithm, "cv_score": float(best_score)})
                    print(f"📊 {algorithm} CV Score: {best_score:.4f}")
                except Exception as e:
                    print(f"⚠️ {algorithm} failed in CV: {str(e)[:100]}... Using holdout validation.")
                    best_score = self._holdout_validation(best_model, X_train, y_train, problem_type)
                    model_comparison.append({"model": algorithm + " (holdout)", "cv_score": float(best_score)})
                    print(f"📊 {algorithm} Holdout Score: {best_score:.4f}")
                    
                hyperparameters = final_hyperparams
                
            except Exception as e:
                print(f"⚠️ Error creating {algorithm}: {e}. Using default RandomForest as fallback.")
                # Use a safe fallback instead of recursive call
                if problem_type == "classification":
                    best_model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    best_model = RandomForestRegressor(n_estimators=100, random_state=42)
                
                # Try to get a score
                try:
                    cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_strategy, 
                                              scoring='accuracy' if problem_type == "classification" else 'r2')
                    best_score = cv_scores.mean()
                except Exception:
                    best_score = 0.0
                
                model_comparison = [{"model": f"{type(best_model).__name__} (fallback)", "cv_score": float(best_score)}]
                hyperparameters = {}
        
        # Train the best model
        best_model.fit(X_train, y_train)
        return best_model, best_score, model_comparison, hyperparameters
    
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

    def _generate_training_insights(self, model, cv_score: float, model_comparison: list, problem_type: str, num_features: int) -> dict:
        """Generate educational insights based on model performance and characteristics."""
        insights = {
            "general_insights": {
                "model_selected": type(model).__name__,
                "cv_score": float(cv_score),
                "problem_type": problem_type,
                "num_features": num_features
            },
            "algorithm_specific_insights": {}
        }

        for item in model_comparison:
            model_name = item["model"]
            if model_name in insights["algorithm_specific_insights"]:
                insights["algorithm_specific_insights"][model_name]["cv_scores"].append(float(item["cv_score"]))
            else:
                insights["algorithm_specific_insights"][model_name] = {
                    "cv_scores": [float(item["cv_score"])],
                    "description": self._get_algorithm_explanation(model_name)["description"]
                }

        return insights

    def _get_cv_strategy(self, cv_type: str, cv_folds: int, problem_type: str, y_train=None):
        """Get cross-validation strategy based on user selection and data characteristics"""
        
        # For classification, adapt the number of folds based on data characteristics
        if problem_type == "classification" and y_train is not None:
            adapted_folds = self._adapt_cv_folds(y_train, cv_folds)
            if adapted_folds != cv_folds:
                print(f"🔄 Adapted CV folds from {cv_folds} to {adapted_folds} based on data characteristics")
                cv_folds = adapted_folds
        
        print(f"🔄 Using {cv_type} with {cv_folds} folds")
        
        # Check if we can use stratification for classification
        can_stratify = False
        if problem_type == "classification" and y_train is not None:
            can_stratify = self._can_use_stratification(y_train, min_samples_per_class=cv_folds)
            if not can_stratify:
                print(f"⚠️ Warning: Class distribution incompatible with {cv_folds}-fold stratified CV. Using regular KFold instead.")
        
        # Handle edge case where CV folds are still too many
        if cv_folds > len(y_train) // 2:
            cv_folds = max(2, len(y_train) // 3)  # At least 2 folds, but safe number
            print(f"🔄 Further reduced CV folds to {cv_folds} due to small dataset size")
        
        if cv_type == "stratified_kfold" and problem_type == "classification" and can_stratify:
            return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        elif cv_type == "kfold" or (cv_type == "stratified_kfold" and not can_stratify):
            return KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        elif cv_type == "time_series":
            return TimeSeriesSplit(n_splits=cv_folds)
        else:
            # Default to appropriate strategy
            if problem_type == "classification" and can_stratify:
                return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            else:
                return KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    def _adapt_cv_folds(self, y_train, requested_folds: int) -> int:
        """Adapt the number of CV folds based on the smallest class size and dataset characteristics"""
        from collections import Counter
        
        class_counts = Counter(y_train)
        min_class_size = min(class_counts.values())
        total_samples = len(y_train)
        
        # Calculate safe number of folds
        # For stratified CV: need at least 1 sample per class per fold
        # For regular CV: need reasonable fold sizes
        
        # Conservative approach: ensure each fold has reasonable data
        safe_folds_by_class = min_class_size  # Each class should appear in each fold at least once
        safe_folds_by_size = max(2, total_samples // 20)  # At least 20 samples per fold on average
        
        adapted_folds = min(requested_folds, safe_folds_by_class, safe_folds_by_size)
        adapted_folds = max(2, adapted_folds)  # Always have at least 2 folds
        
        if adapted_folds != requested_folds:
            print(f"🔍 CV adaptation analysis:")
            print(f"   Minimum class size: {min_class_size}")
            print(f"   Total samples: {total_samples}")
            print(f"   Requested folds: {requested_folds}")
            print(f"   Safe folds by class: {safe_folds_by_class}")
            print(f"   Safe folds by size: {safe_folds_by_size}")
        
        return adapted_folds
    
    def _holdout_validation(self, model, X_train, y_train, problem_type: str) -> float:
        """Perform simple holdout validation when CV fails"""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, r2_score
        
        # Use 80-20 split for holdout validation
        X_holdout_train, X_holdout_val, y_holdout_train, y_holdout_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # Train model on holdout training set
        model.fit(X_holdout_train, y_holdout_train)
        
        # Evaluate on holdout validation set
        y_holdout_pred = model.predict(X_holdout_val)
        
        if problem_type == "classification":
            return accuracy_score(y_holdout_val, y_holdout_pred)
        else:
            return r2_score(y_holdout_val, y_holdout_pred)
    
    def _get_default_hyperparameters(self, algorithm: str, problem_type: str) -> dict:
        """Get default hyperparameters for each algorithm"""
        defaults = {
            "RandomForestClassifier": {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42
            },
            "RandomForestRegressor": {
                "n_estimators": 100,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42
            },
            "LogisticRegression": {
                "C": 1.0,
                "penalty": "l2",
                "solver": "liblinear",
                "random_state": 42,
                "max_iter": 1000
            },
            "SVC": {
                "C": 1.0,
                "kernel": "rbf",
                "gamma": "scale",
                "random_state": 42
            },
            "DecisionTreeClassifier": {
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42
            },
            "GradientBoostingClassifier": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42
            },
            "KNeighborsClassifier": {
                "n_neighbors": 5,
                "weights": "uniform",
                "metric": "minkowski"
            },
            "GaussianNB": {},
            "LinearRegression": {},
            "Ridge": {
                "alpha": 1.0,
                "random_state": 42
            },
            "Lasso": {
                "alpha": 1.0,
                "random_state": 42,
                "max_iter": 1000
            },
            "SVR": {
                "C": 1.0,
                "kernel": "rbf",
                "gamma": "scale"
            },
            "DecisionTreeRegressor": {
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "random_state": 42
            },
            "GradientBoostingRegressor": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42
            },
            "KNeighborsRegressor": {
                "n_neighbors": 5,
                "weights": "uniform",
                "metric": "minkowski"
            }
        }
        return defaults.get(algorithm, {})

    def _create_model_instance(self, algorithm: str, hyperparameters: dict):
        """Create model instance with specified hyperparameters"""
        model_classes = {
            "RandomForestClassifier": RandomForestClassifier,
            "RandomForestRegressor": RandomForestRegressor,
            "LogisticRegression": LogisticRegression,
            "SVC": SVC,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "GradientBoostingClassifier": GradientBoostingClassifier,
            "KNeighborsClassifier": KNeighborsClassifier,
            "GaussianNB": GaussianNB,
            "LinearRegression": LinearRegression,
            "Ridge": Ridge,
            "Lasso": Lasso,
            "SVR": SVR,
            "DecisionTreeRegressor": DecisionTreeRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor,
            "KNeighborsRegressor": KNeighborsRegressor
        }
        
        if algorithm in model_classes:
            return model_classes[algorithm](**hyperparameters)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    def _get_algorithm_explanation(self, algorithm: str) -> dict:
        """Get educational explanation for each algorithm"""
        explanations = {
            "RandomForestClassifier": {
                "description": "Random Forest combines multiple decision trees to make predictions. Each tree is trained on a random subset of data and features.",
                "strengths": ["Handles overfitting well", "Works with missing values", "Provides feature importance", "Good for both linear and non-linear relationships"],
                "weaknesses": ["Can be slow on large datasets", "Less interpretable than single decision tree"],
                "when_to_use": "Good general-purpose algorithm for most classification tasks, especially when you need feature importance."
            },
            "LogisticRegression": {
                "description": "Logistic Regression uses the logistic function to model the probability of class membership. Despite its name, it's used for classification.",
                "strengths": ["Fast training and prediction", "Provides probability estimates", "Less prone to overfitting", "Highly interpretable"],
                "weaknesses": ["Assumes linear relationship", "Sensitive to outliers", "Requires feature scaling"],
                "when_to_use": "When you need interpretable results and have linear relationships between features and target."
            },
            "SVC": {
                "description": "Support Vector Classifier finds the optimal boundary (hyperplane) that separates different classes with maximum margin.",
                "strengths": ["Effective in high dimensions", "Memory efficient", "Versatile with different kernels"],
                "weaknesses": ["Slow on large datasets", "Requires feature scaling", "No probability estimates by default"],
                "when_to_use": "When you have high-dimensional data or need to capture complex non-linear relationships."
            },
            "DecisionTreeClassifier": {
                "description": "Decision Tree creates a model that predicts target values by learning simple decision rules inferred from data features.",
                "strengths": ["Highly interpretable", "Requires little data preparation", "Handles both numerical and categorical data"],
                "weaknesses": ["Prone to overfitting", "Unstable (small data changes can result in very different trees)"],
                "when_to_use": "When you need maximum interpretability and can handle potential overfitting through pruning."
            },
            "GradientBoostingClassifier": {
                "description": "Gradient Boosting builds models sequentially, where each new model corrects errors made by previous models.",
                "strengths": ["Often achieves high accuracy", "Handles missing values", "No need for data preprocessing"],
                "weaknesses": ["Prone to overfitting", "Longer training time", "Requires hyperparameter tuning"],
                "when_to_use": "When you want high accuracy and can invest time in hyperparameter tuning."
            },
            "KNeighborsClassifier": {
                "description": "K-Nearest Neighbors classifies data points based on the majority class among k nearest neighbors in feature space.",
                "strengths": ["Simple to understand", "No assumptions about data distribution", "Works well with small datasets"],
                "weaknesses": ["Computationally expensive for large datasets", "Sensitive to irrelevant features", "Requires feature scaling"],
                "when_to_use": "When you have small datasets and local patterns are important for classification."
            },
            "GaussianNB": {
                "description": "Naive Bayes assumes features are independent and follow a Gaussian distribution. Despite the 'naive' assumption, it often works well.",
                "strengths": ["Fast training and prediction", "Works well with small datasets", "Not sensitive to irrelevant features"],
                "weaknesses": ["Strong independence assumption", "Assumes Gaussian distribution"],
                "when_to_use": "When you have small datasets or need a fast baseline model, especially for text classification."
            }
        }
        
        # Add regression explanations
        regression_explanations = {
            "LinearRegression": {
                "description": "Linear Regression models the relationship between features and target as a linear equation.",
                "strengths": ["Fast and simple", "Highly interpretable", "No hyperparameters to tune"],
                "weaknesses": ["Assumes linear relationship", "Sensitive to outliers"],
                "when_to_use": "When you need interpretability and believe the relationship is linear."
            },
            "Ridge": {
                "description": "Ridge Regression adds L2 regularization to Linear Regression to prevent overfitting.",
                "strengths": ["Reduces overfitting", "Handles multicollinearity", "Stable solutions"],
                "weaknesses": ["Doesn't perform feature selection", "Still assumes linear relationship"],
                "when_to_use": "When you have multicollinearity or want to prevent overfitting in linear models."
            },
            "Lasso": {
                "description": "Lasso Regression adds L1 regularization to Linear Regression, which can perform automatic feature selection.",
                "strengths": ["Automatic feature selection", "Reduces overfitting", "Sparse solutions"],
                "weaknesses": ["May select only one feature from correlated groups", "Still assumes linear relationship"],
                "when_to_use": "When you want automatic feature selection and have many irrelevant features."
            }
        }
        
        explanations.update(regression_explanations)
        
        return explanations.get(algorithm, {
            "description": f"{algorithm} - Algorithm explanation not available.",
            "strengths": ["Refer to documentation for details"],
            "weaknesses": ["Refer to documentation for details"],
            "when_to_use": "Refer to algorithm documentation for guidance."
        })