import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

class ProblemTypeAgent:
    """
    Problem Type Detection Agent
    
    Follows the staged approach:
    1. Load & Inspect: Basic checks, fix obvious issues
    2. Min Clean: Standardize, fix types for reliable detection
    3. Identify Type: Find target & problem type
    4. Deep Clean: Proceed according to task requirements
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.decision_log = []
        self.step_log = []
        
    def _log_step(self, step: str, details: str):
        """Log each step for transparency"""
        self.step_log.append({
            "step": step,
            "details": details,
            "timestamp": pd.Timestamp.now()
        })
        self.logger.info(f"Step: {step} - {details}")
    
    def _log_decision(self, decision_type: str, reasoning: str, action_taken: str):
        """Log decisions with rationale"""
        self.decision_log.append({
            "decision_type": decision_type,
            "reasoning": reasoning,
            "action_taken": action_taken,
            "timestamp": pd.Timestamp.now()
        })
    
    def analyze_problem_type(self, data: pd.DataFrame) -> Dict:
        """
        Main method to analyze and identify the problem type
        
        Returns:
            Dict containing problem type, target variable, and recommendations
        """
        self._log_step("START", "Beginning problem type analysis")
        
        # Stage 1: Load & Inspect
        inspection_results = self._load_and_inspect(data)
        
        # Stage 2: Min Clean (minimal cleaning for reliable detection)
        min_cleaned_data = self._minimal_clean(data)
        
        # Stage 3: Identify Type
        problem_analysis = self._identify_problem_type(min_cleaned_data)
        
        # Stage 4: Prepare recommendations for Deep Clean
        recommendations = self._prepare_deep_clean_recommendations(problem_analysis)
        
        return {
            "problem_type": problem_analysis["problem_type"],
            "target_variable": problem_analysis.get("target_variable"),
            "target_characteristics": problem_analysis.get("target_characteristics", {}),
            "data_characteristics": inspection_results,
            "cleaning_recommendations": recommendations,
            "decision_log": self.decision_log,
            "step_log": self.step_log
        }
    
    def _load_and_inspect(self, data: pd.DataFrame) -> Dict:
        """Stage 1: Basic checks, fix obvious issues"""
        self._log_step("LOAD_INSPECT", "Performing basic data inspection")
        
        inspection = {
            "original_shape": data.shape,
            "data_types": data.dtypes.to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "unique_counts": {col: data[col].nunique() for col in data.columns},
            "sample_values": {col: data[col].dropna().head(3).tolist() for col in data.columns},
            "obvious_issues": []
        }
        
        # Check for obvious issues
        for col in data.columns:
            # Check for all-null columns
            if data[col].isnull().all():
                inspection["obvious_issues"].append(f"Column '{col}' is completely null")
            
            # Check for single-value columns
            if data[col].nunique() == 1:
                inspection["obvious_issues"].append(f"Column '{col}' has only one unique value")
            
            # Check for extremely high cardinality in categorical data
            if data[col].dtype == 'object' and data[col].nunique() > data.shape[0] * 0.8:
                inspection["obvious_issues"].append(f"Column '{col}' has extremely high cardinality")
        
        self._log_decision("INSPECTION", 
                          f"Found {len(inspection['obvious_issues'])} obvious issues", 
                          "Issues logged for later handling")
        
        return inspection
    
    def _minimal_clean(self, data: pd.DataFrame) -> pd.DataFrame:
        """Stage 2: Standardize, fix types for reliable detection"""
        self._log_step("MIN_CLEAN", "Performing minimal cleaning for reliable detection")
        
        data_copy = data.copy()
        
        # Fix obvious data type issues
        for col in data_copy.columns:
            # Convert obvious numeric strings to numeric
            if data_copy[col].dtype == 'object':
                # Check if it's actually numeric
                try:
                    pd.to_numeric(data_copy[col], errors='raise')
                    data_copy[col] = pd.to_numeric(data_copy[col], errors='coerce')
                    self._log_decision("TYPE_CONVERSION", 
                                     f"Column '{col}' converted from object to numeric", 
                                     "Numeric conversion applied")
                except (ValueError, TypeError):
                    pass
            
            # Convert obvious boolean strings
            if data_copy[col].dtype == 'object':
                unique_vals = data_copy[col].dropna().unique()
                if len(unique_vals) == 2:
                    bool_patterns = [
                        {'0', '1'}, {'true', 'false'}, {'yes', 'no'}, 
                        {'y', 'n'}, {'t', 'f'}, {'positive', 'negative'}
                    ]
                    if set(str(v).lower() for v in unique_vals) in bool_patterns:
                        data_copy[col] = data_copy[col].astype('category')
                        self._log_decision("TYPE_CONVERSION", 
                                         f"Column '{col}' converted to categorical (binary)", 
                                         "Binary categorical conversion applied")
        
        # Handle obvious missing value patterns (but don't impute target yet)
        for col in data_copy.columns:
            missing_pct = data_copy[col].isnull().sum() / len(data_copy)
            if missing_pct > 0.5:  # More than 50% missing
                self._log_decision("MISSING_VALUES", 
                                 f"Column '{col}' has {missing_pct:.1%} missing values", 
                                 "Flagged for potential removal")
        
        return data_copy
    
    def _identify_problem_type(self, data: pd.DataFrame) -> Dict:
        """Stage 3: Find target & problem type"""
        self._log_step("IDENTIFY_TYPE", "Identifying problem type and target variable")
        
        # Target detection indicators
        target_indicators = {
            'binary_classification': [
                'target', 'label', 'class', 'outcome', 'result', 'prediction',
                'disease', 'survived', 'churn', 'click', 'buy', 'convert',
                'positive', 'negative', 'yes', 'no', 'true', 'false'
            ],
            'regression': [
                'price', 'cost', 'salary', 'income', 'revenue', 'sales',
                'amount', 'value', 'score', 'rating', 'prediction', 'target',
                'grade', 'performance', 'result', 'outcome', 'achievement'
            ],
            'multiclass': [
                'category', 'class', 'type', 'group', 'level', 'grade',
                'target', 'label', 'outcome'
            ]
        }
        
        # Feature exclusion patterns (columns that are clearly features, not targets)
        feature_exclusion_patterns = [
            'id', 'index', 'name', 'date', 'time', 'timestamp',
            'school', 'education', 'parent', 'gender', 'age', 'location',
            'address', 'phone', 'email', 'city', 'state', 'country',
            'type', 'category', 'status', 'method', 'source'
        ]
        
        target_candidates = []
        
        for col in data.columns:
            col_lower = col.lower()
            score = 0
            reasons = []
            
            # Check for feature exclusion patterns first
            is_likely_feature = False
            for pattern in feature_exclusion_patterns:
                if pattern in col_lower:
                    score -= 20  # Heavy penalty for feature patterns
                    reasons.append(f"Column name contains '{pattern}' - likely a feature, not target")
                    is_likely_feature = True
                    break
            
            # Check for target indicators (only if not excluded as feature)
            if not is_likely_feature:
                for problem_type, indicators in target_indicators.items():
                    for indicator in indicators:
                        if indicator in col_lower:
                            score += 10
                            reasons.append(f"Column name contains '{indicator}' indicator")
                            break
                
                # Additional scoring for target preference patterns
                if 'final' in col_lower and ('grade' in col_lower or 'score' in col_lower):
                    score += 15  # Strong preference for final grades/scores
                    reasons.append("Contains 'final' + grade/score - strong target indicator")
                elif 'target' in col_lower:
                    score += 15  # Strong preference for anything with 'target'
                    reasons.append("Contains 'target' - strong target indicator")
                elif 'previous' in col_lower or 'initial' in col_lower:
                    score -= 5  # Slight penalty for previous/initial values (likely features)
                    reasons.append("Contains 'previous/initial' - likely input feature")
            
            # Analyze data characteristics
            unique_vals = data[col].nunique()
            total_rows = len(data[col])
            unique_ratio = unique_vals / total_rows
            
            # Binary classification patterns
            if data[col].dtype in ['int64', 'bool'] or (data[col].dtype == 'category' and unique_vals == 2):
                unique_values = data[col].dropna().unique()
                if len(unique_values) == 2:
                    binary_patterns = [
                        {0, 1}, {True, False}, {'0', '1'}, {'yes', 'no'},
                        {'true', 'false'}, {'positive', 'negative'}, {'Y', 'N'}
                    ]
                    if set(str(v).lower() for v in unique_values) in [set(str(v).lower() for v in pattern) for pattern in binary_patterns]:
                        score += 15
                        reasons.append(f"Binary values {unique_values} strongly suggest classification target")
            
            # Data type and cardinality analysis
            if data[col].dtype in ['float64', 'int64'] and not is_likely_feature:
                score += 5  # Boost for numeric columns
                reasons.append(f"Numeric data type ({data[col].dtype}) suggests potential target")
            
            # Cardinality analysis
            if unique_ratio < 0.1:  # Very low cardinality
                if data[col].dtype in ['object', 'category']:
                    score += 3  # Reduced score for categorical low cardinality
                    reasons.append(f"Low cardinality ({unique_vals} unique values) suggests categorical target")
                else:
                    score += 8  # Higher score for numeric low cardinality (could be ordinal)
                    reasons.append(f"Low cardinality numeric ({unique_vals} unique values) suggests ordinal target")
            elif unique_ratio > 0.5:  # High cardinality
                score += 8  # Higher boost for high cardinality (likely continuous)
                reasons.append(f"High cardinality ({unique_vals} unique values) suggests continuous target")
            
            if score > 0:
                target_candidates.append({
                    'column': col,
                    'score': score,
                    'reasons': reasons,
                    'unique_values': unique_vals,
                    'dtype': str(data[col].dtype),
                    'unique_ratio': unique_ratio
                })
        
        # Sort by score
        target_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        if target_candidates:
            best_candidate = target_candidates[0]
            target_col = best_candidate['column']
            target_data = data[target_col]
            
            # Determine problem type
            if target_data.dtype in ['object', 'category'] or target_data.nunique() < 20:
                if target_data.nunique() == 2:
                    problem_type = "Binary Classification"
                else:
                    problem_type = "Multi-class Classification"
            else:
                problem_type = "Regression"
            
            # Analyze target characteristics
            target_characteristics = {
                "data_type": str(target_data.dtype),
                "unique_values": target_data.nunique(),
                "missing_count": target_data.isnull().sum(),
                "missing_percentage": target_data.isnull().sum() / len(target_data)
            }
            
            if problem_type in ["Binary Classification", "Multi-class Classification"]:
                value_counts = target_data.value_counts()
                target_characteristics["class_distribution"] = value_counts.to_dict()
                if len(value_counts) == 2:
                    ratio = min(value_counts) / max(value_counts)
                    target_characteristics["class_balance_ratio"] = ratio
                    if ratio < 0.3:
                        target_characteristics["imbalance_warning"] = True
            
            self._log_decision("TARGET_IDENTIFICATION", 
                             f"Identified target: {target_col} with score {best_candidate['score']}", 
                             f"Problem type: {problem_type}")
            
            return {
                "problem_type": problem_type,
                "target_variable": target_col,
                "target_characteristics": target_characteristics,
                "target_candidates": target_candidates,
                "supervised": True
            }
        else:
            # No clear target found - unsupervised learning
            self._log_decision("TARGET_IDENTIFICATION", 
                             "No clear target variable identified", 
                             "Unsupervised learning problem")
            
            # Determine unsupervised problem type
            if self._has_natural_clusters(data):
                problem_type = "Clustering"
            elif self._has_temporal_patterns(data):
                problem_type = "Time Series"
            else:
                problem_type = "Dimensionality Reduction"
            
            return {
                "problem_type": problem_type,
                "target_variable": None,
                "supervised": False
            }
    
    def _has_natural_clusters(self, data: pd.DataFrame) -> bool:
        """Check if data has natural clustering patterns"""
        # Simple heuristic: check if numeric columns have reasonable distributions
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            return True
        return False
    
    def _has_temporal_patterns(self, data: pd.DataFrame) -> bool:
        """Check if data has temporal patterns"""
        # Check for date/time columns
        date_indicators = ['date', 'time', 'year', 'month', 'day', 'hour', 'minute']
        for col in data.columns:
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in date_indicators):
                return True
        return False
    
    def _prepare_deep_clean_recommendations(self, problem_analysis: Dict) -> Dict:
        """Stage 4: Prepare recommendations for task-specific cleaning"""
        self._log_step("DEEP_CLEAN_PREP", "Preparing task-specific cleaning recommendations")
        
        recommendations = {
            "target_handling": {},
            "feature_handling": {},
            "validation_strategy": {},
            "algorithm_recommendations": []
        }
        
        if problem_analysis["supervised"]:
            target_col = problem_analysis["target_variable"]
            target_chars = problem_analysis["target_characteristics"]
            
            # Target handling recommendations
            if target_chars["missing_count"] > 0:
                recommendations["target_handling"]["missing_targets"] = "drop"
                self._log_decision("TARGET_HANDLING", 
                                 f"Found {target_chars['missing_count']} missing target values", 
                                 "Recommend dropping missing target rows")
            
            # Problem-specific recommendations
            if problem_analysis["problem_type"] == "Binary Classification":
                recommendations["validation_strategy"]["cv_method"] = "stratified"
                recommendations["algorithm_recommendations"] = ["RandomForest", "LogisticRegression", "SVM"]
                
                if target_chars.get("imbalance_warning"):
                    recommendations["target_handling"]["class_imbalance"] = "handle"
                    self._log_decision("CLASS_IMBALANCE", 
                                     f"Class balance ratio: {target_chars['class_balance_ratio']:.2f}", 
                                     "Recommend handling class imbalance")
            
            elif problem_analysis["problem_type"] == "Multi-class Classification":
                recommendations["validation_strategy"]["cv_method"] = "stratified"
                recommendations["algorithm_recommendations"] = ["RandomForest", "SVM", "KNN"]
            
            elif problem_analysis["problem_type"] == "Regression":
                recommendations["validation_strategy"]["cv_method"] = "kfold"
                recommendations["algorithm_recommendations"] = ["RandomForest", "XGBoost", "LinearRegression"]
        
        else:  # Unsupervised
            if problem_analysis["problem_type"] == "Clustering":
                recommendations["algorithm_recommendations"] = ["KMeans", "DBSCAN", "Hierarchical"]
                recommendations["validation_strategy"]["evaluation"] = "silhouette_score"
            
            elif problem_analysis["problem_type"] == "Time Series":
                recommendations["algorithm_recommendations"] = ["ARIMA", "Prophet", "LSTM"]
                recommendations["validation_strategy"]["cv_method"] = "timeseries"
        
        return recommendations 