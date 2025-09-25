import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import openai
import os
import dotenv
from datetime import datetime
from typing import Dict, List, Tuple, Any
import streamlit as st
dotenv.load_dotenv()


class CleanerAgent:
    def __init__(self):
        # Try Streamlit secrets first, fallback to environment variable
        api_key = None
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except:
            api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set it in Streamlit secrets or environment variables.")
        
        self.openai_client = openai.OpenAI(api_key=api_key)
        # Enhanced transparency tracking
        self.decision_log = []
        self.step_log = []
        self.explanations = {}
        self.ml_concepts_used = set()
        
    def clean_data(self, data: pd.DataFrame, cleaning_options: dict = None) -> tuple[pd.DataFrame, dict]:
        """Clean and preprocess the dataset with full transparency and explanations"""
        # Initialize transparency tracking
        self.decision_log = []
        self.step_log = []
        self.explanations = {}
        self.ml_concepts_used = set()
        
        # Set excluded columns from cleaning options
        if cleaning_options and 'exclude_from_imputation' in cleaning_options:
            self.exclude_from_imputation = cleaning_options['exclude_from_imputation']
        else:
            self.exclude_from_imputation = []
        
        self._log_step("🚀 Starting Data Cleaning Process", {
            "original_shape": data.shape,
            "columns": list(data.columns),
            "data_types": data.dtypes.to_dict(),
            "excluded_columns": self.exclude_from_imputation
        })
        
        cleaned_state = {
            "original_shape": data.shape,
            "cleaned_shape": None,
            "missing_values_info": {},
            "outliers_info": [],
            "data_type_fixes": {},
            "cleaning_summary": None,
            # Enhanced transparency fields
            "decision_log": [],
            "step_log": [],
            "explanations": {},
            "ml_concepts_used": [],
            "educational_insights": {}
        }
        
        # Ensure cleaning_options is not None
        if cleaning_options is None:
            cleaning_options = {}
        
        try:
            print(f"📋 Original data shape: {data.shape}")
            
            # Step 1: Analyze data quality
            self._analyze_data_quality(data)
            
            # Step 2: Handle missing values (target-aware)
            self._log_step("🔍 Analyzing Missing Values", {"step": "missing_values_analysis"})
            data_cleaned = self._handle_missing_values(data, cleaning_options.get("missing_value_strategy", "Agent Decides"), cleaning_options.get("column_missing_strategies", {}))
            cleaned_state["missing_values_info"] = self.missing_values_log
            
            # Step 3: Handle outliers (target-aware)
            self._log_step("📊 Analyzing Outliers", {"step": "outlier_analysis"})
            data_cleaned = self._handle_outliers(data_cleaned, cleaning_options.get("outlier_strategy", "Agent Decides"), cleaning_options.get("iqr_multiplier", 1.5))
            cleaned_state["outliers_info"] = self.outliers_log
            
            # Step 4: Fix data types (target-aware)
            self._log_step("🔧 Optimizing Data Types", {"step": "data_type_optimization"})
            data_cleaned = self._fix_data_types(data_cleaned, cleaning_options.get("data_type_strategy", "Agent Decides"), cleaning_options.get("manual_type_conversions", {}))
            cleaned_state["data_type_fixes"] = self.dtype_fixes_log
            
            # Step 5: Handle target column specifically (NEW)
            if cleaning_options.get("drop_missing_targets", False):
                data_cleaned = self._handle_target_column(data_cleaned, cleaning_options)
            
            # Step 6: Generate comprehensive cleaning summary
            self._log_step("📝 Generating Cleaning Report", {"step": "summary_generation"})
            cleaning_summary = self._generate_cleaning_summary(data, data_cleaned)
            cleaned_state["cleaning_summary"] = cleaning_summary.get("summary", "LLM summary not generated.")
            
            # Step 7: Compile transparency information
            cleaned_state["decision_log"] = self.decision_log
            cleaned_state["step_log"] = self.step_log
            cleaned_state["explanations"] = self.explanations
            cleaned_state["ml_concepts_used"] = list(self.ml_concepts_used)
            cleaned_state["educational_insights"] = self._generate_educational_insights()
            
            # Update cleaned shape
            cleaned_state["cleaned_shape"] = data_cleaned.shape
            
            self._log_step("✅ Data Cleaning Completed", {
                "final_shape": data_cleaned.shape,
                "total_decisions": len(self.decision_log),
                "concepts_learned": len(self.ml_concepts_used)
            })
            
            print(f"✅ Cleaned data shape: {data_cleaned.shape}")
            return data_cleaned, cleaned_state
            
        except Exception as e:
            print(f"❌ Error during data cleaning: {str(e)}")
            raise
    
    def _log_step(self, description: str, details: dict = None):
        """Log each step of the cleaning process with timestamp and details"""
        step_entry = {
            "timestamp": datetime.now().isoformat(),
            "description": description,
            "details": details or {}
        }
        self.step_log.append(step_entry)
        
    def _log_decision(self, decision_type: str, column: str, reasoning: str, action_taken: str, alternatives_considered: List[str] = None):
        """Log decision rationale for transparency"""
        decision_entry = {
            "timestamp": datetime.now().isoformat(),
            "decision_type": decision_type,
            "column": column,
            "reasoning": reasoning,
            "action_taken": action_taken,
            "alternatives_considered": alternatives_considered or [],
            "ml_concept": self._get_ml_concept_for_decision(decision_type)
        }
        self.decision_log.append(decision_entry)
        
        # Track ML concepts used
        if decision_entry["ml_concept"]:
            self.ml_concepts_used.add(decision_entry["ml_concept"])
    
    def _get_ml_concept_for_decision(self, decision_type: str) -> str:
        """Map decision types to ML concepts for educational purposes"""
        concept_mapping = {
            "missing_value_imputation": "Data Imputation",
            "outlier_handling": "Outlier Detection & Treatment",
            "data_type_conversion": "Feature Engineering",
            "data_quality_analysis": "Exploratory Data Analysis"
        }
        return concept_mapping.get(decision_type, "Data Preprocessing")
    
    def _analyze_data_quality(self, data: pd.DataFrame):
        """Comprehensive data quality analysis with transparency"""
        self._log_step("🔍 Conducting Data Quality Assessment", {})
        
        # Missing values analysis
        missing_analysis = {}
        total_missing = 0
        for col in data.columns:
            missing_count = data[col].isnull().sum()
            missing_pct = (missing_count / len(data)) * 100
            if missing_count > 0:
                missing_analysis[col] = {
                    "count": int(missing_count),
                    "percentage": round(missing_pct, 2),
                    "severity": "Low" if missing_pct < 5 else "Medium" if missing_pct < 20 else "High"
                }
                total_missing += missing_count
        
        self._log_step("📊 Missing Values Analysis Complete", {
            "total_missing_values": int(total_missing),
            "columns_affected": len(missing_analysis),
            "missing_analysis": missing_analysis
        })
        
        # Data types analysis
        dtype_analysis = {}
        for col in data.columns:
            dtype_info = {
                "current_type": str(data[col].dtype),
                "unique_values": int(data[col].nunique()),
                "suggested_type": self._suggest_optimal_dtype(data[col])
            }
            dtype_analysis[col] = dtype_info
            
        self._log_step("🏷️ Data Types Analysis Complete", {"dtype_analysis": dtype_analysis})
        
        return {
            "missing_analysis": missing_analysis,
            "dtype_analysis": dtype_analysis
        }
    
    def _suggest_optimal_dtype(self, series: pd.Series) -> str:
        """Suggest optimal data type for a column"""
        if series.dtype == 'object':
            # Check if it could be numeric
            try:
                pd.to_numeric(series.dropna(), errors='raise')
                return "numeric"
            except:
                # Check if it could be datetime
                try:
                    pd.to_datetime(series.dropna(), errors='raise')
                    return "datetime"
                except:
                    # Check cardinality for categorical
                    unique_ratio = series.nunique() / len(series)
                    if unique_ratio < 0.5:
                        return "categorical"
                    else:
                        return "text"
        elif series.dtype in ['int64', 'float64']:
            unique_ratio = series.nunique() / len(series)
            if unique_ratio < 0.1 and series.nunique() < 20:
                return "categorical"
            else:
                return "numeric"
        else:
            return str(series.dtype)
    
    def _handle_target_column(self, data: pd.DataFrame, cleaning_options: dict) -> pd.DataFrame:
        """Handle target column specifically - drop missing targets, don't impute target"""
        self._log_step("🎯 Handling Target Column", {"step": "target_column_handling"})
        
        exclude_from_imputation = cleaning_options.get("exclude_from_imputation", [])
        drop_missing_targets = cleaning_options.get("drop_missing_targets", False)
        
        data_copy = data.copy()
        
        # Handle excluded columns (usually target columns)
        for col in exclude_from_imputation:
            if col in data_copy.columns:
                missing_count = data_copy[col].isnull().sum()
                if missing_count > 0:
                    if drop_missing_targets:
                        # Drop rows with missing targets
                        original_shape = data_copy.shape
                        data_copy = data_copy.dropna(subset=[col])
                        dropped_count = original_shape[0] - data_copy.shape[0]
                        
                        self._log_decision("TARGET_MISSING_HANDLING", 
                                         col,
                                         f"Target column '{col}' has {missing_count} missing values", 
                                         f"Dropped {dropped_count} rows with missing targets",
                                         ["Impute with mean", "Impute with median", "Keep missing values"])
                        
                        self._log_step("🎯 Target Missing Values Handled", {
                            "target_column": col,
                            "missing_count": missing_count,
                            "dropped_rows": dropped_count,
                            "remaining_shape": data_copy.shape
                        })
                        
                        print(f"🔧 Dropped {dropped_count} rows with missing targets in '{col}'")
                    else:
                        # Just log the missing values without imputing
                        self._log_decision("TARGET_MISSING_HANDLING", 
                                         col,
                                         f"Target column '{col}' has {missing_count} missing values", 
                                         "Missing targets preserved (no imputation)",
                                         ["Drop rows", "Impute with mode", "Keep missing values"])
                        
                        print(f"⚠️  Target column '{col}' has {missing_count} missing values (not imputed)")
        
        return data_copy
    
    def _handle_missing_values(self, data: pd.DataFrame, strategy: str = "Agent Decides", column_strategies: dict = None) -> pd.DataFrame:
        """Handle missing values with comprehensive transparency and educational explanations"""
        data_copy = data.copy()
        self.missing_values_log = {}
        
        if column_strategies is None:
            column_strategies = {}
        
        # Get excluded columns (target columns that shouldn't be imputed)
        exclude_from_imputation = getattr(self, 'exclude_from_imputation', [])
        
        self._log_step("🔍 Starting Missing Values Treatment", {
            "total_missing": int(data.isnull().sum().sum()),
            "strategy": strategy,
            "columns_with_missing": data.columns[data.isnull().any()].tolist(),
            "excluded_columns": exclude_from_imputation
        })
        
        # Check if any column has a "Drop Rows" strategy
        drop_rows_columns = [col for col, strat in column_strategies.items() if strat == "Drop Rows"]
        if drop_rows_columns or strategy == "Drop Rows":
            initial_rows = data_copy.shape[0]
            if drop_rows_columns:
                # Drop rows with missing values in specific columns
                reasoning = f"Removing rows with missing values in critical columns: {drop_rows_columns}. This ensures data completeness for essential features."
                self._log_decision("missing_value_imputation", str(drop_rows_columns), reasoning, "Drop rows", 
                                 ["Impute with mean", "Impute with median", "Impute with mode"])
                data_copy.dropna(subset=drop_rows_columns, inplace=True)
                self.missing_values_log["strategy_applied"] = f"Dropped rows with missing values in: {drop_rows_columns}"
            else:
                # Global drop rows strategy
                reasoning = "Removing all rows with any missing values to ensure complete data records."
                self._log_decision("missing_value_imputation", "all_columns", reasoning, "Drop all incomplete rows",
                                 ["Global imputation", "Column-specific imputation"])
                data_copy.dropna(inplace=True)
                self.missing_values_log["strategy_applied"] = "Dropped rows with any missing values"
            
            rows_removed = initial_rows - data_copy.shape[0]
            self.missing_values_log["rows_removed"] = rows_removed
            self._log_step(f"📉 Removed {rows_removed} rows with missing values", {"rows_removed": rows_removed})
        
        # Handle remaining columns with imputation strategies
        for column in data_copy.columns:
            missing_count = data_copy[column].isnull().sum()
            if missing_count > 0:
                # Skip excluded columns (target columns)
                if column in exclude_from_imputation:
                    self._log_decision("missing_value_imputation", column, 
                                     f"Column '{column}' is excluded from imputation (likely target variable)", 
                                     "Skip imputation", ["Impute with mean", "Impute with median", "Impute with mode"])
                    self.missing_values_log[column] = "skipped (excluded from imputation)"
                    print(f"⚠️  Skipping imputation for excluded column '{column}' ({missing_count} missing values)")
                    continue
                
                # Use column-specific strategy if available, otherwise use global strategy
                col_strategy = column_strategies.get(column, strategy)
                
                if col_strategy == "Drop Rows":
                    continue  # Already handled above
                
                # Analyze column characteristics for decision making
                column_info = self._analyze_column_characteristics(data_copy[column])
                
                if data_copy[column].dtype in ['object', 'category']:
                    # For categorical data
                    if col_strategy in ["Agent Decides", "Impute with Mode"]:
                        mode_val = data_copy[column].mode()
                        if len(mode_val) > 0:
                            reasoning = f"Imputing '{column}' with mode ('{mode_val[0]}') because it's categorical data and mode preserves the most common pattern. Missing: {missing_count} values ({(missing_count/len(data_copy)*100):.1f}%)"
                            self._log_decision("missing_value_imputation", column, reasoning, f"Impute with mode: {mode_val[0]}", 
                                             ["Drop rows", "Create 'Unknown' category"])
                            data_copy[column] = data_copy[column].fillna(mode_val[0])
                            self.missing_values_log[column] = f"mode (strategy: {col_strategy})"
                    else:
                        self.missing_values_log[column] = f"skipped (categorical, strategy: {col_strategy})"
                else:
                    # For numerical data
                    if col_strategy == "Impute with Mean":
                        mean_val = data_copy[column].mean()
                        reasoning = f"Imputing '{column}' with mean ({mean_val:.2f}) because user selected mean strategy. Note: Mean is sensitive to outliers but preserves the overall average."
                        self._log_decision("missing_value_imputation", column, reasoning, f"Impute with mean: {mean_val:.2f}",
                                         ["Impute with median", "Impute with mode", "Drop rows"])
                        data_copy[column] = data_copy[column].fillna(mean_val)
                        self.missing_values_log[column] = f"mean (strategy: {col_strategy})"
                    elif col_strategy == "Impute with Median":
                        median_val = data_copy[column].median()
                        reasoning = f"Imputing '{column}' with median ({median_val}) because user selected median strategy. Median is robust to outliers and good for skewed distributions."
                        self._log_decision("missing_value_imputation", column, reasoning, f"Impute with median: {median_val}",
                                         ["Impute with mean", "Impute with mode", "Drop rows"])
                        data_copy[column] = data_copy[column].fillna(median_val)
                        self.missing_values_log[column] = f"median (strategy: {col_strategy})"
                    elif col_strategy == "Impute with Mode":
                        mode_val = data_copy[column].mode()
                        if len(mode_val) > 0:
                            reasoning = f"Imputing '{column}' with mode ({mode_val[0]}) because user selected mode strategy. Mode preserves the most frequent value pattern."
                            self._log_decision("missing_value_imputation", column, reasoning, f"Impute with mode: {mode_val[0]}",
                                             ["Impute with mean", "Impute with median", "Drop rows"])
                            data_copy[column] = data_copy[column].fillna(mode_val[0])
                            self.missing_values_log[column] = f"mode (strategy: {col_strategy})"
                    elif col_strategy == "Agent Decides":
                        # Intelligent decision based on data characteristics
                        if column_info["is_skewed"]:
                            median_val = data_copy[column].median()
                            reasoning = f"Imputing '{column}' with median ({median_val}) because the data is skewed (skewness: {column_info['skewness']:.2f}). Median is more robust than mean for skewed distributions."
                            action = f"Impute with median: {median_val}"
                        else:
                            mean_val = data_copy[column].mean()
                            reasoning = f"Imputing '{column}' with mean ({mean_val:.2f}) because the data is approximately normal (skewness: {column_info['skewness']:.2f}). Mean preserves the central tendency."
                            action = f"Impute with mean: {mean_val:.2f}"
                        
                        self._log_decision("missing_value_imputation", column, reasoning, action,
                                         ["Impute with mode", "Drop rows", "Forward fill"])
                        data_copy[column] = data_copy[column].fillna(median_val if column_info["is_skewed"] else mean_val)
                        self.missing_values_log[column] = f"intelligent ({'median' if column_info['is_skewed'] else 'mean'})"
                    else:
                        self.missing_values_log[column] = f"skipped (strategy: {col_strategy})"
        
        return data_copy
    
    def _analyze_column_characteristics(self, series: pd.Series) -> dict:
        """Analyze column characteristics to make informed decisions"""
        characteristics = {}
        
        if pd.api.types.is_numeric_dtype(series):
            # Calculate skewness
            skewness = series.skew()
            characteristics["skewness"] = skewness
            characteristics["is_skewed"] = abs(skewness) > 1.0
            characteristics["mean"] = series.mean()
            characteristics["median"] = series.median()
            characteristics["std"] = series.std()
        
        characteristics["missing_count"] = series.isnull().sum()
        characteristics["missing_percentage"] = (series.isnull().sum() / len(series)) * 100
        characteristics["unique_count"] = series.nunique()
        characteristics["data_type"] = str(series.dtype)
        
        return characteristics
    
    def _handle_outliers(self, data: pd.DataFrame, strategy: str = "Agent Decides", iqr_multiplier: float = 1.5) -> pd.DataFrame:
        """Handle outliers with comprehensive transparency and educational explanations"""
        data_copy = data.copy()
        numeric_columns = data_copy.select_dtypes(include=[np.number]).columns
        self.outliers_log = []
        
        self._log_step("📊 Starting Outlier Detection and Treatment", {
            "numeric_columns": list(numeric_columns),
            "strategy": strategy,
            "iqr_multiplier": iqr_multiplier
        })
        
        if strategy == "No Outlier Handling":
            reasoning = "User selected no outlier handling. Outliers will remain in the dataset, which may affect model performance but preserves data authenticity."
            self._log_decision("outlier_handling", "all_columns", reasoning, "No action taken", 
                             ["Cap outliers", "Remove outliers", "Transform data"])
            self.outliers_log.append("No outlier handling applied (user selected)")
            print(f"🔧 Outliers handled: {self.outliers_log}")
            return data_copy
        
        for column in numeric_columns:
            # Calculate outlier boundaries using IQR method
            Q1 = data_copy[column].quantile(0.25)
            Q3 = data_copy[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - iqr_multiplier * IQR
            upper = Q3 + iqr_multiplier * IQR
            
            # Count outliers
            outliers_lower = (data_copy[column] < lower).sum()
            outliers_upper = (data_copy[column] > upper).sum()
            total_outliers = outliers_lower + outliers_upper
            
            if total_outliers > 0:
                outlier_percentage = (total_outliers / len(data_copy)) * 100
                
                if strategy == "Cap Outliers (IQR)":
                    reasoning = f"Capping outliers in '{column}' using IQR method (multiplier: {iqr_multiplier}). Found {total_outliers} outliers ({outlier_percentage:.1f}%). Capping preserves data points while reducing extreme values."
                    self._log_decision("outlier_handling", column, reasoning, f"Cap {total_outliers} outliers",
                                     ["Remove outliers", "No action", "Log transformation"])
                    
                    original = data_copy[column].copy()
                    capped = np.clip(original, lower, upper)
                    if not original.equals(capped):
                        self.outliers_log.append(f"{column}: capped (IQR×{iqr_multiplier})")
                        data_copy[column] = capped
                        
                elif strategy == "Remove Outliers (IQR)":
                    reasoning = f"Removing outliers in '{column}' using IQR method. Found {total_outliers} outliers ({outlier_percentage:.1f}%). Removal ensures clean data but reduces sample size."
                    self._log_decision("outlier_handling", column, reasoning, f"Remove {total_outliers} outliers",
                                     ["Cap outliers", "No action", "Transform data"])
                    
                    initial_rows = data_copy.shape[0]
                    data_copy = data_copy[~((data_copy[column] < lower) | (data_copy[column] > upper))]
                    if initial_rows != data_copy.shape[0]:
                        self.outliers_log.append(f"{column}: removed (IQR×{iqr_multiplier})")
                        
                elif strategy == "Agent Decides": 
                    # Intelligent decision based on outlier severity
                    if outlier_percentage < 5.0:
                        reasoning = f"Capping outliers in '{column}' because outlier percentage ({outlier_percentage:.1f}%) is low. Capping preserves sample size while reducing extreme influence."
                        action = f"Cap {total_outliers} outliers (low percentage)"
                    else:
                        reasoning = f"Capping outliers in '{column}' because outlier percentage ({outlier_percentage:.1f}%) is significant. Removing would lose too much data."
                        action = f"Cap {total_outliers} outliers (high percentage - removal not recommended)"
                    
                    self._log_decision("outlier_handling", column, reasoning, action,
                                     ["Remove outliers", "No action", "Robust scaling"])
                    
                    original = data_copy[column].copy()
                    capped = np.clip(original, lower, upper)
                    if not original.equals(capped):
                        self.outliers_log.append(f"{column}: capped by agent (IQR×{iqr_multiplier})")
                        data_copy[column] = capped
        
        self._log_step("✅ Outlier Treatment Complete", {"columns_processed": len(numeric_columns)})
        print(f"🔧 Outliers handled: {self.outliers_log}")
        return data_copy
    
    def _fix_data_types(self, data: pd.DataFrame, strategy: str = "Agent Decides", manual_conversions: dict = None) -> pd.DataFrame:
        """Fix data types with comprehensive transparency and educational explanations"""
        data_copy = data.copy()
        self.dtype_fixes_log = {}
        
        if manual_conversions is None:
            manual_conversions = {}
        
        self._log_step("🔧 Starting Data Type Optimization", {
            "strategy": strategy,
            "manual_conversions": manual_conversions,
            "current_dtypes": data_copy.dtypes.to_dict()
        })
        
        # Apply manual conversions first
        for column, conversion_type in manual_conversions.items():
            if column in data_copy.columns:
                original_dtype = str(data_copy[column].dtype)
                try:
                    if conversion_type == "Force Numeric":
                        reasoning = f"Converting '{column}' to numeric as manually specified. This enables mathematical operations and numerical analysis."
                        self._log_decision("data_type_conversion", column, reasoning, f"Manual conversion: {original_dtype} → numeric",
                                         ["Keep as text", "Convert to categorical"])
                        data_copy[column] = pd.to_numeric(data_copy[column], errors='coerce')
                        self.dtype_fixes_log[column] = f"{original_dtype} → numeric (manual)"
                    elif conversion_type == "Force Categorical":
                        reasoning = f"Converting '{column}' to categorical as manually specified. This saves memory and enables category-specific operations."
                        self._log_decision("data_type_conversion", column, reasoning, f"Manual conversion: {original_dtype} → categorical",
                                         ["Keep as text", "Convert to numeric"])
                        data_copy[column] = data_copy[column].astype('category')
                        self.dtype_fixes_log[column] = f"{original_dtype} → category (manual)"
                    elif conversion_type == "Force DateTime":
                        reasoning = f"Converting '{column}' to datetime as manually specified. This enables time-based operations and analysis."
                        self._log_decision("data_type_conversion", column, reasoning, f"Manual conversion: {original_dtype} → datetime",
                                         ["Keep as text", "Convert to numeric"])
                        data_copy[column] = pd.to_datetime(data_copy[column], errors='coerce')
                        self.dtype_fixes_log[column] = f"{original_dtype} → datetime (manual)"
                    elif conversion_type == "Force Text":
                        reasoning = f"Converting '{column}' to text as manually specified. This preserves string operations and text analysis capabilities."
                        self._log_decision("data_type_conversion", column, reasoning, f"Manual conversion: {original_dtype} → string",
                                         ["Keep current type", "Convert to categorical"])
                        data_copy[column] = data_copy[column].astype('str')
                        self.dtype_fixes_log[column] = f"{original_dtype} → str (manual)"
                except Exception as e:
                    self.dtype_fixes_log[column] = f"Manual conversion failed: {str(e)}"
        
        # If strategy is "Manual Only", skip automatic conversions
        if strategy == "Manual Only":
            self._log_step("✅ Data Type Optimization Complete (Manual Only)", {"conversions": len(self.dtype_fixes_log)})
            print(f"🔧 Data types fixed (manual only): {self.dtype_fixes_log}")
            return data_copy
        
        # Apply automatic conversions for remaining columns
        if strategy == "Convert Automatically" or strategy == "Agent Decides":
            for column in data_copy.columns:
                # Skip columns that were manually converted
                if column in manual_conversions:
                    continue
                    
                original_dtype = str(data_copy[column].dtype)
                
                # Only try to convert to numeric if not already numeric
                if not pd.api.types.is_numeric_dtype(data_copy[column]):
                    try:
                        converted = pd.to_numeric(data_copy[column], errors='ignore')
                        if converted.dtype != data_copy[column].dtype:
                            reasoning = f"Converting '{column}' from {original_dtype} to numeric because values can be interpreted as numbers. This enables mathematical operations."
                            self._log_decision("data_type_conversion", column, reasoning, f"Auto conversion: {original_dtype} → {str(converted.dtype)}",
                                             ["Keep as text", "Convert to categorical"])
                            self.dtype_fixes_log[column] = f"{original_dtype} → {str(converted.dtype)} (auto)"
                            data_copy[column] = converted
                            continue # Move to next column if successfully converted
                    except Exception:
                        pass # Not convertible to numeric
                
                # Only try to convert to datetime if object or category
                if data_copy[column].dtype in ['object', 'category']:
                    try:
                        converted = pd.to_datetime(data_copy[column], errors='coerce')
                        # Only convert if a significant portion of values are valid dates
                        valid_date_ratio = converted.notna().sum() / len(converted)
                        if valid_date_ratio > 0.8 and converted.dtype != data_copy[column].dtype:
                            reasoning = f"Converting '{column}' to datetime because {valid_date_ratio*100:.1f}% of values are valid dates. This enables time-based analysis."
                            self._log_decision("data_type_conversion", column, reasoning, f"Auto conversion: {original_dtype} → datetime",
                                             ["Keep as text", "Convert to categorical"])
                            self.dtype_fixes_log[column] = f"{original_dtype} → {str(converted.dtype)} (auto)"
                            data_copy[column] = converted
                    except Exception:
                        continue # Cannot convert to datetime either
        
        self._log_step("✅ Data Type Optimization Complete", {"conversions": len(self.dtype_fixes_log)})
        print(f"🔧 Data types fixed: {self.dtype_fixes_log}")
        return data_copy
    
    def _generate_educational_insights(self) -> dict:
        """Generate educational insights about the cleaning process"""
        insights = {}
        
        # ML concepts explanations
        concept_explanations = {
            "Data Imputation": {
                "definition": "The process of filling in missing values in a dataset using statistical methods or domain knowledge.",
                "why_important": "Missing data can cause algorithms to fail or produce biased results. Proper imputation preserves data integrity.",
                "common_methods": ["Mean imputation", "Median imputation", "Mode imputation", "Forward fill", "Predictive imputation"],
                "best_practices": "Choose method based on data distribution: median for skewed data, mean for normal data, mode for categorical data."
            },
            "Outlier Detection & Treatment": {
                "definition": "Identifying and handling data points that significantly differ from other observations in the dataset.",
                "why_important": "Outliers can skew statistical measures and negatively impact model performance, especially for linear models.",
                "common_methods": ["IQR method", "Z-score method", "Isolation Forest", "Local Outlier Factor"],
                "best_practices": "Understand if outliers are errors or valid extreme values. Capping preserves data while removing extreme influence."
            },
            "Feature Engineering": {
                "definition": "The process of selecting, modifying, or creating features from raw data to improve model performance.",
                "why_important": "Proper data types enable appropriate algorithms and operations. Wrong types can cause errors or poor performance.",
                "common_methods": ["Type conversion", "Encoding categorical variables", "Scaling numerical features", "Creating derived features"],
                "best_practices": "Convert to appropriate types early in the pipeline. Use categorical type for memory efficiency with repeated strings."
            },
            "Exploratory Data Analysis": {
                "definition": "The process of analyzing and investigating data sets to summarize their main characteristics and discover patterns.",
                "why_important": "Understanding your data is crucial before applying machine learning algorithms. It reveals quality issues and informs preprocessing decisions.",
                "common_methods": ["Statistical summaries", "Data visualization", "Correlation analysis", "Distribution analysis"],
                "best_practices": "Always start with EDA. Look for missing values, outliers, distributions, and relationships between variables."
            }
        }
        
        # Add explanations for concepts used in this cleaning session
        for concept in self.ml_concepts_used:
            if concept in concept_explanations:
                insights[concept] = concept_explanations[concept]
        
        # Add decision summary
        insights["decision_summary"] = {
            "total_decisions": len(self.decision_log),
            "decision_types": list(set(d["decision_type"] for d in self.decision_log)),
            "columns_affected": len(set(d["column"] for d in self.decision_log)),
            "transparency_level": "Full transparency with step-by-step logging and rationale"
        }
        
        return insights
    
    def _generate_cleaning_summary(self, original_data: pd.DataFrame, cleaned_data: pd.DataFrame) -> dict:
        """Generate comprehensive cleaning summary using LLM with transparency focus"""
        try:
            # Prepare detailed information for LLM
            cleaning_stats = {
                "original_shape": original_data.shape,
                "cleaned_shape": cleaned_data.shape,
                "missing_before": int(original_data.isnull().sum().sum()),
                "missing_after": int(cleaned_data.isnull().sum().sum()),
                "decisions_made": len(self.decision_log),
                "concepts_used": list(self.ml_concepts_used)
            }
            
            prompt = f"""
            Analyze this transparent data cleaning process:
            
            Original Dataset: {cleaning_stats['original_shape']} (rows, columns)
            Cleaned Dataset: {cleaning_stats['cleaned_shape']} (rows, columns)
            Missing Values: {cleaning_stats['missing_before']} → {cleaning_stats['missing_after']}
            Decisions Made: {cleaning_stats['decisions_made']}
            ML Concepts Applied: {', '.join(cleaning_stats['concepts_used'])}
            
            Key Features of this Cleaning Process:
            - Full transparency with decision rationale logging
            - Step-by-step breakdown of all actions
            - Educational explanations of ML concepts
            - Intelligent agent decisions based on data characteristics
            
            Provide a brief educational summary emphasizing the transparency and learning aspects.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
 
            print("✅ GPT-4o educational summary generated.")
            print("📄 Summary Content:", response.choices[0].message.content)
            
            return {
                "summary": response.choices[0].message.content,
                "missing_values_removed": cleaning_stats["missing_before"],
                "outliers_handled": True,
                "data_types_optimized": True,
                "transparency_features": {
                    "decision_logging": True,
                    "step_tracking": True,
                    "educational_insights": True,
                    "ml_concept_explanations": True
                }
            }
            
        except Exception as e:
            return {
                "summary": f"Error generating summary: {str(e)}",
                "missing_values_removed": int(original_data.isnull().sum().sum()),
                "outliers_handled": True,
                "data_types_optimized": True,
                "transparency_features": {
                    "decision_logging": True,
                    "step_tracking": True,
                    "educational_insights": True,
                    "ml_concept_explanations": True
                }
            }