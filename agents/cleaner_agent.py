import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import openai
import os
import dotenv
dotenv.load_dotenv()


class CleanerAgent:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def clean_data(self, data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Clean and preprocess the dataset"""
        cleaned_state = {
            "original_shape": data.shape,
            "cleaned_shape": None,
            "missing_values_info": {},
            "outliers_info": [],
            "data_type_fixes": {},
            "cleaning_summary": None
        }
        
        try:
            print(f"📋 Original data shape: {data.shape}")
            
            # Handle missing values
            data_cleaned = self._handle_missing_values(data)
            cleaned_state["missing_values_info"] = self.missing_values_log
            
            # Handle outliers
            data_cleaned = self._handle_outliers(data_cleaned)
            cleaned_state["outliers_info"] = self.outliers_log
            
            # Fix data types
            data_cleaned = self._fix_data_types(data_cleaned)
            cleaned_state["data_type_fixes"] = self.dtype_fixes_log
            
            # Generate cleaning summary using LLM
            cleaning_summary = self._generate_cleaning_summary(data, data_cleaned)
            cleaned_state["cleaning_summary"] = cleaning_summary.get("summary", "LLM summary not generated.")
            
            # Update cleaned shape
            cleaned_state["cleaned_shape"] = data_cleaned.shape
            
            print(f"✅ Cleaned data shape: {data_cleaned.shape}")
            return data_cleaned, cleaned_state
            
        except Exception as e:
            print(f"❌ Data cleaning failed: {str(e)}")
            raise
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        data_copy = data.copy()
        self.missing_values_log = {}
        
        for column in data_copy.columns:
            if data_copy[column].isnull().sum() > 0:
                if data_copy[column].dtype in ['object', 'category']:
                    # Fill categorical with mode
                    data_copy[column] = data_copy[column].fillna(data_copy[column].mode()[0])
                    self.missing_values_log[column] = "mode"
                else:
                    # Fill numerical with median
                    data_copy[column] = data_copy[column].fillna(data_copy[column].median())
                    self.missing_values_log[column] = "median"

        print(f"🔧 Missing values handled: {self.missing_values_log}")
        return data_copy
    
    def _handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method"""
        data_copy = data.copy()
        numeric_columns = data_copy.select_dtypes(include=[np.number]).columns
        self.outliers_log = []
        
        for column in numeric_columns:
            Q1 = data_copy[column].quantile(0.25)
            Q3 = data_copy[column].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            original = data_copy[column].copy()
            capped = np.clip(original, lower, upper)
            if not original.equals(capped):
                self.outliers_log.append(column)
                data_copy[column] = capped
        
        print(f"🔧 Outliers for columns: {self.outliers_log}")
        return data_copy
    
    def _fix_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fix data types automatically"""
        data_copy = data.copy()
        self.dtype_fixes_log = {}
        
        for column in data_copy.columns:
            original_dtype = str(data_copy[column].dtype)
            try:
                converted = pd.to_numeric(data_copy[column], errors='ignore')
                if converted.dtype != data_copy[column].dtype:
                    self.dtype_fixes_log[column] = f"{original_dtype} → {str(converted.dtype)}"
                    data_copy[column] = converted
            except Exception:
                continue
        
        print(f"🔧 Data types fixed: {self.dtype_fixes_log}")
        return data_copy
    
    def _generate_cleaning_summary(self, original_data: pd.DataFrame, cleaned_data: pd.DataFrame) -> dict:
        """Generate cleaning summary using LLM"""
        try:
            prompt = f"""
            Analyze the data cleaning process:
            - Original shape: {original_data.shape}
            - Cleaned shape: {cleaned_data.shape}
            - Missing values before: {original_data.isnull().sum().sum()}
            - Missing values after: {cleaned_data.isnull().sum().sum()}
            
            Provide a brief summary of the cleaning process and its impact.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )

            print("✅ GPT-4o summary generated.")
            print("📄 Summary Content:", response.choices[0].message.content)
            
            return {
                "summary": response.choices[0].message.content,
                "missing_values_removed": int(original_data.isnull().sum().sum()),
                "outliers_handled": True,
                "data_types_optimized": True
            }
            
        except Exception as e:
            return {
                "summary": f"Error generating summary: {str(e)}",
                "missing_values_removed": int(original_data.isnull().sum().sum()),
                "outliers_handled": True,
                "data_types_optimized": True
            }