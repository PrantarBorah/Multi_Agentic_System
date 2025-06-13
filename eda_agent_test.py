








import os
import dotenv
import logging
import pandas as pd
from agents.eda_agent import EDAAgent

# 1. Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("eda_agent_test.log"), logging.StreamHandler()]
)

def main():
    # 2. Load environment and cleaned data
    dotenv.load_dotenv()
    data_path = "intermediate_data/cleaned_titanic.csv"
    df = pd.read_csv(data_path)
    logging.info(f"✅ Loaded cleaned data with shape: {df.shape}")

    # 3. Initialize EDAAgent
    agent = EDAAgent()
    logging.info("✅ Initialized EDAAgent")

    # 4. Step-by-step EDA with explicit logging
    logging.info("▶️ _generate_summary_stats")
    stats = agent._generate_summary_stats(df)
    logging.info(f"   Summary stats keys: {list(stats.keys())}")

    logging.info("▶️ _analyze_data_types")
    dtypes = agent._analyze_data_types(df)
    logging.info(f"   Columns: {dtypes['total_columns']}")

    logging.info("▶️ _compute_correlations")
    corrs = agent._compute_correlations(df)
    logging.info(f"   High correlations: {corrs['high_correlations']}")

    logging.info("▶️ _create_visualizations")
    vizs = agent._create_visualizations(df)
    logging.info(f"   Generated visualizations: {vizs}")

    logging.info("▶️ _detect_target_and_problem_type")
    target, problem = agent._detect_target_and_problem_type(df)
    logging.info(f"   Target: {target}, Problem type: {problem}")

    logging.info("▶️ _generate_ai_insights")
    insights = agent._generate_ai_insights(df, {
        'problem_type': problem,
        'target_variable': target,
        'correlations': corrs
    })
    logging.info(f"   AI insights summary: {insights.get('ai_insights')}")

    logging.info("✅ EDA workflow test completed")

if __name__ == "__main__":
    main()










# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats
# import os
# import warnings
# warnings.filterwarnings('ignore')

# # Optional OpenAI import for testing
# try:
#     import openai
#     OPENAI_AVAILABLE = True
# except ImportError:
#     OPENAI_AVAILABLE = False
#     print("⚠️  OpenAI not available. Running in test mode without LLM insights.")

# # Import the CleanerAgent for testing workflow
# try:
#     from cleaner_agent_test import CleanerAgent, create_sample_data
#     CLEANER_AVAILABLE = True
# except ImportError:
#     CLEANER_AVAILABLE = False
#     print("⚠️  CleanerAgent not available. Will create basic cleaned state for testing.")

# class EDAAgent:
#     def __init__(self):
#         if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
#             self.openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
#             self.use_openai = True
#         else:
#             self.openai_client = None
#             self.use_openai = False
#             print("🔄 Running EDA without OpenAI integration")
    
#     def perform_eda(self, cleaned_state: dict) -> dict:
#         """Perform comprehensive exploratory data analysis"""
#         data = cleaned_state["data"]
        
#         eda_results = {
#             "summary_stats": {},
#             "data_types": {},
#             "correlations": {},
#             "visualizations": [],
#             "insights": {},
#             "target_variable": None,
#             "problem_type": None
#         }
        
#         try:
#             print("📊 Generating summary statistics...")
#             eda_results["summary_stats"] = self._generate_summary_stats(data)
            
#             print("🔍 Analyzing data types...")
#             eda_results["data_types"] = self._analyze_data_types(data)
            
#             print("🔗 Computing correlations...")
#             eda_results["correlations"] = self._compute_correlations(data)
            
#             print("📈 Creating visualizations...")
#             eda_results["visualizations"] = self._create_visualizations(data)
            
#             print("🎯 Detecting target variable...")
#             eda_results["target_variable"], eda_results["problem_type"] = self._detect_target_and_problem_type(data)
            
#             print("🧠 Generating insights...")
#             eda_results["insights"] = self._generate_ai_insights(data, eda_results)
            
#             print("✅ EDA completed successfully!")
#             return eda_results
            
#         except Exception as e:
#             print(f"❌ EDA failed: {str(e)}")
#             raise
    
#     def _generate_summary_stats(self, data: pd.DataFrame) -> dict:
#         """Generate summary statistics"""
#         numeric_data = data.select_dtypes(include=[np.number])
#         categorical_data = data.select_dtypes(include=['object', 'category'])
        
#         print(f"   📋 Found {len(numeric_data.columns)} numeric and {len(categorical_data.columns)} categorical columns")
        
#         return {
#             "shape": data.shape,
#             "numeric_summary": numeric_data.describe().to_dict() if len(numeric_data.columns) > 0 else {},
#             "categorical_summary": {col: data[col].value_counts().head().to_dict() 
#                                   for col in categorical_data.columns},
#             "missing_values": data.isnull().sum().to_dict()
#         }
    
#     def _analyze_data_types(self, data: pd.DataFrame) -> dict:
#         """Analyze data types and suggest improvements"""
#         numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
#         categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
#         print(f"   🔢 Numeric columns: {numeric_cols}")
#         print(f"   📝 Categorical columns: {categorical_cols}")
        
#         return {
#             "dtypes": data.dtypes.to_dict(),
#             "numeric_columns": numeric_cols,
#             "categorical_columns": categorical_cols,
#             "total_columns": len(data.columns)
#         }
    
#     def _compute_correlations(self, data: pd.DataFrame) -> dict:
#         """Compute correlation matrix for numeric variables"""
#         numeric_data = data.select_dtypes(include=[np.number])
        
#         if len(numeric_data.columns) > 1:
#             corr_matrix = numeric_data.corr()
#             high_corrs = self._find_high_correlations(corr_matrix)
#             print(f"   🔗 Found {len(high_corrs)} high correlations (>0.7)")
            
#             return {
#                 "correlation_matrix": corr_matrix.to_dict(),
#                 "high_correlations": high_corrs
#             }
#         else:
#             print("   ⚠️  Not enough numeric columns for correlation analysis")
#             return {"correlation_matrix": {}, "high_correlations": []}
    
#     def _find_high_correlations(self, corr_matrix: pd.DataFrame, threshold: float = 0.7) -> list:
#         """Find highly correlated variable pairs"""
#         high_corrs = []
#         for i in range(len(corr_matrix.columns)):
#             for j in range(i+1, len(corr_matrix.columns)):
#                 corr_val = corr_matrix.iloc[i, j]
#                 if abs(corr_val) > threshold:
#                     high_corrs.append({
#                         "var1": corr_matrix.columns[i],
#                         "var2": corr_matrix.columns[j],
#                         "correlation": round(corr_val, 3)
#                     })
#         return high_corrs
    
#     def _create_visualizations(self, data: pd.DataFrame) -> list:
#         """Create and save visualizations"""
#         visualizations = []
        
#         try:
#             # Set style
#             plt.style.use('default')  # Use default style for better compatibility
            
#             # 1. Correlation heatmap
#             numeric_data = data.select_dtypes(include=[np.number])
#             if len(numeric_data.columns) > 1:
#                 plt.figure(figsize=(10, 8))
#                 corr_matrix = numeric_data.corr()
                
#                 # Create heatmap
#                 sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
#                            square=True, linewidths=0.5)
#                 plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
#                 plt.tight_layout()
                
#                 # Save plot
#                 filename = 'correlation_heatmap.png'
#                 plt.savefig(filename, dpi=300, bbox_inches='tight')
#                 plt.close()
#                 visualizations.append(filename)
#                 print(f"   📊 Saved correlation heatmap: {filename}")
            
#             # 2. Distribution plots for numeric variables
#             if len(numeric_data.columns) > 0:
#                 n_cols = min(len(numeric_data.columns), 4)
#                 fig, axes = plt.subplots(2, 2, figsize=(15, 10))
#                 axes = axes.ravel()
                
#                 for i, col in enumerate(numeric_data.columns[:4]):
#                     if i < 4:
#                         # Create histogram with KDE
#                         sns.histplot(data[col].dropna(), kde=True, ax=axes[i], alpha=0.7)
#                         axes[i].set_title(f'Distribution of {col}', fontweight='bold')
#                         axes[i].grid(True, alpha=0.3)
                
#                 # Hide empty subplots
#                 for i in range(len(numeric_data.columns), 4):
#                     axes[i].set_visible(False)
                
#                 plt.tight_layout()
#                 filename = 'distributions.png'
#                 plt.savefig(filename, dpi=300, bbox_inches='tight')
#                 plt.close()
#                 visualizations.append(filename)
#                 print(f"   📈 Saved distribution plots: {filename}")
            
#             # 3. Box plots for numeric variables
#             if len(numeric_data.columns) > 0:
#                 plt.figure(figsize=(12, 6))
#                 numeric_data.boxplot()
#                 plt.title('Box Plots - Outlier Detection', fontsize=14, fontweight='bold')
#                 plt.xticks(rotation=45)
#                 plt.tight_layout()
                
#                 filename = 'boxplots.png'
#                 plt.savefig(filename, dpi=300, bbox_inches='tight')
#                 plt.close()
#                 visualizations.append(filename)
#                 print(f"   📦 Saved box plots: {filename}")
            
#             # 4. Categorical variable plots
#             categorical_data = data.select_dtypes(include=['object', 'category'])
#             if len(categorical_data.columns) > 0:
#                 n_cat_cols = min(len(categorical_data.columns), 2)
#                 fig, axes = plt.subplots(1, n_cat_cols, figsize=(12, 5))
#                 if n_cat_cols == 1:
#                     axes = [axes]
                
#                 for i, col in enumerate(categorical_data.columns[:2]):
#                     value_counts = data[col].value_counts().head(10)
#                     axes[i].bar(range(len(value_counts)), value_counts.values)
#                     axes[i].set_title(f'Distribution of {col}', fontweight='bold')
#                     axes[i].set_xticks(range(len(value_counts)))
#                     axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
#                     axes[i].grid(True, alpha=0.3)
                
#                 plt.tight_layout()
#                 filename = 'categorical_distributions.png'
#                 plt.savefig(filename, dpi=300, bbox_inches='tight')
#                 plt.close()
#                 visualizations.append(filename)
#                 print(f"   📊 Saved categorical distributions: {filename}")
            
#             return visualizations
            
#         except Exception as e:
#             print(f"   ⚠️  Warning: Visualization creation failed: {str(e)}")
#             return []
    
#     def _detect_target_and_problem_type(self, data: pd.DataFrame) -> tuple:
#         """Detect likely target variable and problem type"""
#         # Simple heuristic: last column is often the target
#         target_column = data.columns[-1]
        
#         # Determine problem type based on target variable
#         unique_values = data[target_column].nunique()
        
#         if data[target_column].dtype in ['object', 'category']:
#             problem_type = "classification"
#         elif unique_values < 10 and unique_values > 1:
#             problem_type = "classification"
#         else:
#             problem_type = "regression"
        
#         print(f"   🎯 Detected target: '{target_column}' ({problem_type})")
#         print(f"   📊 Target has {unique_values} unique values")
        
#         return target_column, problem_type
    
#     def _generate_ai_insights(self, data: pd.DataFrame, eda_results: dict) -> dict:
#         """Generate insights about the dataset"""
#         try:
#             data_quality_score = self._calculate_data_quality_score(data)
#             recommendations = self._generate_recommendations(eda_results)
            
#             if self.use_openai:
#                 prompt = f"""
#                 Analyze this dataset and provide key insights:
#                 - Shape: {data.shape}
#                 - Columns: {list(data.columns)}
#                 - Problem type: {eda_results['problem_type']}
#                 - Target variable: {eda_results['target_variable']}
#                 - High correlations: {len(eda_results['correlations']['high_correlations'])}
#                 - Data quality score: {data_quality_score}
                
#                 Provide 3-5 key insights about this dataset for machine learning.
#                 """
                
#                 response = self.openai_client.chat.completions.create(
#                     model="gpt-4o",
#                     messages=[{"role": "user", "content": prompt}],
#                     max_tokens=300
#                 )
                
#                 ai_insights = response.choices[0].message.content
#             else:
#                 ai_insights = self._generate_basic_insights(data, eda_results, data_quality_score)
            
#             return {
#                 "ai_insights": ai_insights,
#                 "data_quality_score": data_quality_score,
#                 "recommendations": recommendations
#             }
            
#         except Exception as e:
#             return {
#                 "ai_insights": f"Error generating insights: {str(e)}",
#                 "data_quality_score": 0.5,
#                 "recommendations": ["Review data quality", "Check for missing values"]
#             }
    
#     def _generate_basic_insights(self, data: pd.DataFrame, eda_results: dict, quality_score: float) -> str:
#         """Generate basic insights without AI"""
#         insights = []
        
#         # Data shape insight
#         insights.append(f"Dataset contains {data.shape[0]} rows and {data.shape[1]} columns")
        
#         # Data quality insight
#         if quality_score >= 0.9:
#             insights.append("Data quality is excellent with minimal missing values")
#         elif quality_score >= 0.7:
#             insights.append("Data quality is good but some cleaning may be beneficial")
#         else:
#             insights.append("Data quality needs improvement - significant missing values detected")
        
#         # Correlation insight
#         high_corrs = len(eda_results['correlations']['high_correlations'])
#         if high_corrs > 0:
#             insights.append(f"Found {high_corrs} highly correlated feature pairs - consider feature selection")
        
#         # Problem type insight
#         problem_type = eda_results['problem_type']
#         target = eda_results['target_variable']
#         insights.append(f"Dataset appears suitable for {problem_type} with '{target}' as target variable")
        
#         # Feature distribution insight
#         numeric_cols = len(eda_results['data_types']['numeric_columns'])
#         cat_cols = len(eda_results['data_types']['categorical_columns'])
#         insights.append(f"Feature mix: {numeric_cols} numeric and {cat_cols} categorical features")
        
#         return ". ".join(insights) + "."
    
#     def _calculate_data_quality_score(self, data: pd.DataFrame) -> float:
#         """Calculate a simple data quality score"""
#         total_cells = data.shape[0] * data.shape[1]
#         missing_cells = data.isnull().sum().sum()
#         completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
#         return round(completeness, 3)
    
#     def _generate_recommendations(self, eda_results: dict) -> list:
#         """Generate recommendations based on EDA"""
#         recommendations = []
        
#         # Algorithm recommendations
#         if eda_results['problem_type'] == "classification":
#             recommendations.append("Consider classification algorithms: Random Forest, XGBoost, or Logistic Regression")
#         else:
#             recommendations.append("Consider regression algorithms: Linear/Ridge Regression, Random Forest, or XGBoost")
        
#         # Correlation recommendations
#         if len(eda_results['correlations']['high_correlations']) > 0:
#             recommendations.append("Remove highly correlated features to avoid multicollinearity")
        
#         # Feature engineering recommendations
#         if len(eda_results['data_types']['categorical_columns']) > 0:
#             recommendations.append("Apply encoding techniques for categorical variables (Label/One-Hot encoding)")
        
#         # Data preprocessing recommendations
#         numeric_cols = len(eda_results['data_types']['numeric_columns'])
#         if numeric_cols > 0:
#             recommendations.append("Consider feature scaling/normalization for numeric variables")
        
#         return recommendations

# def create_mock_cleaned_state():
#     """Create a mock cleaned state for testing when CleanerAgent is not available"""
#     # Create sample data similar to cleaner_agent_test.py
#     np.random.seed(42)
    
#     data = {
#         'age': [25, 30, 28, 35, 45, 28, 32, 29, 31],
#         'salary': [50000, 60000, 55000, 62000, 45000, 58000, 52000, 58000, 51000],
#         'department': ['HR', 'IT', 'Finance', 'Finance', 'IT', 'HR', 'Finance', 'IT', 'HR'],
#         'experience': [5, 7, 3, 8, 4, 6, 9, 2, 10],
#         'performance_score': [8.5, 9.2, 7.8, 8.4, 8.9, 9.5, 8.1, 7.9, 8.7],
#         'city': ['New York', 'Boston', 'Chicago', 'New York', 'Boston', 'Chicago', 'Chicago', 'New York', 'Boston']
#     }
    
#     df = pd.DataFrame(data)
    
#     return {
#         "data": df,
#         "cleaning_summary": {
#             "missing_values_removed": 0,
#             "outliers_handled": True,
#             "data_types_optimized": True
#         },
#         "transformations": ["missing_values_handled", "outliers_handled", "data_types_fixed"]
#     }

# def print_eda_results(eda_results: dict):
#     """Print comprehensive EDA results"""
#     print("\n" + "="*80)
#     print("📊 EXPLORATORY DATA ANALYSIS RESULTS")
#     print("="*80)
    
#     # Summary Statistics
#     print("\n📋 SUMMARY STATISTICS:")
#     print(f"   Dataset shape: {eda_results['summary_stats']['shape']}")
#     print(f"   Data quality score: {eda_results['insights']['data_quality_score']}")
    
#     # Data Types
#     print(f"\n🔍 DATA TYPES:")
#     print(f"   Numeric columns ({len(eda_results['data_types']['numeric_columns'])}): {eda_results['data_types']['numeric_columns']}")
#     print(f"   Categorical columns ({len(eda_results['data_types']['categorical_columns'])}): {eda_results['data_types']['categorical_columns']}")
    
#     # Correlations
#     print(f"\n🔗 CORRELATIONS:")
#     high_corrs = eda_results['correlations']['high_correlations']
#     if high_corrs:
#         print(f"   Found {len(high_corrs)} high correlations:")
#         for corr in high_corrs:
#             print(f"     • {corr['var1']} ↔ {corr['var2']}: {corr['correlation']}")
#     else:
#         print("   No high correlations found (threshold: 0.7)")
    
#     # Target Detection
#     print(f"\n🎯 TARGET ANALYSIS:")
#     print(f"   Target variable: {eda_results['target_variable']}")
#     print(f"   Problem type: {eda_results['problem_type']}")
    
#     # Visualizations
#     print(f"\n📈 VISUALIZATIONS:")
#     if eda_results['visualizations']:
#         for viz in eda_results['visualizations']:
#             print(f"   ✓ Created: {viz}")
#     else:
#         print("   No visualizations created")
    
#     # Insights
#     print(f"\n🧠 KEY INSIGHTS:")
#     print(f"   {eda_results['insights']['ai_insights']}")
    
#     # Recommendations
#     print(f"\n💡 RECOMMENDATIONS:")
#     for i, rec in enumerate(eda_results['insights']['recommendations'], 1):
#         print(f"   {i}. {rec}")

# def main():
#     """Test the EDAAgent"""
#     print("🚀 Testing EDAAgent")
#     print("="*50)
    
#     # Get cleaned data
#     if CLEANER_AVAILABLE:
#         print("📝 Creating sample data and cleaning it...")
#         sample_data = create_sample_data()
#         cleaner = CleanerAgent()
#         cleaned_state = cleaner.clean_data(sample_data)
#         print("✅ Data cleaned successfully!")
#     else:
#         print("📝 Creating mock cleaned state...")
#         cleaned_state = create_mock_cleaned_state()
#         print("✅ Mock cleaned state created!")
    
#     print(f"\n🤖 Initializing EDAAgent...")
#     eda_agent = EDAAgent()
    
#     print(f"\n📊 Starting EDA process...")
#     print("-" * 50)
    
#     # Perform EDA
#     eda_results = eda_agent.perform_eda(cleaned_state)
    
#     print("-" * 50)
    
#     # Display results
#     print_eda_results(eda_results)
    
#     print(f"\n✅ EDA test completed successfully!")
#     print(f"\n📁 Check the current directory for generated visualization files:")
#     for viz in eda_results['visualizations']:
#         print(f"   • {viz}")

# if __name__ == "__main__":
#     main()