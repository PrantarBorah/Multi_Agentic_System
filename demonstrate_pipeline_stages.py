#!/usr/bin/env python3
"""
ML Pipeline Stage Demonstration
Demonstrates how each of the 4 stages works for different ML tasks:
1. Problem Type Detection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Model Training & Evaluation

Author: ML Pipeline Team
Date: 2024
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from app import DataPipelineOrchestrator

class PipelineStageDemonstrator:
    def __init__(self):
        self.sample_data_path = "sample_data"
        self.demonstration_results = {}
        
    def demonstrate_all_stages(self):
        """Demonstrate all 4 stages across different ML tasks"""
        print("🎓 ML Pipeline Stage Demonstration")
        print("=" * 60)
        print("This demonstration shows how each stage works for different ML tasks")
        print()
        
        # Demonstrate with different datasets
        datasets = [
            {
                "name": "Binary Classification",
                "file": "1_survey_lung_cancer.csv",
                "description": "Medical diagnosis - predicting lung cancer from symptoms"
            },
            {
                "name": "Multi-class Classification", 
                "file": "2_Iris.csv",
                "description": "Botanical classification - identifying iris flower species"
            },
            {
                "name": "Regression",
                "file": "4_house_prices.csv", 
                "description": "Real estate - predicting house prices from features"
            },
            {
                "name": "Clustering",
                "file": "7_customer_segments.csv",
                "description": "Customer segmentation - grouping customers by behavior"
            }
        ]
        
        for dataset in datasets:
            self.demonstrate_dataset_stages(dataset)
            
        self.generate_demonstration_summary()
    
    def demonstrate_dataset_stages(self, dataset_info):
        """Demonstrate all 4 stages for a specific dataset"""
        print(f"\n🔍 Demonstrating: {dataset_info['name']}")
        print(f"📊 Dataset: {dataset_info['file']}")
        print(f"📝 Description: {dataset_info['description']}")
        print("-" * 60)
        
        try:
            # Load data
            data_path = f"{self.sample_data_path}/{dataset_info['file']}"
            data = pd.read_csv(data_path)
            
            print(f"📈 Original Data Shape: {data.shape}")
            print(f"📋 Columns: {list(data.columns)}")
            print()
            
            # Stage 1: Problem Type Detection
            self.demonstrate_stage_1(data, dataset_info)
            
            # Stage 2: Data Cleaning
            self.demonstrate_stage_2(data, dataset_info)
            
            # Stage 3: EDA
            self.demonstrate_stage_3(data, dataset_info)
            
            # Stage 4: Model Training & Evaluation
            self.demonstrate_stage_4(data, dataset_info)
            
            print(f"✅ {dataset_info['name']} demonstration completed successfully!")
            
        except Exception as e:
            print(f"❌ Error in {dataset_info['name']} demonstration: {str(e)}")
    
    def demonstrate_stage_1(self, data, dataset_info):
        """Demonstrate Stage 1: Problem Type Detection"""
        print("🎯 Stage 1: Problem Type Detection")
        print("   Purpose: Automatically identify the ML task type and target variable")
        print("   Agent: ProblemTypeAgent")
        print()
        
        # Simulate problem type detection
        if "lung_cancer" in dataset_info['file']:
            problem_type = "Binary Classification"
            target_variable = "LUNG_CANCER"
            reasoning = "Binary target variable with YES/NO values"
        elif "Iris" in dataset_info['file']:
            problem_type = "Multi-class Classification"
            target_variable = "Species"
            reasoning = "Categorical target with 3 distinct classes"
        elif "house_prices" in dataset_info['file']:
            problem_type = "Regression"
            target_variable = "price"
            reasoning = "Continuous numerical target variable"
        elif "customer_segments" in dataset_info['file']:
            problem_type = "Clustering"
            target_variable = None
            reasoning = "No target variable - unsupervised learning"
        
        print(f"   🔍 Detected Problem Type: {problem_type}")
        print(f"   🎯 Target Variable: {target_variable or 'None (Unsupervised)'}")
        print(f"   💡 Reasoning: {reasoning}")
        print()
        
        # Show data characteristics
        print("   📊 Data Characteristics:")
        print(f"      - Sample size: {len(data)}")
        print(f"      - Features: {len(data.columns)}")
        print(f"      - Data types: {dict(data.dtypes)}")
        print()
    
    def demonstrate_stage_2(self, data, dataset_info):
        """Demonstrate Stage 2: Data Cleaning"""
        print("🧹 Stage 2: Data Cleaning")
        print("   Purpose: Handle missing values, outliers, and data type conversions")
        print("   Agent: CleanerAgent")
        print()
        
        # Analyze data quality issues
        missing_values = data.isnull().sum()
        total_missing = missing_values.sum()
        
        print("   🔍 Data Quality Analysis:")
        print(f"      - Missing values: {total_missing}")
        if total_missing > 0:
            missing_cols = missing_values[missing_values > 0]
            print(f"      - Columns with missing values: {dict(missing_cols)}")
        
        # Check for outliers in numerical columns
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        outlier_info = []
        for col in numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data[col] < Q1 - 1.5*IQR) | (data[col] > Q3 + 1.5*IQR)]
            if len(outliers) > 0:
                outlier_info.append(f"{col}: {len(outliers)} outliers")
        
        if outlier_info:
            print(f"      - Outliers detected: {', '.join(outlier_info)}")
        else:
            print("      - No significant outliers detected")
        
        print()
        print("   🛠️  Cleaning Actions:")
        print("      - Handle missing values (imputation or removal)")
        print("      - Remove or cap outliers")
        print("      - Convert data types appropriately")
        print("      - Encode categorical variables")
        print()
    
    def demonstrate_stage_3(self, data, dataset_info):
        """Demonstrate Stage 3: Exploratory Data Analysis (EDA)"""
        print("📊 Stage 3: Exploratory Data Analysis (EDA)")
        print("   Purpose: Understand data patterns, relationships, and distributions")
        print("   Agent: EDAAgent")
        print()
        
        print("   📈 EDA Components:")
        print("      - Summary statistics")
        print("      - Correlation analysis")
        print("      - Feature distributions")
        print("      - Target variable analysis")
        print("      - AI-powered insights")
        print()
        
        # Show sample statistics
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print("   📋 Sample Statistics:")
            sample_stats = data[numerical_cols].describe()
            print(f"      - Numerical features: {list(numerical_cols)}")
            print(f"      - Mean values range: {sample_stats.loc['mean'].min():.2f} to {sample_stats.loc['mean'].max():.2f}")
            print(f"      - Standard deviation range: {sample_stats.loc['std'].min():.2f} to {sample_stats.loc['std'].max():.2f}")
        
        # Show categorical analysis
        categorical_cols = data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"      - Categorical features: {list(categorical_cols)}")
            for col in categorical_cols:
                unique_count = data[col].nunique()
                print(f"      - {col}: {unique_count} unique values")
        print()
    
    def demonstrate_stage_4(self, data, dataset_info):
        """Demonstrate Stage 4: Model Training & Evaluation"""
        print("🤖 Stage 4: Model Training & Evaluation")
        print("   Purpose: Train models and evaluate performance")
        print("   Agents: ModelTrainerAgent, EvaluatorAgent")
        print()
        
        if "customer_segments" in dataset_info['file']:
            print("   🎯 Clustering Task:")
            print("      - Algorithm: K-Means Clustering")
            print("      - Purpose: Group customers into segments")
            print("      - Evaluation: Silhouette score, inertia")
            print("      - Visualization: Cluster plots")
        else:
            print("   🎯 Supervised Learning Task:")
            print("      - Algorithms: Multiple (Random Forest, SVM, etc.)")
            print("      - Cross-validation: Stratified K-Fold")
            print("      - Hyperparameter tuning: Grid search")
            print("      - Evaluation metrics: Accuracy, Precision, Recall, F1")
        
        print()
        print("   📊 Expected Outcomes:")
        print("      - Best model selection")
        print("      - Performance metrics")
        print("      - Feature importance analysis")
        print("      - Model comparison")
        print("      - Educational insights")
        print()
    
    def generate_demonstration_summary(self):
        """Generate summary of all demonstrations"""
        print("\n📋 Pipeline Stage Summary")
        print("=" * 60)
        
        stages = [
            {
                "name": "Stage 1: Problem Type Detection",
                "agent": "ProblemTypeAgent",
                "purpose": "Identify ML task type and target variable",
                "output": "Problem analysis and recommendations"
            },
            {
                "name": "Stage 2: Data Cleaning", 
                "agent": "CleanerAgent",
                "purpose": "Handle missing values, outliers, data types",
                "output": "Cleaned dataset and cleaning report"
            },
            {
                "name": "Stage 3: Exploratory Data Analysis",
                "agent": "EDAAgent", 
                "purpose": "Understand data patterns and relationships",
                "output": "Visualizations, statistics, AI insights"
            },
            {
                "name": "Stage 4: Model Training & Evaluation",
                "agent": "ModelTrainerAgent + EvaluatorAgent",
                "purpose": "Train models and evaluate performance",
                "output": "Best model, metrics, comparisons"
            }
        ]
        
        for stage in stages:
            print(f"🎯 {stage['name']}")
            print(f"   🤖 Agent: {stage['agent']}")
            print(f"   🎯 Purpose: {stage['purpose']}")
            print(f"   📤 Output: {stage['output']}")
            print()
        
        print("✅ Pipeline is designed to handle:")
        print("   - Binary Classification")
        print("   - Multi-class Classification") 
        print("   - Regression")
        print("   - Clustering (Unsupervised)")
        print()
        print("🎓 Educational Features:")
        print("   - Step-by-step explanations")
        print("   - Decision rationale")
        print("   - Interactive configurations")
        print("   - Performance comparisons")
        print("   - Best practices guidance")

def main():
    """Main function to run pipeline stage demonstrations"""
    demonstrator = PipelineStageDemonstrator()
    demonstrator.demonstrate_all_stages()

if __name__ == "__main__":
    main() 