import os
import json
import pandas as pd
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from agents.cleaner_agent import CleanerAgent
from agents.eda_agent import EDAAgent
from agents.model_trainer_agent import ModelTrainerAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.problem_type_agent import ProblemTypeAgent
from utils.data_utils import load_data, save_state

load_dotenv()

class DataPipelineOrchestrator:
    def __init__(self, data_path: str, openai_api_key: str = None, cleaning_options: dict = None, eda_options: dict = None, model_training_options: dict = None, evaluation_options: dict = None):
        self.data_path = data_path
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Please set it in Streamlit secrets or environment variables.")
        
        self.pipeline_state = {
            "original_data": None,
            "problem_analysis": None,
            "cleaned_data": None,
            "eda_results": None,
            "model_results": None,
            "evaluation_results": None
        }
        
        # Initialize agents with API key
        self.problem_type_agent = ProblemTypeAgent()
        self.cleaner_agent = CleanerAgent(openai_api_key=self.openai_api_key)
        self.eda_agent = EDAAgent(openai_api_key=self.openai_api_key)
        self.model_trainer_agent = ModelTrainerAgent(openai_api_key=self.openai_api_key)
        self.evaluator_agent = EvaluatorAgent(openai_api_key=self.openai_api_key)
        
        self.cleaning_options = cleaning_options if cleaning_options is not None else {}
        self.eda_options = eda_options if eda_options is not None else {}
        self.model_training_options = model_training_options if model_training_options is not None else {}
        self.evaluation_options = evaluation_options if evaluation_options is not None else {}
    
    def run_pipeline(self, cleaning_config=None, eda_config=None, training_config=None, evaluation_config=None, progress_callback=None):
        """Execute the complete data pipeline using CrewAI orchestration"""
        try:
            # Load initial data
            print("🚀 Starting AI Agent Data Pipeline...")
            if progress_callback:
                progress_callback("starting", 0, "Loading data...")
            
            self.pipeline_state["original_data"] = load_data(self.data_path)
            
            # Update options with provided configurations
            if cleaning_config:
                self.cleaning_options.update(cleaning_config)
            if eda_config:
                self.eda_options.update(eda_config)
            if training_config:
                self.model_training_options.update(training_config)
            if evaluation_config:
                self.evaluation_options.update(evaluation_config)
            
            # Stage 1: Problem Type Detection (NEW)
            print("🔍 Stage 1: Problem Type Detection...")
            if progress_callback:
                progress_callback("problem_detection", 20, "Detecting problem type and target variable...")
            self._execute_problem_type_detection()
            
            # Stage 2: Data Cleaning (conditional based on problem type)
            print("🧹 Stage 2: Data Cleaning...")
            if progress_callback:
                progress_callback("cleaning", 40, "Cleaning and preprocessing data...")
            self._execute_cleaning()
            
            # Stage 3: Exploratory Data Analysis
            print("📊 Stage 3: Exploratory Data Analysis...")
            if progress_callback:
                progress_callback("eda", 60, "Analyzing data patterns and relationships...")
            self._execute_eda()
            
            # Stage 4: Model Training (conditional based on problem type)
            print("🤖 Stage 4: Model Training...")
            if progress_callback:
                progress_callback("training", 80, "Training machine learning models...")
            self._execute_training()
            
            # Stage 5: Model Evaluation
            print("📈 Stage 5: Model Evaluation...")
            if progress_callback:
                progress_callback("evaluation", 90, "Evaluating model performance...")
            self._execute_evaluation()
            
            # Save final results
            if progress_callback:
                progress_callback("completed", 100, "Pipeline completed successfully!")
            self._save_final_results()
            
            print("✅ Pipeline completed successfully!")
            
        except Exception as e:
            print(f"❌ Pipeline Error: {str(e)}")
            if progress_callback:
                progress_callback("failed", 0, f"Pipeline failed: {str(e)}")
            raise
    
    def _execute_problem_type_detection(self):
        """Execute problem type detection as the first stage"""
        try:
            # Analyze problem type using the new agent
            problem_analysis = self.problem_type_agent.analyze_problem_type(
                self.pipeline_state["original_data"]
            )
            
            self.pipeline_state["problem_analysis"] = problem_analysis
            
            # Log the results
            print(f"🎯 Problem Type: {problem_analysis['problem_type']}")
            if problem_analysis.get('target_variable'):
                print(f"🎯 Target Variable: {problem_analysis['target_variable']}")
                target_chars = problem_analysis['target_characteristics']
                print(f"📊 Target Characteristics:")
                print(f"   - Data Type: {target_chars['data_type']}")
                print(f"   - Unique Values: {target_chars['unique_values']}")
                print(f"   - Missing Values: {target_chars['missing_count']} ({target_chars['missing_percentage']:.1%})")
                
                if target_chars.get('imbalance_warning'):
                    print(f"⚠️  Class Imbalance Detected (Ratio: {target_chars['class_balance_ratio']:.2f})")
            else:
                print("🔍 Unsupervised Learning Problem")
            
            # Update cleaning options based on problem type
            self._update_cleaning_options(problem_analysis)
            
        except Exception as e:
            print(f"❌ Problem Type Detection Error: {str(e)}")
            raise
    
    def _update_cleaning_options(self, problem_analysis: dict):
        """Update cleaning options based on problem type analysis"""
        recommendations = problem_analysis.get('cleaning_recommendations', {})
        
        # Update target handling
        target_handling = recommendations.get('target_handling', {})
        if 'missing_targets' in target_handling:
            self.cleaning_options['target_missing_strategy'] = target_handling['missing_targets']
        
        if 'class_imbalance' in target_handling:
            self.cleaning_options['handle_class_imbalance'] = True
        
        # Update validation strategy
        validation_strategy = recommendations.get('validation_strategy', {})
        if 'cv_method' in validation_strategy:
            self.model_training_options['cv_strategy'] = validation_strategy['cv_method']
        
        # Update algorithm recommendations
        algorithm_recommendations = recommendations.get('algorithm_recommendations', [])
        if algorithm_recommendations:
            self.model_training_options['recommended_algorithms'] = algorithm_recommendations
        
        print("⚙️  Updated pipeline configuration based on problem type analysis")
    
    def _execute_cleaning(self):
        """Execute data cleaning with problem type awareness"""
        try:
            # Get problem analysis
            problem_analysis = self.pipeline_state["problem_analysis"]
            target_variable = problem_analysis.get('target_variable')
            
            # Update cleaning options for target-aware cleaning
            if target_variable:
                # Don't impute target column - drop missing targets instead
                self.cleaning_options['exclude_from_imputation'] = [target_variable]
                self.cleaning_options['drop_missing_targets'] = True
                
                # Handle class imbalance if detected
                target_chars = problem_analysis['target_characteristics']
                if target_chars.get('imbalance_warning'):
                    self.cleaning_options['handle_class_imbalance'] = True
            
            # Translate UI parameter names to cleaner agent parameter names
            if 'imputation_method' in self.cleaning_options:
                self.cleaning_options['missing_value_strategy'] = self.cleaning_options['imputation_method']
            if 'outlier_method' in self.cleaning_options:
                self.cleaning_options['outlier_strategy'] = self.cleaning_options['outlier_method']
            
            print(f"🧹 Cleaning with options: {self.cleaning_options}")
            
            # Execute cleaning with updated options
            cleaned_result = self.cleaner_agent.clean_data(
                self.pipeline_state["original_data"], 
                self.cleaning_options
            )
            
            # Handle tuple return (data, cleaned_state)
            if isinstance(cleaned_result, tuple):
                self.pipeline_state["cleaned_data"] = cleaned_result[0]
                self.pipeline_state["cleaning_summary_results"] = cleaned_result[1]
            else:
                self.pipeline_state["cleaned_data"] = cleaned_result
            
            print(f"✅ Data cleaned successfully")
            print(f"📊 Original shape: {self.pipeline_state['original_data'].shape}")
            print(f"📊 Cleaned shape: {self.pipeline_state['cleaned_data'].shape}")
            
        except Exception as e:
            print(f"❌ Cleaning Error: {str(e)}")
            raise
    
    def _execute_eda(self):
        """Execute EDA step"""
        print("\n📊 Step 2: Exploratory Data Analysis...")
        # Pass through detected target/problem type from problem detection to keep consistency
        problem_analysis = self.pipeline_state.get("problem_analysis", {})
        eda_opts = dict(self.eda_options) if isinstance(self.eda_options, dict) else {}
        if problem_analysis.get("target_variable"):
            eda_opts["force_target_variable"] = problem_analysis["target_variable"]
        if problem_analysis.get("problem_type"):
            eda_opts["force_problem_type"] = problem_analysis["problem_type"]

        self.pipeline_state["eda_results"] = self.eda_agent.perform_eda(
            self.pipeline_state["cleaned_data"],
            eda_opts
        )
        
    def _execute_training(self):
        """Execute model training step"""
        print("\n🤖 Step 3: Model Training...")
        # Pass problem type and target variable from problem analysis to ensure consistency
        problem_analysis = self.pipeline_state.get("problem_analysis", {})
        
        print(f"🤖 Training with options: {self.model_training_options}")
        
        self.pipeline_state["model_results"] = self.model_trainer_agent.train_model(
            self.pipeline_state["cleaned_data"],
            self.pipeline_state["eda_results"],
            model_training_options=self.model_training_options
        )
        
    def _execute_evaluation(self):
        """Execute model evaluation step"""
        print("\n📈 Step 4: Model Evaluation...")
        
        print(f"📈 Evaluation with options: {self.evaluation_options}")
        
        self.pipeline_state["evaluation_results"] = self.evaluator_agent.evaluate_model(
            self.pipeline_state["model_results"],
            self.pipeline_state["cleaned_data"],
            self.evaluation_options
        )
    
    def _save_final_results(self):
        """Save pipeline results"""
        save_state(self.pipeline_state, "pipeline_results.json")
        print(f"💾 Results saved to pipeline_results.json")

if __name__ == "__main__":
    # Example usage
    data_path = "sample_data/5_HousingData.csv"
    orchestrator = DataPipelineOrchestrator(data_path)
    orchestrator.run_pipeline()