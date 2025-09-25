#!/usr/bin/env python3
"""
Comprehensive ML Pipeline Robustness Test
Tests the 4-stage pipeline across different ML tasks:
1. Binary Classification
2. Multi-class Classification  
3. Regression
4. Clustering (Unsupervised)

Author: ML Pipeline Team
Date: 2024
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from app import DataPipelineOrchestrator

class PipelineRobustnessTester:
    def __init__(self):
        self.test_results = {}
        self.sample_data_path = "sample_data"
        
    def run_comprehensive_tests(self):
        """Run tests across all ML task types"""
        print("🧪 Starting Comprehensive ML Pipeline Robustness Tests")
        print("=" * 60)
        
        # Test 1: Binary Classification
        self.test_binary_classification()
        
        # Test 2: Multi-class Classification
        self.test_multiclass_classification()
        
        # Test 3: Regression
        self.test_regression()
        
        # Test 4: Clustering (Unsupervised)
        self.test_clustering()
        
        # Generate comprehensive report
        self.generate_test_report()
        
    def test_binary_classification(self):
        """Test binary classification with lung cancer dataset"""
        print("\n🔍 Test 1: Binary Classification (Lung Cancer)")
        print("-" * 50)
        
        try:
            # Test with lung cancer dataset
            data_path = f"{self.sample_data_path}/1_survey_lung_cancer.csv"
            orchestrator = DataPipelineOrchestrator(data_path)
            
            # Run pipeline
            results = orchestrator.run_pipeline()
            
            # Validate results
            self.validate_classification_results(results, "binary")
            
            self.test_results["binary_classification"] = {
                "status": "PASSED",
                "dataset": "lung_cancer_survey",
                "target_variable": "LUNG_CANCER",
                "problem_type": "Binary Classification",
                "sample_size": len(pd.read_csv(data_path)),
                "timestamp": datetime.now().isoformat()
            }
            
            print("✅ Binary Classification Test PASSED")
            
        except Exception as e:
            print(f"❌ Binary Classification Test FAILED: {str(e)}")
            self.test_results["binary_classification"] = {
                "status": "FAILED",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def test_multiclass_classification(self):
        """Test multi-class classification with Iris dataset"""
        print("\n🔍 Test 2: Multi-class Classification (Iris)")
        print("-" * 50)
        
        try:
            # Test with Iris dataset
            data_path = f"{self.sample_data_path}/2_Iris.csv"
            orchestrator = DataPipelineOrchestrator(data_path)
            
            # Run pipeline
            results = orchestrator.run_pipeline()
            
            # Validate results
            self.validate_classification_results(results, "multiclass")
            
            self.test_results["multiclass_classification"] = {
                "status": "PASSED",
                "dataset": "iris_flowers",
                "target_variable": "Species",
                "problem_type": "Multi-class Classification",
                "sample_size": len(pd.read_csv(data_path)),
                "timestamp": datetime.now().isoformat()
            }
            
            print("✅ Multi-class Classification Test PASSED")
            
        except Exception as e:
            print(f"❌ Multi-class Classification Test FAILED: {str(e)}")
            self.test_results["multiclass_classification"] = {
                "status": "FAILED",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def test_regression(self):
        """Test regression with house prices dataset"""
        print("\n🔍 Test 3: Regression (House Prices)")
        print("-" * 50)
        
        try:
            # Test with house prices dataset
            data_path = f"{self.sample_data_path}/4_house_prices.csv"
            orchestrator = DataPipelineOrchestrator(data_path)
            
            # Run pipeline
            results = orchestrator.run_pipeline()
            
            # Validate results
            self.validate_regression_results(results)
            
            self.test_results["regression"] = {
                "status": "PASSED",
                "dataset": "house_prices",
                "target_variable": "price",
                "problem_type": "Regression",
                "sample_size": len(pd.read_csv(data_path)),
                "timestamp": datetime.now().isoformat()
            }
            
            print("✅ Regression Test PASSED")
            
        except Exception as e:
            print(f"❌ Regression Test FAILED: {str(e)}")
            self.test_results["regression"] = {
                "status": "FAILED",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def test_clustering(self):
        """Test clustering with customer segments dataset"""
        print("\n🔍 Test 4: Clustering (Customer Segments)")
        print("-" * 50)
        
        try:
            # Test with customer segments dataset
            data_path = f"{self.sample_data_path}/7_customer_segments.csv"
            orchestrator = DataPipelineOrchestrator(data_path)
            
            # Run pipeline
            results = orchestrator.run_pipeline()
            
            # Validate results
            self.validate_clustering_results(results)
            
            self.test_results["clustering"] = {
                "status": "PASSED",
                "dataset": "customer_segments",
                "problem_type": "Clustering (Unsupervised)",
                "sample_size": len(pd.read_csv(data_path)),
                "timestamp": datetime.now().isoformat()
            }
            
            print("✅ Clustering Test PASSED")
            
        except Exception as e:
            print(f"❌ Clustering Test FAILED: {str(e)}")
            self.test_results["clustering"] = {
                "status": "FAILED",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def validate_classification_results(self, results, classification_type):
        """Validate classification pipeline results"""
        required_keys = [
            "problem_analysis", "cleaned_data", "eda_results", 
            "model_results", "evaluation_results"
        ]
        
        for key in required_keys:
            if key not in results:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate problem analysis
        problem_analysis = results["problem_analysis"]
        if classification_type == "binary":
            expected_type = "Binary Classification"
        else:
            expected_type = "Multi-class Classification"
            
        if problem_analysis.get("problem_type") != expected_type:
            raise ValueError(f"Expected {expected_type}, got {problem_analysis.get('problem_type')}")
        
        # Validate model results
        model_results = results["model_results"]
        if "cv_score" not in model_results:
            raise ValueError("Missing cross-validation score")
        
        cv_score = model_results["cv_score"]
        if not (0 <= cv_score <= 1):
            raise ValueError(f"Invalid CV score: {cv_score}")
    
    def validate_regression_results(self, results):
        """Validate regression pipeline results"""
        required_keys = [
            "problem_analysis", "cleaned_data", "eda_results", 
            "model_results", "evaluation_results"
        ]
        
        for key in required_keys:
            if key not in results:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate problem analysis
        problem_analysis = results["problem_analysis"]
        if problem_analysis.get("problem_type") != "Regression":
            raise ValueError(f"Expected Regression, got {problem_analysis.get('problem_type')}")
        
        # Validate model results
        model_results = results["model_results"]
        if "cv_score" not in model_results:
            raise ValueError("Missing cross-validation score")
    
    def validate_clustering_results(self, results):
        """Validate clustering pipeline results"""
        required_keys = [
            "problem_analysis", "cleaned_data", "eda_results"
        ]
        
        for key in required_keys:
            if key not in results:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate problem analysis
        problem_analysis = results["problem_analysis"]
        if "Clustering" not in problem_analysis.get("problem_type", ""):
            raise ValueError(f"Expected Clustering, got {problem_analysis.get('problem_type')}")
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n📊 Test Report Summary")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["status"] == "PASSED")
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result["status"] == "PASSED" else "❌"
            print(f"{status_icon} {test_name.replace('_', ' ').title()}: {result['status']}")
            if result["status"] == "PASSED":
                print(f"   Dataset: {result.get('dataset', 'N/A')}")
                print(f"   Target: {result.get('target_variable', 'N/A')}")
                print(f"   Problem Type: {result.get('problem_type', 'N/A')}")
                print(f"   Sample Size: {result.get('sample_size', 'N/A')}")
            else:
                print(f"   Error: {result.get('error', 'Unknown error')}")
            print()
        
        # Save detailed report
        report_path = "pipeline_robustness_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        print(f"📄 Detailed report saved to: {report_path}")
        
        # Overall assessment
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! Pipeline is robust across all ML task types.")
        elif passed_tests >= total_tests * 0.75:
            print("⚠️  Most tests passed. Pipeline needs minor improvements.")
        else:
            print("🚨 Multiple tests failed. Pipeline needs significant improvements.")

def main():
    """Main function to run comprehensive pipeline tests"""
    tester = PipelineRobustnessTester()
    tester.run_comprehensive_tests()

if __name__ == "__main__":
    main() 