#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""

def test_imports():
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        import pandas as pd
        print("✅ Pandas imported successfully")
        
        import numpy as np
        print("✅ NumPy imported successfully")
        
        import sklearn
        print("✅ Scikit-learn imported successfully")
        
        import plotly
        print("✅ Plotly imported successfully")
        
        import openai
        print("✅ OpenAI imported successfully")
        
        from dotenv import load_dotenv
        print("✅ Python-dotenv imported successfully")
        
        from crewai import Agent, Task, Crew, Process
        print("✅ CrewAI imported successfully")
        
        import matplotlib
        print("✅ Matplotlib imported successfully")
        
        import seaborn
        print("✅ Seaborn imported successfully")
        
        import scipy
        print("✅ SciPy imported successfully")
        
        import joblib
        print("✅ Joblib imported successfully")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    test_imports()
