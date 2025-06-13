import os
import dotenv
import logging
import pandas as pd
from agents.model_trainer_agent import ModelTrainerAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("model_trainer_agent_test.log"),
        logging.StreamHandler()
    ]
)


def main():
    # Load environment and cleaned data
    dotenv.load_dotenv()
    data_path = "intermediate_data/cleaned_titanic.csv"
    df = pd.read_csv(data_path)
    logging.info(f"✅ Loaded cleaned data with shape: {df.shape}")

    # Initialize ModelTrainerAgent
    agent = ModelTrainerAgent()
    logging.info("✅ Initialized ModelTrainerAgent")

    # Classification workflow: target 'Survived'
    logging.info("🧪 Testing classification training")
    eda_results_clf = {"problem_type": "classification", "target_variable": "Survived"}
    cleaned_state = {"data": df}
    results_clf = agent.train_model(cleaned_state, eda_results_clf)
    logging.info(f"   • Model type: {results_clf['model_type']}")
    logging.info(f"   • Preprocessing: {results_clf['preprocessing']}")
    logging.info(f"   • Training metrics: {results_clf['training_metrics']}")
    logging.info(f"   • Model saved at: {results_clf['model_path']}")
    logging.info(f"   • AI insights: {results_clf['ai_insights']}")

    # Regression workflow: target 'Fare'
    logging.info("🧪 Testing regression training")
    eda_results_reg = {"problem_type": "regression", "target_variable": "Fare"}
    results_reg = agent.train_model(cleaned_state, eda_results_reg)
    logging.info(f"   • Model type: {results_reg['model_type']}")
    logging.info(f"   • Preprocessing: {results_reg['preprocessing']}")
    logging.info(f"   • Training metrics: {results_reg['training_metrics']}")
    logging.info(f"   • Model saved at: {results_reg['model_path']}")
    logging.info(f"   • AI insights: {results_reg['ai_insights']}")

    logging.info("✅ ModelTrainerAgent test completed successfully")


if __name__ == "__main__":
    main()
