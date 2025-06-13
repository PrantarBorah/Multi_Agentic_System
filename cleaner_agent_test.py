import os
import dotenv
import pandas as pd
import logging
from agents.cleaner_agent import CleanerAgent

# 1. Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("cleaner_agent_test.log"),
        logging.StreamHandler()
    ]
)

def main():
    logging.info("🚀 Starting CleanerAgent test")

    # 2. Load environment variables
    dotenv.load_dotenv()  # Load .env file into environment variables [[1]]  
    logging.info("🔑 Environment variables loaded")

    # 3. Load raw dataset
    data_path = "sample_data/3_titanic.csv"
    df = pd.read_csv(data_path)
    logging.info(f"✅ Loaded raw data with shape: {df.shape}")

    # 4. Initialize CleanerAgent
    agent = CleanerAgent()
    logging.info("✅ Initialized CleanerAgent")

    # 5. Run cleaning process
    logging.info("📋 Running clean_data()")
    result = agent.clean_data(df)
    cleaned_df = result.get("data")
    logging.info(f"✅ Cleaning completed. Cleaned data shape: {cleaned_df.shape}")

    # 6. Log AI-generated summary
    summary = result.get(
        "llm_summary",
        result.get("cleaning_summary", {}).get("summary", "No LLM summary generated.")
    )
    logging.info("📄 Cleaning summary:\n" + summary)

    # 7. Log transformations applied
    logging.info(f"🔧 Transformations: {result['transformations']}")

    # 8. Save cleaned data for next agent
    output_path = "intermediate_data/cleaned_titanic.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cleaned_df.to_csv(output_path, index=False)
    logging.info(f"📂 Cleaned data saved to: {output_path}")

if __name__ == "__main__":
    main()












# import os
# import dotenv
# import pandas as pd
# from agents.cleaner_agent import CleanerAgent

# # Load environment variables from .env
# dotenv.load_dotenv()

# # Load dataset
# data_path = "sample_data/3_titanic.csv"
# df = pd.read_csv(data_path)

# # Initialize cleaner agent
# agent = CleanerAgent()

# # Clean data
# result = agent.clean_data(df)
# cleaned_df = result.get("data")
# summary_prompt = result.get("llm_summary", result.get("cleaning_summary", {}).get("summary", "No LLM summary generated."))
# print("\n📄 LLM Summary:\n", summary_prompt)
# # Output results
# print("✅ Cleaning Completed.")

# print(f"🔧 Transformations: {result['transformations']}")
# print("\n📄 LLM Summary:\n", summary_prompt)

# # Save cleaned data for next agent
# output_path = "intermediate_data/cleaned_titanic.csv"
# os.makedirs(os.path.dirname(output_path), exist_ok=True)
# cleaned_df.to_csv(output_path, index=False)
# print(f"📂 Cleaned data saved to: {output_path}")
