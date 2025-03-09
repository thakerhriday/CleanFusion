from decision_engine.decision_engine import DecisionEngine
from utils.logger import Logger

if __name__ == "__main__":
    logger = Logger()
    file_path = "sample_data.csv"

    engine = DecisionEngine(file_path)
    engine.run_pipeline()

    logger.info("✅ Process Completed Successfully")
