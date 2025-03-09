import pandas as pd
from decision_engine.decision_engine import DecisionEngine
from utils.logger import setup_logger

logger = setup_logger()

try:
    engine = DecisionEngine(data_path="sample_data.csv")
    cleaned_data = engine.run_pipeline()
    logger.info("Pipeline executed successfully.")
except Exception as e:
    logger.error(f"Pipeline execution failed: {e}")
