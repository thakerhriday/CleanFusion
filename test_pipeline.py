import pandas as pd
from decision_engine.decision_engine import DecisionEngine
from utils.logger import Logger

if __name__ == "__main__":
    logger = Logger()
    
    try:
        sample_data = pd.read_csv('sample_data.csv')
        logger.info("Sample data loaded successfully.")
        
        engine = DecisionEngine(sample_data)
        cleaned_data = engine.run_pipeline()
        
        logger.info("Pipeline executed successfully. Cleaned data saved as **cleaned_data.csv**.")
        print("Pipeline executed successfully. Check 'cleaned_data.csv' for results.")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
