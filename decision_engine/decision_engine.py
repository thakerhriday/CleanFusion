import pandas as pd
import logging
from utils.logger import setup_logger
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

class DecisionEngine:
    def __init__(self, data_path):
        self.data_path = data_path
        self.logger = setup_logger()

    def load_data(self):
        try:
            data = pd.read_csv(self.data_path)
            self.logger.info("Sample data loaded successfully.")
            return data
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise e

    def identify_target_column(self, data):
        assumed_target = data.columns[-1]
        self.logger.info(f"Assuming '{assumed_target}' as target column.")

        if data[assumed_target].nunique() == 2 or data[assumed_target].nunique() > 10:
            self.logger.info(f"Confirmed '{assumed_target}' as target based on data characteristics.")
            return assumed_target

        # AI-driven confirmation if assumption seems ambiguous
        self.logger.warning("Target column assumption unclear. Using AI to confirm...")

        potential_targets = [col for col in data.columns if data[col].nunique() < 20 and data[col].dtype in ['int64', 'float64']]

        if not potential_targets:
            raise ValueError("No clear target column identified. Please specify the target manually.")

        scores = {}
        for col in potential_targets:
            features = data.drop(columns=[col])
            target = data[col]

            # Split and train
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
            model = RandomForestClassifier() if target.nunique() == 2 else RandomForestRegressor()
            model.fit(X_train, y_train)

            # Calculate performance
            prediction = model.predict(X_test)
            score = accuracy_score(y_test, prediction) if target.nunique() == 2 else -mean_squared_error(y_test, prediction)
            scores[col] = score

        best_target = max(scores, key=scores.get)
        self.logger.info(f"AI-identified target column: '{best_target}'")
        return best_target

    def run_pipeline(self):
        try:
            data = self.load_data()
            target_column = self.identify_target_column(data)
            self.logger.info(f"Target column finalized: {target_column}")
            return data
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise e
