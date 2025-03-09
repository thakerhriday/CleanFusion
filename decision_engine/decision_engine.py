import pandas as pd
from data_assessment.data_assessment import DataAssessment
from data_cleaning.missing_value_handler import MissingValueHandler
from text_vectorization.text_vectorizer import TextHandler
from utils.logger import Logger

class DecisionEngine:
    def __init__(self, data):
        self.data = data
        self.assessment = DataAssessment(data)
        self.missing_value_handler = MissingValueHandler()
        self.text_handler = TextHandler()
        self.logger = Logger()

    def run_pipeline(self):
        profile_summary, text_columns, missing_values = self.assessment.assess_data()
        dataset_size = self.assessment.dataset_size()

        cleaned_data = self.data.copy()

        for column, missing_count in missing_values.items():
            if missing_count > 0:
                if column in text_columns:
                    cleaned_data[column] = self.missing_value_handler.handle_text_missing(cleaned_data[column])
                elif cleaned_data[column].dtype in ['int64', 'float64']:
                    method = 'median' if dataset_size == 'large' else 'knn'
                    cleaned_data[column] = self.missing_value_handler.handle_numerical_missing(cleaned_data[column], method)
                else:
                    cleaned_data[column] = self.missing_value_handler.handle_categorical_missing(cleaned_data[column])

        for column in text_columns:
            cleaned_data[column] = self.text_handler.vectorize_text(cleaned_data[column])

        self.logger.info("Data cleaning and text vectorization completed successfully.")
        cleaned_data.to_csv("cleaned_data.csv", index=False)
        self.logger.info("Cleaned data saved as **cleaned_data.csv**")
        return cleaned_data
