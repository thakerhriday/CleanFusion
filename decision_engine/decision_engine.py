import numpy as np
import pandas as pd
from data_assessment.data_assessment import DataAssessment
from data_cleaning.missing_value_handler import MissingValueHandler
from data_cleaning.outlier_handler import OutlierHandler
from text_vectorization.text_vectorizer import TextHandler

class DecisionEngine:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def run_pipeline(self):
        # Step 1: Data Assessment
        assessor = DataAssessment(self.file_path)
        assessor.load_data()
        assessor.identify_data_types()
        assessor.generate_profile_summary()

        # Step 2: Missing Value Handling
        cleaner = MissingValueHandler(assessor.data)
        cleaned_data = cleaner.handle_missing_values()

        # Step 3: Outlier Detection and Handling
        outlier_handler = OutlierHandler(cleaned_data)
        for column in cleaned_data.select_dtypes(include=[np.number]).columns:
            outliers = outlier_handler.detect_outliers_iqr(column)
            if not outliers.empty:
                print(f"\n⚠️ Outliers detected in {column}. Applying IQR-based handling.")
                outlier_handler.handle_outliers_iqr(column)

        # Step 4: Text and Categorical Data Processing
        text_handler = TextHandler(cleaned_data)
        text_columns = text_handler.detect_text_columns()
        categories, text_data = self.classify_columns(cleaned_data, text_columns)

        text_handler.encode_categorical_columns(categories)
        text_handler.vectorize_text_columns(text_data)

        # Step 5: Save Processed Data
        text_handler.save_processed_data("final_cleaned_data.csv")

        print("\n✅ Data Preprocessing Pipeline Completed Successfully!")

    def classify_columns(self, data, text_columns):
        categories, text_data = [], []

        for column in text_columns:
            unique_values = data[column].nunique()
            max_length = data[column].str.len().max()
            value_entropy = self.calculate_entropy(data[column])

            if unique_values <= 20 or (value_entropy < 1 and max_length <= 10):
                categories.append(column)
            else:
                text_data.append(column)

        print(f"\n➡️ Categorical Columns: {categories}")
        print(f"➡️ Text Columns for Vectorization: {text_data}")
        return categories, text_data

    def calculate_entropy(self, series):
        value_counts = series.value_counts(normalize=True)
        return -(value_counts * np.log2(value_counts)).sum()
