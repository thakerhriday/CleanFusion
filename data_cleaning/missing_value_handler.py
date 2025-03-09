import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from utils.logger import Logger

class MissingValueHandler:
    def __init__(self):
        self.logger = Logger()

    def handle_numerical_missing(self, column, method='mean'):
        if method == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif method == 'median':
            imputer = SimpleImputer(strategy='median')
        elif method == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        else:
            raise ValueError(f"Invalid method for numerical imputation: {method}")

        filled_data = imputer.fit_transform(column.values.reshape(-1, 1)).flatten()
        self.logger.info(f"Numerical column '{column.name}' imputed using {method} strategy.")
        return filled_data

    def handle_categorical_missing(self, column):
        most_frequent_value = column.mode()[0]
        filled_data = column.fillna(most_frequent_value)
        self.logger.info(f"Categorical column '{column.name}' imputed using mode ('{most_frequent_value}').")
        return filled_data

    def handle_text_missing(self, column):
        filled_data = column.fillna("UNKNOWN_TEXT")
        self.logger.info(f"Text column '{column.name}' imputed with 'UNKNOWN_TEXT'.")
        return filled_data
