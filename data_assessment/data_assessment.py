import pandas as pd
import numpy as np
from scipy.stats import entropy
from utils.logger import Logger

class DataAssessment:
    def __init__(self, data):
        self.data = data
        self.logger = Logger()

    def assess_data(self):
        profile_summary = self.data.describe(include='all').transpose()
        profile_summary['Missing Values'] = self.data.isnull().sum()
        text_columns = []
        missing_values = self.data.isnull().sum()

        for column in self.data.columns:
            if self.data[column].dtype == 'object':
                if self._calculate_entropy(self.data[column]) > 1.5:
                    text_columns.append(column)
                    self.logger.info(f"'{column}' identified as text data based on entropy analysis.")
                else:
                    self.logger.info(f"'{column}' identified as categorical data (low entropy).")

        self.logger.info("Data profiling completed successfully.")
        return profile_summary, text_columns, missing_values

    def _calculate_entropy(self, column):
        value_counts = column.value_counts(normalize=True)
        return entropy(value_counts)

    def dataset_size(self):
        row_count, col_count = self.data.shape
        if row_count > 10000:
            self.logger.info("Dataset identified as **Large Dataset**.")
            return 'large'
        else:
            self.logger.info("Dataset identified as **Small/Medium Dataset**.")
            return 'small'
