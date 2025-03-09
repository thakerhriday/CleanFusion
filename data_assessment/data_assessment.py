import pandas as pd
import numpy as np

class DataAssessor:
    def __init__(self, data):
        self.data = data

    def assess_dataset_size(self):
        row_count = self.data.shape[0]
        if row_count < 10000:
            return "small"
        elif row_count < 100000:
            return "medium"
        else:
            return "large"

    def identify_data_types(self):
        data_types = {
            "numerical": [],
            "categorical": [],
            "text": []
        }

        for column in self.data.columns:
            unique_ratio = self.data[column].nunique() / len(self.data)

            if pd.api.types.is_numeric_dtype(self.data[column]):
                data_types["numerical"].append(column)
            elif pd.api.types.is_string_dtype(self.data[column]) or unique_ratio > 0.3:
                data_types["text"].append(column)
            else:
                data_types["categorical"].append(column)

        return data_types

    def suggest_preprocessing_strategies(self):
        dataset_size = self.assess_dataset_size()
        data_types = self.identify_data_types()

        strategies = {}

        for column in data_types['numerical']:
            if dataset_size == "small":
                strategies[column] = "KNN Imputation"
            elif dataset_size == "medium":
                strategies[column] = "Mean/Median Imputation"
            else:
                strategies[column] = "Chunk-based Imputation"

        for column in data_types['categorical']:
            strategies[column] = "Mode Imputation or Target Encoding"

        for column in data_types['text']:
            if dataset_size == "small":
                strategies[column] = "BERT Embedding"
            else:
                strategies[column] = "TF-IDF Vectorization"

        return strategies
