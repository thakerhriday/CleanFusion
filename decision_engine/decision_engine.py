import pandas as pd
import numpy as np
from utils.logger import Logger
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.impute import SimpleImputer
from transformers import pipeline

class DecisionEngine:
    def __init__(self, data_path):
        self.data_path = 'sample_data.csv' 
        self.logger = Logger()

    def load_data(self):
        try:
            data = pd.read_csv(self.data_path)
            self.logger.info("Sample data loaded successfully.")
            return data
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise e

    def handle_categorical_data(self, data):
        cat_columns = data.select_dtypes(include='object').columns.tolist()
        if not cat_columns:
            self.logger.info("No categorical columns detected.")
            return data
        
        self.logger.info(f"Detected categorical columns: {cat_columns}")

        for column in cat_columns:
            unique_count = data[column].nunique()

            # Missing Value Handling
            missing_ratio = data[column].isnull().mean()
            if missing_ratio > 0:
                if missing_ratio < 0.1:
                    data[column].fillna(data[column].mode()[0], inplace=True)
                    self.logger.info(f"{column} → Mode Imputation applied for missing values.")
                else:
                    data[column] = self.bert_based_imputation(data[column])
                    self.logger.info(f"{column} → AI-based Imputation (BERT) applied for missing values.")

            # Encoding Strategies
            if unique_count <= 10:
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded_data = encoder.fit_transform(data[[column]])
                encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out())
                data = pd.concat([data.drop(column, axis=1), encoded_df], axis=1)
                self.logger.info(f"{column} → One-Hot Encoding applied.")
            
            elif 11 <= unique_count <= 50:
                encoder = TargetEncoder()
                data[column] = encoder.fit_transform(data[column], data['target'])
                self.logger.info(f"{column} → Target Encoding applied.")
            
            else:
                encoder = OrdinalEncoder()
                data[column] = encoder.fit_transform(data[[column]])
                self.logger.info(f"{column} → Ordinal Encoding applied for high cardinality data.")
        
        return data

    def bert_based_imputation(self, column):
        nlp = pipeline('fill-mask', model='bert-base-uncased')
        return column.apply(lambda x: x if pd.notna(x) else nlp(f"{x} [MASK]")[0]['token_str'])

    def run_pipeline(self):
        try:
            data = self.load_data()
            data = self.handle_categorical_data(data)
            self.logger.info("Categorical data handling completed successfully.")
            return data
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise e
