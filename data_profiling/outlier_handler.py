import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

class OutlierHandler:
    def __init__(self, data):
        self.data = data

    def detect_outliers_iqr(self, column):
        q1 = self.data[column].quantile(0.25)
        q3 = self.data[column].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = self.data[(self.data[column] < lower_bound) | (self.data[column] > upper_bound)]
        return outliers

    def handle_outliers_iqr(self, column):
        q1 = self.data[column].quantile(0.25)
        q3 = self.data[column].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Cap values at the lower and upper bounds
        self.data[column] = np.where(self.data[column] < lower_bound, lower_bound, self.data[column])
        self.data[column] = np.where(self.data[column] > upper_bound, upper_bound, self.data[column])

        print(f"\n✅ Outliers handled using IQR in column: {column}")

    def detect_outliers_zscore(self, column, threshold=3):
        mean = self.data[column].mean()
        std_dev = self.data[column].std()

        z_scores = (self.data[column] - mean) / std_dev
        outliers = self.data[np.abs(z_scores) > threshold]

        return outliers

    def handle_outliers_isolation_forest(self):
        model = IsolationForest(contamination=0.05, random_state=42)
        outlier_flags = model.fit_predict(self.data.select_dtypes(include=[np.number]))

        self.data['Outlier_Flag'] = outlier_flags
        self.data = self.data[self.data['Outlier_Flag'] == 1].drop('Outlier_Flag', axis=1)

        print("\n✅ Outliers handled using Isolation Forest")

    def save_cleaned_data(self, output_path="outlier_cleaned_data.csv"):
        self.data.to_csv(output_path, index=False)
        print(f"\n✅ Data with outliers handled saved as '{output_path}'")
