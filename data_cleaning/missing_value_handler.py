import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestRegressor

class MissingValueHandler:
    def __init__(self, data):
        self.data = data

    def detect_missing_values(self):
        missing_summary = self.data.isnull().sum()
        missing_columns = missing_summary[missing_summary > 0].index.tolist()
        
        print("\nMissing Value Summary:")
        print(missing_summary[missing_summary > 0])
        return missing_columns

    def handle_missing_values(self):
        missing_columns = self.detect_missing_values()

        for column in missing_columns:
            missing_percentage = (self.data[column].isnull().sum() / len(self.data)) * 100

            if missing_percentage <= 10:
                print(f"{column} → Using Mean/Median Imputation")
                imputer = SimpleImputer(strategy='mean')
                self.data[column] = imputer.fit_transform(self.data[[column]])
            
            elif 10 < missing_percentage <= 30:
                print(f"{column} → Using KNN Imputation")
                imputer = KNNImputer(n_neighbors=3)
                self.data[column] = imputer.fit_transform(self.data[[column]])
            
            else:
                print(f"{column} → Using Predictive Imputation")
                self.data[column] = self._predictive_imputation(column)

        print("\nMissing values handled successfully.")
        return self.data

    def save_cleaned_data(self, output_path="cleaned_data.csv"):
        self.data.to_csv(output_path, index=False)
        print(f"\n✅ Cleaned data saved successfully as '{output_path}'")

    def _predictive_imputation(self, target_column):
        known_data = self.data.dropna(subset=[target_column])
        unknown_data = self.data[self.data[target_column].isnull()]

        features = known_data.drop(target_column, axis=1).select_dtypes(include='number')
        target = known_data[target_column]

        model = RandomForestRegressor()
        model.fit(features, target)

        predicted_values = model.predict(unknown_data.drop(target_column, axis=1).select_dtypes(include='number'))
        self.data.loc[self.data[target_column].isnull(), target_column] = predicted_values
        
        return self.data[target_column]
