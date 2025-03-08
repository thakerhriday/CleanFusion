import pandas as pd

class DataAssessment:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.profile = {}

    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Data loaded successfully with {self.data.shape[0]} rows and {self.data.shape[1]} columns.")
        except Exception as e:
            print(f"Error loading file: {e}")
            return None

    def identify_data_types(self):
        if self.data is None:
            print("No data loaded. Please load the dataset first.")
            return
        
        for column in self.data.columns:
            dtype = self.data[column].dtype
            if pd.api.types.is_numeric_dtype(self.data[column]):
                self.profile[column] = 'Numerical'
            elif pd.api.types.is_string_dtype(self.data[column]):
                self.profile[column] = 'Text'
            elif pd.api.types.is_categorical_dtype(self.data[column]):
                self.profile[column] = 'Categorical'
            else:
                self.profile[column] = 'Unknown'
        
        print("Data types identified successfully:")
        for column, dtype in self.profile.items():
            print(f"{column}: {dtype}")

    def generate_profile_summary(self):
        if self.data is None:
            print("No data loaded. Please load the dataset first.")
            return
        
        summary = self.data.describe(include='all').transpose()
        missing_values = self.data.isnull().sum().to_frame('Missing Values')
        profile_summary = summary.merge(missing_values, left_index=True, right_index=True)
        
        print("\nData Profile Summary:")
        print(profile_summary[['count', 'unique', 'top', 'freq', 'mean', 'std', 'min', 'max', 'Missing Values']])
    
# Example Usage (For Testing)
if __name__ == "__main__":
    file_path = "../sample_data.csv"  # Adjust the path as needed
    assessor = DataAssessment(file_path)
    assessor.load_data()
    assessor.identify_data_types()
    assessor.generate_profile_summary()
