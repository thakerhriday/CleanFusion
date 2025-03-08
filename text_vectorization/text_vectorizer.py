import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class TextHandler:
    def __init__(self, data):
        self.data = data
        self.processed_data = pd.DataFrame()

    def detect_text_columns(self):
        text_columns = [col for col in self.data.columns if self.data[col].dtype == 'object']
        if text_columns:
            print(f"\nDetected Text Columns: {text_columns}")
        else:
            print("\nNo text columns detected.")
        return text_columns

    def classify_text_columns(self, text_columns):
        categories = []
        text_data = []

        for column in text_columns:
            unique_values = self.data[column].nunique()
            max_length = self.data[column].str.len().max()

            if unique_values <= 20 or max_length <= 10:  
                categories.append(column)
            else:
                text_data.append(column)

        print(f"\n➡️ Categorical Columns: {categories}")
        print(f"➡️ Text Columns for Vectorization: {text_data}")
        
        return categories, text_data

    def encode_categorical_columns(self, categories):
        for column in categories:
            encoder = LabelEncoder()
            self.data[column] = encoder.fit_transform(self.data[column].astype(str))
        print("\n✅ Categorical data encoded successfully.")
    
    def vectorize_text_columns(self, text_data):
        for column in text_data:
            unique_values = self.data[column].nunique()

            if unique_values <= 100:
                print(f"{column} → Using CountVectorizer")
                vectorizer = CountVectorizer()
            else:
                print(f"{column} → Using TF-IDF Vectorizer")
                vectorizer = TfidfVectorizer()

            vectorized_features = vectorizer.fit_transform(self.data[column].fillna(''))
            feature_names = [f"{column}_{name}" for name in vectorizer.get_feature_names_out()]
            vectorized_df = pd.DataFrame(vectorized_features.toarray(), columns=feature_names)

            self.processed_data = pd.concat([self.processed_data, vectorized_df], axis=1)
        
        print("\n✅ Text vectorization complete.")
        self.processed_data = pd.concat([self.data.drop(text_data, axis=1), self.processed_data], axis=1)

    def save_processed_data(self, output_path="processed_data.csv"):
        self.processed_data.to_csv(output_path, index=False)
        print(f"\n✅ Processed data saved successfully as '{output_path}'")
