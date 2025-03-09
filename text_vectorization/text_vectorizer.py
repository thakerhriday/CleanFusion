import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from utils.logger import Logger

class TextHandler:
    def __init__(self):
        self.logger = Logger()
        self.bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def vectorize_text(self, column):
        if column.nunique() > 50:  # Assume >50 unique values = likely free-text data
            vectorized_data = self.bert_model.encode(column.fillna("UNKNOWN_TEXT").tolist())
            self.logger.info(f"Text column '{column.name}' vectorized using **BERT Embeddings**.")
        else:
            label_encoder = LabelEncoder()
            vectorized_data = label_encoder.fit_transform(column.fillna("UNKNOWN_TEXT"))
            self.logger.info(f"Text column '{column.name}' encoded using **Label Encoding** (Categorical).")
        
        return vectorized_data
