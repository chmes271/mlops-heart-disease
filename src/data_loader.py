import os
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", 
    "ca", "thal", "target"
]

def fetch_data(output_path: str = "data/raw/heart.csv") -> pd.DataFrame:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_csv(DATA_URL, names=COLUMN_NAMES, na_values="?")
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    df = fetch_data()
    print(f"Data downloaded successfully. Shape: {df.shape}")