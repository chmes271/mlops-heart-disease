import os
import pandas as pd
import numpy as np
from src.data_loader import fetch_data
from src.preprocess import build_preprocessor, prepare_data

def test_data_fetching():
    df = fetch_data("data/raw/test_heart.csv")
    assert not df.empty
    assert df.shape[1] == 14
    if os.path.exists("data/raw/test_heart.csv"):
        os.remove("data/raw/test_heart.csv")

def test_preprocessing_output_shape():
    df = pd.DataFrame({
        "age": [63, 67], "sex": [1, 1], "cp": [1, 4], 
        "trestbps": [145, 160], "chol": [233, 286], "fbs": [1, 0], 
        "restecg": [2, 2], "thalach": [150, 108], "exang": [0, 1], 
        "oldpeak": [2.3, 1.5], "slope": [3, 2], "ca": [0, 3], 
        "thal": [6, 3], "target": [0, 2]
    })
    X, y = prepare_data(df)
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    
    assert len(y) == 2
    assert y.iloc[0] == 0
    assert y.iloc[1] == 1
    assert X_transformed.shape[0] == 2
    assert not np.isnan(X_transformed).any()