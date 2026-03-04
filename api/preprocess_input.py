import pandas as pd
import joblib

from src.data_cleaning import clean_telco_data
from src.feature_engineering import add_feature_engineering
from src.preprocessing import encode_features


def preprocess_input(data_dict):

    df = pd.DataFrame([data_dict])

    # cleaning
    df = clean_telco_data(df)

    # feature engineering
    df = add_feature_engineering(df)

    # encoding
    df = encode_features(df)
    df = df.fillna(0)
    # load training columns
    columns = joblib.load("models/columns.pkl")

    for col in columns:
        if col not in df.columns:
            df[col] = 0

    df = df[columns]

    # scaling
    scaler = joblib.load("models/scaler.pkl")

    X_scaled = scaler.transform(df)

    return X_scaled