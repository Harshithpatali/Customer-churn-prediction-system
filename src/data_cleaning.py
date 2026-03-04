import pandas as pd
import numpy as np


def clean_telco_data(df: pd.DataFrame):

    df = df.copy()

    df.replace({
        "No internet service": "No",
        "No phone service": "No"
    }, inplace=True)

    binary_cols = [
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "PaperlessBilling",
        "Churn"
    ]

    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Female": 1, "Male": 0})

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        df[col].fillna(df[col].median(), inplace=True)

    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    return df