import pandas as pd

CATEGORICAL_COLUMNS = [
    "InternetService",
    "Contract",
    "PaymentMethod"
]


def encode_features(df: pd.DataFrame):

    df = df.copy()

    df = pd.get_dummies(df, columns=CATEGORICAL_COLUMNS)

    return df


def split_features_target(df):

    X = df.drop("Churn", axis=1)

    y = df["Churn"]

    return X, y