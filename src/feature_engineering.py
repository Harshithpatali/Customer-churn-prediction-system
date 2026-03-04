import pandas as pd


def add_feature_engineering(df: pd.DataFrame):

    df = df.copy()

    df["tenure_bin"] = pd.cut(
        df["tenure"],
        bins=5,
        labels=False
    )

    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    df["monthly_ratio"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    return df