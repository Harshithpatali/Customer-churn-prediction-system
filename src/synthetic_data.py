import numpy as np
import pandas as pd


def expand_dataset(df: pd.DataFrame, target_size=100000):

    df = df.copy()

    current_size = len(df)

    if current_size >= target_size:
        return df

    multiplier = int(target_size / current_size) + 1

    expanded = pd.concat([df] * multiplier, ignore_index=True)

    expanded = expanded.sample(frac=1).reset_index(drop=True)

    expanded = expanded.iloc[:target_size]

    numeric_cols = expanded.select_dtypes(include=[np.number]).columns

    numeric_cols = [c for c in numeric_cols if c != "Churn"]

    noise = np.random.normal(0, 0.01, expanded[numeric_cols].shape)

    expanded[numeric_cols] = expanded[numeric_cols] + noise

    return expanded