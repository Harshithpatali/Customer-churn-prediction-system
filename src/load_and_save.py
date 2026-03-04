import os
import pandas as pd

DATA_DIR = "data"


def load_raw_data(filename="telco.csv"):

    path = os.path.join(DATA_DIR, filename)

    df = pd.read_csv(path)

    print(f"Loaded {filename} shape:", df.shape)

    return df


def save_dataframe(df, filename):

    path = os.path.join(DATA_DIR, filename)

    df.to_csv(path, index=False)

    print(f"Saved -> {filename} shape:", df.shape)