import pandas as pd

from src.load_and_save import load_raw_data, save_dataframe
from src.data_cleaning import clean_telco_data
from src.feature_engineering import add_feature_engineering
from src.preprocessing import encode_features, split_features_target
from src.smote_balance import apply_smote


def main():

    # -------------------------
    # LOAD RAW DATA
    # -------------------------

    df = load_raw_data("telco.csv")

    save_dataframe(df, "raw_telco.csv")

    # -------------------------
    # CLEAN DATA
    # -------------------------

    df = clean_telco_data(df)

    save_dataframe(df, "cleaned_telco.csv")

    # -------------------------
    # FEATURE ENGINEERING
    # -------------------------

    df = add_feature_engineering(df)

    save_dataframe(df, "feature_engineered_telco.csv")

    # -------------------------
    # ENCODING
    # -------------------------

    df = encode_features(df)

    save_dataframe(df, "encoded_telco.csv")

    # -------------------------
    # SMOTE
    # -------------------------

    X, y = split_features_target(df)

    X_res, y_res = apply_smote(X, y)

    smote_df = pd.DataFrame(X_res, columns=X.columns)

    smote_df["Churn"] = y_res

    save_dataframe(smote_df, "smote_balanced_telco.csv")

    print("Pipeline completed successfully")


if __name__ == "__main__":
    main()