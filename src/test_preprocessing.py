from src.load_and_save import load_dataframe
from src.preprocessing import encode_features, split_features_target
from src.synthetic_data import expand_dataset
from src.smote_balance import apply_smote
from src.scaling import scale_features


def main():

    df = load_dataframe("feature_engineered_telco.csv")

    print("Original shape:", df.shape)

    df = encode_features(df)

    print("After encoding:", df.shape)

    df = expand_dataset(df)

    print("After expansion:", df.shape)

    X, y = split_features_target(df)

    X, y = apply_smote(X, y)

    print("After SMOTE:", X.shape)

    X_scaled, scaler = scale_features(X)

    print("Scaled shape:", X_scaled.shape)


if __name__ == "__main__":
    main()