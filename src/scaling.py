import joblib
from sklearn.preprocessing import StandardScaler


def scale_features(X, scaler_path="models/scaler.pkl"):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, scaler_path)

    print("Scaler saved")

    return X_scaled


def load_scaler(path="models/scaler.pkl"):

    scaler = joblib.load(path)

    return scaler