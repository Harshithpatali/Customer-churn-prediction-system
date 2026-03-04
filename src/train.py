import joblib
import pandas as pd

from sklearn.model_selection import train_test_split

from src.scaling import scale_features
from src.optuna_tuning import tune_hyperparameters
from src.residual_attention_model import build_model
from src.evaluate import evaluate_model
from src.mlflow_tracking import start_mlflow, log_params, log_metrics, log_model


def main():

    print("Loading dataset...")

    df = pd.read_csv("data/smote_balanced_telco.csv")

    print("Dataset shape:", df.shape)

    X = df.drop("Churn", axis=1)

    y = df["Churn"]

    # Save feature column order
    joblib.dump(X.columns.tolist(), "models/columns.pkl")

    print("Feature columns saved")

    # Convert to numpy
    X = X.values
    y = y.values

    # Scaling
    X_scaled = scale_features(X)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Starting Optuna tuning...")

    best_params = tune_hyperparameters(X_train, y_train)

    print("Training final model...")

    start_mlflow()

    model = build_model(
        input_dim=X_train.shape[1],
        dropout_rate=best_params["dropout"],
        learning_rate=best_params["learning_rate"]
    )

    model.fit(
        X_train,
        y_train,
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        validation_split=0.2
    )

    metrics = evaluate_model(model, X_test, y_test)

    log_params(best_params)

    log_metrics(metrics)

    log_model(model, X_train[:100])

    model.save("models/attention_model.keras")

    print("Training complete")

    print("Final metrics:", metrics)


if __name__ == "__main__":
    main()