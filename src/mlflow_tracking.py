import mlflow
import mlflow.keras
from mlflow.models import infer_signature
import pandas as pd


def start_mlflow():
    mlflow.set_experiment("Customer_Churn_Attention_Model")


def log_params(params):

    for k, v in params.items():
        mlflow.log_param(k, v)


def log_metrics(metrics):

    for k, v in metrics.items():
        mlflow.log_metric(k, v)


def log_model(model, X_sample):

    signature = infer_signature(X_sample, model.predict(X_sample))

    mlflow.keras.log_model(
        model,
        name="attention_model",
        signature=signature
    )