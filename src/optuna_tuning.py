import optuna
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.residual_attention_model import build_model


def objective(trial, X, y):

    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    dropout = trial.suggest_float("dropout", 0.1, 0.5)

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    epochs = trial.suggest_int("epochs", 10, 25)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = []

    for train_idx, val_idx in skf.split(X, y):

        X_train, X_val = X[train_idx], X[val_idx]

        y_train, y_val = y[train_idx], y[val_idx]

        model = build_model(
            input_dim=X.shape[1],
            dropout_rate=dropout,
            learning_rate=learning_rate
        )

        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        score = model.evaluate(X_val, y_val, verbose=0)

        scores.append(score[-1])  # ROC-AUC

    return np.mean(scores)


def tune_hyperparameters(X, y):

    study = optuna.create_study(direction="maximize")

    study.optimize(lambda trial: objective(trial, X, y), n_trials=5)

    print("Best parameters:", study.best_params)

    return study.best_params