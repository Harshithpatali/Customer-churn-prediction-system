from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def evaluate_model(model, X_test, y_test):

    preds = model.predict(X_test)

    preds_binary = (preds > 0.5).astype(int)

    accuracy = accuracy_score(y_test, preds_binary)
    precision = precision_score(y_test, preds_binary)
    recall = recall_score(y_test, preds_binary)
    roc = roc_auc_score(y_test, preds)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc
    }

    return metrics