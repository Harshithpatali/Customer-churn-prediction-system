import tensorflow as tf
import numpy as np

from api.preprocess_input import preprocess_input


model = tf.keras.models.load_model("models/attention_model.keras")


def predict_churn(data):

    X = preprocess_input(data)

    prob = model.predict(X)[0][0]

    prediction = "Churn" if prob > 0.5 else "No Churn"

    return {

        "probability": float(prob),

        "prediction": prediction
    }