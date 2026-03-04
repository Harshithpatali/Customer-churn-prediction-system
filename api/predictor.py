import onnxruntime as ort
import numpy as np

from api.preprocess_input import preprocess_input


# Load ONNX model
session = ort.InferenceSession("models/attention_model.onnx")

input_name = session.get_inputs()[0].name


def predict_churn(data):

    X = preprocess_input(data)

    prob = session.run(
        None,
        {input_name: X.astype(np.float32)}
    )[0][0][0]

    prediction = "Churn" if prob > 0.5 else "No Churn"

    return {
        "probability": float(prob),
        "prediction": prediction
    }