import tensorflow as tf
import tf2onnx
import os

MODEL_PATH = "models/attention_model.keras"
ONNX_PATH = "models/attention_model.onnx"

model = tf.keras.models.load_model(MODEL_PATH)

spec = (tf.TensorSpec((None, model.input_shape[1]), tf.float32, name="input"),)

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13
)

with open(ONNX_PATH, "wb") as f:
    f.write(model_proto.SerializeToString())

print("ONNX model saved to:", ONNX_PATH)