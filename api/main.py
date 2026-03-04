from fastapi import FastAPI
from api.schema import CustomerData
from api.predictor import predict_churn


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Residual Attention Neural Network",
    version="1.0"
)


@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict")
def predict(data: CustomerData):

    result = predict_churn(data.dict())

    return result