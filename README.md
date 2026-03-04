# 📉 Customer Churn Prediction System
### Residual Attention Neural Network

An end-to-end Machine Learning system that predicts whether a telecom customer is likely to churn using a **Residual Attention Neural Network**. This project covers the full lifecycle from data engineering and hyperparameter optimization to cloud-native deployment.

---

## 🚀 Live Application
* **🌐 Streamlit UI:** [Customer Churn UI](https://customer-churn-prediction-system-eoz2hnc3sacvv2qz8asotz.streamlit.app)
* **⚡ FastAPI Backend:** [Production API](https://customer-churn-prediction-system-gdvi.onrender.com)
* **📑 API Documentation:** [Swagger UI](https://customer-churn-prediction-system-gdvi.onrender.com/docs)

---

## 📊 Model Performance
The model was evaluated using a Stratified K-Fold approach to ensure robust generalization.

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 0.8347 |
| **Precision** | 0.8176 |
| **Recall** | 0.8618 |
| **ROC-AUC** | 0.9263 |

> **Note:** The high ROC-AUC and Recall indicate the system is exceptionally strong at identifying "at-risk" customers, minimizing false negatives—a critical requirement for retention strategies.

---

## 🧠 Model Architecture
The core engine is a custom **Residual Attention Neural Network** designed to capture both deep non-linear relationships and specific feature importance.



* **Input Layer:** Raw customer features.
* **Dense(128):** Initial feature extraction.
* **Residual Block:** (Dense → Dense + Skip Connection) to prevent vanishing gradients and allow for deeper feature learning.
* **Attention Layer:** Dynamic feature weighting to focus on high-impact customer behavior patterns.
* **Dense(64) & Dropout:** Regularization to prevent overfitting.
* **Sigmoid Output:** Probability-based binary classification.

---

## ⚙️ Data & Pipeline
### Dataset
Utilizes the **IBM Telco Customer Churn Dataset**, covering:
* **Demographics:** Gender, Seniority, Partners, Dependents.
* **Services:** Phone, Multiple Lines, Internet (DSL/Fiber), Security, Backup, Tech Support.
* **Account:** Contract type, Paperless billing, Payment method, Tenure.
* **Financials:** Monthly Charges, Total Charges.

### Preprocessing Workflow
1.  **Cleaning:** Handling missing values and data type conversion.
2.  **Feature Engineering:** * `tenure_bin`: Grouping customers by loyalty stages.
    * `avg_monthly_spend`: Historical spending patterns.
    * `monthly_ratio`: Ratio of current charges to total value.
3.  **Class Balancing:** **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance.
4.  **Scaling:** Standard scaling for numerical convergence.

---

## 🏗 System Architecture
The system is decoupled into a microservices-style architecture for scalability.



1.  **User Interface:** Built with Streamlit for interactive data entry and visualization.
2.  **API Layer:** FastAPI serves the model, handling request validation and preprocessing.
3.  **Inference Engine:** TensorFlow models are converted to **ONNX** format.
4.  **Optimization:** Using **ONNX Runtime** for faster inference and significantly lower memory footprint compared to full TF/Keras environments.

---

## 🛠 Technologies Used
* **ML Frameworks:** TensorFlow, Scikit-learn, SMOTE.
* **Optimization:** **Optuna** (Bayesian Optimization) for hyperparameter tuning.
* **MLOps:** **MLflow** for experiment tracking (metrics, params, artifacts).
* **Backend:** FastAPI, Uvicorn, Pydantic.
* **Frontend:** Streamlit.
* **Deployment:** Render (API), Streamlit Cloud (UI).

---

## 📦 Project Structure
```text
customer-churn-prediction-system/
├── data/                   # Raw and processed datasets
├── models/                 # attention_model.onnx, scaler.pkl, columns.pkl
├── src/                    # Core Logic
│   ├── data_pipeline.py    # Cleaning & Engineering
│   ├── train.py            # Main training script
│   ├── residual_attention_model.py
│   ├── optuna_tuning.py    # Bayesian optimization logic
│   └── mlflow_tracking.py  # Experiment logs
├── api/                    # FastAPI Implementation
│   ├── main.py             # API Routes
│   └── predictor.py        # ONNX Inference engine
└── frontend/               # Streamlit Application
🖥 Example API Request
Endpoint: POST /predict

Payload:

JSON
{
 "gender":"Male",
 "SeniorCitizen":0,
 "Partner":"Yes",
 "Dependents":"No",
 "tenure":12,
 "PhoneService":"Yes",
 "MultipleLines":"No",
 "InternetService":"DSL",
 "OnlineSecurity":"No",
 "OnlineBackup":"Yes",
 "DeviceProtection":"No",
 "TechSupport":"No",
 "StreamingTV":"No",
 "StreamingMovies":"No",
 "Contract":"Month-to-month",
 "PaperlessBilling":"Yes",
 "PaymentMethod":"Electronic check",
 "MonthlyCharges":70,
 "TotalCharges":800
}
Response:

JSON
{
 "probability": 0.83,
 "prediction": "Churn"
}
🎯 Future Roadmap
[ ] Explainability: Integrate SHAP values into the UI for local prediction transparency.

[ ] Monitoring: Build a real-time dashboard to track model drift in production.

[ ] CI/CD: Automated retraining pipeline triggered by data drift detection.

[ ] Orchestration: Migration to Kubernetes (K8s) for high availability and auto-scaling.

Author: [Harshith Devraj]

Machine Learning | Data Science | AI
