import joblib
import pandas as pd

model = joblib.load("models/churn_model_v1.pkl")

def predict(sample: dict):
    df = pd.DataFrame([sample])
    prob = model.predict_proba(df)[0][1]
    return prob
