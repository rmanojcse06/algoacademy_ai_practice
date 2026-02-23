import joblib
import pandas as pd
from src.config.config import MODEL_PATH

pipeline = joblib.load(MODEL_PATH)

def predict(input_dict: dict):
    df = pd.DataFrame([input_dict])
    return pipeline.predict(df)[0]