import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from config import DATA_PATH, TARGET_COL

model = joblib.load("models/churn_model_v1.pkl")

df = pd.read_csv(DATA_PATH)
df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

print("Accuracy :", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred))
print("Recall   :", recall_score(y, y_pred))
print("F1 Score :", f1_score(y, y_pred))
print("ROC AUC  :", roc_auc_score(y, y_prob))

print("\nClassification Report\n")
print(classification_report(y, y_pred))
