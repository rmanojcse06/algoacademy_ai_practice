import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from config import DATA_PATH, TARGET_COL
from preprocessing import preprocessor
from data_validation import validate

df = pd.read_csv(DATA_PATH)
df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

validate(df)

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "models/churn_model_v1.pkl")
print("✅ Model trained and saved")
