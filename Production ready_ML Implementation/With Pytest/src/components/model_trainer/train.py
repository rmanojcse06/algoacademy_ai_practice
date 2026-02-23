import pandas as pd
import joblib
import logging
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.config.config import RAW_DATA_DIR, MODEL_PATH, TARGET_COL, RANDOM_STATE
from src.components.data_validation.data_validation import validate_and_save
from src.components.data_transformation.preprocessing import (
    preprocessor, save_processed
)

logger = logging.getLogger(__name__)

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

def train():
    raw_file = next(RAW_DATA_DIR.glob("*.csv"))
    df = pd.read_csv(raw_file)

    validate_and_save(df, raw_file.name)
    save_processed(df)

    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    best_pipeline, best_auc = None, -1
    mlflow.set_experiment("churn_experiment")

    for name, model in MODELS.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline([
                ("preprocessing", preprocessor),
                ("model", model)
            ])
            pipeline.fit(X_train, y_train)
            auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])

            mlflow.log_param("model", name)
            mlflow.log_metric("roc_auc", auc)
            mlflow.sklearn.log_model(pipeline, "model")

            if auc > best_auc:
                best_auc = auc
                best_pipeline = pipeline

    joblib.dump(best_pipeline, MODEL_PATH)
    logger.info("Saved best pipeline with ROC-AUC %.4f", best_auc)

    return best_pipeline, X_test, y_test