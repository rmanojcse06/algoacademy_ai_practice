from src.config.logging_config import *
from src.components.model_trainer.train import train
from src.components.model_evaluation.evaluate import evaluate
from src.components.drift_detection.drift_detection import detect_drift

if __name__ == "__main__":
    model, X_test, y_test = train()
    metrics = evaluate(model, X_test, y_test)
    drift = detect_drift()

    print("Metrics:", metrics)
    print("Drift:", drift)