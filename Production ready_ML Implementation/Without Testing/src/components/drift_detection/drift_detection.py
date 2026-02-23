import json
import pandas as pd
from scipy.stats import ks_2samp
from src.config.config import PROCESSED_DATA_DIR, DRIFT_REPORT_PATH
from src.components.data_transformation.preprocessing import NUM_COLS

def detect_drift():
    files = sorted(PROCESSED_DATA_DIR.glob("processed_*.csv"))
    if len(files) < 2:
        raise ValueError("Need at least two processed files for drift detection")

    ref = pd.read_csv(files[-2])
    cur = pd.read_csv(files[-1])

    report = {}
    for col in NUM_COLS:
        _, p = ks_2samp(ref[col], cur[col])
        report[col] = {
            "p_value": float(p),
            "drift_detected": p < 0.05
        }

    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    return report