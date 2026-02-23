# Production-Ready ML Churn System

## Data Lifecycle Design

This repository follows a strict **lifecycle-based data layout**:

```
data/
├── raw/        # Immutable source data
├── validated/  # Schema-validated data
└── processed/  # Feature-ready, versioned data
```

### Why this design?
- Ensures data lineage and reproducibility
- Prevents silent schema breakage
- Aligns with real production ML systems

## Drift Detection Strategy

Drift detection operates on **versioned processed data**:
- Each training run snapshots processed data with a timestamp
- Drift compares the latest snapshot vs the previous one
- No separate reference/current folders required

This mirrors real-world systems where data evolves over time.

## Model Persistence

The full sklearn **Pipeline** (preprocessing + model) is persisted.
Models are never saved in isolation.

## Running the pipeline

```bash
pip install -r requirements.txt
python src/pipeline/main.py
```

Artifacts generated:
- models/churn_pipeline.joblib
- data/validated/*.csv
- data/processed/processed_*.csv
- reports/drift_report.json
- logs/pipeline.log