import pandas as pd
from src.schema.schema import SCHEMA

def test_schema_columns():
    df = pd.DataFrame(columns=SCHEMA.keys())
    assert set(df.columns) == set(SCHEMA.keys())