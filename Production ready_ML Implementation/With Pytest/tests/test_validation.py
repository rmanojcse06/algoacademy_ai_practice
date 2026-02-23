import pandas as pd
import pytest
from src.components.data_validation.data_validation import validate_and_save

def test_validation_raises():
    df = pd.DataFrame({"foo": [1]})
    with pytest.raises(ValueError):
        validate_and_save(df, "bad.csv")