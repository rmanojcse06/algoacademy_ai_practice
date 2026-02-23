def validate(df):
    assert df.isnull().mean().max() < 0.4, "Too many missing values"
    assert df.duplicated().sum() == 0, "Duplicate rows found"
    assert "Churn" in df.columns, "Target column missing"
