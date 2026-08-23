from pathlib import Path

import pandas as pd


def clean_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Fill missing numeric values using a simple strategy."""
    cleaned = df.copy()
    numeric_cols = cleaned.select_dtypes(include="number").columns

    if strategy == "median":
        for col in numeric_cols:
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
    elif strategy == "mean":
        for col in numeric_cols:
            cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
    else:
        raise ValueError("strategy must be 'median' or 'mean'")

    return cleaned


def split_features_target(df: pd.DataFrame, target_col: str = "failure") -> tuple[pd.DataFrame, pd.Series]:
    """Split the DataFrame into feature matrix and target series."""
    features = df.drop(columns=[target_col], errors="ignore")
    target = df[target_col]
    return features, target


def prepare_model_ready_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned frame ready for modelling with basic ID handling."""
    cleaned = clean_missing_values(df)
    if "timestamp" in cleaned.columns:
        cleaned = cleaned.drop(columns=["timestamp"])
    if "turbine_id" in cleaned.columns:
        cleaned = cleaned.drop(columns=["turbine_id"])
    return cleaned
