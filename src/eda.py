from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "wind_turbine_detection.csv"


def load_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw turbine dataset."""
    return pd.read_csv(path)


def basic_summary(df: pd.DataFrame) -> dict:
    """Return a compact summary for initial EDA."""
    return {
        "shape": df.shape,
        "missing_values": df.isna().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "target_distribution": df["failure"].value_counts().to_dict(),
    }


def describe_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics for numeric variables."""
    numeric = df.select_dtypes(include="number")
    return numeric.describe().T
