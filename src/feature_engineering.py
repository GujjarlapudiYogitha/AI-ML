import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic time-based features from the timestamp column."""
    if "timestamp" not in df.columns:
        return df.copy()

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["month"] = data["timestamp"].dt.month
    return data


def add_rolling_features(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Construct a few rolling statistics for monitoring-related channels."""
    feature_frame = df.copy()
    for col in ["drivetrain_vibration_rms_mmps", "gearbox_bearing_temp_C", "main_bearing_temp_C", "power_output_kW"]:
        if col in feature_frame.columns:
            feature_frame[f"{col}_rolling_mean"] = feature_frame[col].shift(1).rolling(window=window, min_periods=1).mean()
            feature_frame[f"{col}_rolling_std"] = feature_frame[col].shift(1).rolling(window=window, min_periods=1).std().fillna(0)
    return feature_frame
