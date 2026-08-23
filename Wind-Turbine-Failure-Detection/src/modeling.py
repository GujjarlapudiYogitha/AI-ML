from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .eda import load_data
from .feature_engineering import add_rolling_features, add_time_features
from .preprocessing import prepare_model_ready_frame, split_features_target


def build_baseline_pipeline(random_state: int = 42) -> Pipeline:
    """Create a simple baseline model pipeline for drivetrain-failure classification."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(n_estimators=200, random_state=random_state, class_weight="balanced")),
        ]
    )


def prepare_training_data(data_path: str | Path | None = None) -> tuple:
    """Load data and return features and target for model training."""
    df = load_data(data_path) if data_path is not None else load_data()
    df = add_time_features(df)
    df = add_rolling_features(df)
    df = prepare_model_ready_frame(df)
    X, y = split_features_target(df, target_col="failure")
    return X, y
