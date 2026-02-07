"""Feature engineering module — transforms raw data into model-ready features."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)

SCALER_MAP = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def create_preprocessor(config: dict) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer based on pipeline config.

    Args:
        config: Pipeline configuration dictionary.

    Returns:
        Fitted-ready ColumnTransformer.
    """
    feature_cfg = config["features"]
    numerical_cols = feature_cfg["numerical"]
    categorical_cols = feature_cfg["categorical"]

    scaling_method = feature_cfg.get("scaling_method", "standard")
    scaler_cls = SCALER_MAP.get(scaling_method, StandardScaler)

    numerical_pipeline = Pipeline([
        ("scaler", scaler_cls()),
    ])

    categorical_pipeline = Pipeline([
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features from raw columns.

    Args:
        df: Raw DataFrame.

    Returns:
        DataFrame with additional engineered features.
    """
    df = df.copy()

    # Average charge per month of tenure
    df["charge_per_tenure"] = np.where(
        df["tenure"] > 0,
        df["total_charges"] / df["tenure"],
        df["monthly_charges"],
    )

    # Tenure buckets
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[0, 6, 12, 24, 48, 72],
        labels=["0-6m", "6-12m", "1-2y", "2-4y", "4-6y"],
    ).astype(str)

    # High-value customer flag
    df["high_value"] = (df["monthly_charges"] > df["monthly_charges"].quantile(0.75)).astype(int)

    # Support intensity (tickets per tenure month)
    df["support_intensity"] = np.where(
        df["tenure"] > 0,
        df["num_support_tickets"] / df["tenure"],
        df["num_support_tickets"],
    )

    logger.info("Added 4 engineered features, total columns: %d", len(df.columns))
    return df


def build_features(df: pd.DataFrame, config: dict, fit: bool = True,
                   preprocessor: ColumnTransformer | None = None) -> tuple:
    """Full feature engineering step.

    Args:
        df: Input DataFrame.
        config: Pipeline configuration.
        fit: Whether to fit the preprocessor (True for training, False for inference).
        preprocessor: Pre-fitted preprocessor for inference mode.

    Returns:
        Tuple of (feature_array, target_array, fitted_preprocessor, feature_names).
    """
    feature_cfg = config["features"]

    # Handle outliers before engineering
    if feature_cfg.get("handle_outliers", False):
        threshold = feature_cfg.get("outlier_threshold", 3.0)
        df = _clip_outliers(df, feature_cfg["numerical"], threshold)

    # Add engineered features
    df = add_engineered_features(df)

    # Update column lists to include engineered features
    numerical_cols = feature_cfg["numerical"] + ["charge_per_tenure", "support_intensity"]
    categorical_cols = feature_cfg["categorical"] + ["tenure_bucket"]

    # Build extended config for the preprocessor
    extended_config = {
        "features": {
            **feature_cfg,
            "numerical": numerical_cols,
            "categorical": categorical_cols,
        }
    }

    if fit:
        preprocessor = create_preprocessor(extended_config)
        target = df[feature_cfg["target"]].values
        X = preprocessor.fit_transform(df)
        logger.info("Fitted preprocessor — output shape: %s", X.shape)
    else:
        if preprocessor is None:
            raise ValueError("preprocessor must be provided when fit=False")
        target = df[feature_cfg["target"]].values if feature_cfg["target"] in df.columns else None
        X = preprocessor.transform(df)

    # Get feature names from the fitted preprocessor
    feature_names = preprocessor.get_feature_names_out().tolist()

    return X, target, preprocessor, feature_names


def save_preprocessor(preprocessor: ColumnTransformer, path: str) -> None:
    """Persist the fitted preprocessor to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    logger.info("Saved preprocessor to %s", path)


def load_preprocessor(path: str) -> ColumnTransformer:
    """Load a fitted preprocessor from disk."""
    return joblib.load(path)


def _clip_outliers(df: pd.DataFrame, columns: list[str], threshold: float) -> pd.DataFrame:
    """Clip outliers using z-score thresholding."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                lower = mean - threshold * std
                upper = mean + threshold * std
                clipped = df[col].clip(lower, upper)
                n_clipped = (df[col] != clipped).sum()
                if n_clipped > 0:
                    logger.debug("Clipped %d outliers in '%s'", n_clipped, col)
                df[col] = clipped
    return df
