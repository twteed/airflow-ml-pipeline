"""Model explainability module — SHAP-based feature importance and prediction explanations."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def _get_explainer(model, X_background: np.ndarray):
    """Select the appropriate SHAP explainer based on model type.

    Uses TreeExplainer for tree-based models (Random Forest, XGBoost)
    and LinearExplainer for linear models (Logistic Regression).

    Args:
        model: Trained sklearn-compatible model.
        X_background: Background dataset for the explainer.

    Returns:
        SHAP explainer instance.
    """
    model_class = type(model).__name__

    if model_class in ("RandomForestClassifier", "XGBClassifier",
                       "GradientBoostingClassifier", "DecisionTreeClassifier"):
        logger.info("Using TreeExplainer for %s", model_class)
        return shap.TreeExplainer(model)
    elif model_class in ("LogisticRegression", "SGDClassifier", "RidgeClassifier"):
        logger.info("Using LinearExplainer for %s", model_class)
        return shap.LinearExplainer(model, X_background)
    else:
        logger.info("Using KernelExplainer (fallback) for %s", model_class)
        return shap.KernelExplainer(model.predict_proba, X_background)


def compute_shap_values(
    model: Any,
    X: np.ndarray,
    feature_names: list[str],
    config: dict,
) -> dict:
    """Compute SHAP values for a model on the given dataset.

    Args:
        model: Trained sklearn-compatible model.
        X: Feature matrix (post-preprocessing).
        feature_names: List of feature names matching X columns.
        config: Pipeline configuration with optional explainability settings.

    Returns:
        Dictionary with:
            - shap_values: np.ndarray of SHAP values
            - feature_names: list of feature names
            - expected_value: base value (model's average output)
            - sample_size: number of rows used
    """
    explain_cfg = config.get("explainability", {})
    sample_size = explain_cfg.get("sample_size", 500)

    # Sample data if larger than configured sample size
    if X.shape[0] > sample_size:
        rng = np.random.RandomState(42)
        indices = rng.choice(X.shape[0], size=sample_size, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X

    explainer = _get_explainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)

    # For binary classifiers, shap_values may be a list of [class_0, class_1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use class 1 (churn) explanations

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

    logger.info(
        "Computed SHAP values — shape: %s, sample_size: %d",
        shap_values.shape, X_sample.shape[0],
    )

    return {
        "shap_values": shap_values,
        "feature_names": feature_names,
        "expected_value": float(expected_value),
        "sample_size": X_sample.shape[0],
    }


def generate_feature_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    max_features: int | None = None,
) -> list[dict]:
    """Compute global feature importance from SHAP values.

    Importance is defined as mean |SHAP value| across all samples.

    Args:
        shap_values: SHAP value matrix (n_samples, n_features).
        feature_names: Feature names matching columns of shap_values.
        max_features: Maximum number of features to return (top-N).

    Returns:
        List of {"feature": str, "importance": float} dicts, sorted descending.
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance = [
        {"feature": name, "importance": round(float(score), 6)}
        for name, score in zip(feature_names, mean_abs_shap)
    ]
    importance.sort(key=lambda x: x["importance"], reverse=True)

    if max_features is not None:
        importance = importance[:max_features]

    logger.info(
        "Top 5 features: %s",
        [(item["feature"], item["importance"]) for item in importance[:5]],
    )

    return importance


def save_explainability_report(
    shap_result: dict,
    importance: list[dict],
    output_path: str,
) -> None:
    """Persist the explainability report as JSON.

    Args:
        shap_result: Output from compute_shap_values (without raw SHAP array).
        importance: Output from generate_feature_importance.
        output_path: Path to save the JSON report.
    """
    report = {
        "expected_value": shap_result["expected_value"],
        "sample_size": shap_result["sample_size"],
        "feature_importance": importance,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved explainability report to %s", output_path)


def explain_single_prediction(
    model: Any,
    preprocessor,
    input_data: pd.DataFrame,
    feature_names: list[str],
    config: dict,
) -> dict:
    """Explain a single prediction with per-feature SHAP contributions.

    Args:
        model: Trained sklearn model.
        preprocessor: Fitted ColumnTransformer.
        input_data: Raw input DataFrame (1 or more rows).
        feature_names: Feature names from the preprocessor.
        config: Pipeline configuration.

    Returns:
        Dictionary with prediction, probability, and per-feature contributions.
    """
    from ml_pipeline.feature_engineering import add_engineered_features

    input_data = add_engineered_features(input_data)
    X = preprocessor.transform(input_data)

    prediction = model.predict(X)
    proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    explainer = _get_explainer(model, X)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]

    results = []
    for i in range(X.shape[0]):
        contributions = [
            {"feature": name, "contribution": round(float(val), 6)}
            for name, val in zip(feature_names, shap_values[i])
        ]
        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        result = {
            "prediction": int(prediction[i]),
            "base_value": round(float(expected_value), 6),
            "contributions": contributions,
        }
        if proba is not None:
            result["probability"] = round(float(proba[i]), 6)

        results.append(result)

    return results[0] if len(results) == 1 else results
