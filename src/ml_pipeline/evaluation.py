"""Model evaluation module — computes metrics, compares models, and selects a champion."""

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

METRIC_FUNCTIONS = {
    "accuracy": accuracy_score,
    "precision": lambda y, p: precision_score(y, p, zero_division=0),
    "recall": lambda y, p: recall_score(y, p, zero_division=0),
    "f1": lambda y, p: f1_score(y, p, zero_division=0),
    "roc_auc": roc_auc_score,
}


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metrics: list[str] | None = None,
) -> dict:
    """Evaluate a trained model on the test set.

    Args:
        model: Trained sklearn-compatible model.
        X_test: Test feature matrix.
        y_test: Test target vector.
        metrics: List of metric names to compute.

    Returns:
        Dictionary of metric name -> score.
    """
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    y_pred = model.predict(X_test)

    scores = {}
    for metric_name in metrics:
        if metric_name not in METRIC_FUNCTIONS:
            logger.warning("Unknown metric: %s, skipping", metric_name)
            continue

        if metric_name == "roc_auc" and hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            scores[metric_name] = float(METRIC_FUNCTIONS[metric_name](y_test, y_proba))
        elif metric_name == "roc_auc":
            scores[metric_name] = float(METRIC_FUNCTIONS[metric_name](y_test, y_pred))
        else:
            scores[metric_name] = float(METRIC_FUNCTIONS[metric_name](y_test, y_pred))

    # Additional detailed outputs
    scores["classification_report"] = classification_report(y_test, y_pred, output_dict=True)
    scores["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()

    logger.info(
        "Evaluation — accuracy: %.4f, f1: %.4f, roc_auc: %.4f",
        scores.get("accuracy", 0),
        scores.get("f1", 0),
        scores.get("roc_auc", 0),
    )

    return scores


def evaluate_all_models(
    trained_models: dict[str, dict],
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
) -> dict[str, dict]:
    """Evaluate all trained models and log metrics to MLflow.

    Args:
        trained_models: Output from train_all_models().
        X_test: Test feature matrix.
        y_test: Test target vector.
        config: Pipeline configuration.

    Returns:
        Dictionary mapping model names to their evaluation scores.
    """
    eval_cfg = config["evaluation"]
    metrics = eval_cfg.get("metrics", ["accuracy", "precision", "recall", "f1", "roc_auc"])
    all_scores = {}

    for model_name, model_info in trained_models.items():
        model = model_info["model"]
        run_id = model_info.get("run_id")

        scores = evaluate_model(model, X_test, y_test, metrics)
        all_scores[model_name] = {"metrics": scores, "run_id": run_id}

        # Log test metrics to MLflow
        if run_id:
            with mlflow.start_run(run_id=run_id):
                for metric_name in metrics:
                    if metric_name in scores:
                        mlflow.log_metric(f"test_{metric_name}", scores[metric_name])

        logger.info(
            "%s — test scores: %s",
            model_name,
            {k: f"{v:.4f}" for k, v in scores.items() if isinstance(v, float)},
        )

    return all_scores


def select_champion(
    all_scores: dict[str, dict],
    config: dict,
) -> tuple[str, dict] | None:
    """Select the best-performing model as the champion.

    Args:
        all_scores: Evaluation scores from evaluate_all_models().
        config: Pipeline configuration.

    Returns:
        Tuple of (champion_model_name, result_dict) or None if no model qualifies.
    """
    eval_cfg = config["evaluation"]
    comparison_metric = eval_cfg.get("comparison_metric", "f1")
    min_f1 = eval_cfg.get("min_f1_score", 0.0)
    min_roc_auc = eval_cfg.get("min_roc_auc", 0.0)

    # Filter models that meet minimum thresholds
    qualifying = {}
    for model_name, result in all_scores.items():
        scores = result["metrics"]
        f1 = scores.get("f1", 0)
        roc_auc = scores.get("roc_auc", 0)

        if f1 >= min_f1 and roc_auc >= min_roc_auc:
            qualifying[model_name] = result
        else:
            logger.warning(
                "%s did not meet thresholds (f1=%.4f >= %.4f, roc_auc=%.4f >= %.4f)",
                model_name, f1, min_f1, roc_auc, min_roc_auc,
            )

    if not qualifying:
        logger.error("No model met minimum thresholds: f1 >= %s, roc_auc >= %s", min_f1, min_roc_auc)
        return None

    # Select champion by comparison metric
    champion_name = max(
        qualifying,
        key=lambda name: qualifying[name]["metrics"].get(comparison_metric, 0),
    )

    logger.info(
        "Champion model: %s (test %s: %.4f)",
        champion_name,
        comparison_metric,
        qualifying[champion_name]["metrics"][comparison_metric],
    )

    return champion_name, qualifying[champion_name]


def save_evaluation_report(
    all_scores: dict[str, dict],
    champion_name: str,
    output_path: str,
) -> None:
    """Save evaluation report as JSON.

    Args:
        all_scores: All model evaluation scores.
        champion_name: Name of the selected champion model.
        output_path: Path to save the report.
    """
    report = {
        "champion": champion_name,
        "models": {},
    }
    for name, result in all_scores.items():
        report["models"][name] = {
            k: v for k, v in result["metrics"].items()
            if isinstance(v, (int, float))
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved evaluation report to %s", output_path)
