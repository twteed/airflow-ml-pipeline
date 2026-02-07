"""Model training module — trains and tunes multiple classifiers with cross-validation."""

import logging
from typing import Any

import mlflow
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "xgboost": XGBClassifier,
}


def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict,
    cv_folds: int = 5,
    scoring: str = "f1",
    random_state: int = 42,
) -> tuple[Any, dict]:
    """Train a single model with hyperparameter tuning via grid search.

    Args:
        model_name: Key from MODEL_REGISTRY.
        X_train: Training feature matrix.
        y_train: Training target vector.
        param_grid: Hyperparameter search space.
        cv_folds: Number of cross-validation folds.
        scoring: Scoring metric for grid search.
        random_state: Random seed.

    Returns:
        Tuple of (best_estimator, cv_results_dict).
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_REGISTRY.keys())}")

    model_cls = MODEL_REGISTRY[model_name]

    # Set random state if the model supports it
    base_params = {}
    if model_name == "xgboost":
        base_params = {
            "random_state": random_state,
            "eval_metric": "logloss",
            "use_label_encoder": False,
        }
    elif model_name in ("logistic_regression", "random_forest"):
        base_params = {"random_state": random_state}

    model = model_cls(**base_params)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    logger.info("Training %s with %d parameter combinations...", model_name, _count_combinations(param_grid))
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    cv_results = {
        "best_params": grid_search.best_params_,
        "best_cv_score": grid_search.best_score_,
        "mean_train_score": grid_search.cv_results_["mean_train_score"][grid_search.best_index_],
        "std_cv_score": grid_search.cv_results_["std_test_score"][grid_search.best_index_],
    }

    logger.info(
        "%s — best CV %s: %.4f (± %.4f), params: %s",
        model_name,
        scoring,
        cv_results["best_cv_score"],
        cv_results["std_cv_score"],
        cv_results["best_params"],
    )

    return best_model, cv_results


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
) -> dict[str, dict]:
    """Train all enabled models specified in config.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        config: Pipeline configuration dictionary.

    Returns:
        Dictionary mapping model names to their trained model and results.
    """
    training_cfg = config["training"]
    results = {}

    mlflow_cfg = config.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "mlruns"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "churn_prediction"))

    for model_name, model_cfg in training_cfg["models"].items():
        if not model_cfg.get("enabled", False):
            logger.info("Skipping disabled model: %s", model_name)
            continue

        with mlflow.start_run(run_name=model_name, nested=True):
            best_model, cv_results = train_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                param_grid=model_cfg["params"],
                cv_folds=training_cfg.get("cv_folds", 5),
                scoring=training_cfg.get("scoring_metric", "f1"),
                random_state=training_cfg.get("random_state", 42),
            )

            # Log to MLflow
            mlflow.log_params(cv_results["best_params"])
            mlflow.log_metric(f"cv_{training_cfg.get('scoring_metric', 'f1')}", cv_results["best_cv_score"])
            mlflow.log_metric("cv_std", cv_results["std_cv_score"])
            mlflow.log_metric("mean_train_score", cv_results["mean_train_score"])
            mlflow.sklearn.log_model(best_model, artifact_path=model_name)

            results[model_name] = {
                "model": best_model,
                "cv_results": cv_results,
                "run_id": mlflow.active_run().info.run_id,
            }

    logger.info("Trained %d models", len(results))
    return results


def _count_combinations(param_grid: dict) -> int:
    """Count total number of hyperparameter combinations."""
    count = 1
    for values in param_grid.values():
        count *= len(values)
    return count
