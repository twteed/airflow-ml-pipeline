"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_config(tmp_path):
    """Minimal pipeline config for testing."""
    return {
        "data": {
            "n_samples": 500,
            "test_size": 0.2,
            "validation_size": 0.1,
            "random_state": 42,
            "raw_data_path": str(tmp_path / "raw.csv"),
            "processed_data_path": str(tmp_path / "processed.csv"),
        },
        "features": {
            "numerical": [
                "tenure", "monthly_charges", "total_charges",
                "num_support_tickets", "avg_monthly_usage_gb",
            ],
            "categorical": ["contract_type", "payment_method", "internet_service"],
            "target": "churn",
            "scaling_method": "standard",
            "handle_outliers": True,
            "outlier_threshold": 3.0,
        },
        "training": {
            "models": {
                "logistic_regression": {
                    "enabled": True,
                    "params": {"C": [1.0], "penalty": ["l2"], "max_iter": [200]},
                },
                "random_forest": {
                    "enabled": True,
                    "params": {
                        "n_estimators": [50],
                        "max_depth": [5],
                        "min_samples_split": [2],
                        "min_samples_leaf": [1],
                    },
                },
            },
            "cv_folds": 3,
            "scoring_metric": "f1",
            "random_state": 42,
        },
        "evaluation": {
            "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
            "min_f1_score": 0.3,
            "min_roc_auc": 0.3,
            "comparison_metric": "f1",
        },
        "deployment": {
            "model_registry_path": str(tmp_path / "registry"),
            "champion_model_path": str(tmp_path / "champion"),
            "serving_port": 8080,
            "min_performance_threshold": 0.3,
        },
        "mlflow": {
            "experiment_name": "test_experiment",
            "tracking_uri": str(tmp_path / "mlruns"),
        },
    }


@pytest.fixture
def sample_dataframe():
    """Small sample DataFrame for testing."""
    rng = np.random.RandomState(42)
    n = 200
    return pd.DataFrame({
        "tenure": rng.randint(1, 72, n),
        "contract_type": rng.choice(
            ["month-to-month", "one-year", "two-year"], n
        ),
        "payment_method": rng.choice(
            ["electronic_check", "mailed_check", "bank_transfer", "credit_card"], n
        ),
        "internet_service": rng.choice(["fiber_optic", "dsl", "none"], n),
        "monthly_charges": rng.uniform(20, 100, n).round(2),
        "total_charges": rng.uniform(100, 5000, n).round(2),
        "num_support_tickets": rng.poisson(2, n),
        "avg_monthly_usage_gb": rng.exponential(15, n).round(2),
        "churn": rng.choice([0, 1], n, p=[0.7, 0.3]),
    })
