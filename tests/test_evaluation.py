"""Tests for model evaluation module."""

import json

import numpy as np

from ml_pipeline.evaluation import (
    evaluate_all_models,
    evaluate_model,
    save_evaluation_report,
    select_champion,
)
from ml_pipeline.feature_engineering import build_features
from ml_pipeline.training import train_all_models


class TestEvaluateModel:
    def test_returns_expected_metrics(self, sample_dataframe, sample_config):
        X, y, _, _ = build_features(sample_dataframe, sample_config, fit=True)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)

        scores = evaluate_model(model, X, y)
        assert "accuracy" in scores
        assert "f1" in scores
        assert "roc_auc" in scores
        assert "confusion_matrix" in scores
        assert 0 <= scores["accuracy"] <= 1
        assert 0 <= scores["f1"] <= 1

    def test_custom_metrics(self, sample_dataframe, sample_config):
        X, y, _, _ = build_features(sample_dataframe, sample_config, fit=True)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)

        scores = evaluate_model(model, X, y, metrics=["accuracy", "f1"])
        assert "accuracy" in scores
        assert "f1" in scores
        # roc_auc not requested
        assert "roc_auc" not in scores


class TestSelectChampion:
    def test_selects_best_model(self):
        all_scores = {
            "model_a": {"metrics": {"f1": 0.70, "roc_auc": 0.75}, "run_id": None},
            "model_b": {"metrics": {"f1": 0.85, "roc_auc": 0.80}, "run_id": None},
        }
        config = {
            "evaluation": {
                "comparison_metric": "f1",
                "min_f1_score": 0.5,
                "min_roc_auc": 0.5,
            }
        }
        result = select_champion(all_scores, config)
        assert result is not None
        name, _ = result
        assert name == "model_b"

    def test_filters_below_threshold(self):
        all_scores = {
            "model_a": {"metrics": {"f1": 0.30, "roc_auc": 0.40}, "run_id": None},
        }
        config = {
            "evaluation": {
                "comparison_metric": "f1",
                "min_f1_score": 0.5,
                "min_roc_auc": 0.5,
            }
        }
        result = select_champion(all_scores, config)
        assert result is None


class TestSaveEvaluationReport:
    def test_creates_json_file(self, tmp_path):
        all_scores = {
            "lr": {"metrics": {"f1": 0.8, "accuracy": 0.85, "confusion_matrix": [[10, 2], [3, 15]]}},
        }
        path = str(tmp_path / "report.json")
        save_evaluation_report(all_scores, "lr", path)

        with open(path) as f:
            report = json.load(f)
        assert report["champion"] == "lr"
        assert "lr" in report["models"]
