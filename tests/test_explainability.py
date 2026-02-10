"""Tests for model explainability module."""

import json

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml_pipeline.explainability import (
    compute_shap_values,
    explain_single_prediction,
    generate_feature_importance,
    save_explainability_report,
)
from ml_pipeline.feature_engineering import build_features


class TestComputeShapValues:
    def test_returns_shap_dict_linear(self, sample_dataframe, sample_config):
        """Test SHAP computation with LinearExplainer (Logistic Regression)."""
        X, y, _, feature_names = build_features(sample_dataframe, sample_config, fit=True)
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)

        result = compute_shap_values(model, X, feature_names, sample_config)

        assert "shap_values" in result
        assert "feature_names" in result
        assert "expected_value" in result
        assert "sample_size" in result
        assert result["shap_values"].shape[1] == len(feature_names)
        assert result["feature_names"] == feature_names
        assert isinstance(result["expected_value"], float)

    def test_returns_shap_dict_tree(self, sample_dataframe, sample_config):
        """Test SHAP computation with TreeExplainer (Random Forest)."""
        X, y, _, feature_names = build_features(sample_dataframe, sample_config, fit=True)
        model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X, y)

        result = compute_shap_values(model, X, feature_names, sample_config)

        assert result["shap_values"].shape[1] == len(feature_names)
        assert result["sample_size"] <= X.shape[0]

    def test_samples_large_datasets(self, sample_dataframe, sample_config):
        """Test that SHAP sampling is applied for large datasets."""
        X, y, _, feature_names = build_features(sample_dataframe, sample_config, fit=True)
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)

        # Set sample_size smaller than dataset
        sample_config["explainability"] = {"sample_size": 50}
        result = compute_shap_values(model, X, feature_names, sample_config)
        assert result["sample_size"] == 50
        assert result["shap_values"].shape[0] == 50


class TestGenerateFeatureImportance:
    def test_sorted_descending(self, sample_dataframe, sample_config):
        X, y, _, feature_names = build_features(sample_dataframe, sample_config, fit=True)
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)

        result = compute_shap_values(model, X, feature_names, sample_config)
        importance = generate_feature_importance(result["shap_values"], feature_names)

        assert len(importance) == len(feature_names)
        # Check descending order
        values = [item["importance"] for item in importance]
        assert values == sorted(values, reverse=True)
        # Each entry has the right keys
        for item in importance:
            assert "feature" in item
            assert "importance" in item
            assert isinstance(item["importance"], float)

    def test_max_features_limit(self, sample_dataframe, sample_config):
        X, y, _, feature_names = build_features(sample_dataframe, sample_config, fit=True)
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)

        result = compute_shap_values(model, X, feature_names, sample_config)
        importance = generate_feature_importance(
            result["shap_values"], feature_names, max_features=3,
        )
        assert len(importance) == 3


class TestSaveExplainabilityReport:
    def test_creates_json_file(self, tmp_path, sample_dataframe, sample_config):
        X, y, _, feature_names = build_features(sample_dataframe, sample_config, fit=True)
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)

        shap_result = compute_shap_values(model, X, feature_names, sample_config)
        importance = generate_feature_importance(shap_result["shap_values"], feature_names)

        output_path = str(tmp_path / "explain_report.json")
        save_explainability_report(shap_result, importance, output_path)

        with open(output_path) as f:
            report = json.load(f)

        assert "expected_value" in report
        assert "sample_size" in report
        assert "feature_importance" in report
        assert len(report["feature_importance"]) == len(feature_names)


class TestExplainSinglePrediction:
    def test_returns_contributions(self, sample_dataframe, sample_config):
        X, y, preprocessor, feature_names = build_features(
            sample_dataframe, sample_config, fit=True,
        )
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)

        # Explain one row
        single_row = sample_dataframe.iloc[:1]
        explanation = explain_single_prediction(
            model, preprocessor, single_row, feature_names, sample_config,
        )

        assert "prediction" in explanation
        assert "base_value" in explanation
        assert "contributions" in explanation
        assert "probability" in explanation
        assert explanation["prediction"] in (0, 1)
        assert 0.0 <= explanation["probability"] <= 1.0
        assert len(explanation["contributions"]) == len(feature_names)
        # Contributions should be sorted by absolute value
        abs_vals = [abs(c["contribution"]) for c in explanation["contributions"]]
        assert abs_vals == sorted(abs_vals, reverse=True)

    def test_batch_returns_list(self, sample_dataframe, sample_config):
        X, y, preprocessor, feature_names = build_features(
            sample_dataframe, sample_config, fit=True,
        )
        model = LogisticRegression(max_iter=200, random_state=42)
        model.fit(X, y)

        # Explain multiple rows
        batch = sample_dataframe.iloc[:3]
        explanations = explain_single_prediction(
            model, preprocessor, batch, feature_names, sample_config,
        )

        assert isinstance(explanations, list)
        assert len(explanations) == 3
        for exp in explanations:
            assert "prediction" in exp
            assert "contributions" in exp
