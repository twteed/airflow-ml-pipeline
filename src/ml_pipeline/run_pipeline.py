"""Standalone pipeline runner — executes the full ML pipeline without Airflow."""

import logging
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

from ml_pipeline.config import load_config
from ml_pipeline.data_ingestion import ingest_data
from ml_pipeline.deployment import promote_model
from ml_pipeline.evaluation import evaluate_all_models, save_evaluation_report, select_champion
from ml_pipeline.explainability import compute_shap_values, generate_feature_importance, save_explainability_report
from ml_pipeline.feature_engineering import build_features, save_preprocessor
from ml_pipeline.training import train_all_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(config_path: str | None = None) -> dict:
    """Execute the full ML pipeline end-to-end.

    Returns:
        Dictionary with pipeline results.
    """
    logger.info("=" * 60)
    logger.info("Starting ML Pipeline")
    logger.info("=" * 60)

    config = load_config(config_path)

    # Step 1 — Data Ingestion
    logger.info("Step 1/5: Data Ingestion")
    df = ingest_data(config)
    logger.info("Ingested %d records", len(df))

    # Step 2 — Feature Engineering
    logger.info("Step 2/5: Feature Engineering")
    X, y, preprocessor, feature_names = build_features(df, config, fit=True)
    preprocessor_path = str(
        Path(config["deployment"]["champion_model_path"]).parent / "preprocessor.joblib"
    )
    save_preprocessor(preprocessor, preprocessor_path)
    logger.info("Feature matrix shape: %s", X.shape)

    # Train / test split
    test_size = config["data"].get("test_size", 0.2)
    random_state = config["data"].get("random_state", 42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    logger.info("Train: %d, Test: %d", len(X_train), len(X_test))

    # Step 3 — Model Training
    logger.info("Step 3/5: Model Training")
    trained_models = train_all_models(X_train, y_train, config)

    # Step 4 — Evaluation
    logger.info("Step 4/5: Model Evaluation")
    evaluation_results = evaluate_all_models(trained_models, X_test, y_test, config)
    result = select_champion(evaluation_results, config)

    if result is None:
        logger.error("Pipeline failed — no model meets performance thresholds")
        return {"success": False, "reason": "No model meets thresholds"}

    champion_name, champion_result = result
    report_path = str(
        Path(config["deployment"]["champion_model_path"]).parent / "evaluation_report.json"
    )
    save_evaluation_report(evaluation_results, champion_name, report_path)

    # Step 4.5 — Explainability
    explain_cfg = config.get("explainability", {})
    feature_importance = None
    if explain_cfg.get("enabled", False):
        logger.info("Step 4.5: Model Explainability (SHAP)")
        champion_model = trained_models[champion_name]["model"]
        shap_result = compute_shap_values(champion_model, X_test, feature_names, config)
        max_features = explain_cfg.get("max_display_features", 10)
        feature_importance = generate_feature_importance(
            shap_result["shap_values"], feature_names, max_features=max_features,
        )
        explain_path = str(
            Path(config["deployment"]["champion_model_path"]).parent / "explainability_report.json"
        )
        save_explainability_report(shap_result, feature_importance, explain_path)
        logger.info("Top features: %s", [f["feature"] for f in feature_importance[:5]])
    else:
        logger.info("Explainability disabled — skipping SHAP computation")

    # Step 5 — Deployment
    logger.info("Step 5/5: Model Deployment")
    champion_model = trained_models[champion_name]["model"]
    deploy_path = promote_model(
        model=champion_model,
        preprocessor=preprocessor,
        model_name=champion_name,
        metrics=champion_result["metrics"],
        feature_names=feature_names,
        config=config,
        feature_importance=feature_importance,
    )

    logger.info("=" * 60)
    logger.info("Pipeline complete — champion: %s", champion_name)
    logger.info("Deployed to: %s", deploy_path)
    logger.info("=" * 60)

    return {
        "success": True,
        "champion": champion_name,
        "metrics": {
            k: v for k, v in champion_result["metrics"].items() if isinstance(v, float)
        },
        "deploy_path": deploy_path,
    }


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_pipeline(config_path)
