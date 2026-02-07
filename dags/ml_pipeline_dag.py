"""
ML Pipeline DAG — Orchestrates the end-to-end machine learning pipeline.

Steps:
    1. Data Ingestion    — Generate/load and validate customer churn data
    2. Feature Engineering — Transform raw data into model-ready features
    3. Model Training     — Train multiple classifiers with hyperparameter tuning
    4. Model Evaluation   — Evaluate models and select champion
    5. Model Deployment   — Promote champion model for serving

Schedule: Weekly (every Sunday at 2 AM UTC)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

logger = logging.getLogger(__name__)

default_args = {
    "owner": "ml-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _load_config(**kwargs):
    import yaml

    config_path = kwargs.get("config_path", "/opt/airflow/config/pipeline_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    kwargs["ti"].xcom_push(key="config", value=config)
    return config


def _ingest_data(**kwargs):
    from ml_pipeline.data_ingestion import ingest_data

    config = kwargs["ti"].xcom_pull(task_ids="load_config", key="config")
    df = ingest_data(config)
    raw_path = config["data"]["raw_data_path"]
    df.to_csv(raw_path, index=False)
    kwargs["ti"].xcom_push(key="data_path", value=raw_path)
    kwargs["ti"].xcom_push(key="n_records", value=len(df))
    return raw_path


def _build_features(**kwargs):
    import joblib
    import numpy as np
    import pandas as pd
    from ml_pipeline.feature_engineering import build_features

    config = kwargs["ti"].xcom_pull(task_ids="load_config", key="config")
    data_path = kwargs["ti"].xcom_pull(task_ids="ingest_data", key="data_path")

    df = pd.read_csv(data_path)
    X, y, preprocessor, feature_names = build_features(df, config, fit=True)

    # Save artifacts for downstream tasks
    from pathlib import Path
    artifacts_dir = Path(config["data"]["processed_data_path"]).parent
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(artifacts_dir / "X.npy"), X)
    np.save(str(artifacts_dir / "y.npy"), y)
    joblib.dump(preprocessor, str(artifacts_dir / "preprocessor.joblib"))

    kwargs["ti"].xcom_push(key="artifacts_dir", value=str(artifacts_dir))
    kwargs["ti"].xcom_push(key="feature_names", value=feature_names)
    kwargs["ti"].xcom_push(key="n_features", value=X.shape[1])
    return str(artifacts_dir)


def _train_models(**kwargs):
    import mlflow
    import numpy as np
    from sklearn.model_selection import train_test_split
    from ml_pipeline.training import train_all_models

    config = kwargs["ti"].xcom_pull(task_ids="load_config", key="config")
    artifacts_dir = kwargs["ti"].xcom_pull(task_ids="build_features", key="artifacts_dir")

    X = np.load(f"{artifacts_dir}/X.npy")
    y = np.load(f"{artifacts_dir}/y.npy")

    test_size = config["data"].get("test_size", 0.2)
    random_state = config["data"].get("random_state", 42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    np.save(f"{artifacts_dir}/X_test.npy", X_test)
    np.save(f"{artifacts_dir}/y_test.npy", y_test)

    mlflow.set_tracking_uri(config["mlflow"].get("tracking_uri", "mlruns"))

    import joblib
    with mlflow.start_run(run_name="pipeline_run"):
        trained_models = train_all_models(X_train, y_train, config)

    joblib.dump(trained_models, f"{artifacts_dir}/trained_models.joblib")
    kwargs["ti"].xcom_push(key="model_names", value=list(trained_models.keys()))
    return list(trained_models.keys())


def _evaluate_models(**kwargs):
    import joblib
    import numpy as np
    from ml_pipeline.evaluation import evaluate_all_models, save_evaluation_report, select_champion

    config = kwargs["ti"].xcom_pull(task_ids="load_config", key="config")
    artifacts_dir = kwargs["ti"].xcom_pull(task_ids="build_features", key="artifacts_dir")

    X_test = np.load(f"{artifacts_dir}/X_test.npy")
    y_test = np.load(f"{artifacts_dir}/y_test.npy")
    trained_models = joblib.load(f"{artifacts_dir}/trained_models.joblib")

    evaluation_results = evaluate_all_models(trained_models, X_test, y_test, config)
    result = select_champion(evaluation_results, config)

    if result is None:
        kwargs["ti"].xcom_push(key="champion_name", value=None)
        return None

    champion_name, champion_result = result
    save_evaluation_report(evaluation_results, champion_name, f"{artifacts_dir}/evaluation_report.json")

    kwargs["ti"].xcom_push(key="champion_name", value=champion_name)
    kwargs["ti"].xcom_push(key="champion_metrics", value={
        k: v for k, v in champion_result["metrics"].items() if isinstance(v, (int, float))
    })
    return champion_name


def _check_champion(**kwargs):
    """Branch: deploy if we have a champion, otherwise notify."""
    champion_name = kwargs["ti"].xcom_pull(task_ids="evaluate_models", key="champion_name")
    if champion_name:
        return "deploy_model"
    return "notify_no_champion"


def _deploy_model(**kwargs):
    import joblib
    from ml_pipeline.deployment import promote_model

    config = kwargs["ti"].xcom_pull(task_ids="load_config", key="config")
    artifacts_dir = kwargs["ti"].xcom_pull(task_ids="build_features", key="artifacts_dir")
    champion_name = kwargs["ti"].xcom_pull(task_ids="evaluate_models", key="champion_name")
    champion_metrics = kwargs["ti"].xcom_pull(task_ids="evaluate_models", key="champion_metrics")
    feature_names = kwargs["ti"].xcom_pull(task_ids="build_features", key="feature_names")

    trained_models = joblib.load(f"{artifacts_dir}/trained_models.joblib")
    preprocessor = joblib.load(f"{artifacts_dir}/preprocessor.joblib")
    champion_model = trained_models[champion_name]["model"]

    deploy_path = promote_model(
        model=champion_model,
        preprocessor=preprocessor,
        model_name=champion_name,
        metrics=champion_metrics,
        feature_names=feature_names,
        config=config,
    )

    logger.info("Model deployed to %s", deploy_path)
    return deploy_path


def _notify_no_champion(**kwargs):
    logger.warning("No model met performance thresholds — skipping deployment")


with DAG(
    dag_id="ml_pipeline",
    default_args=default_args,
    description="End-to-end ML pipeline: ingest -> features -> train -> evaluate -> deploy",
    schedule="0 2 * * 0",  # Every Sunday at 2 AM UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "churn-prediction", "pipeline"],
    doc_md=__doc__,
) as dag:

    load_config = PythonOperator(
        task_id="load_config",
        python_callable=_load_config,
    )

    ingest_data = PythonOperator(
        task_id="ingest_data",
        python_callable=_ingest_data,
    )

    build_features = PythonOperator(
        task_id="build_features",
        python_callable=_build_features,
    )

    train_models = PythonOperator(
        task_id="train_models",
        python_callable=_train_models,
    )

    evaluate_models = PythonOperator(
        task_id="evaluate_models",
        python_callable=_evaluate_models,
    )

    check_champion = BranchPythonOperator(
        task_id="check_champion",
        python_callable=_check_champion,
    )

    deploy_model = PythonOperator(
        task_id="deploy_model",
        python_callable=_deploy_model,
    )

    notify_no_champion = PythonOperator(
        task_id="notify_no_champion",
        python_callable=_notify_no_champion,
    )

    # DAG dependency chain
    (
        load_config
        >> ingest_data
        >> build_features
        >> train_models
        >> evaluate_models
        >> check_champion
        >> [deploy_model, notify_no_champion]
    )
