"""
Data Quality DAG — Runs data validation checks on a daily schedule.

Independent from the main training pipeline, this DAG monitors incoming
data quality and alerts on drift or anomalies.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

default_args = {
    "owner": "ml-engineering",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def _validate_data_quality(**kwargs):
    import pandas as pd
    import yaml

    config_path = kwargs.get("config_path", "/opt/airflow/config/pipeline_config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    raw_path = config["data"]["raw_data_path"]

    try:
        df = pd.read_csv(raw_path)
    except FileNotFoundError:
        logger.warning("No data file found at %s — skipping validation", raw_path)
        return {"status": "skipped", "reason": "no data file"}

    from ml_pipeline.data_ingestion import validate_data

    results = validate_data(df)

    # Basic distribution shift detection
    drift_report = {}
    for col in config["features"]["numerical"]:
        if col in df.columns:
            drift_report[col] = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "null_pct": float(df[col].isnull().mean()),
            }

    if "churn" in df.columns:
        churn_rate = float(df["churn"].mean())
        drift_report["churn_rate"] = churn_rate
        if churn_rate < 0.05 or churn_rate > 0.95:
            logger.warning("Extreme class imbalance detected: churn_rate=%.3f", churn_rate)

    results["drift"] = drift_report
    kwargs["ti"].xcom_push(key="validation_results", value=results)
    return results


with DAG(
    dag_id="data_quality_check",
    default_args=default_args,
    description="Daily data quality validation and drift detection",
    schedule="0 6 * * *",  # Daily at 6 AM UTC
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "data-quality", "monitoring"],
) as dag:

    validate = PythonOperator(
        task_id="validate_data_quality",
        python_callable=_validate_data_quality,
    )
