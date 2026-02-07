# ML Pipeline Orchestration with Apache Airflow

An end-to-end machine learning pipeline for **customer churn prediction**, orchestrated with Apache Airflow. The project demonstrates production-grade ML engineering practices including automated data ingestion, feature engineering, model training with hyperparameter tuning, evaluation with champion/challenger selection, and model deployment with a REST API.

## Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌────────────────┐    ┌──────────────┐    ┌─────────────┐
│   Data       │    │   Feature        │    │   Model        │    │   Model      │    │   Model     │
│   Ingestion  │───▶│   Engineering    │───▶│   Training     │───▶│   Evaluation │───▶│   Deployment│
│              │    │                  │    │                │    │              │    │             │
│ • Generate/  │    │ • Derived feats  │    │ • Grid search  │    │ • Metrics    │    │ • Champion  │
│   load data  │    │ • Scaling        │    │ • Cross-val    │    │ • Thresholds │    │   promotion │
│ • Validate   │    │ • Encoding       │    │ • MLflow log   │    │ • Selection  │    │ • REST API  │
└─────────────┘    └──────────────────┘    └────────────────┘    └──────────────┘    └─────────────┘
```

**Orchestrated by Apache Airflow** with a weekly retraining schedule and daily data quality monitoring.

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | Apache Airflow 2.9 |
| ML Framework | scikit-learn, XGBoost |
| Experiment Tracking | MLflow |
| Data Processing | pandas, NumPy |
| Model Serving | Flask + Gunicorn |
| Containerization | Docker & Docker Compose |
| Testing | pytest |

## Project Structure

```
├── dags/
│   ├── ml_pipeline_dag.py          # Main pipeline DAG (weekly)
│   └── data_quality_dag.py         # Data quality monitoring DAG (daily)
├── src/ml_pipeline/
│   ├── config.py                   # Configuration management
│   ├── data_ingestion.py           # Data generation, loading, validation
│   ├── feature_engineering.py      # Feature transforms & preprocessing
│   ├── training.py                 # Model training with hyperparameter tuning
│   ├── evaluation.py               # Metrics, champion selection
│   ├── deployment.py               # Model promotion & REST serving
│   └── run_pipeline.py             # Standalone runner (no Airflow needed)
├── tests/                          # Unit tests for each module
├── config/pipeline_config.yaml     # Pipeline configuration
├── docker/Dockerfile
├── docker-compose.yaml             # Airflow + Postgres + MLflow stack
└── Makefile
```

## Pipeline Details

### 1. Data Ingestion
Generates a synthetic customer churn dataset (10K records) with realistic feature correlations. Includes schema validation, range checks, class balance verification, and duplicate detection.

### 2. Feature Engineering
- **Derived features**: charge-per-tenure ratio, tenure buckets, high-value flag, support intensity
- **Preprocessing**: StandardScaler for numerical, OneHotEncoder for categorical
- **Outlier handling**: z-score clipping with configurable threshold
- Fitted preprocessor is persisted for inference consistency

### 3. Model Training
Trains three classifiers with grid search hyperparameter tuning and stratified k-fold cross-validation:
- **Logistic Regression** — interpretable baseline
- **Random Forest** — ensemble with feature importance
- **XGBoost** — gradient boosting for maximum performance

All experiments are tracked in MLflow with parameters, metrics, and model artifacts.

### 4. Model Evaluation
Computes accuracy, precision, recall, F1, and ROC-AUC on a held-out test set. Applies configurable minimum thresholds to filter candidates, then selects the champion by the best comparison metric.

### 5. Model Deployment
Promotes the champion model to a versioned deployment directory with:
- Serialized model + preprocessor artifacts
- Metadata JSON with metrics, parameters, and timestamp
- Automatic archival of previous champion
- Flask REST API for real-time predictions

### Airflow DAGs
- **`ml_pipeline`** — Full training pipeline, runs weekly (Sunday 2 AM UTC). Uses `BranchPythonOperator` to skip deployment if no model meets thresholds.
- **`data_quality_check`** — Daily validation and drift monitoring.

## Quick Start

### Run Locally (without Docker)

```bash
# Install dependencies
pip install -e .
pip install -r requirements.txt

# Run the full pipeline
python -m ml_pipeline.run_pipeline

# Run tests
pytest tests/ -v
```

### Run with Docker Compose

```bash
# Start Airflow + Postgres + MLflow
docker compose up -d --build

# Access services:
#   Airflow UI:  http://localhost:8080  (admin / admin)
#   MLflow UI:   http://localhost:5000

# Optionally start the model serving API:
docker compose --profile serving up -d model-server
# Prediction API: http://localhost:8081

# Stop all services
docker compose down -v
```

### API Usage

```bash
# Health check
curl http://localhost:8081/health

# Single prediction
curl -X POST http://localhost:8081/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 12,
    "contract_type": "month-to-month",
    "payment_method": "electronic_check",
    "internet_service": "fiber_optic",
    "monthly_charges": 79.99,
    "total_charges": 959.88,
    "num_support_tickets": 3,
    "avg_monthly_usage_gb": 45.2
  }'

# Model info
curl http://localhost:8081/model/info
```

## Configuration

All pipeline parameters are defined in `config/pipeline_config.yaml`:
- Data generation and split ratios
- Feature lists and preprocessing methods
- Model hyperparameter search spaces
- Evaluation metrics and minimum thresholds
- Deployment paths

Environment variables (see `.env.example`) override config for Docker deployments.

## Testing

```bash
# Full test suite with coverage
make test

# Quick run
make test-quick
```
