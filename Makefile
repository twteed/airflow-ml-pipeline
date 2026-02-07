.PHONY: help install test lint run-local docker-up docker-down clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -e ".[dev]"
	pip install -r requirements.txt

test:  ## Run tests with coverage
	pytest tests/ -v --cov=src/ml_pipeline --cov-report=term-missing

test-quick:  ## Run tests without coverage
	pytest tests/ -v

lint:  ## Run linting
	ruff check src/ tests/ dags/
	ruff format --check src/ tests/ dags/

format:  ## Auto-format code
	ruff format src/ tests/ dags/

run-pipeline:  ## Run the ML pipeline locally (without Airflow)
	python -m ml_pipeline.run_pipeline

docker-up:  ## Start all services with Docker Compose
	docker compose up -d --build

docker-down:  ## Stop all services
	docker compose down -v

clean:  ## Remove generated files
	rm -rf data/*.csv data/*.parquet
	rm -rf models/*.joblib models/*.pkl models/*.json
	rm -rf mlruns/
	rm -rf __pycache__ .pytest_cache .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
