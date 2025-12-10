.PHONY: help install sync lock dvc-run-pipeline clean test dvc-push dvc-pull dvc-status dvc-dag pipeline-and-push

help:
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:
	uv sync

sync:
	uv sync

lock:
	uv lock

clean:
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete

setup: install
	@echo "Project setup complete"

dvc-run-pipeline:
	uv run dvc repro

dvc-push: ## push dvc data
	uv run dvc push

dvc-pull: ## pull dvc data
	uv run dvc pull

dvc-status: ## check DVC status
	uv run dvc status

dvc-dag:
	uv run dvc dag

dvc-pipeline-and-push: run-pipeline push ## Run pipeline and push results
	@echo "Pipeline completed and pushed to remote"

