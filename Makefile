.PHONY: help install sync lock run-pipeline clean test

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies using uv
	uv sync

sync: ## Sync dependencies with uv.lock
	uv sync

lock: ## Update uv.lock file
	uv lock

run-pipeline: ## Run DVC pipeline
	uv run dvc repro

clean: ## Clean generated files
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete

test: ## Run tests
	@echo "No tests defined yet"

setup: install ## Setup project (install dependencies)
	@echo "Project setup complete"

