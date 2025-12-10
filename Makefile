.PHONY: help install sync lock run-pipeline clean test

help:
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:
	uv sync

sync:
	uv sync

lock:
	uv lock

run-pipeline:
	uv run dvc repro

clean:
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".DS_Store" -delete

test:
	@echo "No tests defined yet"

setup: install
	@echo "Project setup complete"

