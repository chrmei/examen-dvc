# Mineral Flotation - DVC & DagsHub Exam Project

## Project Status

✅ **Plan Created** - A comprehensive execution plan has been created in `task/Plan.md` outlining the complete ML workflow setup.

## Project Overview

This repository contains the architecture for implementing a mineral flotation data modeling solution using DVC for version control and DagsHub for collaboration.

## Project Structure

```
mineral_flotation/
├── data/
│   ├── processed_data/
│   └── raw_data/
├── metrics/
├── models/
│   ├── data/
│   └── models/
├── src/
│   ├── data/
│   └── models/
├── task/
│   ├── Plan.md
│   ├── Task.md
│   └── Schema_exam.png
├── pyproject.toml
├── uv.lock
├── Makefile
└── README.md
```

## Setup

### Prerequisites

- Python 3.11.14 (specified in `.python-version`)
- `uv` package manager

### Installation

```bash
# Install dependencies using uv
make install

# Or manually
uv sync
```

### Available Make Commands

- `make install` - Install dependencies using uv
- `make sync` - Sync dependencies with uv.lock
- `make lock` - Update uv.lock file
- `make run-pipeline` - Run DVC pipeline
- `make clean` - Clean generated files
- `make help` - Show all available commands

## Dependencies

- dagshub==0.6.3
- dvc==3.64.2
- dvc-s3==3.2.2
- numpy==2.3.5
- pandas==2.3.3
- scikit-learn==1.7.2

## Data

You can download the raw dataset from:
https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv

## Submission

The exam submission will be the link to your DagsHub repository. Make sure to add `licence.pedago` as a collaborator with read-only access for grading.

## Next Steps

Refer to `task/Plan.md` for the detailed step-by-step execution plan.
