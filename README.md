# Mineral Flotation - DVC & DagsHub Exam Project

## Project Overview

This repository implements a complete machine learning workflow for mineral flotation data modeling. The project uses DVC (Data Version Control) for managing data and model versions, and DagsHub for collaboration and experiment tracking. The workflow includes data validation, feature engineering, model training, and evaluation, all orchestrated through a DVC pipeline.

## Project Status

✅ **Plan Created** - A comprehensive execution plan has been created in `task/Plan.md` outlining the complete ML workflow setup.

✅ **Scripts Implemented** - All data preprocessing and model training scripts have been created:
- Data validation (`src/data/validate_data.py`)
- Feature engineering (`src/data/feature_engineering.py`)
- Data splitting (`src/data/data_split.py`)
- Data normalization (`src/data/normalize.py`)
- Grid search (`src/models/grid_search.py`)
- Model training (`src/models/training.py`)
- Model evaluation (`src/models/evaluate.py`)

## Project Structure

```
mineral_flotation/
├── data/
│   ├── processed/
│   └── raw/
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

The project requires the following Python packages (see `requirements.txt` or `pyproject.toml` for versions):

- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms and utilities
- **dvc** - Data version control
- **dvc-s3** - DVC S3 remote support
- **dagshub** - DagsHub integration

### Installation

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

Or using uv (recommended):

```bash
uv sync
```

## Workflow Pipeline

The project uses a DVC pipeline with 7 stages:

1. **validate** - Validates and cleans raw data
2. **feature_engineering** - Creates engineered features (including optional time-series features)
3. **split** - Splits data into train/test sets
4. **normalize** - Normalizes features using StandardScaler (fitted on training data only)
5. **gridsearch** - Performs hyperparameter tuning
6. **train** - Trains the final model with best parameters
7. **evaluate** - Evaluates model performance and generates metrics

### Running the Pipeline

```bash
# Run the entire pipeline
dvc repro

# Run a specific stage
dvc repro validate

# Run from a specific stage onwards
dvc repro train
```

## Data

The raw dataset can be downloaded from:
https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv

Place it in `data/raw/raw.csv` before running the pipeline.

## DVC & DagsHub Setup

### Initialize DVC

```bash
dvc init
```

### Configure DagsHub Remote

```bash
dvc remote add -d origin <DAGS_HUB_REMOTE_URL>
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <YOUR_DAGS_HUB_USERNAME>
dvc remote modify origin --local password <YOUR_DAGS_HUB_TOKEN>
```

### Push Data and Models

```bash
# After running the pipeline
dvc push
```

## Key Features

- **Strict Train/Test Separation**: All preprocessing (scaling, etc.) is fitted only on training data to prevent data leakage
- **Scaler Persistence**: The fitted scaler is saved and reused for inference
- **Comprehensive Data Validation**: Automatic detection and handling of missing values, invalid dates, and data quality issues
- **Flexible Feature Engineering**: Optional time-series cyclic features for model comparison
- **Reproducible Pipeline**: DVC ensures reproducibility across different environments

## Submission

The exam submission will be the link to your DagsHub repository. Make sure to add `licence.pedago` as a collaborator with read-only access for grading.

## Documentation

For detailed implementation steps and execution plan, refer to `task/Plan.md`.
