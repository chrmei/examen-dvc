# Mineral Flotation - DVC & DagsHub Exam Project

## Project Overview

This repository implements a machine learning workflow for mineral flotation data modeling using DVC for version control and DagsHub for collaboration. The workflow includes data validation, feature engineering, model training, and evaluation, all orchestrated through a DVC pipeline.

## Project Status

✅ **Complete** - All core scripts and DVC pipeline stages are implemented and functional:
- Data validation (`src/data/validate_data.py`)
- Feature engineering (`src/data/feature_engineering.py`)
- Data splitting (`src/data/data_split.py`)
- Data normalization (`src/data/normalize.py`)
- Grid search (`src/models/grid_search.py`)
- Model training (`src/models/training.py`)
- Model evaluation (`src/models/evaluate.py`)

⚠️ **Note on Time-Dependent Features**: While the feature engineering script supports optional time-series cyclic features (via `--use-time-features` flag), the complete end-to-end workflow (creation, gridsearch, training, evaluation) with time-dependent data is not currently implemented as a separate pipeline flow. This could be added in the future to enable model comparison between time-series and non-time-series variants.

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

The project requires:
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms and utilities
- **xgboost** - Gradient boosting model
- **dvc** - Data version control
- **dvc-s3** - DVC S3 remote support
- **dagshub** - DagsHub integration

See `requirements.txt` or `pyproject.toml` for specific versions.

## Workflow Pipeline

The DVC pipeline consists of 7 stages:

1. **validate** - Validates and cleans raw data (handles missing values, invalid dates, data quality checks)
2. **feature_engineering** - Creates engineered features (interaction/ratio features; time-series features disabled by default)
3. **split** - Splits data into train/test sets (80/20 split)
4. **normalize** - Normalizes features using StandardScaler (fitted on training data only, scaler persisted for inference)
5. **gridsearch** - Performs hyperparameter tuning using XGBoost with GridSearchCV
6. **train** - Trains the final XGBoost model with best parameters
7. **evaluate** - Evaluates model performance and generates metrics (MSE, RMSE, R², MAE)


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
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/chrmei/examen-dvc.s3
dvc remote modify origin --local auth basic
dvc remote modify origin --local access_key_id *************************
dvc remote modify origin --local secret_access_key *************************
```

For reference:

```bash
# Stage 1: validate
dvc stage add -n validate \
  -d data/raw/raw.csv \
  -d src/data/validate_data.py \
  -o data/processed/raw_validated.csv \
  python src/data/validate_data.py

# Stage 2: feature_engineering
dvc stage add -n feature_engineering \
  -d data/processed/raw_validated.csv \
  -d src/data/feature_engineering.py \
  -o data/processed/raw_engineered.csv \
  python src/data/feature_engineering.py

# Stage 3: split
dvc stage add -n split \
  -d data/processed/raw_engineered.csv \
  -d src/data/data_split.py \
  -o data/processed/X_train.csv \
  -o data/processed/X_test.csv \
  -o data/processed/y_train.csv \
  -o data/processed/y_test.csv \
  python src/data/data_split.py

# Stage 4: normalize
dvc stage add -n normalize \
  -d data/processed/X_train.csv \
  -d data/processed/X_test.csv \
  -d data/processed/y_train.csv \
  -d data/processed/y_test.csv \
  -d src/data/normalize.py \
  -o data/processed/X_train_scaled.csv \
  -o data/processed/X_test_scaled.csv \
  -o models/data/scaler.pkl \
  python src/data/normalize.py

# Stage 5: gridsearch
dvc stage add -n gridsearch \
  -d data/processed/X_train_scaled.csv \
  -d data/processed/y_train.csv \
  -d src/models/grid_search.py \
  -o models/data/best_params.pkl \
  -o models/data/grid_search_results.csv \
  python src/models/grid_search.py

# Stage 6: train
dvc stage add -n train \
  -d models/data/best_params.pkl \
  -d data/processed/X_train_scaled.csv \
  -d data/processed/y_train.csv \
  -d src/models/training.py \
  -o models/models/trained_model.pkl \
  -o models/models/trained_model_with_history.pkl \
  python src/models/training.py

# Stage 7: evaluate
dvc stage add -n evaluate \
  -d models/models/trained_model.pkl \
  -d data/processed/X_test_scaled.csv \
  -d data/processed/y_test.csv \
  -d src/models/evaluate.py \
  -o data/processed/predictions.csv \
  -o metrics/scores.json \
  -o metrics/learning_curve.png \
  python src/models/evaluate.py
```

### Push Data and Models

```bash
# After running the pipeline
dvc push
```

## Key Features

- **Strict Train/Test Separation**: All preprocessing fitted only on training data to prevent data leakage
- **Scaler Persistence**: Fitted scaler saved and reused for inference
- **Data Validation**: Automatic handling of missing values, invalid dates, and data quality issues
- **Reproducible Pipeline**: DVC ensures reproducibility across environments

## Submission

The exam submission will be the link to your DagsHub repository. Make sure to add `licence.pedago` as a collaborator with read-only access for grading.

## Documentation

For detailed implementation steps and execution plan, refer to `task/Plan.md`.
