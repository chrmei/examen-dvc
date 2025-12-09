# Mineral Flotation DVC & DagsHub Exam Setup Plan

## Overview

This plan sets up a complete ML workflow for mineral flotation data modeling using DVC for version control and DagsHub for collaboration. The workflow includes data preprocessing, model training, and evaluation scripts connected via a DVC pipeline.

### Fix Required (must address)
- Enforce strict train/test separation across all stages (no fitting on combined data, no leakage of test statistics into training).
- Persist the fitted scaler for inference (previously missing); downstream scripts must load the saved scaler.

## Step-by-Step Execution Plan

### Phase 1: Repository Setup

**Step 1: Fork the repository**

- Go to https://github.com/DataScientest-Studio/examen-dvc
- Click "Fork" to create your own copy

**Step 2: Clone your forked repository**

```bash
cd /home/christianm/Projects/Repos/mineral_flotation
git clone https://github.com/YOUR_USERNAME/examen-dvc
cd examen-dvc
```

**Step 3: Set up Python environment with pyenv**

```bash
# Install Python version (if not already installed)
pyenv install 3.11.0  # or your preferred Python version

# Set local Python version for this project
pyenv local 3.11.0

# Verify Python version
python --version

# Optional: Create a virtual environment for isolation (recommended)
python -m venv venv
source venv/bin/activate
```

**Step 4: Install required packages**

```bash
pip install pandas numpy scikit-learn dvc dvc-s3 dagshub
```

**Step 5: Download the raw dataset**

```bash
mkdir -p data/raw
curl -o data/raw/raw.csv https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv
```

### Phase 2: Script Creation

**Step 6: Data Exploration**

- Perform comprehensive data exploration using Jupyter notebook
- Analyze data quality, distributions, correlations, and relationships
- Document findings and insights
- Result: `notebooks/data_exploration.ipynb` (existing notebook contains the exploration results)

**Step 7: Create data validation script (`src/data/validate_data.py`)**

- Load `raw.csv` from `data/raw/`
- Validate and clean data before feature engineering to ensure automation robustness:
  - **Missing value handling**: Detect and handle missing values with configurable strategies (drop, forward_fill, backward_fill, mean, median, zero)
  - **Date validation**: Validate date column format, convert invalid dates, handle missing dates
  - **Data quality checks**:
    - Validate required columns exist
    - Detect and remove duplicate rows
    - Validate numeric data types
    - Detect and handle infinite values
    - Check for negative values in columns that should be positive
  - Generate comprehensive validation report with warnings and errors
- Save validated dataset to `data/processed/raw_validated.csv`
- This step ensures the pipeline is future-proof and handles data quality issues automatically

**Step 8: Create feature engineering script (`src/data/feature_engineering.py`)**

- Load `raw_validated.csv` from `data/processed/` (validated data from Step 7)
- Convert `date` column to datetime format (with validation checks)
- Add a config flag to make time-series feature engineering optional so we can train one model with time features and one without; if enabled, create cyclic encodings for time columns to preserve periodicity while avoiding discontinuities:
  - `hour`: Hour of day (0-23)
  - `day_of_week`: Day of week (0-6, Monday=0)
  - `month`: Month (1-12)
  - `day`: Day of month (1-31)
  - `is_weekend`: Boolean flag (1 if Saturday/Sunday, 0 otherwise)
- Create interaction/ratio features:
  - `amina_starch_ratio`: Ratio of amina_flow to starch_flow
  - `flow_total`: Sum of all flow features (ave_flot_air_flow, starch_flow, amina_flow, ore_pulp_flow)
  - `pulp_flow_density`: Product of ore_pulp_flow and ore_pulp_density
  - `ph_density_interaction`: Product of ore_pulp_pH and ore_pulp_density
- Keep all original features
- Save engineered dataset to `data/processed/raw_engineered.csv`
  - If time features are disabled, still save the dataset without the derived time columns so downstream steps can train/evaluate the non-time-series variant.

**Step 9: Create data splitting script (`src/data/data_split.py`)**

- Load `raw_engineered.csv` from `data/processed/` (instead of `raw.csv` from `data/raw/`)
- Extract target variable `silica_concentrate` (last column)
- Exclude `date` column and any time-based categorical columns (keep only numerical features)
- Split into train/test sets (80/20 split recommended) with strict train/test separation (no transformations fit on the full dataset and no leakage of test statistics)
- Save 4 files to `data/processed/`:
  - `X_train.csv`
  - `X_test.csv`
  - `y_train.csv`
  - `y_test.csv`

**Step 10: Create data normalization script (`src/data/normalize.py`)**

- Load X_train, X_test, y_train, y_test from `data/processed/`
- Validate data quality (check for missing/infinite values) before scaling
- Apply StandardScaler fit **only on X_train** to enforce strict train/test separation, then transform X_test
- Save scaled datasets to `data/processed/`:
  - `X_train_scaled.csv`
  - `X_test_scaled.csv`
- Persist the fitted scaler for inference/reuse (no scaler persistence is a fix-required item): save to `models/data/scaler.pkl` and load it in evaluation/inference scripts to ensure identical preprocessing.

**Step 11: Create grid search script (`src/models/grid_search.py`)**

- Load normalized training data (X_train_scaled, y_train)
- Choose a regression model (e.g., RandomForestRegressor, GradientBoostingRegressor, or LinearRegression)
- Define parameter grid for hyperparameter tuning
- Perform GridSearchCV with cross-validation
- Save best parameters as `best_params.pkl` to `models/data/`

**Step 12: Create model training script (`src/models/training.py`)**

- Load best parameters from `models/data/best_params.pkl`
- Load normalized training data (X_train_scaled, y_train)
- Train model with best parameters
- Save trained model as `trained_model.pkl` to `models/models/`

**Step 13: Create model evaluation script (`src/models/evaluate.py`)**

- Load trained model from `models/models/trained_model.pkl`
- Load test data (X_test_scaled, y_test)
- Make predictions on test set
- Calculate evaluation metrics: MSE, RMSE, R², MAE
- Save predictions to `data/processed/predictions.csv`
- Save metrics as `scores.json` to `metrics/`

**Step 14: Create supporting files**

- `requirements.txt`: List all Python dependencies (pandas, numpy, scikit-learn, dvc, dvc-s3, dagshub)
- `.gitignore`: Configure to exclude:
  - `__pycache__/`
  - `*.pyc`
  - `.dvc/` (but keep `.dvc/config` tracked)
  - `data/raw/*`
  - `data/processed/*`
  - `models/*.pkl` (but keep directories)
  - `.python-version` (optional - track for consistency if desired)
- `README.md`: Add project documentation

### Phase 3: DVC Pipeline Setup

**Step 15: Create DVC pipeline configuration (`dvc.yaml`)**

Define 7 pipeline stages:

1. **validate**: 
   - Runs `src/data/validate_data.py`
   - Depends on: `data/raw/raw.csv`
   - Outputs: `data/processed/raw_validated.csv`
   - Purpose: Validate and clean raw data (handle missing values, validate dates, check data quality)

2. **feature_engineering**: 
   - Runs `src/data/feature_engineering.py`
   - Depends on: `data/processed/raw_validated.csv` (from validate stage)
   - Outputs: `data/processed/raw_engineered.csv`
   - Supports toggling time-series cyclic features on/off so we can produce both feature sets for model comparison

3. **split**: 
   - Runs `src/data/data_split.py`
   - Depends on: `data/processed/raw_engineered.csv` (from feature_engineering stage)
   - Outputs: `data/processed/X_train.csv`, `data/processed/X_test.csv`, `data/processed/y_train.csv`, `data/processed/y_test.csv`

4. **normalize**: 
   - Runs `src/data/normalize.py`
   - Depends on: split stage outputs
   - Outputs: `data/processed/X_train_scaled.csv`, `data/processed/X_test_scaled.csv`, `models/data/scaler.pkl` (fitted on train only to avoid leakage and reused for inference)

5. **gridsearch**: 
   - Runs `src/models/grid_search.py`
   - Depends on: normalize stage outputs
   - Outputs: `models/data/best_params.pkl`

6. **train**: 
   - Runs `src/models/training.py`
   - Depends on: gridsearch output and normalize outputs
   - Outputs: `models/models/trained_model.pkl`

7. **evaluate**: 
   - Runs `src/models/evaluate.py`
   - Depends on: train output and normalize outputs
   - Outputs: `data/processed/predictions.csv`, `metrics/scores.json`

**Step 16: Initialize DVC repository**


```bash
dvc init
```

### Phase 4: DagsHub Setup

**Step 17: Create DagsHub account**

- Go to https://dagshub.com and sign up/login (if you don't have an account)

**Step 18: Connect GitHub repository to DagsHub**

- In DagsHub, click "New Repository"
- Select "Connect GitHub Repository"
- Choose your forked `examen-dvc` repository
- Follow the connection wizard

**Step 19: Get DagsHub credentials**

- In DagsHub, go to your repository settings
- Navigate to "Remote" section
- Copy the DVC remote URL (format: `https://dagshub.com/USERNAME/REPO_NAME.dvc`)
- Get your DagsHub token from: Settings → Access Tokens → Generate Token

**Step 20: Configure DVC remote**

```bash
dvc remote add -d origin <DAGS_HUB_REMOTE_URL>
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <YOUR_DAGS_HUB_USERNAME>
dvc remote modify origin --local password <YOUR_DAGS_HUB_TOKEN>
```

### Phase 5: Run Pipeline and Push

**Step 21: Run DVC pipeline**

```bash
dvc repro
```

This will execute all pipeline stages and create `dvc.lock` file.

**Step 22: Commit and push to Git**

```bash
git add .
git commit -m "Initial commit: Add ML workflow scripts and DVC pipeline"
git push origin main
```

**Step 23: Push data and models to DagsHub**

```bash
dvc push
```

### Phase 6: Collaboration Setup

**Step 24: Share repository with examiner**

- In DagsHub repository settings → Collaborators
- Add `licence.pedago` with read-only access

### Phase 7: Submission

**Step 25: Create submission file**

- Create `submission.md` with the following content:
  - Your name
  - Your first name
  - Your email address
  - Link to your DagsHub repository
- Zip this file and submit on the exam platform

## Validation Checklist

After completing all steps, verify:

- [ ] All 7 scripts exist in `src/data/` or `src/models/` (including validate_data.py)
- [ ] `dvc.yaml` defines all 7 pipeline stages (including validate stage)
- [ ] `dvc.lock` file exists after running `dvc repro`
- [ ] `.dvc/config` contains DagsHub remote configuration
- [ ] `data/processed/raw_validated.csv` exists (from validate stage)
- [ ] `data/processed/raw_engineered.csv` exists
- [ ] Time-series feature engineering toggle works (datasets produced with and without cyclic time features)
- [ ] `models/models/trained_model.pkl` exists
- [ ] `models/data/best_params.pkl` exists
- [ ] `models/data/scaler.pkl` is saved and reused in evaluation/inference
- [ ] `metrics/scores.json` contains evaluation metrics
- [ ] `data/processed/predictions.csv` exists
- [ ] Strict train/test separation maintained (no leakage; scalers/encoders fit on train only)
- [ ] DagsHub repository shows data in the *data* tab
- [ ] DagsHub repository shows model in the *models* tab
- [ ] DagsHub pipeline visualization matches the expected schema (Schema_exam.png)
- [ ] Repository is shared with `licence.pedago` (read-only)
- [ ] All required files are committed and pushed to Git
- [ ] All data and models are pushed to DagsHub via DVC

## Expected Directory Structure

After completion, your repository should have:

```
examen-dvc/
├── data/
│   ├── processed/
│   │   ├── raw_validated.csv
│   │   ├── raw_engineered.csv
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   ├── y_test.csv
│   │   ├── X_train_scaled.csv
│   │   ├── X_test_scaled.csv
│   │   └── predictions.csv
│   └── raw/
│       └── raw.csv
├── metrics/
│   └── scores.json
├── models/
│   ├── data/
│   │   ├── best_params.pkl
│   │   └── scaler.pkl
│   └── models/
│       └── trained_model.pkl
├── src/
│   ├── data/
│   │   ├── validate_data.py
│   │   ├── feature_engineering.py
│   │   ├── data_split.py
│   │   └── normalize.py
│   └── models/
│       ├── grid_search.py
│       ├── training.py
│       └── evaluate.py
├── .dvc/
│   └── config
├── .gitignore
├── dvc.yaml
├── dvc.lock
├── requirements.txt
├── README.md
└── submission.md
```
