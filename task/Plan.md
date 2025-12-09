# Mineral Flotation DVC & DagsHub Exam Setup Plan

## Overview

This plan sets up a complete ML workflow for mineral flotation data modeling using DVC for version control and DagsHub for collaboration. The workflow includes data preprocessing, model training, and evaluation scripts connected via a DVC pipeline.

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
curl -o data/raw_data/raw.csv https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv
```

### Phase 2: Script Creation

**Step 6: Data Exploration**

- Perform comprehensive data exploration using Jupyter notebook
- Analyze data quality, distributions, correlations, and relationships
- Document findings and insights
- Result: `notebooks/data_exploration.ipynb` (existing notebook contains the exploration results)

**Step 7: Create data splitting script (`src/data/split_data.py`)**

- Load `raw.csv` from `data/raw/`
- Extract target variable `silica_concentrate` (last column)
- Split into train/test sets (80/20 split recommended)
- Save 4 files to `data/processed/`:
  - `X_train.csv`
  - `X_test.csv`
  - `y_train.csv`
  - `y_test.csv`

**Step 8: Create data normalization script (`src/data/normalize_data.py`)**

- Load X_train, X_test, y_train, y_test from `data/processed/`
- Apply StandardScaler to X_train and X_test
- Save scaled datasets to `data/processed/`:
  - `X_train_scaled.csv`
  - `X_test_scaled.csv`

**Step 9: Create grid search script (`src/models/grid_search.py`)**

- Load normalized training data (X_train_scaled, y_train)
- Choose a regression model (e.g., RandomForestRegressor, GradientBoostingRegressor, or LinearRegression)
- Define parameter grid for hyperparameter tuning
- Perform GridSearchCV with cross-validation
- Save best parameters as `best_params.pkl` to `models/data/`

**Step 10: Create model training script (`src/models/train_model.py`)**

- Load best parameters from `models/data/best_params.pkl`
- Load normalized training data (X_train_scaled, y_train)
- Train model with best parameters
- Save trained model as `trained_model.pkl` to `models/models/`

**Step 11: Create model evaluation script (`src/models/evaluate_model.py`)**

- Load trained model from `models/models/trained_model.pkl`
- Load test data (X_test_scaled, y_test)
- Make predictions on test set
- Calculate evaluation metrics: MSE, RMSE, R², MAE
- Save predictions to `data/processed/predictions.csv`
- Save metrics as `scores.json` to `metrics/`

**Step 12: Create supporting files**

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

**Step 13: Create DVC pipeline configuration (`dvc.yaml`)**

Define 5 pipeline stages:

1. **split**: 
   - Runs `src/data/split_data.py`
   - Depends on: `data/raw/raw.csv`
   - Outputs: `data/processed/X_train.csv`, `data/processed/X_test.csv`, `data/processed/y_train.csv`, `data/processed/y_test.csv`

2. **normalize**: 
   - Runs `src/data/normalize_data.py`
   - Depends on: split stage outputs
   - Outputs: `data/processed/X_train_scaled.csv`, `data/processed/X_test_scaled.csv`

3. **gridsearch**: 
   - Runs `src/models/grid_search.py`
   - Depends on: normalize stage outputs
   - Outputs: `models/data/best_params.pkl`

4. **train**: 
   - Runs `src/models/train_model.py`
   - Depends on: gridsearch output and normalize outputs
   - Outputs: `models/models/trained_model.pkl`

5. **evaluate**: 
   - Runs `src/models/evaluate_model.py`
   - Depends on: train output and normalize outputs
   - Outputs: `data/processed/predictions.csv`, `metrics/scores.json`

**Step 14: Initialize DVC repository**

```bash
dvc init
```

### Phase 4: DagsHub Setup

**Step 15: Create DagsHub account**

- Go to https://dagshub.com and sign up/login (if you don't have an account)

**Step 16: Connect GitHub repository to DagsHub**

- In DagsHub, click "New Repository"
- Select "Connect GitHub Repository"
- Choose your forked `examen-dvc` repository
- Follow the connection wizard

**Step 17: Get DagsHub credentials**

- In DagsHub, go to your repository settings
- Navigate to "Remote" section
- Copy the DVC remote URL (format: `https://dagshub.com/USERNAME/REPO_NAME.dvc`)
- Get your DagsHub token from: Settings → Access Tokens → Generate Token

**Step 18: Configure DVC remote**

```bash
dvc remote add -d origin <DAGS_HUB_REMOTE_URL>
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <YOUR_DAGS_HUB_USERNAME>
dvc remote modify origin --local password <YOUR_DAGS_HUB_TOKEN>
```

### Phase 5: Run Pipeline and Push

**Step 19: Run DVC pipeline**

```bash
dvc repro
```

This will execute all pipeline stages and create `dvc.lock` file.

**Step 20: Commit and push to Git**

```bash
git add .
git commit -m "Initial commit: Add ML workflow scripts and DVC pipeline"
git push origin main
```

**Step 21: Push data and models to DagsHub**

```bash
dvc push
```

### Phase 6: Collaboration Setup

**Step 22: Share repository with examiner**

- In DagsHub repository settings → Collaborators
- Add `licence.pedago` with read-only access

### Phase 7: Submission

**Step 23: Create submission file**

- Create `submission.md` with the following content:
  - Your name
  - Your first name
  - Your email address
  - Link to your DagsHub repository
- Zip this file and submit on the exam platform

## Validation Checklist

After completing all steps, verify:

- [ ] All 5 scripts exist in `src/data/` or `src/models/`
- [ ] `dvc.yaml` defines all 5 pipeline stages
- [ ] `dvc.lock` file exists after running `dvc repro`
- [ ] `.dvc/config` contains DagsHub remote configuration
- [ ] `models/models/trained_model.pkl` exists
- [ ] `models/data/best_params.pkl` exists
- [ ] `metrics/scores.json` contains evaluation metrics
- [ ] `data/processed/predictions.csv` exists
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
│   │   └── best_params.pkl
│   └── models/
│       └── trained_model.pkl
├── src/
│   ├── data/
│   │   ├── split_data.py
│   │   └── normalize_data.py
│   └── models/
│       ├── grid_search.py
│       ├── train_model.py
│       └── evaluate_model.py
├── .dvc/
│   └── config
├── .gitignore
├── dvc.yaml
├── dvc.lock
├── requirements.txt
├── README.md
└── submission.md
```
