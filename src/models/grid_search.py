"""
Grid Search Script

This script performs hyperparameter tuning using GridSearchCV to find the best
parameters for an XGBoost model. The best parameters are saved for use in model training.

Input: data/processed/X_train_scaled.csv, data/processed/y_train.csv
Output: models/data/best_params.pkl
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


def grid_search(
    input_dir: str = "data/processed",
    output_path: str = "models/data/best_params.pkl",
    cv: int = 5,
    n_jobs: int = -1,
    random_state: int = 42,
):
    """
    Perform grid search to find optimal hyperparameters for XGBoost.

    Parameters:
    -----------
    input_dir : str
        Directory containing the normalized training data (default: data/processed)
    output_path : str
        Path to save the best parameters pickle file (default: models/data/best_params.pkl)
    cv : int
        Number of cross-validation folds (default: 5)
    n_jobs : int
        Number of parallel jobs for grid search (default: -1, use all cores)
    random_state : int
        Random seed for reproducibility (default: 42)
    """
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_dir)

    # Load normalized training data
    print(f"Loading normalized training data from {input_dir}...")
    X_train_scaled = pd.read_csv(input_path / "X_train_scaled.csv")
    y_train = pd.read_csv(input_path / "y_train.csv").squeeze()

    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Target variable shape: {y_train.shape}")

    # Validate data quality
    print("\nValidating data quality...")
    
    # Check for missing values
    train_missing = X_train_scaled.isna().sum().sum()
    y_missing = y_train.isna().sum()
    
    if train_missing > 0:
        raise ValueError(
            f"Training features contain {train_missing} missing values. "
            "Please ensure data is properly preprocessed."
        )
    
    if y_missing > 0:
        raise ValueError(
            f"Target variable contains {y_missing} missing values. "
            "Please ensure data is properly preprocessed."
        )
    
    # Check for infinite values
    train_inf = np.isinf(X_train_scaled.select_dtypes(include=[np.number])).sum().sum()
    # Handle y_train as Series or array
    if isinstance(y_train, pd.Series):
        y_inf = np.isinf(y_train).sum()
    else:
        y_inf = np.isinf(np.array(y_train)).sum()
    
    if train_inf > 0:
        raise ValueError(
            f"Training features contain {train_inf} infinite values. "
            "Please ensure data is properly preprocessed."
        )
    
    if y_inf > 0:
        raise ValueError(
            f"Target variable contains {y_inf} infinite values. "
            "Please ensure data is properly preprocessed."
        )
    
    print("✓ Data validation passed: no missing or infinite values")

    # Display target variable statistics
    print("\n" + "=" * 60)
    print("TARGET VARIABLE STATISTICS")
    print("=" * 60)
    print(f"Mean: {y_train.mean():.3f}")
    print(f"Std:  {y_train.std():.3f}")
    print(f"Min:  {y_train.min():.3f}")
    print(f"Max:  {y_train.max():.3f}")

    # Initialize XGBoost
    print("\n" + "=" * 60)
    print("INITIALIZING GRID SEARCH")
    print("=" * 60)
    print("Model: XGBRegressor")
    print(f"Cross-validation folds: {cv}")
    print(f"Parallel jobs: {n_jobs}")

    # Define parameter grid for XGBoost
    # Balanced grid covering key hyperparameters for regression
    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 0.9, 1.0],
        "min_child_weight": [1, 2, 4],  # XGBoost equivalent of min_samples_split/leaf
    }

    print("\nParameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\nTotal parameter combinations: {total_combinations}")
    print(f"Total fits (with CV): {total_combinations * cv}")

    # Initialize base model using XGBoost sklearn API
    base_model = xgb.XGBRegressor(
        random_state=random_state,
        objective="reg:squarederror",  # For regression
        eval_metric="rmse",  # Evaluation metric
    )

    # Perform grid search with cross-validation
    print("\n" + "=" * 60)
    print("RUNNING GRID SEARCH")
    print("=" * 60)
    print("This may take several minutes...")

    grid_search_cv = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_mean_squared_error",  # Use negative MSE (higher is better)
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True,
    )

    # Fit grid search
    grid_search_cv.fit(X_train_scaled, y_train)

    # Get best parameters
    best_params = grid_search_cv.best_params_
    best_score = grid_search_cv.best_score_

    print("\n" + "=" * 60)
    print("GRID SEARCH RESULTS")
    print("=" * 60)
    print(f"Best cross-validation score (neg MSE): {best_score:.6f}")
    print(f"Best cross-validation RMSE: {np.sqrt(-best_score):.6f}")
    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    # Display top 5 parameter combinations
    results_df = pd.DataFrame(grid_search_cv.cv_results_)
    top_5 = results_df.nlargest(5, "mean_test_score")[
        ["mean_test_score", "std_test_score"]
    ]
    top_5["mean_test_rmse"] = np.sqrt(-top_5["mean_test_score"])
    top_5["std_test_rmse"] = top_5["std_test_score"] / (2 * np.sqrt(-top_5["mean_test_score"]))

    print("\nTop 5 parameter combinations:")
    print(top_5.to_string())

    # Save best parameters
    print(f"\nSaving best parameters to {output_path}...")
    dump(best_params, output_path)
    print(f"✓ Successfully saved best parameters to {output_path}")

    # Also save the full grid search results for reference (optional)
    results_path = output_file.parent / "grid_search_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"✓ Grid search results saved to {results_path}")

    return best_params, grid_search_cv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform grid search for hyperparameter tuning"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed",
        help="Directory containing normalized training data (default: data/processed)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/data/best_params.pkl",
        help="Path to save best parameters pickle file (default: models/data/best_params.pkl)",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (default: -1, use all cores)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    grid_search(
        input_dir=args.input_dir,
        output_path=args.output,
        cv=args.cv,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
    )
