"""
Model Evaluation Script

This script loads a trained model and evaluates its performance on both training
and test sets. It calculates various regression metrics (MSE, RMSE, R², MAE),
plots the learning curve if training history is available, and saves both
predictions and metrics for analysis.

Input: models/models/trained_model.pkl (or trained_model_with_history.pkl),
       data/processed/X_train_scaled.csv, data/processed/y_train.csv,
       data/processed/X_test_scaled.csv, data/processed/y_test.csv
Output: data/processed/predictions.csv, metrics/scores.json, metrics/learning_curve.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
import xgboost as xgb


def evaluate_model(
    model_path: str = "models/models/trained_model.pkl",
    input_dir: str = "data/processed",
    predictions_path: str = "data/processed/predictions.csv",
    metrics_path: str = "metrics/scores.json",
    learning_curve_path: str = "metrics/learning_curve.png",
    history_path: str = None,
):
    """
    Evaluate a trained model on both training and test sets, plot learning curve
    if available, and save predictions and metrics.

    Parameters:
    -----------
    model_path : str
        Path to the trained model pickle file (default: models/models/trained_model.pkl)
    input_dir : str
        Directory containing the data CSV files (default: data/processed)
    predictions_path : str
        Path to save predictions CSV file (default: data/processed/predictions.csv)
    metrics_path : str
        Path to save metrics JSON file (default: metrics/scores.json)
    learning_curve_path : str
        Path to save learning curve plot (default: metrics/learning_curve.png)
    history_path : str, optional
        Path to model with training history. If None, tries to load from
        models/models/trained_model_with_history.pkl (default: None)
    """
    # Create output directories if they don't exist
    model_file = Path(model_path)
    input_path = Path(input_dir)
    predictions_file = Path(predictions_path)
    metrics_file = Path(metrics_path)
    learning_curve_file = Path(learning_curve_path)
    
    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    learning_curve_file.parent.mkdir(parents=True, exist_ok=True)

    # Try to load training history
    if history_path is None:
        history_path = "models/models/trained_model_with_history.pkl"
    
    history_file = Path(history_path)
    training_history = None
    evals_result = None
    
    if history_file.exists():
        print(f"Loading training history from {history_path}...")
        training_history = load(history_path)
        if isinstance(training_history, dict) and "evals_result" in training_history:
            evals_result = training_history["evals_result"]
            model = training_history["model"]
            print("✓ Training history loaded successfully")
        else:
            print("⚠ Training history file exists but doesn't contain expected format")
            model = load(model_path)
    else:
        print(f"⚠ Training history not found at {history_path}, loading model only...")
        model = load(model_path)

    # Load trained model if not already loaded
    if not isinstance(model, xgb.core.Booster) and not hasattr(model, 'predict'):
        print(f"Loading trained model from {model_path}...")
        if not model_file.exists():
            raise FileNotFoundError(
                f"Trained model file not found at {model_path}. "
                "Please run training.py first."
            )
        model = load(model_path)
    
    print("✓ Model loaded successfully")

    # Load training data for evaluation
    print(f"\nLoading training data from {input_dir}...")
    X_train_scaled = pd.read_csv(input_path / "X_train_scaled.csv")
    y_train = pd.read_csv(input_path / "y_train.csv").squeeze()

    # Load test data
    print(f"Loading test data from {input_dir}...")
    X_test_scaled = pd.read_csv(input_path / "X_test_scaled.csv")
    y_test = pd.read_csv(input_path / "y_test.csv").squeeze()

    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")

    # Validate data quality
    print("\nValidating data quality...")
    
    # Check for missing values
    train_missing = X_train_scaled.isna().sum().sum()
    test_missing = X_test_scaled.isna().sum().sum()
    y_train_missing = y_train.isna().sum()
    y_test_missing = y_test.isna().sum()
    
    if train_missing > 0 or test_missing > 0:
        raise ValueError(
            f"Features contain missing values (train: {train_missing}, test: {test_missing}). "
            "Please ensure data is properly preprocessed."
        )
    
    if y_train_missing > 0 or y_test_missing > 0:
        raise ValueError(
            f"Target variable contains missing values (train: {y_train_missing}, test: {y_test_missing}). "
            "Please ensure data is properly preprocessed."
        )
    
    # Check for infinite values
    train_inf = np.isinf(X_train_scaled.select_dtypes(include=[np.number])).sum().sum()
    test_inf = np.isinf(X_test_scaled.select_dtypes(include=[np.number])).sum().sum()
    
    if isinstance(y_train, pd.Series):
        y_train_inf = np.isinf(y_train).sum()
    else:
        y_train_inf = np.isinf(np.array(y_train)).sum()
    
    if isinstance(y_test, pd.Series):
        y_test_inf = np.isinf(y_test).sum()
    else:
        y_test_inf = np.isinf(np.array(y_test)).sum()
    
    if train_inf > 0 or test_inf > 0:
        raise ValueError(
            f"Features contain infinite values (train: {train_inf}, test: {test_inf}). "
            "Please ensure data is properly preprocessed."
        )
    
    if y_train_inf > 0 or y_test_inf > 0:
        raise ValueError(
            f"Target variable contains infinite values (train: {y_train_inf}, test: {y_test_inf}). "
            "Please ensure data is properly preprocessed."
        )
    
    print("✓ Data validation passed: no missing or infinite values")

    # Plot learning curve if training history is available
    if evals_result is not None:
        print("\n" + "=" * 60)
        print("PLOTTING LEARNING CURVE")
        print("=" * 60)
        
        train_rmse = evals_result["train"]["rmse"]
        val_rmse = evals_result["validation"]["rmse"]
        iterations = range(1, len(train_rmse) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, train_rmse, label="Training RMSE", linewidth=2, alpha=0.8)
        plt.plot(iterations, val_rmse, label="Validation RMSE", linewidth=2, alpha=0.8)
        plt.xlabel("Boosting Iteration", fontsize=12)
        plt.ylabel("RMSE", fontsize=12)
        plt.title("XGBoost Learning Curve - Model Convergence", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(learning_curve_path, dpi=300, bbox_inches="tight")
        print(f"✓ Learning curve saved to {learning_curve_path}")
        plt.close()
        
        # Display convergence statistics
        print("\nConvergence Statistics:")
        print(f"  Initial Training RMSE:   {train_rmse[0]:.6f}")
        print(f"  Final Training RMSE:      {train_rmse[-1]:.6f}")
        print(f"  Initial Validation RMSE: {val_rmse[0]:.6f}")
        print(f"  Final Validation RMSE:   {val_rmse[-1]:.6f}")
        print(f"  Training Improvement:    {train_rmse[0] - train_rmse[-1]:.6f}")
        print(f"  Validation Improvement:  {val_rmse[0] - val_rmse[-1]:.6f}")
        
        # Check for overfitting
        final_gap = val_rmse[-1] - train_rmse[-1]
        if final_gap > 0.1:
            print(f"\n⚠ Warning: Potential overfitting detected (validation RMSE is {final_gap:.6f} higher than training RMSE)")
        else:
            print(f"\n✓ Model shows good generalization (validation gap: {final_gap:.6f})")

    # Evaluate on training set
    print("\n" + "=" * 60)
    print("EVALUATING ON TRAINING SET")
    print("=" * 60)
    
    # Handle XGBoost Booster objects (from native API) vs sklearn estimators
    if isinstance(model, xgb.core.Booster):
        dtrain = xgb.DMatrix(X_train_scaled)
        y_train_pred = model.predict(dtrain)
    else:
        y_train_pred = model.predict(X_train_scaled)
    
    train_mse = np.mean((y_train - y_train_pred) ** 2)
    train_rmse = np.sqrt(train_mse)
    train_mae = np.mean(np.abs(y_train - y_train_pred))
    ss_res_train = np.sum((y_train - y_train_pred) ** 2)
    ss_tot_train = np.sum((y_train - np.mean(y_train)) ** 2)
    train_r2 = 1 - (ss_res_train / ss_tot_train) if ss_tot_train != 0 else 0.0
    
    print("\nTraining Set Performance Metrics:")
    print(f"MSE:  {train_mse:.6f}")
    print(f"RMSE: {train_rmse:.6f}")
    print(f"MAE:  {train_mae:.6f}")
    print(f"R²:   {train_r2:.6f}")

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)
    print("Predicting on test set...")
    
    if isinstance(model, xgb.core.Booster):
        dtest = xgb.DMatrix(X_test_scaled)
        y_test_pred = model.predict(dtest)
    else:
        y_test_pred = model.predict(X_test_scaled)
    
    print(f"✓ Predictions completed: {len(y_test_pred)} predictions")

    # Calculate evaluation metrics
    print("\n" + "=" * 60)
    print("CALCULATING EVALUATION METRICS")
    print("=" * 60)
    
    # Mean Squared Error (MSE)
    test_mse = np.mean((y_test - y_test_pred) ** 2)
    
    # Root Mean Squared Error (RMSE)
    test_rmse = np.sqrt(test_mse)
    
    # Mean Absolute Error (MAE)
    test_mae = np.mean(np.abs(y_test - y_test_pred))
    
    # R² Score (Coefficient of Determination)
    ss_res_test = np.sum((y_test - y_test_pred) ** 2)
    ss_tot_test = np.sum((y_test - np.mean(y_test)) ** 2)
    test_r2 = 1 - (ss_res_test / ss_tot_test) if ss_tot_test != 0 else 0.0

    # Display metrics
    print("\nTest Set Performance Metrics:")
    print(f"MSE:  {test_mse:.6f}")
    print(f"RMSE: {test_rmse:.6f}")
    print(f"MAE:  {test_mae:.6f}")
    print(f"R²:   {test_r2:.6f}")

    # Create predictions DataFrame (test set)
    predictions_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_test_pred,
        "residual": y_test - y_test_pred,
        "abs_residual": np.abs(y_test - y_test_pred),
    })

    # Save predictions
    print(f"\nSaving predictions to {predictions_path}...")
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✓ Successfully saved predictions to {predictions_path}")

    # Prepare metrics dictionary
    metrics = {
        "train": {
            "mse": float(train_mse),
            "rmse": float(train_rmse),
            "mae": float(train_mae),
            "r2": float(train_r2),
        },
        "test": {
            "mse": float(test_mse),
            "rmse": float(test_rmse),
            "mae": float(test_mae),
            "r2": float(test_r2),
        },
    }

    # Save metrics
    print(f"Saving metrics to {metrics_path}...")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Successfully saved metrics to {metrics_path}")

    # Display summary statistics of predictions
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY STATISTICS (TEST SET)")
    print("=" * 60)
    print(f"Predicted mean: {y_test_pred.mean():.3f}")
    print(f"Predicted std:  {y_test_pred.std():.3f}")
    print(f"Predicted min:  {y_test_pred.min():.3f}")
    print(f"Predicted max:  {y_test_pred.max():.3f}")
    print(f"\nResidual mean:  {predictions_df['residual'].mean():.6f} (should be ~0)")
    print(f"Residual std:   {predictions_df['residual'].std():.6f}")

    return metrics, predictions_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on training and test sets"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/models/trained_model.pkl",
        help="Path to trained model pickle file (default: models/models/trained_model.pkl)",
    )
    parser.add_argument(
        "--history",
        type=str,
        default=None,
        help="Path to model with training history (default: models/models/trained_model_with_history.pkl)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed",
        help="Directory containing data CSV files (default: data/processed)",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="data/processed/predictions.csv",
        help="Path to save predictions CSV file (default: data/processed/predictions.csv)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="metrics/scores.json",
        help="Path to save metrics JSON file (default: metrics/scores.json)",
    )
    parser.add_argument(
        "--learning-curve",
        type=str,
        default="metrics/learning_curve.png",
        help="Path to save learning curve plot (default: metrics/learning_curve.png)",
    )

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        input_dir=args.input_dir,
        predictions_path=args.predictions,
        metrics_path=args.metrics,
        learning_curve_path=args.learning_curve,
        history_path=args.history,
    )

