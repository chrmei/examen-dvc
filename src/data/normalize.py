"""
Data Normalization Script

This script loads the train/test split data and applies StandardScaler normalization
to the feature matrices. The scaler is fitted on the training data and then applied
to both training and test sets.

Input: data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv
Output: data/processed/X_train_scaled.csv, X_test_scaled.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler


def normalize_data(
    input_dir: str = "data/processed",
    output_dir: str = "data/processed",
    scaler_path: str = "models/data/scaler.pkl",
):
    """
    Normalize the training and test feature matrices using StandardScaler.

    Parameters:
    -----------
    input_dir : str
        Directory containing the input CSV files (default: data/processed)
    output_dir : str
        Directory to save the normalized CSV files (default: data/processed)
    scaler_path : str
        Path to persist the fitted scaler for inference/reuse (default: models/data/scaler.pkl)
    """
    # Create output directories if they don't exist
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    scaler_file = Path(scaler_path)
    scaler_file.parent.mkdir(parents=True, exist_ok=True)

    # Load train/test split data
    print(f"Loading data from {input_dir}...")
    X_train = pd.read_csv(input_path / "X_train.csv")
    X_test = pd.read_csv(input_path / "X_test.csv")

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Validate data quality before scaling
    print("\nValidating data quality...")
    
    # Check for missing values
    train_missing = X_train.isna().sum().sum()
    test_missing = X_test.isna().sum().sum()
    
    if train_missing > 0:
        raise ValueError(
            f"Training set contains {train_missing} missing values. "
            "Please run validate_data.py to handle missing values before normalization."
        )
    
    if test_missing > 0:
        raise ValueError(
            f"Test set contains {test_missing} missing values. "
            "Please run validate_data.py to handle missing values before normalization."
        )
    
    # Check for infinite values
    train_inf = np.isinf(X_train.select_dtypes(include=[np.number])).sum().sum()
    test_inf = np.isinf(X_test.select_dtypes(include=[np.number])).sum().sum()
    
    if train_inf > 0:
        raise ValueError(
            f"Training set contains {train_inf} infinite values. "
            "Please run validate_data.py to handle infinite values before normalization."
        )
    
    if test_inf > 0:
        raise ValueError(
            f"Test set contains {test_inf} infinite values. "
            "Please run validate_data.py to handle infinite values before normalization."
        )
    
    print("✓ Data validation passed: no missing or infinite values")

    # Initialize StandardScaler
    print("\nInitializing StandardScaler...")
    scaler = StandardScaler()

    # Fit scaler on training data only
    print("Fitting StandardScaler on training data...")
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform test data using the scaler fitted on training data
    print("Transforming test data...")
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame with original column names
    X_train_scaled_df = pd.DataFrame(
        X_train_scaled, columns=X_train.columns, index=X_train.index
    )
    X_test_scaled_df = pd.DataFrame(
        X_test_scaled, columns=X_test.columns, index=X_test.index
    )

    # Save normalized datasets
    print(f"\nSaving normalized datasets to {output_dir}...")
    X_train_scaled_df.to_csv(output_path / "X_train_scaled.csv", index=False)
    X_test_scaled_df.to_csv(output_path / "X_test_scaled.csv", index=False)

    # Persist the fitted scaler for inference / evaluation reuse
    print(f"Persisting fitted scaler to {scaler_file}...")
    dump(scaler, scaler_file)

    print("✓ Successfully saved:")
    print(f"  - {output_path / 'X_train_scaled.csv'}")
    print(f"  - {output_path / 'X_test_scaled.csv'}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print("\nTraining set (scaled) statistics:")
    print(f"  Mean: {X_train_scaled_df.mean().mean():.6f} (should be ~0)")
    print(f"  Std:  {X_train_scaled_df.std().mean():.6f} (should be ~1)")
    print(f"  Min:  {X_train_scaled_df.min().min():.3f}")
    print(f"  Max:  {X_train_scaled_df.max().max():.3f}")

    print("\nTest set (scaled) statistics:")
    print(f"  Mean: {X_test_scaled_df.mean().mean():.6f}")
    print(f"  Std:  {X_test_scaled_df.std().mean():.6f}")
    print(f"  Min:  {X_test_scaled_df.min().min():.3f}")
    print(f"  Max:  {X_test_scaled_df.max().max():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize train/test feature matrices using StandardScaler"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed",
        help="Directory containing input CSV files (default: data/processed)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for normalized CSV files (default: data/processed)",
    )
    parser.add_argument(
        "--scaler-path",
        type=str,
        default="models/data/scaler.pkl",
        help="Path to persist fitted scaler (default: models/data/scaler.pkl)",
    )

    args = parser.parse_args()

    normalize_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        scaler_path=args.scaler_path,
    )

