"""
Data Splitting Script

This script loads the feature-engineered mineral flotation data, extracts the target variable,
and splits the data into training and testing sets (80/20 split).

Input: data/processed/raw_engineered.csv
Output: data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import argparse


def split_data(
    input_path: str = "data/processed/raw_engineered.csv",
    output_dir: str = "data/processed",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split the feature-engineered data into training and testing sets.

    Parameters:
    -----------
    input_path : str
        Path to the feature-engineered CSV file (default: data/processed/raw_engineered.csv)
    output_dir : str
        Directory to save the split data files
    test_size : float
        Proportion of data to use for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load raw data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Data shape: {df.shape}")

    # Validate data quality
    print("\nValidating data quality...")
    
    # Check for missing values
    missing_values = df.isna().sum().sum()
    if missing_values > 0:
        raise ValueError(
            f"Dataset contains {missing_values} missing values. "
            "Please run validate_data.py to handle missing values before splitting."
        )
    
    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        import numpy as np
        inf_values = np.isinf(df[numeric_cols]).sum().sum()
        if inf_values > 0:
            raise ValueError(
                f"Dataset contains {inf_values} infinite values. "
                "Please run validate_data.py to handle infinite values before splitting."
            )
    
    print("✓ Data validation passed: no missing or infinite values")

    # Extract target variable (last column: silica_concentrate)
    target_column = df.columns[-1]
    print(f"Target variable: {target_column}")

    # Separate features and target
    # Exclude the date column (first column) and target column (last column)
    feature_columns = df.columns[1:-1]  # Skip date and target
    X = df[feature_columns]
    y = df[target_column]

    print(f"Feature columns: {list(feature_columns)}")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Target variable range: [{y.min():.3f}, {y.max():.3f}]")

    # Split into train and test sets
    print(
        f"\nSplitting data into train/test sets ({1-test_size:.0%}/{test_size:.0%})..."
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Training set size: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
    print(f"Test set size: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")

    # Save split data
    print(f"\nSaving split data to {output_dir}...")
    X_train.to_csv(output_path / "X_train.csv", index=False)
    X_test.to_csv(output_path / "X_test.csv", index=False)
    y_train.to_csv(output_path / "y_train.csv", index=False)
    y_test.to_csv(output_path / "y_test.csv", index=False)

    print("✓ Successfully saved:")
    print(f"  - {output_path / 'X_train.csv'}")
    print(f"  - {output_path / 'X_test.csv'}")
    print(f"  - {output_path / 'y_train.csv'}")
    print(f"  - {output_path / 'y_test.csv'}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print("\nTraining set target statistics:")
    print(f"  Mean: {y_train.mean():.3f}")
    print(f"  Std:  {y_train.std():.3f}")
    print(f"  Min:  {y_train.min():.3f}")
    print(f"  Max:  {y_train.max():.3f}")

    print(f"\nTest set target statistics:")
    print(f"  Mean: {y_test.mean():.3f}")
    print(f"  Std:  {y_test.std():.3f}")
    print(f"  Min:  {y_test.min():.3f}")
    print(f"  Max:  {y_test.max():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split feature-engineered data into train/test sets")
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/raw_engineered.csv",
        help="Path to feature-engineered CSV file (default: data/processed/raw_engineered.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for split data (default: data/processed)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    split_data(
        input_path=args.input,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
