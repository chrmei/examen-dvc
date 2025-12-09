"""
Feature Engineering Script

This script loads the raw mineral flotation data and creates engineered features
including time-based features and interaction/ratio features.

Input: data/raw/raw.csv
Output: data/processed/raw_engineered.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def engineer_features(
    input_path: str = "data/raw/raw.csv",
    output_path: str = "data/processed/raw_engineered.csv",
):
    """
    Create engineered features from raw data.

    Parameters:
    -----------
    input_path : str
        Path to the raw CSV file
    output_path : str
        Path to save the engineered CSV file
    """
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load raw data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Original data shape: {df.shape}")

    # Convert date column to datetime format
    print("\nConverting date column to datetime...")
    df['date'] = pd.to_datetime(df['date'])

    # Extract time-based features
    print("Extracting time-based features...")
    
    # Extract base time features (temporary, will be converted to cyclical)
    hour = df['date'].dt.hour  # Hour of day (0-23)
    day_of_week = df['date'].dt.dayofweek  # Day of week (0-6, Monday=0)
    month = df['date'].dt.month  # Month (1-12)
    day = df['date'].dt.day  # Day of month (1-31)
    
    # Create cyclical encodings for month, day, and hour
    print("Creating cyclical encodings for time features...")
    
    # Month: cyclical encoding (handles missing months gracefully)
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    
    # Day of month: cyclical encoding
    df['day_sin'] = np.sin(2 * np.pi * day / 31)
    df['day_cos'] = np.cos(2 * np.pi * day / 31)
    
    # Hour: cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # Keep day_of_week and is_weekend as-is (they're already well-represented)
    df['day_of_week'] = day_of_week
    df['is_weekend'] = (day_of_week >= 5).astype(int)  # Boolean flag (1 if Saturday/Sunday, 0 otherwise)

    # Create interaction/ratio features
    print("Creating interaction/ratio features...")
    
    # amina_starch_ratio: Ratio of amina_flow to starch_flow
    # Handle division by zero by replacing inf with NaN, then fill NaN with 0
    df['amina_starch_ratio'] = df['amina_flow'] / df['starch_flow'].replace(0, np.nan)
    df['amina_starch_ratio'] = df['amina_starch_ratio'].fillna(0)
    
    # flow_total: Sum of all flow features
    df['flow_total'] = (
        df['ave_flot_air_flow'] + 
        df['starch_flow'] + 
        df['amina_flow'] + 
        df['ore_pulp_flow']
    )
    
    # pulp_flow_density: Product of ore_pulp_flow and ore_pulp_density
    df['pulp_flow_density'] = df['ore_pulp_flow'] * df['ore_pulp_density']
    
    # ph_density_interaction: Product of ore_pulp_pH and ore_pulp_density
    df['ph_density_interaction'] = df['ore_pulp_pH'] * df['ore_pulp_density']

    print(f"\nEngineered data shape: {df.shape}")
    print(f"New features added: {df.shape[1] - len(pd.read_csv(input_path).columns)}")
    
    # Display new feature names
    original_cols = pd.read_csv(input_path).columns.tolist()
    new_features = [col for col in df.columns if col not in original_cols]
    print(f"\nNew features created:")
    for feat in new_features:
        print(f"  - {feat}")

    # Save engineered dataset
    print(f"\nSaving engineered dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"✓ Successfully saved engineered dataset to {output_path}")

    # Print summary statistics for new features
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS FOR NEW FEATURES")
    print("=" * 60)
    print(df[new_features].describe())

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create engineered features from raw data")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/raw.csv",
        help="Path to raw CSV file (default: data/raw/raw.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/raw_engineered.csv",
        help="Path to save engineered CSV file (default: data/processed/raw_engineered.csv)",
    )

    args = parser.parse_args()

    engineer_features(
        input_path=args.input,
        output_path=args.output,
    )

