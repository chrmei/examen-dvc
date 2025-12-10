import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def engineer_features(
    input_path: str = "data/processed/raw_validated.csv",
    output_path: str = "data/processed/raw_engineered.csv",
    use_time_features: bool = False,
):
    """
    Create engineered features from validated data.

    Parameters:
    -----------
    input_path : str
        Path to the validated CSV file (default: data/processed/raw_validated.csv)
    output_path : str
        Path to save the engineered CSV file
    use_time_features : bool
        Whether to add time-series features (cyclical encodings). Set to False to
        produce a version without time features (for comparison / ablation).
    """
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Original data shape: {df.shape}")

    original_cols = df.columns.tolist()

    if 'date' not in df.columns:
        raise ValueError("Date column not found in dataset. Please run validate_data.py first.")

    print("\nConverting date column to datetime...")
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        invalid_dates = df['date'].isna().sum()
        if invalid_dates > 0:
            raise ValueError(f"Found {invalid_dates} invalid dates after conversion. Please run validate_data.py first.")
    else:
        print("Date column is already in datetime format")

    if use_time_features:
        print("Extracting time-based features (cyclical encodings)...")

        hour = df["date"].dt.hour
        day_of_week = df["date"].dt.dayofweek
        month = df["date"].dt.month
        day = df["date"].dt.day

        # cyclic encodings
        df["month_sin"] = np.sin(2 * np.pi * month / 12)
        df["month_cos"] = np.cos(2 * np.pi * month / 12)

        df["day_sin"] = np.sin(2 * np.pi * day / 31)
        df["day_cos"] = np.cos(2 * np.pi * day / 31)

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        df["day_of_week"] = day_of_week
        df["is_weekend"] = (day_of_week >= 5).astype(int)
    else:
        print("Time-based feature engineering disabled; skipping cyclic encodings.")

    print("Creating interaction/ratio features...")
    
    df['amina_starch_ratio'] = df['amina_flow'] / df['starch_flow'].replace(0, np.nan)
    df['amina_starch_ratio'] = df['amina_starch_ratio'].fillna(0)
    
    df['flow_total'] = (
        df['ave_flot_air_flow'] + 
        df['starch_flow'] + 
        df['amina_flow'] + 
        df['ore_pulp_flow']
    )
    
    df['pulp_flow_density'] = df['ore_pulp_flow'] * df['ore_pulp_density']
    
    df['ph_density_interaction'] = df['ore_pulp_pH'] * df['ore_pulp_density']

    new_features = [col for col in df.columns if col not in original_cols]
    
    print(f"\nEngineered data shape: {df.shape}")
    print(f"New features added: {len(new_features)}")
    
    print(f"\nNew features created:")
    for feat in new_features:
        print(f"  - {feat}")

    print(f"\nSaving engineered dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"✓ Successfully saved engineered dataset to {output_path}")

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
        default="data/processed/raw_validated.csv",
        help="Path to validated CSV file (default: data/processed/raw_validated.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/raw_engineered.csv",
        help="Path to save engineered CSV file (default: data/processed/raw_engineered.csv)",
    )
    parser.add_argument(
        "--use-time-features",
        action="store_true",
        default=False,
        help="Enable cyclic time-based features (default: enabled)",
    )
    parser.add_argument(
        "--no-time-features",
        action="store_false",
        dest="use_time_features",
        help="Disable time-based feature engineering",
    )

    args = parser.parse_args()

    engineer_features(
        input_path=args.input,
        output_path=args.output,
        use_time_features=args.use_time_features,
    )

