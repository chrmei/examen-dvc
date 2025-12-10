import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import warnings
from typing import Tuple, Dict


def validate_and_clean_data(
    input_path: str = "data/raw/raw.csv",
    output_path: str = "data/processed/raw_validated.csv",
    missing_value_strategy: str = "drop",
    date_format: str = "infer",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Validate and clean raw data before feature engineering.

    Parameters:
    -----------
    input_path : str
        Path to the raw CSV file
    output_path : str
        Path to save the validated CSV file
    missing_value_strategy : str
        Strategy for handling missing values: 'drop', 'forward_fill', 'backward_fill', 'mean', 'median', 'zero'
        date_format : str
        Date format to use when parsing dates. Use 'infer' for automatic detection.

    Returns:
    --------
    df : pd.DataFrame
        Cleaned and validated dataframe
    validation_report : dict
        Dictionary containing validation statistics and issues found
    """
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    validation_report = {
        "original_shape": None,
        "final_shape": None,
        "missing_values": {},
        "invalid_dates": 0,
        "rows_dropped": 0,
        "columns_checked": [],
        "warnings": [],
        "errors": [],
    }

    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
        validation_report["original_shape"] = df.shape
        print(f"Original data shape: {df.shape}")
    except Exception as e:
        error_msg = f"Failed to load CSV file: {str(e)}"
        validation_report["errors"].append(error_msg)
        raise FileNotFoundError(error_msg)

    print("\nValidating column structure...")
    expected_columns = [
        "date",
        "ave_flot_air_flow",
        "ave_flot_level",
        "iron_feed",
        "starch_flow",
        "amina_flow",
        "ore_pulp_flow",
        "ore_pulp_pH",
        "ore_pulp_density",
        "silica_concentrate",
    ]
    
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        error_msg = f"Missing required columns: {missing_columns}"
        validation_report["errors"].append(error_msg)
        raise ValueError(error_msg)
    
    print(f"✓ All required columns present ({len(expected_columns)} columns)")

    print("\nValidating date column...")
    if "date" not in df.columns:
        error_msg = "Date column not found in dataset"
        validation_report["errors"].append(error_msg)
        raise ValueError(error_msg)

    missing_dates_before = df["date"].isna().sum()
    if missing_dates_before > 0:
        warning_msg = f"Found {missing_dates_before} missing date values"
        validation_report["warnings"].append(warning_msg)
        print(f"⚠ {warning_msg}")

    print("Converting date column to datetime format...")
    try:
        if date_format == "infer":
            df["date"] = pd.to_datetime(
                df["date"], errors="coerce"
            )
        else:
            df["date"] = pd.to_datetime(df["date"], format=date_format, errors="coerce")
        
        invalid_dates = df["date"].isna().sum()
        validation_report["invalid_dates"] = invalid_dates
        
        if invalid_dates > 0:
            warning_msg = f"Found {invalid_dates} invalid date values (converted to NaT)"
            validation_report["warnings"].append(warning_msg)
            print(f"⚠ {warning_msg}")
            
            invalid_date_indices = df[df["date"].isna()].index[:5]
            if len(invalid_date_indices) > 0:
                print("  Examples of invalid dates:")
                original_df = pd.read_csv(input_path)
                for idx in invalid_date_indices:
                    print(f"    Row {idx}: '{original_df.loc[idx, 'date']}'")
    except Exception as e:
        error_msg = f"Failed to parse date column: {str(e)}"
        validation_report["errors"].append(error_msg)
        raise ValueError(error_msg)

    print("\nChecking for duplicate rows...")
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        warning_msg = f"Found {duplicates} duplicate rows"
        validation_report["warnings"].append(warning_msg)
        print(f"⚠ {warning_msg}")
        df = df.drop_duplicates()
        print(f"✓ Removed {duplicates} duplicate rows")

    print("\nChecking for missing values in numeric columns...")
    numeric_columns = [
        "ave_flot_air_flow",
        "ave_flot_level",
        "iron_feed",
        "starch_flow",
        "amina_flow",
        "ore_pulp_flow",
        "ore_pulp_pH",
        "ore_pulp_density",
        "silica_concentrate",
    ]

    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        missing_count = df[col].isna().sum()
        validation_report["missing_values"][col] = {
            "count": int(missing_count),
            "percentage": float(missing_count / len(df) * 100) if len(df) > 0 else 0.0,
        }
        
        if missing_count > 0:
            print(f"  {col}: {missing_count} missing values ({missing_count/len(df)*100:.2f}%)")

    total_missing = sum([v["count"] for v in validation_report["missing_values"].values()])
    if total_missing > 0:
        print(f"\nHandling missing values using strategy: '{missing_value_strategy}'...")
        
        rows_before = len(df)
        
        if missing_value_strategy == "drop":
            df = df.dropna()
            rows_dropped = rows_before - len(df)
            validation_report["rows_dropped"] = rows_dropped
            print(f"✓ Dropped {rows_dropped} rows with missing values")
            
        elif missing_value_strategy == "forward_fill":
            df = df.ffill()
            print("✓ Forward-filled missing values")
            
        elif missing_value_strategy == "backward_fill":
            df = df.bfill()
            print("✓ Backward-filled missing values")
            
        elif missing_value_strategy == "mean":
            for col in numeric_columns:
                if col in df.columns and df[col].isna().sum() > 0:
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)
                    print(f"✓ Filled {col} missing values with mean: {mean_val:.3f}")
                    
        elif missing_value_strategy == "median":
            for col in numeric_columns:
                if col in df.columns and df[col].isna().sum() > 0:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    print(f"✓ Filled {col} missing values with median: {median_val:.3f}")
                    
        elif missing_value_strategy == "zero":
            df[numeric_columns] = df[numeric_columns].fillna(0)
            print("✓ Filled missing values with zero")
        else:
            warning_msg = f"Unknown missing value strategy: {missing_value_strategy}. No action taken."
            validation_report["warnings"].append(warning_msg)
            print(f"⚠ {warning_msg}")

    print("\nValidating numeric columns...")
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        if not pd.api.types.is_numeric_dtype(df[col]):
            warning_msg = f"Column {col} is not numeric (type: {df[col].dtype})"
            validation_report["warnings"].append(warning_msg)
            print(f"⚠ {warning_msg}")
            
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                print(f"  Converted {col} to numeric")
            except Exception as e:
                error_msg = f"Failed to convert {col} to numeric: {str(e)}"
                validation_report["errors"].append(error_msg)
                print(f"✗ {error_msg}")

    print("\nChecking for extreme outliers...")
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            warning_msg = f"Column {col} contains {inf_count} infinite values"
            validation_report["warnings"].append(warning_msg)
            print(f"⚠ {warning_msg}")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            if missing_value_strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif missing_value_strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif missing_value_strategy == "zero":
                df[col] = df[col].fillna(0)
            elif missing_value_strategy == "drop":
                df = df.dropna(subset=[col])
        
        positive_only_cols = [
            "ave_flot_air_flow",
            "ave_flot_level",
            "iron_feed",
            "starch_flow",
            "amina_flow",
            "ore_pulp_flow",
            "ore_pulp_density",
            "silica_concentrate",
        ]
        
        if col in positive_only_cols:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                warning_msg = f"Column {col} contains {negative_count} negative values (should be positive)"
                validation_report["warnings"].append(warning_msg)
                print(f"⚠ {warning_msg}")
                if missing_value_strategy == "drop":
                    df = df[df[col] >= 0]
                else:
                    df.loc[df[col] < 0, col] = 0

    if df["date"].isna().sum() > 0:
        if missing_value_strategy == "drop":
            df = df.dropna(subset=["date"])
            print("✓ Dropped rows with invalid dates")
        else:
            df["date"] = df["date"].ffill().bfill()
            if df["date"].isna().sum() > 0:
                df = df.dropna(subset=["date"])
                print("✓ Dropped rows with invalid dates that couldn't be filled")

    remaining_missing = df.isna().sum().sum()
    if remaining_missing > 0:
        warning_msg = f"Warning: {remaining_missing} missing values still remain after cleaning"
        validation_report["warnings"].append(warning_msg)
        print(f"⚠ {warning_msg}")
    else:
        print("✓ No missing values remaining")

    validation_report["final_shape"] = df.shape
    validation_report["columns_checked"] = list(df.columns)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Original shape: {validation_report['original_shape']}")
    print(f"Final shape: {validation_report['final_shape']}")
    print(f"Rows dropped: {validation_report['rows_dropped']}")
    print(f"Invalid dates found: {validation_report['invalid_dates']}")
    print(f"Warnings: {len(validation_report['warnings'])}")
    print(f"Errors: {len(validation_report['errors'])}")
    
    if validation_report["warnings"]:
        print("\nWarnings:")
        for warning in validation_report["warnings"]:
            print(f"  ⚠ {warning}")
    
    if validation_report["errors"]:
        print("\nErrors:")
        for error in validation_report["errors"]:
            print(f"  ✗ {error}")

    print(f"\nSaving validated dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print(f"✓ Successfully saved validated dataset to {output_path}")

    return df, validation_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate and clean raw data before feature engineering"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/raw.csv",
        help="Path to raw CSV file (default: data/raw/raw.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/raw_validated.csv",
        help="Path to save validated CSV file (default: data/processed/raw_validated.csv)",
    )
    parser.add_argument(
        "--missing-strategy",
        type=str,
        default="drop",
        choices=["drop", "forward_fill", "backward_fill", "mean", "median", "zero"],
        help="Strategy for handling missing values (default: drop)",
    )
    parser.add_argument(
        "--date-format",
        type=str,
        default="infer",
        help="Date format for parsing (default: infer for automatic detection)",
    )

    args = parser.parse_args()

    validate_and_clean_data(
        input_path=args.input,
        output_path=args.output,
        missing_value_strategy=args.missing_strategy,
        date_format=args.date_format,
    )

