import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
import xgboost as xgb


def train_model(
    best_params_path: str = "models/data/best_params.pkl",
    input_dir: str = "data/processed",
    output_path: str = "models/models/trained_model.pkl",
    history_path: str = "models/models/trained_model_with_history.pkl",
    random_state: int = 42,
    validation_split: float = 0.2,
):
    """
    Train an XGBoost model with best hyperparameters and save training history.

    Parameters:
    -----------
    best_params_path : str
        Path to the best parameters pickle file (default: models/data/best_params.pkl)
    input_dir : str
        Directory containing the normalized training data (default: data/processed)
    output_path : str
        Path to save the trained model pickle file (default: models/models/trained_model.pkl)
    history_path : str
        Path to save the model with training history (default: models/models/trained_model_with_history.pkl)
    random_state : int
        Random seed for reproducibility (default: 42)
    validation_split : float
        Fraction of training data to use for validation tracking (default: 0.2)
    """
    # Create output directories if they don't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    history_file = Path(history_path)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_dir)
    best_params_file = Path(best_params_path)

    print(f"Loading best parameters from {best_params_path}...")
    if not best_params_file.exists():
        raise FileNotFoundError(
            f"Best parameters file not found at {best_params_path}. "
            "Please run grid_search.py first."
        )
    
    best_params = load(best_params_path)
    print("✓ Best parameters loaded successfully")
    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    print(f"\nLoading normalized training data from {input_dir}...")
    X_train_scaled = pd.read_csv(input_path / "X_train_scaled.csv")
    y_train = pd.read_csv(input_path / "y_train.csv").squeeze()

    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Target variable shape: {y_train.shape}")

    print("\nValidating data quality...")
    
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
    
    train_inf = np.isinf(X_train_scaled.select_dtypes(include=[np.number])).sum().sum()
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

    print("\n" + "=" * 60)
    print("PREPARING TRAINING DATA")
    print("=" * 60)
    print(f"Splitting training data (validation split: {validation_split})...")
    
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train_scaled,
        y_train,
        test_size=validation_split,
        random_state=random_state,
        shuffle=True,
    )
    
    print(f"Training set for fitting: {X_train_fit.shape}")
    print(f"Validation set for tracking: {X_val.shape}")

    print("\n" + "=" * 60)
    print("INITIALIZING MODEL")
    print("=" * 60)
    print("Model: XGBRegressor")
    
    xgb_params = best_params.copy()
    num_boost_round = xgb_params.pop("n_estimators", 100)
    
    xgb_params["seed"] = random_state
    xgb_params["objective"] = "reg:squarederror"
    xgb_params["eval_metric"] = "rmse"
    
    xgb_params.pop("random_state", None)
    
    print("\nModel parameters:")
    for param, value in xgb_params.items():
        print(f"  {param}: {value}")
    print(f"  n_estimators: {num_boost_round}")

    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)
    print("Training with evaluation set tracking...")
    
    dtrain = xgb.DMatrix(X_train_fit, label=y_train_fit)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    evals_result = {}
    
    model = xgb.train(
        params=xgb_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "validation")],
        evals_result=evals_result,
        verbose_eval=False,
    )
    
    print("✓ Model training completed successfully")

    print(f"\nSaving trained model to {output_path}...")
    dump(model, output_path)
    print(f"✓ Successfully saved trained model to {output_path}")

    print(f"\nSaving model with training history to {history_path}...")
    training_history = {
        "model": model,
        "evals_result": evals_result,
        "validation_split": validation_split,
        "random_state": random_state,
    }
    dump(training_history, history_path)
    print(f"✓ Successfully saved training history to {history_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train model with best hyperparameters"
    )
    parser.add_argument(
        "--best-params",
        type=str,
        default="models/data/best_params.pkl",
        help="Path to best parameters pickle file (default: models/data/best_params.pkl)",
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
        default="models/models/trained_model.pkl",
        help="Path to save trained model pickle file (default: models/models/trained_model.pkl)",
    )
    parser.add_argument(
        "--history",
        type=str,
        default="models/models/trained_model_with_history.pkl",
        help="Path to save model with training history (default: models/models/trained_model_with_history.pkl)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.2,
        help="Fraction of training data for validation tracking (default: 0.2)",
    )

    args = parser.parse_args()

    train_model(
        best_params_path=args.best_params,
        input_dir=args.input_dir,
        output_path=args.output,
        history_path=args.history,
        random_state=args.random_state,
        validation_split=args.validation_split,
    )

