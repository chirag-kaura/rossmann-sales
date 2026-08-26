from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "splits" / "validation.csv"


def rmspe(actual, predicted):
    mask = actual != 0

    return np.sqrt(
        np.mean(
            ((actual[mask] - predicted[mask]) / actual[mask]) ** 2
        )
    )


def main():

    print("Loading datasets...")

    train = pd.read_csv(
        TRAIN_PATH,
        low_memory=False
    )

    validation = pd.read_csv(
        VALIDATION_PATH,
        low_memory=False
    )

    print(f"Train shape: {train.shape}")
    print(f"Validation shape: {validation.shape}")

    # -----------------------------------------
    # Target
    # -----------------------------------------

    target = "Sales"

    # -----------------------------------------
    # Features
    # -----------------------------------------

    feature_columns = [
        "Store",
        "DayOfWeek",
        "Open",
        "Promo",
        "SchoolHoliday",
        "CompetitionDistance",
        "StoreType",
        "Assortment",
        "Year",
        "Month",
        "Day",
        "WeekOfYear",
        "HasCompetitionDistance",
        "HasPromo2",
        "StoreAge",
        "Sales_Lag_1",
        "Sales_Lag_7",
        "Sales_Lag_14",
        "Sales_Rolling_Mean_7",
        "Sales_Rolling_Mean_14",
        "Sales_Rolling_Mean_30",
    ]

    X_train = train[feature_columns].copy()
    y_train = train[target].copy()

    X_validation = validation[feature_columns].copy()
    y_validation = validation[target].copy()

    # -----------------------------------------
    # Handle categorical variables
    # -----------------------------------------

    categorical_columns = [
        "StoreType",
        "Assortment",
    ]

    X_train = pd.get_dummies(
        X_train,
        columns=categorical_columns,
        dtype=int
    )

    X_validation = pd.get_dummies(
        X_validation,
        columns=categorical_columns,
        dtype=int
    )

    # Make validation columns identical to training
    X_validation = X_validation.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    # -----------------------------------------
    # Handle missing values
    # -----------------------------------------

    X_train = X_train.fillna(0)
    X_validation = X_validation.fillna(0)

    print(f"Final training features: {X_train.shape}")
    print(f"Final validation features: {X_validation.shape}")

    # -----------------------------------------
    # Train model
    # -----------------------------------------

    print("\nTraining Random Forest...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print("✓ Model training completed")

    # -----------------------------------------
    # Validation prediction
    # -----------------------------------------

    print("\nGenerating validation predictions...")

    predictions = model.predict(X_validation)

    score = rmspe(
        y_validation.values,
        predictions
    )

    print(f"\nValidation RMSPE: {score:.4f}")



    # -----------------------------------------
    # Feature importance
    # -----------------------------------------

    importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    print("\nTop 15 Feature Importances:")
    print(
        importance.head(15).to_string(index=False)
    )
    

    print("\n================================")
    print("RANDOM FOREST COMPLETED")
    print("================================")


if __name__ == "__main__":
    main()