from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "validation.csv"
)


# --------------------------------------------------
# Metric
# --------------------------------------------------

def rmspe(actual, predicted):

    mask = actual != 0

    actual = actual[mask]
    predicted = predicted[mask]

    return np.sqrt(
        np.mean(
            ((actual - predicted) / actual) ** 2
        )
    )


# --------------------------------------------------
# Data preparation
# --------------------------------------------------

def prepare_features(train, validation):

    target = "Sales"

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

    X_validation = X_validation.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    X_train = X_train.fillna(0)
    X_validation = X_validation.fillna(0)

    return (
        X_train,
        y_train,
        X_validation,
        y_validation
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("Loading datasets...")

    train = pd.read_csv(
        TRAIN_PATH,
        low_memory=False,
        dtype={"StateHoliday": "string"}
    )

    validation = pd.read_csv(
        VALIDATION_PATH,
        low_memory=False,
        dtype={"StateHoliday": "string"}
    )

    print(f"Train shape: {train.shape}")
    print(f"Validation shape: {validation.shape}")

    (
        X_train,
        y_train,
        X_validation,
        y_validation
    ) = prepare_features(
        train,
        validation
    )

    print(
        f"Final training features: {X_train.shape}"
    )

    print(
        f"Final validation features: {X_validation.shape}"
    )

    # --------------------------------------------------
    # Train XGBoost
    # --------------------------------------------------

    print("\nTraining XGBoost...")

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_validation, y_validation)
        ],
        verbose=False
    )

    print("✓ Model training completed")

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    print(
        "\nGenerating validation predictions..."
    )

    predictions = model.predict(
        X_validation
    )

    predictions = np.maximum(
        predictions,
        0
    )

    score = rmspe(
        y_validation.values,
        predictions
    )

    print(
        f"\nValidation RMSPE: {score:.4f}"
    )

    # --------------------------------------------------
    # Feature importance
    # --------------------------------------------------

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
        importance.head(15).to_string(
            index=False
        )
    )

    print("\n================================")
    print("XGBOOST COMPLETED")
    print("================================")


if __name__ == "__main__":
    main()