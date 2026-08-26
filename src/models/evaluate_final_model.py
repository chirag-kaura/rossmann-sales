from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "test.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest_final.pkl"
)


# --------------------------------------------------
# RMSPE
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
# Main
# --------------------------------------------------

def main():

    print("Loading final model...")

    model_package = joblib.load(
        MODEL_PATH
    )

    model = model_package["model"]
    feature_columns = model_package["features"]

    print("✓ Model loaded")

    print("\nLoading test dataset...")

    test = pd.read_csv(
        TEST_PATH,
        low_memory=False,
        dtype={"StateHoliday": "string"}
    )

    print(f"Test shape: {test.shape}")

    # -----------------------------------------
    # Prepare features
    # -----------------------------------------

    feature_input = [
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

    X_test = test[feature_input].copy()

    y_test = test["Sales"].copy()

    # -----------------------------------------
    # Encode categorical variables
    # -----------------------------------------

    categorical_columns = [
        "StoreType",
        "Assortment",
    ]

    X_test = pd.get_dummies(
        X_test,
        columns=categorical_columns,
        dtype=int
    )

    # Ensure exactly the same feature structure
    X_test = X_test.reindex(
        columns=feature_columns,
        fill_value=0
    )

    X_test = X_test.fillna(0)

    print(
        f"Final test features: {X_test.shape}"
    )

    # -----------------------------------------
    # Predict
    # -----------------------------------------

    print("\nGenerating test predictions...")

    predictions = model.predict(
        X_test
    )

    predictions = np.maximum(
        predictions,
        0
    )

    # -----------------------------------------
    # Evaluate
    # -----------------------------------------

    score = rmspe(
        y_test.values,
        predictions
    )

    print(
        f"\nFinal Test RMSPE: {score:.4f}"
    )

    # -----------------------------------------
    # Compare with baseline
    # -----------------------------------------

    baseline_predictions = test[
        "Sales_Lag_7"
    ].values

    baseline_mask = (
        (y_test.values != 0)
        & (~np.isnan(baseline_predictions))
    )

    baseline_score = rmspe(
        y_test.values[baseline_mask],
        baseline_predictions[baseline_mask]
    )

    print(
        f"Baseline Test RMSPE: {baseline_score:.4f}"
    )

    improvement = (
        (baseline_score - score)
        / baseline_score
        * 100
    )

    print(
        f"Improvement over baseline: "
        f"{improvement:.2f}%"
    )

    print("\n================================")
    print("FINAL TEST EVALUATION COMPLETED")
    print("================================")


if __name__ == "__main__":
    main()