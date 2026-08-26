from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"
VALIDATION_PATH = PROJECT_ROOT / "data" / "splits" / "validation.csv"
TEST_PATH = PROJECT_ROOT / "data" / "splits" / "test.csv"


def rmspe(actual, predicted):

    mask = actual != 0

    actual = actual[mask]
    predicted = predicted[mask]

    return np.sqrt(
        np.mean(
            ((actual - predicted) / actual) ** 2
        )
    )


def prepare_features(data):

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

    X = data[feature_columns].copy()

    X = pd.get_dummies(
        X,
        columns=["StoreType", "Assortment"],
        dtype=int
    )

    X = X.fillna(0)

    return X


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

    test = pd.read_csv(
        TEST_PATH,
        low_memory=False,
        dtype={"StateHoliday": "string"}
    )

    print(f"Train shape: {train.shape}")
    print(f"Validation shape: {validation.shape}")
    print(f"Test shape: {test.shape}")

    # -----------------------------------------
    # Prepare training data
    # -----------------------------------------

    X_train = prepare_features(train)
    y_train = train["Sales"]

    # -----------------------------------------
    # Prepare validation data
    # -----------------------------------------

    X_validation = prepare_features(validation)
    y_validation = validation["Sales"]

    # Ensure same feature structure
    X_validation = X_validation.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    # -----------------------------------------
    # Prepare test data
    # -----------------------------------------

    X_test = prepare_features(test)
    y_test = test["Sales"]

    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    print(
        f"Final training features: {X_train.shape}"
    )

    # -----------------------------------------
    # MLflow experiment
    # -----------------------------------------

    mlflow.set_experiment(
        "rossmann-sales"
    )

    with mlflow.start_run(
        run_name="random-forest-final-metrics"
    ):

        # -------------------------------------
        # Parameters
        # -------------------------------------

        n_estimators = 100
        max_depth = 20
        min_samples_leaf = 2
        random_state = 42

        mlflow.log_params({
            "model": "RandomForestRegressor",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state,
            "training_rows": len(X_train),
            "feature_count": X_train.shape[1],
        })

        # -------------------------------------
        # Train model
        # -------------------------------------

        print("\nTraining Random Forest...")

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train
        )

        print("✓ Training completed")

        # -------------------------------------
        # Validation prediction
        # -------------------------------------

        print("\nEvaluating validation set...")

        validation_predictions = model.predict(
            X_validation
        )

        validation_predictions = np.maximum(
            validation_predictions,
            0
        )

        validation_rmspe = rmspe(
            y_validation.values,
            validation_predictions
        )

        print(
            f"Validation RMSPE: "
            f"{validation_rmspe:.4f}"
        )

        # -------------------------------------
        # Test prediction
        # -------------------------------------

        print("\nEvaluating test set...")

        test_predictions = model.predict(
            X_test
        )

        test_predictions = np.maximum(
            test_predictions,
            0
        )

        test_rmspe = rmspe(
            y_test.values,
            test_predictions
        )

        print(
            f"Test RMSPE: "
            f"{test_rmspe:.4f}"
        )

        # -------------------------------------
        # Baseline test performance
        # -------------------------------------

        baseline_predictions = test[
            "Sales_Lag_7"
        ].values

        baseline_mask = (
            (y_test.values != 0)
            & (~np.isnan(baseline_predictions))
        )

        baseline_test_rmspe = rmspe(
            y_test.values[baseline_mask],
            baseline_predictions[baseline_mask]
        )

        # -------------------------------------
        # Improvement
        # -------------------------------------

        improvement = (
            (baseline_test_rmspe - test_rmspe)
            / baseline_test_rmspe
            * 100
        )

        print(
            f"Baseline Test RMSPE: "
            f"{baseline_test_rmspe:.4f}"
        )

        print(
            f"Improvement over baseline: "
            f"{improvement:.2f}%"
        )

        # -------------------------------------
        # Log metrics to MLflow
        # -------------------------------------

        mlflow.log_metrics({
            "validation_rmspe": validation_rmspe,
            "test_rmspe": test_rmspe,
            "baseline_test_rmspe": baseline_test_rmspe,
            "improvement_over_baseline_percent": improvement
        })

        # -------------------------------------
        # Tags
        # -------------------------------------

        mlflow.set_tags({
            "project": "rossmann-sales",
            "model_type": "random_forest",
            "stage": "final",
            "evaluation": "validation_and_test"
        })

        # -------------------------------------
        # Log model
        # -------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="random_forest_model"
        )

        print("\n✓ Parameters logged")
        print("✓ Metrics logged")
        print("✓ Model logged")

        print(
            f"\nRun ID: "
            f"{mlflow.active_run().info.run_id}"
        )

    print("\n================================")
    print("MLFLOW METRICS LOGGING COMPLETED")
    print("================================")


if __name__ == "__main__":
    main()