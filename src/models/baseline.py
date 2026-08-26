from pathlib import Path
import pandas as pd
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "validation.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "test.csv"
)


def rmspe(actual, predicted):

    mask = actual != 0

    return np.sqrt(
        np.mean(
            ((actual[mask] - predicted[mask]) / actual[mask]) ** 2
        )
    )


def evaluate_baseline(df, dataset_name):

    actual = df["Sales"].values
    predicted = df["Sales_Lag_7"].values

    # Remove rows where lag is unavailable
    mask = ~np.isnan(predicted)

    actual = actual[mask]
    predicted = predicted[mask]

    score = rmspe(actual, predicted)

    print(f"\n{dataset_name} Baseline")
    print(f"Records evaluated: {len(actual)}")
    print(f"RMSPE: {score:.4f}")

    return score


def main():

    print("Loading validation and test datasets...")

    validation = pd.read_csv(
        VALIDATION_PATH,
        low_memory=False
    )

    test = pd.read_csv(
        TEST_PATH,
        low_memory=False
    )

    validation["Date"] = pd.to_datetime(validation["Date"])
    test["Date"] = pd.to_datetime(test["Date"])

    validation_score = evaluate_baseline(
        validation,
        "Validation"
    )

    test_score = evaluate_baseline(
        test,
        "Test"
    )

    print("\n================================")
    print("BASELINE MODEL COMPLETED")
    print("================================")

    print(f"Validation RMSPE: {validation_score:.4f}")
    print(f"Test RMSPE:       {test_score:.4f}")


if __name__ == "__main__":
    main()