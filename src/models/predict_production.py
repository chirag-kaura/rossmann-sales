from pathlib import Path

import mlflow
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = PROJECT_ROOT / "data" / "splits" / "test.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "predictions"

OUTPUT_PATH = (
    OUTPUT_DIR / "predictions_mlflow.csv"
)

MODEL_URI = (
    "models:/rossmann-sales-random-forest/Production"
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

    print("Loading test data...")

    test = pd.read_csv(
        TEST_PATH,
        low_memory=False,
        dtype={"StateHoliday": "string"}
    )

    print(f"Input shape: {test.shape}")

    print("\nPreparing features...")

    X_test = prepare_features(test)

    print(
        f"Feature shape: {X_test.shape}"
    )

    print("\nLoading Production model from MLflow...")

    model = mlflow.pyfunc.load_model(
        MODEL_URI
    )

    print("✓ Production model loaded")

    print("\nGenerating predictions...")

    predictions = model.predict(
        X_test
    )

    predictions = np.maximum(
        predictions,
        0
    )

    output = pd.DataFrame({
        "Store": test["Store"],
        "Date": test["Date"],
        "PredictedSales": predictions
    })

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n✓ Predictions generated")

    print(
        f"Predictions saved to:\n"
        f"{OUTPUT_PATH}"
    )

    print("\nSample predictions:")

    print(
        output.head(10)
    )

    print("\n================================")
    print("MLFLOW PRODUCTION PREDICTION COMPLETED")
    print("================================")


if __name__ == "__main__":
    main()