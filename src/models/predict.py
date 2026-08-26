from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "random_forest_final.pkl"
)


def load_model():

    package = joblib.load(MODEL_PATH)

    return (
        package["model"],
        package["features"]
    )


def prepare_data(data, feature_columns):

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

    X = data[feature_input].copy()

    categorical_columns = [
        "StoreType",
        "Assortment",
    ]

    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        dtype=int
    )

    X = X.reindex(
        columns=feature_columns,
        fill_value=0
    )

    X = X.fillna(0)

    return X


def predict(data):

    model, feature_columns = load_model()

    X = prepare_data(
        data,
        feature_columns
    )

    predictions = model.predict(X)

    predictions = np.maximum(
        predictions,
        0
    )

    return predictions


def main():

    print("Loading test data...")

    test_path = (
        PROJECT_ROOT
        / "data"
        / "splits"
        / "test.csv"
    )

    data = pd.read_csv(
        test_path,
        low_memory=False
    )

    print(f"Input shape: {data.shape}")

    print("\nGenerating predictions...")

    predictions = predict(data)

    output = data[
        ["Store", "Date"]
    ].copy()

    output["PredictedSales"] = predictions

    output_path = (
        PROJECT_ROOT
        / "data"
        / "predictions"
        / "predictions.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output.to_csv(
        output_path,
        index=False
    )

    print("\n✓ Predictions generated")

    print(
        f"Predictions saved to:\n{output_path}"
    )

    print("\nSample predictions:")
    print(output.head(10))

    print("\n================================")
    print("PREDICTION PIPELINE COMPLETED")
    print("================================")


if __name__ == "__main__":
    main()