from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "train.csv"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "random_forest_final.pkl"


def main():

    print("Loading training dataset...")

    train = pd.read_csv(
        TRAIN_PATH,
        low_memory=False,
        dtype={"StateHoliday": "string"}
    )

    print(f"Training shape: {train.shape}")

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

    categorical_columns = [
        "StoreType",
        "Assortment",
    ]

    X_train = pd.get_dummies(
        X_train,
        columns=categorical_columns,
        dtype=int
    )

    X_train = X_train.fillna(0)

    print(f"Final training features: {X_train.shape}")

    print("\nTraining final Random Forest...")

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

    print("✓ Final model training completed")

    # Save model and feature schema together
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    model_package = {
        "model": model,
        "features": X_train.columns.tolist()
    }

    joblib.dump(
        model_package,
        MODEL_PATH
    )

    print("\nFinal model saved to:")
    print(MODEL_PATH)

    print("\n================================")
    print("FINAL MODEL CREATED")
    print("================================")


if __name__ == "__main__":
    main()