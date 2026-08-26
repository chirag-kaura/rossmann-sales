from pathlib import Path
import pandas as pd

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
STORE_PATH = PROJECT_ROOT / "data" / "raw" / "store.csv"


# Expected columns
EXPECTED_TRAIN_COLUMNS = {
    "Store",
    "DayOfWeek",
    "Date",
    "Sales",
    "Customers",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
}

EXPECTED_STORE_COLUMNS = {
    "Store",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2",
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "PromoInterval",
}

def validate_data():
    print("Starting data validation...\n")

    # 1. Check files
    assert TRAIN_PATH.exists(), f"Train file not found: {TRAIN_PATH}"
    assert STORE_PATH.exists(), f"Store file not found: {STORE_PATH}"

    print("✓ Required files exist")

    # Load data
    train = pd.read_csv(TRAIN_PATH)
    store = pd.read_csv(STORE_PATH)

    # 2. Check columns
    assert set(train.columns) == EXPECTED_TRAIN_COLUMNS, (
        "Train dataset columns do not match expected schema"
    )

    assert set(store.columns) == EXPECTED_STORE_COLUMNS, (
        "Store dataset columns do not match expected schema"
    )

    print("✓ Column/schema validation passed")

    # 3. Check duplicates
    train_duplicates = train.duplicated().sum()
    store_duplicates = store.duplicated().sum()

    assert train_duplicates == 0, (
        f"Train dataset contains {train_duplicates} duplicate rows"
    )

    assert store_duplicates == 0, (
        f"Store dataset contains {store_duplicates} duplicate rows"
    )

    print("✓ Duplicate validation passed")

    # 4. Check Store IDs
    assert train["Store"].notna().all(), "Train contains missing Store IDs"
    assert store["Store"].notna().all(), "Store contains missing Store IDs"

    assert train["Store"].nunique() == 1115, (
        f"Expected 1115 stores in train, found {train['Store'].nunique()}"
    )

    assert store["Store"].nunique() == 1115, (
        f"Expected 1115 stores in store dataset, found {store['Store'].nunique()}"
    )

    print("✓ Store ID validation passed")

    # 5. Check Sales
    assert train["Sales"].notna().all(), "Sales contains missing values"
    assert (train["Sales"] >= 0).all(), "Sales contains negative values"

    print("✓ Sales validation passed")

    # 6. Check Date
    dates = pd.to_datetime(train["Date"], errors="coerce")

    assert dates.notna().all(), "Invalid dates found in Date column"

    print("✓ Date validation passed")

    # 7. Check Open values
    assert train["Open"].isin([0, 1]).all(), (
        "Open column contains values other than 0 or 1"
    )

    # 8. Check Promo values
    assert train["Promo"].isin([0, 1]).all(), (
        "Promo column contains values other than 0 or 1"
    )

    # 9. Check SchoolHoliday values
    assert train["SchoolHoliday"].isin([0, 1]).all(), (
        "SchoolHoliday column contains values other than 0 or 1"
    )

    print("✓ Binary feature validation passed")

    print("\n================================")
    print("DATA VALIDATION PASSED")
    print("================================")

if __name__ == "__main__":
    validate_data()