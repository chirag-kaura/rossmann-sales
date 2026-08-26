from pathlib import Path

import pandas as pd

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "raw" / "train.csv"
STORE_PATH = PROJECT_ROOT / "data" / "raw" / "store.csv"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "train_processed.csv"

# --------------------------------------------------
# Load data
# --------------------------------------------------

def load_data():
    print("Loading raw datasets...")

    train = pd.read_csv(TRAIN_PATH, low_memory=False)
    store = pd.read_csv(STORE_PATH)

    print(f"Train shape: {train.shape}")
    print(f"Store shape: {store.shape}")

    return train, store


# --------------------------------------------------
# Clean train data
# --------------------------------------------------

def clean_train_data(train):

    print("\nCleaning train dataset...")

    # Convert Date to datetime
    train["Date"] = pd.to_datetime(train["Date"], errors="coerce")

    # Standardize StateHoliday
    train["StateHoliday"] = train["StateHoliday"].astype(str).str.strip()

    # Convert "0.0" to "0"
    train["StateHoliday"] = train["StateHoliday"].replace(
        {"0.0": "0", "nan": "0"}
    )

    # Convert numeric columns
    numeric_columns = [
        "Store",
        "DayOfWeek",
        "Sales",
        "Customers",
        "Open",
        "Promo",
        "SchoolHoliday",
    ]

    for column in numeric_columns:
        train[column] = pd.to_numeric(
            train[column],
            errors="coerce"
        )

    return train


# --------------------------------------------------
# Clean store data
# --------------------------------------------------

def clean_store_data(store):

    print("\nCleaning store dataset...")

    numeric_columns = [
        "Store",
        "CompetitionDistance",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2",
        "Promo2SinceWeek",
        "Promo2SinceYear",
    ]

    for column in numeric_columns:
        store[column] = pd.to_numeric(
            store[column],
            errors="coerce"
        )

    # Standardize categorical columns
    categorical_columns = [
        "StoreType",
        "Assortment",
        "PromoInterval",
    ]

    for column in categorical_columns:
        store[column] = store[column].astype("string").str.strip()

    return store


# --------------------------------------------------
# Merge datasets
# --------------------------------------------------

def merge_datasets(train, store):

    print("\nMerging train and store datasets...")

    merged = train.merge(
        store,
        on="Store",
        how="left",
        validate="many_to_one"
    )

    print(f"Merged shape: {merged.shape}")

    return merged


# --------------------------------------------------
# Validate processed data
# --------------------------------------------------

def validate_processed_data(data):

    print("\nValidating processed dataset...")

    # Date validation
    assert data["Date"].notna().all(), \
        "Invalid or missing dates found."

    # Store validation
    assert data["Store"].notna().all(), \
        "Missing Store values found."

    # Sales validation
    assert data["Sales"].notna().all(), \
        "Missing Sales values found."

    assert (data["Sales"] >= 0).all(), \
        "Negative Sales values found."

    # Row count should remain unchanged
    assert len(data) == 1017209, \
        "Unexpected number of rows after processing."

    print("✓ Processed data validation passed")


# --------------------------------------------------
# Save processed data
# --------------------------------------------------

def save_data(data):

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    data.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"\nProcessed dataset saved to:")
    print(OUTPUT_PATH)


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------

def main():

    train, store = load_data()

    train = clean_train_data(train)

    store = clean_store_data(store)

    merged = merge_datasets(
        train,
        store
    )

    validate_processed_data(
        merged
    )

    save_data(
        merged
    )

    print("\n================================")
    print("DATA PROCESSING COMPLETED")
    print("================================")


if __name__ == "__main__":
    main()