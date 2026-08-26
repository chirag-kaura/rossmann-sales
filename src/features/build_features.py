from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_processed.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "train_features.csv"
)


# --------------------------------------------------
# Load processed data
# --------------------------------------------------

def load_data():

    print("Loading processed dataset...")

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
        dtype={"StateHoliday": "string"}
    )

    print(f"Input shape: {df.shape}")

    return df


# --------------------------------------------------
# Create date features
# --------------------------------------------------

def create_date_features(df):

    print("\nCreating date features...")

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

    return df


# --------------------------------------------------
# Create basic business features
# --------------------------------------------------

def create_business_features(df):

    print("\nCreating business features...")

    # Competition information availability
    df["HasCompetitionDistance"] = (
        df["CompetitionDistance"].notna().astype(int)
    )

    # Promo2 availability
    df["HasPromo2"] = (
        df["Promo2"].fillna(0).astype(int)
    )

    # Store age at observation date
    df["StoreAge"] = (
        df["Year"]
        - df["CompetitionOpenSinceYear"]
    )

    # Prevent invalid negative store-age values
    df["StoreAge"] = df["StoreAge"].clip(lower=0)

    return df


# --------------------------------------------------
# Create Lag Features
# --------------------------------------------------


def create_lag_features(df):

    print("\nCreating lag features...")

    df = df.sort_values(
        ["Store", "Date"]
    ).copy()

    df["Sales_Lag_1"] = (
        df.groupby("Store")["Sales"]
        .shift(1)
    )

    df["Sales_Lag_7"] = (
        df.groupby("Store")["Sales"]
        .shift(7)
    )

    df["Sales_Lag_14"] = (
        df.groupby("Store")["Sales"]
        .shift(14)
    )

    return df


# --------------------------------------------------
# Create Rolling Features
# --------------------------------------------------


def create_rolling_features(df):

    print("\nCreating rolling features...")

    df = df.sort_values(
        ["Store", "Date"]
    ).copy()

    # Shift first so today's Sales is never included
    df["Sales_Rolling_Mean_7"] = (
        df.groupby("Store")["Sales"]
        .transform(
            lambda x: x.shift(1).rolling(7).mean()
        )
    )

    df["Sales_Rolling_Mean_14"] = (
        df.groupby("Store")["Sales"]
        .transform(
            lambda x: x.shift(1).rolling(14).mean()
        )
    )

    df["Sales_Rolling_Mean_30"] = (
        df.groupby("Store")["Sales"]
        .transform(
            lambda x: x.shift(1).rolling(30).mean()
        )
    )

    return df



# --------------------------------------------------
# Validate features
# --------------------------------------------------

def validate_features(df):

    print("\nValidating feature dataset...")

    required_columns = [
        "Store",
        "Date",
        "Sales",
        "Customers",
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
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

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    assert not missing_columns, (
        f"Missing feature columns: {missing_columns}"
    )

    assert df["Date"].notna().all(), (
        "Missing or invalid dates found."
    )

    assert len(df) == 1017209, (
        "Unexpected row count after feature engineering."
    )

    print("✓ Feature validation passed")


# --------------------------------------------------
# Save features
# --------------------------------------------------

def save_features(df):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nFeature dataset saved to:")
    print(OUTPUT_PATH)


# --------------------------------------------------
# Main pipeline
# --------------------------------------------------

def main():

    df = load_data()

    df = create_date_features(df)

    df = create_business_features(df)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    validate_features(df)

    save_features(df)

    print("\n================================")
    print("FEATURE ENGINEERING COMPLETED")
    print("================================")


if __name__ == "__main__":
    main()