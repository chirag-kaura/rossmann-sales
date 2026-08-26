from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
)


def main():

    print("Loading feature dataset...")

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False,
        dtype={"StateHoliday": "string"}
    )

    df["Date"] = pd.to_datetime(df["Date"])

    print(f"Dataset shape: {df.shape}")
    print(
        f"Date range: {df['Date'].min().date()} "
        f"to {df['Date'].max().date()}"
    )

    # -----------------------------
    # Time-based split
    # -----------------------------

    train = df[
        (df["Date"] >= "2013-01-01") &
        (df["Date"] <= "2014-12-31")
    ]

    validation = df[
        (df["Date"] >= "2015-01-01") &
        (df["Date"] <= "2015-06-30")
    ]

    test = df[
        (df["Date"] >= "2015-07-01") &
        (df["Date"] <= "2015-07-31")
    ]

    # -----------------------------
    # Validation
    # -----------------------------

    assert len(train) > 0
    assert len(validation) > 0
    assert len(test) > 0

    assert train["Date"].max() < validation["Date"].min()
    assert validation["Date"].max() < test["Date"].min()

    assert len(train) + len(validation) + len(test) == len(df)

    print("\nSplit validation passed")

    # -----------------------------
    # Save
    # -----------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    validation.to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False
    )

    test.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    print("\nTrain:")
    print(train.shape)
    print(train["Date"].min(), "to", train["Date"].max())

    print("\nValidation:")
    print(validation.shape)
    print(validation["Date"].min(), "to", validation["Date"].max())

    print("\nTest:")
    print(test.shape)
    print(test["Date"].min(), "to", test["Date"].max())

    print("\n================================")
    print("DATA SPLIT COMPLETED")
    print("================================")


if __name__ == "__main__":
    main()