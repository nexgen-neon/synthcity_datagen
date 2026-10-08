import pandas as pd


INPUT = "data/real/export.csv"
OUTPUT = "data/real/lapd_5k.csv"

SAMPLE_SIZE = 5_000
RANDOM_STATE = 42


def main():

    print("Loading real LAPD dataset...")

    df = pd.read_csv(INPUT)

    print(f"Original shape: {df.shape}")


    # ---------------------------------------------------------
    # Sample 5,000 real records
    # ---------------------------------------------------------

    df = df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    print(f"Sampled shape: {df.shape}")


    # ---------------------------------------------------------
    # Keep only useful columns
    # ---------------------------------------------------------

    columns = [
        "AREA NAME",
        "Crm Cd Desc",
        "Vict Age",
        "Vict Sex",
        "Vict Descent",
        "Premis Desc",
        "Status",
        "LAT",
        "LON",
    ]

    df = df[columns].copy()


    # ---------------------------------------------------------
    # Clean Vict Age
    # ---------------------------------------------------------

    df["Vict Age"] = pd.to_numeric(
        df["Vict Age"],
        errors="coerce"
    )

    valid_ages = df.loc[
        df["Vict Age"] >= 0,
        "Vict Age"
    ]

    median_age = valid_ages.median()

    df.loc[
        df["Vict Age"] < 0,
        "Vict Age"
    ] = median_age

    df["Vict Age"] = df["Vict Age"].fillna(
        median_age
    )


    # ---------------------------------------------------------
    # Clean categorical columns
    # ---------------------------------------------------------

    categorical_columns = [
        "AREA NAME",
        "Crm Cd Desc",
        "Vict Sex",
        "Vict Descent",
        "Premis Desc",
        "Status",
    ]

    for column in categorical_columns:

        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

        df.loc[
            df[column] == "",
            column
        ] = "Unknown"


    # ---------------------------------------------------------
    # Group very rare categories
    # ---------------------------------------------------------

    MIN_FREQUENCY = 5

    for column in categorical_columns:

        counts = df[column].value_counts()

        rare_values = counts[
            counts < MIN_FREQUENCY
        ].index

        df.loc[
            df[column].isin(rare_values),
            column
        ] = "Unknown"


    # ---------------------------------------------------------
    # Clean coordinates
    # ---------------------------------------------------------

    df["LAT"] = pd.to_numeric(
        df["LAT"],
        errors="coerce"
    ).fillna(0)

    df["LON"] = pd.to_numeric(
        df["LON"],
        errors="coerce"
    ).fillna(0)


    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    print("\nFinal dataset")
    print("-----------------------------")

    print("Shape:", df.shape)

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nData types:")
    print(df.dtypes)

    print("\nVict Age range:")
    print(
        df["Vict Age"].min(),
        "to",
        df["Vict Age"].max()
    )

    print("\nLAT range:")
    print(
        df["LAT"].min(),
        "to",
        df["LAT"].max()
    )

    print("\nLON range:")
    print(
        df["LON"].min(),
        "to",
        df["LON"].max()
    )


    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    df.to_csv(
        OUTPUT,
        index=False
    )

    print("\nDataset saved:")
    print(OUTPUT)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()