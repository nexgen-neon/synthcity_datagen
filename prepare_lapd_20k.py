import pandas as pd


INPUT = "data/real/export.csv"
OUTPUT = "data/real/lapd_20k.csv"

SAMPLE_SIZE = 20_000
RANDOM_STATE = 42


def main():

    # ---------------------------------------------------------
    # 1. Load the real LAPD dataset
    # ---------------------------------------------------------

    print("Loading dataset...")

    df = pd.read_csv(INPUT)

    print(f"Original shape: {df.shape}")

    print("\nAvailable columns:")
    print(df.columns.tolist())


    # ---------------------------------------------------------
    # 2. Sample 20,000 real records
    # ---------------------------------------------------------

    df = df.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE
    ).reset_index(drop=True)

    print(f"\nSampled shape: {df.shape}")


    # ---------------------------------------------------------
    # 3. Columns we want to use
    # ---------------------------------------------------------

    requested_columns = [
        "AREA",
        "AREA NAME",
        "Rpt Dist No",
        "Part 1-2",
        "Crm Cd",
        "Crm Cd Desc",
        "Vict Age",
        "Vict Sex",
        "Vict Descent",
        "Premis Cd",
        "Premis Desc",
        "Weapon Used Cd",
        "Weapon Desc",
        "Status",
        "Status Desc",
        "LAT",
        "LON",
    ]


    # ---------------------------------------------------------
    # 4. Find which columns actually exist
    # ---------------------------------------------------------

    available_columns = [
        column
        for column in requested_columns
        if column in df.columns
    ]

    missing_columns = [
        column
        for column in requested_columns
        if column not in df.columns
    ]


    print("\nColumns selected:")
    print(available_columns)

    if missing_columns:
        print("\nColumns not found:")
        print(missing_columns)


    # ---------------------------------------------------------
    # 5. Make sure we have enough useful columns
    # ---------------------------------------------------------

    if len(available_columns) < 5:
        raise ValueError(
            "Too many requested columns are missing. "
            "Check the column names printed above."
        )


    df = df[available_columns].copy()


    # ---------------------------------------------------------
    # 6. Identify numeric and categorical columns
    # ---------------------------------------------------------

    numeric_columns = [
        column
        for column in [
            "AREA",
            "Rpt Dist No",
            "Part 1-2",
            "Crm Cd",
            "Vict Age",
            "Premis Cd",
            "Weapon Used Cd",
            "LAT",
            "LON",
        ]
        if column in df.columns
    ]


    categorical_columns = [
        column
        for column in [
            "AREA NAME",
            "Crm Cd Desc",
            "Vict Sex",
            "Vict Descent",
            "Premis Desc",
            "Weapon Desc",
            "Status",
            "Status Desc",
        ]
        if column in df.columns
    ]


    # ---------------------------------------------------------
    # 7. Clean numeric columns
    # ---------------------------------------------------------

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        df[column] = df[column].fillna(0)


    # ---------------------------------------------------------
    # 8. Clean Vict Age
    # ---------------------------------------------------------

    if "Vict Age" in df.columns:

        valid_ages = df.loc[
            df["Vict Age"] >= 0,
            "Vict Age"
        ]

        if len(valid_ages) > 0:

            median_age = valid_ages.median()

            df.loc[
                df["Vict Age"] < 0,
                "Vict Age"
            ] = median_age


    # ---------------------------------------------------------
    # 9. Clean categorical columns
    # ---------------------------------------------------------

    for column in categorical_columns:

        df[column] = df[column].fillna("Unknown")

        df[column] = df[column].astype(str)

        df.loc[
            df[column].str.strip() == "",
            column
        ] = "Unknown"


    # ---------------------------------------------------------
    # 10. Group extremely rare categories
    # ---------------------------------------------------------

    MIN_CATEGORY_FREQUENCY = 5

    for column in categorical_columns:

        counts = df[column].value_counts()

        rare_values = counts[
            counts < MIN_CATEGORY_FREQUENCY
        ].index

        df.loc[
            df[column].isin(rare_values),
            column
        ] = "Unknown"


    # ---------------------------------------------------------
    # 11. Convert coded fields to categorical strings
    # ---------------------------------------------------------

    coded_categorical_columns = [
        "AREA",
        "Rpt Dist No",
        "Part 1-2",
        "Crm Cd",
        "Premis Cd",
        "Weapon Used Cd",
    ]

    for column in coded_categorical_columns:

        if column in df.columns:

            df[column] = df[column].astype(str)


    # ---------------------------------------------------------
    # 12. Final validation
    # ---------------------------------------------------------

    print("\n======================================")
    print("FINAL DATASET")
    print("======================================")

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")

    print(
        df.isna()
        .sum()
        .sort_values(ascending=False)
    )


    # ---------------------------------------------------------
    # 13. Save
    # ---------------------------------------------------------

    df.to_csv(
        OUTPUT,
        index=False
    )

    print("\n======================================")
    print("SUCCESS")
    print("======================================")

    print(f"Output: {OUTPUT}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()