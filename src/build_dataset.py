import pandas as pd
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

PROCESSED_DIR = Path("data/processed")
WEATHER_DIR = PROCESSED_DIR / "weather"

DEMAND_FILE = PROCESSED_DIR / "swpp_demand_2020_2025.csv"

OUTPUT_FILE = PROCESSED_DIR / "modeling_dataset.csv"


# ------------------------------------------------------------
# Weather files
# ------------------------------------------------------------

weather_files = {
    "kc": WEATHER_DIR / "kansas_city_weather_2020_2025.csv",
    "wichita": WEATHER_DIR / "wichita_weather_2020_2025.csv",
    "oklahoma": WEATHER_DIR / "oklahoma_weather_2020_2025.csv",
    "omaha": WEATHER_DIR / "omaha_weather_2020_2025.csv",
    "fargo": WEATHER_DIR / "fargo_weather_2020_2025.csv"
}


# ------------------------------------------------------------
# Weather features we want to keep
# ------------------------------------------------------------

weather_features = [
    "temperature",
    "dew_point_temperature",
    "relative_humidity",
    "wind_speed"
]


# ============================================================
# Helper functions
# ============================================================

def validate_timestamp(df, name):

    if "timestamp_utc" not in df.columns:
        raise ValueError(
            f"{name} does not contain a timestamp_utc column."
        )

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        errors="coerce"
    )

    missing_timestamps = df["timestamp_utc"].isna().sum()

    duplicates = df["timestamp_utc"].duplicated().sum()

    if missing_timestamps > 0:
        raise ValueError(
            f"{name} contains {missing_timestamps} invalid timestamps."
        )

    if duplicates > 0:
        raise ValueError(
            f"{name} contains {duplicates} duplicate timestamps."
        )

    return df


# ============================================================
# Load demand
# ============================================================

print("\n")
print("=" * 60)
print("LOADING SWPP DEMAND")
print("=" * 60)

demand_df = pd.read_csv(DEMAND_FILE)

demand_df = validate_timestamp(
    demand_df,
    "SWPP demand"
)

print("Rows:", len(demand_df))
print("Columns:", demand_df.columns.tolist())

print(
    "First timestamp:",
    demand_df["timestamp_utc"].min()
)

print(
    "Last timestamp:",
    demand_df["timestamp_utc"].max()
)


# ============================================================
# Detect demand column
# ============================================================

possible_demand_columns = [
    "demand_mwh",
    "demand_mw",
    "demand",
    "value",
    "Demand",
    "DEMAND"
]

demand_column = None

for column in possible_demand_columns:

    if column in demand_df.columns:
        demand_column = column
        break


if demand_column is None:

    raise ValueError(
        "Could not automatically identify the demand column.\n"
        f"Available columns: {demand_df.columns.tolist()}"
    )


print("\nDemand column found:", demand_column)


# Rename to a consistent name
if demand_column != "demand_mw":

    demand_df = demand_df.rename(
        columns={
            demand_column: "demand_mw"
        }
    )


# Keep only fields needed for modeling
demand_df = demand_df[
    [
        "timestamp_utc",
        "demand_mw"
    ]
].copy()


# ============================================================
# Start combined dataset with demand
# ============================================================

combined_df = demand_df.copy()


# ============================================================
# Load and merge weather
# ============================================================

for prefix, file_path in weather_files.items():

    print("\n")
    print("=" * 60)
    print(f"LOADING WEATHER: {prefix.upper()}")
    print("=" * 60)

    weather_df = pd.read_csv(file_path)

    weather_df = validate_timestamp(
        weather_df,
        f"{prefix} weather"
    )

    print("Rows:", len(weather_df))

    # --------------------------------------------------------
    # Check required features
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in weather_features
        if column not in weather_df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{prefix} is missing columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Keep only timestamp + weather measurements
    #
    # STATION and Station_name are useful for provenance,
    # but not useful as ML features in this table.
    # --------------------------------------------------------

    weather_df = weather_df[
        [
            "timestamp_utc",
            *weather_features
        ]
    ].copy()

    # --------------------------------------------------------
    # Prefix weather columns
    #
    # Example:
    # temperature -> kc_temperature
    # --------------------------------------------------------

    rename_map = {
        feature: f"{prefix}_{feature}"
        for feature in weather_features
    }

    weather_df = weather_df.rename(
        columns=rename_map
    )

    # --------------------------------------------------------
    # Merge with combined dataset
    # --------------------------------------------------------

    rows_before = len(combined_df)

    combined_df = combined_df.merge(
        weather_df,
        on="timestamp_utc",
        how="left",
        validate="one_to_one"
    )

    rows_after = len(combined_df)

    print(
        f"Rows before merge: {rows_before}"
    )

    print(
        f"Rows after merge:  {rows_after}"
    )

    if rows_before != rows_after:

        raise ValueError(
            f"Row count changed after merging {prefix}."
        )


# ============================================================
# Sort final dataset
# ============================================================

combined_df = combined_df.sort_values(
    "timestamp_utc"
).reset_index(drop=True)


# ============================================================
# Validate combined dataset
# ============================================================

print("\n")
print("=" * 60)
print("COMBINED DATASET VALIDATION")
print("=" * 60)

print(
    "Total rows:",
    len(combined_df)
)

print(
    "Total columns:",
    len(combined_df.columns)
)

print(
    "Duplicate timestamps:",
    combined_df["timestamp_utc"].duplicated().sum()
)

print(
    "First timestamp:",
    combined_df["timestamp_utc"].min()
)

print(
    "Last timestamp:",
    combined_df["timestamp_utc"].max()
)


# ============================================================
# Check missing demand
# ============================================================

missing_demand = combined_df[
    "demand_mw"
].isna().sum()

print(
    "\nMissing demand values:",
    missing_demand
)


# ============================================================
# Check missing weather
# ============================================================

weather_columns = [
    column
    for column in combined_df.columns
    if column != "timestamp_utc"
    and column != "demand_mw"
]


print("\nMissing weather values by column:")

print(
    combined_df[
        weather_columns
    ].isna().sum()
)


# ------------------------------------------------------------
# Find rows containing ANY missing weather value
# ------------------------------------------------------------

combined_df["has_missing_weather"] = (
    combined_df[
        weather_columns
    ]
    .isna()
    .any(axis=1)
)


missing_weather_rows = (
    combined_df[
        "has_missing_weather"
    ].sum()
)


print(
    "\nRows with at least one missing weather feature:",
    missing_weather_rows
)


# ============================================================
# Show missing date range
# ============================================================

missing_rows = combined_df[
    combined_df["has_missing_weather"]
]


if len(missing_rows) > 0:

    print(
        "\nFirst row with missing weather:",
        missing_rows["timestamp_utc"].min()
    )

    print(
        "Last row with missing weather:",
        missing_rows["timestamp_utc"].max()
    )


# ============================================================
# Create ML-ready dataset
# ============================================================

modeling_df = combined_df[
    ~combined_df["has_missing_weather"]
].copy()


# Drop helper flag
modeling_df = modeling_df.drop(
    columns=["has_missing_weather"]
)


# ============================================================
# Final validation
# ============================================================

print("\n")
print("=" * 60)
print("ML-READY DATASET")
print("=" * 60)

print(
    "Rows before removing missing weather:",
    len(combined_df)
)

print(
    "Rows removed:",
    len(combined_df) - len(modeling_df)
)

print(
    "Final rows:",
    len(modeling_df)
)

print(
    "Final columns:",
    len(modeling_df.columns)
)

print(
    "Remaining missing values:",
    modeling_df.isna().sum().sum()
)

print(
    "Duplicate timestamps:",
    modeling_df[
        "timestamp_utc"
    ].duplicated().sum()
)


# ============================================================
# Save
# ============================================================

modeling_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved dataset to:")
print(OUTPUT_FILE)