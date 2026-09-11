import pandas as pd
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

MODELING_FILE = Path(
    "data/processed/modeling_dataset.csv"
)

DEMAND_FILE = Path(
    "data/processed/swpp_demand_2020_2025.csv"
)

OUTPUT_FILE = Path(
    "data/processed/feature_dataset.csv"
)


# ============================================================
# Load modeling dataset
# ============================================================

print("=" * 60)
print("LOADING MODELING DATASET")
print("=" * 60)

df = pd.read_csv(MODELING_FILE)

df["timestamp_utc"] = pd.to_datetime(
    df["timestamp_utc"],
    errors="coerce"
)

df = df.sort_values(
    "timestamp_utc"
).reset_index(drop=True)

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("First timestamp:", df["timestamp_utc"].min())
print("Last timestamp:", df["timestamp_utc"].max())


# ============================================================
# Validate modeling dataset
# ============================================================

if df["timestamp_utc"].isna().any():
    raise ValueError(
        "Invalid timestamps found in modeling dataset."
    )

if df["timestamp_utc"].duplicated().any():
    raise ValueError(
        "Duplicate timestamps found in modeling dataset."
    )

if "demand_mw" not in df.columns:
    raise ValueError(
        "demand_mw column not found in modeling dataset."
    )


# ============================================================
# Create calendar features
# ============================================================

print("\n" + "=" * 60)
print("CREATING CALENDAR FEATURES")
print("=" * 60)

df["hour"] = (
    df["timestamp_utc"]
    .dt.hour
)

df["day_of_week"] = (
    df["timestamp_utc"]
    .dt.dayofweek
)

df["month"] = (
    df["timestamp_utc"]
    .dt.month
)

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

print("Created:")
print("- hour")
print("- day_of_week")
print("- month")
print("- is_weekend")


# ============================================================
# Load complete SWPP demand history
# ============================================================

print("\n" + "=" * 60)
print("LOADING COMPLETE DEMAND HISTORY")
print("=" * 60)

demand_history = pd.read_csv(
    DEMAND_FILE
)

demand_history["timestamp_utc"] = pd.to_datetime(
    demand_history["timestamp_utc"],
    errors="coerce"
)

demand_history = demand_history.sort_values(
    "timestamp_utc"
).reset_index(drop=True)


# ============================================================
# Validate demand history
# ============================================================

if demand_history["timestamp_utc"].isna().any():
    raise ValueError(
        "Invalid timestamps found in demand history."
    )

if demand_history["timestamp_utc"].duplicated().any():
    raise ValueError(
        "Duplicate timestamps found in demand history."
    )


# ------------------------------------------------------------
# Standardize demand column name
# ------------------------------------------------------------

if "demand_mwh" in demand_history.columns:

    demand_history = demand_history.rename(
        columns={
            "demand_mwh": "demand_mw"
        }
    )


if "demand_mw" not in demand_history.columns:

    raise ValueError(
        "Could not find demand column in demand history."
    )


# ------------------------------------------------------------
# Make sure demand quality exists
# ------------------------------------------------------------

if "demand_quality" not in demand_history.columns:

    raise ValueError(
        "demand_quality column not found in demand history. "
        "Run validate_electricity.py first."
    )


valid_quality_values = {
    "valid",
    "flatline"
}

unexpected_quality = set(
    demand_history["demand_quality"]
    .dropna()
    .unique()
) - valid_quality_values

if unexpected_quality:

    raise ValueError(
        "Unexpected demand quality values found: "
        f"{unexpected_quality}"
    )


print("Rows:", len(demand_history))

print(
    "First timestamp:",
    demand_history["timestamp_utc"].min()
)

print(
    "Last timestamp:",
    demand_history["timestamp_utc"].max()
)

print(
    "Missing demand values:",
    demand_history["demand_mw"].isna().sum()
)

print("\nDemand quality counts:")

print(
    demand_history["demand_quality"]
    .value_counts()
)


# ============================================================
# Validate hourly demand timeline
# ============================================================

expected_timestamps = pd.date_range(
    start=demand_history["timestamp_utc"].min(),
    end=demand_history["timestamp_utc"].max(),
    freq="h"
)

actual_timestamps = pd.DatetimeIndex(
    demand_history["timestamp_utc"]
)

missing_demand_timestamps = (
    expected_timestamps
    .difference(actual_timestamps)
)

print(
    "\nMissing timestamps in complete demand history:",
    len(missing_demand_timestamps)
)


if len(missing_demand_timestamps) > 0:

    raise ValueError(
        "Demand history is not a continuous hourly timeline."
    )


# ============================================================
# Create demand lag features
# ============================================================

print("\n" + "=" * 60)
print("CREATING DEMAND LAG FEATURES")
print("=" * 60)

demand_history["demand_lag_1h"] = (
    demand_history["demand_mw"]
    .shift(1)
)

demand_history["demand_lag_24h"] = (
    demand_history["demand_mw"]
    .shift(24)
)

demand_history["demand_lag_168h"] = (
    demand_history["demand_mw"]
    .shift(168)
)

print("Created:")
print("- demand_lag_1h")
print("- demand_lag_24h")
print("- demand_lag_168h")


# ============================================================
# Create lag quality features
# ============================================================

print("\n" + "=" * 60)
print("CREATING DEMAND QUALITY FEATURES")
print("=" * 60)

demand_history["demand_lag_1h_quality"] = (
    demand_history["demand_quality"]
    .shift(1)
)

demand_history["demand_lag_24h_quality"] = (
    demand_history["demand_quality"]
    .shift(24)
)

demand_history["demand_lag_168h_quality"] = (
    demand_history["demand_quality"]
    .shift(168)
)

print("Created:")
print("- demand_quality")
print("- demand_lag_1h_quality")
print("- demand_lag_24h_quality")
print("- demand_lag_168h_quality")


# ============================================================
# Create rolling demand features
# ============================================================

print("\n" + "=" * 60)
print("CREATING ROLLING DEMAND FEATURES")
print("=" * 60)

# Shift by one hour first so the current target value
# is never included in its own rolling statistic.

demand_history["demand_rolling_24h"] = (
    demand_history["demand_mw"]
    .shift(1)
    .rolling(
        window=24
    )
    .mean()
)

demand_history["demand_rolling_168h"] = (
    demand_history["demand_mw"]
    .shift(1)
    .rolling(
        window=168
    )
    .mean()
)

print("Created:")
print("- demand_rolling_24h")
print("- demand_rolling_168h")


# ============================================================
# Inspect feature nulls
# ============================================================

demand_feature_columns = [
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_lag_168h",
    "demand_rolling_24h",
    "demand_rolling_168h"
]

print(
    "\nMissing values created in complete demand history:"
)

print(
    demand_history[
        demand_feature_columns
    ]
    .isna()
    .sum()
)


# ============================================================
# Keep timestamp + engineered demand features + quality
# ============================================================

demand_features = demand_history[
    [
        "timestamp_utc",

        # Current target quality
        "demand_quality",

        # Demand features
        *demand_feature_columns,

        # Lag quality
        "demand_lag_1h_quality",
        "demand_lag_24h_quality",
        "demand_lag_168h_quality"
    ]
].copy()


# ============================================================
# Merge demand features into modeling dataset
# ============================================================

print("\n" + "=" * 60)
print("MERGING DEMAND FEATURES")
print("=" * 60)

rows_before_merge = len(df)

df = df.merge(
    demand_features,
    on="timestamp_utc",
    how="left",
    validate="one_to_one"
)

rows_after_merge = len(df)

print(
    "Rows before merge:",
    rows_before_merge
)

print(
    "Rows after merge:",
    rows_after_merge
)


if rows_before_merge != rows_after_merge:

    raise ValueError(
        "Row count changed while merging demand features."
    )


# ============================================================
# Check engineered feature nulls after merge
# ============================================================

print(
    "\nMissing engineered demand features after merge:"
)

print(
    df[
        demand_feature_columns
    ]
    .isna()
    .sum()
)


# ============================================================
# Remove rows without sufficient demand history
# ============================================================

print("\n" + "=" * 60)
print("REMOVING INCOMPLETE FEATURE ROWS")
print("=" * 60)

rows_before_history_filter = len(df)

df = df.dropna(
    subset=demand_feature_columns
).copy()

rows_after_history_filter = len(df)

print(
    "Rows before filtering:",
    rows_before_history_filter
)

print(
    "Rows removed for insufficient history:",
    rows_before_history_filter
    - rows_after_history_filter
)

print(
    "Rows remaining:",
    rows_after_history_filter
)


# ============================================================
# Remove rows affected by flatline demand data
# ============================================================

print("\n" + "=" * 60)
print("FILTERING DEMAND QUALITY")
print("=" * 60)

rows_before_quality_filter = len(df)


# ------------------------------------------------------------
# Count each reason separately before removing rows
# ------------------------------------------------------------

bad_target = (
    df["demand_quality"] != "valid"
)

bad_lag_24h = (
    df["demand_lag_24h_quality"] != "valid"
)

bad_lag_168h = (
    df["demand_lag_168h_quality"] != "valid"
)


print(
    "Rows with invalid target demand:",
    bad_target.sum()
)

print(
    "Rows with invalid 24h lag:",
    bad_lag_24h.sum()
)

print(
    "Rows with invalid 168h lag:",
    bad_lag_168h.sum()
)


# ------------------------------------------------------------
# A row is usable only if all three are valid
# ------------------------------------------------------------

valid_demand_row = (
    (~bad_target)
    & (~bad_lag_24h)
    & (~bad_lag_168h)
)

df = df[
    valid_demand_row
].copy()


rows_after_quality_filter = len(df)

print(
    "\nRows removed because of demand quality:",
    rows_before_quality_filter
    - rows_after_quality_filter
)

print(
    "Rows remaining after demand QC:",
    rows_after_quality_filter
)


# ============================================================
# Quality removals by year
# ============================================================

quality_check = demand_features.copy()

quality_check["year"] = (
    quality_check["timestamp_utc"]
    .dt.year
)

quality_check["invalid_target"] = (
    quality_check["demand_quality"] != "valid"
)

quality_check["invalid_lag_24h"] = (
    quality_check["demand_lag_24h_quality"] != "valid"
)

quality_check["invalid_lag_168h"] = (
    quality_check["demand_lag_168h_quality"] != "valid"
)

quality_check["invalid_for_modeling"] = (
    quality_check["invalid_target"]
    | quality_check["invalid_lag_24h"]
    | quality_check["invalid_lag_168h"]
)

print("\nDemand quality issues by year:")

quality_summary = (
    quality_check
    .groupby("year")
    .agg(
        invalid_target=(
            "invalid_target",
            "sum"
        ),
        invalid_lag_24h=(
            "invalid_lag_24h",
            "sum"
        ),
        invalid_lag_168h=(
            "invalid_lag_168h",
            "sum"
        ),
        invalid_for_modeling=(
            "invalid_for_modeling",
            "sum"
        )
    )
)

print(
    quality_summary.to_string()
)


# ============================================================
# Verify no bad demand quality remains
# ============================================================

remaining_bad_target = (
    df["demand_quality"] != "valid"
).sum()

remaining_bad_24h = (
    df["demand_lag_24h_quality"] != "valid"
).sum()

remaining_bad_168h = (
    df["demand_lag_168h_quality"] != "valid"
).sum()

if (
    remaining_bad_target > 0
    or remaining_bad_24h > 0
    or remaining_bad_168h > 0
):

    raise ValueError(
        "Invalid demand-quality rows remain "
        "after filtering."
    )


# ============================================================
# Remove quality helper columns
# ============================================================

# These columns were needed to perform QC,
# but the model itself should not learn from them.

quality_columns = [
    "demand_quality",
    "demand_lag_1h_quality",
    "demand_lag_24h_quality",
    "demand_lag_168h_quality"
]

df = df.drop(
    columns=quality_columns
)


# ============================================================
# Create chronological dataset splits
# ============================================================

print("\n" + "=" * 60)
print("CREATING DATASET SPLITS")
print("=" * 60)

year = (
    df["timestamp_utc"]
    .dt.year
)

df["split"] = "unknown"


# ------------------------------------------------------------
# Training: 2020-2023
# ------------------------------------------------------------

df.loc[
    year.between(
        2020,
        2023
    ),
    "split"
] = "train"


# ------------------------------------------------------------
# Validation: 2024
# ------------------------------------------------------------

df.loc[
    year == 2024,
    "split"
] = "validation"


# ------------------------------------------------------------
# Testing: 2025
# ------------------------------------------------------------

df.loc[
    year == 2025,
    "split"
] = "test"


# ------------------------------------------------------------
# Make sure everything was assigned correctly
# ------------------------------------------------------------

unknown_rows = (
    df["split"] == "unknown"
).sum()

if unknown_rows > 0:

    raise ValueError(
        f"{unknown_rows} rows were not assigned a split."
    )


print(
    df["split"]
    .value_counts()
)


# ============================================================
# Split counts with percentages
# ============================================================

print("\nFinal split percentages:")

split_counts = (
    df["split"]
    .value_counts()
)

for split_name, count in split_counts.items():

    percentage = (
        count
        / len(df)
        * 100
    )

    print(
        f"{split_name}: "
        f"{count} rows "
        f"({percentage:.2f}%)"
    )


# ============================================================
# Final sort
# ============================================================

df = df.sort_values(
    "timestamp_utc"
).reset_index(drop=True)


# ============================================================
# Final validation
# ============================================================

print("\n" + "=" * 60)
print("FINAL FEATURE DATASET")
print("=" * 60)

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)

print(
    "Remaining missing values:",
    df.isna().sum().sum()
)

print(
    "Duplicate timestamps:",
    df["timestamp_utc"]
    .duplicated()
    .sum()
)

print(
    "First timestamp:",
    df["timestamp_utc"].min()
)

print(
    "Last timestamp:",
    df["timestamp_utc"].max()
)


# ============================================================
# Print final features
# ============================================================

print("\nFinal columns:")

for column in df.columns:
    print("-", column)


# ============================================================
# Save feature dataset
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)

print(
    "Saved feature dataset to:"
)

print(
    OUTPUT_FILE
)