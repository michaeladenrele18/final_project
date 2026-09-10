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
    "Missing timestamps in complete demand history:",
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
# Keep only timestamp + engineered demand features
# ============================================================

demand_features = demand_history[
    [
        "timestamp_utc",
        *demand_feature_columns
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

rows_before_filtering = len(df)

df = df.dropna(
    subset=demand_feature_columns
).copy()

rows_after_filtering = len(df)

print(
    "Rows before filtering:",
    rows_before_filtering
)

print(
    "Rows removed:",
    rows_before_filtering - rows_after_filtering
)

print(
    "Rows remaining:",
    rows_after_filtering
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