import pandas as pd
from pathlib import Path


FILE_PATH = Path("data/processed/swpp_demand_2020_2025.csv")

MIN_INVALID_FLATLINE_HOURS = 12


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(FILE_PATH)

df["timestamp_utc"] = pd.to_datetime(
    df["timestamp_utc"]
)

df["demand_mwh"] = pd.to_numeric(
    df["demand_mwh"],
    errors="coerce"
)


# ---------------------------------------------------------
# RANGE VALIDATION
# ---------------------------------------------------------

invalid_demand = (
    (df["demand_mwh"] < 0)
    | (df["demand_mwh"] > 100000)
)

invalid_count = invalid_demand.sum()

if invalid_count > 0:
    print(
        "Invalid demand values replaced with NaN:",
        invalid_count
    )

df.loc[
    invalid_demand,
    "demand_mwh"
] = pd.NA


# ---------------------------------------------------------
# BASIC VALIDATION
# ---------------------------------------------------------

duplicates = (
    df["timestamp_utc"]
    .duplicated()
    .sum()
)

print(
    "Duplicate timestamps:",
    duplicates
)

print(
    "Missing demand values:",
    df["demand_mwh"]
    .isna()
    .sum()
)


# ---------------------------------------------------------
# BUILD COMPLETE HOURLY TIMELINE
# ---------------------------------------------------------

expected = pd.date_range(
    start="2020-01-01 00:00:00",
    end="2025-12-31 23:00:00",
    freq="h"
)

missing = expected.difference(
    df["timestamp_utc"]
)

print(
    "Expected hours:",
    len(expected)
)

print(
    "Actual rows before cleaning:",
    len(df)
)

print("Missing timestamps:")
print(missing)


# ---------------------------------------------------------
# REINDEX TO COMPLETE TIMELINE
# ---------------------------------------------------------

df = (
    df
    .set_index("timestamp_utc")
    .reindex(expected)
)

df.index.name = "timestamp_utc"


# ---------------------------------------------------------
# INTERPOLATE MISSING VALUES
# ---------------------------------------------------------

# This handles isolated missing timestamps / invalid values.
# Flatline values have not been flagged yet and are NOT
# modified by this interpolation step.

df["demand_mwh"] = (
    df["demand_mwh"]
    .interpolate(
        method="time",
        limit_area="inside"
    )
)


# ---------------------------------------------------------
# RETURN TIMESTAMP TO NORMAL COLUMN
# ---------------------------------------------------------

df = df.reset_index()


# ---------------------------------------------------------
# DETECT ALL SUSPICIOUS FLATLINES
# ---------------------------------------------------------

flatline_group = (
    df["demand_mwh"]
    .ne(
        df["demand_mwh"].shift()
    )
    .cumsum()
)

flatline_runs = (
    df
    .groupby(flatline_group)
    .agg(
        start_time=(
            "timestamp_utc",
            "first"
        ),
        end_time=(
            "timestamp_utc",
            "last"
        ),
        demand_mwh=(
            "demand_mwh",
            "first"
        ),
        consecutive_hours=(
            "demand_mwh",
            "size"
        )
    )
)

suspicious_flatlines = (
    flatline_runs[
        flatline_runs[
            "consecutive_hours"
        ] >= 6
    ]
)

print(
    "\nSuspicious demand flatlines "
    "(6+ hours):"
)

print(
    suspicious_flatlines
    .to_string(index=False)
)


# ---------------------------------------------------------
# FLAG INVALID FLATLINES
# ---------------------------------------------------------

df["demand_quality"] = "valid"

flatline_size = (
    df
    .groupby(flatline_group)[
        "demand_mwh"
    ]
    .transform("size")
)

invalid_flatline = (
    flatline_size
    >= MIN_INVALID_FLATLINE_HOURS
)

df.loc[
    invalid_flatline,
    "demand_quality"
] = "flatline"

flatline_count = (
    invalid_flatline.sum()
)

print(
    "\nDemand observations flagged "
    "as flatline:",
    flatline_count
)


# ---------------------------------------------------------
# QUALITY SUMMARY
# ---------------------------------------------------------

print(
    "\nDemand quality counts:"
)

print(
    df["demand_quality"]
    .value_counts()
)


# ---------------------------------------------------------
# QUALITY SUMMARY BY YEAR
# ---------------------------------------------------------

df["year"] = (
    df["timestamp_utc"]
    .dt.year
)

quality_by_year = (
    df
    .groupby(
        [
            "year",
            "demand_quality"
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)

print(
    "\nDemand quality by year:"
)

print(
    quality_by_year
    .to_string()
)

df = df.drop(
    columns="year"
)


# ---------------------------------------------------------
# FINAL VALIDATION
# ---------------------------------------------------------

print()

print("Cleaning complete!")

print(
    "Final rows:",
    len(df)
)

print(
    "Remaining missing values:",
    df["demand_mwh"]
    .isna()
    .sum()
)

print(
    "Duplicate timestamps:",
    df["timestamp_utc"]
    .duplicated()
    .sum()
)


# ---------------------------------------------------------
# SAVE CLEANED DATA
# ---------------------------------------------------------

df.to_csv(
    FILE_PATH,
    index=False
)

print(
    "Saved to:",
    FILE_PATH
)