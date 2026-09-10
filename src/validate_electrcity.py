import pandas as pd
from pathlib import Path

FILE_PATH = Path("data/processed/swpp_demand_2020_2025.csv")

# Load data
df = pd.read_csv(FILE_PATH)

# Convert timestamp and demand columns
df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
df["demand_mwh"] = pd.to_numeric(df["demand_mwh"], errors="coerce")

# Check duplicates
duplicates = df["timestamp_utc"].duplicated().sum()

print("Duplicate timestamps:", duplicates)
print("Missing demand values:", df["demand_mwh"].isna().sum())

# Create the complete expected hourly timeline
expected = pd.date_range(
    start="2020-01-01 00:00:00",
    end="2025-12-31 23:00:00",
    freq="h"
)

# Find missing timestamps
missing = expected.difference(df["timestamp_utc"])

print("Expected hours:", len(expected))
print("Actual rows before cleaning:", len(df))
print("Missing timestamps:")
print(missing)

# Reindex onto complete hourly timeline
df = (
    df.set_index("timestamp_utc")
      .reindex(expected)
)

df.index.name = "timestamp_utc"

# Interpolate missing demand values
df["demand_mwh"] = df["demand_mwh"].interpolate(method="time")

# Return timestamp to a normal column
df = df.reset_index()

# Save OVER the existing CSV
df.to_csv(FILE_PATH, index=False)

print()
print("Cleaning complete!")
print("Final rows:", len(df))
print("Remaining missing values:", df["demand_mwh"].isna().sum())
print("Saved to:", FILE_PATH)