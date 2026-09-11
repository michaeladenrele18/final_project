from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

import pandas as pd
from pathlib import Path


# ============================================================
# Load complete SWPP demand data
# ============================================================

FILE_PATH = Path("data/processed/swpp_demand_2020_2025.csv")

df = pd.read_csv(FILE_PATH)

df["timestamp_utc"] = pd.to_datetime(
    df["timestamp_utc"]
)

df["demand_mwh"] = pd.to_numeric(
    df["demand_mwh"],
    errors="coerce"
)


# ============================================================
# Rename columns for StatsForecast
# ============================================================

df = df.rename(
    columns={
        "timestamp_utc": "ds",
        "demand_mwh": "y"
    }
)

df["unique_id"] = "SWPP"

df = df[
    [
        "unique_id",
        "ds",
        "y"
    ]
]


# ============================================================
# Create train / validation / test sets
# ============================================================

train_df = df[
    df["ds"].dt.year <= 2023
].copy()

validation_df = df[
    df["ds"].dt.year == 2024
].copy()

test_df = df[
    df["ds"].dt.year == 2025
].copy()


print("=" * 60)
print("DATA SPLITS")
print("=" * 60)

print("\nTRAINING")
print("Rows:", len(train_df))
print("Start:", train_df["ds"].min())
print("End:", train_df["ds"].max())

print("\nVALIDATION")
print("Rows:", len(validation_df))
print("Start:", validation_df["ds"].min())
print("End:", validation_df["ds"].max())

print("\nTEST")
print("Rows:", len(test_df))
print("Start:", test_df["ds"].min())
print("End:", test_df["ds"].max())


# ============================================================
# Combine training + validation for rolling evaluation
# ============================================================

cv_df = df[
    df["ds"].dt.year <= 2024
].copy()


# ============================================================
# Initialize 24-hour Seasonal Naive model
# ============================================================

model = SeasonalNaive(
    season_length=24
)

sf = StatsForecast(
    models=[model],
    freq="h"
)


# ============================================================
# Rolling cross-validation across 2024
# ============================================================

cv_results = sf.cross_validation(
    df=cv_df,
    h=24,
    step_size=24,
    n_windows=366
)


# ============================================================
# Inspect cross-validation output
# ============================================================

print("\n" + "=" * 60)
print("CROSS-VALIDATION RESULTS")
print("=" * 60)

print("\nRows:", len(cv_results))

print("\nFirst forecast window:")
print(
    cv_results.head(24)
)

print("\nLast forecast window:")
print(
    cv_results.tail(24)
)


# ============================================================
# Basic validation checks
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION CHECKS")
print("=" * 60)

print(
    "First forecast timestamp:",
    cv_results["ds"].min()
)

print(
    "Last forecast timestamp:",
    cv_results["ds"].max()
)

print(
    "Number of forecast windows:",
    cv_results["cutoff"].nunique()
)

print(
    "Total predictions:",
    len(cv_results)
)

# ============================================================
# Evaluate Seasonal Naive baseline
# ============================================================

actual = cv_results["y"]
predicted = cv_results["SeasonalNaive"]

mae = mean_absolute_error(
    actual,
    predicted
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predicted
    )
)

mape = np.mean(
    np.abs(
        (actual - predicted) / actual
    )
) * 100


print("\n" + "=" * 60)
print("SEASONAL NAIVE (24H) PERFORMANCE")
print("=" * 60)

print(f"MAE:  {mae:.2f} MW")
print(f"RMSE: {rmse:.2f} MW")
print(f"MAPE: {mape:.2f}%")