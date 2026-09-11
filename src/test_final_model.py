import pandas as pd
import numpy as np

from pathlib import Path

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# Configuration
# ============================================================

FEATURE_FILE = Path(
    "data/processed/feature_dataset.csv"
)


# ============================================================
# Load feature dataset
# ============================================================

print("=" * 60)
print("LOADING FEATURE DATASET")
print("=" * 60)

df = pd.read_csv(
    FEATURE_FILE
)

df["timestamp_utc"] = pd.to_datetime(
    df["timestamp_utc"]
)

df = df.sort_values(
    "timestamp_utc"
).reset_index(drop=True)

print(
    "Rows:",
    len(df)
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
# Features
# ============================================================

feature_columns = [

    # Calendar
    "hour",
    "day_of_week",
    "month",
    "is_weekend",

    # Demand history
    "demand_lag_24h",
    "demand_lag_168h",

    # Kansas City weather
    "kc_temperature",
    "kc_dew_point_temperature",
    "kc_relative_humidity",
    "kc_wind_speed",

    # Wichita weather
    "wichita_temperature",
    "wichita_dew_point_temperature",
    "wichita_relative_humidity",
    "wichita_wind_speed",

    # Oklahoma weather
    "oklahoma_temperature",
    "oklahoma_dew_point_temperature",
    "oklahoma_relative_humidity",
    "oklahoma_wind_speed",

    # Omaha weather
    "omaha_temperature",
    "omaha_dew_point_temperature",
    "omaha_relative_humidity",
    "omaha_wind_speed",

    # Fargo weather
    "fargo_temperature",
    "fargo_dew_point_temperature",
    "fargo_relative_humidity",
    "fargo_wind_speed"
]


TARGET = "demand_mw"


# ============================================================
# Verify required columns
# ============================================================

required_columns = (
    feature_columns
    + [
        TARGET,
        "timestamp_utc",
        "split"
    ]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns: "
        f"{missing_columns}"
    )


# ============================================================
# Final train/test split
# ============================================================

print("\n" + "=" * 60)
print("CREATING FINAL TRAIN / TEST SPLIT")
print("=" * 60)


# Train on everything that was previously
# training OR validation data.
#
# 2020-2024 -> final training
# 2025      -> final testing

train_df = df[
    df["split"].isin(
        [
            "train",
            "validation"
        ]
    )
].copy()

test_df = df[
    df["split"] == "test"
].copy()


print(
    "Final training rows:",
    len(train_df)
)

print(
    "Final testing rows:",
    len(test_df)
)

print(
    "Training period:",
    train_df["timestamp_utc"].min(),
    "to",
    train_df["timestamp_utc"].max()
)

print(
    "Testing period:",
    test_df["timestamp_utc"].min(),
    "to",
    test_df["timestamp_utc"].max()
)


# ============================================================
# Prepare X and y
# ============================================================

X_train = train_df[
    feature_columns
]

y_train = train_df[
    TARGET
]


X_test = test_df[
    feature_columns
]

y_test = test_df[
    TARGET
]


# ============================================================
# Final XGBoost model
# ============================================================

print("\n" + "=" * 60)
print("TRAINING FINAL XGBOOST MODEL")
print("=" * 60)


model = XGBRegressor(

    max_depth=4,

    learning_rate=0.03,

    n_estimators=300,

    subsample=1.0,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42,

    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


print(
    "Training complete."
)


# ============================================================
# Predict 2025
# ============================================================

predictions = model.predict(
    X_test
)


# ============================================================
# Evaluation metrics
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

mape = np.mean(
    np.abs(
        (
            y_test.values
            - predictions
        )
        / y_test.values
    )
) * 100


print("\n" + "=" * 60)
print("FINAL 2025 TEST PERFORMANCE")
print("=" * 60)

print(
    f"MAE:  {mae:.2f} MW"
)

print(
    f"RMSE: {rmse:.2f} MW"
)

print(
    f"MAPE: {mape:.2f}%"
)


# ============================================================
# Build results dataframe
# ============================================================

results = test_df[
    [
        "timestamp_utc",
        TARGET
    ]
].copy()

results[
    "predicted_demand_mw"
] = predictions

results[
    "error_mw"
] = (
    results[TARGET]
    - results["predicted_demand_mw"]
)

results[
    "absolute_error_mw"
] = (
    results["error_mw"]
    .abs()
)

results[
    "percentage_error"
] = (
    results["absolute_error_mw"]
    / results[TARGET]
    * 100
)


# ============================================================
# Worst prediction
# ============================================================

worst_index = (
    results["absolute_error_mw"]
    .idxmax()
)

worst_result = results.loc[
    worst_index
]


print("\n" + "=" * 60)
print("WORST 2025 PREDICTION")
print("=" * 60)

print(
    "Timestamp:",
    worst_result["timestamp_utc"]
)

print(
    f"Actual: "
    f"{worst_result[TARGET]:.2f} MW"
)

print(
    f"Predicted: "
    f"{worst_result['predicted_demand_mw']:.2f} MW"
)

print(
    f"Absolute Error: "
    f"{worst_result['absolute_error_mw']:.2f} MW"
)

print(
    f"Percentage Error: "
    f"{worst_result['percentage_error']:.2f}%"
)


# ============================================================
# Features at worst prediction
# ============================================================

worst_timestamp = pd.Timestamp(
    worst_result["timestamp_utc"]
)

worst_feature_row = test_df[
    test_df["timestamp_utc"]
    == worst_timestamp
].iloc[0]


print(
    "\nFeatures at worst prediction:"
)

print(
    worst_feature_row[
        feature_columns
    ]
)


# ============================================================
# Rows around worst prediction
# ============================================================

window_start = (
    worst_timestamp
    - pd.Timedelta(hours=3)
)

window_end = (
    worst_timestamp
    + pd.Timedelta(hours=3)
)

window = test_df[
    (
        test_df["timestamp_utc"]
        >= window_start
    )
    &
    (
        test_df["timestamp_utc"]
        <= window_end
    )
]


print(
    "\nRows around worst prediction:"
)

print(
    window[
        [
            "timestamp_utc",
            TARGET,
            "demand_lag_24h",
            "demand_lag_168h",
            "hour",
            "day_of_week",
            "month"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# First 24 predictions
# ============================================================

print("\n" + "=" * 60)
print("FIRST 24-HOUR 2025 FORECAST")
print("=" * 60)

print(
    results[
        [
            "timestamp_utc",
            TARGET,
            "predicted_demand_mw",
            "error_mw"
        ]
    ]
    .head(24)
    .to_string(
        index=False
    )
)


# ============================================================
# Feature importance
# ============================================================

feature_importance = pd.DataFrame(
    {
        "feature": feature_columns,

        "importance":
            model.feature_importances_
    }
)

feature_importance = (
    feature_importance
    .sort_values(
        "importance",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\n" + "=" * 60)
print("FINAL XGBOOST FEATURE IMPORTANCE")
print("=" * 60)

print(
    feature_importance
    .to_string(
        index=False
    )
)


# ============================================================
# Error by hour
# ============================================================

analysis_df = test_df[
    [
        "timestamp_utc",
        "hour",
        "month"
    ]
].copy()

analysis_df[
    "actual"
] = y_test.values

analysis_df[
    "prediction"
] = predictions

analysis_df[
    "absolute_error"
] = np.abs(
    analysis_df["actual"]
    - analysis_df["prediction"]
)

analysis_df[
    "percentage_error"
] = (
    analysis_df["absolute_error"]
    / analysis_df["actual"]
    * 100
)


hour_error = (
    analysis_df
    .groupby("hour")
    .agg(
        MAE=(
            "absolute_error",
            "mean"
        ),
        MAPE=(
            "percentage_error",
            "mean"
        )
    )
)


print("\n" + "=" * 60)
print("2025 ERROR BY HOUR")
print("=" * 60)

print(
    hour_error.round(2)
)


# ============================================================
# Error by month
# ============================================================

month_error = (
    analysis_df
    .groupby("month")
    .agg(
        MAE=(
            "absolute_error",
            "mean"
        ),
        MAPE=(
            "percentage_error",
            "mean"
        )
    )
)


print("\n" + "=" * 60)
print("2025 ERROR BY MONTH")
print("=" * 60)

print(
    month_error.round(2)
)


# ============================================================
# Save final predictions
# ============================================================

OUTPUT_FILE = Path(
    "data/processed/final_2025_predictions.csv"
)

results.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 60)
print("FINAL TEST COMPLETE")
print("=" * 60)

print(
    "Predictions saved to:",
    OUTPUT_FILE
)