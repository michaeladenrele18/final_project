import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

FILE_PATH = "data/processed/feature_dataset.csv"

df = pd.read_csv(FILE_PATH)

df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])

# ---------------------------------------------------------
# FEATURES
# ---------------------------------------------------------

features = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",

    "demand_lag_24h",
    "demand_lag_168h",

    "kc_temperature",
    "kc_dew_point_temperature",
    "kc_relative_humidity",
    "kc_wind_speed",

    "wichita_temperature",
    "wichita_dew_point_temperature",
    "wichita_relative_humidity",
    "wichita_wind_speed",

    "oklahoma_temperature",
    "oklahoma_dew_point_temperature",
    "oklahoma_relative_humidity",
    "oklahoma_wind_speed",

    "omaha_temperature",
    "omaha_dew_point_temperature",
    "omaha_relative_humidity",
    "omaha_wind_speed",

    "fargo_temperature",
    "fargo_dew_point_temperature",
    "fargo_relative_humidity",
    "fargo_wind_speed"
]

target = "demand_mw"

# ---------------------------------------------------------
# SPLIT DATA
# ---------------------------------------------------------

train = df[df["split"] == "train"].copy()
validation = df[df["split"] == "validation"].copy()

X_train = train[features]
y_train = train[target]

X_val = validation[features]
y_val = validation[target]

print("Training rows:", len(train))
print("Validation rows:", len(validation))

print("\nTraining target range:")
print("Min:", y_train.min())
print("Max:", y_train.max())

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

print("\nTraining XGBoost...")

model.fit(
    X_train,
    y_train
)

# ---------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------

predictions = model.predict(X_val)

# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

mae = mean_absolute_error(y_val, predictions)
rmse = np.sqrt(mean_squared_error(y_val, predictions))
mape = mean_absolute_percentage_error(y_val, predictions) * 100

print("\n" + "=" * 60)
print("XGBOOST PERFORMANCE")
print("=" * 60)

print(f"MAE:  {mae:.2f} MW")
print(f"RMSE: {rmse:.2f} MW")
print(f"MAPE: {mape:.2f}%")

# ---------------------------------------------------------
# WORST PREDICTION
# ---------------------------------------------------------

errors = np.abs(y_val.to_numpy() - predictions)

worst_index = np.argmax(errors)

worst_row = validation.iloc[worst_index]

print("\nWorst prediction:")

print("Timestamp:", worst_row["timestamp_utc"])
print(f"Predicted: {predictions[worst_index]:.2f} MW")
print(f"Actual: {y_val.iloc[worst_index]:.2f} MW")
print(f"Absolute Error: {errors[worst_index]:.2f} MW")

print("\nFeatures at worst prediction:")

print(
    worst_row[features]
)

# ---------------------------------------------------------
# FIRST 24-HOUR FORECAST
# ---------------------------------------------------------

forecast_24 = validation[
    ["timestamp_utc", "demand_mw"]
].copy()

forecast_24["predicted_demand_mw"] = predictions

forecast_24["error_mw"] = (
    forecast_24["demand_mw"]
    - forecast_24["predicted_demand_mw"]
)

forecast_24 = forecast_24.head(24)

print("\n" + "=" * 60)
print("FIRST 24-HOUR XGBOOST FORECAST")
print("=" * 60)

print(
    forecast_24[
        [
            "timestamp_utc",
            "demand_mw",
            "predicted_demand_mw",
            "error_mw"
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print("\n" + "=" * 60)
print("TOP 10 XGBOOST FEATURES")
print("=" * 60)

print(
    importance_df.to_string(index=False)
)

# ---------------------------------------------------------
# FEATURE IMPORTANCE BY CATEGORY
# ---------------------------------------------------------

categories = {
    "Demand History": [
        "demand_lag_24h",
        "demand_lag_168h"
    ],

    "Calendar": [
        "hour",
        "day_of_week",
        "month",
        "is_weekend"
    ],

    "Temperature": [
        "kc_temperature",
        "wichita_temperature",
        "oklahoma_temperature",
        "omaha_temperature",
        "fargo_temperature"
    ],

    "Humidity / Dew Point": [
        "kc_dew_point_temperature",
        "kc_relative_humidity",
        "wichita_dew_point_temperature",
        "wichita_relative_humidity",
        "oklahoma_dew_point_temperature",
        "oklahoma_relative_humidity",
        "omaha_dew_point_temperature",
        "omaha_relative_humidity",
        "fargo_dew_point_temperature",
        "fargo_relative_humidity"
    ],

    "Wind": [
        "kc_wind_speed",
        "wichita_wind_speed",
        "oklahoma_wind_speed",
        "omaha_wind_speed",
        "fargo_wind_speed"
    ]
}

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE BY CATEGORY")
print("=" * 60)

for category, category_features in categories.items():

    total_importance = importance_df[
        importance_df["feature"].isin(category_features)
    ]["importance"].sum()

    print(f"{category:<25} {total_importance:.4f}")

# ---------------------------------------------------------
# ERROR ANALYSIS
# ---------------------------------------------------------

error_analysis = validation[
    ["timestamp_utc", "demand_mw", "hour", "month"]
].copy()

error_analysis["prediction"] = predictions

error_analysis["absolute_error"] = np.abs(
    error_analysis["demand_mw"]
    - error_analysis["prediction"]
)

error_analysis["percentage_error"] = (
    error_analysis["absolute_error"]
    / error_analysis["demand_mw"]
) * 100


# ---------------------------------------------------------
# ERROR BY HOUR
# ---------------------------------------------------------

hourly_errors = (
    error_analysis
    .groupby("hour")
    .agg(
        MAE=("absolute_error", "mean"),
        MAPE=("percentage_error", "mean")
    )
)

print("\n" + "=" * 60)
print("XGBOOST ERROR BY HOUR")
print("=" * 60)

print(hourly_errors.round(2).to_string())


# ---------------------------------------------------------
# ERROR BY MONTH
# ---------------------------------------------------------

monthly_errors = (
    error_analysis
    .groupby("month")
    .agg(
        MAE=("absolute_error", "mean"),
        MAPE=("percentage_error", "mean")
    )
)

print("\n" + "=" * 60)
print("XGBOOST ERROR BY MONTH")
print("=" * 60)

print(monthly_errors.round(2).to_string())


# ---------------------------------------------------------
# BEST / WORST HOURS
# ---------------------------------------------------------

print("\nBest hour by MAE:")
print(hourly_errors["MAE"].idxmin())

print("Worst hour by MAE:")
print(hourly_errors["MAE"].idxmax())

print("\nBest month by MAE:")
print(monthly_errors["MAE"].idxmin())

print("Worst month by MAE:")
print(monthly_errors["MAE"].idxmax())

# ---------------------------------------------------------
# ERROR VS TEMPERATURE
# ---------------------------------------------------------

temperature_columns = [
    "kc_temperature",
    "wichita_temperature",
    "oklahoma_temperature",
    "omaha_temperature",
    "fargo_temperature"
]

error_analysis["avg_temperature"] = validation[
    temperature_columns
].mean(axis=1)

error_analysis["temperature_range"] = pd.cut(
    error_analysis["avg_temperature"],
    bins=[-np.inf, 0, 10, 20, 30, np.inf],
    labels=[
        "< 0 C",
        "0-10 C",
        "10-20 C",
        "20-30 C",
        "> 30 C"
    ]
)

temperature_errors = (
    error_analysis
    .groupby(
        "temperature_range",
        observed=True
    )
    .agg(
        observations=("absolute_error", "size"),
        MAE=("absolute_error", "mean"),
        MAPE=("percentage_error", "mean")
    )
)

print("\n" + "=" * 60)
print("XGBOOST ERROR BY REGIONAL TEMPERATURE")
print("=" * 60)

print(temperature_errors.round(2).to_string())