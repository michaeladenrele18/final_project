import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Load the feature-engineered dataset
df = pd.read_csv("data/processed/feature_dataset.csv")


# Select features for the Linear Regression model
# demand_lag_1h is excluded because it would not be available
# for every hour in a true 24-hour-ahead forecast
features = [
    # Calendar features
    "hour",
    "day_of_week",
    "month",
    "is_weekend",

    # Historical demand features
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

    # Oklahoma City weather
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
    "fargo_wind_speed",
]


# Use 2020-2023 for training and 2024 for validation
# 2025 remains untouched for final testing
train = df[df["split"] == "train"]
validation = df[df["split"] == "validation"]


# Separate input features (X) from the target demand (y)
X_train = train[features]
y_train = train["demand_mw"]

X_val = validation[features]
y_val = validation["demand_mw"]


# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)


# Predict electricity demand for the 2024 validation data
predictions = model.predict(X_val)

# Find the worst prediction
worst_index = np.argmin(predictions)

print("\nWorst prediction:")
print(f"Timestamp: {validation.iloc[worst_index]['timestamp_utc']}")
print(f"Predicted: {predictions[worst_index]:.2f} MW")
print(f"Actual: {y_val.iloc[worst_index]:.2f} MW")

# Inspect the features used for the worst prediction
print("\nFeatures at worst prediction:")
print(validation.iloc[worst_index][features])

# Look at the surrounding rows
print("\nRows around worst prediction:")
print(
    validation.iloc[worst_index - 2 : worst_index + 3][
        ["timestamp_utc", "demand_mw"] + features
    ]
)

# Calculate forecasting error metrics
mae = mean_absolute_error(y_val, predictions)

rmse = np.sqrt(
    mean_squared_error(y_val, predictions)
)

mape = np.mean(
    np.abs((y_val - predictions) / y_val)
) * 100


# Display model performance
print("\n" + "=" * 60)
print("LINEAR REGRESSION PERFORMANCE")
print("=" * 60)

print(f"MAE:  {mae:.2f} MW")
print(f"RMSE: {rmse:.2f} MW")
print(f"MAPE: {mape:.2f}%")