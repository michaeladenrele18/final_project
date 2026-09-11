# SPP Electricity Demand Forecasting

An end-to-end data engineering and machine learning project for forecasting hourly electricity demand across the Southwest Power Pool (SPP) region using historical electricity demand, weather observations, temporal features, and lagged demand.

The project combines **electrical power systems, data engineering, time-series forecasting, and machine learning** to explore how historical grid behavior and environmental conditions can be used to predict future electricity consumption.

---

# Project Overview

Electricity demand changes continuously based on factors such as:

- Time of day
- Day of the week
- Seasonal patterns
- Temperature
- Humidity
- Weather conditions
- Recent electricity consumption
- Daily and weekly demand cycles

Accurate demand forecasting helps power-system operators plan generation, maintain grid reliability, and balance electricity supply and demand.

This project builds a reproducible pipeline that:

1. Extracts historical electricity demand.
2. Processes hourly weather data.
3. Detects and handles data-quality issues.
4. Combines electricity and weather observations.
5. Engineers time-series forecasting features.
6. Splits data chronologically.
7. Creates realistic rolling forecasting baselines.
8. Trains machine-learning models.
9. Evaluates models using consistent metrics.
10. Investigates unexpected model behavior to identify upstream data-quality problems.

---

# Architecture

```text
                         DATA SOURCES
                              │
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
         U.S. EIA Data                  NOAA GHCNh
         SPP Demand                     Weather Data
               │                             │
               ▼                             ▼
       Demand Extraction              Weather Processing
               │                             │
               ▼                             ▼
       Demand Validation              Weather Validation
               │                             │
               ▼                             ▼
     Range / Quality Checks         Range / Quality Checks
               │                             │
               ▼                             ▼
     Missing-Value Handling        Short-Gap Interpolation
               │                             │
               └──────────────┬──────────────┘
                              │
                              ▼
                       Dataset Builder
                              │
                              ▼
                    Modeling Dataset
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
       Complete Demand History       Weather + Demand
                │                           │
                ▼                           │
       Lag / Rolling Features               │
                │                           │
                └─────────────┬─────────────┘
                              │
                              ▼
                    Feature Engineering
                              │
                              ▼
                 Chronological Splitting
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
             TRAIN       VALIDATION        TEST
           2020-2023        2024           2025
                │
                ▼
       Forecasting Baselines
                │
                ▼
       Machine Learning Models
                │
                ▼
         Model Evaluation
                │
                ▼
        Forecast Comparison
```

---

# Data Sources

## Electricity Demand

Historical electricity demand is sourced from the **U.S. Energy Information Administration (EIA)** electric system operating data.

The project uses hourly electricity demand for:

**Southwest Power Pool (SWPP)**

Series:

```text
EBA.SWPP-ALL.D.H
```

Time period:

```text
January 1, 2020
through
December 31, 2025
```

All timestamps are processed in:

```text
UTC
```

After extraction and validation, the complete demand dataset contains:

```text
52,608 hourly observations
```

The validated demand timeline contains:

```text
Missing demand values: 0
Missing timestamps: 0
Duplicate timestamps: 0
```

---

# Electricity Demand Validation

Electricity-demand validation is performed by:

```text
src/validate_electricity.py
```

The script performs:

- Timestamp parsing
- Numeric conversion
- Duplicate detection
- Missing timestamp detection
- Reindexing onto the complete expected hourly timeline
- Missing-value interpolation
- Demand-range validation
- Final dataset validation

A project-level physical plausibility check is used to detect obviously corrupted demand measurements.

Values outside:

```text
0 MW to 100,000 MW
```

are treated as invalid and converted to missing values before interpolation.

This threshold is used as a project-level sanity check for gross data corruption rather than as an official EIA quality-control limit.

---

# Electricity Data Quality Issue Discovered

During Random Forest model development, the model produced an extreme forecast of approximately:

```text
966,023 MW
```

while normal SPP demand values were only in the tens of thousands of MW.

Further investigation showed that the training target contained:

```text
Timestamp: 2023-06-13 02:00:00
Demand:    3,621,097 MW
```

The surrounding demand values were:

```text
2023-06-12 23:00   33,604 MW
2023-06-13 00:00   33,639 MW
2023-06-13 01:00   33,011 MW
2023-06-13 02:00   3,621,097 MW
2023-06-13 03:00   31,086 MW
2023-06-13 04:00   29,565 MW
2023-06-13 05:00   27,647 MW
```

This clearly indicated a corrupted isolated observation.

The validation pipeline was updated to detect unrealistic demand values before interpolation.

The corrupted value was replaced with a missing value and interpolated between its neighboring hourly observations.

This issue demonstrated that model failures can expose data-quality problems that are not detected by checking only for:

```text
missing values
duplicate timestamps
missing timestamps
```

---

# Weather Data

Weather observations are sourced from NOAA's:

**Global Historical Climatology Network Hourly (GHCNh)**

Because the Southwest Power Pool spans a large geographic area, the project uses several representative weather stations rather than relying on a single location.

| Location | NOAA Station |
|---|---|
| Kansas City | USW00003947 |
| Wichita | USW00003928 |
| Oklahoma City | USW00013967 |
| Omaha | USW00014942 |
| Fargo | USW00014914 |

Weather features currently include:

```text
temperature
dew_point_temperature
relative_humidity
wind_speed
```

Weather measurements from each station are preserved separately in the final dataset.

For example:

```text
kc_temperature
wichita_temperature
oklahoma_temperature
omaha_temperature
fargo_temperature
```

This allows the machine-learning models to learn regional weather patterns across the SPP footprint instead of relying on one averaged weather value.

---

# Weather Processing

Raw NOAA GHCNh data can contain multiple observations within the same hour.

For example:

```text
00:00
00:53
01:00
01:53
```

The weather pipeline converts this data into one representative observation per hour.

For each station and year, the pipeline:

1. Parses NOAA timestamps.
2. Converts weather columns to numeric values.
3. Checks weather values for physical plausibility.
4. Floors timestamps into hourly intervals.
5. Calculates each observation's distance from minute `:53`.
6. Selects the observation closest to `:53`.
7. Normalizes the selected timestamp to the top of the hour.
8. Reindexes the dataset onto a complete hourly timeline.
9. Detects missing observations.
10. Interpolates only short gaps.
11. Preserves long gaps as missing.
12. Validates final timestamps and duplicate counts.

NOAA GHCNh timestamps are treated as:

```text
UTC
```

The timestamp normalization performed by the project is hourly binning, not a timezone conversion.

---

# Weather Range Validation

Initial weather validation only checked:

```text
missing observations
duplicate timestamps
missing timestamps
```

During Linear Regression testing, the model produced an impossible prediction of approximately:

```text
-2,997,113 MW
```

Investigation of the corresponding input features revealed corrupted Omaha weather observations, including values such as:

```text
Temperature:       -61.0 °C
Relative humidity: 31,287%
```

This showed that a dataset can contain:

```text
no missing values
```

while still containing invalid measurements.

Weather validation was therefore expanded to include project-level physical plausibility checks.

Current validation ranges are:

| Feature | Valid Range |
|---|---:|
| Temperature | -60°C to 60°C |
| Dew Point | -70°C to 40°C |
| Relative Humidity | 0% to 100% |
| Wind Speed | 0 to 75 m/s |

Values outside these ranges are converted to missing values before the existing interpolation logic is applied.

These ranges are practical project-level sanity bounds intended to catch gross data corruption. They are not official NOAA quality-control thresholds.

---

# Missing Weather Strategy

Small weather-data gaps are reasonable to interpolate.

However, interpolating across long multi-day gaps could create artificial weather measurements.

The project therefore uses:

```text
Maximum interpolation gap: 6 hours
```

Any missing sequence longer than six consecutive hours remains missing.

This prevents the weather preprocessing pipeline from creating unrealistic values over large reporting gaps.

Isolated invalid observations detected by the range-validation checks can therefore be repaired using the same short-gap interpolation process.

---

# NOAA Reporting Gap

A significant multi-day reporting gap was discovered across several weather stations in 2025.

The outage occurred approximately around:

```text
August 29, 2025
through
September 2, 2025
```

Because several stations experienced similar missing periods, these values are not blindly interpolated.

Instead, the affected observations remain missing until the final modeling dataset is created.

---

# Processed Weather Dataset Validation

Each processed station dataset contains the full expected hourly timeline:

```text
52,608 rows
```

with:

```text
Missing timestamps: 0
Duplicate timestamps: 0
```

Remaining missing weather rows after range validation and short-gap interpolation:

| Station | Rows With Missing Weather |
|---|---:|
| Kansas City | 91 |
| Wichita | 116 |
| Oklahoma City | 90 |
| Omaha | 103 |
| Fargo | 110 |

---

# Dataset Construction

The script:

```text
src/build_dataset.py
```

combines the processed electricity-demand dataset with all five weather datasets.

All datasets are joined using:

```text
timestamp_utc
```

The join is validated as:

```text
one-to-one
```

to prevent duplicate observations from silently appearing in the modeling dataset.

---

# Modeling Dataset

Before filtering incomplete weather observations:

```text
Rows: 52,608
Columns: 22
Duplicate timestamps: 0
Missing demand values: 0
```

The dataset contains:

```text
1 timestamp
1 electricity demand target
20 weather variables
```

for a total of:

```text
22 columns
```

---

## Missing Weather Filtering

After combining the five stations, the pipeline identifies every row containing at least one missing weather feature.

Results:

```text
Rows before filtering: 52,608
Rows removed: 150
Final modeling rows: 52,458
Remaining missing values: 0
Duplicate timestamps: 0
```

Only approximately:

```text
0.29%
```

of the original timeline is removed.

The final modeling dataset is saved as:

```text
data/processed/modeling_dataset.csv
```

---

# Modeling Dataset Features

The base modeling dataset contains:

```text
timestamp_utc
demand_mw

kc_temperature
kc_dew_point_temperature
kc_relative_humidity
kc_wind_speed

wichita_temperature
wichita_dew_point_temperature
wichita_relative_humidity
wichita_wind_speed

oklahoma_temperature
oklahoma_dew_point_temperature
oklahoma_relative_humidity
oklahoma_wind_speed

omaha_temperature
omaha_dew_point_temperature
omaha_relative_humidity
omaha_wind_speed

fargo_temperature
fargo_dew_point_temperature
fargo_relative_humidity
fargo_wind_speed
```

---

# Feature Engineering

Feature engineering is performed by:

```text
src/feature_engineering.py
```

The feature-engineering pipeline creates three major categories of features:

1. Calendar features
2. Lagged electricity-demand features
3. Rolling electricity-demand features

---

# Calendar Features

The following features are extracted from:

```text
timestamp_utc
```

## Hour

```text
hour
```

Values:

```text
0-23
```

This helps the model learn daily electricity-use patterns.

## Day of Week

```text
day_of_week
```

Values:

```text
0 = Monday
1 = Tuesday
2 = Wednesday
3 = Thursday
4 = Friday
5 = Saturday
6 = Sunday
```

## Month

```text
month
```

Values:

```text
1-12
```

This helps capture seasonal electricity-demand behavior.

## Weekend Indicator

```text
is_weekend
```

Values:

```text
0 = weekday
1 = weekend
```

This helps model differences between weekday and weekend electricity usage.

---

# Demand Lag Features

Historical electricity demand is one of the strongest predictors of future demand.

The pipeline creates:

```text
demand_lag_1h
demand_lag_24h
demand_lag_168h
```

These represent:

```text
demand_lag_1h
= electricity demand exactly 1 hour earlier

demand_lag_24h
= electricity demand exactly 24 hours earlier

demand_lag_168h
= electricity demand exactly 7 days earlier
```

---

# Timestamp-Aware Demand Engineering

The final modeling dataset has timestamps removed because of incomplete weather observations.

Because of this, directly calculating:

```python
df["demand_mw"].shift(24)
```

on the filtered modeling dataset would not always represent exactly 24 hours earlier.

For example:

```text
24 rows earlier
```

would not necessarily equal:

```text
24 clock-hours earlier
```

after a weather observation has been removed.

To prevent this issue, all demand-history features are calculated from the original complete electricity-demand dataset:

```text
swpp_demand_2020_2025.csv
```

which contains the uninterrupted:

```text
52,608-hour timeline
```

The engineered demand features are then merged back onto the cleaned modeling dataset using:

```text
timestamp_utc
```

This ensures that:

```text
demand_lag_24h
```

actually represents electricity demand exactly 24 hours earlier.

---

# Rolling Demand Features

The pipeline also creates rolling historical demand averages:

```text
demand_rolling_24h
demand_rolling_168h
```

These represent average electricity demand over:

```text
Previous 24 hours
Previous 168 hours
```

---

## Preventing Target Leakage

Rolling averages are shifted before the rolling calculation.

Example:

```python
demand_history["demand_rolling_24h"] = (
    demand_history["demand_mw"]
    .shift(1)
    .rolling(window=24)
    .mean()
)
```

The current target value is therefore never included in its own rolling feature.

This prevents direct:

```text
target leakage
```

during feature construction.

---

# Final Feature Dataset

After feature engineering:

```text
Initial modeling rows: 52,458
Rows removed for insufficient demand history: 168
Final rows: 52,290
Final columns: 32
Missing values: 0
Duplicate timestamps: 0
```

The first usable observation is:

```text
2020-01-08 00:00:00
```

because the 168-hour weekly lag requires one complete week of previous demand history.

The final timestamp is:

```text
2025-12-31 23:00:00
```

The final dataset is saved as:

```text
data/processed/feature_dataset.csv
```

---

# Final Features

The full feature dataset contains:

```text
timestamp_utc
demand_mw

kc_temperature
kc_dew_point_temperature
kc_relative_humidity
kc_wind_speed

wichita_temperature
wichita_dew_point_temperature
wichita_relative_humidity
wichita_wind_speed

oklahoma_temperature
oklahoma_dew_point_temperature
oklahoma_relative_humidity
oklahoma_wind_speed

omaha_temperature
omaha_dew_point_temperature
omaha_relative_humidity
omaha_wind_speed

fargo_temperature
fargo_dew_point_temperature
fargo_relative_humidity
fargo_wind_speed

hour
day_of_week
month
is_weekend

demand_lag_1h
demand_lag_24h
demand_lag_168h

demand_rolling_24h
demand_rolling_168h

split
```

---

# Train / Validation / Test Strategy

Because electricity demand is a time series, the project does not use a random train/test split.

Instead, observations are split chronologically.

```text
2020 ─┐
2021  │
2022  ├── TRAIN
2023 ─┘

2024 ─── VALIDATION

2025 ─── TEST
```

Final feature-dataset split sizes:

```text
Train:       34,885
Validation:   8,775
Test:         8,630
```

For the complete electricity-demand series used by the forecasting baselines:

```text
Training:    35,064 hours
Validation:   8,784 hours
Test:         8,760 hours
```

The difference occurs because the feature dataset removes observations with insufficient lag history or incomplete weather features, while the baseline models operate on the complete validated electricity-demand timeline.

---

# Why Chronological Splitting Matters

A random train/test split could allow the model to train on future electricity-demand behavior while evaluating on older observations.

That would not represent a realistic forecasting environment.

The chronological split ensures:

```text
past data → predicts future data
```

which better reflects how electricity-demand forecasting systems operate in practice.

The 2025 dataset remains untouched during model development and model selection.

---

# Forecasting Objective

The primary forecasting objective is:

```text
24-hour-ahead hourly electricity-demand forecasting
```

The goal is to predict the next 24 hourly electricity-demand values.

Conceptually:

```text
Historical Demand + Weather + Time Features
                    │
                    ▼
           Forecast Next 24 Hours
```

This resembles a day-ahead forecasting problem used in electric-grid operations.

---

# Rolling Forecast Evaluation

A single Seasonal Naive forecast from the end of 2023 through all of 2024 would not represent a realistic day-ahead forecasting system.

For example, a forecast for July 2024 should be allowed to use actual observations from earlier in 2024.

The baseline models therefore use:

```text
rolling-origin evaluation
```

also known as:

```text
walk-forward validation
```

The baseline evaluation process behaves like:

```text
Data available through Dec 31, 2023
                │
                ▼
       Forecast Jan 1, 2024
          Next 24 hours
                │
                ▼
       Observe Jan 1 demand
                │
                ▼
Data available through Jan 1, 2024
                │
                ▼
       Forecast Jan 2, 2024
          Next 24 hours
                │
                ▼
              ...
                │
                ▼
       Forecast Dec 31, 2024
```

The baseline validation configuration is:

```text
Forecast horizon:   24 hours
Step size:          24 hours
Forecast windows:   366
Total predictions:  8,784
```

Because 2024 is a leap year:

```text
366 days × 24 hours = 8,784 predictions
```

---

# Repository Structure

```text
.
├── data/
│   ├── raw/
│   │   ├── electricity/
│   │   └── weather/
│   │
│   └── processed/
│       ├── weather/
│       ├── swpp_demand_2020_2025.csv
│       ├── modeling_dataset.csv
│       └── feature_dataset.csv
│
├── src/
│   ├── extract_swpp_demand.py
│   ├── validate_electricity.py
│   ├── validate_weather.py
│   ├── check_weather_gaps.py
│   ├── build_dataset.py
│   ├── feature_engineering.py
│   ├── train_baseline.py
│   ├── train_linear_regression.py
│   └── train_random_forest.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

Large raw and generated processed datasets are excluded from Git version control.

---

# Pipeline Components

## 1. Electricity Extraction

```text
src/extract_swpp_demand.py
```

Extracts hourly SPP electricity demand from the EIA bulk electricity dataset.

```text
EBA.txt
    │
    ▼
Extract SWPP Demand
    │
    ▼
SWPP Hourly Demand
```

---

## 2. Electricity Validation

```text
src/validate_electricity.py
```

Validates and cleans the electricity-demand timeline.

Responsibilities include:

- Timestamp parsing
- Numeric conversion
- Missing timestamp detection
- Duplicate detection
- Hourly reindexing
- Unrealistic demand detection
- Missing demand handling
- Time interpolation
- Final timeline validation

Final output:

```text
52,608 continuous hourly demand observations
```

---

## 3. Weather Validation

```text
src/validate_weather.py
```

Processes and validates NOAA weather observations.

Responsibilities include:

- NOAA PSV parsing
- Numeric conversion
- Physical range checks
- Hourly observation selection
- Timestamp normalization
- Missing observation detection
- Short-gap interpolation
- Long-gap preservation
- Station-level validation

---

## 4. Weather Gap Analysis

```text
src/check_weather_gaps.py
```

Analyzes missing weather periods.

The script reports:

```text
Total missing hours
Longest consecutive gap
Gap start time
Gap end time
```

This allows long reporting gaps to be detected before interpolation.

---

## 5. Dataset Construction

```text
src/build_dataset.py
```

Combines:

```text
SPP electricity demand
+
Kansas City weather
+
Wichita weather
+
Oklahoma City weather
+
Omaha weather
+
Fargo weather
```

using:

```text
timestamp_utc
```

Responsibilities include:

- Loading processed data
- Renaming weather features
- One-to-one joins
- Row-count validation
- Missing-weather analysis
- Filtering incomplete observations
- Saving the modeling dataset

Output:

```text
data/processed/modeling_dataset.csv
```

---

## 6. Feature Engineering

```text
src/feature_engineering.py
```

Creates:

```text
Calendar features
Demand lag features
Rolling demand features
Chronological split labels
```

Demand-history features are calculated from the complete demand timeline before being merged onto the modeling dataset.

Output:

```text
data/processed/feature_dataset.csv
```

---

## 7. Baseline Forecasting

```text
src/train_baseline.py
```

Creates and evaluates Seasonal Naive electricity-demand forecasts.

The baseline pipeline:

1. Loads the complete electricity-demand history.
2. Converts the data into StatsForecast format.
3. Separates training, validation, and test periods.
4. Performs rolling 24-hour forecasting across 2024.
5. Compares predictions against actual demand.
6. Calculates MAE, RMSE, and MAPE.

StatsForecast uses the following column structure:

```text
unique_id
ds
y
```

where:

```text
unique_id = SWPP
ds        = timestamp
y         = actual electricity demand
```

---

## 8. Linear Regression

```text
src/train_linear_regression.py
```

Trains a simple supervised-learning model using:

```text
Calendar features
24-hour demand lag
168-hour demand lag
Regional weather features
```

The model is trained using:

```text
2020-2023
```

and evaluated using:

```text
2024
```

The 2025 test set remains untouched.

---

## 9. Random Forest

```text
src/train_random_forest.py
```

Trains a nonlinear tree-based model using the same core feature set as Linear Regression.

The model is trained using:

```text
2020-2023
```

and evaluated using:

```text
2024
```

This creates a direct comparison between a simple linear model and a nonlinear ensemble model.

---

# Modeling Strategy

The project compares simple historical-demand baselines against increasingly capable machine-learning models.

Current modeling progression:

```text
Seasonal Naive Baselines
          │
          ▼
   Linear Regression
          │
          ▼
     Random Forest
          │
          ▼
       XGBoost
          │
          ▼
   Model Comparison
          │
          ▼
     Final 2025 Test
```

The purpose of the baseline models is to establish a meaningful performance threshold.

A more complex model should provide measurable improvement over simply using recent historical electricity demand.

---

# Baseline Models

Two Seasonal Naive forecasting baselines are evaluated.

These models are implemented using:

```text
StatsForecast
```

---

## 24-Hour Seasonal Naive

The daily Seasonal Naive model uses:

```python
SeasonalNaive(season_length=24)
```

For each future hour, the prediction is based on the demand observed at the same hour one day earlier.

Conceptually:

```text
Forecast at time t
=
Demand at time t - 24 hours
```

Example:

```text
Forecast:
Tuesday 3:00 PM

Uses:
Monday 3:00 PM demand
```

The model is evaluated using a:

```text
24-hour forecast horizon
```

and the forecasting origin moves forward one day after every prediction window.

---

## 168-Hour Seasonal Naive

The weekly Seasonal Naive model uses:

```python
SeasonalNaive(season_length=168)
```

because:

```text
24 hours × 7 days = 168 hours
```

For each future hour, the prediction is based on the corresponding hour from one week earlier.

Conceptually:

```text
Forecast at time t
=
Demand at time t - 168 hours
```

The value `168` represents the seasonal lookback period, not the forecasting horizon.

---

# Baseline Evaluation Results

Both Seasonal Naive models were evaluated across the full 2024 validation period using rolling 24-hour forecasting.

The evaluation contains:

```text
366 forecast windows
24 predictions per window
8,784 total predictions
```

Results:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| **Seasonal Naive - 24 Hour** | **1,457.18 MW** | **2,003.98 MW** | **4.34%** |
| Seasonal Naive - 168 Hour | 2,823.37 MW | 3,865.90 MW | 8.27% |

---

# Baseline Interpretation

The 24-hour Seasonal Naive baseline substantially outperformed the 168-hour weekly baseline.

## Daily Seasonal Naive

```text
MAE:  1,457.18 MW
RMSE: 2,003.98 MW
MAPE: 4.34%
```

A MAPE of:

```text
4.34%
```

means the daily Seasonal Naive forecast differs from actual electricity demand by approximately:

```text
4.34% on average
```

across the 2024 validation period.

The MAE indicates that the model's hourly forecast is typically off by approximately:

```text
1,457 MW
```

---

## Daily vs Weekly Demand Persistence

The results show that:

```text
Demand 24 hours earlier
```

is substantially more predictive of future SPP hourly demand than:

```text
Demand 168 hours earlier
```

for the 2024 validation period.

MAPE comparison:

```text
24-hour Seasonal Naive:  4.34%
168-hour Seasonal Naive: 8.27%
```

This establishes the daily Seasonal Naive model as the project's:

```text
PRIMARY BASELINE
```

---

# Linear Regression

Linear Regression serves as the first supervised machine-learning model.

The model uses:

```text
hour
day_of_week
month
is_weekend

demand_lag_24h
demand_lag_168h

20 regional weather features
```

For the first day-ahead machine-learning comparison, the following engineered features are intentionally excluded:

```text
demand_lag_1h
demand_rolling_24h
demand_rolling_168h
```

because their availability requires additional care when forecasting all 24 future hours at once.

---

## Linear Regression Results

After electricity and weather data-quality corrections:

```text
MAE:  3,791.94 MW
RMSE: 4,950.42 MW
MAPE: 10.93%
```

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Seasonal Naive - 24 Hour | **1,457.18 MW** | **2,003.98 MW** | **4.34%** |
| Linear Regression | 3,791.94 MW | 4,950.42 MW | 10.93% |

Linear Regression performs substantially worse than the daily Seasonal Naive baseline.

---

# Linear Regression Interpretation

The result demonstrates that adding more features does not automatically create a better forecasting model.

Electricity demand has nonlinear relationships with several variables.

Temperature is one example.

Demand can increase during:

```text
very cold weather
```

because of heating load, while also increasing during:

```text
very hot weather
```

because of cooling load.

A simple linear model cannot naturally represent this type of relationship without additional feature transformations.

Calendar variables also contain nonlinear and cyclical behavior.

For example:

```text
hour = 23
```

and:

```text
hour = 0
```

are adjacent in real time even though they appear numerically far apart.

The poor Linear Regression performance motivated testing a nonlinear tree-based model.

---

# Random Forest

Random Forest is the first nonlinear machine-learning model used in the project.

It is implemented using:

```text
scikit-learn
```

The model uses the same main inputs as the Linear Regression model:

```text
Calendar features
24-hour demand lag
168-hour demand lag
Regional weather observations
```

Using the same general feature set makes the comparison between models more meaningful.

---

# Random Forest Results

After correcting both weather and electricity-demand data-quality problems, Random Forest achieved:

```text
MAE:  1,137.73 MW
RMSE: 1,535.01 MW
MAPE: 3.36%
```

Compared with the primary 24-hour Seasonal Naive baseline:

```text
Seasonal Naive

MAE:  1,457.18 MW
RMSE: 2,003.98 MW
MAPE: 4.34%
```

Random Forest improves all three evaluation metrics.

---

# Random Forest Improvement Over Baseline

Approximate improvement compared with the 24-hour Seasonal Naive baseline:

```text
MAE improvement:  ~21.9%
RMSE improvement: ~23.4%
MAPE improvement: ~22.6%
```

This means the Random Forest model successfully demonstrates that:

```text
Weather
+
Calendar information
+
Historical demand
+
Nonlinear machine learning
```

can outperform a strong previous-day persistence forecast on the 2024 validation period.

---

# Random Forest Error Analysis

The largest absolute Random Forest validation error occurred at:

```text
Timestamp: 2024-07-20 05:00:00 UTC
```

Prediction:

```text
26,252.86 MW
```

Actual demand:

```text
43,551.00 MW
```

Absolute error:

```text
17,298.14 MW
```

Relevant historical demand features were:

```text
demand_lag_24h:  24,991 MW
demand_lag_168h: 39,485 MW
```

Demand around the timestamp was:

```text
2024-07-20 03:00   43,575 MW
2024-07-20 04:00   43,575 MW
2024-07-20 05:00   43,551 MW
2024-07-20 06:00   32,377 MW
2024-07-20 07:00   30,988 MW
```

The model substantially underestimated this unusual high-demand period.

The 24-hour lag was especially low relative to the target:

```text
24,991 MW vs 43,551 MW
```

which may have contributed to the underprediction.

This demonstrates that even the best current model can struggle when current demand behavior differs sharply from recent daily patterns.

---

# Current Model Comparison

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| **Random Forest** | **1,137.73 MW** | **1,535.01 MW** | **3.36%** |
| Seasonal Naive - 24 Hour | 1,457.18 MW | 2,003.98 MW | 4.34% |
| Seasonal Naive - 168 Hour | 2,823.37 MW | 3,865.90 MW | 8.27% |
| Linear Regression | 3,791.94 MW | 4,950.42 MW | 10.93% |
| XGBoost | TBD | TBD | TBD |

Current strongest validation model:

```text
Random Forest
```

The 2025 test dataset remains untouched.

---

# Evaluation Metrics

Forecast performance is measured using three metrics.

## MAE

Mean Absolute Error

```text
MAE = average absolute prediction error
```

Mathematically:

```text
MAE = mean(|actual - predicted|)
```

MAE provides an intuitive measure of the typical hourly demand-forecast error.

For example:

```text
Random Forest MAE = 1,137.73 MW
```

means the prediction differs from actual demand by approximately:

```text
1,138 MW on average
```

---

## RMSE

Root Mean Squared Error

```text
RMSE = square root of mean squared prediction error
```

RMSE penalizes large forecasting errors more heavily than MAE.

A significantly higher RMSE than MAE can indicate that the model occasionally produces large forecasting misses.

---

## MAPE

Mean Absolute Percentage Error

```text
MAPE = average absolute percentage prediction error
```

Conceptually:

```text
MAPE
=
mean(
    |actual - predicted|
    --------------------
           actual
) × 100
```

For example:

```text
Random Forest MAPE = 3.36%
```

means the model differs from actual demand by approximately:

```text
3.36% on average
```

across the validation observations.

---

# Challenges and Observations

This project has revealed several important challenges associated with working with real-world energy data.

---

## 1. Missing Values Are Not the Only Data-Quality Problem

Initial validation focused on:

```text
missing timestamps
duplicate timestamps
missing values
```

However, both the electricity and weather datasets contained values that were technically present but clearly unrealistic.

Examples included:

```text
Relative humidity: 31,287%
Electricity demand: 3,621,097 MW
```

Neither value would have been detected by a simple missing-value check.

This led to adding physical plausibility checks to both validation pipelines.

### Observation

Reliable machine-learning pipelines need to validate:

```text
whether a value exists
```

and:

```text
whether the value makes sense
```

---

## 2. Model Errors Helped Identify Data Problems

Two major data-quality problems were discovered because machine-learning models produced extreme predictions.

Linear Regression initially produced approximately:

```text
-2,997,113 MW
```

which led to discovering corrupted NOAA weather values.

Random Forest later produced approximately:

```text
966,023 MW
```

which led to discovering the corrupted EIA demand observation.

### Observation

Unexpected model behavior should not always be treated as a model problem.

It can also indicate:

```text
bad training data
incorrect feature engineering
data leakage
pipeline errors
```

Model diagnostics became an additional form of data-quality monitoring.

---

## 3. Long Weather Gaps Should Not Be Blindly Interpolated

Some NOAA weather stations contained multi-day reporting gaps.

Interpolating across these periods would create artificial weather observations that were never measured.

The pipeline therefore distinguishes between:

```text
short gaps
```

and:

```text
long gaps
```

using a maximum interpolation length of six hours.

### Observation

Missing-value handling should depend on the length and meaning of the missing period rather than applying one interpolation rule to every gap.

---

## 4. Row-Based Time-Series Shifts Can Be Incorrect

The modeling dataset removes some timestamps because of missing weather observations.

This means:

```python
df.shift(24)
```

does not necessarily represent:

```text
exactly 24 clock-hours earlier
```

if calculated after rows have been removed.

### Observation

Time-series feature engineering should respect timestamps rather than assuming every remaining row is separated by exactly one hour.

This motivated calculating lag features from the complete demand timeline and merging them back by timestamp.

---

## 5. A Strong Simple Baseline Is Difficult to Beat

The 24-hour Seasonal Naive model achieved:

```text
MAPE: 4.34%
```

without using:

```text
weather
machine learning
feature engineering
```

It simply uses demand from the previous day.

Linear Regression achieved:

```text
MAPE: 10.93%
```

and performed significantly worse.

### Observation

A more complicated model is not automatically better.

Strong forecasting projects need meaningful baselines so that model complexity is justified by measurable improvement.

---

## 6. Nonlinear Models Fit the Problem Better

Random Forest reduced validation MAPE to:

```text
3.36%
```

compared with:

```text
4.34%
```

for the daily Seasonal Naive model and:

```text
10.93%
```

for Linear Regression.

### Observation

Electricity demand appears to contain nonlinear interactions among:

```text
weather
time
seasonality
historical demand
```

Tree-based models are better suited to learning these patterns than the initial untransformed Linear Regression model.

---

## 7. Daily Demand Is More Predictive Than Weekly Demand

The baseline results were:

```text
24-hour Seasonal Naive MAPE:  4.34%
168-hour Seasonal Naive MAPE: 8.27%
```

### Observation

For this dataset and validation period, demand from the previous day contains much stronger predictive information than demand from the same hour one week earlier.

---

## 8. Forecast Errors Can Still Be Large During Unusual Periods

Random Forest achieved strong average performance but still produced an error of more than:

```text
17,000 MW
```

during its worst validation observation.

### Observation

Average metrics alone do not fully describe model behavior.

Future analysis should examine:

```text
peak-demand periods
extreme weather
rapid demand transitions
seasonal changes
large-error timestamps
```

---

## 9. Forecast Feature Availability Matters

Features can be valid historically but unavailable when generating a real future forecast.

For example:

```text
demand_lag_1h
```

is known for the immediate next hour, but not necessarily for hour 24 of a forecast generated all at once.

Similarly, observed future weather is not available in production.

### Observation

A feature should not only be predictive.

It must also be:

```text
available at forecast time
```

This is why the first supervised day-ahead models exclude the one-hour lag and rolling-demand features.

---

## 10. UTC Simplifies Integration but May Hide Local Behavior

Using UTC creates a consistent timeline across EIA and NOAA datasets.

However, electricity usage is closely tied to human behavior and local clock time.

### Observation

Future models may benefit from adding local-time or regional-time calendar features while preserving UTC as the system's canonical timestamp.

---

# Running the Project

Clone the repository:

```bash
git clone <repository-url>
```

Move into the project directory:

```bash
cd final_project
```

---

## Create Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

Core machine-learning and forecasting dependencies include:

```bash
pip install pandas numpy scikit-learn statsforecast xgboost
```

---

# Run the Data Pipeline

The pipeline should be executed in the following order.

## 1. Extract Electricity Demand

```bash
python src/extract_swpp_demand.py
```

## 2. Validate Electricity Demand

```bash
python src/validate_electricity.py
```

## 3. Process and Validate Weather

```bash
python src/validate_weather.py
```

## 4. Optional Weather Gap Analysis

```bash
python src/check_weather_gaps.py
```

## 5. Build Modeling Dataset

```bash
python src/build_dataset.py
```

## 6. Engineer Features

```bash
python src/feature_engineering.py
```

## 7. Train and Evaluate Baselines

```bash
python src/train_baseline.py
```

## 8. Train Linear Regression

```bash
python src/train_linear_regression.py
```

## 9. Train Random Forest

```bash
python src/train_random_forest.py
```

---

# Data Storage

Large datasets are intentionally excluded from GitHub.

The repository primarily contains:

```text
source code
pipeline logic
configuration
documentation
```

rather than large raw or generated processed data files.

Raw datasets must be downloaded from the corresponding EIA and NOAA sources before executing the full pipeline.

---

# Technology Stack

## Data Engineering

- Python
- Pandas
- NumPy
- CSV processing
- PSV processing
- ETL pipelines
- Data validation
- Physical range checks
- Time-series processing
- Missing-data analysis
- Timestamp-aware feature engineering

## Time-Series Forecasting

- StatsForecast
- Seasonal Naive forecasting
- Rolling-origin evaluation
- Walk-forward validation
- Daily seasonality
- Weekly seasonality

## Machine Learning

- Scikit-learn
- Linear Regression
- Random Forest
- XGBoost

## Model Evaluation

- MAE
- RMSE
- MAPE
- Chronological validation
- Error analysis
- Outlier investigation

## Data Sources

- U.S. Energy Information Administration
- NOAA Global Historical Climatology Network Hourly

## Development

- Git
- GitHub
- Python virtual environments

---

# Current Project Status

## Completed

```text
✓ EIA electricity-demand extraction

✓ Demand timestamp validation

✓ Demand range validation

✓ Corrupted demand-value detection

✓ Continuous 52,608-hour demand timeline

✓ NOAA weather ingestion

✓ NOAA station selection

✓ Hourly weather normalization

✓ Weather physical-range validation

✓ Missing-weather gap detection

✓ Short-gap interpolation

✓ Long-gap preservation

✓ Multi-station weather processing

✓ Electricity + weather dataset integration

✓ Modeling dataset validation

✓ Calendar feature engineering

✓ Timestamp-aware demand lag features

✓ Rolling demand features

✓ Target-leakage prevention

✓ Chronological train / validation / test splitting

✓ 24-hour-ahead forecasting objective defined

✓ Rolling-origin baseline validation

✓ StatsForecast integration

✓ 24-hour Seasonal Naive baseline

✓ 168-hour Seasonal Naive baseline

✓ Linear Regression model

✓ Random Forest model

✓ MAE evaluation

✓ RMSE evaluation

✓ MAPE evaluation

✓ Model error investigation

✓ Baseline performance comparison

✓ Random Forest outperforming primary baseline
```

---

# Current Results

## 24-Hour Seasonal Naive

```text
MAE:  1,457.18 MW
RMSE: 2,003.98 MW
MAPE: 4.34%
```

## 168-Hour Seasonal Naive

```text
MAE:  2,823.37 MW
RMSE: 3,865.90 MW
MAPE: 8.27%
```

## Linear Regression

```text
MAE:  3,791.94 MW
RMSE: 4,950.42 MW
MAPE: 10.93%
```

## Random Forest

```text
MAE:  1,137.73 MW
RMSE: 1,535.01 MW
MAPE: 3.36%
```

Current best validation model:

```text
Random Forest
```

---

# Next Steps

```text
1. Train XGBoost

2. Evaluate XGBoost on 2024 validation data

3. Compare XGBoost against Random Forest

4. Investigate feature importance

5. Analyze errors by hour of day

6. Analyze errors by month and season

7. Visualize actual vs predicted demand

8. Evaluate model behavior during peak-demand periods

9. Consider model hyperparameter tuning

10. Select the strongest validation model

11. Lock model selection

12. Evaluate the final selected model once on the untouched 2025 test set

13. Compare final performance with the Seasonal Naive baseline

14. Document final model results
```

---

# Future Architecture

The current implementation uses local files so the project can remain manageable during initial development.

A future data-engineering version could use:

```text
               EIA
                │
                │
               NOAA
                │
                ▼
       Automated Ingestion
                │
                ▼
             AWS S3
        Raw Data Storage
                │
                ▼
        ETL / Validation
                │
                ▼
           Processed S3
                │
                ▼
            Snowflake
                │
                ▼
       Feature Engineering
                │
                ▼
        ML Model Training
                │
                ▼
         Forecast Service
                │
           ┌────┴────┐
           ▼         ▼
          API     Dashboard
```

---

# Future Data Engineering Improvements

Potential future improvements include:

- Automated EIA ingestion
- Automated NOAA ingestion
- AWS S3 raw-data storage
- Snowflake data warehousing
- ETL orchestration
- Pipeline scheduling
- Automated data-quality monitoring
- Statistical anomaly detection
- Data lineage
- Schema validation
- Automated data refreshes
- Model experiment tracking
- Automated retraining
- Model versioning
- Forecast APIs
- Interactive dashboards
- Cloud deployment
- Infrastructure as code

---

# Future Forecasting Improvements

Potential forecasting improvements include:

- XGBoost
- Gradient-boosted trees
- Hyperparameter tuning
- More SPP weather stations
- Additional weather variables
- Cyclical hour features
- Cyclical day-of-week features
- Cyclical month features
- Holiday features
- Extreme-weather indicators
- Heating-degree-day features
- Cooling-degree-day features
- Regional weather aggregates
- Temperature-demand interaction features
- Additional lag periods
- Forecast-safe rolling features
- Forecast uncertainty intervals
- Peak-demand prediction
- Seasonal error analysis

---

# Forecasting Caveats

The current supervised machine-learning models use observed historical weather values during validation.

For a true production day-ahead forecasting system, future observed weather would not yet be known.

A production forecasting pipeline would therefore need to replace target-time observed weather with:

```text
weather forecasts
```

such as numerical weather prediction or forecast API data.

Conceptually:

```text
Historical Model Development
----------------------------

Actual Demand
+
Observed Weather
        │
        ▼
      Model


Production Forecasting
----------------------

Historical Demand
+
Future Weather Forecast
        │
        ▼
      Model
        │
        ▼
Next 24 Hours of Demand
```

The current weather inputs should therefore be interpreted as:

```text
historical observed-weather proxy features
```

rather than a complete production day-ahead weather pipeline.

---

# Forecast Horizon Considerations

Some engineered lag features require additional care in a real 24-hour-ahead forecasting environment.

For example:

```text
demand_lag_1h
```

is available when predicting the next immediate hour.

However, when forecasting all 24 future hours simultaneously, the actual demand one hour before later forecast horizons may not yet be known.

For that reason, the initial Linear Regression and Random Forest comparisons use:

```text
demand_lag_24h
demand_lag_168h
```

rather than:

```text
demand_lag_1h
```

The current rolling-demand features are also excluded from these first direct day-ahead model comparisons because later horizons could depend on demand observations that would not yet exist at the forecast cutoff.

Future versions may address this using:

- Horizon-specific models
- Recursive forecasting
- Direct multi-step forecasting
- Forecast-cutoff-aware rolling features
- Lag restrictions based on real feature availability

---

# UTC and Local-Time Considerations

The current pipeline uses:

```text
UTC
```

throughout the project.

This provides consistent timestamps across electricity and weather datasets.

However, electricity usage behavior is often strongly associated with local clock time.

The Southwest Power Pool spans multiple geographic regions and time zones.

A future improvement could therefore include:

```text
local-time calendar features
```

in addition to UTC-based features.

---

# Forecast Benchmarking

A future extension could compare the project's machine-learning forecasts against SPP's published day-ahead demand forecasts.

Potential comparison:

```text
Project ML Forecast
        vs
SPP Day-Ahead Forecast
        vs
Seasonal Naive Baseline
        vs
Actual Demand
```

The EIA bulk dataset also contains the SWPP day-ahead forecast series:

```text
EBA.SWPP-ALL.DF.H
```

which could provide a future industry-oriented benchmark.

---

# Model Comparison Framework

All forecasting models are evaluated using a common set of metrics.

Current validation results:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| **Random Forest** | **1,137.73 MW** | **1,535.01 MW** | **3.36%** |
| Seasonal Naive - 24 Hour | 1,457.18 MW | 2,003.98 MW | 4.34% |
| Seasonal Naive - 168 Hour | 2,823.37 MW | 3,865.90 MW | 8.27% |
| Linear Regression | 3,791.94 MW | 4,950.42 MW | 10.93% |
| XGBoost | TBD | TBD | TBD |

The 2025 test dataset will remain untouched until model selection is complete.

This prevents repeated model decisions from indirectly overfitting the final test period.

---

# Key Findings So Far

The project has produced several important findings:

```text
1. Previous-day demand is a very strong forecasting baseline.

2. Previous-week demand performs substantially worse than previous-day demand.

3. Basic Linear Regression does not capture the nonlinear structure of the problem well.

4. Random Forest outperforms the primary Seasonal Naive baseline.

5. Weather and electricity datasets can contain extreme corrupted values even when no data is missing.

6. Model diagnostics can help reveal upstream data-quality problems.

7. Timestamp-aware feature engineering is necessary after removing incomplete observations.

8. Short and long missing-data gaps should be treated differently.

9. Feature availability must be considered when designing a realistic forecasting model.

10. The 2025 test set should remain untouched until final model selection.
```

---

# Long-Term Goal

The long-term goal of this project is to evolve beyond a standalone machine-learning experiment into an:

**end-to-end energy data and forecasting platform**

that demonstrates how modern data-engineering and machine-learning techniques can be applied to real-world electrical power-system data.

The project is intended to demonstrate experience across:

```text
Electrical Power Systems
        +
Data Engineering
        +
Time-Series Forecasting
        +
Machine Learning
        +
Cloud Data Architecture
```

The project also provides a foundation for exploring career paths at the intersection of:

```text
Energy Systems
Power Grid Analytics
Data Engineering
Machine Learning
Cloud Engineering
Forecasting
```