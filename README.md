# SPP Electricity Demand Forecasting

An end-to-end data engineering and machine learning project for forecasting hourly electricity demand across the Southwest Power Pool (SPP) region using historical electricity demand, weather observations, temporal features, lagged demand, and rolling demand statistics.

The project combines **electrical power systems, data engineering, time-series forecasting, and machine learning** to explore how historical grid behavior and environmental conditions can be used to predict future electricity consumption.

---

## Project Overview

Electricity demand changes continuously based on factors such as:

- Time of day
- Day of the week
- Seasonal patterns
- Temperature
- Humidity
- Weather conditions
- Recent electricity consumption
- Daily and weekly demand cycles

Accurate demand forecasting helps power system operators plan generation, maintain grid reliability, and balance electricity supply and demand.

This project builds a reproducible pipeline that:

1. Extracts historical electricity demand.
2. Processes hourly weather data.
3. Detects and handles data-quality issues.
4. Combines electricity and weather observations.
5. Engineers time-series forecasting features.
6. Splits data chronologically.
7. Creates realistic rolling forecasting baselines.
8. Trains and evaluates machine-learning models.
9. Compares model performance using consistent forecasting metrics.

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
       Demand Validation              Gap Detection
                                             │
                                             ▼
                                  Short-Gap Interpolation
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
       Rolling Baseline Forecasts
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

The demand timeline contains:

```text
Missing demand values: 0
Missing timestamps: 0
Duplicate timestamps: 0
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

This allows machine-learning models to learn regional weather patterns across the SPP footprint instead of using one averaged weather value.

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
2. Floors timestamps into hourly intervals.
3. Calculates the distance of each observation from minute `:53`.
4. Selects the observation closest to `:53`.
5. Normalizes the selected timestamp to the top of the hour.
6. Reindexes the dataset onto a complete hourly timeline.
7. Detects missing observations.
8. Interpolates only short gaps.
9. Preserves long gaps as missing.
10. Validates final timestamps and duplicate counts.

NOAA GHCNh timestamps are treated as:

```text
UTC
```

The timestamp normalization performed by the project is hourly binning, not a timezone conversion.

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

Remaining missing weather rows after short-gap interpolation:

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

The final modeling dataset has 150 timestamps removed because of missing weather observations.

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

after a removed weather observation.

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

The current target value is therefore never included in its own feature.

This prevents:

```text
target leakage
```

during machine-learning training.

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

## Why Chronological Splitting Matters

A random train/test split could allow the model to train on future electricity-demand behavior while evaluating on older observations.

That would not represent a realistic forecasting environment.

The chronological split ensures:

```text
past data → predicts future data
```

which better reflects how electricity-demand forecasting systems operate in practice.

---

# Forecasting Objective

The primary forecasting objective is:

```text
24-hour-ahead hourly electricity-demand forecasting
```

The model predicts the next 24 hourly electricity-demand values.

Conceptually:

```text
Historical Demand + Weather + Time Features
                    │
                    ▼
           Forecast Next 24 Hours
```

This resembles the day-ahead forecasting problem commonly used in electric-grid operations.

---

# Rolling Forecast Evaluation

A single forecast from the end of 2023 through all of 2024 would not represent a realistic day-ahead forecasting system.

For example, a forecast for July 2024 should be allowed to use observations from earlier in 2024.

The project therefore uses:

```text
rolling-origin evaluation
```

also known as:

```text
walk-forward validation
```

The evaluation process behaves like:

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

The validation configuration is:

```text
Forecast horizon: 24 hours
Step size:        24 hours
Forecast windows: 366
Total predictions: 8,784
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
│   ├── validate_demand.py
│   ├── validate_weather.py
│   ├── check_weather_gaps.py
│   ├── build_dataset.py
│   ├── feature_engineering.py
│   └── train_baseline.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

Large raw and processed datasets are excluded from Git version control.

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

## 2. Demand Validation

```text
src/validate_demand.py
```

Validates the electricity-demand timeline.

Responsibilities include:

- Timestamp parsing
- Missing timestamp detection
- Duplicate detection
- Hourly reindexing
- Missing demand handling
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

Processes NOAA weather observations.

Responsibilities include:

- NOAA PSV parsing
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

Creates and evaluates simple Seasonal Naive electricity-demand forecasts.

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

# Modeling Strategy

The project compares simple historical-demand baselines against increasingly capable machine-learning models.

The modeling progression is:

```text
Seasonal Naive Baselines
          │
          ▼
   Linear Regression
          │
          ▼
   Tree-Based Models
          │
          ▼
       XGBoost
          │
          ▼
   Model Comparison
```

The purpose of the baseline models is to establish a meaningful performance threshold.

A more complex model should provide measurable improvement over simply using recent historical electricity demand.

---

# Baseline Models

Two Seasonal Naive forecasting baselines are currently evaluated.

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

The model is still evaluated using a:

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

Example:

```text
Forecast:
Tuesday 3:00 PM

Uses:
Previous Tuesday 3:00 PM demand
```

The forecast horizon remains:

```text
24 hours
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

means that the daily Seasonal Naive forecast differs from actual electricity demand by approximately:

```text
4.34% on average
```

across the 2024 validation period.

The MAE indicates that the model's hourly forecast is typically off by approximately:

```text
1,457 MW
```

The larger RMSE indicates that some forecasting periods contain significantly larger errors that are penalized more strongly by the squared-error calculation.

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

The weekly baseline's MAE is also nearly twice as large as the daily baseline.

This establishes the daily Seasonal Naive model as the project's current:

```text
PRIMARY BASELINE
```

---

# Baseline to Beat

Future machine-learning models will be compared against:

```text
Seasonal Naive - 24 Hour
```

with baseline performance of:

```text
MAE:  1,457.18 MW
RMSE: 2,003.98 MW
MAPE: 4.34%
```

A successful machine-learning model should ideally improve on all three metrics.

The key modeling question becomes:

```text
Can weather, calendar behavior, lagged demand,
and rolling demand statistics outperform
a simple previous-day electricity-demand forecast?
```

---

# Planned Machine Learning Models

The next forecasting models include:

- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost

The models will use combinations of:

```text
Weather features
Calendar features
Historical demand
Lagged demand
Rolling demand statistics
```

---

# Linear Regression

Linear Regression will serve as the first machine-learning model after the Seasonal Naive baselines.

Its purpose is to establish whether a simple supervised-learning model can improve on daily demand persistence.

Potential inputs include:

```text
Weather features
hour
day_of_week
month
is_weekend
demand_lag_1h
demand_lag_24h
demand_lag_168h
demand_rolling_24h
demand_rolling_168h
```

The Linear Regression results will be compared directly against:

```text
MAE:  1,457.18 MW
RMSE: 2,003.98 MW
MAPE: 4.34%
```

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
MAE = 1,457 MW
```

means the prediction is approximately:

```text
1,457 MW away from actual demand on average
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

MAPE makes forecast performance easier to interpret relative to actual electricity demand.

For example:

```text
MAPE = 4.34%
```

means the model differs from actual demand by approximately:

```text
4.34% on average
```

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

The current pipeline should be executed in the following order.

## 1. Extract Electricity Demand

```bash
python src/extract_swpp_demand.py
```

## 2. Validate Demand

```bash
python src/validate_demand.py
```

## 3. Process Weather

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

rather than large raw or processed data files.

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
- Gradient Boosting
- XGBoost

## Model Evaluation

- MAE
- RMSE
- MAPE
- Chronological validation
- Rolling 24-hour forecasting

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

✓ Demand validation

✓ Continuous 52,608-hour demand timeline

✓ NOAA weather ingestion

✓ NOAA station selection

✓ Hourly weather normalization

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

✓ Rolling-origin validation implemented

✓ StatsForecast integration

✓ 24-hour Seasonal Naive baseline

✓ 168-hour Seasonal Naive baseline

✓ MAE evaluation

✓ RMSE evaluation

✓ MAPE evaluation

✓ Baseline performance comparison

✓ Primary baseline established
```

---

# Current Baseline Results

```text
24-Hour Seasonal Naive

MAE:  1,457.18 MW
RMSE: 2,003.98 MW
MAPE: 4.34%
```

```text
168-Hour Seasonal Naive

MAE:  2,823.37 MW
RMSE: 3,865.90 MW
MAPE: 8.27%
```

Current best baseline:

```text
24-Hour Seasonal Naive
```

---

# Next Steps

```text
1. Train Linear Regression

2. Evaluate Linear Regression on 2024 validation data

3. Compare Linear Regression against the 24-hour baseline

4. Train Random Forest

5. Train Gradient Boosting

6. Train XGBoost

7. Compare all model performance

8. Select the strongest validation model

9. Evaluate the selected model on untouched 2025 test data

10. Analyze feature importance

11. Analyze errors by hour of day

12. Analyze errors by month and season

13. Visualize actual vs predicted electricity demand

14. Evaluate behavior during high-demand periods

15. Document final model results
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
- Data-quality monitoring
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
- Additional rolling windows
- Forecast uncertainty intervals
- Peak-demand prediction
- Seasonal error analysis

---

# Forecasting Caveats

The current project uses observed historical weather values for model development.

For a true production day-ahead forecasting system, future weather observations would not yet be known.

A production forecasting pipeline would therefore need to replace observed future weather with:

```text
weather forecasts
```

such as numerical weather prediction or forecast API data.

Conceptually:

```text
Historical Training
-------------------

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

This distinction will be important as the project moves from historical model evaluation toward a production-style forecasting architecture.

---

# Forecast Horizon Considerations

Some engineered lag features require additional care in a real 24-hour-ahead forecasting environment.

For example:

```text
demand_lag_1h
```

is available when predicting the next immediate hour.

However, when forecasting all 24 future hours simultaneously, the actual demand one hour before later forecast horizons may not yet be known.

Future versions of the project may address this using:

- Horizon-specific models
- Recursive forecasting
- Direct multi-step forecasting
- Lag restrictions based on forecast availability

The current project will document feature availability carefully to avoid unrealistic forecasting assumptions.

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

This would provide a more realistic industry benchmark for evaluating the forecasting system.

---

# Model Comparison Framework

As additional models are trained, results will be recorded in a common comparison table.

Current results:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| **Seasonal Naive - 24 Hour** | **1,457.18 MW** | **2,003.98 MW** | **4.34%** |
| Seasonal Naive - 168 Hour | 2,823.37 MW | 3,865.90 MW | 8.27% |
| Linear Regression | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD |
| Gradient Boosting | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD |

The 2025 test dataset will remain untouched until model selection is complete.

This prevents repeated model decisions from indirectly overfitting the final test period.

---

# Long-Term Goal

The long-term goal of this project is to evolve beyond a standalone machine-learning notebook into an:

**end-to-end energy data and forecasting platform**

that demonstrates how modern data engineering and machine-learning techniques can be applied to real-world electrical power-system data.

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