# SPP Electricity Demand Forecasting

An end-to-end data engineering and machine learning project for forecasting hourly electricity demand across the Southwest Power Pool (SPP) region using historical electricity demand, weather observations, temporal features, lagged demand, and rolling demand statistics.

The project combines **electrical power systems, data engineering, and machine learning** to explore how historical grid behavior and environmental conditions can be used to predict future electricity consumption.

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
7. Trains and evaluates forecasting models.

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
          Forecast Models
                │
                ▼
         Model Evaluation
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

### Hour

```text
hour
```

Values:

```text
0-23
```

This helps the model learn daily electricity-use patterns.

---

### Day of Week

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

---

### Month

```text
month
```

Values:

```text
1-12
```

This helps capture seasonal electricity-demand behavior.

---

### Weekend Indicator

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

Final split sizes:

```text
Train:       34,885
Validation:   8,775
Test:         8,630
```

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

# Repository Structure

```text
.
├── data/
│   ├── raw/
│   │   ├── electricity/
│   │   └── weather/
│   │
│   └── processed/
│       └── weather/
│
├── src/
│   ├── extract_swpp_demand.py
│   ├── validate_demand.py
│   ├── validate_weather.py
│   ├── check_weather_gaps.py
│   ├── build_dataset.py
│   └── feature_engineering.py
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

# 2. Demand Validation

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

# 3. Weather Validation

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

# 4. Weather Gap Analysis

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

# 5. Dataset Construction

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

# 6. Feature Engineering

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

# Modeling Strategy

The project will compare simple electricity-demand baselines against machine-learning models.

The planned progression is:

```text
Naive Forecasting Baselines
        │
        ▼
Linear Regression
        │
        ▼
Tree-Based Models
        │
        ▼
XGBoost
```

---

# Baseline Models

Before training more complex machine-learning models, simple historical-demand forecasts will establish baseline performance.

## 24-Hour Persistence

Prediction:

```text
Predicted Demand
=
Demand 24 hours earlier
```

Feature:

```text
demand_lag_24h
```

---

## 168-Hour Persistence

Prediction:

```text
Predicted Demand
=
Demand 168 hours earlier
```

Feature:

```text
demand_lag_168h
```

These baselines establish the minimum performance that machine-learning models should outperform.

---

# Planned Machine Learning Models

Potential forecasting models include:

- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost

The models will use combinations of:

```text
Weather features
Calendar features
Historical demand
Rolling demand statistics
```

---

# Evaluation Metrics

Forecast performance will be measured using:

## MAE

Mean Absolute Error

```text
MAE = average absolute prediction error
```

MAE provides an intuitive measure of the typical demand-forecast error.

---

## RMSE

Root Mean Squared Error

```text
RMSE = square root of mean squared prediction error
```

RMSE penalizes large forecasting errors more heavily.

---

## MAPE

Mean Absolute Percentage Error

```text
MAPE = average percentage prediction error
```

MAPE makes forecast error easier to interpret relative to actual electricity demand.

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

---

# Run the Data Pipeline

The current pipeline should be executed in the following order.

### 1. Extract Electricity Demand

```bash
python src/extract_swpp_demand.py
```

### 2. Validate Demand

```bash
python src/validate_demand.py
```

### 3. Process Weather

```bash
python src/validate_weather.py
```

### 4. Optional Weather Gap Analysis

```bash
python src/check_weather_gaps.py
```

### 5. Build Modeling Dataset

```bash
python src/build_dataset.py
```

### 6. Engineer Features

```bash
python src/feature_engineering.py
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

---

## Machine Learning

- Scikit-learn
- XGBoost

---

## Data Sources

- U.S. Energy Information Administration
- NOAA Global Historical Climatology Network Hourly

---

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
```

---

## Next Steps

```text
1. Create naive forecasting baselines

2. Evaluate 24-hour persistence

3. Evaluate 168-hour persistence

4. Train Linear Regression

5. Train tree-based models

6. Train XGBoost

7. Compare model performance

8. Analyze feature importance

9. Visualize actual vs predicted demand
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

# Future Improvements

Potential future improvements include:

- Automated EIA ingestion
- Automated NOAA ingestion
- AWS S3 raw-data storage
- Snowflake data warehousing
- ETL orchestration
- Pipeline scheduling
- Data-quality monitoring
- Data lineage
- Model experiment tracking
- Automated retraining
- Forecast APIs
- Interactive dashboards
- Cloud deployment
- More SPP weather stations
- Additional weather variables
- Cyclical calendar features
- Holiday features
- Extreme-weather indicators
- Forecast uncertainty intervals

---

# Forecast Benchmarking

A future extension could compare the project's machine-learning forecasts against SPP's published day-ahead demand forecasts.

Potential comparison:

```text
Project ML Forecast
        vs
SPP Day-Ahead Forecast
        vs
Actual Demand
```

This would provide a more realistic benchmark for evaluating the forecasting system.

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
Time-Series Analysis
        +
Machine Learning
        +
Cloud Data Architecture
```