# SPP Electricity Demand Forecasting

An end-to-end data engineering and machine learning project for forecasting hourly electricity demand across the Southwest Power Pool (SPP) region using historical electricity demand, regional weather observations, calendar features, and lagged demand.

The project combines **electrical power systems, data engineering, time-series forecasting, and machine learning** to explore how historical grid behavior and environmental conditions can be used to predict future electricity consumption. :contentReference[oaicite:0]{index=0}

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

1. Extracts historical SPP electricity demand from EIA data.
2. Processes hourly weather observations from NOAA.
3. Detects missing, unrealistic, and suspicious values.
4. Combines electricity and weather observations.
5. Engineers calendar and historical-demand features.
6. Splits data chronologically to prevent future information from entering model training.
7. Establishes forecasting baselines.
8. Trains Linear Regression, Random Forest, and XGBoost models.
9. Tunes XGBoost using a dedicated validation year.
10. Retrains the selected model using all available pre-test data.
11. Evaluates the final model on a held-out 2025 test period.
12. Uses model-error analysis to identify additional upstream data-quality problems.

---

# Architecture

```text
                         DATA SOURCES
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
          U.S. EIA Data                 NOAA GHCNh
          SPP Demand                    Weather Data
                │                           │
                ▼                           ▼
        Demand Extraction             Weather Processing
                │                           │
                ▼                           ▼
        Demand Validation             Weather Validation
                │                           │
                ▼                           ▼
       Range + Temporal QC          Range + Missing QC
                │                           │
                └─────────────┬─────────────┘
                              │
                              ▼
                       Dataset Builder
                              │
                              ▼
                       Modeling Dataset
                              │
                              ▼
                      Feature Engineering
                              │
                              ▼
                  Chronological Data Split
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
              TRAIN       VALIDATION       TEST
            2020-2023        2024          2025
                │             │
                └──────┬──────┘
                       ▼
               Model Development
                       │
                       ▼
               XGBoost Tuning
                       │
                       ▼
              Best Model Selected
                       │
                       ▼
             Retrain on 2020-2024
                       │
                       ▼
                 Test on 2025
                       │
                       ▼
               Final Evaluation
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

All timestamps are processed in UTC.

The complete validated demand timeline contains:

```text
52,608 hourly observations
```

with:

```text
Missing demand values: 0
Missing timestamps:    0
Duplicate timestamps:  0
```

The original project used a continuous hourly EIA demand series and a project-level range check of 0–100,000 MW to identify grossly corrupted measurements. :contentReference[oaicite:1]{index=1} :contentReference[oaicite:2]{index=2}

---

# Electricity Demand Validation

Electricity-demand validation is performed by:

```text
src/validate_electricity.py
```

The validation pipeline performs:

- Timestamp parsing
- Numeric conversion
- Duplicate detection
- Missing timestamp detection
- Reindexing onto the expected hourly timeline
- Demand-range validation
- Isolated missing-value interpolation
- Temporal flatline detection
- Demand-quality labeling
- Final timeline validation

---

# Demand Range Validation

A project-level physical plausibility check is used to detect obviously corrupted demand measurements.

Values outside:

```text
0 MW to 100,000 MW
```

are treated as invalid before interpolation.

This is a practical project-level sanity check rather than an official EIA quality-control threshold.

One major corrupted observation discovered during model development was:

```text
Timestamp: 2023-06-13 02:00 UTC
Demand:    3,621,097 MW
```

while neighboring observations were approximately 28,000–34,000 MW.

This observation was converted to missing and repaired using the existing time-interpolation pipeline.

The discovery demonstrated that checking only for missing values, duplicate timestamps, and missing timestamps is not enough for real-world data. The original README documented this corrupted observation and how model behavior exposed the upstream problem. :contentReference[oaicite:3]{index=3}

---

# Temporal Demand Quality Validation

A second and more subtle electricity-data problem was discovered during final model evaluation.

Some demand measurements remained within physically reasonable ranges but repeated the **exact same demand value for unusually long periods**.

Examples included runs lasting:

```text
12 hours
24 hours
48 hours
62 hours
82 hours
96 hours
133 hours
155 hours
```

These values would pass ordinary checks because they were:

```text
numeric
non-missing
within the valid demand range
associated with valid timestamps
```

but the temporal behavior was suspicious.

The demand-validation pipeline was therefore expanded to detect consecutive identical demand observations.

### Detection Rule

Runs of at least:

```text
6 consecutive identical hourly observations
```

are reported for inspection.

Runs of at least:

```text
12 consecutive identical hourly observations
```

are automatically labeled:

```text
demand_quality = "flatline"
```

All other observations are labeled:

```text
demand_quality = "valid"
```

The 12-hour threshold is a project-level temporal quality-control rule and is not an official EIA threshold.

Long flatlines are **not interpolated** because replacing multi-hour or multi-day sequences with synthetic demand would introduce a large amount of artificial target data.

Instead, they remain in the validated demand dataset with an explicit quality label and are excluded later from model training and evaluation.

---

# Demand Quality Results

Across the complete 2020–2025 demand timeline:

```text
Valid demand observations:      51,327
Flagged flatline observations:   1,281
Total observations:             52,608
```

This was an important finding because the demand series could contain:

```text
0 missing values
0 missing timestamps
0 duplicate timestamps
```

while still containing substantial temporal data-quality problems.

---

# Lag Quality Propagation

Demand quality must also be considered when creating historical-demand features.

The supervised models use:

```text
demand_lag_24h
demand_lag_168h
```

A current target can be valid while one of its lagged inputs points to a previously flagged flatline.

For example:

```text
Current demand       = valid
Demand 24 hours ago  = flatline
```

Using that observation would still introduce suspicious demand information into the model.

The feature-engineering pipeline therefore tracks quality for the current target and its historical lag sources.

A modeling observation is retained only when:

```text
current target quality = valid
24-hour lag quality    = valid
168-hour lag quality   = valid
```

This causes more modeling rows to be removed than the number of directly flagged target observations.

---

# Demand Quality Filtering Results

After removing rows with insufficient history:

```text
Rows before demand QC: 52,290
```

Quality checks identified:

```text
Rows with invalid target demand: 1,267
Rows with invalid 24h lag:       1,264
Rows with invalid 168h lag:      1,281
```

Because some observations violate multiple conditions, these counts overlap.

Total rows removed because of demand quality:

```text
2,859
```

Final feature dataset:

```text
49,431 rows
32 columns
0 missing values
0 duplicate timestamps
```

Demand-quality issues affecting modeling by year:

| Year | Invalid Target | Invalid 24h Lag | Invalid 168h Lag | Invalid for Modeling |
|---|---:|---:|---:|---:|
| 2020 | 36 | 60 | 204 | 261 |
| 2021 | 127 | 127 | 127 | 264 |
| 2022 | 0 | 0 | 0 | 0 |
| 2023 | 12 | 12 | 12 | 36 |
| 2024 | 396 | 396 | 396 | 847 |
| 2025 | 710 | 710 | 710 | 1,636 |

The later years, particularly 2024 and 2025, contained substantially more suspicious flatline behavior.

---

# Weather Data

Weather observations are sourced from NOAA's **Global Historical Climatology Network Hourly (GHCNh)**.

Because SPP covers a large geographic area, five representative weather locations are used:

| Location | NOAA Station |
|---|---|
| Kansas City | USW00003947 |
| Wichita | USW00003928 |
| Oklahoma City | USW00013967 |
| Omaha | USW00014942 |
| Fargo | USW00014914 |

Weather features include:

```text
temperature
dew_point_temperature
relative_humidity
wind_speed
```

Measurements from each station remain separate rather than being averaged together so that the models can learn regional weather differences. :contentReference[oaicite:4]{index=4}

---

# Weather Processing and Validation

NOAA GHCNh can contain multiple observations during an hour.

The pipeline:

1. Parses NOAA timestamps.
2. Converts weather measurements to numeric values.
3. Applies physical plausibility checks.
4. Groups observations into hourly intervals.
5. Selects the observation closest to minute `:53`.
6. Normalizes the selected observation to the top of the hour.
7. Reindexes onto a complete hourly timeline.
8. Detects missing observations.
9. Interpolates short gaps.
10. Preserves long gaps.
11. Validates timestamps and duplicates.

NOAA timestamps are treated as UTC.

---

# Weather Range Validation

Model-error analysis also exposed corrupted NOAA observations.

An early Linear Regression model produced a physically impossible forecast of approximately:

```text
-2,997,113 MW
```

Investigation revealed weather values including:

```text
Temperature:       -61.0 °C
Relative humidity: 31,287%
```

The weather-validation pipeline was expanded with project-level physical plausibility checks:

| Feature | Valid Range |
|---|---:|
| Temperature | -60°C to 60°C |
| Dew Point | -70°C to 40°C |
| Relative Humidity | 0% to 100% |
| Wind Speed | 0 to 75 m/s |

Values outside these ranges are converted to missing before interpolation.

These bounds are practical sanity checks rather than official NOAA QC thresholds. :contentReference[oaicite:5]{index=5}

---

# Missing Weather Strategy

Only short weather gaps are interpolated.

```text
Maximum interpolation gap: 6 hours
```

Longer missing sequences remain missing rather than creating artificial multi-day weather patterns.

A significant reporting gap was observed across several weather stations around:

```text
August 29, 2025
through
September 2, 2025
```

These observations were not blindly interpolated.

Remaining missing weather rows after processing:

| Station | Rows With Missing Weather |
|---|---:|
| Kansas City | 91 |
| Wichita | 116 |
| Oklahoma City | 90 |
| Omaha | 103 |
| Fargo | 110 |

The six-hour interpolation strategy and station-level missing counts were established during weather preprocessing. :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}

---

# Dataset Construction

The script:

```text
src/build_dataset.py
```

combines SPP demand with all five processed weather datasets using:

```text
timestamp_utc
```

Joins are validated as:

```text
one-to-one
```

to prevent duplicate observations from silently entering the modeling dataset.

Before weather filtering:

```text
Rows:                 52,608
Columns:              22
Duplicate timestamps: 0
Missing demand:       0
```

After filtering timestamps with incomplete weather:

```text
Rows before filtering: 52,608
Rows removed:             150
Modeling rows:         52,458
```

Only approximately 0.29% of the original timeline is removed because of incomplete weather. :contentReference[oaicite:8]{index=8}

---

# Feature Engineering

Feature engineering is performed by:

```text
src/feature_engineering.py
```

The pipeline creates:

### Calendar Features

```text
hour
day_of_week
month
is_weekend
```

### Demand Lag Features

```text
demand_lag_1h
demand_lag_24h
demand_lag_168h
```

### Rolling Demand Features

```text
demand_rolling_24h
demand_rolling_168h
```

---

# Timestamp-Aware Demand Engineering

Weather filtering removes some timestamps from the modeling dataset.

Therefore:

```python
df["demand_mw"].shift(24)
```

on the filtered dataset would not always mean:

```text
exactly 24 clock-hours earlier
```

Demand-history features are instead calculated using the original complete 52,608-hour demand timeline and merged back into the modeling dataset by:

```text
timestamp_utc
```

This guarantees that `demand_lag_24h` really represents demand exactly 24 hours earlier. :contentReference[oaicite:9]{index=9}

---

# Forecast-Time Feature Availability

The project creates additional features such as:

```text
demand_lag_1h
demand_rolling_24h
demand_rolling_168h
```

but the first direct day-ahead machine-learning models intentionally exclude them.

The forecasting objective is to predict the next 24 hourly values at once.

For later hours in that forecast horizon, a target-relative one-hour lag or rolling window could require demand observations that would not yet exist when the forecast is issued.

The primary supervised feature set therefore uses:

```text
Calendar features
24-hour demand lag
168-hour demand lag
Regional weather variables
```

This keeps the historical-demand inputs available across the complete 24-hour forecasting horizon.

---

# Final Feature Dataset

After insufficient-history and demand-quality filtering:

```text
Rows:                 49,431
Columns:              32
Missing values:       0
Duplicate timestamps: 0

First timestamp:
2020-01-08 00:00 UTC

Last timestamp:
2025-12-31 23:00 UTC
```

The first usable observation occurs one week into 2020 because the 168-hour lag requires a complete week of historical demand.

---

# Chronological Data Splitting

Electricity demand is a time series, so random train/test splitting is not used.

Initial model development uses:

```text
2020 ─┐
2021  │
2022  ├── TRAIN
2023 ─┘

2024 ─── VALIDATION

2025 ─── TEST
```

After final quality filtering:

```text
Train (2020-2023):  34,492 rows
Validation (2024):   7,928 rows
Test (2025):         7,011 rows
```

Total:

```text
49,431 modeling observations
```

Chronological splitting ensures that models learn from earlier observations before being evaluated on later observations, rather than allowing future demand behavior into training. The original design used the same chronological 2020–2023 / 2024 / 2025 structure. :contentReference[oaicite:10]{index=10}

---

# Test-Set Methodology Note

The 2025 period was originally intended to remain completely untouched during model development.

During an earlier 2025 evaluation, however, error analysis exposed long sequences of suspicious repeated demand measurements. This motivated the temporal flatline quality-control rule.

The QC rule was then applied **uniformly across the entire 2020–2025 dataset**, rather than selectively removing 2025 errors.

Model selection and hyperparameter tuning continued to use the 2024 validation period rather than 2025.

For this reason, 2025 is described as the project's:

```text
held-out final evaluation period
```

rather than claiming that it remained completely untouched throughout development.

No additional hyperparameter or preprocessing changes are made based on the final cleaned 2025 performance.

---

# Forecasting Objective

The primary objective is:

```text
24-hour-ahead hourly electricity-demand forecasting
```

The model estimates the next 24 hourly SPP demand values using:

```text
Historical Demand
       +
Regional Weather
       +
Calendar Information
       ↓
24-Hour Demand Forecast
```

---

# Important Weather Limitation

The historical evaluation currently uses **observed target-time weather**.

That means the model receives the weather conditions that actually occurred during the historical forecast period.

In a real production day-ahead forecasting system, those observations would not yet be available.

A production implementation would instead require:

```text
weather forecasts available at forecast issue time
```

Therefore, current weather inputs should be interpreted as **oracle/proxy weather** for evaluating the relationship between weather and demand.

Replacing observed weather with historical forecast-weather archives is an important future improvement.

---

# Machine Learning Models

The project evaluates increasingly capable approaches:

```text
Seasonal Naive
      ↓
Linear Regression
      ↓
Random Forest
      ↓
XGBoost
      ↓
XGBoost Hyperparameter Tuning
      ↓
Final XGBoost
```

---

# Seasonal Naive Baseline

The primary baseline predicts demand using demand from the same hour one day earlier:

```text
Prediction(t) = Demand(t - 24 hours)
```

A second baseline uses the same hour one week earlier:

```text
Prediction(t) = Demand(t - 168 hours)
```

The daily baseline established that previous-day demand is a strong predictor and provided a meaningful threshold for the machine-learning models.

Because demand-quality filtering creates gaps in the final eligible modeling observations, fair post-QC baseline comparison should evaluate baseline predictions on the same eligible target timestamps used by the supervised models.

---

# Post-QC Validation Results

After implementing the expanded demand-quality pipeline, all supervised models were retrained using the cleaned feature dataset.

## Linear Regression

```text
MAE:  1,347.18 MW
RMSE: 1,824.94 MW
MAPE: 3.93%
```

The corrected result was substantially better than the earlier Linear Regression result, showing how strongly corrupted observations had affected the initial evaluation.

---

# Random Forest

Post-QC validation performance:

```text
MAE:  1,109.69 MW
RMSE: 1,502.15 MW
MAPE: 3.22%
```

Worst validation prediction:

```text
Timestamp:  2024-07-20 05:00 UTC
Actual:     43,551 MW
Predicted:  26,563 MW
```

The observation survived the expanded data-quality checks, suggesting that it represents a genuinely difficult demand pattern rather than the type of long flatline automatically excluded by the QC pipeline.

The target was much larger than demand 24 hours earlier:

```text
Target:          43,551 MW
24-hour lag:     24,991 MW
168-hour lag:    39,485 MW
```

This illustrates the difficulty of forecasting rapid changes when recent daily demand is not representative of current conditions.

---

# XGBoost

The initial post-QC XGBoost configuration achieved:

```text
MAE:  1,074.52 MW
RMSE: 1,390.50 MW
MAPE: 3.16%
```

This outperformed both Linear Regression and Random Forest on the cleaned 2024 validation set.

---

# XGBoost Feature Importance

Before final retraining, the XGBoost validation model placed most importance on historical demand and regional temperature.

Major features included:

| Feature | Importance |
|---|---:|
| demand_lag_24h | 0.6619 |
| Wichita temperature | 0.0961 |
| Oklahoma temperature | 0.0680 |
| demand_lag_168h | 0.0400 |
| is_weekend | 0.0230 |
| Kansas City temperature | 0.0209 |
| Omaha temperature | 0.0192 |

Grouped importance:

```text
Demand History:       70.19%
Temperature:          20.61%
Calendar:              5.25%
Humidity / Dew Point:  3.27%
Wind:                   0.68%
```

This indicates that recent electricity demand is the strongest source of predictive information, while weather provides additional information needed to adjust those historical patterns.

---

# Performance by Temperature

The validation model performed best under moderate regional temperatures and worse during temperature extremes.

```text
Temperature     MAE
< 0°C           1,637.56 MW
0-10°C            904.88 MW
10-20°C           871.28 MW
20-30°C         1,128.02 MW
> 30°C          1,235.39 MW
```

Cold conditions were particularly difficult.

This supports the expectation that electricity demand becomes harder to predict during unusual weather and high heating or cooling demand.

---

# XGBoost Hyperparameter Tuning

A predefined grid of 72 XGBoost configurations was evaluated using the cleaned 2024 validation dataset.

The best configuration was:

```python
XGBRegressor(
    max_depth=4,
    learning_rate=0.03,
    n_estimators=300,
    subsample=1.0,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)
```

Best 2024 validation performance:

```text
MAE:  1,023.97 MW
RMSE: 1,347.40 MW
MAPE: 3.01%
```

Improvement over the original post-QC XGBoost configuration:

```text
MAE improvement:   8.57%
RMSE improvement:  7.75%
MAPE improvement:  9.84%
```

The top two configurations were extremely close:

```text
subsample = 1.0 → MAE 1,023.97 MW
subsample = 0.8 → MAE 1,024.46 MW
```

This suggests that the general model configuration was relatively stable and that further aggressive tuning would provide little practical benefit while increasing the risk of overfitting the validation period.

---

# Final Model Training

After selecting the hyperparameters using 2024 validation data, the final model is retrained using all eligible observations from:

```text
2020 through 2024
```

Final training observations:

```text
42,420
```

The final model is then evaluated against eligible observations from:

```text
2025
```

Final test observations:

```text
7,011
```

The model is never fit using the 2025 demand targets.

---

# Final 2025 Test Results

The final tuned XGBoost model achieved:

| Metric | 2024 Validation | 2025 Final Test |
|---|---:|---:|
| MAE | 1,023.97 MW | **1,117.60 MW** |
| RMSE | 1,347.40 MW | **1,429.34 MW** |
| MAPE | 3.01% | **3.18%** |

Final test performance:

```text
MAE:  1,117.60 MW
RMSE: 1,429.34 MW
MAPE: 3.18%
```

The relatively small increase from validation to test error indicates that the selected model generalized reasonably well from the 2020–2024 development period to the later 2025 period.

A 3.18% MAPE means that, across retained 2025 observations, the forecast differed from actual demand by approximately 3.18% on average.

---

# Final 2025 Worst Prediction

The largest retained 2025 error occurred at:

```text
Timestamp:        2025-07-11 05:00 UTC
Actual:           47,720.00 MW
Predicted:        39,376.69 MW
Absolute error:    8,343.31 MW
Percentage error:     17.48%
```

Historical-demand features were:

```text
demand_lag_24h:  39,796 MW
demand_lag_168h: 37,408 MW
```

The target was substantially higher than both lagged demand values, contributing to the underprediction.

This timestamp is also notable because it occurs at the end of an 11-hour sequence of identical `47,720 MW` observations.

The automatic flatline threshold is 12 hours, so this sequence is reported by the QC system but intentionally retained.

The threshold was **not changed after final evaluation** simply to remove the worst model error.

This preserves the predefined preprocessing rule and avoids changing the dataset based on final test performance.

---

# Final Model Feature Importance

After retraining on 2020–2024, the final model's strongest features were:

| Feature | Importance |
|---|---:|
| demand_lag_24h | 0.4778 |
| demand_lag_168h | 0.1623 |
| Wichita temperature | 0.1437 |
| Oklahoma temperature | 0.0690 |
| Kansas City temperature | 0.0398 |
| Omaha temperature | 0.0185 |
| Wichita dew point | 0.0148 |
| is_weekend | 0.0119 |
| Omaha dew point | 0.0114 |
| day_of_week | 0.0104 |

The two historical-demand features alone account for approximately:

```text
64%
```

of total feature importance.

This reinforces the importance of daily and weekly persistence in electricity-demand forecasting.

Temperature is the next major source of predictive information.

---

# 2025 Error by Hour

The model performed best during the morning hours and was generally less accurate around midnight and early morning.

Selected results:

| Hour | MAE | MAPE |
|---:|---:|---:|
| 1 | 1,453.09 MW | 3.82% |
| 5 | 1,119.34 MW | 3.27% |
| 8 | 886.22 MW | 2.87% |
| 9 | 858.62 MW | 2.80% |
| 10 | **854.25 MW** | **2.79%** |
| 11 | 880.05 MW | 2.82% |
| 18 | 1,097.45 MW | 3.03% |
| 23 | 1,453.88 MW | 3.80% |

The lowest hourly MAPE occurred around:

```text
10:00 UTC
```

while some of the largest errors occurred around:

```text
00:00-03:00 UTC
23:00 UTC
```

This suggests that forecast difficulty varies meaningfully across the daily load cycle.

---

# 2025 Error by Month

Monthly performance also varied:

| Month | MAE | MAPE |
|---|---:|---:|
| January | 1,435.59 MW | 3.84% |
| February | 1,274.16 MW | 3.56% |
| March | 823.91 MW | 2.69% |
| April | **818.31 MW** | 2.71% |
| May | 1,086.77 MW | 3.44% |
| June | 1,262.12 MW | 3.56% |
| July | 988.49 MW | **2.43%** |
| August | 1,212.30 MW | 2.96% |
| September | 1,116.27 MW | 3.15% |
| October | 881.10 MW | 2.77% |
| November | 987.05 MW | 3.14% |
| December | **1,490.50 MW** | **4.08%** |

December was the most difficult month by both MAE and MAPE.

Interestingly, July contained the single largest retained error while still producing the lowest monthly MAPE.

This demonstrates why individual worst-case errors and aggregate metrics should both be examined.

---

# Final Model Comparison

Post-QC 2024 validation performance:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Linear Regression | 1,347.18 MW | 1,824.94 MW | 3.93% |
| Random Forest | 1,109.69 MW | 1,502.15 MW | 3.22% |
| XGBoost | 1,074.52 MW | 1,390.50 MW | 3.16% |
| **Tuned XGBoost** | **1,023.97 MW** | **1,347.40 MW** | **3.01%** |

Final selected model:

```text
Tuned XGBoost
```

Final 2025 evaluation:

```text
MAE:  1,117.60 MW
RMSE: 1,429.34 MW
MAPE: 3.18%
```

---

# Evaluation Metrics

## MAE

Mean Absolute Error:

```text
MAE = mean(|actual - predicted|)
```

MAE measures the typical size of an hourly forecast error in MW.

For the final model:

```text
MAE = 1,117.60 MW
```

meaning the retained 2025 predictions differ from actual demand by approximately 1,118 MW on average.

## RMSE

Root Mean Squared Error gives additional weight to large forecast errors.

```text
Final RMSE = 1,429.34 MW
```

## MAPE

Mean Absolute Percentage Error:

```text
MAPE = mean(|actual - predicted| / actual) × 100
```

Final:

```text
MAPE = 3.18%
```

---

# Challenges and Findings

## 1. Missing Values Are Not the Only Data-Quality Problem

Initial validation focused on:

```text
missing timestamps
duplicate timestamps
missing values
```

However, values can exist and still be incorrect.

Examples discovered during this project included:

```text
Relative humidity: 31,287%
Electricity demand: 3,621,097 MW
```

This motivated physical plausibility checks for both weather and demand.

### Finding

Data validation must answer both:

```text
Does the value exist?
```

and:

```text
Does the value make sense?
```

---

## 2. Valid Individual Values Can Form Invalid Sequences

The flatline issue introduced another class of data-quality problem.

Values such as:

```text
33,813 MW
37,455 MW
48,701 MW
```

are individually plausible SPP demand values.

The problem becomes visible only when exactly the same value appears for dozens or hundreds of consecutive hours.

### Finding

Data quality must be evaluated across both:

```text
individual observations
```

and:

```text
temporal sequences
```

Range checks alone cannot detect this type of failure.

---

## 3. Model Errors Became Data-Quality Signals

Several major pipeline problems were discovered because model predictions behaved unexpectedly.

Linear Regression exposed corrupted weather.

Random Forest exposed an extreme corrupted electricity-demand target.

Final XGBoost error analysis exposed long demand flatlines.

### Finding

Model diagnostics can serve as an additional layer of data-quality monitoring.

An extreme prediction or error should trigger investigation of:

```text
model behavior
feature engineering
data leakage
input quality
target quality
```

rather than automatically being blamed on the algorithm.

---

## 4. Data Quality Can Propagate Through Lag Features

A bad demand observation does not affect only one target row.

Because the project uses:

```text
demand_lag_24h
demand_lag_168h
```

one corrupted sequence can reappear later as model input.

### Finding

Quality metadata must propagate through feature engineering.

This is why the project checks target quality as well as the quality of the observations used to create important lag features.

---

## 5. Long Gaps Should Not Be Blindly Interpolated

Short isolated gaps can reasonably be estimated.

Long gaps are different.

Interpolating across days of missing weather or demand can create artificial patterns that never occurred.

### Finding

The project uses conservative repair strategies:

```text
Short weather gaps → interpolate

Long weather gaps → preserve as missing

Isolated invalid demand → interpolate

Long demand flatlines → flag and exclude
```

---

## 6. Row-Based Time-Series Operations Can Be Dangerous

Once timestamps are removed because of missing weather:

```python
df.shift(24)
```

does not necessarily mean 24 hours earlier.

### Finding

Time-series features should be constructed using the complete timestamp-aware history and joined back by timestamp.

---

## 7. Forecast-Time Availability Matters

A feature can be historically valid but unavailable when a real forecast is generated.

Examples include:

```text
one-hour target-relative demand lags
target-relative rolling windows
observed future weather
```

### Finding

Predictive power alone is not enough.

A production forecasting feature must also be available at forecast time.

---

## 8. Historical Demand Is the Strongest Predictor

The final model places approximately 64% of its feature importance on:

```text
demand_lag_24h
demand_lag_168h
```

### Finding

SPP electricity demand contains strong daily and weekly persistence.

Weather improves the model's ability to adjust those historical patterns.

---

## 9. Temperature Matters More Than Most Other Weather Variables

Regional temperature variables consistently rank above:

```text
humidity
dew point
wind speed
```

with Wichita and Oklahoma temperature particularly important.

### Finding

Temperature appears to be the most useful environmental signal in the current model.

---

## 10. Extreme Conditions Are Harder to Forecast

Validation performance was worse during very cold and very hot conditions than during moderate temperatures.

### Finding

Demand behavior becomes more difficult to estimate during weather extremes, likely because heating and cooling loads introduce stronger nonlinear behavior.

---

## 11. Average Performance Can Hide Large Errors

The final model achieved:

```text
MAPE = 3.18%
```

but still had a worst retained error of:

```text
8,343 MW
```

### Finding

A strong average forecasting metric does not guarantee small errors at every timestamp.

Grid forecasting analysis should examine:

```text
MAE
RMSE
MAPE
worst-case errors
hourly behavior
monthly behavior
extreme weather
peak demand
```

---

## 12. Validation and Test Performance Should Be Close, Not Identical

The tuned XGBoost achieved:

```text
2024 Validation MAPE: 3.01%
2025 Test MAPE:       3.18%
```

The test result is slightly worse but remains close to validation performance.

### Finding

This is a healthier result than seeing a dramatic test-performance collapse and suggests reasonable generalization to the later year.

---

## 13. More Hyperparameter Tuning Was Not Necessarily Better

The two strongest XGBoost configurations differed by less than 1 MW in validation MAE.

### Finding

Once several configurations perform almost identically, continued tuning provides little practical value and increases the chance of fitting model-selection decisions too closely to the validation year.

---

## 14. Data Cleaning Decisions Must Not Chase Test Performance

The final worst prediction occurs inside an 11-hour constant-demand sequence.

The predefined automatic flatline threshold is 12 hours.

Lowering the threshold after seeing the final test error could improve reported metrics but would make the preprocessing decision dependent on test performance.

### Finding

Data-quality rules should be defined generally and applied consistently, even when that means retaining difficult observations.

---

## 15. Observed Weather Is Not the Same as Forecast Weather

The historical experiment uses weather observations from the target period.

A real day-ahead system would not know the exact future temperature, humidity, or wind speed.

### Finding

A production implementation should replace observed target-time weather with archived weather forecasts that would actually have been available when each demand forecast was issued.

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
│       ├── feature_dataset.csv
│       ├── xgboost_tuning_results.csv
│       └── final_2025_predictions.csv
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
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   ├── tune_xgboost.py
│   └── test_final_model.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

Large raw and generated datasets are excluded from Git version control.

---

# Running the Project

## Create a Virtual Environment

```bash
python -m venv .venv
```

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Pipeline Execution

Run the pipeline in order.

## 1. Extract SPP Demand

```bash
python src/extract_swpp_demand.py
```

## 2. Validate Electricity Demand

```bash
python src/validate_electricity.py
```

This stage performs range validation, interpolation of isolated missing values, flatline detection, and demand-quality labeling.

## 3. Process Weather

```bash
python src/validate_weather.py
```

## 4. Analyze Weather Gaps

```bash
python src/check_weather_gaps.py
```

## 5. Build Modeling Dataset

```bash
python src/build_dataset.py
```

## 6. Engineer Features and Apply Demand QC

```bash
python src/feature_engineering.py
```

## 7. Train Baseline

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

## 10. Train XGBoost

```bash
python src/train_xgboost.py
```

## 11. Tune XGBoost

```bash
python src/tune_xgboost.py
```

## 12. Run Final 2025 Evaluation

```bash
python src/test_final_model.py
```

---

# Technology Stack

## Data Engineering

- Python
- Pandas
- NumPy
- CSV / PSV processing
- ETL pipeline design
- Data validation
- Range validation
- Temporal data-quality checks
- Missing-data analysis
- Timestamp-aware joins
- Time-series feature engineering
- Quality-flag propagation

## Machine Learning

- scikit-learn
- XGBoost
- StatsForecast
- Linear Regression
- Random Forest
- Gradient-boosted decision trees
- Hyperparameter tuning

## Power and Energy

- Electricity-demand forecasting
- Load persistence
- Daily demand cycles
- Weekly demand cycles
- Weather-sensitive load
- SPP regional demand

---

# Current Project Results

The final selected model is:

```text
Tuned XGBoost
```

with:

```text
2024 Validation
----------------
MAE:  1,023.97 MW
RMSE: 1,347.40 MW
MAPE: 3.01%

2025 Final Evaluation
---------------------
MAE:  1,117.60 MW
RMSE: 1,429.34 MW
MAPE: 3.18%
```

The project demonstrates that a combination of:

```text
Historical electricity demand
+
Regional weather
+
Calendar information
+
Nonlinear machine learning
```

can provide accurate hourly SPP demand estimates while also showing that **data quality is as important as model selection**.

---

# Future Improvements

## Data Engineering

The current pipeline is local and file-based.

A future production-oriented architecture could migrate toward:

```text
EIA + NOAA
     ↓
Python Ingestion
     ↓
Amazon S3
     ↓
ETL / Validation
     ↓
Snowflake
     ↓
Feature Pipeline
     ↓
Model Training
     ↓
Forecast API
     ↓
Dashboard
```

Potential improvements include:

- Automated scheduled ingestion
- Incremental data loading
- Cloud object storage
- Snowflake warehouse integration
- Pipeline orchestration
- Data-quality monitoring
- Schema validation
- Data lineage
- Model artifact storage
- Automated retraining
- Forecast monitoring

## Forecasting

Potential modeling improvements include:

- Historical forecast-weather data
- SPP-specific weather weighting
- Local-time calendar features
- Holiday indicators
- Heating Degree Days
- Cooling Degree Days
- Better extreme-weather features
- Forecast-horizon-specific models
- Direct multi-horizon forecasting
- Peak-demand-specific evaluation
- Prediction intervals
- SHAP-based model interpretation
- Additional boosting approaches such as LightGBM or CatBoost

---

# Key Takeaway

The most important lesson from this project was not simply that XGBoost produced the lowest forecasting error.

The project repeatedly showed that **model performance depends on the quality and temporal correctness of the entire data pipeline**.

Several of the most valuable improvements came from investigating model failures:

```text
Impossible Linear Regression prediction
        ↓
Corrupted weather discovered
        ↓
Weather range validation added

Extreme Random Forest prediction
        ↓
Corrupted demand discovered
        ↓
Demand range validation added

Large 2025 XGBoost error
        ↓
Long demand flatlines discovered
        ↓
Temporal demand QC added
        ↓
Lag-quality propagation added
```

The final result is therefore not only a machine-learning model, but a more robust end-to-end forecasting pipeline that performs:

```text
data ingestion
validation
cleaning
temporal quality control
feature engineering
model training
hyperparameter selection
held-out evaluation
error analysis
```

while achieving a final 2025 MAPE of:

```text
3.18%
```

on the retained eligible test observations.