from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_DATASET_PATH = (
    PROJECT_ROOT / "data" / "processed" / "feature_dataset.csv"
)

FINAL_PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "final_2025_predictions.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "docs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def save_figure(filename):
    output_path = OUTPUT_DIR / filename

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Saved: {output_path}")


def load_data():
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    features = pd.read_csv(FEATURE_DATASET_PATH)
    predictions = pd.read_csv(FINAL_PREDICTIONS_PATH)

    features["timestamp_utc"] = pd.to_datetime(
        features["timestamp_utc"]
    )

    predictions["timestamp_utc"] = pd.to_datetime(
        predictions["timestamp_utc"]
    )

    # Recompute errors just to guarantee consistency
    predictions["error_mw"] = (
        predictions["demand_mw"]
        - predictions["predicted_demand_mw"]
    )

    predictions["absolute_error_mw"] = (
        predictions["error_mw"].abs()
    )

    predictions["ape"] = (
        predictions["absolute_error_mw"]
        / predictions["demand_mw"]
        * 100
    )

    predictions["hour"] = predictions["timestamp_utc"].dt.hour
    predictions["month"] = predictions["timestamp_utc"].dt.month

    print(f"Feature rows: {len(features)}")
    print(f"Prediction rows: {len(predictions)}")
    print()

    return features, predictions


# ============================================================
# 1. FULL 2025 ACTUAL VS PREDICTED
# ============================================================

def plot_full_2025_forecast(predictions):
    print("Generating full 2025 actual vs predicted plot...")

    plt.figure(figsize=(16, 6))

    plt.plot(
        predictions["timestamp_utc"],
        predictions["demand_mw"],
        label="Actual Demand",
        linewidth=1
    )

    plt.plot(
        predictions["timestamp_utc"],
        predictions["predicted_demand_mw"],
        label="Predicted Demand",
        linewidth=1
    )

    plt.title("SPP Actual vs Predicted Electricity Demand - 2025")
    plt.xlabel("Date")
    plt.ylabel("Electricity Demand (MW)")
    plt.legend()
    plt.grid(alpha=0.25)

    save_figure("01_actual_vs_predicted_2025.png")


# ============================================================
# 2. SAMPLE 7-DAY FORECAST
# ============================================================

def plot_sample_week(predictions):
    print("Generating sample 7-day forecast plot...")

    start_date = predictions["timestamp_utc"].min()

    end_date = start_date + pd.Timedelta(days=7)

    sample = predictions[
        (predictions["timestamp_utc"] >= start_date)
        & (predictions["timestamp_utc"] < end_date)
    ]

    plt.figure(figsize=(14, 6))

    plt.plot(
        sample["timestamp_utc"],
        sample["demand_mw"],
        marker="o",
        markersize=2,
        label="Actual Demand"
    )

    plt.plot(
        sample["timestamp_utc"],
        sample["predicted_demand_mw"],
        marker="o",
        markersize=2,
        label="Predicted Demand"
    )

    plt.title("SPP Actual vs Predicted Demand - First 7 Days of 2025")
    plt.xlabel("Timestamp")
    plt.ylabel("Electricity Demand (MW)")
    plt.legend()
    plt.grid(alpha=0.25)

    save_figure("02_actual_vs_predicted_first_week.png")


# ============================================================
# 3. MODEL COMPARISON - MAPE
# ============================================================

def plot_model_mape_comparison():
    print("Generating model MAPE comparison...")

    model_results = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest",
            "XGBoost",
            "Tuned XGBoost"
        ],
        "MAPE": [
            3.93,
            3.22,
            3.16,
            3.01
        ]
    })

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        model_results["Model"],
        model_results["MAPE"]
    )

    plt.title("2024 Validation MAPE by Model")
    plt.ylabel("MAPE (%)")
    plt.xlabel("Model")

    for bar, value in zip(bars, model_results["MAPE"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.03,
            f"{value:.2f}%",
            ha="center"
        )

    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.25)

    save_figure("03_model_mape_comparison.png")


# ============================================================
# 4. MODEL COMPARISON - MAE
# ============================================================

def plot_model_mae_comparison():
    print("Generating model MAE comparison...")

    model_results = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest",
            "XGBoost",
            "Tuned XGBoost"
        ],
        "MAE": [
            1347.18,
            1109.69,
            1074.52,
            1023.97
        ]
    })

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        model_results["Model"],
        model_results["MAE"]
    )

    plt.title("2024 Validation MAE by Model")
    plt.ylabel("MAE (MW)")
    plt.xlabel("Model")

    for bar, value in zip(bars, model_results["MAE"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 15,
            f"{value:,.0f}",
            ha="center"
        )

    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.25)

    save_figure("04_model_mae_comparison.png")


# ============================================================
# 5. VALIDATION VS TEST
# ============================================================

def plot_validation_vs_test():
    print("Generating validation vs test comparison...")

    results = pd.DataFrame({
        "Dataset": [
            "2024 Validation",
            "2025 Test"
        ],
        "MAPE": [
            3.01,
            3.18
        ]
    })

    plt.figure(figsize=(8, 6))

    bars = plt.bar(
        results["Dataset"],
        results["MAPE"]
    )

    plt.title("Tuned XGBoost Validation vs Final Test Performance")
    plt.ylabel("MAPE (%)")

    for bar, value in zip(bars, results["MAPE"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.03,
            f"{value:.2f}%",
            ha="center"
        )

    plt.grid(axis="y", alpha=0.25)

    save_figure("05_validation_vs_test_mape.png")


# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================

def plot_feature_importance():
    print("Generating feature importance plot...")

    feature_importance = pd.DataFrame({
        "Feature": [
            "demand_lag_24h",
            "demand_lag_168h",
            "wichita_temperature",
            "oklahoma_temperature",
            "kc_temperature",
            "omaha_temperature",
            "wichita_dew_point_temperature",
            "is_weekend",
            "omaha_dew_point_temperature",
            "day_of_week"
        ],
        "Importance": [
            0.477794,
            0.162307,
            0.143715,
            0.068996,
            0.039848,
            0.018453,
            0.014766,
            0.011943,
            0.011434,
            0.010416
        ]
    })

    feature_importance = feature_importance.sort_values(
        "Importance"
    )

    plt.figure(figsize=(11, 7))

    plt.barh(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )

    plt.title("Final XGBoost Feature Importance")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.grid(axis="x", alpha=0.25)

    save_figure("06_feature_importance.png")


# ============================================================
# 7. ERROR BY HOUR
# ============================================================

def plot_error_by_hour(predictions):
    print("Generating error by hour plot...")

    hourly = (
        predictions.groupby("hour")
        .agg(
            MAE=("absolute_error_mw", "mean"),
            MAPE=("ape", "mean")
        )
        .reset_index()
    )

    plt.figure(figsize=(11, 6))

    plt.plot(
        hourly["hour"],
        hourly["MAE"],
        marker="o"
    )

    plt.title("2025 Forecast Error by Hour of Day")
    plt.xlabel("Hour of Day (UTC)")
    plt.ylabel("Mean Absolute Error (MW)")
    plt.xticks(range(24))
    plt.grid(alpha=0.25)

    save_figure("07_error_by_hour.png")


# ============================================================
# 8. ERROR BY MONTH
# ============================================================

def plot_error_by_month(predictions):
    print("Generating error by month plot...")

    monthly = (
        predictions.groupby("month")
        .agg(
            MAE=("absolute_error_mw", "mean"),
            MAPE=("ape", "mean")
        )
        .reset_index()
    )

    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec"
    ]

    monthly["month_name"] = monthly["month"].apply(
        lambda x: month_names[x - 1]
    )

    plt.figure(figsize=(11, 6))

    bars = plt.bar(
        monthly["month_name"],
        monthly["MAPE"]
    )

    plt.title("2025 Forecast MAPE by Month")
    plt.xlabel("Month")
    plt.ylabel("MAPE (%)")
    plt.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, monthly["MAPE"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.03,
            f"{value:.2f}",
            ha="center",
            fontsize=8
        )

    save_figure("08_error_by_month.png")


# ============================================================
# 9. ERROR DISTRIBUTION
# ============================================================

def plot_error_distribution(predictions):
    print("Generating error distribution plot...")

    plt.figure(figsize=(10, 6))

    plt.hist(
        predictions["error_mw"],
        bins=60,
        edgecolor="black",
        alpha=0.8
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=1.5,
        label="Zero Error"
    )

    plt.title("Distribution of 2025 Forecast Errors")
    plt.xlabel("Forecast Error (Actual - Predicted) MW")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    save_figure("09_error_distribution.png")


# ============================================================
# 10. ACTUAL VS PREDICTED SCATTER
# ============================================================

def plot_actual_vs_predicted_scatter(predictions):
    print("Generating actual vs predicted scatter plot...")

    plt.figure(figsize=(8, 8))

    plt.scatter(
        predictions["demand_mw"],
        predictions["predicted_demand_mw"],
        alpha=0.35,
        s=12
    )

    minimum = min(
        predictions["demand_mw"].min(),
        predictions["predicted_demand_mw"].min()
    )

    maximum = max(
        predictions["demand_mw"].max(),
        predictions["predicted_demand_mw"].max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        label="Perfect Prediction"
    )

    plt.title("Actual vs Predicted SPP Demand - 2025")
    plt.xlabel("Actual Demand (MW)")
    plt.ylabel("Predicted Demand (MW)")
    plt.legend()
    plt.grid(alpha=0.25)

    save_figure("10_actual_vs_predicted_scatter.png")


# ============================================================
# 11. WORST ERROR PERIOD
# ============================================================

def plot_worst_error_period(predictions):
    print("Generating worst error period plot...")

    worst_index = predictions["absolute_error_mw"].idxmax()

    worst_timestamp = pd.Timestamp(
        predictions.loc[worst_index, "timestamp_utc"]
    )

    start = worst_timestamp - pd.Timedelta(hours=24)
    end = worst_timestamp + pd.Timedelta(hours=24)

    window = predictions[
        (predictions["timestamp_utc"] >= start)
        & (predictions["timestamp_utc"] <= end)
    ].copy()

    plt.figure(figsize=(14, 6))

    plt.plot(
        window["timestamp_utc"],
        window["demand_mw"],
        marker="o",
        markersize=3,
        label="Actual Demand"
    )

    plt.plot(
        window["timestamp_utc"],
        window["predicted_demand_mw"],
        marker="o",
        markersize=3,
        label="Predicted Demand"
    )

    plt.axvline(
        worst_timestamp,
        linestyle="--",
        linewidth=1.5,
        label="Worst Prediction"
    )

    plt.title(
        "Forecast Behavior Around Largest 2025 Error"
    )

    plt.xlabel("Timestamp")
    plt.ylabel("Electricity Demand (MW)")
    plt.legend()
    plt.grid(alpha=0.25)

    save_figure("11_worst_error_period.png")


# ============================================================
# 12. ERROR VS ACTUAL DEMAND
# ============================================================

def plot_error_vs_demand(predictions):
    print("Generating error vs demand plot...")

    plt.figure(figsize=(10, 6))

    plt.scatter(
        predictions["demand_mw"],
        predictions["absolute_error_mw"],
        alpha=0.35,
        s=12
    )

    plt.title("Forecast Error vs Actual Electricity Demand - 2025")
    plt.xlabel("Actual Demand (MW)")
    plt.ylabel("Absolute Forecast Error (MW)")
    plt.grid(alpha=0.25)

    save_figure("12_error_vs_actual_demand.png")


# ============================================================
# 13. DAILY AVERAGE ACTUAL VS PREDICTED
# ============================================================

def plot_daily_average_forecast(predictions):
    print("Generating daily average demand plot...")

    daily = (
        predictions
        .set_index("timestamp_utc")
        [["demand_mw", "predicted_demand_mw"]]
        .resample("D")
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(16, 6))

    plt.plot(
        daily["timestamp_utc"],
        daily["demand_mw"],
        label="Actual Daily Average"
    )

    plt.plot(
        daily["timestamp_utc"],
        daily["predicted_demand_mw"],
        label="Predicted Daily Average"
    )

    plt.title("Daily Average SPP Demand - 2025")
    plt.xlabel("Date")
    plt.ylabel("Average Demand (MW)")
    plt.legend()
    plt.grid(alpha=0.25)

    save_figure("13_daily_average_actual_vs_predicted.png")


# ============================================================
# 14. DEMAND DISTRIBUTION
# ============================================================

def plot_demand_distribution(predictions):
    print("Generating demand distribution plot...")

    plt.figure(figsize=(10, 6))

    plt.hist(
        predictions["demand_mw"],
        bins=50,
        alpha=0.6,
        label="Actual"
    )

    plt.hist(
        predictions["predicted_demand_mw"],
        bins=50,
        alpha=0.6,
        label="Predicted"
    )

    plt.title("Actual and Predicted Demand Distributions - 2025")
    plt.xlabel("Electricity Demand (MW)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)

    save_figure("14_demand_distribution.png")


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 60)
    print("GENERATING REPORT VISUALIZATIONS")
    print("=" * 60)
    print()

    features, predictions = load_data()

    plot_full_2025_forecast(predictions)
    plot_sample_week(predictions)

    plot_model_mape_comparison()
    plot_model_mae_comparison()
    plot_validation_vs_test()

    plot_feature_importance()

    plot_error_by_hour(predictions)
    plot_error_by_month(predictions)

    plot_error_distribution(predictions)
    plot_actual_vs_predicted_scatter(predictions)

    plot_worst_error_period(predictions)
    plot_error_vs_demand(predictions)

    plot_daily_average_forecast(predictions)
    plot_demand_distribution(predictions)

    print()
    print("=" * 60)
    print("REPORT VISUALIZATIONS COMPLETE")
    print("=" * 60)
    print()
    print(f"Figures saved to:")
    print(OUTPUT_DIR)
    print()


if __name__ == "__main__":
    main()