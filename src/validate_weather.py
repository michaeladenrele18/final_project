import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RAW_WEATHER_DIR = Path("data/raw/weather")
PROCESSED_WEATHER_DIR = Path("data/processed/weather")

START_YEAR = 2020
END_YEAR = 2025

# Only interpolate gaps this many hours or shorter
MAX_INTERPOLATION_GAP = 6


# --------------------------------------------------
# Station configuration
# --------------------------------------------------

stations = {
    "kansas_city": {
        "station_id": "USW00003947"
    },

    "wichita": {
        "station_id": "USW00003928"
    },

    "oklahoma": {
        "station_id": "USW00013967"
    },

    "omaha": {
        "station_id": "USW00014942"
    },

    "fargo": {
        "station_id": "USW00014914"
    }
}


# --------------------------------------------------
# Columns
# --------------------------------------------------

columns_to_keep = [
    "STATION",
    "Station_name",
    "DATE",
    "temperature",
    "dew_point_temperature",
    "relative_humidity",
    "wind_speed"
]

weather_columns = [
    "temperature",
    "dew_point_temperature",
    "relative_humidity",
    "wind_speed"
]


# --------------------------------------------------
# Interpolate ONLY short gaps
# --------------------------------------------------

def interpolate_short_gaps(series, max_gap=6):

    # First calculate normal time-based interpolation
    interpolated = series.interpolate(method="time")

    # Find which values were originally missing
    missing = series.isna()

    # Assign a group number whenever missing/non-missing changes
    groups = missing.ne(missing.shift()).cumsum()

    # Go through each consecutive region
    for _, group in series.groupby(groups):

        group_index = group.index

        # Skip groups that are not missing
        if not missing.loc[group_index].all():
            continue

        gap_length = len(group_index)

        # If gap is longer than allowed,
        # restore it to NaN
        if gap_length > max_gap:
            interpolated.loc[group_index] = pd.NA

    return interpolated


# --------------------------------------------------
# Process one year for one station
# --------------------------------------------------

def process_year(city_name, station_id, year):

    input_path = (
        RAW_WEATHER_DIR
        / city_name
        / f"GHCNh_{station_id}_{year}.psv"
    )

    print("\n" + "-" * 60)
    print(f"Processing {city_name} - {year}")
    print("File:", input_path)

    # --------------------------------------------------
    # Load only columns we actually care about
    #
    # This also avoids the DtypeWarnings we were seeing
    # from unrelated NOAA columns.
    # --------------------------------------------------

    df = pd.read_csv(
        input_path,
        sep="|",
        usecols=columns_to_keep
    )

    # --------------------------------------------------
    # Convert timestamp
    # --------------------------------------------------

    df["DATE"] = pd.to_datetime(
        df["DATE"],
        errors="coerce"
    )

    # Remove rows where DATE could not be parsed
    df = df.dropna(subset=["DATE"])

    # --------------------------------------------------
    # Convert weather columns to numeric
    # --------------------------------------------------

    for column in weather_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------
    # Validate weather ranges
    #
    # NOAA files can occasionally contain malformed or
    # physically unrealistic values. Convert those values
    # to missing values so the normal gap-handling logic
    # below can decide whether they should be interpolated.
    # --------------------------------------------------

    valid_ranges = {
        "temperature": (-60, 60),
        "dew_point_temperature": (-70, 40),
        "relative_humidity": (0, 100),
        "wind_speed": (0, 75)
    }

    for column, (minimum, maximum) in valid_ranges.items():

        invalid = (
            (df[column] < minimum)
            | (df[column] > maximum)
        )

        invalid_count = invalid.sum()

        if invalid_count > 0:
            print(
                f"Invalid {column} values replaced with NaN:",
                invalid_count
            )

        df.loc[invalid, column] = pd.NA

    # --------------------------------------------------
    # Assign each observation to an hour
    # --------------------------------------------------

    df["hour"] = df["DATE"].dt.floor("h")

    # --------------------------------------------------
    # Determine distance from standard :53 reading
    #
    # :53 -> 0
    # :52 -> 1
    # :50 -> 3
    # :00 -> 53
    # --------------------------------------------------

    df["distance_from_53"] = abs(
        df["DATE"].dt.minute - 53
    )

    # --------------------------------------------------
    # Sort so closest reading to :53 comes first
    # --------------------------------------------------

    df = df.sort_values(
        by=[
            "hour",
            "distance_from_53",
            "DATE"
        ]
    )

    # --------------------------------------------------
    # Select one observation per hour
    # --------------------------------------------------

    hourly_df = (
        df
        .drop_duplicates(
            subset="hour",
            keep="first"
        )
        .copy()
    )

    # --------------------------------------------------
    # Normalize to top of hour
    # --------------------------------------------------

    hourly_df["timestamp_utc"] = (
        hourly_df["hour"]
    )

    # --------------------------------------------------
    # Keep final fields
    # --------------------------------------------------

    hourly_df = hourly_df[
        [
            "timestamp_utc",
            "STATION",
            "Station_name",
            "temperature",
            "dew_point_temperature",
            "relative_humidity",
            "wind_speed"
        ]
    ]

    hourly_df = hourly_df.sort_values(
        "timestamp_utc"
    )

    # --------------------------------------------------
    # Expected timeline
    # --------------------------------------------------

    expected_hours = pd.date_range(
        start=f"{year}-01-01 00:00:00",
        end=f"{year}-12-31 23:00:00",
        freq="h"
    )

    actual_hours = pd.DatetimeIndex(
        hourly_df["timestamp_utc"]
    )

    missing_hours = expected_hours.difference(
        actual_hours
    )

    print("Expected hours:", len(expected_hours))

    print(
        "Actual hourly observations:",
        len(hourly_df)
    )

    print(
        "Missing entire hours:",
        len(missing_hours)
    )

    # --------------------------------------------------
    # Reindex onto complete timeline
    # --------------------------------------------------

    hourly_df = (
        hourly_df
        .set_index("timestamp_utc")
        .reindex(expected_hours)
    )

    hourly_df.index.name = "timestamp_utc"

    # --------------------------------------------------
    # Show missing values BEFORE cleaning
    # --------------------------------------------------

    print("\nMissing values before interpolation:")

    print(
        hourly_df[
            weather_columns
        ].isna().sum()
    )

    # --------------------------------------------------
    # Interpolate ONLY short gaps
    # --------------------------------------------------

    for column in weather_columns:

        hourly_df[column] = interpolate_short_gaps(
            hourly_df[column],
            max_gap=MAX_INTERPOLATION_GAP
        )

    # --------------------------------------------------
    # Station metadata
    #
    # Metadata isn't a weather measurement, so it's
    # completely fine to fill this across long gaps.
    # --------------------------------------------------

    hourly_df["STATION"] = (
        hourly_df["STATION"]
        .ffill()
        .bfill()
    )

    hourly_df["Station_name"] = (
        hourly_df["Station_name"]
        .ffill()
        .bfill()
    )

    # --------------------------------------------------
    # Check what remains missing
    # --------------------------------------------------

    print(
        "\nMissing values after short-gap interpolation:"
    )

    print(
        hourly_df[
            weather_columns
        ].isna().sum()
    )

    hourly_df = hourly_df.reset_index()

    return hourly_df


# --------------------------------------------------
# Process complete station
# --------------------------------------------------

def process_station(city_name, station_id):

    print("\n")
    print("=" * 60)

    print(
        f"PROCESSING STATION: {city_name.upper()}"
    )

    print("=" * 60)

    all_years = []

    # --------------------------------------------------
    # Process each year
    # --------------------------------------------------

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        yearly_df = process_year(
            city_name,
            station_id,
            year
        )

        all_years.append(yearly_df)

    # --------------------------------------------------
    # Combine years
    # --------------------------------------------------

    station_df = pd.concat(
        all_years,
        ignore_index=True
    )

    station_df = station_df.sort_values(
        "timestamp_utc"
    )

    # --------------------------------------------------
    # Expected complete 2020-2025 timeline
    # --------------------------------------------------

    expected_timeline = pd.date_range(
        start=f"{START_YEAR}-01-01 00:00:00",
        end=f"{END_YEAR}-12-31 23:00:00",
        freq="h"
    )

    actual_timeline = pd.DatetimeIndex(
        station_df["timestamp_utc"]
    )

    missing_final = (
        expected_timeline.difference(
            actual_timeline
        )
    )

    # --------------------------------------------------
    # Final validation
    # --------------------------------------------------

    print("\n")
    print("=" * 50)

    print(
        f"FINAL VALIDATION: {city_name.upper()}"
    )

    print("=" * 50)

    print(
        "Expected rows:",
        len(expected_timeline)
    )

    print(
        "Actual rows:",
        len(station_df)
    )

    print(
        "Missing timestamps:",
        len(missing_final)
    )

    print(
        "Duplicate timestamps:",
        station_df[
            "timestamp_utc"
        ].duplicated().sum()
    )

    print("\nRemaining missing weather values:")

    print(
        station_df[
            weather_columns
        ].isna().sum()
    )

    print(
        "\nRows containing at least one "
        "missing weather feature:",
        station_df[
            weather_columns
        ].isna().any(axis=1).sum()
    )

    print(
        "\nFirst timestamp:",
        station_df["timestamp_utc"].min()
    )

    print(
        "Last timestamp:",
        station_df["timestamp_utc"].max()
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    output_path = (
        PROCESSED_WEATHER_DIR
        / (
            f"{city_name}_weather_"
            f"{START_YEAR}_{END_YEAR}.csv"
        )
    )

    PROCESSED_WEATHER_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    station_df.to_csv(
        output_path,
        index=False
    )

    print("\nSaved to:", output_path)


# --------------------------------------------------
# Run all stations
# --------------------------------------------------

for city_name, station_info in stations.items():

    process_station(
        city_name,
        station_info["station_id"]
    )