import csv
import json
from pathlib import Path
from datetime import datetime


# -----------------------------
# File paths
# -----------------------------

EBA_PATH = Path("data/raw/electricity/EBA.txt")
OUTPUT_PATH = Path("data/processed/swpp_demand_2020_2025.csv")


# -----------------------------
# Configuration
# -----------------------------

TARGET_SERIES = "EBA.SWPP-ALL.D.H"

START_DATE = datetime(2020, 1, 1, 0)
END_DATE = datetime(2025, 12, 31, 23)


# -----------------------------
# Create output directory
# -----------------------------

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Search EBA bulk file
# -----------------------------

print("Searching for SWPP demand series...")

target_data = None

with EBA_PATH.open("r", encoding="utf-8") as file:

    for line in file:

        obj = json.loads(line)

        if obj.get("series_id") == TARGET_SERIES:

            print("Found:", obj["series_id"])
            print("Name:", obj["name"])
            print("Units:", obj["units"])

            target_data = obj["data"]

            break


if target_data is None:
    raise ValueError(f"Could not find {TARGET_SERIES}")


# -----------------------------
# Filter dates
# -----------------------------

rows = []

for timestamp, value in target_data:

    dt = datetime.strptime(timestamp, "%Y%m%dT%H")

    if START_DATE <= dt <= END_DATE:

        rows.append([
            dt.strftime("%Y-%m-%d %H:%M:%S"),
            value
        ])


# Sort oldest → newest
rows.sort(key=lambda row: row[0])


# -----------------------------
# Write CSV
# -----------------------------

with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "timestamp_utc",
        "demand_mwh"
    ])

    writer.writerows(rows)


# -----------------------------
# Summary
# -----------------------------

print()
print("Extraction complete!")
print("Rows extracted:", len(rows))
print("First row:", rows[0])
print("Last row:", rows[-1])
print("Saved to:", OUTPUT_PATH)