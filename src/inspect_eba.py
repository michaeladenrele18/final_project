import json
from pathlib import Path

EBA_PATH = Path("data/raw/electricity/EBA.txt")

with EBA_PATH.open("r", encoding="utf-8") as file:
    for line in file:
        obj = json.loads(line)

        series_id = obj.get("series_id", "")
        name = obj.get("name", "")

        if "SWPP" in series_id or "Southwest Power Pool" in name:
            if "demand" in name.lower():
                print("\nFOUND SERIES")
                print("Series ID:", series_id)
                print("Name:", name)
                print("Units:", obj.get("units"))
                print("Frequency:", obj.get("f"))
                print("Start:", obj.get("start"))
                print("End:", obj.get("end"))
                print("First 5 data points:", obj.get("data", [])[:5])