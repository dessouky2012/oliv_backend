# aggregate_data.py
import pandas as pd

# Load your DLD dataset
# Make sure dld_data.csv is in the same directory or provide a full path
dld_data = pd.read_csv("dld_data.csv")

# We assume the bedrooms are in a column like 'ROOMS_EN' which might say "1 B/R", "Studio", etc.
# Check your data. If bedrooms are stored differently, adjust accordingly.
# We will trust the sample data you provided that uses "Studio", "1 B/R", "2 B/R", etc.

# Group by AREA_EN, PROP_TYPE_EN, ROOMS_EN and compute min, max, median price and median area
price_stats = dld_data.groupby(["AREA_EN", "PROP_TYPE_EN", "ROOMS_EN"]).agg(
    MIN_PRICE=("TRANS_VALUE", "min"),
    MAX_PRICE=("TRANS_VALUE", "max"),
    MEDIAN_PRICE=("TRANS_VALUE", "median"),
    MEDIAN_AREA=("ACTUAL_AREA", "median")
).reset_index()

# Save to CSV
price_stats.to_csv("price_stats.csv", index=False)

print("price_stats.csv created successfully.")