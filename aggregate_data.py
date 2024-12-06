import pandas as pd

dld_data = pd.read_csv("dld_data.csv")

price_stats = dld_data.groupby(["AREA_EN", "PROP_TYPE_EN", "ROOMS_EN"]).agg(
    MIN_PRICE=("TRANS_VALUE", "min"),
    MAX_PRICE=("TRANS_VALUE", "max"),
    MEDIAN_PRICE=("TRANS_VALUE", "median"),
    MEDIAN_AREA=("ACTUAL_AREA", "median")
).reset_index()

price_stats.to_csv("price_stats.csv", index=False)
print("price_stats.csv created successfully.")