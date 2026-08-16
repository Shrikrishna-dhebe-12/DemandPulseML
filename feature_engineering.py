# ============================================================
# DEMANDPULSE
# STEP 4 : TIME-SERIES FEATURE ENGINEERING
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("amazon_multi_product_demand.csv")

df["date"] = pd.to_datetime(df["date"])

print("\n" + "=" * 70)
print("DEMANDPULSE - STEP 4 : FEATURE ENGINEERING")
print("=" * 70)

print("\nOriginal Shape:")
print(df.shape)


# ============================================================
# 2. SORT DATA
# ============================================================

# Time-series data must be sorted by date
df = df.sort_values(
    ["asin", "date"]
).reset_index(drop=True)

print("\nData sorted by product and date.")


# ============================================================
# 3. DATE FEATURES
# ============================================================

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek
df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
df["day_of_year"] = df["date"].dt.dayofyear
df["quarter"] = df["date"].dt.quarter

print("\nDate features created.")


# ============================================================
# 4. WEEKEND FEATURE
# ============================================================

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)


# ============================================================
# 5. CYCLICAL TIME FEATURES
# ============================================================

# Day of week → cyclical representation
df["day_of_week_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

df["day_of_week_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)


# Month → cyclical representation
df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)


# Day of year → cyclical representation
df["day_of_year_sin"] = np.sin(
    2 * np.pi * df["day_of_year"] / 365
)

df["day_of_year_cos"] = np.cos(
    2 * np.pi * df["day_of_year"] / 365
)

print("Cyclical time features created.")


# ============================================================
# 6. PRICE FEATURES
# ============================================================

df["discount_amount"] = (
    df["mrp"] -
    df["selling_price"]
)

df["price_discount_ratio"] = (
    df["discount_amount"] /
    df["mrp"]
)

print("Price features created.")


# ============================================================
# 7. MARKETING FEATURES
# ============================================================

df["ad_spend_per_view"] = (
    df["ad_spend_inr"] /
    df["page_views"].replace(0, np.nan)
)

df["ad_spend_per_view"] = (
    df["ad_spend_per_view"]
    .fillna(0)
)

print("Marketing features created.")


# ============================================================
# 8. INVENTORY FEATURES
# ============================================================

df["inventory_demand_ratio"] = (
    df["inventory_level"] /
    df["actual_demand"].replace(0, np.nan)
)

df["inventory_demand_ratio"] = (
    df["inventory_demand_ratio"]
    .fillna(0)
)

df["inventory_gap"] = (
    df["inventory_level"] -
    df["actual_demand"]
)

print("Inventory features created.")


# ============================================================
# 9. LAG FEATURES
# ============================================================

# Previous demand for same product
df["demand_lag_1"] = (
    df.groupby("asin")["actual_demand"]
    .shift(1)
)

# Demand 7 days ago
df["demand_lag_7"] = (
    df.groupby("asin")["actual_demand"]
    .shift(7)
)

# Demand 14 days ago
df["demand_lag_14"] = (
    df.groupby("asin")["actual_demand"]
    .shift(14)
)

# Demand 30 days ago
df["demand_lag_30"] = (
    df.groupby("asin")["actual_demand"]
    .shift(30)
)

print("Lag features created.")


# ============================================================
# 10. ROLLING DEMAND FEATURES
# ============================================================

# IMPORTANT:
# shift(1) prevents today's demand from leaking
# into today's rolling features.

df["demand_rolling_7"] = (
    df.groupby("asin")["actual_demand"]
    .transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
)

df["demand_rolling_14"] = (
    df.groupby("asin")["actual_demand"]
    .transform(
        lambda x: x.shift(1).rolling(14).mean()
    )
)

df["demand_rolling_30"] = (
    df.groupby("asin")["actual_demand"]
    .transform(
        lambda x: x.shift(1).rolling(30).mean()
    )
)

print("Rolling demand features created.")


# ============================================================
# 11. ROLLING DEMAND VOLATILITY
# ============================================================

df["demand_rolling_std_7"] = (
    df.groupby("asin")["actual_demand"]
    .transform(
        lambda x: x.shift(1).rolling(7).std()
    )
)

print("Demand volatility feature created.")


# ============================================================
# 12. PREVIOUS UNITS SOLD
# ============================================================

df["units_sold_lag_1"] = (
    df.groupby("asin")["units_sold"]
    .shift(1)
)

df["units_sold_lag_7"] = (
    df.groupby("asin")["units_sold"]
    .shift(7)
)

print("Previous sales features created.")


# ============================================================
# 13. PREVIOUS PRICE
# ============================================================

df["selling_price_lag_1"] = (
    df.groupby("asin")["selling_price"]
    .shift(1)
)

df["selling_price_change"] = (
    df["selling_price"] -
    df["selling_price_lag_1"]
)

print("Price change feature created.")


# ============================================================
# 14. PREVIOUS INVENTORY
# ============================================================

df["inventory_lag_1"] = (
    df.groupby("asin")["inventory_level"]
    .shift(1)
)

df["inventory_change"] = (
    df["inventory_level"] -
    df["inventory_lag_1"]
)

print("Inventory change feature created.")


# ============================================================
# 15. PRODUCT-WISE DEMAND RANK
# ============================================================

df["product_demand_rank"] = (
    df.groupby("date")["actual_demand"]
    .rank(
        method="average",
        ascending=False
    )
)

print("Product demand rank created.")


# ============================================================
# 16. FILL FEATURE MISSING VALUES
# ============================================================

# Lag and rolling features naturally create NaN
# for the first few observations of each product.

feature_columns = [
    "demand_lag_1",
    "demand_lag_7",
    "demand_lag_14",
    "demand_lag_30",
    "demand_rolling_7",
    "demand_rolling_14",
    "demand_rolling_30",
    "demand_rolling_std_7",
    "units_sold_lag_1",
    "units_sold_lag_7",
    "selling_price_lag_1",
    "selling_price_change",
    "inventory_lag_1",
    "inventory_change"
]

for column in feature_columns:

    df[column] = (
        df.groupby("asin")[column]
        .transform(
            lambda x: x.ffill()
        )
    )

    df[column] = df[column].fillna(0)


# ============================================================
# 17. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES AFTER FEATURE ENGINEERING")
print("=" * 70)

print(
    df.isnull().sum()
    .sort_values(ascending=False)
    .head(20)
)


# ============================================================
# 18. DISPLAY NEW FEATURES
# ============================================================

print("\n" + "=" * 70)
print("NEW FEATURE COLUMNS")
print("=" * 70)

original_columns = [
    "date",
    "asin",
    "product_name",
    "mrp",
    "selling_price",
    "discount_percent",
    "customer_rating",
    "total_reviews",
    "page_views",
    "ad_spend_inr",
    "is_sale_season",
    "is_weekend",
    "inventory_level",
    "units_sold",
    "is_stockout",
    "actual_demand"
]

new_features = [
    column
    for column in df.columns
    if column not in original_columns
]

for column in new_features:
    print(column)


# ============================================================
# 19. FEATURE COUNT
# ============================================================

print("\n" + "=" * 70)
print("FEATURE COUNT")
print("=" * 70)

print("Original Columns :", len(original_columns))
print("New Features     :", len(new_features))
print("Total Columns    :", len(df.columns))


# ============================================================
# 20. SAMPLE DATA
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING SAMPLE")
print("=" * 70)

sample_columns = [
    "date",
    "asin",
    "actual_demand",
    "demand_lag_1",
    "demand_lag_7",
    "demand_lag_14",
    "demand_lag_30",
    "demand_rolling_7",
    "demand_rolling_14",
    "demand_rolling_30",
    "discount_percent",
    "selling_price",
    "inventory_level"
]

print(
    df[sample_columns].head(20)
)


# ============================================================
# 21. SAVE FEATURE-ENGINEERED DATA
# ============================================================

output_file = "demandpulse_features.csv"

df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("OUTPUT FILE")
print("=" * 70)

print(
    f"✅ Feature-engineered dataset saved as: {output_file}"
)


# ============================================================
# 22. FINAL SHAPE
# ============================================================

print("\nFinal Dataset Shape:")
print(df.shape)


# ============================================================
# STEP 4 COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("✅ STEP 4 : FEATURE ENGINEERING COMPLETED")
print("=" * 70)