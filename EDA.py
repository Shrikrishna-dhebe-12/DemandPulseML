# ============================================================
# DEMANDPULSE
# STEP 3 : EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("amazon_multi_product_demand.csv")

df["date"] = pd.to_datetime(df["date"])

print("\n" + "=" * 70)
print("DEMANDPULSE - STEP 3 : EDA")
print("=" * 70)

print("\nDataset loaded successfully.")
print("Rows    :", df.shape[0])
print("Columns :", df.shape[1])


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("BASIC INFORMATION")
print("=" * 70)

print(df.info())


# ============================================================
# 3. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(df.isnull().sum())


# ============================================================
# 4. DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE CHECK")
print("=" * 70)

print("Duplicate Rows :", df.duplicated().sum())


# ============================================================
# 5. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

print(df.describe())


# ============================================================
# 6. TARGET ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("TARGET ANALYSIS")
print("=" * 70)

print("Target Variable : actual_demand")

print("\nDemand Statistics:")
print(df["actual_demand"].describe())

print("\nMinimum Demand :", df["actual_demand"].min())
print("Maximum Demand :", df["actual_demand"].max())
print("Average Demand :", round(df["actual_demand"].mean(), 2))
print("Median Demand  :", df["actual_demand"].median())


# ============================================================
# 7. PRODUCT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PRODUCT ANALYSIS")
print("=" * 70)

product_summary = (
    df.groupby("product_name")
    .agg(
        total_demand=("actual_demand", "sum"),
        average_demand=("actual_demand", "mean"),
        total_units_sold=("units_sold", "sum"),
        average_price=("selling_price", "mean"),
        total_page_views=("page_views", "sum"),
        total_ad_spend=("ad_spend_inr", "sum")
    )
    .sort_values("total_demand", ascending=False)
)

print(product_summary)


# ============================================================
# 8. DEMAND BY PRODUCT - BAR CHART
# ============================================================

plt.figure(figsize=(12, 6))

product_demand = (
    df.groupby("product_name")["actual_demand"]
    .sum()
    .sort_values(ascending=False)
)

product_demand.plot(kind="bar")

plt.title("Total Demand by Product")
plt.xlabel("Product")
plt.ylabel("Total Demand")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# ============================================================
# 9. DAILY DEMAND TREND
# ============================================================

daily_demand = (
    df.groupby("date")["actual_demand"]
    .sum()
)

plt.figure(figsize=(14, 6))

plt.plot(daily_demand.index, daily_demand.values)

plt.title("Daily Demand Trend")
plt.xlabel("Date")
plt.ylabel("Total Demand")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# 10. 7-DAY MOVING AVERAGE
# ============================================================

daily_demand_df = daily_demand.to_frame(name="actual_demand")

daily_demand_df["rolling_7_day"] = (
    daily_demand_df["actual_demand"]
    .rolling(7)
    .mean()
)

plt.figure(figsize=(14, 6))

plt.plot(
    daily_demand_df.index,
    daily_demand_df["actual_demand"],
    alpha=0.4,
    label="Daily Demand"
)

plt.plot(
    daily_demand_df.index,
    daily_demand_df["rolling_7_day"],
    linewidth=2,
    label="7-Day Moving Average"
)

plt.title("Daily Demand with 7-Day Moving Average")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ============================================================
# 11. MONTHLY DEMAND
# ============================================================

df["month"] = df["date"].dt.month

monthly_demand = (
    df.groupby("month")["actual_demand"]
    .sum()
)

print("\n" + "=" * 70)
print("MONTHLY DEMAND")
print("=" * 70)

print(monthly_demand)


plt.figure(figsize=(10, 6))

monthly_demand.plot(kind="bar")

plt.title("Total Demand by Month")
plt.xlabel("Month")
plt.ylabel("Total Demand")
plt.tight_layout()
plt.show()


# ============================================================
# 12. YEARLY DEMAND
# ============================================================

df["year"] = df["date"].dt.year

yearly_demand = (
    df.groupby("year")["actual_demand"]
    .sum()
)

print("\n" + "=" * 70)
print("YEARLY DEMAND")
print("=" * 70)

print(yearly_demand)


plt.figure(figsize=(8, 5))

yearly_demand.plot(kind="bar")

plt.title("Total Demand by Year")
plt.xlabel("Year")
plt.ylabel("Total Demand")
plt.tight_layout()
plt.show()


# ============================================================
# 13. WEEKEND VS WEEKDAY DEMAND
# ============================================================

weekend_demand = (
    df.groupby("is_weekend")["actual_demand"]
    .agg(["count", "mean", "sum"])
)

print("\n" + "=" * 70)
print("WEEKEND VS WEEKDAY DEMAND")
print("=" * 70)

print(weekend_demand)


plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="is_weekend",
    y="actual_demand"
)

plt.title("Weekend vs Weekday Demand")
plt.xlabel("Weekend (0 = No, 1 = Yes)")
plt.ylabel("Demand")
plt.tight_layout()
plt.show()


# ============================================================
# 14. SALE SEASON VS NORMAL DEMAND
# ============================================================

sale_demand = (
    df.groupby("is_sale_season")["actual_demand"]
    .agg(["count", "mean", "sum"])
)

print("\n" + "=" * 70)
print("SALE SEASON VS NORMAL DEMAND")
print("=" * 70)

print(sale_demand)


plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="is_sale_season",
    y="actual_demand"
)

plt.title("Sale Season vs Normal Demand")
plt.xlabel("Sale Season (0 = No, 1 = Yes)")
plt.ylabel("Demand")
plt.tight_layout()
plt.show()


# ============================================================
# 15. DISCOUNT VS DEMAND
# ============================================================

discount_demand = (
    df.groupby("discount_percent")["actual_demand"]
    .mean()
)

print("\n" + "=" * 70)
print("DISCOUNT VS DEMAND")
print("=" * 70)

print(discount_demand)


plt.figure(figsize=(10, 6))

plt.scatter(
    df["discount_percent"],
    df["actual_demand"],
    alpha=0.5
)

plt.title("Discount Percentage vs Demand")
plt.xlabel("Discount %")
plt.ylabel("Demand")
plt.tight_layout()
plt.show()


# ============================================================
# 16. SELLING PRICE VS DEMAND
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["selling_price"],
    df["actual_demand"],
    alpha=0.5
)

plt.title("Selling Price vs Demand")
plt.xlabel("Selling Price")
plt.ylabel("Demand")
plt.tight_layout()
plt.show()


# ============================================================
# 17. PAGE VIEWS VS DEMAND
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["page_views"],
    df["actual_demand"],
    alpha=0.5
)

plt.title("Page Views vs Demand")
plt.xlabel("Page Views")
plt.ylabel("Demand")
plt.tight_layout()
plt.show()


# ============================================================
# 18. AD SPEND VS DEMAND
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["ad_spend_inr"],
    df["actual_demand"],
    alpha=0.5
)

plt.title("Advertising Spend vs Demand")
plt.xlabel("Ad Spend (INR)")
plt.ylabel("Demand")
plt.tight_layout()
plt.show()


# ============================================================
# 19. INVENTORY VS DEMAND
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["inventory_level"],
    df["actual_demand"],
    alpha=0.5
)

plt.title("Inventory Level vs Actual Demand")
plt.xlabel("Inventory Level")
plt.ylabel("Actual Demand")
plt.tight_layout()
plt.show()


# ============================================================
# 20. STOCKOUT ANALYSIS
# ============================================================

stockout_summary = (
    df.groupby("is_stockout")
    .agg(
        records=("is_stockout", "count"),
        average_demand=("actual_demand", "mean"),
        average_units_sold=("units_sold", "mean")
    )
)

print("\n" + "=" * 70)
print("STOCKOUT ANALYSIS")
print("=" * 70)

print(stockout_summary)


plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="is_stockout",
    y="actual_demand"
)

plt.title("Demand Distribution by Stockout Status")
plt.xlabel("Stockout (0 = No, 1 = Yes)")
plt.ylabel("Actual Demand")
plt.tight_layout()
plt.show()


# ============================================================
# 21. ACTUAL DEMAND VS UNITS SOLD
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    df["actual_demand"],
    df["units_sold"],
    alpha=0.5
)

plt.title("Actual Demand vs Units Sold")
plt.xlabel("Actual Demand")
plt.ylabel("Units Sold")
plt.tight_layout()
plt.show()


# ============================================================
# 22. CORRELATION ANALYSIS
# ============================================================

numerical_columns = [
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

correlation_matrix = df[numerical_columns].corr()

print("\n" + "=" * 70)
print("CORRELATION WITH ACTUAL DEMAND")
print("=" * 70)

print(
    correlation_matrix["actual_demand"]
    .sort_values(ascending=False)
)


# ============================================================
# 23. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()


# ============================================================
# 24. TOP HIGH-DEMAND RECORDS
# ============================================================

print("\n" + "=" * 70)
print("TOP 10 HIGH-DEMAND RECORDS")
print("=" * 70)

top_demand = (
    df[
        [
            "date",
            "product_name",
            "selling_price",
            "discount_percent",
            "page_views",
            "ad_spend_inr",
            "inventory_level",
            "actual_demand",
            "is_stockout"
        ]
    ]
    .sort_values("actual_demand", ascending=False)
    .head(10)
)

print(top_demand)


# ============================================================
# 25. LOW-DEMAND RECORDS
# ============================================================

print("\n" + "=" * 70)
print("10 LOW-DEMAND RECORDS")
print("=" * 70)

low_demand = (
    df[
        [
            "date",
            "product_name",
            "selling_price",
            "discount_percent",
            "page_views",
            "ad_spend_inr",
            "inventory_level",
            "actual_demand",
            "is_stockout"
        ]
    ]
    .sort_values("actual_demand")
    .head(10)
)

print(low_demand)


# ============================================================
# 26. PRODUCT-WISE DEMAND SUMMARY
# ============================================================

product_demand_summary = (
    df.groupby("product_name")
    .agg(
        min_demand=("actual_demand", "min"),
        max_demand=("actual_demand", "max"),
        avg_demand=("actual_demand", "mean"),
        total_demand=("actual_demand", "sum")
    )
    .sort_values("avg_demand", ascending=False)
)

print("\n" + "=" * 70)
print("PRODUCT-WISE DEMAND SUMMARY")
print("=" * 70)

print(product_demand_summary)


# ============================================================
# 27. SALE SEASON DEMAND BOOST
# ============================================================

normal_demand = df.loc[
    df["is_sale_season"] == 0,
    "actual_demand"
].mean()

sale_season_demand = df.loc[
    df["is_sale_season"] == 1,
    "actual_demand"
].mean()

print("\n" + "=" * 70)
print("SALE SEASON DEMAND COMPARISON")
print("=" * 70)

print("Normal Average Demand     :", round(normal_demand, 2))
print("Sale Season Average Demand:", round(sale_season_demand, 2))

if normal_demand != 0:
    boost_percentage = (
        (sale_season_demand - normal_demand)
        / normal_demand
    ) * 100

    print(
        "Sale Season Demand Boost :",
        round(boost_percentage, 2),
        "%"
    )


# ============================================================
# 28. INVENTORY RISK ANALYSIS
# ============================================================

df["inventory_demand_ratio"] = (
    df["inventory_level"] /
    df["actual_demand"]
)

print("\n" + "=" * 70)
print("INVENTORY / DEMAND RATIO")
print("=" * 70)

print(
    df["inventory_demand_ratio"]
    .describe()
)


# ============================================================
# 29. HIGH STOCKOUT RISK RECORDS
# ============================================================

high_risk = df[
    df["inventory_demand_ratio"] < 1
].copy()

print("\n" + "=" * 70)
print("STOCKOUT RISK RECORDS")
print("=" * 70)

print("High Risk Records :", len(high_risk))

print(
    high_risk[
        [
            "date",
            "product_name",
            "inventory_level",
            "actual_demand",
            "units_sold",
            "is_stockout"
        ]
    ].head(10)
)


# ============================================================
# 30. SAVE EDA SUMMARY
# ============================================================

product_demand_summary.to_csv(
    "product_demand_summary.csv"
)

correlation_matrix.to_csv(
    "demand_correlation_matrix.csv"
)

stockout_summary.to_csv(
    "stockout_summary.csv"
)

print("\nEDA summary files saved successfully.")


# ============================================================
# STEP 3 COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("✅ STEP 3 : EDA COMPLETED SUCCESSFULLY")
print("=" * 70)