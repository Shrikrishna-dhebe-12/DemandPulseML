# ============================================================
# DEMANDPULSE
# STEP 2 : DATA UNDERSTANDING
# ============================================================

import pandas as pd


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("amazon_multi_product_demand.csv")


# ============================================================
# 2. CONVERT DATE COLUMN
# ============================================================

df["date"] = pd.to_datetime(df["date"])


# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET BASIC INFORMATION")
print("=" * 60)

print("Number of Rows    :", df.shape[0])
print("Number of Columns :", df.shape[1])


# ============================================================
# 4. DISPLAY FIRST 5 ROWS
# ============================================================

print("\n" + "=" * 60)
print("FIRST 5 ROWS")
print("=" * 60)

print(df.head())


# ============================================================
# 5. DISPLAY LAST 5 ROWS
# ============================================================

print("\n" + "=" * 60)
print("LAST 5 ROWS")
print("=" * 60)

print(df.tail())


# ============================================================
# 6. COLUMN NAMES
# ============================================================

print("\n" + "=" * 60)
print("COLUMN NAMES")
print("=" * 60)

for column in df.columns:
    print(column)


# ============================================================
# 7. DATA TYPES
# ============================================================

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)


# ============================================================
# 8. DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFO")
print("=" * 60)

df.info()


# ============================================================
# 9. MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()

print(missing_values)


# ============================================================
# 10. TOTAL MISSING VALUES
# ============================================================

print("\nTotal Missing Values :", df.isnull().sum().sum())


# ============================================================
# 11. DUPLICATE ROWS
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE CHECK")
print("=" * 60)

duplicate_rows = df.duplicated().sum()

print("Duplicate Rows :", duplicate_rows)


# ============================================================
# 12. UNIQUE PRODUCTS
# ============================================================

print("\n" + "=" * 60)
print("PRODUCT INFORMATION")
print("=" * 60)

print("Unique Products :", df["asin"].nunique())


# ============================================================
# 13. PRODUCT DISTRIBUTION
# ============================================================

print("\nProduct Distribution:")
print(df["product_name"].value_counts())


# ============================================================
# 14. UNIQUE ASIN VALUES
# ============================================================

print("\n" + "=" * 60)
print("ASIN VALUES")
print("=" * 60)

print(df["asin"].unique())


# ============================================================
# 15. STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe())


# ============================================================
# 16. DEMAND STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("ACTUAL DEMAND STATISTICS")
print("=" * 60)

print(df["actual_demand"].describe())


# ============================================================
# 17. UNITS SOLD STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("UNITS SOLD STATISTICS")
print("=" * 60)

print(df["units_sold"].describe())


# ============================================================
# 18. STOCKOUT INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("STOCKOUT INFORMATION")
print("=" * 60)

print(df["is_stockout"].value_counts())


# ============================================================
# 19. SALE SEASON INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("SALE SEASON INFORMATION")
print("=" * 60)

print(df["is_sale_season"].value_counts())


# ============================================================
# 20. WEEKEND INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("WEEKEND INFORMATION")
print("=" * 60)

print(df["is_weekend"].value_counts())


# ============================================================
# 21. DATE RANGE
# ============================================================

print("\n" + "=" * 60)
print("DATE RANGE")
print("=" * 60)

print("Start Date :", df["date"].min())
print("End Date   :", df["date"].max())


# ============================================================
# 22. NUMERICAL COLUMNS
# ============================================================

print("\n" + "=" * 60)
print("NUMERICAL COLUMNS")
print("=" * 60)

numerical_columns = df.select_dtypes(include="number").columns

print(list(numerical_columns))


# ============================================================
# 23. CATEGORICAL COLUMNS
# ============================================================

print("\n" + "=" * 60)
print("CATEGORICAL COLUMNS")
print("=" * 60)

categorical_columns = df.select_dtypes(include="object").columns

print(list(categorical_columns))


# ============================================================
# 24. TARGET VARIABLE
# ============================================================

print("\n" + "=" * 60)
print("TARGET VARIABLE")
print("=" * 60)

target = "actual_demand"

print("Target Variable :", target)


# ============================================================
# 25. TARGET RANGE
# ============================================================

print("\n" + "=" * 60)
print("TARGET RANGE")
print("=" * 60)

print("Minimum Demand :", df[target].min())
print("Maximum Demand :", df[target].max())
print("Average Demand :", round(df[target].mean(), 2))


# ============================================================
# 26. FINAL DATASET PREVIEW
# ============================================================

print("\n" + "=" * 60)
print("FINAL DATASET PREVIEW")
print("=" * 60)

print(df.head(10))


# ============================================================
# STEP 2 COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("✅ STEP 2 : DATA UNDERSTANDING COMPLETED")
print("=" * 60)