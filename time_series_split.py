# ============================================================
# DEMANDPULSE
# STEP 5 : TIME-SERIES TRAIN / VALIDATION / TEST SPLIT
# ============================================================

import pandas as pd
import numpy as np


# ============================================================
# 1. LOAD FEATURE-ENGINEERED DATA
# ============================================================

df = pd.read_csv("demandpulse_features.csv")

df["date"] = pd.to_datetime(df["date"])

print("\n" + "=" * 70)
print("DEMANDPULSE - STEP 5 : TIME-SERIES DATA SPLIT")
print("=" * 70)

print("\nOriginal Shape:")
print(df.shape)


# ============================================================
# 2. SORT DATA CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    ["date", "asin"]
).reset_index(drop=True)

print("\nData sorted chronologically.")


# ============================================================
# 3. CHECK DATE RANGE
# ============================================================

print("\n" + "=" * 70)
print("DATE RANGE")
print("=" * 70)

print("Start Date :", df["date"].min().date())
print("End Date   :", df["date"].max().date())

print(
    "Total Days :",
    df["date"].dt.date.nunique()
)


# ============================================================
# 4. DEFINE TARGET
# ============================================================

target = "actual_demand"

print("\nTarget Variable:")
print(target)


# ============================================================
# 5. REMOVE COLUMNS THAT SHOULD NOT ENTER MODEL
# ============================================================

# actual_demand = target
#
# units_sold and is_stockout are consequences of demand/inventory.
# Keeping them can create target leakage.
#
# product_name is text and asin is an identifier.
# date itself will not directly enter the model.

drop_columns = [
    "actual_demand",
    "units_sold",
    "is_stockout",
    "product_name",
    "asin",
    "date"
]

available_drop_columns = [
    col
    for col in drop_columns
    if col in df.columns
]


# ============================================================
# 6. CREATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=available_drop_columns
)

y = df[target]


print("\nFeature Matrix Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)


# ============================================================
# 7. TIME-BASED SPLIT
# ============================================================

# 70% → Training
# 15% → Validation
# 15% → Testing

n = len(df)

train_end = int(n * 0.70)

validation_end = int(n * 0.85)


# ============================================================
# 8. CREATE TRAIN DATA
# ============================================================

train_df = df.iloc[
    :train_end
].copy()


# ============================================================
# 9. CREATE VALIDATION DATA
# ============================================================

validation_df = df.iloc[
    train_end:validation_end
].copy()


# ============================================================
# 10. CREATE TEST DATA
# ============================================================

test_df = df.iloc[
    validation_end:
].copy()


# ============================================================
# 11. PRINT SPLIT INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("TIME-SERIES SPLIT")
print("=" * 70)

print("\nTRAINING DATA")
print("-------------------------")
print("Rows :", len(train_df))
print(
    "Date :",
    train_df["date"].min().date(),
    "→",
    train_df["date"].max().date()
)


print("\nVALIDATION DATA")
print("-------------------------")
print("Rows :", len(validation_df))
print(
    "Date :",
    validation_df["date"].min().date(),
    "→",
    validation_df["date"].max().date()
)


print("\nTEST DATA")
print("-------------------------")
print("Rows :", len(test_df))
print(
    "Date :",
    test_df["date"].min().date(),
    "→",
    test_df["date"].max().date()
)


# ============================================================
# 12. CREATE X_train / y_train
# ============================================================

X_train = train_df.drop(
    columns=available_drop_columns
)

y_train = train_df[target]


# ============================================================
# 13. CREATE X_validation / y_validation
# ============================================================

X_validation = validation_df.drop(
    columns=available_drop_columns
)

y_validation = validation_df[target]


# ============================================================
# 14. CREATE X_test / y_test
# ============================================================

X_test = test_df.drop(
    columns=available_drop_columns
)

y_test = test_df[target]


# ============================================================
# 15. CHECK SHAPES
# ============================================================

print("\n" + "=" * 70)
print("MODEL DATA SHAPES")
print("=" * 70)

print("\nX_train      :", X_train.shape)
print("y_train      :", y_train.shape)

print("\nX_validation :", X_validation.shape)
print("y_validation :", y_validation.shape)

print("\nX_test       :", X_test.shape)
print("y_test       :", y_test.shape)


# ============================================================
# 16. CHECK TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("TARGET STATISTICS")
print("=" * 70)

print("\nTraining Target:")
print(y_train.describe())

print("\nValidation Target:")
print(y_validation.describe())

print("\nTesting Target:")
print(y_test.describe())


# ============================================================
# 17. CHECK DATA LEAKAGE
# ============================================================

print("\n" + "=" * 70)
print("DATA LEAKAGE CHECK")
print("=" * 70)

print(
    "\nLatest Training Date :",
    train_df["date"].max().date()
)

print(
    "Validation Start Date:",
    validation_df["date"].min().date()
)

print(
    "Validation End Date  :",
    validation_df["date"].max().date()
)

print(
    "Test Start Date      :",
    test_df["date"].min().date()
)

print(
    "Test End Date        :",
    test_df["date"].max().date()
)


# ============================================================
# 18. VERIFY CHRONOLOGICAL ORDER
# ============================================================

if (
    train_df["date"].max()
    <
    validation_df["date"].min()
    and
    validation_df["date"].max()
    <
    test_df["date"].min()
):

    print(
        "\n✅ No chronological overlap detected."
    )

else:

    print(
        "\n⚠️ Possible chronological overlap!"
    )


# ============================================================
# 19. SAVE TRAIN DATA
# ============================================================

train_df.to_csv(
    "train_demandpulse.csv",
    index=False
)


# ============================================================
# 20. SAVE VALIDATION DATA
# ============================================================

validation_df.to_csv(
    "validation_demandpulse.csv",
    index=False
)


# ============================================================
# 21. SAVE TEST DATA
# ============================================================

test_df.to_csv(
    "test_demandpulse.csv",
    index=False
)


# ============================================================
# 22. SAVE MODEL MATRICES
# ============================================================

X_train.to_csv(
    "X_train.csv",
    index=False
)

y_train.to_csv(
    "y_train.csv",
    index=False
)

X_validation.to_csv(
    "X_validation.csv",
    index=False
)

y_validation.to_csv(
    "y_validation.csv",
    index=False
)

X_test.to_csv(
    "X_test.csv",
    index=False
)

y_test.to_csv(
    "y_test.csv",
    index=False
)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print("""
✅ train_demandpulse.csv
✅ validation_demandpulse.csv
✅ test_demandpulse.csv

✅ X_train.csv
✅ y_train.csv

✅ X_validation.csv
✅ y_validation.csv

✅ X_test.csv
✅ y_test.csv
""")


# ============================================================
# STEP 5 COMPLETED
# ============================================================

print("=" * 70)
print("✅ STEP 5 : TIME-SERIES SPLIT COMPLETED")
print("=" * 70)