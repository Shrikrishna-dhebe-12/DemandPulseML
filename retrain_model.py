# ============================================================
# DEMANDPULSE
# STEP 14 : AUTOMATED MODEL RETRAINING
# ============================================================

import os
import shutil
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "demandpulse_features.csv"
DRIFT_FILE = "drift_summary.csv"

CURRENT_MODEL = "demandpulse_best_model.pkl"
NEW_MODEL = "demandpulse_retrained_model.pkl"

DRIFT_THRESHOLD = 0.30


print("\n" + "=" * 70)
print("DEMANDPULSE - STEP 14 : AUTOMATED MODEL RETRAINING")
print("=" * 70)


# ============================================================
# 1. LOAD FEATURE DATA
# ============================================================

print("\n[1] Loading feature-engineered dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)


# ============================================================
# 2. DEFINE MODEL FEATURES
# ============================================================

target = "actual_demand"

excluded_columns = [
    "date",
    "asin",
    "product_name",
    target
]

feature_columns = [
    column
    for column in df.columns
    if column not in excluded_columns
]

print("\nNumber of model features:", len(feature_columns))


X = df[feature_columns].copy()
y = df[target].copy()


# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(0)


# ============================================================
# 4. TIME-BASED TRAIN / TEST SPLIT
# ============================================================

print("\n[2] Creating time-based train/test split...")

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 5. CHECK DRIFT STATUS
# ============================================================

print("\n[3] Checking drift status...")

drift_detected = False

if os.path.exists(DRIFT_FILE):

    drift_df = pd.read_csv(DRIFT_FILE)

    print("Drift report loaded.")
    print("Drift report shape:", drift_df.shape)

    # Try to find common drift columns
    possible_columns = [
        "drift_score",
        "p_value",
        "psi",
        "statistic"
    ]

    found_column = None

    for column in possible_columns:

        if column in drift_df.columns:
            found_column = column
            break

    if found_column is not None:

        print("Drift metric:", found_column)

        values = pd.to_numeric(
            drift_df[found_column],
            errors="coerce"
        )

        values = values.dropna()

        if len(values) > 0:

            if found_column == "psi":

                max_drift = values.max()

                print("Maximum PSI:", max_drift)

                if max_drift >= DRIFT_THRESHOLD:
                    drift_detected = True

            elif found_column == "p_value":

                min_p_value = values.min()

                print("Minimum p-value:", min_p_value)

                if min_p_value < 0.05:
                    drift_detected = True

    else:

        print(
            "No standard drift metric detected."
        )

        print(
            "Retraining pipeline will continue."
        )

else:

    print(
        "drift_summary.csv not found."
    )

    print(
        "Continuing with retraining."
    )


# ============================================================
# 6. RETRAINING DECISION
# ============================================================

print("\n" + "=" * 70)
print("RETRAINING DECISION")
print("=" * 70)

if drift_detected:

    print("⚠️ Significant drift detected.")
    print("Retraining recommended.")

else:

    print("ℹ️ No significant drift detected.")
    print("Running controlled retraining for pipeline validation.")


# ============================================================
# 7. TRAIN NEW MODEL
# ============================================================

print("\n[4] Training new Random Forest model...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("✅ New model training completed.")


# ============================================================
# 8. MODEL PREDICTION
# ============================================================

print("\n[5] Generating predictions...")

predictions = model.predict(
    X_test
)


# ============================================================
# 9. MODEL EVALUATION
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)


print("\n" + "=" * 70)
print("NEW MODEL PERFORMANCE")
print("=" * 70)

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")


# ============================================================
# 10. LOAD CURRENT PRODUCTION MODEL
# ============================================================

print("\n[6] Checking current production model...")

current_model_score = None

if os.path.exists(CURRENT_MODEL):

    try:

        current_model = joblib.load(
            CURRENT_MODEL
        )

        current_predictions = current_model.predict(
            X_test
        )

        current_mae = mean_absolute_error(
            y_test,
            current_predictions
        )

        current_r2 = r2_score(
            y_test,
            current_predictions
        )

        current_model_score = current_mae

        print("Current model loaded.")

        print(
            f"Current Model MAE : {current_mae:.4f}"
        )

        print(
            f"Current Model R2  : {current_r2:.4f}"
        )

    except Exception as error:

        print(
            "Could not evaluate current model."
        )

        print("Error:", error)

else:

    print(
        "Current production model not found."
    )


# ============================================================
# 11. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

if current_model_score is not None:

    if mae < current_model_score:

        print(
            "✅ New model performs better."
        )

        print(
            "New model will become production model."
        )

        model_selected = "retrained"

    else:

        print(
            "ℹ️ Current model performs better."
        )

        print(
            "Keeping current production model."
        )

        model_selected = "current"

else:

    print(
        "No current model available."
    )

    print(
        "New model will become production model."
    )

    model_selected = "retrained"


# ============================================================
# 12. SAVE RETRAINED MODEL
# ============================================================

joblib.dump(
    model,
    NEW_MODEL
)

print(
    f"\nRetrained model saved: {NEW_MODEL}"
)


# ============================================================
# 13. PROMOTE BEST MODEL
# ============================================================

if model_selected == "retrained":

    shutil.copyfile(
        NEW_MODEL,
        CURRENT_MODEL
    )

    print(
        f"✅ New model promoted to: {CURRENT_MODEL}"
    )

else:

    print(
        "Current production model retained."
    )


# ============================================================
# 14. SAVE RETRAINING REPORT
# ============================================================

report = pd.DataFrame([
    {
        "new_model_mae": mae,
        "new_model_mse": mse,
        "new_model_rmse": rmse,
        "new_model_r2": r2,
        "current_model_mae":
            current_model_score
            if current_model_score is not None
            else np.nan,
        "drift_detected": drift_detected,
        "selected_model": model_selected
    }
])


report.to_csv(
    "retraining_report.csv",
    index=False
)


# ============================================================
# 15. FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("DEMANDPULSE RETRAINING STATUS")
print("=" * 70)

print(
    "Drift Detected :", drift_detected
)

print(
    "New Model      :", NEW_MODEL
)

print(
    "Production     :", CURRENT_MODEL
)

print(
    "Report         : retraining_report.csv"
)

print("\n" + "=" * 70)
print("✅ STEP 14 : AUTOMATED MODEL RETRAINING COMPLETED")
print("=" * 70)