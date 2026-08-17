# ============================================================
# DEMANDPULSE
# STEP 15 : MODEL REGISTRY & VERSIONING
# ============================================================

import os
import json
import shutil
import joblib
import pandas as pd

from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_FILE = "demandpulse_best_model.pkl"

REPORT_FILE = "retraining_report.csv"

REGISTRY_FOLDER = "model_registry"

REGISTRY_FILE = os.path.join(
    REGISTRY_FOLDER,
    "model_registry.csv"
)

PRODUCTION_FILE = os.path.join(
    REGISTRY_FOLDER,
    "production_model.json"
)


print("\n" + "=" * 70)
print("DEMANDPULSE - STEP 15 : MODEL REGISTRY & VERSIONING")
print("=" * 70)


# ============================================================
# 1. CREATE MODEL REGISTRY DIRECTORY
# ============================================================

print("\n[1] Creating model registry...")


os.makedirs(
    REGISTRY_FOLDER,
    exist_ok=True
)


print(
    f"Registry directory: {REGISTRY_FOLDER}"
)


# ============================================================
# 2. CHECK PRODUCTION MODEL
# ============================================================

print("\n[2] Checking production model...")


if not os.path.exists(MODEL_FILE):

    print(
        "❌ Production model not found:"
    )

    print(
        MODEL_FILE
    )

    raise FileNotFoundError(
        "demandpulse_best_model.pkl not found."
    )


print(
    "✅ Production model found."
)


# ============================================================
# 3. LOAD RETRAINING REPORT
# ============================================================

print("\n[3] Loading retraining report...")


if os.path.exists(REPORT_FILE):

    report_df = pd.read_csv(
        REPORT_FILE
    )

    print(
        "Retraining report loaded."
    )

else:

    print(
        "⚠ retraining_report.csv not found."
    )

    report_df = pd.DataFrame()


# ============================================================
# 4. DETERMINE MODEL VERSION
# ============================================================

print("\n[4] Determining model version...")


if os.path.exists(REGISTRY_FILE):

    registry_df = pd.read_csv(
        REGISTRY_FILE
    )

    if len(registry_df) > 0:

        last_version = (
            registry_df["version"]
            .str.replace("v", "", regex=False)
            .astype(int)
            .max()
        )

        new_version_number = (
            last_version + 1
        )

    else:

        new_version_number = 1

else:

    registry_df = pd.DataFrame()

    new_version_number = 1


version = (
    f"v{new_version_number}"
)


print(
    "New model version:",
    version
)


# ============================================================
# 5. CREATE VERSION DIRECTORY
# ============================================================

print("\n[5] Creating version directory...")


version_folder = os.path.join(
    REGISTRY_FOLDER,
    version
)


os.makedirs(
    version_folder,
    exist_ok=True
)


# ============================================================
# 6. COPY MODEL INTO REGISTRY
# ============================================================

print("\n[6] Registering model...")


registered_model_path = os.path.join(
    version_folder,
    "model.pkl"
)


shutil.copyfile(
    MODEL_FILE,
    registered_model_path
)


print(
    "✅ Model registered:"
)

print(
    registered_model_path
)


# ============================================================
# 7. LOAD MODEL
# ============================================================

print("\n[7] Loading registered model...")


model = joblib.load(
    registered_model_path
)


print(
    "✅ Model loaded successfully."
)


# ============================================================
# 8. COLLECT MODEL METRICS
# ============================================================

print("\n[8] Collecting model metrics...")


mae = None
mse = None
rmse = None
r2 = None

selected_model = None
drift_detected = None


if len(report_df) > 0:

    latest_report = (
        report_df.iloc[-1]
    )

    if "new_model_mae" in report_df.columns:
        mae = latest_report["new_model_mae"]

    if "new_model_mse" in report_df.columns:
        mse = latest_report["new_model_mse"]

    if "new_model_rmse" in report_df.columns:
        rmse = latest_report["new_model_rmse"]

    if "new_model_r2" in report_df.columns:
        r2 = latest_report["new_model_r2"]

    if "selected_model" in report_df.columns:
        selected_model = latest_report["selected_model"]

    if "drift_detected" in report_df.columns:
        drift_detected = latest_report["drift_detected"]


print(
    "MAE:",
    mae
)

print(
    "MSE:",
    mse
)

print(
    "RMSE:",
    rmse
)

print(
    "R2:",
    r2
)


# ============================================================
# 9. CREATE REGISTRY RECORD
# ============================================================

print("\n[9] Creating registry record...")


timestamp = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)


registry_record = {

    "version": version,

    "created_at": timestamp,

    "model_type":
        type(model).__name__,

    "mae": mae,

    "mse": mse,

    "rmse": rmse,

    "r2": r2,

    "drift_detected":
        drift_detected,

    "selected_model":
        selected_model,

    "model_path":
        registered_model_path
}


# ============================================================
# 10. UPDATE MODEL REGISTRY
# ============================================================

new_record_df = pd.DataFrame([
    registry_record
])


if os.path.exists(REGISTRY_FILE):

    existing_registry = pd.read_csv(
        REGISTRY_FILE
    )

    updated_registry = pd.concat(
        [
            existing_registry,
            new_record_df
        ],
        ignore_index=True
    )

else:

    updated_registry = (
        new_record_df
    )


updated_registry.to_csv(
    REGISTRY_FILE,
    index=False
)


print(
    "✅ Model registry updated:"
)

print(
    REGISTRY_FILE
)


# ============================================================
# 11. MARK CURRENT PRODUCTION VERSION
# ============================================================

print("\n[10] Updating production model version...")


production_info = {

    "production_version":
        version,

    "updated_at":
        timestamp,

    "model_path":
        registered_model_path,

    "status":
        "production"
}


with open(
    PRODUCTION_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        production_info,
        file,
        indent=4
    )


print(
    "✅ Production version updated."
)


# ============================================================
# 12. DISPLAY MODEL HISTORY
# ============================================================

print("\n" + "=" * 70)
print("MODEL VERSION HISTORY")
print("=" * 70)


display_columns = [
    "version",
    "created_at",
    "model_type",
    "mae",
    "rmse",
    "r2",
    "selected_model"
]


available_columns = [
    column
    for column in display_columns
    if column in updated_registry.columns
]


print(
    updated_registry[
        available_columns
    ].to_string(index=False)
)


# ============================================================
# 13. PRODUCTION MODEL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("CURRENT PRODUCTION MODEL")
print("=" * 70)


print(
    "Version:",
    version
)

print(
    "Model:",
    registered_model_path
)

print(
    "Status: production"
)


# ============================================================
# 14. FINAL REGISTRY STATUS
# ============================================================

print("\n" + "=" * 70)
print("DEMANDPULSE MODEL REGISTRY STATUS")
print("=" * 70)


print(
    "Registry Folder :",
    REGISTRY_FOLDER
)

print(
    "Registry File   :",
    REGISTRY_FILE
)

print(
    "Production Info :",
    PRODUCTION_FILE
)

print(
    "Latest Version  :",
    version
)

print(
    "Total Versions  :",
    len(updated_registry)
)


print("\n" + "=" * 70)
print("✅ STEP 15 : MODEL REGISTRY & VERSIONING COMPLETED")
print("=" * 70)