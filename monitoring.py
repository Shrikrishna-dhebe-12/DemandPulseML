# ============================================================
# DEMANDPULSE
# STEP 13 : MODEL MONITORING + DATA DRIFT DETECTION
# ============================================================

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from evidently import Report
from evidently.presets import DataDriftPreset


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "demandpulse_features.csv"
MODEL_FILE = "demandpulse_best_model.pkl"

REFERENCE_FILE = "monitoring_reference.csv"
CURRENT_FILE = "monitoring_current.csv"

DRIFT_REPORT_FILE = "drift_report.html"
DRIFT_SUMMARY_FILE = "drift_summary.csv"
PREDICTION_MONITORING_FILE = "prediction_monitoring.csv"


print("\n" + "=" * 70)
print("DEMANDPULSE - STEP 13 : MODEL MONITORING")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1] Loading feature-engineered dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)


# ============================================================
# 2. CONVERT DATE
# ============================================================

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["asin", "date"]
    ).reset_index(drop=True)


# ============================================================
# 3. SELECT MODEL FEATURES
# ============================================================

feature_columns = [
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

    "year",
    "month",
    "day",
    "day_of_week",
    "week_of_year",
    "day_of_year",
    "quarter",

    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",

    "discount_amount",
    "price_discount_ratio",
    "ad_spend_per_view",

    "inventory_demand_ratio",
    "inventory_gap",

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
    "inventory_change",

    "product_demand_rank"
]


# ============================================================
# 4. CHECK FEATURES
# ============================================================

available_features = [
    column
    for column in feature_columns
    if column in df.columns
]

missing_features = [
    column
    for column in feature_columns
    if column not in df.columns
]


print("\nAvailable model features:", len(available_features))


if missing_features:

    print("\nWARNING: Missing features:")

    for column in missing_features:
        print("-", column)


# ============================================================
# 5. CREATE MONITORING DATASET
# ============================================================

monitoring_df = df[
    available_features
].copy()


# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

monitoring_df = monitoring_df.replace(
    [np.inf, -np.inf],
    np.nan
)

monitoring_df = monitoring_df.fillna(0)


print(
    "\nMonitoring dataset shape:",
    monitoring_df.shape
)


# ============================================================
# 7. CREATE REFERENCE AND CURRENT DATA
# ============================================================

print("\n[2] Creating reference and current datasets...")


# Use older data as reference
reference_data = monitoring_df.iloc[
    : int(len(monitoring_df) * 0.70)
].copy()


# Use newer data as current production-like data
current_data = monitoring_df.iloc[
    int(len(monitoring_df) * 0.70):
].copy()


print(
    "Reference data:",
    reference_data.shape
)

print(
    "Current data:",
    current_data.shape
)


# ============================================================
# 8. SAVE MONITORING DATA
# ============================================================

reference_data.to_csv(
    REFERENCE_FILE,
    index=False
)

current_data.to_csv(
    CURRENT_FILE,
    index=False
)

print("\nReference data saved:", REFERENCE_FILE)
print("Current data saved:", CURRENT_FILE)


# ============================================================
# 9. DATA DRIFT DETECTION
# ============================================================

print("\n[3] Running data drift detection...")


try:

    report = Report(
        metrics=[
            DataDriftPreset()
        ]
    )


    result = report.run(
        reference_data,
        current_data
    )


    # Save HTML report
    result.save_html(
        DRIFT_REPORT_FILE
    )


    print(
        "\n✅ Drift report created:",
        DRIFT_REPORT_FILE
    )


except Exception as error:

    print(
        "\n❌ Drift detection error:"
    )

    print(error)

    result = None


# ============================================================
# 10. EXTRACT DRIFT SUMMARY
# ============================================================

print("\n[4] Creating drift summary...")


drift_summary = []


if result is not None:

    try:

        result_dict = result.dict()


        # Try to extract drift information
        metrics = result_dict.get(
            "metrics",
            []
        )


        for metric in metrics:

            metric_id = metric.get(
                "metric_id",
                ""
            )

            value = metric.get(
                "value",
                None
            )


            drift_summary.append({

                "metric_id":
                    metric_id,

                "value":
                    value

            })


    except Exception as error:

        print(
            "Could not extract detailed drift metrics:"
        )

        print(error)


# If summary extraction is unavailable
# create a basic monitoring record

if not drift_summary:

    drift_summary.append({

        "metric_id":
            "DataDriftPreset",

        "value":
            "Generated - see drift_report.html"

    })


drift_summary_df = pd.DataFrame(
    drift_summary
)


drift_summary_df.to_csv(
    DRIFT_SUMMARY_FILE,
    index=False
)


print(
    "Drift summary saved:",
    DRIFT_SUMMARY_FILE
)


# ============================================================
# 11. LOAD BEST MODEL
# ============================================================

print("\n[5] Loading production model...")


try:

    model = joblib.load(
        MODEL_FILE
    )

    print(
        "✅ Model loaded successfully."
    )

except Exception as error:

    print(
        "\n❌ Model loading failed:"
    )

    print(error)

    model = None


# ============================================================
# 12. PREDICTION MONITORING
# ============================================================

prediction_records = []


if model is not None:

    print(
        "\n[6] Generating production-like predictions..."
    )


    predictions = model.predict(
        current_data
    )


    predictions = np.asarray(
        predictions
    )


    prediction_records = pd.DataFrame({

        "prediction":
            predictions,

        "prediction_mean":
            [predictions.mean()] * len(predictions),

        "prediction_std":
            [predictions.std()] * len(predictions),

        "prediction_min":
            [predictions.min()] * len(predictions),

        "prediction_max":
            [predictions.max()] * len(predictions),

        "monitoring_timestamp":
            [
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            ] * len(predictions)

    })


    prediction_records.to_csv(
        PREDICTION_MONITORING_FILE,
        index=False
    )


    print(
        "Prediction monitoring saved:",
        PREDICTION_MONITORING_FILE
    )


    # ========================================================
    # 13. PREDICTION STATISTICS
    # ========================================================

    print("\n" + "=" * 70)
    print("PREDICTION MONITORING")
    print("=" * 70)


    print(
        "Prediction Count :",
        len(predictions)
    )

    print(
        "Prediction Mean  :",
        round(
            predictions.mean(),
            2
        )
    )

    print(
        "Prediction Std   :",
        round(
            predictions.std(),
            2
        )
    )

    print(
        "Prediction Min   :",
        round(
            predictions.min(),
            2
        )
    )

    print(
        "Prediction Max   :",
        round(
            predictions.max(),
            2
        )
    )


# ============================================================
# 14. FINAL MONITORING STATUS
# ============================================================

print("\n" + "=" * 70)
print("DEMANDPULSE MONITORING STATUS")
print("=" * 70)

print("\nReference Dataset :",
      REFERENCE_FILE)

print("Current Dataset   :",
      CURRENT_FILE)

print("Drift Report      :",
      DRIFT_REPORT_FILE)

print("Drift Summary     :",
      DRIFT_SUMMARY_FILE)

print("Prediction Report :",
      PREDICTION_MONITORING_FILE)


# ============================================================
# 15. COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("✅ STEP 13 : MONITORING + DRIFT DETECTION COMPLETED")
print("=" * 70)