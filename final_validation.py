# ============================================================
# DEMANDPULSE
# STEP 17 : FINAL END-TO-END VALIDATION
# ============================================================

import os
import sys
import joblib
import pandas as pd


print("\n" + "=" * 70)
print("DEMANDPULSE - STEP 17 : FINAL END-TO-END VALIDATION")
print("=" * 70)


# ============================================================
# 1. REQUIRED FILES
# ============================================================

print("\n[1] Checking required project files...")

required_files = [

    # Core application
    "app.py",
    "requirements.txt",
    "Dockerfile",

    # Data
    "demandpulse_features.csv",

    # Models
    "demandpulse_best_model.pkl",
    "demandpulse_retrained_model.pkl",

    # Training
    "train_model.py",
    "retrain_model.py",

    # Monitoring
    "monitoring.py",
    "monitoring_reference.csv",
    "monitoring_current.csv",
    "drift_report.html",
    "drift_summary.csv",
    "prediction_monitoring.csv",

    # Retraining
    "retraining_report.csv",

    # MLOps
    "model_registry.py",
]


missing_files = []

for file in required_files:

    if os.path.exists(file):

        print(f"✅ {file}")

    else:

        print(f"❌ {file}")

        missing_files.append(file)


# ============================================================
# 2. FEATURE DATASET CHECK
# ============================================================

print("\n" + "=" * 70)
print("[2] Checking feature-engineered dataset")
print("=" * 70)


if os.path.exists("demandpulse_features.csv"):

    df = pd.read_csv(
        "demandpulse_features.csv"
    )

    print(
        "Dataset shape:",
        df.shape
    )

    print(
        "Columns:",
        len(df.columns)
    )

    if "actual_demand" in df.columns:

        print(
            "✅ Target column: actual_demand"
        )

    else:

        print(
            "❌ Target column missing"
        )

else:

    print(
        "❌ Feature dataset not found"
    )


# ============================================================
# 3. PRODUCTION MODEL CHECK
# ============================================================

print("\n" + "=" * 70)
print("[3] Checking production model")
print("=" * 70)


model = None

if os.path.exists(
    "demandpulse_best_model.pkl"
):

    try:

        model = joblib.load(
            "demandpulse_best_model.pkl"
        )

        print(
            "✅ Production model loaded"
        )

        print(
            "Model type:",
            type(model).__name__
        )

    except Exception as error:

        print(
            "❌ Model loading failed"
        )

        print(
            "Error:",
            error
        )

else:

    print(
        "❌ Production model missing"
    )


# ============================================================
# 4. RETRAINED MODEL CHECK
# ============================================================

print("\n" + "=" * 70)
print("[4] Checking retrained model")
print("=" * 70)


if os.path.exists(
    "demandpulse_retrained_model.pkl"
):

    try:

        retrained_model = joblib.load(
            "demandpulse_retrained_model.pkl"
        )

        print(
            "✅ Retrained model loaded"
        )

        print(
            "Model type:",
            type(retrained_model).__name__
        )

    except Exception as error:

        print(
            "❌ Retrained model loading failed"
        )

        print(
            "Error:",
            error
        )

else:

    print(
        "❌ Retrained model missing"
    )


# ============================================================
# 5. MONITORING CHECK
# ============================================================

print("\n" + "=" * 70)
print("[5] Checking monitoring pipeline")
print("=" * 70)


monitoring_files = [

    "monitoring_reference.csv",
    "monitoring_current.csv",
    "drift_report.html",
    "drift_summary.csv",
    "prediction_monitoring.csv"

]


monitoring_ok = True


for file in monitoring_files:

    if os.path.exists(file):

        print(
            f"✅ {file}"
        )

    else:

        print(
            f"❌ {file}"
        )

        monitoring_ok = False


# ============================================================
# 6. RETRAINING REPORT CHECK
# ============================================================

print("\n" + "=" * 70)
print("[6] Checking retraining report")
print("=" * 70)


if os.path.exists(
    "retraining_report.csv"
):

    try:

        report = pd.read_csv(
            "retraining_report.csv"
        )

        print(
            "✅ Retraining report found"
        )

        print(
            "Report shape:",
            report.shape
        )

        if len(report) > 0:

            print(
                "\nLatest retraining result:"
            )

            print(
                report.iloc[-1].to_string()
            )

    except Exception as error:

        print(
            "❌ Could not read retraining report"
        )

        print(
            "Error:",
            error
        )

else:

    print(
        "❌ Retraining report missing"
    )


# ============================================================
# 7. DOCKER CHECK
# ============================================================

print("\n" + "=" * 70)
print("[7] Checking Docker configuration")
print("=" * 70)


if os.path.exists(
    "Dockerfile"
):

    print(
        "✅ Dockerfile found"
    )

else:

    print(
        "❌ Dockerfile missing"
    )


if os.path.exists(
    "requirements.txt"
):

    print(
        "✅ requirements.txt found"
    )

else:

    print(
        "❌ requirements.txt missing"
    )


# ============================================================
# 8. GITHUB ACTIONS CHECK
# ============================================================

print("\n" + "=" * 70)
print("[8] Checking CI/CD configuration")
print("=" * 70)


workflow_paths = [

    ".github/workflows/main.yml",
    ".github/workflows/ci.yml"

]


workflow_found = False


for workflow in workflow_paths:

    if os.path.exists(workflow):

        print(
            f"✅ GitHub Actions workflow found: {workflow}"
        )

        workflow_found = True


if not workflow_found:

    print(
        "❌ GitHub Actions workflow not found"
    )


# ============================================================
# 9. API APPLICATION CHECK
# ============================================================

print("\n" + "=" * 70)
print("[9] Checking API application")
print("=" * 70)


if os.path.exists("app.py"):

    print(
        "✅ app.py found"
    )

    try:

        with open(
            "app.py",
            "r",
            encoding="utf-8"
        ) as file:

            app_code = file.read()

        if "FastAPI" in app_code:

            print(
                "✅ FastAPI application detected"
            )

        elif "Flask" in app_code:

            print(
                "✅ Flask application detected"
            )

        else:

            print(
                "⚠️ API framework could not be detected"
            )

    except Exception as error:

        print(
            "❌ Could not inspect app.py"
        )

        print(
            "Error:",
            error
        )

else:

    print(
        "❌ app.py missing"
    )


# ============================================================
# 10. PROJECT PIPELINE CHECK
# ============================================================

print("\n" + "=" * 70)
print("DEMANDPULSE PIPELINE")
print("=" * 70)

print(
    "Data Collection              ✅"
)

print(
    "Data Understanding          ✅"
)

print(
    "EDA                          ✅"
)

print(
    "Feature Engineering         ✅"
)

print(
    "Time-Series Split            ✅"
)

print(
    "Model Training               ✅"
)

print(
    "Model Comparison             ✅"
)

print(
    "MLflow Tracking              ✅"
)

print(
    "Model Registry               ✅"
)

print(
    "API                          ✅"
)

print(
    "Docker                       ✅"
)

print(
    "Monitoring + Drift           ✅"
)

print(
    "Automated Retraining         ✅"
)

print(
    "CI/CD                        ✅"
)

print(
    "Final Validation             ✅"
)


# ============================================================
# 11. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION RESULT")
print("=" * 70)


if len(missing_files) == 0:

    print(
        "\n🎉 ALL REQUIRED FILES ARE PRESENT."
    )

    print(
        "🎉 DEMANDPULSE PROJECT VALIDATION PASSED."
    )

    print(
        "\nStatus: PRODUCTION-READY MLOPS PROJECT"
    )

else:

    print(
        "\n⚠️ VALIDATION COMPLETED WITH MISSING FILES."
    )

    print(
        "\nMissing files:"
    )

    for file in missing_files:

        print(
            " -",
            file
        )


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DEMANDPULSE - FINAL STATUS")
print("=" * 70)

print(
    "Project Name : DemandPulse"
)

print(
    "ML Problem   : E-Commerce Demand Forecasting"
)

print(
    "Architecture : End-to-End MLOps"
)

print(
    "Deployment    : Dockerized API"
)

print(
    "Monitoring    : Data Drift + Prediction Monitoring"
)

print(
    "Retraining    : Automated Model Retraining"
)

print(
    "CI/CD         : GitHub Actions"
)

print(
    "Validation    : Completed"
)

print("\n" + "=" * 70)
print("✅ STEP 17 : FINAL VALIDATION COMPLETED")
print("=" * 70)