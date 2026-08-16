# ============================================================
# DEMANDPULSE
# STEP 7 : MLFLOW EXPERIMENT TRACKING
# ============================================================

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 1. MLFLOW CONFIGURATION
# ============================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("DemandPulse_Demand_Forecasting")

print("\n" + "=" * 70)
print("DEMANDPULSE - STEP 7 : MLFLOW TRACKING")
print("=" * 70)


# ============================================================
# 2. LOAD DATA
# ============================================================

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()

X_validation = pd.read_csv("X_validation.csv")
y_validation = pd.read_csv("y_validation.csv").squeeze()

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()


# ============================================================
# 3. CLEAN DATA
# ============================================================

for data in [X_train, X_validation, X_test]:
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)


# ============================================================
# 4. DEFINE MODELS
# ============================================================

models = {
    "Linear_Regression": LinearRegression(),

    "Random_Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="reg:squarederror",   # ✅ fixed typo
        n_jobs=-1
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )
}


# ============================================================
# 5. TRAIN + MLFLOW TRACKING
# ============================================================

results = []

for model_name, model in models.items():
    print("\n" + "-" * 70)
    print(f"Training model: {model_name}")
    print("-" * 70)

    with mlflow.start_run(run_name=model_name):

        # Train
        model.fit(X_train, y_train)

        # Validation prediction
        validation_prediction = model.predict(X_validation)

        # Metrics
        mae = mean_absolute_error(y_validation, validation_prediction)
        rmse = np.sqrt(mean_squared_error(y_validation, validation_prediction))
        r2 = r2_score(y_validation, validation_prediction)

        # Log params
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("validation_rows", len(X_validation))
        mlflow.log_param("feature_count", X_train.shape[1])

        if hasattr(model, "get_params"):
            parameters = model.get_params()
            for key, value in parameters.items():
                try:
                    mlflow.log_param(key, value)
                except ValueError:
                    pass   # ✅ narrowed exception

        # Log metrics
        mlflow.log_metric("MAE", float(mae))
        mlflow.log_metric("RMSE", float(rmse))
        mlflow.log_metric("R2_Score", float(r2))

        # ✅ Log model safely
        if model_name == "XGBoost":
            mlflow.xgboost.log_model(model, name="model")
        elif model_name == "LightGBM":
            mlflow.lightgbm.log_model(model, name="model")
        else:
            mlflow.sklearn.log_model(model, name="model")

        # Save local model
        local_model_name = model_name.lower().replace(" ", "_") + "_model.pkl"
        joblib.dump(model, local_model_name)

        # Log artifact
        mlflow.log_artifact(local_model_name)

        # Print results
        print("MAE  :", round(mae, 4))
        print("RMSE :", round(rmse, 4))
        print("R2   :", round(r2, 4))

        # Store results
        results.append({
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2_Score": r2
        })


# ============================================================
# 6. CREATE RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="RMSE", ascending=True)

print("\n" + "=" * 70)
print("MLFLOW MODEL COMPARISON")
print("=" * 70)
print(results_df.to_string(index=False))


# ============================================================
# 7. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]
best_rmse = results_df.iloc[0]["RMSE"]
best_mae = results_df.iloc[0]["MAE"]
best_r2 = results_df.iloc[0]["R2_Score"]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)
print("Model:", best_model_name)
print("RMSE:", round(best_rmse, 4))
print("MAE:", round(best_mae, 4))
print("R2:", round(best_r2, 4))


# ============================================================
# 8. SAVE RESULTS
# ============================================================

results_df.to_csv("mlflow_model_comparison.csv", index=False)


# ============================================================
# 9. SAVE BEST MODEL METADATA
# ============================================================

metadata = {
    "best_model": best_model_name,
    "validation_rmse": float(best_rmse),
    "validation_mae": float(best_mae),
    "validation_r2": float(best_r2),
    "feature_count": int(X_train.shape[1])
}

joblib.dump(metadata, "mlflow_best_model_metadata.pkl")


# ============================================================
# 10. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("""
✅ MLflow Experiment Tracking Completed

Experiment:
DemandPulse_Demand_Forecasting

Tracked:
✔ Model parameters
✔ Training information
✔ Validation metrics
✔ Model artifacts
✔ Model comparison

Created:
✔ mlflow.db (SQLite backend)
✔ mlflow_model_comparison.csv
✔ mlflow_best_model_metadata.pkl
""")
print("=" * 70)
