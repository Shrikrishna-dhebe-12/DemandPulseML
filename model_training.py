# ============================================================
# DEMANDPULSE
# STEP 6 : MODEL TRAINING + EVALUATION
# ============================================================

import pandas as pd
import numpy as np
import joblib
import warnings

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("DEMANDPULSE - STEP 6 : MODEL TRAINING")
print("=" * 70)

X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").squeeze()

X_validation = pd.read_csv("X_validation.csv")
y_validation = pd.read_csv("y_validation.csv").squeeze()

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").squeeze()


print("\nDataset Loaded Successfully")

print("X_train      :", X_train.shape)
print("X_validation :", X_validation.shape)
print("X_test       :", X_test.shape)


# ============================================================
# 2. SAFETY CHECK
# ============================================================

# Remove any accidental unnamed columns

X_train = X_train.loc[
    :, ~X_train.columns.str.contains("^Unnamed")
]

X_validation = X_validation.loc[
    :, ~X_validation.columns.str.contains("^Unnamed")
]

X_test = X_test.loc[
    :, ~X_test.columns.str.contains("^Unnamed")
]


# ============================================================
# 3. CHECK DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(X_train.dtypes)


# ============================================================
# 4. HANDLE INFINITE VALUES
# ============================================================

X_train = X_train.replace(
    [np.inf, -np.inf],
    np.nan
)

X_validation = X_validation.replace(
    [np.inf, -np.inf],
    np.nan
)

X_test = X_test.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

X_train = X_train.fillna(0)
X_validation = X_validation.fillna(0)
X_test = X_test.fillna(0)


# ============================================================
# 6. BASELINE + ML MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ),

    "XGBoost":
        XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective="reg:squarederror",
            n_jobs=-1
        ),

    "LightGBM":
        LGBMRegressor(
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
# 7. MODEL TRAINING
# ============================================================

results = []

trained_models = {}


print("\n" + "=" * 70)
print("MODEL TRAINING STARTED")
print("=" * 70)


for name, model in models.items():

    print("\n" + "-" * 70)
    print(f"Training: {name}")
    print("-" * 70)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # VALIDATION PREDICTION
    # --------------------------------------------------------

    validation_prediction = model.predict(
        X_validation
    )

    # --------------------------------------------------------
    # VALIDATION METRICS
    # --------------------------------------------------------

    validation_mae = mean_absolute_error(
        y_validation,
        validation_prediction
    )

    validation_rmse = np.sqrt(
        mean_squared_error(
            y_validation,
            validation_prediction
        )
    )

    validation_r2 = r2_score(
        y_validation,
        validation_prediction
    )

    print(
        "Validation MAE  :",
        round(validation_mae, 4)
    )

    print(
        "Validation RMSE :",
        round(validation_rmse, 4)
    )

    print(
        "Validation R2   :",
        round(validation_r2, 4)
    )


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append({

        "Model": name,

        "Validation_MAE":
            validation_mae,

        "Validation_RMSE":
            validation_rmse,

        "Validation_R2":
            validation_r2
    })


    trained_models[name] = model


# ============================================================
# 8. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Validation_RMSE",
    ascending=True
).reset_index(drop=True)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(index=False)
)


# ============================================================
# 9. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]


print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    "Best Model:",
    best_model_name
)

print(
    "Validation RMSE:",
    round(
        results_df.iloc[0]["Validation_RMSE"],
        4
    )
)

print(
    "Validation MAE:",
    round(
        results_df.iloc[0]["Validation_MAE"],
        4
    )
)

print(
    "Validation R2:",
    round(
        results_df.iloc[0]["Validation_R2"],
        4
    )
)


# ============================================================
# 10. FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


test_prediction = best_model.predict(
    X_test
)


test_mae = mean_absolute_error(
    y_test,
    test_prediction
)


test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_prediction
    )
)


test_r2 = r2_score(
    y_test,
    test_prediction
)


print(
    "\nTest MAE :",
    round(test_mae, 4)
)

print(
    "Test RMSE:",
    round(test_rmse, 4)
)

print(
    "Test R2  :",
    round(test_r2, 4)
)


# ============================================================
# 11. ACTUAL VS PREDICTED
# ============================================================

comparison_df = pd.DataFrame({

    "Actual_Demand":
        y_test.values,

    "Predicted_Demand":
        test_prediction

})


comparison_df["Prediction_Error"] = (
    comparison_df["Actual_Demand"]
    -
    comparison_df["Predicted_Demand"]
)


print("\n" + "=" * 70)
print("ACTUAL VS PREDICTED")
print("=" * 70)

print(
    comparison_df.head(20)
)


# ============================================================
# 12. SAVE MODEL COMPARISON
# ============================================================

results_df.to_csv(
    "model_comparison.csv",
    index=False
)


# ============================================================
# 13. SAVE PREDICTIONS
# ============================================================

comparison_df.to_csv(
    "test_predictions.csv",
    index=False
)


# ============================================================
# 14. SAVE BEST MODEL
# ============================================================

joblib.dump(
    best_model,
    "demandpulse_best_model.pkl"
)


# ============================================================
# 15. SAVE MODEL METADATA
# ============================================================

model_metadata = {

    "model_name":
        best_model_name,

    "validation_mae":
        float(
            results_df.iloc[0]["Validation_MAE"]
        ),

    "validation_rmse":
        float(
            results_df.iloc[0]["Validation_RMSE"]
        ),

    "validation_r2":
        float(
            results_df.iloc[0]["Validation_R2"]
        ),

    "test_mae":
        float(test_mae),

    "test_rmse":
        float(test_rmse),

    "test_r2":
        float(test_r2),

    "feature_count":
        int(X_train.shape[1])
}


joblib.dump(
    model_metadata,
    "model_metadata.pkl"
)


# ============================================================
# 16. FEATURE IMPORTANCE
# ============================================================

if hasattr(best_model, "feature_importances_"):

    feature_importance = pd.DataFrame({

        "Feature":
            X_train.columns,

        "Importance":
            best_model.feature_importances_

    })

    feature_importance = (
        feature_importance
        .sort_values(
            by="Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )


    print("\n" + "=" * 70)
    print("TOP 20 IMPORTANT FEATURES")
    print("=" * 70)

    print(
        feature_importance.head(20)
        .to_string(index=False)
    )


    feature_importance.to_csv(
        "feature_importance.csv",
        index=False
    )


# ============================================================
# 17. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FILES CREATED")
print("=" * 70)

print("""
✅ model_comparison.csv
✅ test_predictions.csv
✅ demandpulse_best_model.pkl
✅ model_metadata.pkl

If tree-based model won:
✅ feature_importance.csv
""")


print("\n" + "=" * 70)
print("STEP 6 COMPLETED SUCCESSFULLY")
print("=" * 70)