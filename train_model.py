# ============================================================
# DEMANDPULSE
# STEP 9 : MLFLOW EXPERIMENT TRACKING
# ============================================================

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)


# ============================================================
# 1. LOAD FEATURE-ENGINEERED DATA
# ============================================================

df = pd.read_csv("demandpulse_features.csv")

print("=" * 70)
print("DEMANDPULSE - STEP 9 : MLFLOW TRACKING")
print("=" * 70)

print("\nDataset Shape:")
print(df.shape)


# ============================================================
# 2. REMOVE NON-MODEL COLUMNS
# ============================================================

drop_columns = [
    "date",
    "asin",
    "product_name",
    "actual_demand"
]

X = df.drop(
    columns=drop_columns,
    errors="ignore"
)

y = df["actual_demand"]


# ============================================================
# 3. HANDLE DATETIME / NON-NUMERIC COLUMNS
# ============================================================

X = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
)


# ============================================================
# 4. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data :", X_train.shape)
print("Testing Data  :", X_test.shape)


# ============================================================
# 5. MLFLOW SETUP
# ============================================================

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

mlflow.set_experiment(
    "DemandPulse_Demand_Forecasting"
)


# ============================================================
# 6. MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            max_depth=10,
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
}


# ============================================================
# 7. TRAIN + MLFLOW TRACKING
# ============================================================

best_model = None
best_r2 = float("-inf")
best_model_name = None


for model_name, model in models.items():

    print("\n" + "=" * 70)
    print("Training:", model_name)
    print("=" * 70)

    with mlflow.start_run(
        run_name=model_name
    ):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        predictions = model.predict(
            X_test
        )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        r2 = r2_score(
            y_test,
            predictions
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        mse = mean_squared_error(
            y_test,
            predictions
        )

        # ----------------------------------------------------
        # PARAMETERS
        # ----------------------------------------------------

        mlflow.log_param(
            "model_type",
            model_name
        )

        if hasattr(model, "get_params"):

            params = model.get_params()

            for parameter, value in params.items():

                if value is not None:
                    mlflow.log_param(
                        parameter,
                        value
                    )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        mlflow.log_metric(
            "R2_Score",
            r2
        )

        mlflow.log_metric(
            "MAE",
            mae
        )

        mlflow.log_metric(
            "MSE",
            mse
        )

        # ----------------------------------------------------
        # LOG MODEL
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            artifact_path="model"
        )

        # ----------------------------------------------------
        # PRINT RESULTS
        # ----------------------------------------------------

        print("\nR2 Score :", round(r2, 4))
        print("MAE      :", round(mae, 4))
        print("MSE      :", round(mse, 4))

        print(
            "\n✅ MLflow tracking completed"
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if r2 > best_r2:

            best_r2 = r2
            best_model = model
            best_model_name = model_name


# ============================================================
# 8. FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    "Model :", best_model_name
)

print(
    "R2 Score :",
    round(best_r2, 4)
)

print(
    "\n✅ STEP 9 : MLFLOW EXPERIMENT TRACKING COMPLETED"
)

print(
    "Open MLflow UI:"
)

print(
    "http://127.0.0.1:5000"
)