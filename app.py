# ============================================================
# DEMANDPULSE
# STEP 10 : FASTAPI PREDICTION API
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os


# ============================================================
# 1. FASTAPI APP
# ============================================================

app = FastAPI(
    title="DemandPulse API",
    description="Real-Time Demand Forecasting API",
    version="1.0.0"
)


# ============================================================
# 2. LOAD MODEL
# ============================================================

MODEL_FILE = "demandpulse_best_model.pkl"
DATA_FILE = "demandpulse_features.csv"

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_FILE}"
    )

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Feature dataset not found: {DATA_FILE}"
    )


model = joblib.load(MODEL_FILE)

history_df = pd.read_csv(DATA_FILE)

history_df["date"] = pd.to_datetime(
    history_df["date"]
)

history_df = history_df.sort_values(
    ["asin", "date"]
).reset_index(drop=True)


print("=" * 70)
print("DEMANDPULSE API")
print("=" * 70)

print("Model loaded successfully.")
print("Historical feature dataset loaded successfully.")

print("\nModel expected features:")

if hasattr(model, "feature_names_in_"):
    print(list(model.feature_names_in_))
else:
    print("Model does not expose feature_names_in_")


# ============================================================
# 3. REQUEST SCHEMA
# ============================================================

class DemandRequest(BaseModel):

    date: str

    asin: str

    product_name: str

    mrp: float

    selling_price: float

    discount_percent: float

    customer_rating: float

    total_reviews: int

    page_views: int

    ad_spend_inr: float

    is_sale_season: int

    is_weekend: int

    inventory_level: float


# ============================================================
# 4. HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {
        "message": "DemandPulse API Running Successfully",
        "status": "online",
        "endpoint": "/predict"
    }


# ============================================================
# 5. FEATURE ENGINEERING FUNCTION
# ============================================================

def create_features(data: DemandRequest):

    # --------------------------------------------------------
    # Convert request to dictionary
    # --------------------------------------------------------

    input_data = data.model_dump()

    request_date = pd.to_datetime(
        input_data["date"]
    )

    asin = input_data["asin"]


    # --------------------------------------------------------
    # Find historical data for product
    # --------------------------------------------------------

    product_history = history_df[
        history_df["asin"] == asin
    ].copy()


    if product_history.empty:

        raise ValueError(
            f"No historical data found for ASIN: {asin}"
        )


    # --------------------------------------------------------
    # Basic row
    # --------------------------------------------------------

    row = pd.DataFrame([input_data])

    row["date"] = request_date


    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------

    row["year"] = row["date"].dt.year

    row["month"] = row["date"].dt.month

    row["day"] = row["date"].dt.day

    row["day_of_week"] = row["date"].dt.dayofweek

    row["week_of_year"] = (
        row["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    row["day_of_year"] = (
        row["date"]
        .dt.dayofyear
    )

    row["quarter"] = (
        row["date"]
        .dt.quarter
    )


    # --------------------------------------------------------
    # WEEKEND
    # --------------------------------------------------------

    row["is_weekend"] = (
        row["day_of_week"] >= 5
    ).astype(int)


    # --------------------------------------------------------
    # CYCLICAL FEATURES
    # --------------------------------------------------------

    row["day_of_week_sin"] = np.sin(
        2 * np.pi * row["day_of_week"] / 7
    )

    row["day_of_week_cos"] = np.cos(
        2 * np.pi * row["day_of_week"] / 7
    )

    row["month_sin"] = np.sin(
        2 * np.pi * row["month"] / 12
    )

    row["month_cos"] = np.cos(
        2 * np.pi * row["month"] / 12
    )

    row["day_of_year_sin"] = np.sin(
        2 * np.pi * row["day_of_year"] / 365
    )

    row["day_of_year_cos"] = np.cos(
        2 * np.pi * row["day_of_year"] / 365
    )


    # --------------------------------------------------------
    # PRICE FEATURES
    # --------------------------------------------------------

    row["discount_amount"] = (
        row["mrp"] -
        row["selling_price"]
    )

    row["price_discount_ratio"] = (
        row["discount_amount"] /
        row["mrp"].replace(0, np.nan)
    )

    row["price_discount_ratio"] = (
        row["price_discount_ratio"]
        .fillna(0)
    )


    # --------------------------------------------------------
    # MARKETING FEATURES
    # --------------------------------------------------------

    row["ad_spend_per_view"] = (
        row["ad_spend_inr"] /
        row["page_views"].replace(0, np.nan)
    )

    row["ad_spend_per_view"] = (
        row["ad_spend_per_view"]
        .fillna(0)
    )


    # --------------------------------------------------------
    # GET LATEST HISTORICAL PRODUCT DATA
    # --------------------------------------------------------

    product_history = (
        product_history
        .sort_values("date")
    )

    latest = product_history.iloc[-1]


    # --------------------------------------------------------
    # LAG FEATURES
    # --------------------------------------------------------

    row["demand_lag_1"] = (
        product_history["actual_demand"]
        .shift(1)
        .iloc[-1]
    )

    row["demand_lag_7"] = (
        product_history["actual_demand"]
        .shift(7)
        .iloc[-1]
    )

    row["demand_lag_14"] = (
        product_history["actual_demand"]
        .shift(14)
        .iloc[-1]
    )

    row["demand_lag_30"] = (
        product_history["actual_demand"]
        .shift(30)
        .iloc[-1]
    )


    # --------------------------------------------------------
    # ROLLING DEMAND
    # --------------------------------------------------------

    demand_series = (
        product_history["actual_demand"]
    )


    row["demand_rolling_7"] = (
        demand_series
        .shift(1)
        .rolling(7)
        .mean()
        .iloc[-1]
    )

    row["demand_rolling_14"] = (
        demand_series
        .shift(1)
        .rolling(14)
        .mean()
        .iloc[-1]
    )

    row["demand_rolling_30"] = (
        demand_series
        .shift(1)
        .rolling(30)
        .mean()
        .iloc[-1]
    )

    row["demand_rolling_std_7"] = (
        demand_series
        .shift(1)
        .rolling(7)
        .std()
        .iloc[-1]
    )


    # --------------------------------------------------------
    # PREVIOUS UNITS SOLD
    # --------------------------------------------------------

    row["units_sold_lag_1"] = (
        product_history["units_sold"]
        .shift(1)
        .iloc[-1]
    )

    row["units_sold_lag_7"] = (
        product_history["units_sold"]
        .shift(7)
        .iloc[-1]
    )


    # --------------------------------------------------------
    # PREVIOUS PRICE
    # --------------------------------------------------------

    row["selling_price_lag_1"] = (
        product_history["selling_price"]
        .shift(1)
        .iloc[-1]
    )

    row["selling_price_change"] = (
        row["selling_price"].iloc[0]
        -
        row["selling_price_lag_1"].iloc[0]
    )


    # --------------------------------------------------------
    # PREVIOUS INVENTORY
    # --------------------------------------------------------

    row["inventory_lag_1"] = (
        product_history["inventory_level"]
        .shift(1)
        .iloc[-1]
    )

    row["inventory_change"] = (
        row["inventory_level"].iloc[0]
        -
        row["inventory_lag_1"].iloc[0]
    )


    # --------------------------------------------------------
    # INVENTORY FEATURES
    # --------------------------------------------------------

    # For a future prediction, actual demand is unknown.
    # Therefore use latest historical demand as reference.

    latest_demand = float(
        latest["actual_demand"]
    )

    row["inventory_demand_ratio"] = (
        row["inventory_level"] /
        latest_demand
        if latest_demand != 0
        else 0
    )

    row["inventory_gap"] = (
        row["inventory_level"] -
        latest_demand
    )


    # --------------------------------------------------------
    # PRODUCT DEMAND RANK
    # --------------------------------------------------------

    latest_date_data = history_df[
        history_df["date"] ==
        history_df["date"].max()
    ]

    if not latest_date_data.empty:

        rank_map = (
            latest_date_data
            .set_index("asin")["actual_demand"]
            .rank(
                method="average",
                ascending=False
            )
        )

        if asin in rank_map.index:

            row["product_demand_rank"] = (
                rank_map.loc[asin]
            )

        else:

            row["product_demand_rank"] = (
                len(rank_map) + 1
            )

    else:

        row["product_demand_rank"] = 1


    # --------------------------------------------------------
    # FILL NaN
    # --------------------------------------------------------

    row = row.replace(
        [np.inf, -np.inf],
        np.nan
    )

    row = row.fillna(0)


    # --------------------------------------------------------
    # REMOVE NON-MODEL COLUMNS
    # --------------------------------------------------------

    if hasattr(model, "feature_names_in_"):

        expected_features = list(
            model.feature_names_in_
        )

        # Add missing expected features
        for feature in expected_features:

            if feature not in row.columns:

                row[feature] = 0


        # Keep ONLY model features
        row = row[
            expected_features
        ]


    return row


# ============================================================
# 6. PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict_demand(data: DemandRequest):

    try:

        print("\n" + "=" * 70)

        print("NEW PREDICTION REQUEST")

        print("=" * 70)

        print(data.model_dump())


        # ----------------------------------------------------
        # Create model features
        # ----------------------------------------------------

        input_df = create_features(data)


        print("\nFeatures sent to model:")

        print(
            list(input_df.columns)
        )


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_df
        )


        predicted_demand = float(
            prediction[0]
        )


        # ----------------------------------------------------
        # STOCKOUT RISK
        # ----------------------------------------------------

        inventory = (
            data.inventory_level
        )


        if inventory < predicted_demand:

            stockout_risk = "High"

        elif inventory < predicted_demand * 1.20:

            stockout_risk = "Medium"

        else:

            stockout_risk = "Low"


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {

            "status": "success",

            "asin": data.asin,

            "product_name": data.product_name,

            "date": data.date,

            "predicted_demand": round(
                predicted_demand,
                2
            ),

            "inventory_level": inventory,

            "stockout_risk": stockout_risk

        }


    except Exception as e:

        print("\n❌ PREDICTION ERROR:")

        print(str(e))

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )


# ============================================================
# 7. RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )