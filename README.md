````markdown
# 🚀 DemandPulse — End-to-End MLOps Demand Forecasting

DemandPulse is an end-to-end **Machine Learning + MLOps** project for forecasting e-commerce product demand and identifying potential stockout risks.

The project covers the complete ML lifecycle — from data preparation and feature engineering to model training, experiment tracking, model registry, API deployment, monitoring, drift detection, automated retraining, Dockerization, CI/CD, and final validation.

---

## 🎯 Project Objective

The main objective of DemandPulse is to predict future product demand using historical e-commerce data and provide actionable information such as:

- 📈 Predicted product demand
- 📦 Inventory status
- ⚠️ Stockout risk
- 🔍 Data drift detection
- 🤖 Model performance monitoring
- 🔄 Automated model retraining

---

# 🏗️ Project Architecture

```text
                    ┌──────────────────────┐
                    │   E-Commerce Data    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Understanding   │
                    │        + EDA         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Engineering  │
                    │ Time-Series Features │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Time-Series Split    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Model Training       │
                    │ Random Forest        │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    ▼                      ▼
          ┌─────────────────┐    ┌─────────────────┐
          │ MLflow Tracking │    │ Model Registry  │
          └────────┬────────┘    └────────┬────────┘
                   │                      │
                   └──────────┬───────────┘
                              ▼
                    ┌──────────────────────┐
                    │ FastAPI Prediction   │
                    │        API           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Docker Container     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Production Monitoring│
                    │ + Data Drift         │
                    └──────────┬───────────┘
                               │
                         Drift Detected
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Automated Retraining │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Best Model Promoted  │
                    └──────────────────────┘
````

---

# 🧠 Machine Learning Problem

### Problem Type

**Supervised Regression**

### Target Variable

```text
actual_demand
```

### Model

```text
RandomForestRegressor
```

The model predicts product demand based on historical demand, pricing, inventory, marketing activity, time-based patterns, and other engineered features.

---

# 📊 Dataset

The project uses multi-product e-commerce demand data.

Important columns include:

| Column             | Description              |
| ------------------ | ------------------------ |
| `date`             | Product observation date |
| `asin`             | Product identifier       |
| `product_name`     | Product name             |
| `mrp`              | Maximum retail price     |
| `selling_price`    | Actual selling price     |
| `discount_percent` | Product discount         |
| `customer_rating`  | Customer rating          |
| `total_reviews`    | Number of reviews        |
| `page_views`       | Product page views       |
| `ad_spend_inr`     | Advertisement spend      |
| `inventory_level`  | Available inventory      |
| `units_sold`       | Units sold               |
| `actual_demand`    | ML target                |

Final feature-engineered dataset:

```text
1096 rows
49 columns
```

---

# ⚙️ Feature Engineering

DemandPulse creates multiple time-series and business features.

### Date Features

* Year
* Month
* Day
* Day of week
* Week of year
* Day of year
* Quarter
* Weekend indicator

### Cyclical Features

* Day-of-week sine/cosine
* Month sine/cosine
* Day-of-year sine/cosine

### Price Features

* Discount amount
* Price discount ratio
* Previous selling price
* Selling price change

### Marketing Features

* Advertisement spend per page view

### Inventory Features

* Inventory-demand ratio
* Inventory gap
* Previous inventory
* Inventory change

### Demand Features

* Demand lag 1
* Demand lag 7
* Demand lag 14
* Demand lag 30
* Rolling demand 7 days
* Rolling demand 14 days
* Rolling demand 30 days
* Rolling demand volatility

### Sales Features

* Units sold lag 1
* Units sold lag 7

### Product Ranking

* Product demand rank

Time-series leakage was controlled by using previous observations for lag and rolling features.

---

# 🤖 Model Performance

The retraining pipeline produced:

```text
MAE  : 4.1156
MSE  : 51.1094
RMSE : 7.1491
R²   : 0.9951
```

The retrained model achieved a strong validation performance and was selected as the production model.

---

# 🔬 MLflow

MLflow is used for:

* Experiment tracking
* Model parameters
* Model metrics
* Model artifacts
* Model version tracking

Tracked metrics include:

```text
MAE
MSE
R² Score
```

---

# 🗂️ Model Registry

DemandPulse includes a model registry system for managing production models.

Example structure:

```text
model_registry/
│
├── model_registry.csv
├── production_model.json
│
└── v1/
    └── model.pkl
```

The registry allows the pipeline to track and promote production-ready models.

---

# 🌐 FastAPI Prediction API

DemandPulse exposes the trained model through a FastAPI application.

### Example API response

```json
{
  "status": "success",
  "asin": "B073496Y7S",
  "product_name": "boAt Rockerz Wireless Bluetooth Earbuds",
  "date": "2025-12-31",
  "predicted_demand": 142,
  "inventory_level": 100,
  "stockout_risk": "High"
}
```

The API provides:

* Product information
* Prediction date
* Predicted demand
* Current inventory
* Stockout risk

---

# 🐳 Docker

The application is containerized using Docker.

### Build image

```bash
docker build -t demandpulse-api .
```

### Run container

```bash
docker run -p 8000:8000 demandpulse-api
```

API:

```text
http://localhost:8000
```

---

# 📈 Model Monitoring

DemandPulse contains a production-style monitoring pipeline.

Monitoring includes:

### Data Drift

```text
monitoring_reference.csv
monitoring_current.csv
drift_summary.csv
drift_report.html
```

### Prediction Monitoring

```text
prediction_monitoring.csv
```

The monitoring system compares reference and current data to identify changes in the production data distribution.

---

# 🔄 Automated Model Retraining

When data drift is detected, the retraining pipeline can train a new model.

The pipeline:

```text
Detect Drift
     ↓
Train New Model
     ↓
Evaluate New Model
     ↓
Compare With Current Model
     ↓
Select Better Model
     ↓
Promote To Production
```

Retraining outputs:

```text
demandpulse_retrained_model.pkl
retraining_report.csv
```

---

# 🔁 CI/CD

GitHub Actions is used for continuous integration.

The CI pipeline validates the project whenever changes are pushed to GitHub.

The workflow checks the project environment and automated validation process.

---

# 🧪 Final Validation

DemandPulse includes an end-to-end validation script.

The validation checks:

* Required project files
* Feature-engineered dataset
* Production model
* Retrained model
* Monitoring pipeline
* Retraining report
* Docker configuration
* CI/CD configuration
* FastAPI application

Final pipeline status:

```text
Data Collection              ✅
Data Understanding           ✅
EDA                          ✅
Feature Engineering          ✅
Time-Series Split            ✅
Model Training               ✅
Model Comparison             ✅
MLflow Tracking              ✅
Model Registry               ✅
API                          ✅
Docker                       ✅
Monitoring + Drift           ✅
Automated Retraining         ✅
CI/CD                        ✅
Final Validation             ✅
```

---

# 🛠️ Tech Stack

### Programming

* Python

### Data Science

* Pandas
* NumPy
* Scikit-learn

### Machine Learning

* Random Forest Regression

### Experiment Tracking

* MLflow

### API

* FastAPI
* Uvicorn

### Deployment

* Docker

### MLOps

* Model Registry
* Data Drift Detection
* Prediction Monitoring
* Automated Retraining

### CI/CD

* GitHub Actions

### Version Control

* Git
* GitHub

---
---

# 🚀 Run Locally

## 1. Clone Repository

```bash
git clone https://github.com/Shrikrishna-dhebe-12/DemandPulseML.git
```

```bash
cd DemandPulseML
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## 3. Run API

```bash
python app.py
```

## 4. Run Monitoring

```bash
python monitoring.py
```

## 5. Run Retraining

```bash
python retrain_model.py
```

## 6. Run Final Validation

```bash
python final_validation.py
```

---

# 📌 Key MLOps Features

DemandPulse demonstrates a complete production-style ML lifecycle:

```text
Data
 ↓
EDA
 ↓
Feature Engineering
 ↓
Time-Series Validation
 ↓
Model Training
 ↓
Experiment Tracking
 ↓
Model Registry
 ↓
API
 ↓
Docker
 ↓
Monitoring
 ↓
Drift Detection
 ↓
Automated Retraining
 ↓
Model Promotion
 ↓
CI/CD
```

---

# 🎯 Future Improvements

Possible future improvements include:

* Cloud deployment
* Kubernetes deployment
* Automated scheduled retraining
* Real-time monitoring dashboard
* Prometheus + Grafana integration
* Feature store
* Model explainability with SHAP
* Advanced forecasting models
* Automated alerting
* Production database integration

---

# 👨‍💻 Author

**Shrikrishna Dhebe**

B.Sc. Computer Science | Data Science | Machine Learning | MLOps

GitHub:

[https://github.com/Shrikrishna-dhebe-12](https://github.com/Shrikrishna-dhebe-12)

---

# ⭐ Project Status

```text
PROJECT: DemandPulse
TYPE: End-to-End MLOps
STATUS: Production-Ready Project
```

If you find this project useful, consider giving the repository a ⭐.

```
include केलं आहे.
```
