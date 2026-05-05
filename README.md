# 📦 Inventory Demand Forecasting

An end-to-end machine learning pipeline that predicts daily sales for every **store × item** combination using historical data, seasonality signals, and lag-based features — deployed as a **FastAPI web application** with a clean UI.

---

🌐 Live Demo: https://inventory-demand-forecasting-dhof.onrender.com  
⚠️ First load may take ~50 seconds (Render free tier)

---

## 🧠 Problem Statement

Retailers face significant losses due to inaccurate demand estimation, leading to **stockouts** (lost revenue) or **overstock situations** (wasted capital). Traditional forecasting models struggle to adapt to dynamic market conditions.

> **Goal:** Build an ML model that predicts future inventory demand using historical sales data, seasonality, and consumer trends — improving stock management and profitability.

---

## 📊 Dataset

| Column  | Description                                 |
|---------|---------------------------------------------|
| `date`  | Date of sale                                |
| `store` | Store ID (1–10)                             |
| `item`  | Product / Item ID (1–50)                    |
| `sales` | Units sold that day — **target variable**   |

- **913,000 rows** — 10 stores × 50 items × ~5 years (2013–2017)
- **Train split:** 2013-01-01 → 2016-12-31
- **Test split:**  2017-01-01 → 2017-12-31 (hold-out, never seen during training)

---

## 🔁 Pipeline Overview

```
Raw CSV
  │
  ▼
Stage 1 — Data Ingestion        load CSV → split by date → save train/test
  │
  ▼
Stage 2 — Data Transformation   validate → fill missing → clip negatives → log1p(sales)
  │
  ▼
Stage 3 — Feature Engineering   28 features: time + lag + rolling stats
  │
  ▼
Stage 4 — Model Training        LR · RF · XGBoost · LightGBM → auto-select best by SMAPE
  │
  ▼
artifacts/model.pkl             saved best model
  │
  ▼
FastAPI Web App                 UI for single / batch / multi-step predictions
```

---

## ⚙️ Feature Engineering

### Time Features (13 columns)
| Feature | Description |
|---------|-------------|
| `year`, `month`, `day` | Calendar components |
| `day_of_week` | 0 = Monday … 6 = Sunday |
| `is_weekend` | 1 if Saturday or Sunday |
| `quarter` | 1–4 |
| `time_index` | Days since dataset start (captures overall trend) |
| `is_month_start`, `is_month_end` | Payday / end-of-month effects |
| `month_sin`, `month_cos` | Cyclical month encoding |
| `dow_sin`, `dow_cos` | Cyclical day-of-week encoding |

### Lag Features (7 columns) — per store × item
`lag_1`, `lag_7`, `lag_14`, `lag_30`, `lag_90`, `lag_365`, `lag_365_missing`

Sales from N days ago. `lag_365_missing` flags rows where the previous year is unavailable (first year of data).

### Rolling Statistics (6 columns) — per store × item
`rolling_mean_7/14/30`, `rolling_std_7/14/30`

Computed with `shift(1)` — never includes today's sales (no data leakage).

---

## 🤖 Models Trained

| Model | Notes |
|-------|-------|
| **Linear Regression** | Baseline; uses `StandardScaler` via sklearn `Pipeline` |
| **Random Forest** | 100 trees, max_depth=10 |
| **XGBoost** | 1000 estimators, early stopping on validation RMSE |
| **LightGBM** | 1000 estimators, num_leaves=31 — typically wins |

The best model is **automatically selected** by lowest **SMAPE** on the 2017 hold-out test set and saved to `artifacts/model.pkl`.

> **Training runs locally only.** The deployed app serves predictions from the pre-trained model — it does not retrain on the server.

---

## 📈 Evaluation Metrics

All models are evaluated on the **2017 test set** in the original sales scale (after reversing the log transform).

| Metric | Description |
|--------|-------------|
| **MAE** | Mean Absolute Error — average error in units |
| **RMSE** | Root Mean Squared Error — penalises large errors |
| **R²** | Variance explained (1.0 = perfect) |
| **MAPE** | Mean Absolute Percentage Error |
| **SMAPE** | Symmetric MAPE — main ranking metric (lower is better) |

### Typical Results (2017 test set)

| Model | MAE | RMSE | R² | SMAPE |
|-------|-----|------|----|-------|
| LightGBM | ~6.1 | ~7.9 | ~0.936 | ~12.1% |
| XGBoost | ~6.1 | ~8.0 | ~0.936 | ~12.1% |
| Random Forest | ~6.5 | ~8.6 | ~0.926 | ~12.7% |
| Linear Regression | ~6.5 | ~8.5 | ~0.927 | ~12.8% |

Feature importances for all tree-based models are saved to `artifacts/feature_importance.csv`.

---

## 📁 Project Structure

```
inventory-demand-forecasting/
│
├── artifacts/                          generated after training
│   ├── raw_data.csv
│   ├── train.csv / test.csv
│   ├── processed_inventory_demand.csv
│   ├── model.pkl                       best trained model
│   ├── preprocessor.pkl                StandardScaler (if LR not best)
│   ├── feature_importance.csv
│   └── predictions.csv
│
├── config/
│   └── config.yaml                     all paths + hyperparameters
│
├── notebooks/
│   ├── data/train.csv
│   ├── EDA_Inventory_Demand.ipynb
│   └── Model_Training.ipynb
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── feature_engineering.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   ├── pipeline/
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── app/
│   ├── app.py                          FastAPI application
│   ├── static/style.css
│   └── templates/
│       ├── base.html
│       ├── predict.html                single-step prediction form
│       ├── batch.html                  batch prediction form
│       ├── multi.html                  multi-step forecast form
│       └── result.html                 training results (local use)
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🚀 Setup & Run

### 1. Clone and create environment

```bash
git clone https://github.com/your-username/inventory-demand-forecasting.git
cd inventory-demand-forecasting

conda create -n venv2 python=3.11 -y
conda activate venv2
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Train the model (local only)

```bash
python -m src.pipeline.training_pipeline
```

This runs all 4 pipeline stages and saves `artifacts/model.pkl`.
Expected runtime: **5–10 minutes** depending on hardware.

### 4. Start the web app

```bash
uvicorn app.app:app --host 0.0.0.0 --port 5000 --reload
```

Open **http://localhost:5000** in your browser.

---

## 🌐 Web Application

The app lands directly on the prediction form. Use the navbar to switch between modes.

### Pages

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Redirects to `/predict` |
| `/predict` | GET | Single-prediction form |
| `/predict` | POST | Returns predicted units for one date |
| `/predict/batch` | GET | Batch prediction form |
| `/predict/batch` | POST | Predict for multiple store-item-date rows |
| `/predict/multi` | GET | Multi-step forecast form |
| `/predict/multi` | POST | Recursive N-day forecast for one store-item pair |

### Single Prediction (`/predict`)
Enter a Store ID (1–10), Item ID (1–50), and a future date to get the predicted units sold for that day.

### Batch Prediction (`/predict/batch`)
Paste a JSON array of records. Each record needs `store`, `item`, and `date`.

**Example input:**
```json
[
  {"store": 1, "item": 5,  "date": "2018-01-01"},
  {"store": 2, "item": 10, "date": "2018-01-01"}
]
```

Results are shown in a table on the same page.

### Multi-Step Forecast (`/predict/multi`)
Enter a Store ID, Item ID, and number of days (1–90). The model forecasts each day sequentially, feeding each prediction back as a lag input for the next day.

> Accuracy degrades beyond ~14 days as prediction errors compound recursively.

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| Data | Pandas, NumPy |
| ML | Scikit-learn, XGBoost, LightGBM |
| Visualisation | Matplotlib, Seaborn |
| Web | FastAPI, Uvicorn, Jinja2 |
| Config | PyYAML |
| Packaging | setuptools |

---

## ☁️ Deployment (Render)

1. Push the repository to GitHub — **include `artifacts/model.pkl`** so Render does not need to retrain.
2. Create a new **Web Service** on [render.com](https://render.com).
3. Set the build command:
   ```
   pip install -r requirements.txt && pip install -e .
   ```
4. Set the start command:
   ```
   uvicorn app.app:app --host 0.0.0.0 --port 10000
   ```
5. Set environment variable `PORT=10000` if required by Render.
6. Deploy — the app will be live at your Render URL.

> Change `port` in `config/config.yaml` to match the Render port if needed.

---

## 📈 Key Insights from EDA

- Sales show a clear **upward trend** from 2013 to 2017
- Strong **weekly seasonality** — weekends vs weekdays differ significantly
- **November–January** consistently show higher sales (festive season)
- Each **Store × Item pair** has its own unique demand pattern
- `lag_7` and `lag_365` are the most predictive features

---

## ⚠️ Important Notes

- **Predict future dates only** — dates already in the training data (2013–2016) will produce unreliable predictions because lag features will be zero or missing.
- The model is trained on **Store IDs 1–10** and **Item IDs 1–50** only. Inputs outside this range are rejected.
- For best predictions, use dates **after 2017-12-31** (the last date in the dataset).
- **Training is local only** — the deployed app does not have a train button. Run `python -m src.pipeline.training_pipeline` locally and commit `artifacts/model.pkl` to deploy updated models.
