"""
app.py
------
FastAPI web application for Inventory Demand Forecasting.

Routes:
  GET  /                   — Redirects to /predict
  GET  /predict            — Single-prediction form
  POST /predict            — Run single-step prediction
  GET  /predict/batch      — Batch prediction form
  POST /predict/batch      — Run batch prediction (HTML form)
  GET  /predict/multi      — Multi-step forecast form
  POST /predict/multi      — Run multi-step forecast (HTML form)
"""

import os
import json
from datetime import timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils import load_config
from src.logger import logger

# ── App setup ─────────────────────────────────────────────────

app = FastAPI(title="Inventory Demand Forecasting")

# Mount static files and templates relative to this file's location
_BASE_DIR = os.path.dirname(__file__)
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(_BASE_DIR, "static")),
    name="static",
)
templates = Jinja2Templates(directory=os.path.join(_BASE_DIR, "templates"))

cfg        = load_config()
MODEL_PATH = cfg["paths"]["model"]
RAW_DATA   = cfg["paths"]["source_data"]
STORE_MIN  = cfg["app"]["store_min"]
STORE_MAX  = cfg["app"]["store_max"]
ITEM_MIN   = cfg["app"]["item_min"]
ITEM_MAX   = cfg["app"]["item_max"]


# ── Pydantic models for JSON API ──────────────────────────────

class BatchRecord(BaseModel):
    store: int
    item:  int
    date:  str


class BatchRequest(BaseModel):
    records: List[BatchRecord]


class MultiRequest(BaseModel):
    store: int
    item:  int
    days:  int = 7


# ── Helpers ───────────────────────────────────────────────────

def _model_exists() -> bool:
    return os.path.exists(MODEL_PATH)


def _validate_store_item(store: int, item: int) -> Optional[str]:
    """Return an error string if inputs are out of range, else None."""
    if not (STORE_MIN <= store <= STORE_MAX):
        return f"Store ID must be between {STORE_MIN} and {STORE_MAX}."
    if not (ITEM_MIN <= item <= ITEM_MAX):
        return f"Item ID must be between {ITEM_MIN} and {ITEM_MAX}."
    return None


def _load_history(store: int, item: int) -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA, parse_dates=["date"])
    return df[(df["store"] == store) & (df["item"] == item)].copy()


def _render(request: Request, template: str, **ctx):
    ctx.setdefault("flash_msg", None)
    ctx.setdefault("flash_cat", None)
    return templates.TemplateResponse(
        request=request,        #  pass as keyword argument
        name=template,          #  pass as keyword argument  
        context=ctx             #  context does NOT include request anymore
    )


# ── Routes ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return RedirectResponse(url="/predict", status_code=303)


@app.get("/predict", response_class=HTMLResponse)
async def predict_form(request: Request):
    """Render the single-step prediction form."""
    return _render(
        request, "predict.html",
        prediction=None,
        store=None, item=None, date=None,
        store_min=STORE_MIN, store_max=STORE_MAX,
        item_min=ITEM_MIN,   item_max=ITEM_MAX,
        flash_msg=None, flash_cat=None,
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict_submit(
    request: Request,
    store:   int = Form(...),
    item:    int = Form(...),
    date:    str = Form(...),
):
    """Handle single-step prediction form submission."""
    base_ctx = dict(
        prediction=None,
        store=store, item=item, date=date,
        store_min=STORE_MIN, store_max=STORE_MAX,
        item_min=ITEM_MIN,   item_max=ITEM_MAX,
        flash_msg=None, flash_cat=None,
    )

    # Validate model exists
    if not _model_exists():
        return RedirectResponse(url="/", status_code=303)

    # Validate store / item range
    err = _validate_store_item(store, item)
    if err:
        return _render(request, "predict.html", **{**base_ctx, "flash_msg": err, "flash_cat": "danger"})

    # Validate date format
    try:
        pd.to_datetime(date)
    except ValueError:
        return _render(
            request, "predict.html",
            **{**base_ctx, "flash_msg": "Invalid date format. Use YYYY-MM-DD.", "flash_cat": "danger"},
        )

    # Sparse-history warning
    flash_msg = None
    flash_cat = None
    history_check = _load_history(store, item)
    if len(history_check) < 365:
        flash_msg = (
            f"Warning: only {len(history_check)} days of history for store {store}, "
            f"item {item}. Lag features may be incomplete — prediction may be unreliable."
        )
        flash_cat = "warning"

    # Run prediction
    try:
        history = _load_history(store, item)
        future  = pd.DataFrame(
            {"date": [pd.to_datetime(date)], "store": [store], "item": [item]}
        )

        pipeline        = PredictionPipeline(model_path=MODEL_PATH)
        result_df       = pipeline.predict_from_raw(history, future)
        predicted_sales = int(round(float(result_df["predicted_sales"].iloc[0])))

        return _render(
            request, "predict.html",
            **{**base_ctx, "prediction": predicted_sales,
               "flash_msg": flash_msg, "flash_cat": flash_cat},
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return _render(
            request, "predict.html",
            **{**base_ctx, "flash_msg": f"Prediction failed: {str(e)}", "flash_cat": "danger"},
        )


@app.get("/predict/batch", response_class=HTMLResponse)
async def batch_form(request: Request):
    """Render the batch prediction form."""
    return _render(request, "batch.html",
                   store_min=STORE_MIN, store_max=STORE_MAX,
                   item_min=ITEM_MIN,   item_max=ITEM_MAX,
                   results=None, flash_msg=None, flash_cat=None)


@app.post("/predict/batch", response_class=HTMLResponse)
async def batch_submit(request: Request):
    """Handle batch prediction form submission."""
    if not _model_exists():
        return _render(request, "batch.html",
                       store_min=STORE_MIN, store_max=STORE_MAX,
                       item_min=ITEM_MIN,   item_max=ITEM_MAX,
                       results=None,
                       flash_msg="Model not found. Please train first.",
                       flash_cat="danger")
    try:
        form   = await request.form()
        raw    = form.get("records", "").strip()
        import json
        records = json.loads(raw)

        # Validate each record
        for i, rec in enumerate(records):
            err = _validate_store_item(int(rec["store"]), int(rec["item"]))
            if err:
                raise ValueError(f"Record {i+1}: {err}")
            pd.to_datetime(rec["date"])

        future_df         = pd.DataFrame(records)
        future_df["store"] = future_df["store"].astype(int)
        future_df["item"]  = future_df["item"].astype(int)
        future_df["date"]  = pd.to_datetime(future_df["date"])
        history            = pd.read_csv(RAW_DATA, parse_dates=["date"])

        pipeline  = PredictionPipeline(model_path=MODEL_PATH)
        result_df = pipeline.predict_from_raw(history, future_df)
        result_df["predicted_sales"] = result_df["predicted_sales"].round(0).astype(int)
        result_df["date"] = result_df["date"].astype(str)

        return _render(request, "batch.html",
                       store_min=STORE_MIN, store_max=STORE_MAX,
                       item_min=ITEM_MIN,   item_max=ITEM_MAX,
                       results=result_df.to_dict(orient="records"),
                       flash_msg=f"Batch complete — {len(result_df)} prediction(s) returned.",
                       flash_cat="success")

    except (json.JSONDecodeError, KeyError, TypeError):
        return _render(request, "batch.html",
                       store_min=STORE_MIN, store_max=STORE_MAX,
                       item_min=ITEM_MIN,   item_max=ITEM_MAX,
                       results=None,
                       flash_msg="Invalid JSON. Check the format and try again.",
                       flash_cat="danger")
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        return _render(request, "batch.html",
                       store_min=STORE_MIN, store_max=STORE_MAX,
                       item_min=ITEM_MIN,   item_max=ITEM_MAX,
                       results=None,
                       flash_msg=f"Prediction failed: {str(e)}",
                       flash_cat="danger")


@app.get("/predict/multi", response_class=HTMLResponse)
async def multi_form(request: Request):
    """Render the multi-step forecast form."""
    return _render(request, "multi.html",
                   store_min=STORE_MIN, store_max=STORE_MAX,
                   item_min=ITEM_MIN,   item_max=ITEM_MAX,
                   results=None, flash_msg=None, flash_cat=None)


@app.post("/predict/multi", response_class=HTMLResponse)
async def multi_submit(
    request: Request,
    store: int = Form(...),
    item:  int = Form(...),
    days:  int = Form(...),
):
    """Handle multi-step forecast form submission."""
    base_ctx = dict(
        store_min=STORE_MIN, store_max=STORE_MAX,
        item_min=ITEM_MIN,   item_max=ITEM_MAX,
        results=None, store=store, item=item, days=days,
        flash_msg=None, flash_cat=None,
    )

    if not _model_exists():
        return _render(request, "multi.html",
                       **{**base_ctx, "flash_msg": "Model not found. Please train first.",
                          "flash_cat": "danger"})

    err = _validate_store_item(store, item)
    if err:
        return _render(request, "multi.html",
                       **{**base_ctx, "flash_msg": err, "flash_cat": "danger"})

    if not (1 <= days <= 90):
        return _render(request, "multi.html",
                       **{**base_ctx, "flash_msg": "Days must be between 1 and 90.",
                          "flash_cat": "danger"})

    try:
        history      = _load_history(store, item)
        last_date    = pd.to_datetime(history["date"]).max()
        future_dates = [last_date + timedelta(days=i + 1) for i in range(days)]

        pipeline  = PredictionPipeline(model_path=MODEL_PATH)
        result_df = pipeline.predict_multi_step(history, store, item, future_dates)
        result_df["date"] = result_df["date"].astype(str)

        return _render(request, "multi.html",
                       **{**base_ctx,
                          "results": result_df[["date", "predicted_sales"]].to_dict(orient="records"),
                          "flash_msg": f"{days}-day forecast for Store {store}, Item {item} complete.",
                          "flash_cat": "success"})

    except Exception as e:
        logger.error(f"Multi-step prediction failed: {e}")
        return _render(request, "multi.html",
                       **{**base_ctx, "flash_msg": f"Prediction failed: {str(e)}",
                          "flash_cat": "danger"})


# ── Inventory Recommendation Engine ─────────────────────────────────────────
# Interview note:
# Safety stock = Z * σ_lead * sqrt(lead_time)
# Z=1.65 → 95% service level, σ_lead = std of daily sales, lead_time = 7 days
# Reorder point = avg_daily_demand * lead_time + safety_stock
# If current_stock < reorder_point  → REORDER
# If current_stock > 2 * avg_forecast → OVERSTOCK
# Otherwise                          → MONITOR

@app.get("/api/inventory")
async def inventory_recommendation(
    store: int,
    item: int,
    current_stock: float,
    days: int = 30,
):
    """
    Compares predicted demand against current stock and returns:
      - recommendation: REORDER | MONITOR | OVERSTOCK
      - safety_stock, reorder_point, recommended_stock
      - risk_level: HIGH | MEDIUM | LOW
      - alerts: list of alert messages
      - inventory_kpis: card data for the frontend
    """
    if not _model_exists():
        raise HTTPException(status_code=400, detail="Model not found.")
    err = _validate_store_item(store, item)
    if err:
        raise HTTPException(status_code=400, detail=err)
    if current_stock < 0:
        raise HTTPException(status_code=400, detail="current_stock must be >= 0.")
    if not (1 <= days <= 90):
        raise HTTPException(status_code=400, detail="days must be 1–90.")

    try:
        history = _load_history(store, item).sort_values("date")
        last_date    = pd.to_datetime(history["date"]).max()
        future_dates = [last_date + timedelta(days=i + 1) for i in range(days)]

        pipeline    = PredictionPipeline(model_path=MODEL_PATH)
        forecast_df = pipeline.predict_multi_step(history, store, item, future_dates)
        preds       = forecast_df["predicted_sales"].values

        # ── Core inventory calculations ───────────────────────────────────
        LEAD_TIME   = 7          # days to receive a new order
        Z_SCORE     = 1.65       # 95% service level
        daily_std   = float(np.std(preds))
        avg_forecast = float(np.mean(preds))
        total_forecast = float(np.sum(preds))

        safety_stock   = round(Z_SCORE * daily_std * (LEAD_TIME ** 0.5), 1)
        reorder_point  = round(avg_forecast * LEAD_TIME + safety_stock, 1)
        # Recommended stock = cover the full forecast period + safety buffer
        recommended_stock = round(total_forecast + safety_stock, 1)

        # ── Decision logic ────────────────────────────────────────────────
        stock_cover_days = (current_stock / avg_forecast) if avg_forecast > 0 else 999

        if current_stock <= reorder_point:
            recommendation = "REORDER"
            risk_level     = "HIGH"
        elif current_stock > recommended_stock * 1.3:
            recommendation = "OVERSTOCK"
            risk_level     = "LOW"
        else:
            recommendation = "MONITOR"
            risk_level     = "MEDIUM"

        # ── Alerts ────────────────────────────────────────────────────────
        alerts = []
        if current_stock <= reorder_point:
            alerts.append({
                "type": "danger",
                "icon": "🚨",
                "message": f"Stock critically low! Current {current_stock:.0f} units is at or below reorder point ({reorder_point} units). Place order immediately.",
            })
        if current_stock > recommended_stock * 1.3:
            alerts.append({
                "type": "warning",
                "icon": "⚠️",
                "message": f"Overstock detected. Current {current_stock:.0f} units exceeds recommended {recommended_stock} units by {current_stock - recommended_stock:.0f} units.",
            })
        if stock_cover_days < LEAD_TIME:
            alerts.append({
                "type": "danger",
                "icon": "⏰",
                "message": f"Only {stock_cover_days:.1f} days of stock remaining — less than lead time ({LEAD_TIME} days). Urgent reorder needed.",
            })
        peak = float(np.max(preds))
        if peak > avg_forecast * 1.5:
            alerts.append({
                "type": "info",
                "icon": "📈",
                "message": f"Demand spike detected: peak forecast of {peak:.0f} units is {((peak/avg_forecast)-1)*100:.0f}% above average. Consider buffer stock.",
            })

        hist_avg   = float(history["sales"].tail(90).mean())
        growth_pct = round(((avg_forecast - hist_avg) / hist_avg) * 100, 1) if hist_avg else 0

        return JSONResponse({
            "recommendation":  recommendation,
            "risk_level":      risk_level,
            "safety_stock":    safety_stock,
            "reorder_point":   reorder_point,
            "recommended_stock": recommended_stock,
            "stock_cover_days":  round(stock_cover_days, 1),
            "alerts": alerts,
            "inventory_kpis": {
                "predicted_demand":   round(total_forecast, 1),
                "current_stock":      round(current_stock, 1),
                "recommended_stock":  recommended_stock,
                "risk_level":         risk_level,
                "growth_pct":         growth_pct,
            },
        })

    except Exception as e:
        logger.error(f"Inventory recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Dashboard JSON API ────────────────────────────────────

@app.get("/api/dashboard")
async def dashboard_data(store: int, item: int, days: int = 30):
    """
    Returns chart-ready JSON for the dashboard.
    Called by JavaScript (fetch) after the multi-step form submits.

    Returns:
      - historical: last 90 days of real sales
      - forecast:   N-day predicted sales
      - kpis:       summary statistics
      - seasonality_weekly: avg sales by day-of-week
      - seasonality_monthly: avg sales by month
      - store_comparison: avg daily sales per store for this item
    """
    if not _model_exists():
        raise HTTPException(status_code=400, detail="Model not found.")

    err = _validate_store_item(store, item)
    if err:
        raise HTTPException(status_code=400, detail=err)

    if not (1 <= days <= 90):
        raise HTTPException(status_code=400, detail="days must be 1–90.")

    try:
        # ── Historical data (last 90 days for this store-item) ────────────
        history = _load_history(store, item).sort_values("date")
        hist_tail = history.tail(90).copy()
        hist_tail["date"] = hist_tail["date"].astype(str)

        # ── Forecast ────────────────────────────────────────────────
        last_date    = pd.to_datetime(history["date"]).max()
        future_dates = [last_date + timedelta(days=i + 1) for i in range(days)]
        pipeline     = PredictionPipeline(model_path=MODEL_PATH)
        forecast_df  = pipeline.predict_multi_step(history, store, item, future_dates)
        forecast_df["date"] = forecast_df["date"].astype(str)

        preds = forecast_df["predicted_sales"].values

        # Simple confidence band: ±15% of predicted value
        upper = (preds * 1.15).round(2).tolist()
        lower = (preds * 0.85).round(2).tolist()

        # ── KPIs ─────────────────────────────────────────────────────
        hist_avg   = float(hist_tail["sales"].mean())
        pred_avg   = float(np.mean(preds))
        growth_pct = round(((pred_avg - hist_avg) / hist_avg) * 100, 1) if hist_avg else 0
        peak       = float(np.max(preds))
        # Confidence: how stable the forecast is (lower CV = higher confidence)
        cv         = float(np.std(preds) / np.mean(preds)) if np.mean(preds) else 1
        confidence = max(0, min(100, round((1 - cv) * 100, 1)))

        # ── Weekly seasonality (avg sales by day-of-week from history) ───
        history["dow"] = pd.to_datetime(history["date"]).dt.day_name()
        dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        weekly = (
            history.groupby("dow")["sales"].mean()
            .reindex(dow_order).round(2)
        )

        # ── Monthly seasonality (avg sales by month from history) ──────
        history["month"] = pd.to_datetime(history["date"]).dt.month_name()
        month_order = ["January","February","March","April","May","June",
                       "July","August","September","October","November","December"]
        monthly = (
            history.groupby("month")["sales"].mean()
            .reindex(month_order).round(2)
        )

        # ── Store comparison (avg daily sales for this item across all stores) ─
        all_data = pd.read_csv(RAW_DATA, parse_dates=["date"])
        all_item = all_data[all_data["item"] == item]
        store_avg = (
            all_item.groupby("store")["sales"].mean().round(2)
        )

        # ── Heatmap: avg sales per store × item (all stores, items 1–50) ──
        pivot = (
            all_data.groupby(["store", "item"])["sales"]
            .mean().round(2)
            .unstack(level="item")          # columns = item IDs
            .sort_index()                   # rows = store IDs ascending
        )
        heatmap_z      = pivot.values.tolist()                          # 2-D list
        heatmap_stores = [f"Store {s}" for s in pivot.index.tolist()]
        heatmap_items  = [f"Item {i}"  for i in pivot.columns.tolist()]

        return JSONResponse({
            "historical": {
                "dates":  hist_tail["date"].tolist(),
                "sales":  hist_tail["sales"].round(2).tolist(),
            },
            "forecast": {
                "dates":  forecast_df["date"].tolist(),
                "sales":  preds.round(2).tolist(),
                "upper":  upper,
                "lower":  lower,
            },
            "kpis": {
                "predicted_sales": round(float(preds[0]), 2),
                "avg_sales":       round(hist_avg, 2),
                "growth_pct":      growth_pct,
                "peak_demand":     round(peak, 2),
                "confidence":      confidence,
            },
            "seasonality_weekly": {
                "days":  weekly.index.tolist(),
                "sales": weekly.values.tolist(),
            },
            "seasonality_monthly": {
                "months": monthly.index.tolist(),
                "sales":  monthly.values.tolist(),
            },
            "store_comparison": {
                "stores": [f"Store {s}" for s in store_avg.index.tolist()],
                "sales":  store_avg.values.tolist(),
                "active_store": store,
            },
            "heatmap": {
                "z":      heatmap_z,
                "stores": heatmap_stores,
                "items":  heatmap_items,
            },
        })

    except Exception as e:
        logger.error(f"Dashboard API failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Run ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=cfg["app"]["host"],
        port=cfg["app"]["port"],
        reload=cfg["app"]["debug"],
    )
