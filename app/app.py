"""
app.py
------
FastAPI web application for Inventory Demand Forecasting.

Routes:
  GET  /              — Home page (model status)
  POST /train         — Trigger full training pipeline
  GET  /predict       — Single-prediction form
  POST /predict       — Run single-step prediction
  POST /predict/batch — JSON API for batch predictions
  POST /predict/multi — JSON API for multi-step (7/30-day) forecasts
"""

import os
from datetime import timedelta
from typing import List, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.pipeline.training_pipeline import TrainingPipeline
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

# Pre-load history data once at startup to avoid repeated disk reads
try:
    _HISTORY_DF = pd.read_csv(RAW_DATA, parse_dates=["date"])
    logger.info(f"History data pre-loaded: {_HISTORY_DF.shape}")
except Exception:
    _HISTORY_DF = None
    logger.warning("Could not pre-load history data at startup")


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
    df = _HISTORY_DF if _HISTORY_DF is not None else pd.read_csv(RAW_DATA, parse_dates=["date"])
    return df[(df["store"] == store) & (df["item"] == item)].copy()


def _render(request: Request, template: str, **ctx):
    """Shorthand for templates.TemplateResponse."""
    ctx.setdefault("flash_msg", None)
    ctx.setdefault("flash_cat", None)
    return templates.TemplateResponse(template, {"request": request, **ctx})


# ── Routes ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return _render(request, "index.html", model_exists=_model_exists())


@app.post("/train", response_class=HTMLResponse)
async def train(request: Request):
    """Trigger the full training pipeline from the web UI."""
    try:
        logger.info("Training triggered from web UI")
        pipeline = TrainingPipeline()
        best_name, results = pipeline.run(source_path=RAW_DATA)

        results_html = results.to_html(
            index=False,
            classes="results-table",
            border=0,
            float_format=lambda x: f"{x:.4f}",
        )
        return _render(
            request, "result.html",
            best_model=best_name,
            results_table=results_html,
            flash_msg=f"Training complete! Best model: <strong>{best_name}</strong>",
            flash_cat="success",
        )

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return _render(
            request, "index.html",
            model_exists=_model_exists(),
            flash_msg=f"Training failed: {str(e)}",
            flash_cat="danger",
        )


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
        predicted_sales = round(float(result_df["predicted_sales"].iloc[0]), 2)

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


@app.post("/predict/batch")
async def predict_batch(body: BatchRequest):
    """
    JSON API — predict sales for multiple store-item-date combinations.

    Request body:
        {"records": [{"store": 1, "item": 5, "date": "2018-01-01"}, ...]}

    Response:
        {"success": true, "predictions": [...]}
    """
    if not _model_exists():
        raise HTTPException(status_code=400, detail="Model not found. Train first.")

    if not body.records:
        raise HTTPException(status_code=400, detail="No records provided.")

    for i, rec in enumerate(body.records):
        err = _validate_store_item(rec.store, rec.item)
        if err:
            raise HTTPException(status_code=400, detail=f"Record {i}: {err}")
        try:
            pd.to_datetime(rec.date)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Record {i}: invalid date '{rec.date}'.")

    try:
        records_dicts = [r.model_dump() for r in body.records]
        future_df         = pd.DataFrame(records_dicts)
        future_df["date"] = pd.to_datetime(future_df["date"])
        history           = _HISTORY_DF if _HISTORY_DF is not None else pd.read_csv(RAW_DATA, parse_dates=["date"])

        pipeline  = PredictionPipeline(model_path=MODEL_PATH)
        result_df = pipeline.predict_from_raw(history, future_df)
        result_df["predicted_sales"] = result_df["predicted_sales"].round(2)
        result_df["date"] = result_df["date"].astype(str)

        return {"success": True, "predictions": result_df.to_dict(orient="records")}

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/multi")
async def predict_multi(body: MultiRequest):
    """
    JSON API — recursive multi-step forecast for a single store-item pair.

    Request body:
        {"store": 1, "item": 5, "days": 7}

    Response:
        {"success": true, "store": 1, "item": 5, "days": 7, "predictions": [...]}
    """
    if not _model_exists():
        raise HTTPException(status_code=400, detail="Model not found. Train first.")

    err = _validate_store_item(body.store, body.item)
    if err:
        raise HTTPException(status_code=400, detail=err)

    if not (1 <= body.days <= 90):
        raise HTTPException(status_code=400, detail="days must be between 1 and 90.")

    try:
        history      = _load_history(body.store, body.item)
        last_date    = pd.to_datetime(history["date"]).max()
        future_dates = [last_date + timedelta(days=i + 1) for i in range(body.days)]

        pipeline  = PredictionPipeline(model_path=MODEL_PATH)
        result_df = pipeline.predict_multi_step(history, body.store, body.item, future_dates)
        result_df["date"] = result_df["date"].astype(str)

        return {
            "success":     True,
            "store":       body.store,
            "item":        body.item,
            "days":        body.days,
            "predictions": result_df[["date", "predicted_sales"]].to_dict(orient="records"),
        }

    except Exception as e:
        logger.error(f"Multi-step prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Run ───────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=cfg["app"]["host"],
        port=cfg["app"]["port"],
        reload=cfg["app"]["debug"],
    )
