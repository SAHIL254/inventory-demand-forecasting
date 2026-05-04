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
from datetime import timedelta
from typing import List, Optional

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
        result_df["predicted_sales"] = result_df["predicted_sales"].round(2)
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


# ── Run ───────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=cfg["app"]["host"],
        port=cfg["app"]["port"],
        reload=cfg["app"]["debug"],
    )
