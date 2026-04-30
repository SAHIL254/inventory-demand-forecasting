"""
app.py
------
Flask web application for Inventory Demand Forecasting.

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

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

from src.pipeline.training_pipeline import TrainingPipeline
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.utils import load_config
from src.logger import logger



app = Flask(__name__)
app.secret_key = "inventory_forecast_secret"

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


# Helpers 

def _model_exists() -> bool:
    return os.path.exists(MODEL_PATH)


def _validate_store_item(store: int, item: int):
    """Return an error string if inputs are out of range, else None."""
    if not (STORE_MIN <= store <= STORE_MAX):
        return f"Store ID must be between {STORE_MIN} and {STORE_MAX}."
    if not (ITEM_MIN <= item <= ITEM_MAX):
        return f"Item ID must be between {ITEM_MIN} and {ITEM_MAX}."
    return None


def _load_history(store: int, item: int) -> pd.DataFrame:
    df = _HISTORY_DF if _HISTORY_DF is not None else pd.read_csv(RAW_DATA, parse_dates=["date"])
    return df[(df["store"] == store) & (df["item"] == item)].copy()


# Routes 

@app.route("/")
def index():
    return render_template("index.html", model_exists=_model_exists())


@app.route("/train", methods=["POST"])
def train():
    """Trigger the full training pipeline from the web UI."""
    try:
        logger.info("Training triggered from web UI")
        pipeline  = TrainingPipeline()
        best_name, results = pipeline.run(source_path=RAW_DATA)

        results_html = results.to_html(
            index=False,
            classes="results-table",
            border=0,
            float_format=lambda x: f"{x:.4f}",
        )
        flash(
            f"Training complete! Best model: <strong>{best_name}</strong>",
            "success",
        )
        return render_template(
            "result.html",
            best_model=best_name,
            results_table=results_html,
        )

    except Exception as e:
        logger.error(f"Training failed: {e}")
        flash(f"Training failed: {str(e)}", "danger")
        return redirect(url_for("index"))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Single-step prediction form."""
    template_kwargs = dict(
        prediction=None,
        store_min=STORE_MIN, store_max=STORE_MAX,
        item_min=ITEM_MIN,   item_max=ITEM_MAX,
    )

    if request.method == "GET":
        return render_template("predict.html", **template_kwargs)

    #Validate model 
    if not _model_exists():
        flash("Model not found. Please train first.", "danger")
        return redirect(url_for("index"))

    #Parse and validate inputs
    try:
        store    = int(request.form["store"])
        item     = int(request.form["item"])
        date_str = request.form["date"].strip()
    except (ValueError, KeyError):
        flash("Invalid input. Please fill all fields correctly.", "danger")
        return render_template("predict.html", **template_kwargs)

    err = _validate_store_item(store, item)
    if err:
        flash(err, "danger")
        return render_template("predict.html", **template_kwargs)

    try:
        pd.to_datetime(date_str)
    except ValueError:
        flash("Invalid date format. Use YYYY-MM-DD.", "danger")
        return render_template("predict.html", **template_kwargs)

    history_check = _load_history(store, item)
    if len(history_check) < 365:
        flash(
            f"Warning: only {len(history_check)} days of history for store {store}, "
            f"item {item}. Lag features may be incomplete — prediction may be unreliable.",
            "warning"
        )

    #Run prediction
    try:
        history  = _load_history(store, item)
        future   = pd.DataFrame(
            {"date": [pd.to_datetime(date_str)], "store": [store], "item": [item]}
        )

        pipeline        = PredictionPipeline(model_path=MODEL_PATH)
        result_df       = pipeline.predict_from_raw(history, future)
        predicted_sales = round(float(result_df["predicted_sales"].iloc[0]), 2)

        return render_template(
            "predict.html",
            prediction=predicted_sales,
            store=store,
            item=item,
            date=date_str,
            **{k: v for k, v in template_kwargs.items() if k != "prediction"},
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        flash(f"Prediction failed: {str(e)}", "danger")
        return render_template("predict.html", **template_kwargs)


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """
    JSON API — predict sales for multiple store-item-date combinations.

    Request body:
        {"records": [{"store": 1, "item": 5, "date": "2018-01-01"}, ...]}

    Response:
        {"success": true, "predictions": [...]}
    """
    if not _model_exists():
        return jsonify({"success": False, "error": "Model not found. Train first."}), 400

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"success": False, "error": "Invalid JSON body."}), 400

    records = data.get("records", [])
    if not records:
        return jsonify({"success": False, "error": "No records provided."}), 400

    for i, rec in enumerate(records):
        try:
            store = int(rec["store"])
            item  = int(rec["item"])
            pd.to_datetime(rec["date"])
        except (KeyError, ValueError) as e:
            return jsonify({"success": False, "error": f"Record {i} invalid: {e}"}), 400
        err = _validate_store_item(store, item)
        if err:
            return jsonify({"success": False, "error": f"Record {i}: {err}"}), 400

    try:
        future_df         = pd.DataFrame(records)
        future_df["date"] = pd.to_datetime(future_df["date"])
        history           = _HISTORY_DF if _HISTORY_DF is not None else pd.read_csv(RAW_DATA, parse_dates=["date"])

        pipeline  = PredictionPipeline(model_path=MODEL_PATH)
        result_df = pipeline.predict_from_raw(history, future_df)
        result_df["predicted_sales"] = result_df["predicted_sales"].round(2)
        result_df["date"] = result_df["date"].astype(str)

        return jsonify({"success": True, "predictions": result_df.to_dict(orient="records")})

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/predict/multi", methods=["POST"])
def predict_multi():
    """
    JSON API — recursive multi-step forecast for a single store-item pair.

    Request body:
        {"store": 1, "item": 5, "days": 7}

    Response:
        {"success": true, "store": 1, "item": 5, "days": 7, "predictions": [...]}
    """
    if not _model_exists():
        return jsonify({"success": False, "error": "Model not found. Train first."}), 400

    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"success": False, "error": "Invalid JSON body."}), 400

    try:
        store = int(data["store"])
        item  = int(data["item"])
        days  = int(data.get("days", 7))
    except (KeyError, ValueError) as e:
        return jsonify({"success": False, "error": f"Invalid input: {e}"}), 400

    err = _validate_store_item(store, item)
    if err:
        return jsonify({"success": False, "error": err}), 400

    if not (1 <= days <= 90):
        return jsonify({"success": False, "error": "days must be between 1 and 90."}), 400

    try:
        history      = _load_history(store, item)
        last_date    = pd.to_datetime(history["date"]).max()
        future_dates = [last_date + timedelta(days=i + 1) for i in range(days)]

        pipeline  = PredictionPipeline(model_path=MODEL_PATH)
        result_df = pipeline.predict_multi_step(history, store, item, future_dates)
        result_df["date"] = result_df["date"].astype(str)

        return jsonify(
            {
                "success":     True,
                "store":       store,
                "item":        item,
                "days":        days,
                "predictions": result_df[["date", "predicted_sales"]].to_dict(
                    orient="records"
                ),
            }
        )

    except Exception as e:
        logger.error(f"Multi-step prediction failed: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


#Run 

if __name__ == "__main__":
    app.run(
        debug=cfg["app"]["debug"],
        host=cfg["app"]["host"],
        port=cfg["app"]["port"],
    )
