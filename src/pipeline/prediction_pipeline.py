import os
import sys
from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logger
from src.utils import load_config, load_object, save_predictions
from src.components.feature_engineering import FeatureEngineering, FeatureEngineeringConfig


class PredictionPipeline:
    def __init__(
        self,
        model_path: str = load_config()["paths"]["model"],
    ):
        self.model = load_object(model_path)
        self.cfg   = load_config()

    # Core predict (feature-ready DataFrame) 

    def predict(self, feature_df: pd.DataFrame) -> np.ndarray:
        """
        Run the saved model on a feature-ready DataFrame.
        Returns predicted sales in the original (non-log) scale.
        """
        logger.info("Running model inference")
        try:
            predictions_log = self.model.predict(feature_df)
            return np.expm1(predictions_log)
        except Exception as e:
            raise CustomException(e, sys)

    # Single-step prediction from raw data 

    def predict_from_raw(
        self,
        history_df: pd.DataFrame,
        future_df:  pd.DataFrame,
        save_path:  str = None,
    ) -> pd.DataFrame:
        
        logger.info(f"Single-step prediction for {len(future_df)} row(s)")
        try:
            result_df = self._build_and_predict(history_df, future_df)

            path = save_path or self.cfg["paths"]["predictions"]
            save_predictions(result_df, path)

            return result_df

        except Exception as e:
            raise CustomException(e, sys)

    # Multi-step recursive forecasting 

    def predict_multi_step(
        self,
        history_df:   pd.DataFrame,
        store:        int,
        item:         int,
        future_dates: List[pd.Timestamp],
        save_path:    str = None,
    ) -> pd.DataFrame:
        """
        Recursively forecast sales for multiple consecutive future dates for a
        single store-item pair.

        How it works:
          - Day 1 is predicted using only real historical data.
          - Day 2 uses real history + Day 1's prediction as lag_1.
          - Day 3 uses real history + Day 1 & 2 predictions, and so on.

        Args:
            history_df:   Historical sales data for the store-item pair.
            store:        Store ID to forecast.
            item:         Item ID to forecast.
            future_dates: Sorted list of future dates to predict.
            save_path:    Optional CSV path to save results.

        Limitations:
            Forecast accuracy degrades beyond ~14 days as prediction errors
            compound recursively. Use results beyond 30 days with caution.

        Returns:
            DataFrame with columns [date, store, item, predicted_sales].
        """
        logger.info(
            f"Multi-step prediction | store={store}, item={item}, "
            f"steps={len(future_dates)}"
        )
        try:
            history = history_df[
                (history_df["store"] == store) & (history_df["item"] == item)
            ].copy()
            history["date"]      = pd.to_datetime(history["date"])
            history["sales_log"] = np.log1p(history["sales"])

            results = []

            for forecast_date in sorted(future_dates):
                forecast_date = pd.to_datetime(forecast_date)

                future_row = pd.DataFrame(
                    {"date": [forecast_date], "store": [store], "item": [item]}
                )

                pred_df         = self._build_and_predict(history, future_row)
                predicted_sales = float(pred_df["predicted_sales"].iloc[0])

                results.append(
                    {
                        "date":            forecast_date,
                        "store":           store,
                        "item":            item,
                        "predicted_sales": round(predicted_sales, 2),
                    }
                )

                # Feed prediction back into history for the next step
                new_row = pd.DataFrame(
                    {
                        "date":      [forecast_date],
                        "store":     [store],
                        "item":      [item],
                        "sales":     [predicted_sales],
                        "sales_log": [np.log1p(predicted_sales)],
                    }
                )
                history = pd.concat([history, new_row], ignore_index=True)
                logger.info(
                    f"  {forecast_date.date()} → predicted_sales={predicted_sales:.2f}"
                )

            result_df = pd.DataFrame(results)

            path = save_path or self.cfg["paths"]["predictions"]
            save_predictions(result_df, path)

            return result_df

        except Exception as e:
            raise CustomException(e, sys)

    #Internal helper

    def _build_and_predict(
        self,
        history_df: pd.DataFrame,
        future_df:  pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Combine history + future, build features, run model.

        history_df must already have a sales_log column (real values).
        future_df rows get placeholder sales=0 / sales_log=0.0 — these
        values are never used as lag sources for themselves.
        """
        future_df = future_df.copy()
        future_df["sales"]     = 0
        future_df["sales_log"] = 0.0
        future_df["_is_future"] = True          # marker to identify prediction rows

        # Compute sales_log from real history only (skip if already present)
        history_df = history_df.copy()
        history_df["date"] = pd.to_datetime(history_df["date"])
        if "sales_log" not in history_df.columns:
            history_df["sales_log"] = np.log1p(history_df["sales"])
        history_df["_is_future"] = False

        # Keep only the store-item pairs that appear in future_df so that
        # lag/rolling features are computed per-pair only.
        pairs = future_df[["store", "item"]].drop_duplicates()
        history_df = history_df.merge(pairs, on=["store", "item"], how="inner")

        combined = pd.concat([history_df, future_df], ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.sort_values(["store", "item", "date"]).reset_index(drop=True)

        fe = FeatureEngineering(FeatureEngineeringConfig())
        combined = fe._add_time_features(combined)
        combined = fe._add_lag_features(combined)
        combined = fe._add_rolling_features(combined)

        feature_cols = fe._get_feature_cols(combined)

        # Use the _is_future flag — not split_date — to select exactly the
        # rows that need a prediction. This is safe even when history rows
        # share the same date as a future row.
        pred_rows = combined[combined["_is_future"] == True][feature_cols]

        preds = self.predict(pred_rows)

        future_df = future_df.drop(columns=["sales", "sales_log", "_is_future"])
        future_df = future_df.reset_index(drop=True)
        future_df["predicted_sales"] = preds
        return future_df[["date", "store", "item", "predicted_sales"]]
