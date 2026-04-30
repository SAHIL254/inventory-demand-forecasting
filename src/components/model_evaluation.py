import sys

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logger
from src.utils import evaluate_model


class ModelEvaluation:
    def evaluate_all(
        self, models: dict, X_test: pd.DataFrame, y_test_log: pd.Series
    ) -> pd.DataFrame:
        
        logger.info("Evaluating models")
        try:
            y_true = np.expm1(y_test_log)
            rows = []
            for name, model in models.items():
                y_pred_log = model.predict(X_test)
                y_pred     = np.expm1(y_pred_log)
                metrics    = evaluate_model(y_true, y_pred)
                metrics["Model"] = name
                rows.append(metrics)
                logger.info(f"{name}: {metrics}")

            results = pd.DataFrame(rows)[
                ["Model", "MAE", "RMSE", "R2", "MAPE", "SMAPE"]
            ]
            results = results.sort_values(
                by=["SMAPE", "MAPE", "RMSE", "MAE"]
            ).reset_index(drop=True)
            return results

        except Exception as e:
            raise CustomException(e, sys)
