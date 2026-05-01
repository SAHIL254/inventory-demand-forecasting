import os
import sys
import pickle
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)

from src.exception import CustomException
from src.logger import logger


# Config 

def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load YAML config and return as a Python dict."""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Config loaded from: {config_path}")
        return config
    except Exception as e:
        raise CustomException(e, sys)


# Object persistence 

def save_object(file_path: str, obj: Any) -> None:
    """Serialize any Python object to disk using pickle."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str) -> Any:
    """Deserialize a pickle file from disk."""
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)


# Prediction persistence

def save_predictions(df: pd.DataFrame, path: str) -> None:
    """Save a predictions DataFrame to CSV."""
    try:
        dir_name = os.path.dirname(path)
        if dir_name:                        # only makedirs if there's a directory part
            os.makedirs(dir_name, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Predictions saved at: {path}")
    except Exception as e:
        raise CustomException(e, sys)


#Metrics

def smape(preds: np.ndarray, target: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error.
    Excludes pairs where both prediction and target are 0.
    Returns SMAPE as a percentage (0–100).
    """
    preds = np.array(preds)
    target = np.array(target)
    mask = ~((preds == 0) & (target == 0))
    preds, target = preds[mask], target[mask]
    numerator = np.abs(preds - target)
    denominator = (np.abs(preds) + np.abs(target)) / 2
    return float(np.mean(numerator / denominator) * 100)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute MAE, RMSE, R², MAPE, SMAPE.
    Both arrays must be in the original (non-log) sales scale.
    """
    return {
        "MAE":   float(mean_absolute_error(y_true, y_pred)),
        "RMSE":  float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":    float(r2_score(y_true, y_pred)),
        "MAPE":  float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "SMAPE": smape(y_pred, y_true),
    }
