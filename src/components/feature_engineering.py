import sys
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logger
from src.utils import load_config


@dataclass
class FeatureEngineeringConfig:
    lag_days:        List[int] = field(
        default_factory=lambda: load_config()["features"]["lag_days"]
    )
    rolling_windows: List[int] = field(
        default_factory=lambda: load_config()["features"]["rolling_windows"]
    )
    group_cols:      List[str] = field(
        default_factory=lambda: load_config()["features"]["group_cols"]
    )


class FeatureEngineering:
    def __init__(self, config: FeatureEngineeringConfig = FeatureEngineeringConfig()):
        self.config = config

    #Public API 

    def fit_transform(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple:
        
        logger.info("Starting feature engineering")
        try:
            combined = (
                pd.concat([train, test], ignore_index=True)
                .sort_values(["store", "item", "date"])
                .reset_index(drop=True)
            )

            combined = self._add_time_features(combined)
            combined = self._add_lag_features(combined)
            combined = self._add_rolling_features(combined)

            # lag_365 NaNs are already filled with 0 and flagged separately.
            non_365_lags = [f"lag_{l}" for l in self.config.lag_days if l != 365]
            before = len(combined)
            combined = combined.dropna(subset=non_365_lags).reset_index(drop=True)
            logger.info(
                f"Dropped {before - len(combined)} rows with insufficient lag history"
            )

            # Re-split
            split_date = test["date"].min()
            train_fe = combined[combined["date"] < split_date].copy()
            test_fe  = combined[combined["date"] >= split_date].copy()

            feature_cols = self._get_feature_cols(combined)
            target_col   = "sales_log"

            X_train = train_fe[feature_cols]
            y_train = train_fe[target_col]
            X_test  = test_fe[feature_cols]
            y_test  = test_fe[target_col]

            logger.info(f"Feature engineering done | {len(feature_cols)} features")
            return X_train, y_train, X_test, y_test, feature_cols

        except Exception as e:
            raise CustomException(e, sys)

    #Private helpers

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calendar features + cyclical sin/cos encoding for month and day-of-week.
        Cyclical encoding preserves the circular nature of time
        (e.g. December → January is a small step, not a large one).
        """
        df["year"]           = df["date"].dt.year
        df["month"]          = df["date"].dt.month
        df["day"]            = df["date"].dt.day
        df["day_of_week"]    = df["date"].dt.dayofweek
        df["is_weekend"]     = df["day_of_week"].isin([5, 6]).astype(int)
        df["quarter"]        = df["date"].dt.quarter
        df["time_index"]     = (df["date"] - df["date"].min()).dt.days
        df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
        df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)

        # Cyclical encoding
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["dow_sin"]   = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"]   = np.cos(2 * np.pi * df["day_of_week"] / 7)
        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lagged sales_log values per store-item group.
        - lag_1 … lag_90 : left as NaN when history is insufficient
          (rows will be dropped in fit_transform via dropna).
        - lag_365 : filled with 0 when the previous year is unavailable
          (first year of data), and a binary flag lag_365_missing is added.
        """
        grp = df.groupby(self.config.group_cols)["sales_log"]
        for lag in self.config.lag_days:
            col = f"lag_{lag}"
            df[col] = grp.shift(lag)
            if lag == 365:
                df["lag_365_missing"] = df[col].isna().astype(int)
                df[col] = df[col].fillna(0)
            # All other lags stay NaN — handled by dropna in fit_transform
        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        grp = df.groupby(self.config.group_cols)["sales_log"]
        for w in self.config.rolling_windows:
            df[f"rolling_mean_{w}"] = grp.transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean()
            )
            df[f"rolling_std_{w}"] = grp.transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).std().fillna(0)
            )
        return df

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        """Return all columns to be used as model input features."""
        drop = {"date", "sales", "sales_log", "_is_future"}
        return [c for c in df.columns if c not in drop]
