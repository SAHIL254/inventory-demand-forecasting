import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logger
from src.utils import load_config


@dataclass
class DataTransformationConfig:
    processed_data_path: str = load_config()["paths"]["processed_data"]


class DataTransformation:
    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        self.config = config

    def initiate_data_transformation(
        self, train_path: str, test_path: str
    ) -> tuple:
        
        logger.info("Starting data transformation")
        try:
            train = pd.read_csv(train_path, parse_dates=["date"])
            test  = pd.read_csv(test_path,  parse_dates=["date"])

            # Validate columns
            required = {"date", "store", "item", "sales"}
            for name, df in [("train", train), ("test", test)]:
                missing = required - set(df.columns)
                if missing:
                    raise ValueError(f"[{name}] Missing columns: {missing}")
            logger.info("Column validation passed")

            # Handle missing values
            for split_name, split_df in [("train", train), ("test", test)]:
                n_missing = split_df.isnull().sum().sum()
                if n_missing > 0:
                    logger.warning(f"[{split_name}] Found {n_missing} missing values — applying ffill + bfill")
                    split_df.sort_values(["store", "item", "date"], inplace=True)
                    split_df["sales"] = (
                        split_df.groupby(["store", "item"])["sales"]
                        .transform(lambda x: x.ffill().bfill())
                    )

            # Clip negative sales
            train["sales"] = train["sales"].clip(lower=0)
            test["sales"]  = test["sales"].clip(lower=0)
            logger.info("Negative sales clipped to 0")

            #Log-transform target 
            # log1p(x) = log(1 + x) — safe for zero sales
            train["sales_log"] = np.log1p(train["sales"])
            test["sales_log"]  = np.log1p(test["sales"])
            logger.info("log1p transformation applied to sales column")

            #Save combined processed data
            combined = pd.concat([train, test], ignore_index=True)
            os.makedirs(os.path.dirname(self.config.processed_data_path), exist_ok=True)
            combined.to_csv(self.config.processed_data_path, index=False)
            logger.info(f"Processed data saved at: {self.config.processed_data_path}")

            logger.info(
                f"Data transformation complete | "
                f"Train: {train.shape}, Test: {test.shape}"
            )
            return train, test

        except Exception as e:
            raise CustomException(e, sys)
