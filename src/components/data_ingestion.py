import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import logger
from src.utils import load_config


@dataclass
class DataIngestionConfig:
    raw_data_path:   str = load_config()["paths"]["raw_data"]
    train_data_path: str = load_config()["paths"]["train_data"]
    test_data_path:  str = load_config()["paths"]["test_data"]
    split_date:      str = load_config()["split"]["date"]


class DataIngestion:
    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config

    def initiate_data_ingestion(self, source_path: str):
        logger.info("Starting data ingestion")
        try:
            df = pd.read_csv(source_path, parse_dates=["date"])
            logger.info(f"Dataset loaded: shape={df.shape}")

            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)

            train = df[df["date"] < self.config.split_date]
            test  = df[df["date"] >= self.config.split_date]

            train.to_csv(self.config.train_data_path, index=False)
            test.to_csv(self.config.test_data_path,   index=False)

            logger.info(
                f"Train: {train.shape} | "
                f"{train['date'].min().date()} → {train['date'].max().date()}"
            )
            logger.info(
                f"Test:  {test.shape}  | "
                f"{test['date'].min().date()} → {test['date'].max().date()}"
            )
            return self.config.train_data_path, self.config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)
