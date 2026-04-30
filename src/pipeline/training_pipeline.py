import os
import sys
import time

from src.exception import CustomException
from src.logger import logger
from src.utils import load_config
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.feature_engineering import FeatureEngineering, FeatureEngineeringConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


class TrainingPipeline:
    def run(self, source_path: str = None) -> tuple:
        
        cfg = load_config()
        if source_path is None:
            source_path = cfg["paths"]["source_data"]

        pipeline_start = time.time()
        logger.info("=" * 55)
        logger.info("  TRAINING PIPELINE STARTED")
        logger.info("=" * 55)

        # Stage 1: Data Ingestion 
        logger.info("── Stage 1/4: Data Ingestion ──")
        t0 = time.time()
        try:
            ingestion = DataIngestion(DataIngestionConfig())
            train_path, test_path = ingestion.initiate_data_ingestion(source_path)
            logger.info(f"Stage 1 done in {time.time() - t0:.1f}s")
        except Exception as e:
            raise CustomException(f"[Stage 1 - Data Ingestion] {e}", sys)

        # Stage 2: Data Transformation 
        logger.info("── Stage 2/4: Data Transformation ──")
        t0 = time.time()
        try:
            transformation = DataTransformation(DataTransformationConfig())
            train_df, test_df = transformation.initiate_data_transformation(
                train_path, test_path
            )
            logger.info(f"Stage 2 done in {time.time() - t0:.1f}s")
        except Exception as e:
            raise CustomException(f"[Stage 2 - Data Transformation] {e}", sys)

        #Stage 3: Feature Engineering
        logger.info("── Stage 3/4: Feature Engineering ──")
        t0 = time.time()
        try:
            fe = FeatureEngineering(FeatureEngineeringConfig())
            X_train, y_train, X_test, y_test, feature_cols = fe.fit_transform(
                train_df, test_df
            )
            logger.info(
                f"Train shape: {X_train.shape} | Test shape: {X_test.shape}"
            )
            logger.info(f"Stage 3 done in {time.time() - t0:.1f}s")
        except Exception as e:
            raise CustomException(f"[Stage 3 - Feature Engineering] {e}", sys)

        # Stage 4: Model Training & Evaluation 
        logger.info("── Stage 4/4: Model Training & Evaluation ──")
        t0 = time.time()
        try:
            trainer = ModelTrainer(ModelTrainerConfig())
            best_name, results = trainer.initiate_model_training(
                X_train, y_train, X_test, y_test
            )
            logger.info(f"Stage 4 done in {time.time() - t0:.1f}s")
        except Exception as e:
            raise CustomException(f"[Stage 4 - Model Training] {e}", sys)

        #Summary 
        total = time.time() - pipeline_start
        logger.info("=" * 55)
        logger.info(f"  PIPELINE FINISHED in {total:.1f}s | Best: {best_name}")
        logger.info("=" * 55)

        return best_name, results


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    best, results = pipeline.run()
    print(f"\nBest Model: {best}")
    print(results.to_string(index=False))
