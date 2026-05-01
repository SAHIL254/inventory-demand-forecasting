import os
import sys
import pathlib
from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.exception import CustomException
from src.logger import logger
from src.utils import load_config, save_object
from src.components.model_evaluation import ModelEvaluation


@dataclass
class ModelTrainerConfig:
    model_path:        str = load_config()["paths"]["model"]
    preprocessor_path: str = load_config()["paths"]["preprocessor"]


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig = ModelTrainerConfig()):
        self.config = config
        self.cfg    = load_config()

    def initiate_model_training(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test:  pd.DataFrame,
        y_test:  pd.Series,
    ) -> tuple:
        
        logger.info("Starting model training")
        try:
            rf_cfg  = self.cfg["models"]["random_forest"]
            xgb_cfg = self.cfg["models"]["xgboost"]
            lgb_cfg = self.cfg["models"]["lightgbm"]

            # Linear Regression is sensitive to feature scale; tree-based models are not.
            # LR is wrapped in a sklearn Pipeline so prediction stays seamless.
            # Do NOT apply this scaler to RF, XGBoost, or LightGBM inputs.
            scaler         = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled  = scaler.transform(X_test)

            lr = LinearRegression()
            lr.fit(X_train_scaled, y_train)

            # Wrap scaler + LR into a Pipeline so prediction is seamless
            lr_pipeline = Pipeline([("scaler", scaler), ("lr", lr)])
            logger.info("Linear Regression trained")

            # Random Forest 
            rf = RandomForestRegressor(
                n_estimators=rf_cfg["n_estimators"],
                max_depth=rf_cfg["max_depth"],
                random_state=rf_cfg["random_state"],
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            logger.info("Random Forest trained")

            # XGBoost 
            xgb = XGBRegressor(
                n_estimators=xgb_cfg["n_estimators"],
                learning_rate=xgb_cfg["learning_rate"],
                max_depth=xgb_cfg["max_depth"],
                tree_method=xgb_cfg["tree_method"],
                early_stopping_rounds=xgb_cfg["early_stopping_rounds"],
                random_state=xgb_cfg["random_state"],
                eval_metric="rmse",
                n_jobs=-1,
            )
            # Early stopping monitors log-scale RMSE on the validation set.
            # Final model selection uses SMAPE on original scale (model_evaluation.py).
            # These metrics do not always agree; treat best_iteration as approximate.
            xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            logger.info(f"XGBoost trained | best iteration: {xgb.best_iteration}")

            # LightGBM 
            lgb = LGBMRegressor(
                n_estimators=lgb_cfg["n_estimators"],
                learning_rate=lgb_cfg["learning_rate"],
                max_depth=lgb_cfg["max_depth"],
                num_leaves=lgb_cfg["num_leaves"],
                random_state=lgb_cfg["random_state"],
                n_jobs=-1,
                verbose=-1,
                importance_type="gain",
            )
            lgb.fit(X_train, y_train)
            logger.info("LightGBM trained")

            # Evaluate all models on the hold-out test set
            models = {
                "Linear Regression": lr_pipeline,
                "Random Forest":     rf,
                "XGBoost":           xgb,
                "LightGBM":          lgb,
            }

            fi_path = os.path.join(os.path.dirname(self.config.model_path), "feature_importance.csv")
            fi_rows = []
            for name, model in models.items():
                if name == "Linear Regression":
                    continue
                m = model  # tree models stored directly
                importances = m.feature_importances_
                fi_rows.append(
                    pd.DataFrame({
                        "model": name,
                        "feature": X_train.columns.tolist(),
                        "importance": importances
                    })
                )
            if fi_rows:
                pd.concat(fi_rows).sort_values(["model", "importance"], ascending=[True, False])\
                  .to_csv(fi_path, index=False)
                logger.info(f"Feature importances saved at: {fi_path}")

            evaluator = ModelEvaluation()
            results   = evaluator.evaluate_all(models, X_test, y_test)
            print("\n" + results.to_string(index=False) + "\n")

            # Pick best model 
            best_name  = results.iloc[0]["Model"]
            best_model = models[best_name]
            logger.info(f"Best model: {best_name}")

            # Save best model 
            save_object(self.config.model_path, best_model)

            # Save standalone scaler only for tree-based best models.
            # LR already has the scaler embedded inside lr_pipeline.
            if best_name != "Linear Regression":
                save_object(self.config.preprocessor_path, scaler)
            else:
                logger.info(
                    "Best model is LR pipeline — scaler is embedded, "
                    "skipping standalone preprocessor save"
                )

            stale = pathlib.Path(self.config.preprocessor_path)
            if best_name == "Linear Regression" and stale.exists():
                stale.unlink()
                logger.info("Removed stale preprocessor.pkl (LR pipeline has scaler embedded)")

            return best_name, results

        except Exception as e:
            raise CustomException(e, sys)
