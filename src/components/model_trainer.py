import os, sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, find_best_model

from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Initiating model training")
            logging.info("Splitting train & test arr into input and output features")
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],  # i/p
                train_arr[:, -1],  # o/p
                test_arr[:, :-1],  # i/p
                test_arr[:, -1],  # o/p
            )
            models = {
                "Linear": LinearRegression(),
                "Decision_Tree": DecisionTreeRegressor(),
                "Random_Forest": RandomForestRegressor(),
                "GradientBoost": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(),
            }
            params = {
                "Decision_Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                },
                "Random_Forest": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "GradientBoost": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    # "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # "criterion": [
                    #     "squared_error",
                    #     "friedman_mse",
                    # ],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear": {},
                "XGBoost": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "AdaBoost": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            model_report: dict = evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params,
            )
            ## To get best model score from dict
            best_model_name, best_model_obj, best_acc = find_best_model(model_report)

            if best_acc < 0.6:
                logging.info(f"Best model not found")
                raise CustomException("No best model found")
            logging.info(f"Best model found")
            logging.info("Model training completed")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model_obj,
            )

            predicted = best_model_obj.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            return round(r2_square * 100, 2)

        except Exception as e:
            raise CustomException(e, sys)


# # just for testing
# from src.components.data_ingestion import DataIngestionConfig
# from src.components.data_transformation import DataTransformation

# if __name__ == "__main__":
#     obj = DataTransformation()
#     ingestion_config = DataIngestionConfig()
#     train_data, test_data, preprocessor_path = obj.initiate_data_transformation(
#         ingestion_config.train_data_path, ingestion_config.test_data_path
#     )
#     modeltrainer = ModelTrainer()
#     r2score = modeltrainer.initiate_model_trainer(
#         train_arr=train_data, test_arr=test_data
#     )
#     print(r2score)
