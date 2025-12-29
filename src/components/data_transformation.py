# About this module:
# receives train_data_path,test_data_path
# perform transformation on train and test input features
# then saves the preprocessor object in "artifacts" folder
# returns: train_arr,test_arr,preprocessor path

from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
import os, sys, numpy as np, pandas as pd
from src.logger import logging
from src.components.data_ingestion import DataIngestionConfig
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessor_obj(self):
        try:
            cat_columns = ["sex", "smoker", "region"]
            num_columns = ["age", "bmi", "children"]
            cat_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoding", OneHotEncoder()),
                    (
                        "standard_scaling",
                        StandardScaler(with_mean=False),
                    ),  # with_mean=False, means z = x/SD not, z=(x-u)/SD
                    # It is important when dealing with sparse matrices
                ]
            )
            num_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("standard_scaling", StandardScaler()),
                ]
            )

            preprocessor_pipeline = ColumnTransformer(
                [
                    ("one_hot_encoder", cat_pipeline, cat_columns),
                    ("standard_scaler", num_pipeline, num_columns),
                ]
            )
            return preprocessor_pipeline

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info(f"Data transformation begins")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train and test data is completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_preprocessor_obj()

            target_column_name = "expenses"

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # concatenating i/p and o/p features of train,test data
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path,
                obj=preprocessing_obj,
            )
            logging.info(f"Saved preprocessing object.")
            logging.info(f"Data transformation completed")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)


# if __name__ == "__main__":
#     from src.components.data_ingestion import DataIngestion

#     data_ingestion_obj = DataIngestion()
#     train_data_path, test_data_path = data_ingestion_obj.initiate_ingestion()
#     data_transformation_obj = DataTransformation()
#     _, _, preprocessor_path = data_transformation_obj.initiate_data_transformation(
#         train_path=train_data_path, test_path=test_data_path
#     )
#     print(preprocessor_path)
