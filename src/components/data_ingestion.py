# About this module:
# we collect data from data source(csv file,databases,etc) and split it into train_set,test_set
# then store the train_set,test_set and also the raw_data(not splitted) in the 'artifacts' folder

import os, sys, pandas as pd
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from src.logger import logging
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "insurance.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_ingestion(self):
        try:
            logging.info("Data ingestion begins")
            file_path = os.path.join(os.getcwd(), "data_src", "insurance.csv")

            logging.info("Data fetch begins")
            data = pd.read_csv(file_path)
            logging.info("Data fetch completed")

            logging.info("Train Test split begins")
            train_data, test_data = train_test_split(
                data, test_size=0.2, random_state=42
            )
            logging.info("Train Test split completed")

            os.makedirs(
                os.path.dirname(self.data_ingestion_config.train_data_path),
                exist_ok=True,
            )

            logging.info("Storing entire,train,test data begins")
            data.to_csv(
                self.data_ingestion_config.raw_data_path, index=False, header=True
            )  # saving raw data

            train_data.to_csv(
                self.data_ingestion_config.train_data_path, index=False, header=True
            )
            test_data.to_csv(
                self.data_ingestion_config.test_data_path, index=False, header=True
            )
            logging.info("Storing entire,train,test data completed")
            logging.info("Data ingestion completed")

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
