from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from dataclasses import dataclass


@dataclass
class TrainPipeline:
    def __init__(self):
        self.ingest = DataIngestion()
        self.transform = DataTransformation()
        self.train = ModelTrainer()

    def initiate_train_pipeline(self):
        train_data_path, test_data_path = self.ingest.initiate_ingestion()
        train_arr, test_arr, preprocessor_path = (
            self.transform.initiate_data_transformation(
                train_path=train_data_path, test_path=test_data_path
            )
        )
        score = self.train.initiate_model_trainer(
            train_arr=train_arr, test_arr=test_arr
        )
        return score


# if __name__ == "__main__":
#     obj = TrainPipeline()
#     print(obj.initiate_train_pipeline())
