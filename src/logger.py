import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(
    # os.getcwd(), "logs", LOG_FILE
    os.getcwd(),
    "logs",
)  # path of the folder inside which log files are present
os.makedirs(logs_path, exist_ok=True)

# log file path = path of the log file inside the folder
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,  # path where to store logs
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # format of the logs
    level=logging.INFO,
)
# whenever  logging.info(<msg>) is used , the above format is used to store logs in the specified path
