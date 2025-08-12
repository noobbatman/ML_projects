import logging
import os
from datetime import datetime

# Define the log file name with timestamp
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Define the logs directory path (logs directory only, without the file name)
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)  # Create the 'logs' directory if it doesn't exist

# Define the full log file path (logs directory + the log file)
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Set up logging configuration
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='%(asctime)s %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,  # INFO level captures INFO, WARNING, ERROR, CRITICAL
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_logger():
    # Return the logger instance
    return logging.getLogger()

