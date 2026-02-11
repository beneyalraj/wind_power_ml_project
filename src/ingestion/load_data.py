import boto3
import yaml
import logging
import os
import time
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

os.makedirs("logs", exist_ok=True)

# 1. Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/ingestion.log"), 
        logging.StreamHandler()
        ]
)
logger = logging.getLogger("Ingestion")

def load_config():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def download_with_retries():
    config = load_config()
    s3_path = config['data_ingestion']['s3_url']
    local_path = config['data_ingestion']['local_path']
    retries = config['data_ingestion']['retries']
    
    # Parse S3
    path_parts = s3_path.replace("s3://", "").split("/")
    bucket = path_parts[0]
    key = "/".join(path_parts[1:])

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Attempt {attempt}: Downloading {key}...")
            s3.download_file(bucket, key, local_path)
            logger.info("Successfully downloaded data.")
            return True
        except Exception as e:
            logger.error(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(5) # Wait before retry
            else:
                logger.critical("Maximum retries reached. Ingestion failed.")
                raise

if __name__ == "__main__":
    download_with_retries()