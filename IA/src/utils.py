
import logging
import pandas as pd
import os

def configure_logging(level=logging.INFO, logfile=None):
    handlers = [logging.StreamHandler()]
    if logfile:
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        handlers=handlers
    )

def safe_read_csv(path):
    logger = logging.getLogger('utils.safe_read_csv')
    if not os.path.exists(path):
        logger.error("File not found: %s", path)
        raise FileNotFoundError(f"File not found: {path}")
    try:
        df = pd.read_csv(path)
        logger.info("Loaded CSV: %s (shape=%s)", path, df.shape)
        return df
    except Exception as e:
        logger.exception("Error reading CSV %s: %s", path, e)
        raise
