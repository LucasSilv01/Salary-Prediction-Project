
import logging
from src.utils import configure_logging
from src.eda import run_eda
from src.model_regression import run_regression
from src.model_classification import run_classification

def main():
    configure_logging()
    logger = logging.getLogger('main')
    logger.info("Starting Salary Project")
    try:
        run_eda()
        run_regression()
        run_classification()
    except Exception as e:
        logger.exception("Project failed: %s", e)
        print("An error occurred. Check logs for details.")

if __name__ == '__main__':
    main()
