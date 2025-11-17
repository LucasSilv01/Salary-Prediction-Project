import logging
import matplotlib.pyplot as plt
from .utils import safe_read_csv

logger = logging.getLogger('eda')

def run_eda(path='salary_prepared_with_targets.csv'):
    df = safe_read_csv(path)
    logger.info("Starting EDA")
    print('\n--- Head ---\n', df.head())
    print('\n--- Describe ---\n', df.describe(include='all'))
    # Simple plots
    try:
        plt.figure(figsize=(8,4))
        plt.hist(df['salary_numeric_proxy'].dropna(), bins=25)
        plt.title('Histogram of salary_numeric_proxy')
        plt.xlabel('salary_numeric_proxy')
        plt.ylabel('Count')
        plt.show()
    except Exception as e:
        logger.exception("Error plotting histogram: %s", e)