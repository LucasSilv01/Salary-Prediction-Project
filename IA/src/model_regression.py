
import logging
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from .preprocessing import build_preprocessor
from .utils import safe_read_csv

logger = logging.getLogger('model_regression')

def run_regression(path='salary_prepared_with_targets.csv'):
    try:
        df = safe_read_csv(path)
        drop_cols = ['salary','salary_class','salary_numeric_proxy']
        preprocessor, num_features, cat_features = build_preprocessor(df, drop_cols)
        X = df[[c for c in df.columns if c not in drop_cols]]
        y = df['salary_numeric_proxy']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # fit preprocessor and transform
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_proc, y_train)
        preds = model.predict(X_test_proc)

        mae = mean_absolute_error(y_test, preds)
        try:
            rmse = mean_squared_error(y_test, preds, squared=False)
        except TypeError:
            # older sklearn versions may not support `squared` keyword
            rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        logger.info("Regression results - MAE: %.2f, RMSE: %.2f, R2: %.4f", mae, rmse, r2)
        print("Linear Regression: MAE={:.2f}, RMSE={:.2f}, R2={:.4f}".format(mae, rmse, r2))
    except Exception as e:
        logger.exception("Regression pipeline failed: %s", e)
        raise
