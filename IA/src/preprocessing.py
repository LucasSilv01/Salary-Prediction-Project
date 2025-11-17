
import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd

logger = logging.getLogger('preprocessing')

def build_preprocessor(df, drop_cols):
    X = df.drop(columns=drop_cols)
    num_features = X.select_dtypes(include=['number']).columns.tolist()
    cat_features = X.select_dtypes(exclude=['number']).columns.tolist()

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])
    logger.info("Preprocessor built with %d numeric and %d categorical features", len(num_features), len(cat_features))
    return preprocessor, num_features, cat_features
