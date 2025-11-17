
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from .preprocessing import build_preprocessor
from .utils import safe_read_csv

logger = logging.getLogger('model_classification')

def run_classification(path='salary_prepared_with_targets.csv'):
    try:
        df = safe_read_csv(path)
        drop_cols = ['salary','salary_class','salary_numeric_proxy']
        preprocessor, num_features, cat_features = build_preprocessor(df, drop_cols)
        X = df[[c for c in df.columns if c not in drop_cols]]
        y = df['salary_class']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_proc, y_train)
        preds = model.predict(X_test_proc)
        prob = model.predict_proba(X_test_proc)[:,1]

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, prob)

        logger.info("Classification results - Acc: %.4f, Prec: %.4f, Rec: %.4f, F1: %.4f, AUC: %.4f", acc, prec, rec, f1, auc)
        print("Logistic Regression: Acc={:.4f}, Prec={:.4f}, Rec={:.4f}, F1={:.4f}, AUC={:.4f}".format(acc, prec, rec, f1, auc))
    except Exception as e:
        logger.exception("Classification pipeline failed: %s", e)
        raise
