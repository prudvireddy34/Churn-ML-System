from __future__ import annotations

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from data_ingestion.load_data import load_data
from feature_engineering.build_features import clean_raw, split_xy, train_val_split


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    model = LogisticRegression(max_iter=1000)

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    return clf


def train_and_save(model_path: str = "artifacts/logistic_model.pkl") -> float:
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "raw_churn.csv"
    
    df_raw = load_data(csv_path)
    df = clean_raw(df_raw)

    X, y = split_xy(df)
    X_train, X_val, y_train, y_val = train_val_split(X, y)

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    return auc


if __name__ == "__main__":
    # Get the project root for model path
    project_root = Path(__file__).parent.parent
    model_path = project_root / "artifacts" / "logistic_model.pkl"
    
    auc = train_and_save(str(model_path))
    print(f"✅ Logistic Regression trained and saved. ROC-AUC={auc:.4f}")
