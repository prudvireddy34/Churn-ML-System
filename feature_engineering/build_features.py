from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw Telco churn dataset:
    - converts TotalCharges to numeric
    - fills missing values
    - drops customerID
    - maps target to 0/1
    """
    df = df.copy()

    # Convert TotalCharges to numeric; blank strings become NaN
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df = df.fillna(0)

    # Drop identifier if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Map target to 0/1
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    if df["Churn"].isna().any():
        raise ValueError("Target mapping failed: unexpected Churn values found")

    return df


def split_xy(df: pd.DataFrame, target: str = "Churn"):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def train_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


if __name__ == "__main__":
    # quick local sanity check (will be used later from pipeline)
    raw = pd.read_csv("data/raw_churn.csv")
    cleaned = clean_raw(raw)
    X, y = split_xy(cleaned)
    X_train, X_val, y_train, y_val = train_val_split(X, y)
    print("✅ Feature engineering complete")
    print("Train shape:", X_train.shape, "Val shape:", X_val.shape)
