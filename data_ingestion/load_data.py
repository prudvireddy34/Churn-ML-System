from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load churn dataset from CSV with basic validations."""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("Loaded dataframe is empty")

    # Basic sanity: must have target column
    if "Churn" not in df.columns:
        raise ValueError("Expected column 'Churn' not found in dataset")

    return df


if __name__ == "__main__":
    df = load_data("data/raw_churn.csv")
    print("✅ Data loaded successfully")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print(df.head(3))
