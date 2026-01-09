from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from data_ingestion.load_data import load_data
from feature_engineering.build_features import clean_raw, split_xy


MODEL_PATH = "artifacts/logistic_model.pkl"
OUTPUT_PATH = "artifacts/churn_predictions.csv"


def run_inference():
    # Load trained model
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError("Model not found. Train the model first.")

    model = joblib.load(MODEL_PATH)

    # Load and prepare data
    df_raw = load_data("data/raw_churn.csv")
    df_clean = clean_raw(df_raw)

    X, _ = split_xy(df_clean)

    # Predict churn probabilities
    churn_probs = model.predict_proba(X)[:, 1]

    results = pd.DataFrame({
        "churn_probability": churn_probs
    })

    results.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Inference complete. Predictions saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_inference()
