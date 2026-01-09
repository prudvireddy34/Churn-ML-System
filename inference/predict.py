from __future__ import annotations

from pathlib import Path
import argparse
import json
import joblib
import pandas as pd


def predict(
    model_path: str,
    input_csv: str,
    output_csv: str,
    threshold: float = 0.5,
) -> dict:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError("Input CSV is empty")

    model = joblib.load(model_path)

    # Model is a sklearn Pipeline; it can handle raw columns as trained
    proba = model.predict_proba(df)[:, 1]
    preds = (proba >= threshold).astype(int)

    out = df.copy()
    out["churn_probability"] = proba
    out["churn_prediction"] = preds
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    summary = {
        "input_rows": int(df.shape[0]),
        "threshold": float(threshold),
        "predicted_churn_rate": float(preds.mean()),
        "avg_churn_probability": float(proba.mean()),
        "model_path": str(model_path),
        "output_csv": str(output_csv),
    }
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="artifacts/logistic_model.pkl")
    parser.add_argument("--input", default="data/raw_churn.csv")
    parser.add_argument("--output", default="artifacts/predictions.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    summary = predict(args.model, args.input, args.output, args.threshold)
    print("✅ Inference completed")
    print(json.dumps(summary, indent=2))
