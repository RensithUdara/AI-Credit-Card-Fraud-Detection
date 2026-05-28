"""Run fraud predictions with a trained model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "best_model.joblib"
TARGET_COLUMN = "Class"
FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict fraud labels for a CSV file.")
    parser.add_argument("input_csv", type=Path, help="CSV containing transaction rows.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to trained .joblib model. Default: models/best_model.joblib",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "predictions.csv",
        help="Prediction output path. Default: reports/predictions.csv",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Fraud probability threshold when predict_proba is available. Default: 0.50",
    )
    return parser.parse_args()


def predict_transactions(
    model,
    df: pd.DataFrame,
    threshold: float = 0.50,
) -> pd.DataFrame:
    missing_columns = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    features = df[FEATURE_COLUMNS]
    output = df.copy()

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[:, 1]
        output["Fraud_Probability"] = probabilities
        output["Predicted_Class"] = (probabilities >= threshold).astype(int)
    else:
        output["Predicted_Class"] = model.predict(features)

    return output


def main() -> None:
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model not found: {args.model}\n"
            "Run training first: python src/fraud_detection_pipeline.py"
        )

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    model = joblib.load(args.model)
    df = pd.read_csv(args.input_csv)
    output = predict_transactions(
        model,
        df.drop(columns=[TARGET_COLUMN], errors="ignore"),
        threshold=args.threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")


if __name__ == "__main__":
    main()
