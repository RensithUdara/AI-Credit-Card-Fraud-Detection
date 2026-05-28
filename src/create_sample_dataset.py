"""Create a small synthetic dataset for local testing.

This is not the real Kaggle dataset. It only matches the same column structure
so the training and prediction pipeline can be tested without downloading data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
RANDOM_STATE = 42


def create_sample_dataset(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    fraud_rows = rows // 4
    normal_rows = rows - fraud_rows

    normal = pd.DataFrame(
        rng.normal(loc=0.0, scale=1.0, size=(normal_rows, 28)),
        columns=[f"V{i}" for i in range(1, 29)],
    )
    normal["Time"] = np.arange(normal_rows)
    normal["Amount"] = rng.gamma(shape=2.0, scale=30.0, size=normal_rows)
    normal["Class"] = 0

    fraud = pd.DataFrame(
        rng.normal(loc=1.8, scale=1.2, size=(fraud_rows, 28)),
        columns=[f"V{i}" for i in range(1, 29)],
    )
    fraud["Time"] = np.arange(normal_rows, rows)
    fraud["Amount"] = rng.gamma(shape=4.0, scale=55.0, size=fraud_rows)
    fraud["Class"] = 1

    df = pd.concat([normal, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    ordered_columns = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount", "Class"]
    return df[ordered_columns]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = create_sample_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Sample dataset created: {OUTPUT_PATH}")
    print("This file is synthetic demo data, not the real Kaggle dataset.")


if __name__ == "__main__":
    main()
