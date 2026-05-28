"""Train and compare credit card fraud detection models.

The project expects the Kaggle credit card fraud dataset as creditcard.csv.
By default, place it at data/raw/creditcard.csv.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_CANDIDATES = [
    RAW_DATA_DIR / "kaggel-creditcard.csv",
    RAW_DATA_DIR / "kaggle-creditcard.csv",
    RAW_DATA_DIR / "creditcard.csv",
]
DEFAULT_DATASET = DATASET_CANDIDATES[0]
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
TARGET_COLUMN = "Class"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare credit card fraud detection models."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to a Kaggle-format credit card CSV. Default: auto-detect in data/raw.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Test split size. Default: 0.30",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    return parser.parse_args()


def resolve_dataset_path(path: Path | None = None) -> Path:
    if path is not None:
        return path

    for candidate in DATASET_CANDIDATES:
        if candidate.exists():
            return candidate

    return DEFAULT_DATASET


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n\n"
            "Download the Kaggle credit card fraud dataset and save it as one of:\n"
            + "\n".join(f"  {candidate}" for candidate in DATASET_CANDIDATES)
        )

    df = pd.read_csv(path)
    required_columns = {"Time", "Amount", TARGET_COLUMN}
    required_columns.update({f"V{i}" for i in range(1, 29)})
    missing_columns = sorted(required_columns.difference(df.columns))

    if missing_columns:
        raise ValueError(
            "The dataset does not match the expected Kaggle schema. "
            f"Missing columns: {missing_columns}"
        )

    return df


def dataset_profile(df: pd.DataFrame, source_path: Path) -> dict[str, object]:
    class_counts = df[TARGET_COLUMN].value_counts().to_dict()
    fraud_count = int(class_counts.get(1, 0))
    normal_count = int(class_counts.get(0, 0))

    return {
        "source": str(source_path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "normal_transactions": normal_count,
        "fraud_transactions": fraud_count,
        "fraud_rate": fraud_count / len(df) if len(df) else 0,
    }


def create_data_quality_report(df: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include="number")
    quality = pd.DataFrame(
        {
            "Column": df.columns,
            "Missing_Values": df.isna().sum().values,
            "Missing_Rate": df.isna().mean().values,
            "Unique_Values": df.nunique(dropna=True).values,
        }
    )

    if not numeric_df.empty:
        stats = numeric_df.describe().transpose().reset_index()
        stats = stats.rename(columns={"index": "Column"})
        quality = quality.merge(stats, on="Column", how="left")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    quality.to_csv(REPORTS_DIR / "data_quality_report.csv", index=False)

    profile = dataset_profile(df, source_path)
    with (REPORTS_DIR / "dataset_profile.json").open("w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2)

    return quality


def balance_dataset(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    fraud = df[df[TARGET_COLUMN] == 1]
    normal = df[df[TARGET_COLUMN] == 0]

    if fraud.empty or normal.empty:
        raise ValueError("Dataset must contain both normal and fraud transactions.")

    normal_sample = resample(
        normal,
        replace=False,
        n_samples=len(fraud),
        random_state=random_state,
    )

    return (
        pd.concat([fraud, normal_sample])
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )


def build_models(random_state: int) -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", DecisionTreeClassifier(random_state=random_state)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestClassifier(random_state=random_state)),
            ]
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier()),
            ]
        ),
    }


def save_confusion_matrix(
    model_name: str,
    y_test: pd.Series,
    y_pred: pd.Series,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Normal", "Fraud"],
        cmap="Blues",
        values_format="d",
    )
    display.ax_.set_title(model_name)
    output_path = FIGURES_DIR / f"{slugify(model_name)}_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def slugify(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def extract_feature_importance(model: Pipeline, model_name: str, columns: list[str]) -> pd.DataFrame | None:
    estimator = model.named_steps["model"]
    values = None

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = abs(estimator.coef_[0])

    if values is None:
        return None

    importance = pd.DataFrame(
        {
            "Model": model_name,
            "Feature": columns,
            "Importance": values,
        }
    ).sort_values("Importance", ascending=False)
    return importance


def save_feature_importance_plot(importance: pd.DataFrame, model_name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    top_features = importance.head(12).sort_values("Importance")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=top_features, x="Importance", y="Feature", color="#2f81f7")
    plt.title(f"Top Features: {model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{slugify(model_name)}_feature_importance.png", dpi=160)
    plt.close()


def save_amount_by_class_plot(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    sns.boxplot(data=df, x=TARGET_COLUMN, y="Amount")
    plt.yscale("symlog")
    plt.title("Transaction Amount by Class")
    plt.xlabel("Class: 0 = Normal, 1 = Fraud")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "amount_by_class.png", dpi=160)
    plt.close()


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    source_path: Path | None = None,
) -> pd.DataFrame:
    balanced_df = balance_dataset(df, random_state=random_state)
    X = balanced_df.drop(columns=[TARGET_COLUMN])
    y = balanced_df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    results = []
    trained_models = {}
    feature_importance_frames = []
    feature_columns = list(X.columns)

    for model_name, model in build_models(random_state).items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred)

        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision_Fraud": report["1"]["precision"],
                "Recall_Fraud": report["1"]["recall"],
                "F1_Fraud": report["1"]["f1-score"],
                "True_Normal": matrix[0][0],
                "False_Fraud_Alert": matrix[0][1],
                "Missed_Fraud": matrix[1][0],
                "True_Fraud": matrix[1][1],
            }
        )
        trained_models[model_name] = model
        save_confusion_matrix(model_name, y_test, y_pred)
        importance = extract_feature_importance(model, model_name, feature_columns)
        if importance is not None:
            feature_importance_frames.append(importance)
            save_feature_importance_plot(importance, model_name)

    summary = pd.DataFrame(results).sort_values(
        by=["F1_Fraud", "Recall_Fraud"],
        ascending=False,
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(REPORTS_DIR / "model_results_summary.csv", index=False)

    if feature_importance_frames:
        feature_importance = pd.concat(feature_importance_frames, ignore_index=True)
        feature_importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)

    best_model_name = summary.iloc[0]["Model"]
    joblib.dump(trained_models[best_model_name], MODELS_DIR / "best_model.joblib")

    metadata = {
        "dataset": dataset_profile(df, source_path) if source_path else None,
        "balanced_rows_used_for_training": int(len(balanced_df)),
        "test_size": test_size,
        "random_state": random_state,
        "best_model": best_model_name,
    }
    with (REPORTS_DIR / "training_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return summary


def save_class_distribution_plot(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.countplot(data=df, x=TARGET_COLUMN)
    plt.title("Class Distribution: 0 = Normal, 1 = Fraud")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=160)
    plt.close()


def save_eda_outputs(df: pd.DataFrame, source_path: Path) -> None:
    create_data_quality_report(df, source_path)
    save_class_distribution_plot(df)
    save_amount_by_class_plot(df)


def main() -> None:
    args = parse_args()
    dataset_path = resolve_dataset_path(args.data)
    df = load_dataset(dataset_path)
    save_eda_outputs(df, dataset_path)
    summary = train_and_evaluate(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        source_path=dataset_path,
    )

    profile = dataset_profile(df, dataset_path)
    print(f"\nDataset: {profile['source']}")
    print(
        "Rows: "
        f"{profile['rows']:,} | Normal: {profile['normal_transactions']:,} | "
        f"Fraud: {profile['fraud_transactions']:,} ({profile['fraud_rate']:.3%})"
    )
    print("\nModel comparison:")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(f"- {REPORTS_DIR / 'model_results_summary.csv'}")
    print(f"- {MODELS_DIR / 'best_model.joblib'}")
    print(f"- {FIGURES_DIR}")


if __name__ == "__main__":
    main()
