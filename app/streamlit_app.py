"""Streamlit UI for the credit card fraud detection project."""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fraud_detection_pipeline import (  # noqa: E402
    DEFAULT_DATASET,
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    TARGET_COLUMN,
    load_dataset,
    save_class_distribution_plot,
    train_and_evaluate,
)


st.set_page_config(
    page_title="Fraud Detection",
    page_icon="",
    layout="wide",
)


BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
SUMMARY_PATH = REPORTS_DIR / "model_results_summary.csv"
PREDICTION_OUTPUT = REPORTS_DIR / "ui_predictions.csv"
FEATURE_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        h1 {
            margin-bottom: 0.25rem;
        }
        h2, h3 {
            margin-top: 1.25rem;
        }
        .st-emotion-cache-gi0tri,
        .st-emotion-cache-1vbkxwb {
            display: none;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
            margin: 0.85rem 0 1.25rem;
        }
        .metric-grid.three {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }
        .metric-card {
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 1rem 1rem 0.95rem;
            background: #161b22;
            min-height: 104px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .metric-label {
            color: #f8fafc;
            font-size: 0.82rem;
            font-weight: 700;
            line-height: 1.1;
            margin-bottom: 0.65rem;
        }
        .metric-value {
            color: #f8fafc;
            font-size: 2rem;
            line-height: 1;
            font-weight: 500;
        }
        .metric-value.compact {
            font-size: 1.65rem;
            line-height: 1.08;
        }
        .metric-delta {
            color: #22c55e;
            font-size: 0.9rem;
            line-height: 1;
            margin-top: 0.65rem;
        }
        .status-line {
            border-left: 4px solid #2563eb;
            padding: 0.65rem 0.9rem;
            background: #111827;
            color: #f8fafc;
            margin: 1rem 0 1rem;
            font-weight: 700;
        }
        .control-button-spacer {
            height: 1.75rem;
        }
        div[data-testid="stExpander"] {
            margin-bottom: 0.75rem;
        }
        @media (max-width: 900px) {
            .metric-grid,
            .metric-grid.three {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        @media (max-width: 560px) {
            .metric-grid,
            .metric-grid.three {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_grid(items: list[tuple[str, str, str | None]], columns: int = 4) -> None:
    grid_class = "metric-grid three" if columns == 3 else "metric-grid"
    cards = []
    for label, value, delta in items:
        delta_html = f'<div class="metric-delta">+ {escape(delta)}</div>' if delta else ""
        value_class = "metric-value compact" if len(value) > 10 else "metric-value"
        cards.append(
            '<div class="metric-card">'
            f'<div class="metric-label">{escape(label)}</div>'
            f'<div class="{value_class}">{escape(value)}</div>'
            f"{delta_html}"
            "</div>"
        )
    st.markdown(f'<div class="{grid_class}">{"".join(cards)}</div>', unsafe_allow_html=True)


def load_summary() -> pd.DataFrame | None:
    if SUMMARY_PATH.exists():
        return pd.read_csv(SUMMARY_PATH)
    return None


def dataset_status() -> pd.DataFrame | None:
    try:
        return load_dataset(DEFAULT_DATASET)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return None


def show_dataset_overview(df: pd.DataFrame) -> None:
    normal_count = int((df[TARGET_COLUMN] == 0).sum())
    fraud_count = int((df[TARGET_COLUMN] == 1).sum())
    fraud_rate = fraud_count / len(df) if len(df) else 0

    render_metric_grid(
        [
            ("Rows", f"{len(df):,}", None),
            ("Features", f"{len(FEATURE_COLUMNS)}", None),
            ("Normal", f"{normal_count:,}", None),
            ("Fraud", f"{fraud_count:,}", f"{fraud_rate:.2%}"),
        ]
    )


def train_view() -> None:
    st.subheader("Training")
    df = dataset_status()
    if df is None:
        return

    show_dataset_overview(df)

    with st.expander("Preview dataset", expanded=False):
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    control_cols = st.columns([1, 1, 2])
    test_size = control_cols[0].slider("Test size", 0.10, 0.50, 0.30, 0.05)
    random_state = control_cols[1].number_input("Random seed", 0, 9999, 42, 1)
    control_cols[2].markdown('<div class="control-button-spacer"></div>', unsafe_allow_html=True)

    if control_cols[2].button("Train Models", type="primary", use_container_width=True):
        with st.spinner("Training and comparing models..."):
            save_class_distribution_plot(df)
            summary = train_and_evaluate(
                df,
                test_size=float(test_size),
                random_state=int(random_state),
            )
        st.success("Training complete")
        st.dataframe(summary, use_container_width=True, hide_index=True)

    summary = load_summary()
    if summary is not None:
        st.markdown('<div class="status-line">Latest saved model results</div>', unsafe_allow_html=True)
        st.dataframe(summary, use_container_width=True, hide_index=True)


def report_view() -> None:
    st.subheader("Reports")
    summary = load_summary()
    if summary is None:
        st.info("Train the models first to generate reports.")
        return

    best = summary.iloc[0]
    render_metric_grid(
        [
            ("Best Model", str(best["Model"]), None),
            ("Accuracy", f"{best['Accuracy']:.3f}", None),
            ("Fraud Recall", f"{best['Recall_Fraud']:.3f}", None),
            ("Fraud F1", f"{best['F1_Fraud']:.3f}", None),
        ]
    )

    st.dataframe(summary, use_container_width=True, hide_index=True)

    figure_paths = [
        FIGURES_DIR / "class_distribution.png",
        FIGURES_DIR / "logistic_regression_confusion_matrix.png",
        FIGURES_DIR / "decision_tree_confusion_matrix.png",
        FIGURES_DIR / "random_forest_confusion_matrix.png",
        FIGURES_DIR / "knn_confusion_matrix.png",
    ]
    existing_figures = [path for path in figure_paths if path.exists()]
    if existing_figures:
        image_cols = st.columns(2)
        for index, image_path in enumerate(existing_figures):
            image_cols[index % 2].image(
                image_path.read_bytes(),
                caption=image_path.stem.replace("_", " ").title(),
                use_column_width=True,
            )


def prediction_view() -> None:
    st.subheader("Prediction")

    if not BEST_MODEL_PATH.exists():
        st.warning("Train the models first so the app can load models/best_model.joblib.")
        return

    uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV with Time, V1-V28, and Amount columns.")
        return

    input_df = pd.read_csv(uploaded_file)
    missing_columns = [column for column in FEATURE_COLUMNS if column not in input_df.columns]
    if missing_columns:
        st.error(f"Missing required columns: {missing_columns}")
        return

    model = joblib.load(BEST_MODEL_PATH)
    features = input_df[FEATURE_COLUMNS]
    predictions = model.predict(features)

    output = input_df.copy()
    output["Predicted_Class"] = predictions
    if hasattr(model, "predict_proba"):
        output["Fraud_Probability"] = model.predict_proba(features)[:, 1]

    fraud_count = int((output["Predicted_Class"] == 1).sum())
    render_metric_grid(
        [
            ("Rows Scored", f"{len(output):,}", None),
            ("Predicted Fraud", f"{fraud_count:,}", None),
            ("Predicted Normal", f"{len(output) - fraud_count:,}", None),
        ],
        columns=3,
    )

    st.dataframe(output, use_container_width=True, hide_index=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output.to_csv(PREDICTION_OUTPUT, index=False)
    st.download_button(
        "Download Predictions",
        data=output.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


def main() -> None:
    inject_styles()
    st.title("Credit Card Fraud Detection")
    st.caption("Train, compare, and use fraud detection models.")

    tabs = st.tabs(["Train", "Reports", "Predict"])
    with tabs[0]:
        train_view()
    with tabs[1]:
        report_view()
    with tabs[2]:
        prediction_view()


if __name__ == "__main__":
    main()
