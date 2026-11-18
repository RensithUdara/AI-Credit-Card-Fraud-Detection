"""Streamlit UI for the credit card fraud detection project."""

from __future__ import annotations

import base64
import sys
import json
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
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    TARGET_COLUMN,
    dataset_profile,
    load_dataset,
    resolve_dataset_path,
    save_eda_outputs,
    train_and_evaluate,
)
from predict import FEATURE_COLUMNS, predict_transactions  # noqa: E402


st.set_page_config(
    page_title="Fraud Detection",
    page_icon=".",
    layout="wide",
)


BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
SUMMARY_PATH = REPORTS_DIR / "model_results_summary.csv"
METADATA_PATH = REPORTS_DIR / "training_metadata.json"
PREDICTION_OUTPUT = REPORTS_DIR / "ui_predictions.csv"
DATA_QUALITY_PATH = REPORTS_DIR / "data_quality_report.csv"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #0d1117;
            --panel: #151a23;
            --panel-soft: #111827;
            --panel-strong: #1a2230;
            --line: #2d3748;
            --text: #f8fafc;
            --muted: #aab6c5;
            --accent: #ff4b55;
            --accent-blue: #2f81f7;
            --good: #22c55e;
            --warn: #f59e0b;
        }
        .stApp {
            background:
                radial-gradient(circle at 14% 0%, rgba(47, 129, 247, 0.12), transparent 26rem),
                linear-gradient(180deg, #0d1117 0%, #0a0f17 100%);
        }
        header[data-testid="stHeader"] {
            background: transparent;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }
        .app-header {
            background: linear-gradient(135deg, rgba(21, 26, 35, 0.94), rgba(17, 24, 39, 0.76));
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 8px;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.22);
            margin-bottom: 1rem;
            padding: 1.15rem 1.25rem;
        }
        .app-title-row {
            align-items: flex-end;
            display: flex;
            gap: 1rem;
            justify-content: space-between;
        }
        .app-title {
            color: var(--text);
            font-size: clamp(2rem, 4vw, 3rem);
            font-weight: 800;
            letter-spacing: 0;
            line-height: 1.02;
            margin: 0;
        }
        .app-subtitle {
            color: var(--muted);
            font-size: 0.98rem;
            margin-top: 0.65rem;
        }
        .header-meta {
            color: var(--muted);
            font-size: 0.78rem;
            font-weight: 700;
            margin-top: 0.45rem;
        }
        .status-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            justify-content: flex-end;
        }
        .status-pill {
            background: var(--panel-soft);
            border: 1px solid var(--line);
            border-radius: 999px;
            color: var(--text);
            font-size: 0.78rem;
            font-weight: 800;
            padding: 0.45rem 0.7rem;
            white-space: nowrap;
        }
        .status-pill.ready {
            border-color: rgba(34, 197, 94, 0.5);
            color: #86efac;
        }
        .status-pill.warn {
            border-color: rgba(255, 75, 85, 0.5);
            color: #fca5a5;
        }
        .section-title {
            align-items: center;
            display: flex;
            gap: 1rem;
            justify-content: space-between;
            margin: 1.2rem 0 0.75rem;
        }
        .section-title h2 {
            color: var(--text);
            font-size: 1.55rem;
            letter-spacing: 0;
            line-height: 1.1;
            margin: 0;
        }
        .section-note {
            color: var(--muted);
            font-size: 0.88rem;
            text-align: right;
        }
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(17, 24, 39, 0.55);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 8px;
            gap: 0.25rem;
            padding: 0.25rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px;
            color: var(--text);
            height: 2.55rem;
            padding: 0 1rem;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(255, 75, 85, 0.16);
            color: #ffffff;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }
        .metric-grid {
            display: grid;
            gap: 0.9rem;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            margin: 0.85rem 0 1.05rem;
        }
        .metric-grid.three {
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }
        .metric-card {
            background: linear-gradient(180deg, #17202c 0%, #121923 100%);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 8px;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.16);
            display: flex;
            flex-direction: column;
            justify-content: center;
            min-height: 104px;
            padding: 1rem;
        }
        .metric-card:hover {
            border-color: rgba(47, 129, 247, 0.45);
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.82rem;
            font-weight: 800;
            line-height: 1.1;
            margin-bottom: 0.65rem;
        }
        .metric-value {
            color: var(--text);
            font-size: 2rem;
            font-weight: 500;
            line-height: 1;
        }
        .metric-value.compact {
            font-size: 1.55rem;
            line-height: 1.1;
        }
        .metric-delta {
            color: var(--good);
            font-size: 0.9rem;
            line-height: 1;
            margin-top: 0.65rem;
        }
        .status-line {
            background: var(--panel-soft);
            border-left: 4px solid var(--accent-blue);
            border-radius: 0 8px 8px 0;
            color: var(--text);
            font-weight: 800;
            margin: 1rem 0;
            padding: 0.75rem 0.9rem;
        }
        .control-button-spacer {
            height: 1.78rem;
        }
        .info-card {
            background: linear-gradient(135deg, rgba(21, 26, 35, 0.96), rgba(17, 24, 39, 0.9));
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 8px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
            color: var(--text);
            margin: 0.75rem 0 1rem;
            padding: 1rem;
        }
        .info-card-title {
            color: var(--text);
            font-size: 0.92rem;
            font-weight: 800;
            margin-bottom: 0.45rem;
        }
        .info-card-body {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.45;
        }
        .info-card-body strong {
            color: var(--text);
        }
        div[data-testid="stExpander"] {
            background: rgba(17, 24, 39, 0.45);
            border-color: var(--line);
            margin-bottom: 0.75rem;
        }
        div[data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
        }
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 8px;
            font-weight: 800;
            min-height: 2.75rem;
        }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #ff4b55, #e83f49);
            border: 0;
        }
        .figure-title {
            color: var(--text);
            font-size: 0.9rem;
            font-weight: 800;
            margin: 0.6rem 0 0.65rem;
        }
        .figure-frame {
            background: var(--panel);
            border: 1px solid rgba(148, 163, 184, 0.14);
            border-radius: 8px;
            margin-bottom: 1rem;
            padding: 0.85rem;
        }
        .result-normal {
            background: rgba(34, 197, 94, 0.12);
            border: 1px solid rgba(34, 197, 94, 0.35);
            border-radius: 8px;
            color: #bbf7d0;
            font-weight: 800;
            margin-top: 0.75rem;
            padding: 0.85rem 1rem;
        }
        .result-fraud {
            background: rgba(255, 75, 85, 0.14);
            border: 1px solid rgba(255, 75, 85, 0.38);
            border-radius: 8px;
            color: #fecaca;
            font-weight: 800;
            margin-top: 0.75rem;
            padding: 0.85rem 1rem;
        }
        @media (max-width: 900px) {
            .app-title-row {
                align-items: flex-start;
                flex-direction: column;
            }
            .status-pills {
                justify-content: flex-start;
            }
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


def render_header() -> None:
    active_dataset = resolve_dataset_path()
    dataset_ready = active_dataset.exists()
    model_ready = BEST_MODEL_PATH.exists()
    dataset_class = "ready" if dataset_ready else "warn"
    model_class = "ready" if model_ready else "warn"
    dataset_text = "Dataset ready" if dataset_ready else "Dataset missing"
    model_text = "Model ready" if model_ready else "Model missing"

    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-title-row">
                <div>
                    <h1 class="app-title">Credit Card Fraud Detection</h1>
                    <div class="app-subtitle">Model training, evaluation, and transaction scoring</div>
                    <div class="header-meta">{escape(active_dataset.name)}</div>
                </div>
                <div class="status-pills">
                    <div class="status-pill {dataset_class}">{dataset_text}</div>
                    <div class="status-pill {model_class}">{model_text}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-card-title">{escape(title)}</div>
            <div class="info-card-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(prediction: str, probability: float | None) -> None:
    result_class = "result-fraud" if prediction == "Fraud" else "result-normal"
    probability_text = "" if probability is None else f" | Fraud probability: {probability:.3f}"
    st.markdown(
        f'<div class="{result_class}">Prediction: {escape(prediction)}{probability_text}</div>',
        unsafe_allow_html=True,
    )


def render_image_card(image_path: Path) -> None:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    title = image_path.stem.replace("_", " ").title()
    st.markdown(
        f"""
        <div class="figure-frame">
            <div class="figure-title">{escape(title)}</div>
            <img src="data:image/png;base64,{encoded}" style="width:100%; border-radius:6px;" />
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section(title: str, note: str | None = None) -> None:
    note_html = f'<div class="section-note">{escape(note)}</div>' if note else ""
    st.markdown(
        f"""
        <div class="section-title">
            <h2>{escape(title)}</h2>
            {note_html}
        </div>
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


def format_summary(summary: pd.DataFrame) -> pd.DataFrame:
    display = summary.copy()
    for column in ["Accuracy", "Precision_Fraud", "Recall_Fraud", "F1_Fraud"]:
        if column in display:
            display[column] = display[column].map(lambda value: f"{value:.3f}")
    return display


def load_summary() -> pd.DataFrame | None:
    if SUMMARY_PATH.exists():
        return pd.read_csv(SUMMARY_PATH)
    return None


def load_metadata() -> dict[str, object] | None:
    if not METADATA_PATH.exists():
        return None

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_data_quality() -> pd.DataFrame | None:
    if DATA_QUALITY_PATH.exists():
        return pd.read_csv(DATA_QUALITY_PATH)
    return None


def load_feature_importance() -> pd.DataFrame | None:
    if FEATURE_IMPORTANCE_PATH.exists():
        return pd.read_csv(FEATURE_IMPORTANCE_PATH)
    return None


def dataset_status() -> pd.DataFrame | None:
    try:
        return load_dataset(resolve_dataset_path())
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
    active_dataset = resolve_dataset_path()
    render_section("Training", str(active_dataset.relative_to(PROJECT_ROOT)))
    df = dataset_status()
    if df is None:
        return

    show_dataset_overview(df)
    profile = dataset_profile(df, active_dataset)
    render_info_card(
        "Active Dataset",
        "Using <strong>"
        f"{escape(Path(profile['source']).name)}</strong> with "
        f"<strong>{profile['rows']:,}</strong> rows and "
        f"<strong>{profile['fraud_transactions']:,}</strong> fraud transactions.",
    )

    with st.expander("Dataset preview", expanded=False):
        st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    control_cols = st.columns([1, 1, 2], gap="medium")
    test_size = control_cols[0].slider("Test size", 0.10, 0.50, 0.30, 0.05)
    random_state = control_cols[1].number_input("Random seed", 0, 9999, 42, 1)
    control_cols[2].markdown('<div class="control-button-spacer"></div>', unsafe_allow_html=True)

    if control_cols[2].button("Train Models", type="primary", use_container_width=True):
        with st.spinner("Training and comparing models..."):
            save_eda_outputs(df, active_dataset)
            summary = train_and_evaluate(
                df,
                test_size=float(test_size),
                random_state=int(random_state),
                source_path=active_dataset,
            )
        st.success("Training complete")
        st.dataframe(format_summary(summary), use_container_width=True, hide_index=True)

    summary = load_summary()
    if summary is not None:
        st.markdown('<div class="status-line">Latest saved model results</div>', unsafe_allow_html=True)
        st.dataframe(format_summary(summary), use_container_width=True, hide_index=True)


def report_view() -> None:
    render_section("Reports", "saved model outputs")
    summary = load_summary()
    if summary is None:
        st.info("Train the models first to generate reports.")
        return

    metadata = load_metadata()
    if metadata and metadata.get("dataset"):
        dataset = metadata["dataset"]
        render_info_card(
            "Training Source",
            "Trained on <strong>"
            f"{escape(Path(str(dataset['source'])).name)}</strong> with "
            f"<strong>{int(dataset['rows']):,}</strong> rows, "
            f"<strong>{int(dataset['fraud_transactions']):,}</strong> fraud transactions, and "
            f"<strong>{int(metadata['balanced_rows_used_for_training']):,}</strong> balanced rows used.",
        )

    best = summary.iloc[0]
    render_metric_grid(
        [
            ("Best Model", str(best["Model"]), None),
            ("Accuracy", f"{best['Accuracy']:.3f}", None),
            ("Fraud Recall", f"{best['Recall_Fraud']:.3f}", None),
            ("Fraud F1", f"{best['F1_Fraud']:.3f}", None),
        ]
    )

    st.dataframe(format_summary(summary), use_container_width=True, hide_index=True)

    figure_paths = [
        FIGURES_DIR / "class_distribution.png",
        FIGURES_DIR / "amount_by_class.png",
        FIGURES_DIR / "logistic_regression_confusion_matrix.png",
        FIGURES_DIR / "decision_tree_confusion_matrix.png",
        FIGURES_DIR / "random_forest_confusion_matrix.png",
        FIGURES_DIR / "knn_confusion_matrix.png",
        FIGURES_DIR / "random_forest_feature_importance.png",
        FIGURES_DIR / "logistic_regression_feature_importance.png",
    ]
    existing_figures = [path for path in figure_paths if path.exists()]
    if existing_figures:
        render_section("Charts")
        image_cols = st.columns(2, gap="large")
        for index, image_path in enumerate(existing_figures):
            with image_cols[index % 2]:
                render_image_card(image_path)


def insights_view() -> None:
    render_section("Insights", "dataset quality and model drivers")
    df = dataset_status()
    if df is None:
        return

    active_dataset = resolve_dataset_path()
    profile = dataset_profile(df, active_dataset)
    amount_stats = df.groupby(TARGET_COLUMN)["Amount"].agg(["count", "mean", "median", "max"])

    render_info_card(
        "Dataset Health",
        "This view summarizes class imbalance, missing values, amount behavior, "
        "and the strongest model features from the latest training run.",
    )

    render_metric_grid(
        [
            ("Fraud Rate", f"{profile['fraud_rate']:.3%}", None),
            ("Missing Values", f"{int(df.isna().sum().sum()):,}", None),
            ("Median Amount", f"{df['Amount'].median():.2f}", None),
            ("Max Amount", f"{df['Amount'].max():.2f}", None),
        ]
    )

    with st.expander("Amount statistics by class", expanded=True):
        st.dataframe(amount_stats, use_container_width=True)

    quality = load_data_quality()
    if quality is not None:
        with st.expander("Data quality report", expanded=False):
            st.dataframe(quality, use_container_width=True, hide_index=True)
    else:
        st.info("Train once to generate reports/data_quality_report.csv.")

    importance = load_feature_importance()
    if importance is not None:
        model_names = sorted(importance["Model"].unique())
        selected_model = st.selectbox("Feature importance model", model_names)
        selected = importance[importance["Model"] == selected_model].head(15)
        st.bar_chart(selected.set_index("Feature")["Importance"])
        st.dataframe(selected, use_container_width=True, hide_index=True)
    else:
        st.info("Train once to generate reports/feature_importance.csv.")


def prediction_view() -> None:
    render_section("Prediction", "batch scoring")

    if not BEST_MODEL_PATH.exists():
        st.warning("Train the models first so the app can load models/best_model.joblib.")
        return

    model = joblib.load(BEST_MODEL_PATH)
    render_info_card(
        "Scoring Controls",
        "Lower thresholds catch more possible fraud but increase false alerts. "
        "Higher thresholds are stricter and reduce alerts.",
    )
    threshold = st.slider("Fraud probability threshold", 0.05, 0.95, 0.50, 0.05)
    uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV with Time, V1-V28, and Amount columns, or use the manual scorer below.")
    else:
        input_df = pd.read_csv(uploaded_file)
        try:
            output = predict_transactions(model, input_df, threshold=threshold)
        except ValueError as exc:
            st.error(str(exc))
            return

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

    render_section("Manual Scorer", "single transaction")
    active_dataset = dataset_status()
    sample_row = active_dataset[FEATURE_COLUMNS].median(numeric_only=True) if active_dataset is not None else None

    input_cols = st.columns(2)
    time_value = input_cols[0].number_input(
        "Time",
        value=float(sample_row["Time"]) if sample_row is not None else 0.0,
    )
    amount_value = input_cols[1].number_input(
        "Amount",
        value=float(sample_row["Amount"]) if sample_row is not None else 0.0,
    )

    values = {"Time": time_value, "Amount": amount_value}
    with st.expander("Advanced V1-V28 values", expanded=False):
        v_cols = st.columns(4)
        for index in range(1, 29):
            feature = f"V{index}"
            default = float(sample_row[feature]) if sample_row is not None else 0.0
            values[feature] = v_cols[(index - 1) % 4].number_input(feature, value=default)

    if st.button("Score Transaction", use_container_width=True):
        manual_df = pd.DataFrame([{column: values[column] for column in FEATURE_COLUMNS}])
        scored = predict_transactions(model, manual_df, threshold=threshold)
        prediction = "Fraud" if int(scored.iloc[0]["Predicted_Class"]) == 1 else "Normal"
        probability = scored.iloc[0].get("Fraud_Probability")
        render_result(prediction, probability)


def main() -> None:
    inject_styles()
    render_header()

    tabs = st.tabs(["Train", "Insights", "Reports", "Predict"])
    with tabs[0]:
        train_view()
    with tabs[1]:
        insights_view()
    with tabs[2]:
        report_view()
    with tabs[3]:
        prediction_view()


if __name__ == "__main__":
    main()
