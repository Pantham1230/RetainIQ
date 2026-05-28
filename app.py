from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from utils.prediction import build_confusion_matrix_figure, ensure_model, predict_risk, predict_single_row, probability_to_percent
from utils.preprocessing import CUSTOMER_ID_COLUMN, TARGET_COLUMN, clean_customer_data, customer_summary, ensure_dataset, list_dataset_files, split_features_target
from utils.recommendations import generate_retention_recommendations, recommendation_summary
from utils.segmentation import assign_segments, segment_insight_text
from utils.shap_analysis import build_shap_summary_figure, clean_feature_label, feature_importance_from_shap, summarize_customer_explanation
from utils.visualizations import (
    build_churn_by_category_figure,
    build_churn_distribution_figure,
    build_churn_trend_figure,
    build_contract_churn_figure,
    build_correlation_heatmap,
    build_feature_importance_figure,
    build_monthly_charges_figure,
    build_revenue_risk_figure,
    build_risk_distribution_figure,
    build_segment_figure,
    build_segment_scatter_figure,
    build_tenure_figure,
)


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
DEFAULT_DATASET_PATH = DATASET_DIR / "telco_churn.csv"
MODEL_DIR = ROOT / "models"
LOGO_PATH = ROOT / "assets" / "logo.png"


st.set_page_config(page_title="RetainIQ", page_icon="RIQ", layout="wide", initial_sidebar_state="expanded")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

            :root {
                --bg: #f4f7ff;
                --panel: rgba(255, 255, 255, 0.9);
                --panel-strong: rgba(255, 255, 255, 0.98);
                --border: rgba(135, 153, 188, 0.28);
                --text: #1f3550;
                --muted: #5b6f8a;
                --lavender: #cab7ff;
                --blue: #b7d5ff;
                --pink: #f7c6d9;
                --mint: #bfead7;
                --shadow: 0 12px 28px rgba(74, 101, 148, 0.12);
            }

            html, body, [class*='css'] {
                font-family: 'Inter', 'Segoe UI', sans-serif;
                color: var(--text);
            }

            h1, h2, h3, h4, h5, h6,
            p, label, span,
            [data-testid="stMarkdownContainer"] {
                color: var(--text);
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(202, 183, 255, 0.28), transparent 28%),
                    radial-gradient(circle at top right, rgba(183, 213, 255, 0.28), transparent 27%),
                    linear-gradient(180deg, #fcfdff 0%, #f3f7ff 100%);
            }

            section[data-testid='stSidebar'] {
                background: linear-gradient(180deg, #f7f9ff 0%, #edf3ff 100%);
                border-right: 1px solid var(--border);
            }

            section[data-testid='stSidebar'] .stMarkdown,
            section[data-testid='stSidebar'] label,
            section[data-testid='stSidebar'] p,
            section[data-testid='stSidebar'] span {
                color: #2d4563;
            }

            .riq-hero {
                padding: 1.4rem 1.5rem;
                border-radius: 26px;
                background: linear-gradient(135deg, rgba(202, 183, 255, 0.45), rgba(183, 213, 255, 0.42), rgba(191, 234, 215, 0.32));
                border: 1px solid rgba(116, 140, 178, 0.22);
                box-shadow: var(--shadow);
            }

            .riq-hero-row {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                flex-wrap: wrap;
                margin-bottom: 0.55rem;
            }

            .riq-eyebrow {
                display: inline-flex;
                align-items: center;
                gap: 0.45rem;
                padding: 0.3rem 0.7rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.68);
                color: #3d5777;
                font-size: 0.8rem;
                font-weight: 700;
                letter-spacing: 0.02em;
                border: 1px solid rgba(122, 145, 184, 0.2);
            }

            .riq-section-title {
                font-size: 1.16rem;
                font-weight: 800;
                color: #23395a;
                margin: 1.1rem 0 0.35rem 0;
                letter-spacing: -0.02em;
            }

            .riq-section-subtitle {
                color: #627792;
                font-size: 0.9rem;
                margin-bottom: 0.85rem;
            }

            .riq-divider {
                height: 1px;
                background: linear-gradient(90deg, rgba(178, 198, 231, 0), rgba(178, 198, 231, 0.85), rgba(178, 198, 231, 0));
                margin: 0.8rem 0 1rem 0;
            }

            .riq-title {
                font-size: 2.4rem;
                font-weight: 800;
                letter-spacing: -0.04em;
                margin-bottom: 0.25rem;
                color: #23395a;
            }

            .riq-subtitle {
                color: #3f5876;
                font-size: 0.98rem;
                line-height: 1.55;
            }

            .riq-kpi {
                background: var(--panel);
                border: 1px solid var(--border);
                border-radius: 22px;
                padding: 1rem 1.05rem;
                box-shadow: var(--shadow);
                backdrop-filter: blur(12px);
            }

            .riq-kpi div:nth-child(1) {
                color: #486281 !important;
            }

            .riq-kpi div:nth-child(2) {
                color: #1e3653 !important;
            }

            .riq-kpi div:nth-child(3) {
                color: #5e738f !important;
            }

            .riq-card {
                background: var(--panel-strong);
                border: 1px solid var(--border);
                border-radius: 24px;
                padding: 1rem 1.1rem;
                box-shadow: var(--shadow);
                margin-bottom: 1rem;
            }

            .riq-card.compact {
                padding: 0.8rem 0.9rem;
                border-radius: 18px;
            }

            .riq-card strong,
            .riq-card p,
            .riq-card div,
            .riq-card span {
                color: #2a4462;
            }

            .riq-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.35rem 0.75rem;
                border-radius: 999px;
                font-size: 0.82rem;
                font-weight: 700;
                border: 1px solid rgba(255,255,255,0.9);
            }

            .low-risk { background: rgba(191, 234, 215, 0.9); color: #175741; }
            .medium-risk { background: rgba(255, 226, 168, 0.92); color: #8d5a11; }
            .high-risk { background: rgba(248, 198, 220, 0.92); color: #8a2650; }

            .stMetric {
                background: rgba(255,255,255,0.55);
                border: 1px solid var(--border);
                border-radius: 18px;
                padding: 0.5rem;
                box-shadow: var(--shadow);
            }

            [data-testid='stMetricLabel'] {
                color: #486281 !important;
            }

            [data-testid='stMetricValue'] {
                color: #1f3652 !important;
            }

            .stButton button {
                background: linear-gradient(135deg, #b7d5ff, #cab7ff);
                color: #17324c;
                border: none;
                border-radius: 14px;
                padding: 0.65rem 1rem;
                font-weight: 700;
            }

            .stDownloadButton button {
                background: linear-gradient(135deg, #bfead7, #b7d5ff);
                color: #17324c;
                border: none;
                border-radius: 14px;
                padding: 0.65rem 1rem;
                font-weight: 700;
            }

            .stSelectbox [data-baseweb='select'] > div,
            .stMultiSelect [data-baseweb='select'] > div,
            .stTextInput input,
            .stNumberInput input {
                background: rgba(255, 255, 255, 0.96) !important;
                color: #223b58 !important;
                border: 1px solid rgba(137, 157, 194, 0.45) !important;
                border-radius: 12px !important;
            }

            .stRadio [role='radiogroup'] {
                gap: 0.4rem;
            }

            /* Sidebar radios: simple text list without white boxed pills */
            section[data-testid='stSidebar'] .stRadio label {
                background: transparent !important;
                padding: 0.35rem 0.35rem !important;
                border: none !important;
                border-radius: 6px !important;
                margin: 0.08rem 0 !important;
                color: #2d4563 !important;
                display: flex !important;
                align-items: center !important;
                gap: 0.6rem !important;
                line-height: 1.1 !important;
            }

            section[data-testid='stSidebar'] .stRadio label:hover {
                background: rgba(183,213,255,0.06) !important;
                border-color: transparent !important;
            }

            .stSlider [data-baseweb='slider'] {
                padding-top: 0.35rem;
            }

            .stProgress > div > div > div > div {
                background: linear-gradient(90deg, #f6b9d0, #b7d5ff, #bfead7) !important;
            }

            .dataframe { border-radius: 16px; overflow: hidden; }

            div[data-testid='stVerticalBlockBorderWrapper'] {
                background: rgba(255, 255, 255, 0.82);
                border: 1px solid rgba(135, 153, 188, 0.22);
                border-radius: 22px;
                box-shadow: var(--shadow);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dataset(dataset_name: str) -> pd.DataFrame:
    dataset_path = DATASET_DIR / dataset_name
    if not dataset_path.exists() and dataset_name == DEFAULT_DATASET_PATH.name:
        raw = ensure_dataset(dataset_path)
    else:
        raw = pd.read_csv(dataset_path)
    return clean_customer_data(raw)


@st.cache_resource(show_spinner=True)
def load_model_bundle(dataset_name: str) -> Dict[str, object]:
    dataset = load_dataset(dataset_name)
    model_path = model_path_for_dataset(dataset_name)
    return ensure_model(dataset, model_path)


def available_dataset_names() -> List[str]:
    default_exists = DEFAULT_DATASET_PATH.exists()
    if not default_exists:
        ensure_dataset(DEFAULT_DATASET_PATH)
    names = [path.name for path in list_dataset_files(DATASET_DIR)]
    if DEFAULT_DATASET_PATH.name in names:
        names = [DEFAULT_DATASET_PATH.name] + [name for name in names if name != DEFAULT_DATASET_PATH.name]
    return names


def model_path_for_dataset(dataset_name: str) -> Path:
    safe_stem = Path(dataset_name).stem.replace(" ", "_").replace("-", "_")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR / f"{safe_stem}_churn_model.pkl"


def build_filtered_frame(df: pd.DataFrame, filters: Dict[str, object]) -> pd.DataFrame:
    frame = df.copy()
    for column, value in filters.items():
        if value in (None, "All", ""):
            continue
        if column == "tenure_range":
            low, high = value
            frame = frame[(frame["tenure"] >= low) & (frame["tenure"] <= high)]
        else:
            frame = frame[frame[column] == value]
    return frame


def format_kpi_card(label: str, value: str, delta: str = "") -> None:
    st.markdown(
        f"""
        <div class="riq-kpi">
            <div style="color:#6c7f97;font-size:0.86rem;font-weight:700;letter-spacing:0.02em;">{label}</div>
            <div style="font-size:1.7rem;font-weight:800;margin-top:0.25rem;">{value}</div>
            <div style="color:#7a88a4;font-size:0.82rem;min-height:1rem;">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_risk_badge(risk_category: str) -> str:
    class_name = {
        "Low Risk": "low-risk",
        "Medium Risk": "medium-risk",
        "High Risk": "high-risk",
    }.get(risk_category, "medium-risk")
    return f'<span class="riq-pill {class_name}">{risk_category}</span>'


def build_sidebar(df: pd.DataFrame, dataset_name: str) -> Tuple[str, Dict[str, object]]:
    st.sidebar.markdown(
        """
        <div class='riq-card'>
            <div style='display:flex;align-items:center;gap:0.75rem;margin-bottom:0.55rem;'>
                <div style='width:42px;height:42px;border-radius:14px;background:linear-gradient(135deg,#cab7ff,#b7d5ff);display:flex;align-items:center;justify-content:center;font-weight:800;color:#17324c;box-shadow:0 10px 18px rgba(127,154,216,0.22);'>RI</div>
                <div>
                    <div style='font-size:1.45rem;font-weight:800;letter-spacing:-0.03em;color:#213855;'>RetainIQ</div>
                    <div style='color:#5d7390;font-size:0.9rem;margin-top:0.18rem;'>AI-powered customer churn intelligence</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), width="stretch")

    dataset_choices = available_dataset_names()
    selected_dataset = st.sidebar.selectbox(
        "Dataset Selector",
        dataset_choices,
        index=dataset_choices.index(dataset_name) if dataset_name in dataset_choices else 0,
    )
    if selected_dataset != dataset_name:
        st.session_state["selected_dataset"] = selected_dataset
        st.rerun()

    navigation = st.sidebar.radio(
        "Navigation",
        ["Executive Dashboard", "Prediction Lab", "AI Insights", "Segmentation", "Business Insights"],
        index=0,
    )

    st.sidebar.markdown("### Filters")
    gender = st.sidebar.selectbox("Gender", ["All"] + sorted(df["gender"].dropna().unique().tolist()))
    contract = st.sidebar.selectbox("Contract Type", ["All"] + sorted(df["Contract"].dropna().unique().tolist()))
    internet = st.sidebar.selectbox("Internet Service", ["All"] + sorted(df["InternetService"].dropna().unique().tolist()))
    payment = st.sidebar.selectbox("Payment Method", ["All"] + sorted(df["PaymentMethod"].dropna().unique().tolist()))
    senior = st.sidebar.selectbox("Senior Citizen", ["All", 0, 1])
    tenure_range = st.sidebar.slider("Tenure Range", int(df["tenure"].min()), int(df["tenure"].max()), (0, int(df["tenure"].max())))

    st.sidebar.markdown("### Dataset & Model")
    model_path = model_path_for_dataset(dataset_name)
    st.sidebar.caption(f"Dataset: {dataset_name}")
    st.sidebar.caption(f"Rows: {len(df):,}")
    st.sidebar.caption(f"Model file: {'available' if model_path.exists() else 'training required'}")
    st.sidebar.caption(f"Last refresh: {datetime.now().strftime('%b %d, %Y %I:%M %p')}")

    filters = {
        "gender": gender,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
        "SeniorCitizen": senior,
        "tenure_range": tenure_range,
    }
    return navigation, filters


def section_heading(title: str, subtitle: str | None = None) -> None:
    st.markdown(f"<div class='riq-section-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='riq-section-subtitle'>{subtitle}</div>", unsafe_allow_html=True)


def divider() -> None:
    st.markdown("<div class='riq-divider'></div>", unsafe_allow_html=True)


def customer_insight_snippets(df: pd.DataFrame) -> List[str]:
    churn_rate = df["Churn"].astype(str).str.lower().eq("yes").mean()
    monthly_churn = df.groupby("Contract", as_index=False).agg(churn_rate=("Churn", lambda values: values.astype(str).str.lower().eq("yes").mean())).sort_values("churn_rate", ascending=False)
    fiber_rate = df[df["InternetService"] == "Fiber optic"]["Churn"].astype(str).str.lower().eq("yes").mean() if (df["InternetService"] == "Fiber optic").any() else 0
    electronic_rate = df[df["PaymentMethod"] == "Electronic check"]["Churn"].astype(str).str.lower().eq("yes").mean() if (df["PaymentMethod"] == "Electronic check").any() else 0
    snippets = [
        f"Overall churn rate sits at {churn_rate:.1%} in the current selection.",
        f"{monthly_churn.iloc[0]['Contract']} contracts show the strongest churn pressure.",
        f"Fiber optic customers churn at {fiber_rate:.1%}, signaling an experience or pricing gap.",
        f"Electronic check users churn at {electronic_rate:.1%}, which often indicates payment friction.",
    ]
    return snippets


def render_hero(filtered_df: pd.DataFrame) -> None:
    summary = customer_summary(filtered_df)
    st.markdown(
        """
        <div class="riq-hero">
            <div class="riq-hero-row">
                <div class="riq-eyebrow">Retention Intelligence Workspace</div>
                <div class="riq-eyebrow">Live dataset-aware analysis</div>
            </div>
            <div class="riq-title">RetainIQ</div>
            <div class="riq-subtitle">AI-powered customer churn intelligence for product, growth, and retention teams. Predict churn risk, explain the drivers, and surface the next best action for each customer.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        ("Total Customers", f"{summary['total_customers']:,}", "Current filtered cohort"),
        ("Churn Rate", f"{summary['churn_rate']:.1%}", "Observed churn in selection"),
        ("Retained Customers", f"{summary['retained_customers']:,}", "Still active customers"),
        ("High-Risk Customers", f"{summary['high_risk_customers']:,}", "Risk proxy above threshold"),
        ("Revenue Loss Risk", f"${summary['revenue_loss_risk']:,.0f}", "Estimated monthly value at risk"),
    ]
    for container, metric in zip([col1, col2, col3, col4, col5], metrics):
        with container:
            format_kpi_card(*metric)


def render_dashboard(filtered_df: pd.DataFrame, model_bundle: Dict[str, object]) -> None:
    scored_df = predict_risk(model_bundle, filtered_df)
    segmented_df, _ = assign_segments(scored_df)
    st.session_state["scored_df"] = segmented_df

    render_hero(segmented_df)

    divider()
    section_heading("Intelligence Snapshot", "A quick read on churn pressure, customer mix, and the most important signals in the current selection.")
    insight_cols = st.columns(4)
    for idx, snippet in enumerate(customer_insight_snippets(segmented_df)[:4]):
        with insight_cols[idx]:
            st.markdown(f"<div class='riq-card compact'><strong>AI Insight {idx + 1}</strong><div style='margin-top:0.45rem;color:#5f748f;line-height:1.55;'>{snippet}</div></div>", unsafe_allow_html=True)

    chart_rows = [
        [build_churn_distribution_figure(segmented_df), build_contract_churn_figure(segmented_df)],
        [build_monthly_charges_figure(segmented_df), build_tenure_figure(segmented_df)],
        [build_correlation_heatmap(segmented_df), build_risk_distribution_figure(segmented_df)],
        [build_segment_figure(segmented_df), build_revenue_risk_figure(segmented_df)],
    ]
    for left_fig, right_fig in chart_rows:
        left, right = st.columns(2)
        with left:
            st.plotly_chart(left_fig, width="stretch")
        with right:
            st.plotly_chart(right_fig, width="stretch")

    divider()
    section_heading("Customer Search and Export", "Search any customer, inspect their retention risk, and export the filtered cohort when needed.")
    customer_query = st.text_input("Search by Customer ID", placeholder="CUST-00042")
    export_frame = segmented_df.copy()
    if customer_query:
        customer_match = segmented_df[segmented_df["customerID"].astype(str).str.contains(customer_query, case=False, na=False)]
        if not customer_match.empty:
            export_frame = customer_match
            row = customer_match.iloc[[0]]
            explanation = predict_single_row(model_bundle, row)
            recommendation_list = generate_retention_recommendations(row.iloc[0], explanation["risk_category"], explanation["probability"])
            st.markdown("<div class='riq-card'>", unsafe_allow_html=True)
            st.markdown(f"**Customer:** {row.iloc[0]['customerID']}  ")
            st.markdown(f"**Risk:** {format_risk_badge(explanation['risk_category'])}", unsafe_allow_html=True)
            st.write(f"Probability: {explanation['risk_percent']} | Confidence: {explanation['confidence']:.0%} | Urgency: {explanation['risk_urgency']}")
            st.write(f"Recommended action: {recommendation_summary(recommendation_list)}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No customer matched that search string.")

    st.download_button(
        "Export filtered data to CSV",
        data=export_frame.to_csv(index=False).encode("utf-8"),
        file_name="retainiq_filtered_customers.csv",
        mime="text/csv",
    )


def render_prediction_lab(filtered_df: pd.DataFrame, model_bundle: Dict[str, object]) -> None:
    section_heading("Prediction Lab", "Select a customer and review the risk estimate, confidence, and recommended retention action.")
    left, right = st.columns([1.1, 1.2])
    with left:
        customer_ids = filtered_df["customerID"].tolist()
        selected_customer = st.selectbox("Select a customer", customer_ids, index=0 if customer_ids else None)
        if selected_customer and not filtered_df.empty:
            current = filtered_df[filtered_df["customerID"] == selected_customer].iloc[[0]]
            prediction = predict_single_row(model_bundle, current)
            recommendations = generate_retention_recommendations(current.iloc[0], prediction["risk_category"], prediction["probability"])
            st.markdown("<div class='riq-card'>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:1rem;font-weight:800;margin-bottom:0.35rem;'>{selected_customer}</div>", unsafe_allow_html=True)
            st.markdown(f"{format_risk_badge(prediction['risk_category'])}", unsafe_allow_html=True)
            st.progress(min(max(prediction["probability"], 0.0), 1.0))
            st.write(f"Churn risk: {prediction['risk_percent']}")
            st.write(f"Urgency: {prediction['risk_urgency']}")
            st.write(f"Prediction confidence: {prediction['confidence']:.0%}")
            st.write(f"Retention playbook: {recommendation_summary(recommendations)}")
            st.markdown("</div>", unsafe_allow_html=True)
            st.dataframe(current.drop(columns=["ChurnProbability", "RiskCategory", "RiskUrgency", "RiskPercent"], errors="ignore"), width="stretch")

    with right:
        st.markdown("<div class='riq-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:1rem;font-weight:800;color:#23395a;margin-bottom:0.45rem;'>Model performance</div>", unsafe_allow_html=True)
        metrics = model_bundle.get("metrics", {})
        metric_cols = st.columns(5)
        metric_specs = [
            ("Accuracy", metrics.get("accuracy", 0.0)),
            ("Precision", metrics.get("precision", 0.0)),
            ("Recall", metrics.get("recall", 0.0)),
            ("F1 Score", metrics.get("f1", 0.0)),
            ("ROC-AUC", metrics.get("roc_auc", 0.0)),
        ]
        for container, spec in zip(metric_cols, metric_specs):
            with container:
                st.metric(spec[0], f"{spec[1]:.3f}")
        st.plotly_chart(build_confusion_matrix_figure(model_bundle, filtered_df), width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)


def render_ai_insights(filtered_df: pd.DataFrame, model_bundle: Dict[str, object]) -> None:
    section_heading("AI Insights", "SHAP-based explanations for the cohort and one customer-level example.")
    feature_frame, _ = split_features_target(filtered_df)
    feature_frame = feature_frame.drop(columns=[CUSTOMER_ID_COLUMN], errors="ignore")
    sample = feature_frame.sample(min(len(feature_frame), 180), random_state=42) if len(feature_frame) > 180 else feature_frame
    summary_fig, shap_outputs = build_shap_summary_figure(model_bundle, sample)
    importance_frame = feature_importance_from_shap(shap_outputs)

    top_left, top_right = st.columns([1.1, 1.0])
    with top_left:
        st.pyplot(summary_fig, clear_figure=True)
    with top_right:
        st.plotly_chart(build_feature_importance_figure(importance_frame), width="stretch")

    if not filtered_df.empty:
        selected = filtered_df.iloc[[0]]
        explanation = summarize_customer_explanation(
            model_bundle,
            selected.drop(columns=["ChurnProbability", "RiskCategory", "RiskUrgency", "RiskPercent", "Segment", "Cluster"], errors="ignore").drop(columns=[TARGET_COLUMN], errors="ignore"),
        )
        st.markdown("<div class='riq-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:1rem;font-weight:800;color:#23395a;margin-bottom:0.45rem;'>Individual customer explanation</div>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<div style='font-weight:700;color:#3f5876;margin-bottom:0.4rem;'>Top positive churn drivers</div>", unsafe_allow_html=True)
            for _, row in explanation["top_positive"].iterrows():
                st.markdown(f"<div style='margin:0.22rem 0;color:#2a4462;'>• {clean_feature_label(row['feature'])}</div>", unsafe_allow_html=True)
        with col_b:
            st.markdown("<div style='font-weight:700;color:#3f5876;margin-bottom:0.4rem;'>Top protective factors</div>", unsafe_allow_html=True)
            for _, row in explanation["top_negative"].iterrows():
                st.markdown(f"<div style='margin:0.22rem 0;color:#2a4462;'>• {clean_feature_label(row['feature'])}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_segmentation(filtered_df: pd.DataFrame) -> None:
    section_heading("Customer Segmentation", "Clustering that separates loyal, at-risk, high-value, and new customers.")
    segmented_df, counts = assign_segments(filtered_df)
    st.info(segment_insight_text(counts))
    cols = st.columns(4)
    for index, segment in enumerate(["Loyal Customers", "At-Risk Customers", "High-Value Customers", "New Customers"]):
        with cols[index]:
            st.metric(segment, counts.get(segment, 0))
    st.plotly_chart(build_segment_scatter_figure(segmented_df), width="stretch")


def render_business_insights(filtered_df: pd.DataFrame) -> None:
    section_heading("Business Insights Engine", "Filter-aware statements that summarize the strongest commercial signals in the data.")
    insights = customer_insight_snippets(filtered_df)
    trend_fig = build_churn_trend_figure(filtered_df)
    left, right = st.columns([1.0, 1.2])
    with left:
        for insight in insights:
            st.markdown(f"<div class='riq-card'>{insight}</div>", unsafe_allow_html=True)
        if "Contract" in filtered_df.columns:
            st.plotly_chart(build_churn_by_category_figure(filtered_df, "Contract", "Churn by Contract Type"), width="stretch")
    with right:
        st.plotly_chart(trend_fig, width="stretch")
        st.plotly_chart(build_revenue_risk_figure(filtered_df), width="stretch")


def main() -> None:
    inject_styles()
    selected_dataset = st.session_state.get("selected_dataset", DEFAULT_DATASET_PATH.name)
    if selected_dataset not in available_dataset_names():
        selected_dataset = DEFAULT_DATASET_PATH.name
    dataset = load_dataset(selected_dataset)
    model_bundle = load_model_bundle(selected_dataset)
    navigation, filters = build_sidebar(dataset, selected_dataset)
    filtered_df = build_filtered_frame(dataset, filters)

    if filtered_df.empty:
        st.warning("No customers match the current filters. Reset filters to continue.")
        return

    if navigation == "Executive Dashboard":
        render_dashboard(filtered_df, model_bundle)
    elif navigation == "Prediction Lab":
        render_prediction_lab(filtered_df, model_bundle)
    elif navigation == "AI Insights":
        render_ai_insights(filtered_df, model_bundle)
    elif navigation == "Segmentation":
        render_segmentation(filtered_df)
    elif navigation == "Business Insights":
        render_business_insights(filtered_df)


if __name__ == "__main__":
    main()
