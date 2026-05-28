from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .preprocessing import CUSTOMER_ID_COLUMN, TARGET_COLUMN, build_preprocessor, clean_customer_data, split_features_target


def risk_category_from_probability(probability: float) -> str:
    if probability >= 0.7:
        return "High Risk"
    if probability >= 0.4:
        return "Medium Risk"
    return "Low Risk"


def risk_urgency_text(risk_category: str) -> str:
    mapping = {
        "High Risk": "Immediate attention required",
        "Medium Risk": "Offer engagement campaigns",
        "Low Risk": "Stable customer",
    }
    return mapping.get(risk_category, "Monitor regularly")


def probability_to_percent(probability: float) -> str:
    return f"{round(probability * 100):.0f}%"


def train_churn_model(df: pd.DataFrame, model_path: Path, random_state: int = 42) -> Dict[str, object]:
    feature_frame, target = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        feature_frame,
        target,
        test_size=0.2,
        stratify=target,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(x_train)
    classifier = RandomForestClassifier(
        n_estimators=320,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])
    pipeline.fit(x_train, y_train)

    test_probabilities = pipeline.predict_proba(x_test)[:, 1]
    test_predictions = (test_probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, test_predictions)),
        "precision": float(precision_score(y_test, test_predictions, zero_division=0)),
        "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
        "f1": float(f1_score(y_test, test_predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
    }

    model_bundle = {
        "pipeline": pipeline,
        "feature_columns": feature_frame.columns.tolist(),
        "metrics": metrics,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, model_path)
    return model_bundle


def load_model(model_path: Path) -> Dict[str, object] | None:
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def ensure_model(df: pd.DataFrame, model_path: Path) -> Dict[str, object]:
    model_bundle = load_model(model_path)
    if model_bundle is None:
        model_bundle = train_churn_model(df, model_path)
    return model_bundle


def predict_risk(model_bundle: Dict[str, object], customer_frame: pd.DataFrame) -> pd.DataFrame:
    pipeline: Pipeline = model_bundle["pipeline"]
    cleaned = clean_customer_data(customer_frame)
    feature_frame = cleaned.drop(columns=[TARGET_COLUMN], errors="ignore")
    if CUSTOMER_ID_COLUMN in feature_frame.columns:
        identifiers = feature_frame[CUSTOMER_ID_COLUMN].copy()
        feature_frame = feature_frame.drop(columns=[CUSTOMER_ID_COLUMN])
    else:
        identifiers = pd.Series(feature_frame.index.astype(str), index=feature_frame.index)

    probability = pipeline.predict_proba(feature_frame)[:, 1]
    output = customer_frame.copy()
    output["ChurnProbability"] = probability
    output["RiskCategory"] = [risk_category_from_probability(value) for value in probability]
    output["RiskUrgency"] = [risk_urgency_text(value) for value in output["RiskCategory"]]
    output["RiskPercent"] = [probability_to_percent(value) for value in probability]
    output["customerID"] = identifiers.values
    return output


def prediction_confidence(probability: float) -> float:
    centered = abs(probability - 0.5) * 2
    return round(0.55 + 0.45 * centered, 3)


def build_confusion_matrix_figure(model_bundle: Dict[str, object], df: pd.DataFrame) -> go.Figure:
    pipeline: Pipeline = model_bundle["pipeline"]
    feature_frame, target = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        feature_frame,
        target,
        test_size=0.2,
        stratify=target,
        random_state=42,
    )
    predictions = pipeline.predict(x_test)
    matrix = np.array(
        [
            [int(((y_test == 0) & (predictions == 0)).sum()), int(((y_test == 0) & (predictions == 1)).sum())],
            [int(((y_test == 1) & (predictions == 0)).sum()), int(((y_test == 1) & (predictions == 1)).sum())],
        ]
    )
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Predicted Retained", "Predicted Churn"],
            y=["Actual Retained", "Actual Churn"],
            colorscale=[[0, "#fdf2f8"], [0.5, "#c4d9ff"], [1, "#8bc8c0"]],
            text=matrix,
            texttemplate="%{text}",
            hovertemplate="Actual %{y}<br>%{x}: %{z}<extra></extra>",
        )
    )
    figure.update_layout(
        template="plotly_white",
        title="Confusion Matrix",
        height=360,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return figure


def predict_single_row(model_bundle: Dict[str, object], row: pd.DataFrame) -> Dict[str, float | str]:
    scored = predict_risk(model_bundle, row)
    probability = float(scored.iloc[0]["ChurnProbability"])
    return {
        "probability": probability,
        "risk_category": scored.iloc[0]["RiskCategory"],
        "risk_urgency": scored.iloc[0]["RiskUrgency"],
        "risk_percent": scored.iloc[0]["RiskPercent"],
        "confidence": prediction_confidence(probability),
    }
