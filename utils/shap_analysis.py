from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def get_transformed_frame(model_bundle: Dict[str, object], feature_frame: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    pipeline = model_bundle["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    transformed = preprocessor.transform(feature_frame)
    feature_names = list(preprocessor.get_feature_names_out())
    transformed_frame = pd.DataFrame(transformed, columns=feature_names, index=feature_frame.index)
    return transformed_frame, transformed, feature_names


def get_shap_outputs(model_bundle: Dict[str, object], feature_frame: pd.DataFrame) -> Dict[str, object]:
    pipeline = model_bundle["pipeline"]
    classifier = pipeline.named_steps["classifier"]
    transformed_frame, transformed_matrix, feature_names = get_transformed_frame(model_bundle, feature_frame)
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(transformed_matrix)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, -1]
    return {
        "explainer": explainer,
        "shap_values": shap_array,
        "feature_names": feature_names,
        "transformed_frame": transformed_frame,
        "expected_value": explainer.expected_value,
    }


def build_shap_summary_figure(model_bundle: Dict[str, object], feature_frame: pd.DataFrame, max_display: int = 12):
    shap_outputs = get_shap_outputs(model_bundle, feature_frame)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_outputs["shap_values"],
        shap_outputs["transformed_frame"],
        feature_names=shap_outputs["feature_names"],
        max_display=max_display,
        show=False,
    )
    figure = plt.gcf()
    return figure, shap_outputs


def feature_importance_from_shap(shap_outputs: Dict[str, object]) -> pd.DataFrame:
    shap_values = np.asarray(shap_outputs["shap_values"])
    importance = np.abs(shap_values).mean(axis=0)
    output = pd.DataFrame(
        {
            "feature": shap_outputs["feature_names"],
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)
    return output


def summarize_customer_explanation(model_bundle: Dict[str, object], feature_row: pd.DataFrame) -> Dict[str, object]:
    shap_outputs = get_shap_outputs(model_bundle, feature_row)
    shap_values = np.asarray(shap_outputs["shap_values"])[0]
    row_values = shap_outputs["transformed_frame"].iloc[0]
    explanation_frame = pd.DataFrame(
        {
            "feature": shap_outputs["feature_names"],
            "value": row_values.values,
            "impact": shap_values,
        }
    )
    explanation_frame["abs_impact"] = explanation_frame["impact"].abs()
    explanation_frame = explanation_frame.sort_values("abs_impact", ascending=False)
    return {
        "top_positive": explanation_frame[explanation_frame["impact"] > 0].head(3),
        "top_negative": explanation_frame[explanation_frame["impact"] < 0].head(3),
        "all_effects": explanation_frame.head(8),
    }


def clean_feature_label(feature_name: str) -> str:
    label = feature_name.split("__", 1)[-1]
    return label.replace("_", " ")
