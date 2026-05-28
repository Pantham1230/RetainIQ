from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def assign_segments(df: pd.DataFrame, random_state: int = 42) -> Tuple[pd.DataFrame, Dict[str, int]]:
    frame = df.copy()
    required_columns = [column for column in ["tenure", "MonthlyCharges", "TotalCharges", "ChurnRiskProxy", "ChurnProbability"] if column in frame.columns]
    if len(required_columns) < 3:
        frame["Segment"] = "New Customers"
        return frame, {"New Customers": len(frame)}

    features = frame[required_columns].fillna(0).astype(float)
    scaled = StandardScaler().fit_transform(features)
    cluster_count = 4 if len(frame) >= 8 else 2
    kmeans = KMeans(n_clusters=cluster_count, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(scaled)
    frame["Cluster"] = clusters

    cluster_profile = pd.DataFrame(features).groupby(clusters).mean(numeric_only=True)
    tenure_col = "tenure"
    value_col = "TotalCharges" if "TotalCharges" in cluster_profile.columns else features.columns[0]
    risk_col = "ChurnProbability" if "ChurnProbability" in cluster_profile.columns else ("ChurnRiskProxy" if "ChurnRiskProxy" in cluster_profile.columns else features.columns[0])

    cluster_profile["value_rank"] = cluster_profile[value_col].rank(ascending=False, method="dense")
    cluster_profile["tenure_rank"] = cluster_profile[tenure_col].rank(ascending=False, method="dense")
    cluster_profile["risk_rank"] = cluster_profile[risk_col].rank(ascending=True, method="dense")

    mapping: Dict[int, str] = {}
    available_clusters = list(cluster_profile.index)
    loyal_cluster = int(cluster_profile.sort_values(["tenure_rank", "risk_rank"]).index[0])
    high_value_cluster = int(cluster_profile.sort_values(["value_rank", "risk_rank"]).index[0])
    at_risk_cluster = int(cluster_profile.sort_values(["risk_rank"], ascending=False).index[0])
    remaining = [cluster for cluster in available_clusters if cluster not in {loyal_cluster, high_value_cluster, at_risk_cluster}]
    new_cluster = int(remaining[0]) if remaining else loyal_cluster

    mapping[loyal_cluster] = "Loyal Customers"
    mapping[high_value_cluster] = "High-Value Customers"
    mapping[at_risk_cluster] = "At-Risk Customers"
    mapping[new_cluster] = "New Customers"

    frame["Segment"] = frame["Cluster"].map(mapping).fillna("New Customers")
    segment_counts = frame["Segment"].value_counts().to_dict()
    return frame, segment_counts


def segment_insight_text(segment_counts: Dict[str, int]) -> str:
    dominant_segment = max(segment_counts, key=segment_counts.get) if segment_counts else "New Customers"
    return f"Dominant customer segment: {dominant_segment}."
