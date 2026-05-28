from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PASTEL_SEQUENCE = ["#cab7ff", "#b2d7ff", "#f8c8dc", "#b9ead5", "#ffe2a8", "#d8d6ff"]


def _base_layout(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=title,
        height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(family="Segoe UI, sans-serif", color="#27415c"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=PASTEL_SEQUENCE,
    )
    return fig


def build_churn_distribution_figure(df: pd.DataFrame) -> go.Figure:
    counts = df["Churn"].value_counts().reset_index()
    counts.columns = ["Churn", "Count"]
    fig = px.pie(counts, names="Churn", values="Count", hole=0.55, color_discrete_sequence=PASTEL_SEQUENCE)
    fig.update_traces(textposition="inside", textinfo="percent+label", hovertemplate="%{label}: %{value}<extra></extra>")
    return _base_layout(fig, "Churn Distribution", height=360)


def build_contract_churn_figure(df: pd.DataFrame) -> go.Figure:
    summary = (
        df.assign(churn_flag=df["Churn"].astype(str).str.lower().eq("yes").astype(int))
        .groupby("Contract", as_index=False)
        .agg(churn_rate=("churn_flag", "mean"), customers=("customerID", "count"))
    )
    fig = px.bar(summary, x="Contract", y="churn_rate", color="Contract", text_auto=".0%", color_discrete_sequence=PASTEL_SEQUENCE)
    fig.update_traces(hovertemplate="%{x}<br>Churn rate: %{y:.1%}<extra></extra>")
    fig.update_yaxes(tickformat=".0%")
    return _base_layout(fig, "Churn by Contract Type", height=380)


def build_monthly_charges_figure(df: pd.DataFrame) -> go.Figure:
    sample = df.copy()
    if "ChurnProbability" not in sample.columns:
        sample["ChurnProbability"] = sample["Churn"].astype(str).str.lower().eq("yes").astype(int)
    fig = px.scatter(
        sample,
        x="MonthlyCharges",
        y="TotalCharges",
        color="RiskCategory" if "RiskCategory" in sample.columns else "Churn",
        size="ChurnProbability" if "ChurnProbability" in sample.columns else None,
        opacity=0.75,
        color_discrete_sequence=PASTEL_SEQUENCE,
        hover_data=[column for column in ["tenure", "Contract", "PaymentMethod"] if column in sample.columns],
    )
    fig.update_traces(marker=dict(line=dict(width=0.5, color="#ffffff")))
    return _base_layout(fig, "Monthly Charges vs Total Charges", height=400)


def build_tenure_figure(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="tenure", color="Churn", barmode="overlay", nbins=20, color_discrete_sequence=PASTEL_SEQUENCE)
    fig.update_traces(opacity=0.68)
    return _base_layout(fig, "Customer Tenure Analysis", height=360)


def build_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric_columns = [column for column in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "TenureRiskScore", "ChargePressure", "ChargePerTenure", "LifetimeValue", "ContractRiskScore", "EngagementRiskScore", "SupportGapScore", "ChurnRiskProxy"] if column in df.columns]
    correlation = df[numeric_columns].corr(numeric_only=True)
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation.values,
            x=correlation.columns,
            y=correlation.index,
            colorscale=[[0, "#fde2e4"], [0.5, "#d8e8ff"], [1, "#b9ead5"]],
            zmin=-1,
            zmax=1,
            hovertemplate="%{y} x %{x}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(template="plotly_white", title="Correlation Heatmap", height=420, margin=dict(l=30, r=30, t=60, b=30))
    return fig


def build_risk_distribution_figure(df: pd.DataFrame) -> go.Figure:
    if "RiskCategory" not in df.columns:
        risk_counts = pd.DataFrame({"RiskCategory": ["Low Risk", "Medium Risk", "High Risk"], "Count": [0, 0, 0]})
    else:
        risk_counts = df["RiskCategory"].value_counts().reindex(["Low Risk", "Medium Risk", "High Risk"], fill_value=0).reset_index()
        risk_counts.columns = ["RiskCategory", "Count"]
    fig = px.bar(risk_counts, x="RiskCategory", y="Count", color="RiskCategory", color_discrete_sequence=PASTEL_SEQUENCE, text_auto=True)
    return _base_layout(fig, "Churn Risk Distribution", height=360)


def build_feature_importance_figure(importance_frame: pd.DataFrame, max_features: int = 10) -> go.Figure:
    top_features = importance_frame.head(max_features).iloc[::-1]
    fig = px.bar(top_features, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale=["#f4d8ff", "#bcd9ff", "#95d5b2"])
    fig.update_layout(coloraxis_showscale=False)
    return _base_layout(fig, "SHAP Feature Importance", height=420)


def build_segment_figure(df: pd.DataFrame) -> go.Figure:
    if "Segment" not in df.columns:
        return _base_layout(go.Figure(), "Customer Segmentation", height=360)
    counts = df["Segment"].value_counts().reset_index()
    counts.columns = ["Segment", "Count"]
    fig = px.treemap(counts, path=["Segment"], values="Count", color="Segment", color_discrete_sequence=PASTEL_SEQUENCE)
    return _base_layout(fig, "Customer Segmentation", height=360)


def build_revenue_risk_figure(df: pd.DataFrame) -> go.Figure:
    if "RevenueAtRisk" not in df.columns:
        return _base_layout(go.Figure(), "Revenue Risk Analysis", height=360)
    summary = df.groupby("RiskCategory", as_index=False).agg(RevenueAtRisk=("RevenueAtRisk", "sum")) if "RiskCategory" in df.columns else df.groupby("TenureBand", as_index=False).agg(RevenueAtRisk=("RevenueAtRisk", "sum"))
    x_axis = "RiskCategory" if "RiskCategory" in summary.columns else "TenureBand"
    fig = px.bar(summary, x=x_axis, y="RevenueAtRisk", color=x_axis, color_discrete_sequence=PASTEL_SEQUENCE, text_auto="$.2s")
    return _base_layout(fig, "Revenue Risk Analysis", height=360)


def build_churn_trend_figure(df: pd.DataFrame) -> go.Figure:
    trend = df.groupby("TenureBand", as_index=False).agg(churn_rate=("Churn", lambda values: values.astype(str).str.lower().eq("yes").mean())) if "TenureBand" in df.columns else pd.DataFrame({"TenureBand": [], "churn_rate": []})
    fig = px.line(trend, x="TenureBand", y="churn_rate", markers=True)
    fig.update_traces(line=dict(color="#8cb4ff", width=4), marker=dict(size=10, color="#f6b8d0"))
    fig.update_yaxes(tickformat=".0%")
    return _base_layout(fig, "Churn Trend Indicator", height=340)


def build_segment_scatter_figure(df: pd.DataFrame) -> go.Figure:
    if "Segment" not in df.columns:
        return _base_layout(go.Figure(), "Customer Segmentation", height=360)
    fig = px.scatter(
        df,
        x="tenure",
        y="MonthlyCharges",
        color="Segment",
        size="ChurnProbability" if "ChurnProbability" in df.columns else None,
        color_discrete_sequence=PASTEL_SEQUENCE,
        hover_data=[column for column in ["customerID", "Contract", "RiskCategory"] if column in df.columns],
    )
    return _base_layout(fig, "Customer Segmentation Map", height=420)


def build_churn_by_category_figure(df: pd.DataFrame, category: str, title: str) -> go.Figure:
    summary = (
        df.assign(churn_flag=df["Churn"].astype(str).str.lower().eq("yes").astype(int))
        .groupby(category, as_index=False)
        .agg(churn_rate=("churn_flag", "mean"))
        .sort_values("churn_rate", ascending=False)
    )
    fig = px.bar(summary, x=category, y="churn_rate", color=category, color_discrete_sequence=PASTEL_SEQUENCE, text_auto=".0%")
    fig.update_yaxes(tickformat=".0%")
    return _base_layout(fig, title, height=360)
