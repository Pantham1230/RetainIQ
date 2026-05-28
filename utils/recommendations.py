from __future__ import annotations

from typing import List

import pandas as pd


def _add_unique(recommendations: List[str], message: str) -> None:
    if message not in recommendations:
        recommendations.append(message)


def generate_retention_recommendations(customer_row: pd.Series, risk_category: str, churn_probability: float) -> List[str]:
    recommendations: List[str] = []
    contract = str(customer_row.get("Contract", "Unknown"))
    payment_method = str(customer_row.get("PaymentMethod", "Unknown"))
    internet_service = str(customer_row.get("InternetService", "Unknown"))
    tenure = float(customer_row.get("tenure", 0))
    monthly_charges = float(customer_row.get("MonthlyCharges", 0))
    senior_citizen = int(customer_row.get("SeniorCitizen", 0))

    if risk_category == "High Risk" or churn_probability >= 0.7:
        _add_unique(recommendations, "Trigger a dedicated save desk outreach within 24 hours.")
        _add_unique(recommendations, "Offer a meaningful annual plan discount to reduce short-term churn pressure.")
        _add_unique(recommendations, "Provide a premium support or concierge onboarding package.")
    elif risk_category == "Medium Risk":
        _add_unique(recommendations, "Launch a targeted engagement campaign with service reminders and product education.")
        _add_unique(recommendations, "Offer loyalty rewards or bill credits for staying on the platform.")
    else:
        _add_unique(recommendations, "Keep the customer engaged with proactive check-ins and value-based communications.")
        _add_unique(recommendations, "Invite the customer to loyalty and referral programs to deepen stickiness.")

    if contract == "Month-to-month":
        _add_unique(recommendations, "Promote a one-year or two-year contract to stabilize retention risk.")
    if payment_method == "Electronic check":
        _add_unique(recommendations, "Encourage auto-pay or card payments to reduce payment friction.")
    if internet_service == "Fiber optic":
        _add_unique(recommendations, "Bundle value-added digital services to justify the higher-speed plan.")
    if tenure < 12:
        _add_unique(recommendations, "Strengthen the first-90-days journey with onboarding milestones and usage nudges.")
    if monthly_charges > 80:
        _add_unique(recommendations, "Offer a price protection message or tier review to address perceived value.")
    if senior_citizen == 1:
        _add_unique(recommendations, "Provide simplified support channels and priority assistance for accessibility.")

    return recommendations[:5]


def recommendation_summary(recommendations: List[str]) -> str:
    return " ".join(recommendations)
