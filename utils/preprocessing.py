from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Churn"
CUSTOMER_ID_COLUMN = "customerID"

BASE_CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "InternetService",
    "Contract",
    "PaymentMethod",
    "PaperlessBilling",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

BASE_NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "TenureRiskScore",
    "ChargePressure",
    "ChargePerTenure",
    "LifetimeValue",
    "ContractRiskScore",
    "EngagementRiskScore",
    "SupportGapScore",
]

REQUIRED_CATEGORICAL_DEFAULTS: Dict[str, str] = {
    "gender": "Unknown",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "InternetService": "DSL",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "PaperlessBilling": "Yes",
    "MultipleLines": "No",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
}


def list_dataset_files(dataset_dir: Path) -> List[Path]:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(dataset_dir.glob("*.csv"), key=lambda path: path.name.lower())
    return files


def _canonical_key(column_name: str) -> str:
    return "".join(character for character in column_name.lower() if character.isalnum())


def _find_column(columns: List[str], *aliases: str) -> str | None:
    canonical_map = {_canonical_key(column): column for column in columns}
    for alias in aliases:
        if _canonical_key(alias) in canonical_map:
            return canonical_map[_canonical_key(alias)]
    return None


def normalize_customer_schema(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame.columns = [column.strip() for column in frame.columns]
    source_columns = frame.columns.tolist()

    customer_col = _find_column(source_columns, "customerID", "CustomerId", "customer_id", "RowNumber", "id")
    if CUSTOMER_ID_COLUMN not in frame.columns:
        if customer_col is not None:
            frame[CUSTOMER_ID_COLUMN] = frame[customer_col].astype(str)
        else:
            frame[CUSTOMER_ID_COLUMN] = [f"CUST-{index:05d}" for index in range(1, len(frame) + 1)]

    churn_col = _find_column(source_columns, "Churn", "Exited", "Attrition", "target")
    if TARGET_COLUMN not in frame.columns:
        if churn_col is None:
            frame[TARGET_COLUMN] = "No"
        else:
            raw_churn = frame[churn_col]
            if pd.api.types.is_numeric_dtype(raw_churn):
                frame[TARGET_COLUMN] = np.where(pd.to_numeric(raw_churn, errors="coerce").fillna(0) >= 1, "Yes", "No")
            else:
                normalized = raw_churn.astype(str).str.strip().str.lower()
                frame[TARGET_COLUMN] = np.where(normalized.isin(["yes", "true", "1", "churn", "exited"]), "Yes", "No")

    gender_col = _find_column(source_columns, "gender", "Gender", "sex")
    if "gender" not in frame.columns and gender_col is not None:
        frame["gender"] = frame[gender_col].astype(str)

    age_col = _find_column(source_columns, "Age", "age")
    senior_col = _find_column(source_columns, "SeniorCitizen", "seniorcitizen", "is_senior")
    if "SeniorCitizen" not in frame.columns:
        if senior_col is not None:
            frame["SeniorCitizen"] = pd.to_numeric(frame[senior_col], errors="coerce").fillna(0).astype(int)
        elif age_col is not None:
            age_values = pd.to_numeric(frame[age_col], errors="coerce").fillna(0)
            frame["SeniorCitizen"] = (age_values >= 60).astype(int)
        else:
            frame["SeniorCitizen"] = 0

    tenure_col = _find_column(source_columns, "tenure", "Tenure", "monthswithcompany")
    if "tenure" not in frame.columns:
        if tenure_col is not None:
            frame["tenure"] = pd.to_numeric(frame[tenure_col], errors="coerce").fillna(0)
        else:
            frame["tenure"] = 0

    monthly_col = _find_column(source_columns, "MonthlyCharges", "monthlycharges", "monthlycharge")
    total_col = _find_column(source_columns, "TotalCharges", "totalcharges", "balance")
    salary_col = _find_column(source_columns, "EstimatedSalary", "salary")
    balance_col = _find_column(source_columns, "Balance", "accountbalance")
    product_col = _find_column(source_columns, "NumOfProducts", "products", "productcount")

    if "MonthlyCharges" not in frame.columns:
        if monthly_col is not None:
            frame["MonthlyCharges"] = pd.to_numeric(frame[monthly_col], errors="coerce")
        else:
            salary_values = pd.to_numeric(frame[salary_col], errors="coerce").fillna(50000) if salary_col else pd.Series(50000, index=frame.index)
            balance_values = pd.to_numeric(frame[balance_col], errors="coerce").fillna(0) if balance_col else pd.Series(0, index=frame.index)
            product_values = pd.to_numeric(frame[product_col], errors="coerce").fillna(1) if product_col else pd.Series(1, index=frame.index)
            frame["MonthlyCharges"] = (salary_values / 1200 + balance_values / 3000 + product_values * 6).clip(18, 140)

    if "TotalCharges" not in frame.columns:
        if total_col is not None and _canonical_key(total_col) != _canonical_key("Balance"):
            frame["TotalCharges"] = pd.to_numeric(frame[total_col], errors="coerce")
        else:
            balance_values = pd.to_numeric(frame[balance_col], errors="coerce").fillna(0) if balance_col else pd.Series(0, index=frame.index)
            frame["TotalCharges"] = frame["MonthlyCharges"].fillna(0) * frame["tenure"].fillna(0) + balance_values * 0.15

    geography_col = _find_column(source_columns, "Geography", "region")
    has_card_col = _find_column(source_columns, "HasCrCard", "hascreditcard")
    active_col = _find_column(source_columns, "IsActiveMember", "active", "isactive")

    if "Contract" not in frame.columns:
        frame["Contract"] = np.select(
            [frame["tenure"] < 12, frame["tenure"].between(12, 24, inclusive="left"), frame["tenure"] >= 24],
            ["Month-to-month", "One year", "Two year"],
            default="Month-to-month",
        )

    if "PaymentMethod" not in frame.columns:
        if has_card_col is not None:
            has_card = pd.to_numeric(frame[has_card_col], errors="coerce").fillna(0)
            frame["PaymentMethod"] = np.where(has_card >= 1, "Credit card (automatic)", "Electronic check")
        else:
            frame["PaymentMethod"] = "Electronic check"

    if "InternetService" not in frame.columns:
        if geography_col is not None:
            geo = frame[geography_col].astype(str).str.lower()
            frame["InternetService"] = np.where(
                geo.str.contains("france|germany"),
                "Fiber optic",
                np.where(geo.str.contains("spain"), "DSL", "DSL"),
            )
        else:
            frame["InternetService"] = np.where(frame["MonthlyCharges"] > frame["MonthlyCharges"].median(), "Fiber optic", "DSL")

    if "Partner" not in frame.columns:
        if active_col is not None:
            active_values = pd.to_numeric(frame[active_col], errors="coerce").fillna(0)
            frame["Partner"] = np.where(active_values >= 1, "Yes", "No")
        else:
            frame["Partner"] = "No"

    if "Dependents" not in frame.columns:
        product_values = pd.to_numeric(frame[product_col], errors="coerce").fillna(1) if product_col else pd.Series(1, index=frame.index)
        frame["Dependents"] = np.where(product_values >= 2, "Yes", "No")

    for column, default_value in REQUIRED_CATEGORICAL_DEFAULTS.items():
        if column not in frame.columns:
            frame[column] = default_value

    if TARGET_COLUMN in frame.columns:
        frame[TARGET_COLUMN] = np.where(frame[TARGET_COLUMN].astype(str).str.lower().isin(["yes", "true", "1", "churn", "exited"]), "Yes", "No")

    return frame


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def generate_synthetic_telco_data(n_rows: int = 1600, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    customer_ids = [f"CUST-{index:05d}" for index in range(1, n_rows + 1)]

    tenure = rng.integers(0, 73, size=n_rows)
    senior_citizen = rng.choice([0, 1], size=n_rows, p=[0.82, 0.18])
    partner = rng.choice(["Yes", "No"], size=n_rows, p=[0.53, 0.47])
    dependents = np.where(
        (partner == "Yes") & (rng.random(n_rows) > 0.45),
        "Yes",
        rng.choice(["Yes", "No"], size=n_rows, p=[0.28, 0.72]),
    )
    gender = rng.choice(["Female", "Male"], size=n_rows, p=[0.49, 0.51])
    phone_service = rng.choice(["Yes", "No"], size=n_rows, p=[0.91, 0.09])
    paperless_billing = rng.choice(["Yes", "No"], size=n_rows, p=[0.62, 0.38])
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], size=n_rows, p=[0.34, 0.47, 0.19])
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n_rows, p=[0.61, 0.22, 0.17])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_rows,
        p=[0.34, 0.19, 0.24, 0.23],
    )
    multiple_lines = np.where(phone_service == "No", "No phone service", rng.choice(["Yes", "No"], size=n_rows, p=[0.38, 0.62]))
    online_security = rng.choice(["Yes", "No", "No internet service"], size=n_rows, p=[0.28, 0.44, 0.28])
    online_backup = rng.choice(["Yes", "No", "No internet service"], size=n_rows, p=[0.31, 0.41, 0.28])
    device_protection = rng.choice(["Yes", "No", "No internet service"], size=n_rows, p=[0.29, 0.43, 0.28])
    tech_support = rng.choice(["Yes", "No", "No internet service"], size=n_rows, p=[0.27, 0.45, 0.28])
    streaming_tv = rng.choice(["Yes", "No", "No internet service"], size=n_rows, p=[0.39, 0.33, 0.28])
    streaming_movies = rng.choice(["Yes", "No", "No internet service"], size=n_rows, p=[0.41, 0.31, 0.28])

    contract_multiplier = np.select(
        [contract == "Month-to-month", contract == "One year", contract == "Two year"],
        [1.0, 0.82, 0.72],
        default=1.0,
    )
    internet_multiplier = np.select(
        [internet_service == "Fiber optic", internet_service == "DSL", internet_service == "No"],
        [1.25, 1.08, 0.65],
        default=1.0,
    )
    add_on_score = (
        (multiple_lines == "Yes").astype(int)
        + (online_security == "Yes").astype(int)
        + (online_backup == "Yes").astype(int)
        + (device_protection == "Yes").astype(int)
        + (tech_support == "Yes").astype(int)
        + (streaming_tv == "Yes").astype(int)
        + (streaming_movies == "Yes").astype(int)
    )

    monthly_charges = np.clip(
        rng.normal(
            loc=55 + 18 * (internet_service == "Fiber optic") + 8 * (paperless_billing == "Yes") + 4 * senior_citizen,
            scale=12,
            size=n_rows,
        ),
        18,
        125,
    )
    monthly_charges = np.round(monthly_charges * contract_multiplier * internet_multiplier, 2)
    monthly_charges += np.round(add_on_score * 3.25, 2)
    monthly_charges = np.clip(monthly_charges, 18, 140)

    total_charges = np.round(monthly_charges * tenure + rng.normal(0, 20, size=n_rows), 2)
    total_charges = np.where(tenure == 0, 0, np.maximum(total_charges, monthly_charges * tenure * 0.7))

    churn_score = (
        -1.45
        + 1.3 * (contract == "Month-to-month").astype(float)
        + 0.72 * (internet_service == "Fiber optic").astype(float)
        + 0.6 * (payment_method == "Electronic check").astype(float)
        + 0.5 * (paperless_billing == "Yes").astype(float)
        + 0.55 * (senior_citizen == 1).astype(float)
        + 0.85 * (tenure < 12).astype(float)
        + 0.38 * (monthly_charges > np.median(monthly_charges)).astype(float)
        - 0.8 * (contract == "Two year").astype(float)
        - 0.48 * (tech_support == "Yes").astype(float)
        - 0.28 * (partner == "Yes").astype(float)
        - 0.24 * (dependents == "Yes").astype(float)
        - 0.18 * (online_security == "Yes").astype(float)
        - 0.12 * (add_on_score >= 4).astype(float)
    )
    churn_probability = 1 / (1 + np.exp(-churn_score))
    churn = np.where(rng.random(n_rows) < churn_probability, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )

    noisy_rows = rng.choice(df.index, size=max(12, n_rows // 40), replace=False)
    df.loc[noisy_rows[: len(noisy_rows) // 2], "TotalCharges"] = np.nan
    df.loc[noisy_rows[len(noisy_rows) // 2 :], "TotalCharges"] = np.nan
    return df


def ensure_dataset(path: Path, n_rows: int = 1600) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pd.read_csv(path)
    df = generate_synthetic_telco_data(n_rows=n_rows)
    df.to_csv(path, index=False)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()
    tenure = engineered["tenure"].clip(lower=0)
    monthly = engineered["MonthlyCharges"].clip(lower=0)
    total = engineered["TotalCharges"].clip(lower=0)
    max_tenure = max(float(tenure.max()), 1.0)

    engineered["TenureRiskScore"] = 1 - tenure / max_tenure
    engineered["ChargePressure"] = monthly / max(float(monthly.median()), 1.0)
    engineered["ChargePerTenure"] = total / (tenure.replace(0, np.nan) + 1)
    engineered["ChargePerTenure"] = engineered["ChargePerTenure"].fillna(monthly)
    engineered["LifetimeValue"] = monthly * tenure
    engineered["ContractRiskScore"] = engineered["Contract"].map(
        {"Month-to-month": 1.0, "One year": 0.5, "Two year": 0.2}
    ).fillna(0.7)
    engineered["EngagementRiskScore"] = (
        (engineered.get("OnlineSecurity", "No") == "No").astype(float)
        + (engineered.get("TechSupport", "No") == "No").astype(float)
        + (engineered.get("PaperlessBilling", "No") == "Yes").astype(float)
    ) / 3.0
    engineered["SupportGapScore"] = (
        (engineered.get("TechSupport", "No") == "No").astype(float)
        + (engineered.get("OnlineBackup", "No") == "No").astype(float)
        + (engineered.get("DeviceProtection", "No") == "No").astype(float)
    ) / 3.0
    engineered["ChurnRiskProxy"] = (
        0.38 * engineered["TenureRiskScore"]
        + 0.29 * engineered["ChargePressure"]
        + 0.18 * engineered["ContractRiskScore"]
        + 0.15 * engineered["EngagementRiskScore"]
    )
    engineered["TenureBand"] = pd.cut(
        engineered["tenure"],
        bins=[-1, 6, 24, 48, 120],
        labels=["New", "Developing", "Established", "Veteran"],
    ).astype(str)
    engineered["MonthlyChargeBand"] = pd.cut(
        engineered["MonthlyCharges"],
        bins=[0, 35, 70, 100, 200],
        labels=["Budget", "Standard", "Premium", "Elite"],
    ).astype(str)
    engineered["RevenueAtRisk"] = engineered["MonthlyCharges"] * engineered["ChurnRiskProxy"]
    return engineered


def clean_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = normalize_customer_schema(df)
    cleaned.columns = [column.strip() for column in cleaned.columns]
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
        fallback_total = cleaned["MonthlyCharges"].fillna(cleaned["MonthlyCharges"].median()) * cleaned["tenure"].fillna(0)
        cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(fallback_total)

    for column in cleaned.columns:
        if cleaned[column].dtype == "object" and column not in {CUSTOMER_ID_COLUMN, TARGET_COLUMN}:
            cleaned[column] = cleaned[column].fillna("Unknown").astype(str).str.strip()

    numeric_fill_values = {
        "SeniorCitizen": 0,
        "tenure": 0,
        "MonthlyCharges": cleaned["MonthlyCharges"].median() if "MonthlyCharges" in cleaned else 0,
        "TotalCharges": cleaned["TotalCharges"].median() if "TotalCharges" in cleaned else 0,
    }
    for column, value in numeric_fill_values.items():
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(value)

    cleaned = engineer_features(cleaned)
    return cleaned


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    cleaned = clean_customer_data(df)
    if TARGET_COLUMN not in cleaned.columns:
        raise ValueError("Target column 'Churn' is missing from the dataset.")
    y = cleaned[TARGET_COLUMN].map({"Yes": 1, "No": 0}).astype(int)
    feature_frame = cleaned.drop(columns=[TARGET_COLUMN])
    if CUSTOMER_ID_COLUMN in feature_frame.columns:
        feature_frame = feature_frame.drop(columns=[CUSTOMER_ID_COLUMN])
    return feature_frame, y


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    available_columns = set(df.columns)
    categorical = [column for column in BASE_CATEGORICAL_COLUMNS if column in available_columns]
    numeric = [column for column in BASE_NUMERIC_COLUMNS if column in available_columns]
    return categorical, numeric


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    categorical_columns, numeric_columns = get_feature_columns(df)
    return ColumnTransformer(
        transformers=[
            ("categorical", _one_hot_encoder(), categorical_columns),
            ("numeric", StandardScaler(), numeric_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def customer_summary(df: pd.DataFrame) -> dict:
    churn_rate = float((df["Churn"].astype(str).str.lower() == "yes").mean()) if "Churn" in df.columns else 0.0
    high_risk = int((df.get("ChurnRiskProxy", pd.Series(dtype=float)) >= df.get("ChurnRiskProxy", pd.Series(dtype=float)).median()).sum())
    revenue_loss_risk = float(df.get("RevenueAtRisk", pd.Series(dtype=float)).sum())
    return {
        "total_customers": int(len(df)),
        "churn_rate": churn_rate,
        "retained_customers": int(len(df) - (df["Churn"].astype(str).str.lower() == "yes").sum()) if "Churn" in df.columns else 0,
        "high_risk_customers": high_risk,
        "revenue_loss_risk": revenue_loss_risk,
    }
