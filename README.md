# RetainIQ — AI-Powered Customer Churn Intelligence Platform

RetainIQ is a premium Streamlit dashboard for churn intelligence, customer risk scoring, retention actions, and executive reporting. It is designed to feel like a real SaaS product: clean, visual, fast to scan, and focused on business decisions.

## What It Does

- Predicts churn likelihood for each customer
- Groups customers into risk bands and segments
- Explains risk drivers with AI-based insights
- Recommends practical retention actions
- Lets you filter, search, and export the current view
- Supports multiple CSV datasets from the `dataset/` folder

## Key Features

- Executive dashboard with KPI cards and interactive charts
- Customer-level prediction lab with risk percentage and confidence score
- Business insights that surface the strongest churn patterns
- Customer segmentation for loyal, new, at-risk, and high-value groups
- SHAP explainability for global and individual churn analysis
- Export filtered results to CSV
- Local model saving and automatic reuse

## Screenshots

Add your app screenshots here after running the project:

- Dashboard overview: [assets/screenshots/dashboard.png](assets/screenshots/dashboard.png)
- Dashboard alternative: [assets/screenshots/dashboardd.png](assets/screenshots/dashboardd.png)
- Prediction lab: [assets/screenshots/predictivelab_M.png](assets/screenshots/predictivelab_M.png)
- Prediction lab alternate: [assets/screenshots/predictivelab_L.png](assets/screenshots/predictivelab_L.png)
- AI insights: [assets/screenshots/ai_insights.png](assets/screenshots/ai_insights.png)
- Business insights: [assets/screenshots/bussinessInsights.png](assets/screenshots/bussinessInsights.png)
- Segmentation: [assets/screenshots/segmentation.png](assets/screenshots/segmentation.png)

## Demo Link

Add your hosted demo link here once it is available:

- Demo: [replace with your live demo URL](https://example.com)

## Project Structure

```text
RetainIQ/
├── app.py
├── requirements.txt
├── .gitignore
├── dataset/
├── assets/
├── models/
├── utils/
└── README.md
```

## Datasets

The app works with telco-style churn datasets and automatically normalizes common column variations. It can also handle the additional CSV files you placed in the `dataset/` folder.

If the default sample dataset is missing, the app creates a synthetic demo dataset so the dashboard still runs end to end.

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- SHAP
- Matplotlib

## Run Locally

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the app:

   ```bash
   streamlit run app.py
   ```

3. Open the local URL shown in the terminal.

## Future Improvements

- Add scheduled retraining and drift monitoring
- Add campaign performance tracking
- Add uploaded file validation with preview
- Add cohort retention reporting
- Add multi-team views for sales and customer success
