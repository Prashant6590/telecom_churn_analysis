# Telecom Churn Risk System

Scores active telecom customers on churn risk using plan, tenure, usage, billing and complaint data, explains the reasons, and recommends a retention action.

## Run
```bash
pip install -r requirements.txt
python generate_data.py        # optional: creates synthetic data/customers.csv
python churn_analysis.py       # or: --data your.csv --high 0.5 --medium 0.25
```

## Input columns
customer_id, plan, contract_type, payment_method, tenure_months, monthly_charges, avg_data_gb, avg_minutes, usage_change_pct, late_payments_6m, bill_increase_pct, complaints_6m, unresolved_complaints, support_calls_6m, churned (1 = left, 0 = active)

## Outputs (in `outputs/`)
report.md, scored_customers.csv, high_risk_customers.csv, feature_importance.csv, churn_model.joblib
