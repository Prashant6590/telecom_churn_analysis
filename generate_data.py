"""Generate a synthetic telecom customer dataset for testing the churn system.

Replace data/customers.csv with your real export. It needs the same columns
(see COLUMNS below); `churned` is 1 if the customer left, 0 if still active.
"""
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "customer_id", "plan", "contract_type", "payment_method", "tenure_months",
    "monthly_charges", "avg_data_gb", "avg_minutes", "usage_change_pct",
    "late_payments_6m", "bill_increase_pct", "complaints_6m",
    "unresolved_complaints", "support_calls_6m", "churned",
]


def _calibrate_intercept(z: np.ndarray, target_rate: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(z + mid)))).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def generate(n: int = 5000, seed: int = 42, churn_rate: float = 0.20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    plans = ["Basic", "Standard", "Premium", "Family"]
    plan = rng.choice(plans, size=n, p=[0.30, 0.35, 0.15, 0.20])
    base_charge = {"Basic": 399, "Standard": 699, "Premium": 1199, "Family": 899}
    base_data = {"Basic": 8, "Standard": 22, "Premium": 45, "Family": 35}

    tenure = np.clip(rng.gamma(2.0, 14.0, n), 1, 120).round().astype(int)

    contract = np.where(
        tenure < 12,
        rng.choice(["Month-to-month", "12-month", "24-month"], n, p=[0.75, 0.2, 0.05]),
        rng.choice(["Month-to-month", "12-month", "24-month"], n, p=[0.35, 0.35, 0.30]),
    )
    payment = rng.choice(["Auto-debit", "UPI", "Card", "Cash/Cheque"], n, p=[0.35, 0.30, 0.20, 0.15])

    monthly_charges = np.array([base_charge[p] for p in plan]) * rng.normal(1.0, 0.07, n)
    data_gb = np.array([base_data[p] for p in plan]) * rng.lognormal(0.0, 0.45, n)
    minutes = np.clip(rng.normal(420, 180, n), 10, None)

    usage_change = np.clip(rng.normal(-3, 18, n), -90, 80)
    late_payments = rng.poisson(np.where(payment == "Cash/Cheque", 0.9, 0.35))
    bill_increase = rng.normal(2, 7, n)
    spike = rng.random(n) < 0.10
    bill_increase = np.where(spike, rng.uniform(15, 40, n), bill_increase)

    complaints = rng.poisson(0.55, n)
    unresolved = rng.binomial(complaints, 0.25)
    support_calls = complaints + rng.poisson(1.0, n)

    charge_per_gb = monthly_charges / np.clip(data_gb, 0.5, None)

    z = (
        0.40 * complaints
        + 0.90 * unresolved
        + 0.35 * late_payments
        - 0.035 * usage_change
        + 0.045 * np.clip(bill_increase, 0, None)
        - 0.022 * np.clip(tenure, 0, 60)
        + np.where(contract == "Month-to-month", 0.9, np.where(contract == "24-month", -0.7, 0.0))
        + np.where(payment == "Auto-debit", -0.35, 0.0)
        + 0.004 * np.clip(charge_per_gb - 40, 0, None)
        + 0.08 * np.clip(support_calls - 3, 0, None)
        + rng.normal(0, 0.6, n)
    )
    z = z + _calibrate_intercept(z, churn_rate)
    churned = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)],
        "plan": plan,
        "contract_type": contract,
        "payment_method": payment,
        "tenure_months": tenure,
        "monthly_charges": monthly_charges.round(0),
        "avg_data_gb": data_gb.round(1),
        "avg_minutes": minutes.round(0),
        "usage_change_pct": usage_change.round(1),
        "late_payments_6m": late_payments,
        "bill_increase_pct": bill_increase.round(1),
        "complaints_6m": complaints,
        "unresolved_complaints": unresolved,
        "support_calls_6m": support_calls,
        "churned": churned,
    })
    return df[COLUMNS]


if __name__ == "__main__":
    out = Path(__file__).parent / "data" / "customers.csv"
    out.parent.mkdir(exist_ok=True)
    frame = generate()
    frame.to_csv(out, index=False)
    print(f"Wrote {len(frame):,} rows to {out} (churn rate {frame.churned.mean():.1%})")
