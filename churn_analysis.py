#!/usr/bin/env python3
"""Telecom churn-risk system.

Analyses plan, tenure, usage, billing and complaint data to:
  1. score every active customer with a churn probability,
  2. assign a risk tier (High / Medium / Low),
  3. explain WHY each customer is at risk (reason codes),
  4. recommend a retention action,
  5. report which segments churn most and what drives churn overall.

Usage:
    python churn_analysis.py                       # uses data/customers.csv (creates demo data if missing)
    python churn_analysis.py --data my.csv --out results
    python churn_analysis.py --high 0.5 --medium 0.25
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL = ["plan", "contract_type", "payment_method"]
NUMERIC = [
    "tenure_months", "monthly_charges", "avg_data_gb", "avg_minutes",
    "usage_change_pct", "late_payments_6m", "bill_increase_pct",
    "complaints_6m", "unresolved_complaints", "support_calls_6m", "charge_per_gb",
]
REQUIRED = [
    "customer_id", "plan", "contract_type", "payment_method", "tenure_months",
    "monthly_charges", "avg_data_gb", "avg_minutes", "usage_change_pct",
    "late_payments_6m", "bill_increase_pct", "complaints_6m",
    "unresolved_complaints", "support_calls_6m", "churned",
]


# ----------------------------------------------------------------- data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        from generate_data import generate
        path.parent.mkdir(parents=True, exist_ok=True)
        generate().to_csv(path, index=False)
        print(f"No data found, created demo data at {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["charge_per_gb"] = df["monthly_charges"] / df["avg_data_gb"].clip(lower=0.5)
    df["tenure_band"] = pd.cut(df["tenure_months"], [0, 6, 12, 24, 48, 10_000],
                               labels=["0-6m", "7-12m", "13-24m", "25-48m", "49m+"])
    df["complaints_band"] = pd.cut(df["complaints_6m"], [-1, 0, 2, 10_000], labels=["0", "1-2", "3+"])
    df["usage_trend_band"] = pd.cut(df["usage_change_pct"], [-1000, -30, -10, 10, 1000],
                                    labels=["Drop >30%", "Drop 10-30%", "Stable", "Growth >10%"])
    df["bill_change_band"] = pd.cut(df["bill_increase_pct"], [-1000, 0, 15, 1000],
                                    labels=["No increase", "Up to 15%", "Spike >15%"])
    df["late_payment_band"] = pd.cut(df["late_payments_6m"], [-1, 0, 1, 10_000], labels=["0", "1", "2+"])
    return df


# ---------------------------------------------------------------- model
def build_model() -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ("num", "passthrough", NUMERIC),
    ])
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=4,
                                         l2_regularization=1.0, random_state=42)
    return Pipeline([("pre", pre), ("clf", clf)])


def evaluate(df: pd.DataFrame, high_thr: float):
    X, y = df[CATEGORICAL + NUMERIC], df["churned"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    model = build_model().fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]

    top_n = max(1, int(len(p) * 0.10))
    top_decile_rate = y_te.to_numpy()[np.argsort(-p)[:top_n]].mean()
    metrics = {
        "roc_auc": roc_auc_score(y_te, p),
        "avg_precision": average_precision_score(y_te, p),
        "base_rate": y_te.mean(),
        "top_decile_churn_rate": top_decile_rate,
        "top_decile_lift": top_decile_rate / y_te.mean(),
        "precision_at_high": precision_score(y_te, p >= high_thr, zero_division=0),
        "recall_at_high": recall_score(y_te, p >= high_thr, zero_division=0),
        "test_size": len(y_te),
    }
    imp = permutation_importance(model, X_te, y_te, scoring="roc_auc", n_repeats=8, random_state=42)
    importance = (pd.DataFrame({"feature": X_te.columns, "importance": imp.importances_mean})
                  .sort_values("importance", ascending=False).reset_index(drop=True))
    return metrics, importance


def score_all(df: pd.DataFrame) -> np.ndarray:
    """Out-of-fold probabilities: every customer is scored by a model that never saw them."""
    X, y = df[CATEGORICAL + NUMERIC], df["churned"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_predict(build_model(), X, y, cv=cv, method="predict_proba")[:, 1]


# -------------------------------------------------------------- reasons
def add_reasons(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    p90_cost = df["charge_per_gb"].quantile(0.90)
    rules = [
        ("Unresolved complaint", df["unresolved_complaints"] >= 1, 3.0,
         "Escalate to a senior agent; close the open ticket within 48h and follow up"),
        ("Repeated complaints", df["complaints_6m"] >= 3, 2.5,
         "Proactive call from retention team; check for recurring network or billing fault"),
        ("Usage dropped sharply", df["usage_change_pct"] <= -30, 2.5,
         "Check network quality in their area; offer a right-sized plan or loyalty data booster"),
        ("Recent bill increase", df["bill_increase_pct"] >= 15, 2.0,
         "Explain the charge; offer a goodwill credit or bill cap"),
        ("Repeated late payments", df["late_payments_6m"] >= 2, 2.0,
         "Offer a flexible due date and an auto-debit incentive"),
        ("Heavy support contact", df["support_calls_6m"] >= 5, 1.5,
         "Assign a named contact and resolve root cause"),
        ("New customer, no contract", (df["tenure_months"] <= 6) & (df["contract_type"] == "Month-to-month"), 1.5,
         "Onboarding check-in; offer a discount for a 12-month commitment"),
        ("Paying a lot per GB used", df["charge_per_gb"] >= p90_cost, 1.5,
         "Offer a cheaper plan that matches actual usage"),
    ]
    hits = [[] for _ in range(len(df))]
    for name, mask, weight, action in rules:
        for i in np.flatnonzero(mask.to_numpy()):
            hits[i].append((weight, name, action))
    ordered = [sorted(h, reverse=True) for h in hits]
    df["top_reasons"] = ["; ".join(n for _, n, _ in h[:3]) or "No single strong driver" for h in ordered]
    df["recommended_action"] = [h[0][2] if h else "Monitor; no action needed" for h in ordered]
    return df


# --------------------------------------------------------------- report
def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{v:,.3f}" if abs(v) < 10 else f"{v:,.0f}"
    return str(v)


def md_table(df: pd.DataFrame) -> str:
    df = df.reset_index(drop=True)
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(_fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def segment_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    t = (df.groupby(col, observed=True)
           .agg(customers=("customer_id", "count"), churn_rate=("churned", "mean"),
                avg_monthly_charge=("monthly_charges", "mean"))
           .reset_index().rename(columns={col: "segment"}))
    t["churn_rate"] = (t["churn_rate"] * 100).round(1).astype(str) + "%"
    return t


def write_report(df, active, metrics, importance, high_thr, med_thr, out: Path):
    tiers = (active.groupby("risk_tier", observed=True)
             .agg(customers=("customer_id", "count"),
                  avg_churn_prob=("churn_probability", "mean"),
                  monthly_revenue=("monthly_charges", "sum"))
             .reindex(["High", "Medium", "Low"]).reset_index())
    tiers["avg_churn_prob"] = (tiers["avg_churn_prob"] * 100).round(1).astype(str) + "%"
    expected_loss = (active["churn_probability"] * active["monthly_charges"]).sum()
    high_rev = active.loc[active.risk_tier == "High", "monthly_charges"].sum()

    reason_counts = (active[active.risk_tier == "High"]["top_reasons"].str.split("; ").explode()
                     .value_counts().head(6).rename_axis("reason").reset_index(name="high_risk_customers"))
    top10 = active.sort_values("churn_probability", ascending=False).head(10)[
        ["customer_id", "plan", "tenure_months", "monthly_charges", "churn_probability", "top_reasons"]].copy()
    top10["churn_probability"] = (top10["churn_probability"] * 100).round(0).astype(int).astype(str) + "%"

    seg_blocks = []
    for title, col in [("Plan", "plan"), ("Contract", "contract_type"), ("Tenure", "tenure_band"),
                       ("Complaints in last 6 months", "complaints_band"),
                       ("Usage trend (last 3 months vs prior)", "usage_trend_band"),
                       ("Latest bill change", "bill_change_band"),
                       ("Late payments in last 6 months", "late_payment_band")]:
        seg_blocks.append(f"### {title}\n\n{md_table(segment_table(df, col))}")

    report = f"""# Churn Risk Report

## Summary

- Customers analysed: **{len(df):,}** ({len(active):,} active, {int(df.churned.sum()):,} already churned)
- Historical churn rate: **{df.churned.mean():.1%}**
- Active customers flagged **High risk** (probability >= {high_thr:.0%}): **{(active.risk_tier == 'High').sum():,}**, carrying **{high_rev:,.0f}** in monthly revenue
- Expected monthly revenue lost to churn across active base: **{expected_loss:,.0f}**

## Risk tiers (active customers)

{md_table(tiers)}

Tiers: High >= {high_thr:.0%}, Medium >= {med_thr:.0%}, Low below that. Adjust with `--high` and `--medium`
to match your retention team's capacity.

## Most common reasons among High-risk customers

{md_table(reason_counts)}

## Model quality (held-out 25% of customers, n={metrics['test_size']:,})

| Metric | Value |
|---|---|
| ROC AUC | {metrics['roc_auc']:.3f} |
| Average precision | {metrics['avg_precision']:.3f} (base rate {metrics['base_rate']:.3f}) |
| Churn rate in top-scored 10% | {metrics['top_decile_churn_rate']:.1%} ({metrics['top_decile_lift']:.1f}x lift) |
| Precision at High threshold | {metrics['precision_at_high']:.1%} |
| Recall at High threshold | {metrics['recall_at_high']:.1%} |

## What drives churn (permutation importance, drop in AUC)

{md_table(importance.head(10))}

## Churn rate by segment

{(chr(10) * 2).join(seg_blocks)}

## 10 highest-risk active customers

{md_table(top10)}
"""
    (out / "report.md").write_text(report, encoding="utf-8")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).parent
    ap.add_argument("--data", default=str(here / "data" / "customers.csv"))
    ap.add_argument("--out", default=str(here / "outputs"))
    ap.add_argument("--high", type=float, default=0.50, help="probability cut-off for High risk")
    ap.add_argument("--medium", type=float, default=0.25, help="probability cut-off for Medium risk")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = engineer_features(load_data(Path(args.data)))
    metrics, importance = evaluate(df, args.high)
    df["churn_probability"] = score_all(df)
    df["risk_tier"] = pd.cut(df["churn_probability"], [-0.01, args.medium, args.high, 1.01],
                             labels=["Low", "Medium", "High"])
    df = add_reasons(df)

    active = df[df["churned"] == 0].copy()
    cols = ["customer_id", "plan", "contract_type", "tenure_months", "monthly_charges",
            "churn_probability", "risk_tier", "top_reasons", "recommended_action"]
    scored = active.sort_values("churn_probability", ascending=False)[cols].round({"churn_probability": 3})
    scored.to_csv(out / "scored_customers.csv", index=False)
    scored[scored["risk_tier"] == "High"].to_csv(out / "high_risk_customers.csv", index=False)
    importance.to_csv(out / "feature_importance.csv", index=False)
    write_report(df, active, metrics, importance, args.high, args.medium, out)

    final = build_model().fit(df[CATEGORICAL + NUMERIC], df["churned"])
    joblib.dump(final, out / "churn_model.joblib")

    print(f"AUC {metrics['roc_auc']:.3f} | top-decile lift {metrics['top_decile_lift']:.1f}x")
    print(active["risk_tier"].value_counts().reindex(["High", "Medium", "Low"]).to_string())
    print(f"Outputs written to {out}")


if __name__ == "__main__":
    main()
