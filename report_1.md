# Churn Risk Report

## Summary

- Customers analysed: **5,000** (4,038 active, 962 already churned)
- Historical churn rate: **19.2%**
- Active customers flagged **High risk** (probability >= 50%): **154**, carrying **110,033** in monthly revenue
- Expected monthly revenue lost to churn across active base: **468,710**

## Risk tiers (active customers)

| risk_tier | customers | avg_churn_prob | monthly_revenue |
|---|---|---|---|
| High | 154 | 60.2% | 110,033 |
| Medium | 631 | 35.1% | 452,719 |
| Low | 3253 | 10.4% | 2,349,974 |

Tiers: High >= 50%, Medium >= 25%, Low below that. Adjust with `--high` and `--medium`
to match your retention team's capacity.

## Most common reasons among High-risk customers

| reason | high_risk_customers |
|---|---|
| Unresolved complaint | 61 |
| Recent bill increase | 40 |
| Usage dropped sharply | 40 |
| New customer, no contract | 26 |
| Repeated late payments | 20 |
| Paying a lot per GB used | 20 |

## Model quality (held-out 25% of customers, n=1,250)

| Metric | Value |
|---|---|
| ROC AUC | 0.766 |
| Average precision | 0.466 (base rate 0.192) |
| Churn rate in top-scored 10% | 53.6% (2.8x lift) |
| Precision at High threshold | 56.8% |
| Recall at High threshold | 19.2% |

## What drives churn (permutation importance, drop in AUC)

| feature | importance |
|---|---|
| usage_change_pct | 0.080 |
| contract_type | 0.080 |
| tenure_months | 0.020 |
| complaints_6m | 0.019 |
| bill_increase_pct | 0.016 |
| late_payments_6m | 0.008 |
| avg_data_gb | 0.005 |
| payment_method | 0.004 |
| unresolved_complaints | 0.004 |
| charge_per_gb | 0.002 |

## Churn rate by segment

### Plan

| segment | customers | churn_rate | avg_monthly_charge |
|---|---|---|---|
| Basic | 1541 | 19.8% | 400 |
| Family | 984 | 21.3% | 899 |
| Premium | 733 | 16.2% | 1,199 |
| Standard | 1742 | 18.8% | 699 |

### Contract

| segment | customers | churn_rate | avg_monthly_charge |
|---|---|---|---|
| 12-month | 1601 | 14.2% | 725 |
| 24-month | 1286 | 10.2% | 711 |
| Month-to-month | 2113 | 28.5% | 720 |

### Tenure

| segment | customers | churn_rate | avg_monthly_charge |
|---|---|---|---|
| 0-6m | 399 | 31.1% | 702 |
| 7-12m | 706 | 25.9% | 702 |
| 13-24m | 1515 | 19.8% | 721 |
| 25-48m | 1681 | 16.5% | 728 |
| 49m+ | 699 | 11.0% | 721 |

### Complaints in last 6 months

| segment | customers | churn_rate | avg_monthly_charge |
|---|---|---|---|
| 0 | 2888 | 14.1% | 722 |
| 1-2 | 1994 | 25.2% | 713 |
| 3+ | 118 | 44.1% | 745 |

### Usage trend (last 3 months vs prior)

| segment | customers | churn_rate | avg_monthly_charge |
|---|---|---|---|
| Drop >30% | 348 | 36.2% | 702 |
| Drop 10-30% | 1474 | 25.8% | 715 |
| Stable | 2053 | 16.8% | 721 |
| Growth >10% | 1125 | 9.9% | 728 |

### Latest bill change

| segment | customers | churn_rate | avg_monthly_charge |
|---|---|---|---|
| No increase | 1777 | 15.9% | 708 |
| Up to 15% | 2580 | 19.4% | 733 |
| Spike >15% | 643 | 28.0% | 695 |

### Late payments in last 6 months

| segment | customers | churn_rate | avg_monthly_charge |
|---|---|---|---|
| 0 | 3310 | 17.1% | 718 |
| 1 | 1301 | 21.8% | 730 |
| 2+ | 389 | 28.8% | 696 |

## 10 highest-risk active customers

| customer_id | plan | tenure_months | monthly_charges | churn_probability | top_reasons |
|---|---|---|---|---|---|
| C104712 | Standard | 29 | 633 | 89% | Unresolved complaint; Usage dropped sharply |
| C101176 | Standard | 15 | 749 | 83% | Unresolved complaint |
| C100192 | Basic | 5 | 400 | 82% | Paying a lot per GB used; New customer, no contract |
| C103050 | Family | 7 | 861 | 81% | Unresolved complaint; Recent bill increase |
| C103974 | Basic | 33 | 406 | 80% | Unresolved complaint; Usage dropped sharply; Repeated late payments |
| C103450 | Premium | 2 | 1,115 | 79% | Unresolved complaint; New customer, no contract |
| C104046 | Basic | 17 | 448 | 79% | Unresolved complaint |
| C104831 | Basic | 20 | 352 | 78% | Unresolved complaint; Usage dropped sharply; Repeated complaints |
| C103630 | Basic | 20 | 368 | 77% | Usage dropped sharply; Paying a lot per GB used |
| C103097 | Standard | 7 | 745 | 77% | Unresolved complaint; Repeated complaints; Recent bill increase |
