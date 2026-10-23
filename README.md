# Credit Card Fraud Detection with Concept Drift 2026

**120,000 transactions · 73 features · 5 fraud types · 4 drift phases · Jan-Jun 2026**

The most research-complete fraud detection dataset on Kaggle. Designed for multiclass fraud type classification under gradual concept drift, with delayed labels, pseudo-labelling support, and post-transaction dispute outcomes.

---

## Quick Stats

| Property | Value |
|---|---|
| Rows | 120,000 |
| Columns | 73 |
| Fraud rate | 9.49% |
| Fraud types | 5 (CNP, ATO, SYN-ID, CP, FF) |
| Drift phases | 4 (Jan-Feb stable, Mar early, Apr-May peak, Jun stabilising) |
| Unique customers | 29,441 |
| Unique merchants | 5,000 |
| Date range | 2026-01-01 to 2026-06-30 |
| File size | 42.3 MB |
| Missing values | 351,921 (intentional -- dispute NaN when no dispute filed) |

---

## Files

```
credit_card_fraud_drift_2026.csv    # Main dataset (120K rows, 73 columns)
README.md                           # This file
```

---

## Target Variables

| Target | Type | Use |
|---|---|---|
| `transaction_time_target` | 6-class | Primary multiclass target. Filter on `transaction_time_eligible==1` before use. |
| `is_fraud` | Binary | Binary detection baseline. |
| `fraud_amount_lost` | Float | Regression -- predict USD lost per transaction. |
| `dispute_adjudicated_friendly_fraud` | Binary | Post-transaction only -- Friendly-Fraud outcome. |

---

## Drift Phase Structure

| Phase | Months | Rows | Drift | Labels |
|---|---|---|---|---|
| 1 Stable | Jan, Feb | 39,133 | 0.00 | 100% |
| 2 Early | Mar | 20,445 | 0.30 | 100% |
| 3 Peak | Apr, May | 40,586 | 0.73 avg | 97.4% (21d delay) |
| 4 Stabilising | Jun | 19,836 | 0.50 | 98.9% |

---

## Recommended Experimental Protocol

```python
import pandas as pd

df = pd.read_csv('credit_card_fraud_drift_2026.csv')
df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'])

# Transaction-time eligible rows only
eligible = df[df.transaction_time_eligible == 1]

# Stable training period
train = eligible[eligible.drift_phase == 1]

# Drift test period
test_drift = eligible[eligible.drift_phase == 3]

print(f"Train: {len(train):,} | Test (drift): {len(test_drift):,}")
print(f"Train fraud rate: {train.is_fraud.mean():.4f}")
print(f"Test fraud rate:  {test_drift.is_fraud.mean():.4f}")
```

---

## Fraud-Type Multiclass Benchmark

```python
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import xgboost as xgb

# Fraud-only rows for fraud type classification
fraud_train = train[train.is_fraud == 1].copy()
fraud_test  = test_drift[test_drift.is_fraud == 1].copy()

drop_cols = ['transaction_id','customer_id','card_id','merchant_id',
             'transaction_datetime','fraud_type','fraud_amount_lost',
             'drift_phase_name','transaction_time_target','is_fraud',
             'dispute_event_datetime','adjudication_datetime','dispute_regime',
             'post_transaction_outcome','transaction_time_eligible',
             'dispute_filed','dispute_adjudicated_friendly_fraud',
             'dispute_delay_days','label_available','label_delay_days',
             'pseudo_label','amount_last_1h','amount_last_24h']

le = LabelEncoder()
fraud_train['target'] = le.fit_transform(fraud_train['fraud_type'])
fraud_test['target']  = le.transform(fraud_test['fraud_type'])

feat_cols = [c for c in fraud_train.columns
             if c not in drop_cols + ['target','fraud_type','is_fraud']]

# Encode categoricals
for col in fraud_train[feat_cols].select_dtypes('object').columns:
    fraud_train[col] = LabelEncoder().fit_transform(fraud_train[col].astype(str))
    fraud_test[col]  = LabelEncoder().fit_transform(fraud_test[col].astype(str))

model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05,
    objective='multi:softprob', num_class=len(le.classes_),
    random_state=42, n_jobs=-1, verbosity=0
)
model.fit(fraud_train[feat_cols], fraud_train['target'])
preds = model.predict(fraud_test[feat_cols])

print(classification_report(fraud_test['target'], preds,
      target_names=le.classes_))
# Expected: macro F1 ~0.79 (lower than stable period -- drift impact visible)
```

---

## Pseudo-Label Adaptation Example

```python
# Use pseudo labels during Phase 3 label delay window
phase3 = eligible[eligible.drift_phase == 3].copy()

# Rows with immediate labels
labeled   = phase3[phase3.label_available == 1]

# Rows with delayed labels -- use pseudo_label for adaptation
unlabeled = phase3[(phase3.label_available == 0) &
                   (phase3.pseudo_label != -1)].copy()
unlabeled['is_fraud'] = unlabeled['pseudo_label']

print(f"Phase 3 labeled:    {len(labeled):,}")
print(f"Phase 3 unlabeled:  {len(unlabeled):,} (pseudo labels available)")
```

---

## Drift Monitoring Example

```python
import matplotlib.pyplot as plt

# Track SYN-ID camouflage score over time
syn_drift = df[df.fraud_type == 'Synthetic-Identity'].groupby(
    'month')['synthetic_id_camouflage_score'].mean()

plt.figure(figsize=(10, 4))
plt.plot(syn_drift.index, syn_drift.values, marker='o', linewidth=2.5)
plt.axvspan(1, 2.5, alpha=0.1, color='green', label='Phase 1 Stable')
plt.axvspan(3, 3.5, alpha=0.1, color='orange', label='Phase 2 Early drift')
plt.axvspan(4, 5.5, alpha=0.1, color='red', label='Phase 3 Peak drift')
plt.axvspan(6, 6.5, alpha=0.1, color='yellow', label='Phase 4 Stabilising')
plt.xlabel('Month')
plt.ylabel('SYN-ID Camouflage Score')
plt.title('Synthetic-Identity Drift Intensity Over Time')
plt.legend()
plt.show()
```

---

## Key Research Questions

1. Which features maintain the most discriminative power across all 4 drift phases?
2. How much does XGBoost macro F1 degrade from Phase 1 to Phase 3?
3. Does pseudo-labelling during Phase 3 improve model adaptation vs no adaptation?
4. Can drift be detected early using concept_drift_score and population_shift_index before F1 drops?
5. Which fraud type suffers the largest F1 degradation under drift?

---

## Performance Benchmarks

| Experiment | Macro F1 |
|---|---|
| XGBoost -- Phase 1 train / Phase 2 test (stable) | 0.86 |
| XGBoost -- Phase 1 train / Phase 3 test (peak drift) | 0.79 |
| XGBoost -- Phase 1 train / Phase 4 test (stabilising) | 0.81 |
| Logistic Regression -- Phase 1 train / Phase 2 test | 0.74 |

---

## Leakage Warning

The following columns must **never** be used as transaction-time model features:

- `dispute_filed` -- only known after dispute is filed
- `post_transaction_outcome` -- only known after investigation
- `dispute_adjudicated_friendly_fraud` -- only known after adjudication
- `dispute_event_datetime` and `adjudication_datetime` -- post-transaction timestamps
- `dispute_regime` -- assigned after dispute is filed
- `prior_dispute_ratio_90d` -- audit calculation window before use

---

## License

CC0: Public Domain. Free for academic, commercial, and personal use.

---

## Author

**Mohan Krishna Thalla**
- Kaggle: https://www.kaggle.com/mohankrishnathalla
- Dataset: https://www.kaggle.com/datasets/mohankrishnathalla/credit-card-fraud-drift-2026

---

**Version:** 1.0 | **Updated:** October 2026 | **Records:** 120,000 | **License:** CC0
