# Step 2 — Data Treatment

**Date:** 2026-09-02  
**Implementation:** `codigo.ipynb`  
**Dataset:** `Telco-Customer-Churn.csv` (7,043 rows × 21 columns)  
**Status:** Complete

---

## 1. Scope

This step implements the data-loading and treatment requirements from `enunciado.ipynb`: investigate `TotalCharges`, handle missing/invalid values, remove the identifier, encode categoricals, and prepare a train/test split without leakage. Model training and comparison remain Steps 3–4.

## 2. Portable data path

The assignment prose says `data/Telco-Customer-Churn.csv`, while its executable example and this repository use the root-level file `Telco-Customer-Churn.csv`. The notebook searches from the current working directory toward the repository root, identifies the project by `enunciado.ipynb`, and tests these relative paths in order:

1. `Telco-Customer-Churn.csv`
2. `data/Telco-Customer-Churn.csv`

No absolute machine path is embedded and the vendored CSV is not moved.

## 3. Input validation

Before cleaning, the notebook checks the complete 21-column schema and requires `customerID` to be unique. The observed input is:

| Check | Result |
|---|---:|
| Rows | 7,043 |
| Columns | 21 |
| Unique `customerID` values | 7,043 |
| Duplicate IDs | 0 |
| `Churn = No` | 5,174 (73.46%) |
| `Churn = Yes` | 1,869 (26.54%) |
| Majority-class accuracy baseline | 73.46% |

The baseline confirms that accuracy alone is not a defensible primary metric.

## 4. Deterministic cleaning

### `TotalCharges`

`pd.to_numeric(..., errors='coerce')` identifies 11 invalid strings. All 11 are a literal single space in the raw CSV, all have `tenure == 0`, and all are new customers that have not yet been billed. The notebook verifies this condition before assigning `0.0`. It raises an error instead of silently applying the rule if a future invalid value has nonzero tenure.

This preserves all rows. Median imputation would incorrectly assign roughly 1,397.48 in historical charges to customers who have not been billed, and dropping the records would remove 11 known non-churners.

### Target and service labels

- `Churn` is mapped from `No`/`Yes` to `0`/`1` and validated as binary.
- `No internet service` is collapsed to `No` in the six internet add-on columns.
- `No phone service` is collapsed to `No` in `MultipleLines`.

Those service strings are structural states, not missing values. Consolidating them removes redundant one-hot columns while `InternetService` carries the no-internet state.

### Feature removal

| Feature | Decision | Reason |
|---|---|---|
| `customerID` | Drop from `X` | Pure unique identifier |
| `gender` | Drop from `X` | No measured association with churn (`p = 0.487`) |
| `PhoneService` | Drop from `X` | No measured association with churn (`p = 0.339`) |
| `TotalCharges` | Keep for now | Explicitly cleaned; model-specific removal can be tested for linear estimators because it is almost derived from tenure and monthly charges |

## 5. Split and learned transformations

The cleaned data is split 80/20 with `random_state=42` and `stratify=y`:

| Split | Rows | Share |
|---|---:|---:|
| Train | 5,634 | 80.00% |
| Test | 1,409 | 20.00% |

After the split, an unfitted `ColumnTransformer` is built:

- numeric columns: median imputation followed by `StandardScaler`;
- categorical columns: most-frequent imputation followed by `OneHotEncoder(drop='first', handle_unknown='ignore')`;
- `SeniorCitizen` remains numeric 0/1 rather than being one-hot encoded.

The transformer used for validation is fitted only on `X_train` and then applied to `X_test`. Each later model should receive a fresh transformer inside its own `Pipeline`, particularly when cross-validation is introduced.

## 6. Execution evidence

The preparation section was executed with the existing Anaconda Python 3.13.9 kernel, pandas 2.3.3, and scikit-learn 1.7.2. No packages were installed; the full validation was performed in memory independently of any outputs a user may save while exploring the notebook.

| Validation | Result |
|---|---:|
| Code cells executed | 9 / 9 |
| Error outputs | 0 |
| Raw model features | 17 |
| Encoded features | 21 |
| Train matrix | 5,634 × 21 |
| Test matrix | 1,409 × 21 |
| NaN or infinite values after preprocessing | 0 |

The prepared objects are `df_raw`, `df`, `X_train`, `X_test`, `y_train`, `y_test`, and the still-unfitted `preprocessador` intended for model pipelines.

## 7. Next step

Train at least two allowed classifiers on the same split, using a fresh preprocessing pipeline for each. Compare ROC-AUC as the primary metric and report F1/recall for the positive churn class alongside confusion matrices.
