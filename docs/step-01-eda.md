# Step 1 — Exploratory Data Analysis

**Date:** 2026-09-01
**Dataset:** `Telco-Customer-Churn.csv` (7043 rows × 21 columns)
**Status:** Complete
**Method note:** performed with stdlib `csv` + numpy/scipy only, because pandas/scikit-learn were not installed at the time. No packages were installed during this step. All figures below are computed from the full dataset, not a sample.

---

## 1. The assignment

From `enunciado.ipynb` — *"Hackathon 1 — Desafio de Classificação"*. 90 minutes, teams of three, no final presentation ("the goal is to practice under real time pressure").

| Item | Requirement |
|---|---|
| **Objective** | Build the best possible **binary classification** model to predict `Churn` (Yes/No) within the time available |
| **Target** | `Churn` — Yes/No |
| **Metric** | Not fixed. Must suit the imbalance: *"think precision, recall, F1 or AUC — not just accuracy"* |
| **Algorithms** | Any from Module 3: KNN, Naive Bayes, Logistic Regression, SVM, Trees, Ensemble |
| **Minimum** | Train **at least 2 different models** and compare them |
| **Deliverable** | A notebook that runs start-to-finish + a synthesis cell |

The setup cell pre-imports `classification_report`, `confusion_matrix`, `f1_score`, `roc_auc_score` — so **F1 and ROC-AUC are the intended candidates**.

### The synthesis cell (final deliverable)

Four fields must be filled in:

1. Best model
2. Chosen metric and the value obtained
3. Main technical decision and why
4. What you would try next with more time

### Explicit hint planted in the enunciado

> **Atenção:** the `TotalCharges` column has text type, not numeric — there is a reason for this. Investigate before converting directly.
>
> *(code cell)* Look for rows where the value is not a valid number.

This is a deliberate trap. It is solved in §4 below.

### Suggested time budget

| Phase | Minutes |
|---|---|
| Load + explore | 10 |
| Cleaning (missing values, encoding, `customerID`) | 20–25 |
| Train ≥2 models | 30–35 |
| Evaluate + compare | 15 |
| Review + synthesis | 10 |

### Discrepancy found

The enunciado's prose says the file lives in `data/`. It does not — it sits at the repo root, and the notebook's actual code cell reads `pd.read_csv('Telco-Customer-Churn.csv')`. **The code is correct; the prose is wrong.** Do not "fix" this by moving the file — that would break the notebook.

---

## 2. Environment

| Package | Status |
|---|---|
| python3 | **3.12.3** |
| numpy | **2.5.1** |
| scipy | **1.18.0** |
| pandas | **MISSING** |
| scikit-learn | **MISSING** |
| matplotlib | **MISSING** |
| seaborn | **MISSING** |

The notebook's setup cell imports pandas, matplotlib, seaborn and sklearn, so **as of this step the notebook cannot execute past cell 7**. Installing these four is a prerequisite for Step 2.

---

## 3. Dataset at a glance

**7043 rows × 21 columns.** No header anomalies, no encoding issues.

| Column | Type | Cardinality | Note |
|---|---|---|---|
| `customerID` | text | 7043 (unique) | identifier — drop |
| `gender` | cat | 2 | Male 50.48% / Female 49.52% |
| `SeniorCitizen` | int 0/1 | 2 | 16.21% seniors (1142) |
| `Partner` | cat | 2 | Yes 48.30% |
| `Dependents` | cat | 2 | Yes 29.96% |
| `tenure` | int | 73 | months, 0–72 |
| `PhoneService` | cat | 2 | Yes 90.32% |
| `MultipleLines` | cat | 3 | incl. "No phone service" (682) |
| `InternetService` | cat | 3 | Fiber 43.96% / DSL 34.37% / No 21.67% |
| `OnlineSecurity` … `StreamingMovies` (6 cols) | cat | 3 each | each incl. "No internet service" (1526) |
| `Contract` | cat | 3 | M2M 55.02% / 2yr 24.07% / 1yr 20.91% |
| `PaperlessBilling` | cat | 2 | Yes 59.22% |
| `PaymentMethod` | cat | 4 | Electronic check 33.58% largest |
| `MonthlyCharges` | float | 1585 | 18.25–118.75 |
| `TotalCharges` | **text** | 6531 | 11 blanks — see §4 |
| `Churn` | cat (target) | 2 | Yes 26.54% |

### Numeric summary

| col | n | miss | min | q25 | median | mean | q75 | max | std |
|---|---|---|---|---|---|---|---|---|---|
| SeniorCitizen | 7043 | 0 | 0 | 0 | 0 | 0.16 | 0 | 1 | 0.37 |
| tenure | 7043 | 0 | 0 | 9 | 29 | 32.37 | 55 | 72 | 24.56 |
| MonthlyCharges | 7043 | 0 | 18.25 | 35.50 | 70.35 | 64.76 | 89.85 | 118.75 | 30.09 |
| TotalCharges | **7032** | **11** | 18.80 | 401.45 | 1397.47 | 2283.30 | 3794.74 | 8684.80 | 2266.77 |

---

## 4. Data quality issues, ranked by impact

### 4.1 `TotalCharges` — 11 disguised-missing values (the planted trap)

The values are literally `' '` (a **single space**, not an empty string). That is exactly why pandas types the column as text rather than numeric, why `.isnull()` does not catch them on read, and why a naive `.astype(float)` raises `ValueError`.

> **Erratum (corrected in Step 2).** This section originally said pandas types the column as `object`. Under **pandas 3.0.5** — the version subsequently installed — the dtype is **`str`**, because pandas 3.0 promotes the dedicated string dtype to the default. Every consequence above is unchanged, but **a guard written as `dtype == 'object'` silently never fires**. Test values, or use `not pd.api.types.is_numeric_dtype(...)`, as `src/data_prep.py` does. Most published tutorials for this dataset predate pandas 3 and will say `object`.

**The diagnosis the enunciado is fishing for:**

- All 11 rows have **`tenure == 0`** — and those are the *only* 11 rows in the entire dataset with `tenure == 0`. These are brand-new customers who have not been billed yet.
- All 11 have **`Churn == 'No'`** (zero churners).
- Contract: 10 × `Two year`, 1 × `One year`. `MonthlyCharges` ranges 19.70–80.85.

The 11 customerIDs:

```
4472-LVYGI, 3115-CZMZD, 5709-LVOEQ, 4367-NUYAO, 1371-DWPAZ, 7644-OMVMY,
3213-VVOLG, 2520-SGTTA, 2923-ARZLG, 4075-WKNIU, 2775-SEFEE
```

**Correct treatment — impute `TotalCharges = 0`.** They have been billed nothing. Median imputation (1397.47) would be badly wrong for a tenure-0 customer. Dropping the 11 rows is also defensible (0.16% of the data) but silently removes 11 non-churners.

```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
```

### 4.2 Six columns encode "No internet service" redundantly

`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`: the value `'No internet service'` occurs in **exactly the same 1526 rows in all six columns**, and is a **perfect 1:1 match** with `InternetService == 'No'` (verified — identical boolean masks).

Likewise `MultipleLines == 'No phone service'` is a **perfect match** for the 682 rows with `PhoneService == 'No'`.

One-hot encoding as-is creates **7 dummy columns perfectly collinear** with `InternetService_No` / `PhoneService_No`. Collapse `'No internet service'` → `'No'` and `'No phone service'` → `'No'` before encoding. This is the highest-value cleaning step after `TotalCharges`.

### 4.3 `TotalCharges ≈ tenure × MonthlyCharges` — near-deterministic

| Check | Value |
|---|---|
| `corr(TotalCharges, tenure × MonthlyCharges)` | **0.9996** |
| OLS `TotalCharges ~ a + b·(tenure × MonthlyCharges)` | a = −0.93, b = 1.0005, **R² = 0.9991** |
| Median ratio `TC / (tenure·MC)` | **1.0000** (p5 = 0.925, p95 = 1.075) |
| Residual | median 0.00, std 67.25 |
| `corr(tenure, TotalCharges)` | 0.826 |
| `corr(MonthlyCharges, TotalCharges)` | 0.651 |

`TotalCharges` carries almost no information beyond the other two. This destabilises logistic-regression coefficients — strong argument for dropping it or replacing it with the residual.

### 4.4 22 rows are duplicates once `customerID` is ignored

20 distinct feature vectors, 42 rows involved. All are `tenure == 1` low-service customers, so they are plausibly genuine distinct customers rather than data errors. The churn label is **not** always consistent across them (e.g. `tenure=1, MonthlyCharges=20.2` appears as both Yes and No). Leave them in — but they set a small irreducible error floor.

### 4.5 No other disguised missing values

Every column was swept for `''`, `' '`, `NA`, `N/A`, `null`, `None`, `?`, `-`, `nan`, `unknown`, and for leading/trailing whitespace. **`TotalCharges` is the only column with any hit.** Zero duplicate `customerID`s, zero fully-duplicate rows, no constant columns (most degenerate is `PhoneService` at 90.32% `Yes`).

### 4.6 `customerID` is 100% unique

Pure identifier — drop it. Leaving it in and label-encoding it would be a serious and easy-to-make mistake.

---

## 5. Target and class balance

| Churn | count | % |
|---|---|---|
| No | 5174 | **73.46%** |
| Yes | 1869 | **26.54%** |

**Ratio 2.77 : 1.** Moderately imbalanced — not severe, but more than enough to make accuracy useless.

> The all-"No" baseline scores **73.46% accuracy** with **0% recall on the class the business cares about.** This is the number the notebook's "stop and think" cell asks for, and the bar any accuracy claim must clear. A model reporting ~75–80% accuracy is barely beating a constant predictor.

**Metric recommendation:** report **ROC-AUC** as the primary comparison metric (threshold-free, robust to this imbalance), and **F1 / recall on the `Yes` class** as the business-facing metric — the retention team's cost function is asymmetric, since a missed churner costs a lost customer while a false positive costs a discount offer.

---

## 6. What predicts churn

Base rate = **26.54%**.

### 6.1 Categoricals, ranked by Cramér's V

| Feature | Cramér's V | chi² p | Verdict |
|---|---|---|---|
| **Contract** | **0.410** | 5.9e-258 | strongest single predictor |
| **OnlineSecurity** | 0.347 | 2.7e-185 | strong |
| **TechSupport** | 0.343 | 1.4e-180 | strong |
| **InternetService** | 0.323 | 9.6e-160 | strong |
| **PaymentMethod** | 0.303 | 3.7e-140 | strong |
| OnlineBackup | 0.292 | 2.1e-131 | moderate |
| DeviceProtection | 0.282 | 5.5e-122 | moderate |
| StreamingMovies | 0.231 | 2.7e-82 | moderate (mostly an internet proxy) |
| StreamingTV | 0.231 | 5.5e-82 | moderate (same) |
| PaperlessBilling | 0.192 | 4.1e-58 | weak-moderate |
| Dependents | 0.164 | 4.9e-43 | weak |
| SeniorCitizen | 0.151 | 1.5e-36 | weak |
| Partner | 0.150 | 2.1e-36 | weak |
| MultipleLines | 0.040 | 3.5e-03 | ~noise |
| **PhoneService** | 0.011 | **0.339** | **useless** |
| **gender** | 0.008 | **0.487** | **useless** |

### 6.2 Numerics

| Feature | r (point-biserial) | Univariate AUC | Cohen's d | Churn=No mean/med | Churn=Yes mean/med |
|---|---|---|---|---|---|
| **tenure** | **−0.352** | **0.740** (inverted) | −0.852 | 37.57 / 38.0 | **17.98 / 10.0** |
| TotalCharges | −0.200 | 0.652 (inverted) | −0.461 | 2555.34 / 1683.60 | 1531.80 / 703.55 |
| MonthlyCharges | +0.193 | 0.621 | +0.446 | 61.27 / 64.43 | **74.44 / 79.65** |
| SeniorCitizen | +0.151 | 0.563 | — | — | — |

`tenure` is the strongest numeric by a wide margin.

### 6.3 High-signal segments

Every segment below has n ≥ 200 — none of this is small-sample noise.

| Feature = value | n | churn rate | Δ vs base | lift |
|---|---|---|---|---|
| PaymentMethod = **Electronic check** | 2365 | **45.29%** | +18.75 pp | 1.71× |
| Contract = **Month-to-month** | 3875 | **42.71%** | +16.17 pp | 1.61× |
| InternetService = **Fiber optic** | 3096 | **41.89%** | +15.36 pp | 1.58× |
| OnlineSecurity = No | 3498 | 41.77% | +15.23 pp | 1.57× |
| SeniorCitizen = 1 | 1142 | 41.68% | +15.14 pp | 1.57× |
| TechSupport = No | 3473 | 41.64% | +15.10 pp | 1.57× |
| PaperlessBilling = Yes | 4171 | 33.57% | +7.03 pp | 1.26× |
| Contract = **One year** | 1473 | **11.27%** | −15.27 pp | 0.42× |
| InternetService = **No** | 1526 | **7.40%** | −19.13 pp | 0.28× |
| Contract = **Two year** | 1695 | **2.83%** | −23.71 pp | **0.11×** |

`Contract` spans **39.88 pp** from Two year (2.83%) to Month-to-month (42.71%) — the widest spread of any feature.

### 6.4 Tenure is monotone and very strong

| tenure (months) | n | churn rate |
|---|---|---|
| 0 | 11 | 0.00% |
| 1–3 | 1051 | **56.80%** |
| 4–6 | 419 | 44.63% |
| 7–12 | 705 | 35.89% |
| 13–24 | 1024 | 28.71% |
| 25–36 | 832 | 21.63% |
| 37–48 | 762 | 19.03% |
| 49–60 | 832 | 14.42% |
| 61–72 | 1407 | **6.61%** |

### 6.5 The `Contract × InternetService` interaction

| Contract / Internet | n | churn rate |
|---|---|---|
| Month-to-month / Fiber optic | 2128 | **54.61%** |
| Month-to-month / DSL | 1223 | 32.22% |
| One year / Fiber optic | 539 | 19.29% |
| Month-to-month / No | 524 | 18.89% |
| One year / DSL | 570 | 9.30% |
| Two year / Fiber optic | 429 | 7.23% |
| Two year / DSL | 628 | 1.91% |
| Two year / No | 638 | **0.78%** |

A **70× spread** between the best and worst cell. Tree and ensemble models find this natively; logistic regression needs the explicit interaction term or it underperforms here.

### 6.6 Add-on count is protective — except at 1

| add-ons (of 6) | n | churn rate |
|---|---|---|
| 0 | 2219 | 21.41% |
| **1** | 966 | **45.76%** ← anomaly |
| 2 | 1033 | 35.82% |
| 3 | 1118 | 27.37% |
| 4 | 852 | 22.30% |
| 5 | 571 | 12.43% |
| 6 | 284 | 5.28% |

The spike at 1 is a **composition artifact, not a real U-shape**. The 0-addon bucket is dominated by the 1526 no-internet customers (7.40% churn), while the 1-addon bucket is dominated by fiber customers whose single add-on is a *streaming* service:

- `StreamingMovies` only + Fiber = **68.84% churn** (n=138)
- `StreamingTV` only + Fiber = **66.92% churn** (n=130)

Those are the highest-churn segments in the whole dataset. A hand-built `n_services` feature would encode this misleadingly — let the model see the individual columns.

### 6.7 `MonthlyCharges` is non-monotone — important

By decile: D1 8.66% → D2 9.38% → rising to D9 **40.91%** → then D10 (102.60–118.75) **drops back to 24.68%**.

Reason: D10 is 40% two-year contracts with a mean of 4.80 add-ons, versus D9 which is 62% month-to-month with 3.20 add-ons. **The most expensive customers are loyal bundlers.**

Plain logistic regression fits a monotone effect and gets the top decile backwards. **This alone argues for tree-based models**, or for an explicit interaction term. Note also that `MonthlyCharges` is nearly a proxy for `InternetService` (mean 91.50 fiber / 58.10 DSL / 21.08 none — near-disjoint ranges).

### 6.8 Useless features

| Feature | Churn spread | p | Action |
|---|---|---|---|
| `gender` | 26.92% (F) vs 26.16% (M) — 0.76 pp | **0.487** | Drop — pure noise |
| `PhoneService` | 26.71% vs 24.93% — 1.78 pp | **0.339** | Drop — also 90.32% one value |
| `MultipleLines` | 3.68 pp, V = 0.040 | 3.5e-03 | Marginal — keep only for trees |

---

## 7. Modeling gotchas and recommendations

### 7.1 Leakage — low risk, two things to watch

1. No time-ordering column, no post-cancellation fields. The feature set is a clean customer snapshot. **No obvious target leak.**
2. The subtle one: `TotalCharges` and `tenure` are **outcome-correlated by construction** — a customer who churned early necessarily has low tenure and low total charges. This is legitimate signal for a "who will churn" model, but it **cannot be interpreted causally**, and if the business use case is "score a customer at signup", tenure is not available. This is exactly the kind of point the synthesis cell's "main technical decision" field is asking for.
3. Do **not** fit the imputer/scaler/encoder before the train-test split. With only 11 missing values it barely matters numerically, but a leaking pipeline is the classic hackathon deduction. Use `sklearn.pipeline.Pipeline` + `ColumnTransformer`.

### 7.2 Collinearity to resolve

- `TotalCharges` is 99.91% explained by `tenure × MonthlyCharges` and correlates 0.826 with `tenure`. For **LogReg / SVM / KNN: drop it**, or replace it with `TotalCharges − tenure×MonthlyCharges` (median 0.00, std 67.25 — captures plan changes and promos, genuinely new information). For trees/ensembles, leaving it in is harmless but adds nothing.
- `Contract` and `tenure` are entangled (mean tenure 18.04 M2M / 42.04 one-year / 56.74 two-year) but both carry independent signal — **keep both**.

### 7.3 Type conversions required

| Column | Action |
|---|---|
| `TotalCharges` | `pd.to_numeric(..., errors='coerce').fillna(0)` — `' '` is not caught by `isnull()`; `.astype(float)` raises |
| `Churn` | map `{'No': 0, 'Yes': 1}` |
| `SeniorCitizen` | already 0/1 int — do **not** one-hot it |
| all other categoricals | one-hot with `drop_first=True`, *after* collapsing the "No internet/phone service" levels |

### 7.4 Columns to drop

`customerID` (7043 unique) · `gender` (p=0.487) · `PhoneService` (p=0.339) · `TotalCharges` (linear models only)

### 7.5 Expected shape and performance

- **Feature count after clean encoding:** ~7 binary + `Contract`(3) + `InternetService`(3) + `PaymentMethod`(4) + 3 numerics ≈ **25–30 columns**. Small and dense — every allowed algorithm trains in seconds on 7043 rows.
- **Scale the numerics** for KNN / SVM / LogReg. `tenure` 0–72 vs `TotalCharges` 0–8685 will otherwise let `TotalCharges` dominate any distance metric.
- **Expected ceiling: ROC-AUC 0.84–0.85, F1(Yes) ≈ 0.60–0.63.** If a model reports **AUC > 0.90, something has leaked.** If it reports 79% accuracy, point at the 73.46% baseline.
- Stratify the split on `Churn`; use `class_weight='balanced'`; tune the decision threshold rather than using the default 0.5.

---

## 8. Suggested split of work for the trio

The enunciado explicitly asks the team to divide the work up front.

| Person | Owns |
|---|---|
| **A** | Cleaning: the `TotalCharges` fix, the "No internet service" collapse, and the `Pipeline` / `ColumnTransformer` scaffold |
| **B** | Linear / interpretable model — LogReg with `class_weight='balanced'`, giving the retention team readable drivers |
| **C** | Ensemble — RandomForest / GradientBoosting, plus threshold tuning and the final comparison table |

---

## 9. Open items carried into Step 2

- [ ] Install `pandas`, `scikit-learn`, `matplotlib`, `seaborn` — the notebook cannot run without them
- [ ] Decide finally between dropping `TotalCharges` vs. using the residual
- [ ] Confirm the primary metric with the team (recommendation: ROC-AUC primary, F1-Yes reported alongside)
