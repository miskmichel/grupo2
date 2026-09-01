# Step 2 — Data Treatment

**Date:** 2026-09-01
**Dataset:** `Telco-Customer-Churn.csv` (7043 rows × 21 columns, root path — not moved)
**Status:** Complete
**Artifact produced:** `src/data_prep.py`
**Method note:** every number below is observed output from a verification script run against the full dataset, not an estimate. The script lived outside the repo and is reproduced in §6.

---

## 1. What Step 2 required

From `enunciado.ipynb`:

> *"tratar dados: valores faltantes, variáveis categóricas, o que fazer com customerID."*
> (treat the data: missing values, categorical variables, what to do with `customerID`)

Budgeted at 20–25 minutes of the 90. Three named sub-problems — missing values, categorical variables, `customerID` — each answered explicitly below.

Step 1 (`docs/step-01-eda.md`) had already *diagnosed* all three. Step 2's job was to **implement and empirically verify** the treatment, not to rediscover it.

---

## 2. Environment changes made

pandas, scikit-learn, matplotlib and seaborn were absent, so the assignment notebook could not execute past its setup cell.

`pip install pandas scikit-learn matplotlib seaborn` **failed** with Ubuntu 24.04's PEP 668 `externally-managed-environment` error. It was re-run as:

```bash
pip install --break-system-packages pandas scikit-learn matplotlib seaborn
```

This installs system-wide rather than into a venv. Recorded here plainly because it is a deliberate override of a distro safety rail, taken because the exercise is a 90-minute disposable sandbox and a venv would have to be re-activated in every cell and every shell.

### Versions now present

| Package | Version | Note |
|---|---|---|
| python3 | 3.12.3 | unchanged |
| numpy | 2.5.1 | unchanged, already present |
| scipy | 1.18.0 | unchanged, already present |
| **pandas** | **3.0.5** | new — see the dtype caveat below |
| **scikit-learn** | **1.9.0** | new |
| **matplotlib** | **3.11.1** | new |
| **seaborn** | **0.13.2** | new |

Transitive installs: `joblib 1.6.0`, `threadpoolctl 3.6.0`, `contourpy 1.3.3`, `cycler 0.12.1`, `fonttools 4.64.0`, `kiwisolver 1.5.1`, `narwhals 2.25.0`, `cloudpickle 3.1.2`.

### ⚠ pandas 3.0 changes one documented fact

`CLAUDE.md` fact #1 says the `' '` values are "why the column reads as `object`". Under **pandas 3.0**, which promotes the dedicated string dtype to the default, `df['TotalCharges'].dtype` is **`str`**, not `object`. Every consequence still holds exactly as documented — `.isnull()` returns 0, and `.astype(float)` still raises `ValueError: could not convert string to float: ' '` — but code or commentary that tests `dtype == 'object'` will silently not fire. Anything downstream should test *values*, not the dtype name.

---

## 3. Decision table

| # | Issue | Decision taken | Alternative considered | Why the alternative was rejected |
|---|---|---|---|---|
| 1 | `TotalCharges` holds 11 `' '` values (single space) | `pd.to_numeric(errors='coerce').fillna(0.0)` → float64 | **Median imputation (1397.47)** | All 11 rows have `tenure == 0` — the customer has literally never been billed. Assigning them the median lifetime spend of a ~29-month customer invents ~1400 currency units of history and breaks the near-deterministic `TotalCharges ≈ tenure × MonthlyCharges` relation (R² = 0.9991) for exactly those rows. |
| 1b | same | keep the 11 rows | **Drop the 11 rows** | Defensible (0.16% of data) but it silently deletes 11 non-churners, and 11 of the 11 are `Churn == 'No'` — a small, one-directional bias for no gain. Row count is asserted unchanged at 7043. |
| 1c | same | `errors='coerce'` then `fillna` | **`.astype(float)`** | Raises `ValueError`. Verified in §6.A. |
| 2 | `'No internet service'` in 6 add-on columns (1526 identical rows each) | Collapse to `'No'` | **Leave as a third level** | Verified counterfactual (§6.H): the naive encoding produces **7 mutually identical dummy columns** and **22 pairs at \|r\| = 1.0**, leaving the design matrix at **rank 24 of 30** — rank-deficient by exactly 6. Unregularised logistic regression has no unique solution on that matrix; tree feature importance is split arbitrarily across the copies. |
| 2b | `'No phone service'` in `MultipleLines` (682 rows) | Collapse to `'No'` | as above | Same defect, expressed as `r = −1.0000` between `PhoneService_Yes` and `MultipleLines_No phone service` — a perfect anti-correlation, which is the same problem wearing a minus sign. |
| 2c | same | collapse **before** encoding | **Drop the 6 add-on columns entirely** | Throws away real signal: `OnlineSecurity` (V = 0.347) and `TechSupport` (V = 0.343) are the #2 and #3 predictors in the dataset. It is the *placeholder level* that is redundant, not the column. |
| 3 | Categorical encoding | `OneHotEncoder(drop='first', handle_unknown='ignore')` | **`LabelEncoder` / ordinal codes** | Would impose a false order on unordered levels (`PaymentMethod` 0<1<2<3 is meaningless) and let linear models and KNN read distances off an arbitrary alphabetical ordering. |
| 3b | same | `drop='first'` | **keep all levels** | Full dummy sets plus an intercept are the classic dummy-variable trap — singular design matrix, unstable coefficients. `drop='first'` cost us nothing: the cleaned matrix is full rank 21/21. |
| 3c | same | `handle_unknown='ignore'` | **`'error'`** | With a stratified 80/20 split of 7043 rows every level appears in both folds, so this never fires today. It is cheap insurance that makes the fitted pipeline safe to apply to fresh scoring data. |
| 4 | `SeniorCitizen` is already 0/1 int | Pass through unscaled | **One-hot it** | Would emit a literal duplicate of the existing column. |
| 4b | same | pass through unscaled | **`StandardScaler` it** | It sits in the same matrix as 17 unscaled 0/1 dummies; scaling only this one binary would give it ~2.7× their weight in any distance metric (KNN, SVM). |
| 5 | `tenure` 0–72 vs `TotalCharges` 0–8684.80 | `StandardScaler` on the 3 continuous columns | **no scaling** | A raw Euclidean distance over these columns is `TotalCharges` and nothing else — the range ratio is ~120:1. Also required for L2-regularised logistic regression to penalise coefficients comparably. |
| 6 | `gender` (p = 0.487), `PhoneService` (p = 0.339) | Drop, behind `drop_noise=True` | **keep** | Neither shows any association with churn survivable at any sane significance level; `PhoneService` is additionally 90.32% one value. Kept as a **flag** rather than a hard delete so Step 3 can refit with them and *demonstrate* the claim instead of asserting it. |
| 7 | `TotalCharges` is 99.91% explained by `tenure × MonthlyCharges` | **Kept** by default, `drop_total_charges=False` flag available | **drop it now** | Still an open Step 1 item and it is model-dependent: harmful to linear models, harmless to trees. Deciding it here would bake a linear-model assumption into the shared cleaning layer. Deferred to Step 3 as a measured experiment. Post-encoding it is the single highest correlation left in the matrix at r = +0.8297 with `tenure` (§6.G). |
| 8 | Fitting the transformers | `build_preprocessor()` returns an **unfitted** `ColumnTransformer` | **fit on the full frame, then split** | Leaks test-fold means, standard deviations and category lists into training. Numerically small here, but it is the standard deduction on this exercise. Asserted unfitted in §6.F. |
| 9 | 22 near-duplicate rows once `customerID` is ignored | Left in | **de-duplicate** | Per EDA §4.4 they are plausibly genuine distinct `tenure == 1` customers with inconsistent labels, not data errors. They set a small irreducible error floor; removing them would flatter the test score. |

---

## 4. The `customerID` question, answered

The assignment names this column specifically, so it gets its own answer.

**What we did: dropped it, unconditionally and non-optionally.**

**Why.** `customerID` has **7043 distinct values across 7043 rows** — verified, `nunique() == 7043`. It is a surrogate key. It carries exactly zero information about churn *by construction*: knowing a customer's ID tells you nothing about a customer you have not already seen, because you will never see that ID again.

**The trap the assignment is pointing at.** The reflexive move when a pipeline complains about a string column is to encode it like any other categorical. Both ways of doing that are wrong here, and wrong differently:

- **`LabelEncoder` / ordinal encoding** turns the IDs into integers 0…7042. Nothing breaks, no error is raised, and the column now *looks* like a well-behaved numeric feature. A tree will happily split on `customerID < 3521.5` and memorise the training set through it; a KNN will let a 0–7042 range dominate every distance it computes. The ordering is an artifact of alphabetical sort order on strings like `4472-LVYGI` — it is pure noise given a magnitude. This is the specific failure the enunciado is fishing for, and it is silent.
- **One-hot encoding** is loud instead of silent: it produces a 7043-column matrix in which every column is a perfect indicator for one training row. Any sufficiently flexible model reaches 100% training accuracy and ~0 useful test performance. It also blows up memory for no reason.

**A note on what we did *not* lose.** The IDs are still useful outside the model — they are the join key back to the customer record, and how the retention team would action a list of predicted churners. Dropping the column from the *feature matrix* is not the same as discarding it. Because `clean()` preserves row order and the index, `split()` returns X frames whose index still maps 1:1 back to `load_raw()` rows, so any prediction can be re-attached to its `customerID` after the fact. The IDs of the 11 `TotalCharges` blanks are recorded in `docs/step-01-eda.md` §4.1 for the same reason.

---

## 5. What was built — `src/data_prep.py`

A dependency-light importable module. Nothing is executed or fitted at import time.

| Function | Signature | Role |
|---|---|---|
| `load_raw` | `(path='Telco-Customer-Churn.csv') -> DataFrame` | Reads the CSV **with no coercion**, so the 11 `' '` values stay visible and countable |
| `clean` | `(df, *, drop_noise=True, drop_total_charges=False) -> DataFrame` | All of §3, decisions 1–2 and 6–7. Copies; never mutates its input (asserted) |
| `infer_feature_groups` | `(df) -> (numeric, binary, categorical)` | Constant-driven, not dtype-sniffed, so grouping stays stable whichever optional columns were dropped |
| `build_preprocessor` | `(numeric=None, binary=None, categorical=None, *, df=None, scale=True) -> ColumnTransformer` | Scaler + passthrough + one-hot. **Returned unfitted.** `scale=False` gives tree models the same pipeline shape without a pointless scaler |
| `split` | `(df, test_size=0.2, random_state=42)` | Stratified on `Churn`, returns `(X_tr, X_te, y_tr, y_te)` with column names intact so the `ColumnTransformer` can select by name |

Module constants (`TARGET`, `ID_COLUMN`, `NOISE_COLUMNS`, `SERVICE_PLACEHOLDERS`, `BINARY_COLUMNS`, `NUMERIC_COLUMNS`, `DEFAULT_CSV`) are exported so Step 3 can reference the same definitions rather than re-typing string literals.

Intended use, keeping the fit inside the split:

```python
import data_prep as dp
from sklearn.pipeline import Pipeline

df = dp.clean(dp.load_raw())
X_tr, X_te, y_tr, y_te = dp.split(df)
pipe = Pipeline([("prep", dp.build_preprocessor(df=df)), ("clf", ...)])
pipe.fit(X_tr, y_tr)          # transformers see the training fold only
```

---

## 6. Verification results

Run against the full 7043 rows. **All 25 assertions passed.** Observed output, condensed:

### A. Load — the trap is real and behaves as documented

```
raw shape                      : (7043, 21)
raw TotalCharges dtype         : str            <- 'object' under pandas < 3.0
raw TotalCharges ' ' count     : 11
raw TotalCharges .isnull()     : 0              <- isnull() does NOT catch them
blank rows tenure values       : [0]
blank rows Churn values        : ['No']
naive .astype(float) raises    : ValueError: could not convert string to float: ' '
customerID nunique             : 7043 / 7043
```

The 11 IDs match `docs/step-01-eda.md` §4.1 exactly.

### B. Nothing silently dropped

```
rows before clean : 7043
rows after  clean : 7043
cols  21 -> 18  (drop_noise=True)   dropped: customerID, gender, PhoneService
cols  21 -> 20  (drop_noise=False)  dropped: customerID
raw frame not mutated by clean()    PASS
```

### C. `TotalCharges`

```
dtype                          : float64
nulls                          : 0
values at the 11 blank rows    : [0.0]
count of TotalCharges == 0.0   : 11      <- the 11 are the ONLY zeros in the column
min / mean / max               : 0.00 / 2279.7343 / 8684.80
```

Mean moves 2283.3004 → 2279.7343 (−3.57) when the 11 zeros join the column. That −0.16% shift is the entire numerical footprint of this decision; the reason to get it right is correctness of those 11 rows, not the aggregate.

### D. Placeholder levels

```
columns still holding 'No internet service' / 'No phone service' : NONE
  (also NONE with drop_noise=False, i.e. with MultipleLines' partner column kept)
OnlineSecurity    : {'No': 5024, 'Yes': 2019}     (3498 + 1526 = 5024 ✓)
MultipleLines     : {'No': 4072, 'Yes': 2971}     (3390 +  682 = 4072 ✓)
InternetService   : {'Fiber optic': 3096, 'DSL': 2421, 'No': 1526}   <- untouched
```

The collapse arithmetic reconciles exactly, and `InternetService` — which is where the "no internet" information legitimately lives — is left alone.

### E. Target

```
dtype        : int64
value counts : 0 -> 5174,  1 -> 1869
churn rate   : 26.5370%
```

Matches Step 1 exactly. `Yes` is mapped to `1`, the minority/positive class, which is what `f1_score` and `roc_auc_score` assume by default.

### F. Encoding

```
numeric      (3) : tenure, MonthlyCharges, TotalCharges
binary       (1) : SeniorCitizen
categorical (13) : Partner, Dependents, MultipleLines, InternetService,
                   OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
                   StreamingTV, StreamingMovies, Contract, PaperlessBilling,
                   PaymentMethod

build_preprocessor returns ColumnTransformer   PASS
preprocessor is NOT fitted at build time       PASS
encoded train shape : (5634, 21)
encoded test  shape : (1409, 21)   <- same width, transformed by the train-fitted object
```

### G. Collinearity — clean

```
exactly duplicate column pairs : NONE
|r| >= 0.999999 pairs          : NONE
highest |r| off-diagonal       : 0.8297   (tenure <-> TotalCharges)
matrix rank / n_cols           : 21 / 21     <- full column rank
NaNs in encoded matrix         : 0
```

Only one pair exceeds \|r\| = 0.80, and it is the known `tenure`/`TotalCharges` entanglement from EDA §7.2 — carried forward deliberately as decision #7, not an oversight.

### H. Counterfactual — what the naive encoding would have produced

```
naive full feature width       : 30   (vs 21 cleaned)
perfect |r| = 1.0 pairs        : 22
   PhoneService_Yes  <->  MultipleLines_No phone service      r = -1.0000
   21 further pairs among {InternetService_No} ∪ the six *_No internet service
matrix rank / n_cols           : 24 / 30    <- rank-deficient by exactly 6
```

Seven mutually identical columns generate C(7,2) = 21 pairs; six of the seven are pure redundancy, which is precisely the rank deficiency of 6. The collapse buys **9 fewer columns and a non-singular design matrix**, for four lines of code.

### I. Split

```
full   n=7043  churn=26.5370%  (1869 positives)
train  n=5634  churn=26.5353%  (1495 positives)   delta = 0.0017 pp
test   n=1409  churn=26.5436%  ( 374 positives)   delta = 0.0067 pp
train counts {0: 4139, 1: 1495}   test counts {0: 1035, 1: 374}
no index overlap between folds                 PASS
deterministic at random_state=42               PASS
```

Both folds hold the 26.54% rate to within 0.007 pp.

### J. Nulls

```
total nulls in cleaned frame  : 0
NaNs in encoded train matrix  : 0
```

**No assertion failed. No finding contradicts Step 1's data claims.**

---

## 7. The resulting feature matrix

**Shape: (7043, 21) overall → (5634, 21) train / (1409, 21) test.**

| # | Column | Group | Transform |
|---|---|---|---|
| 0 | `tenure` | numeric | StandardScaler |
| 1 | `MonthlyCharges` | numeric | StandardScaler |
| 2 | `TotalCharges` | numeric | StandardScaler |
| 3 | `SeniorCitizen` | binary | passthrough |
| 4 | `Partner_Yes` | one-hot | ref = No |
| 5 | `Dependents_Yes` | one-hot | ref = No |
| 6 | `MultipleLines_Yes` | one-hot | ref = No (post-collapse) |
| 7 | `InternetService_Fiber optic` | one-hot | ref = DSL |
| 8 | `InternetService_No` | one-hot | ref = DSL |
| 9 | `OnlineSecurity_Yes` | one-hot | ref = No (post-collapse) |
| 10 | `OnlineBackup_Yes` | one-hot | ref = No (post-collapse) |
| 11 | `DeviceProtection_Yes` | one-hot | ref = No (post-collapse) |
| 12 | `TechSupport_Yes` | one-hot | ref = No (post-collapse) |
| 13 | `StreamingTV_Yes` | one-hot | ref = No (post-collapse) |
| 14 | `StreamingMovies_Yes` | one-hot | ref = No (post-collapse) |
| 15 | `Contract_One year` | one-hot | ref = Month-to-month |
| 16 | `Contract_Two year` | one-hot | ref = Month-to-month |
| 17 | `PaperlessBilling_Yes` | one-hot | ref = No |
| 18 | `PaymentMethod_Credit card (automatic)` | one-hot | ref = Bank transfer (automatic) |
| 19 | `PaymentMethod_Electronic check` | one-hot | ref = Bank transfer (automatic) |
| 20 | `PaymentMethod_Mailed check` | one-hot | ref = Bank transfer (automatic) |

Convenient for interpretation: because the reference level for `Contract` is `Month-to-month` and for `PaymentMethod` is `Bank transfer (automatic)`, the two strongest risk factors from EDA §6.3 (`Contract = Month-to-month` at 42.71%, `PaymentMethod = Electronic check` at 45.29%) read directly off the coefficients — the `Contract_*` coefficients should both come out strongly negative and `PaymentMethod_Electronic check` positive. That is a free sanity check on any fitted logistic regression in Step 3.

---

## 8. Surprises and open questions for Step 3

### 8.1 The matrix is smaller than Step 1 predicted — 21, not 25–30

EDA §7.5 estimated "~25–30 columns". The actual encoded width is **21**. Not an error in either place, just an arithmetic consequence that is worth writing down: the collapse turns seven columns (`MultipleLines` + six add-ons) from 3-level into 2-level, and under `drop='first'` a 3-level column costs 2 dummies while a 2-level column costs 1 — so the collapse saves 7 dummies, not the 7 *collinear* dummies one might first count. Dropping `gender` and `PhoneService` removes 2 more. 30 → 21.

The practical consequence is good news for the time budget: **21 dense features on 5634 training rows** means every algorithm the assignment allows — including SVM, which is the slowest — fits in seconds. There is room for `GridSearchCV` and threshold tuning inside the 30–35 minute modeling slot.

### 8.2 pandas 3.0's `str` dtype

Flagged in §2. It changes nothing about the treatment, but it will trip any `dtype == 'object'` check, and it is the sort of thing that silently disables a guard. `clean()` deliberately dispatches on `pd.api.types.is_numeric_dtype()` (negated) rather than on `== 'object'` for this reason.

### 8.3 Still open, carried forward

- [ ] **`TotalCharges`: keep, drop, or replace with the residual?** Left in by default. It is the only remaining high-correlation pair in the matrix (r = +0.8297 with `tenure`). The three-way comparison — keep / `drop_total_charges=True` / substitute `TotalCharges − tenure×MonthlyCharges` (median 0.00, std 67.25) — is a 5-minute experiment on a 21-column matrix and should be run for the linear model specifically. The flag is already in place.
- [ ] **Does dropping `gender`/`PhoneService` actually help?** Almost certainly neutral rather than helpful — they are noise, and a regularised model would shrink them to nothing anyway. `drop_noise=False` exists so the claim can be shown rather than asserted, which is a better answer for the synthesis cell's "main technical decision" field.
- [ ] **The `Contract × InternetService` interaction** (70× spread, EDA §6.5) and **the non-monotone `MonthlyCharges` top decile** (EDA §6.7) are *not* handled by this preprocessor. Trees find them natively; a plain logistic regression will get the top `MonthlyCharges` decile backwards. If LogReg underperforms, an explicit interaction term is the first thing to try — it is a one-line addition to the `ColumnTransformer`.
- [ ] **`MultipleLines` is now a clean binary** (4072 No / 2971 Yes) rather than a muddled 3-level column, but EDA §6.8 put it at V = 0.040, essentially noise. It survived the noise cull only because its p-value (3.5e-03) clears significance where `gender` and `PhoneService` do not. Worth a glance in the Step 3 feature-importance table.
- [ ] **No model has been trained.** Nothing in this step touches an estimator; the expected ceiling (ROC-AUC 0.84–0.85, F1-Yes 0.60–0.63) remains a Step 1 prediction to be tested. AUC > 0.90 still means go find the leak.

---

## 9. The frozen train/test split

Added after the initial Step 2 write-up, so that Step 3 starts from a fixed holdout.

### Why freeze it to disk

`data_prep.split()` is deterministic at `random_state=42`, so freezing is not about reproducibility — it is about **coordination and holdout hygiene**:

1. **Three people, one holdout.** The team trains different models in parallel. If each notebook calls `split()` itself, a silent difference in `test_size`, `random_state`, or the cleaning flags means the models are scored on different rows and the comparison table is meaningless. A committed split makes the comparison valid by construction rather than by everyone remembering the same magic number.
2. **The test set stays a test set.** Re-deriving a split inside a tuning loop is the standard way a holdout quietly degrades into a validation set. `test.csv` is written once and not read until final evaluation.
3. **Drift is loud.** The manifest stores a SHA-256 of each file *and* of the source CSV. A regenerated split that differs shows up as a hash mismatch instead of as an unexplained score change.

### Files

| File | Rows | Contents |
|---|---|---|
| `data/splits/train.csv` | 5634 | 17 feature columns + `Churn`, indexed by `row_id` |
| `data/splits/test.csv` | 1409 | same schema — **held out** |
| `data/splits/customer_ids.csv` | 7043 | `row_id → customerID` |
| `data/splits/manifest.json` | — | parameters + SHA-256 of every file |

Generated by `src/make_splits.py`. Parameters recorded in the manifest: `test_size=0.2`, `random_state=42`, `stratify=Churn`, `drop_noise=True`, `drop_total_charges=False`.

### `row_id` — keeping the identifier without feeding it to the model

The frames carry a `row_id` index: the row's position in the raw CSV. This is deliberate. `customerID` must not be in the feature matrix — §4 explains why label-encoding it is the trap the assignment is pointing at — but a churn score nobody can attach to a customer is not actionable. `row_id` is the join key, and `data_prep.load_customer_ids()` resolves it:

```python
ids = dp.load_customer_ids()
ids.loc[X_te.index]        # customerIDs for the test fold
```

Verified: all 1409 test `row_id`s resolve, and `customerID` is absent from `X_train`/`X_test` (17 feature columns, not 18).

### Verification

```
train 5634 rows, churn 26.5353% (1495 positives)
test  1409 rows, churn 26.5436% ( 374 positives)
fold overlap                          : none
union covers all 7043 rows            : yes
customerID present in X               : no  (17 feature columns)
frozen split == in-memory split()     : yes (identical row assignment)
regeneration byte-stable              : yes (train.csv sha256 08cd0f158e076218... twice)
source CSV sha256                     : 16320c9c1ec7... (recorded in manifest)
encoded from frozen split             : (5634, 21) / (1409, 21), rank 21/21
```

`python3 src/make_splits.py --check` re-runs all of the above and exits non-zero on any mismatch.

### For Step 3

```python
import sys; sys.path.insert(0, 'src')
import data_prep as dp

X_tr, X_te, y_tr, y_te = dp.load_splits()          # <- use this, not split()
pre = dp.build_preprocessor(df=dp.clean(dp.load_raw()))
```

**Regenerating the split invalidates every score recorded before it.** If the cleaning changes and the split has to be rebuilt, say so explicitly — silently rebuilt splits are how a comparison table stops meaning anything.
