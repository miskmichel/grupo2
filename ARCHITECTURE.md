# Architecture

How this repository is laid out and why.

## Repository map

```
hackaton-classificacao/
├── README.md                  # entry point / index
├── CLAUDE.md                  # working guidance + verified facts about the data
├── ARCHITECTURE.md            # this file
├── CHANGES.md                 # chronological changelog
├── METHODOLOGY.md             # what we did at each step, and why
│
├── enunciado.ipynb            # THE ASSIGNMENT (read-only — do not edit)
├── codigo.ipynb               # executable data preparation and, next, modeling
├── Telco-Customer-Churn.csv   # the dataset (root path — do not move)
│
└── docs/
    ├── step-01-eda.md         # Step 1 — exploratory data analysis
    ├── step-02-data-treatment.md # Step 2 — cleaning and preprocessing
    ├── step-03-logistic-regression.md # Step 3 — first model and metrics
    └── step-04-model-comparison.md # Step 4 — Naive Bayes and comparison
```

### Why the CSV sits at the root

The enunciado's prose text refers to `data/Telco-Customer-Churn.csv`, but its executable cell reads:

```python
df = pd.read_csv('Telco-Customer-Churn.csv')
```

The code is authoritative. Moving the file into `data/` would break the assignment notebook, so the file stays at the root and the prose is treated as an erratum. This is recorded here so nobody "tidies" it later.

### Why `enunciado.ipynb` is read-only

It is the assignment as issued, and it is tracked against `upstream`. Editing it would create merge conflicts if the instructor pushes a fix, and would muddy the record of what was actually asked. All work happens in a separate notebook.

## Document roles

| File | Answers | Updated |
|---|---|---|
| `README.md` | "What is this and where do I start?" | Rarely |
| `CLAUDE.md` | "What must I not get wrong?" | When a new hard fact is established |
| `ARCHITECTURE.md` | "Where does everything live and why?" | When structure changes |
| `METHODOLOGY.md` | "What did we do, in what order, and why?" | Every step |
| `CHANGES.md` | "What changed and when?" | Every change |
| `docs/step-NN-*.md` | "What did we find at step N?" | Once per step, then frozen |

The split matters: `METHODOLOGY.md` is the **narrative** (decisions and reasoning, readable start to finish), `CHANGES.md` is the **ledger** (dated, terse, append-only), and `docs/step-NN-*.md` holds the **evidence** (full numbers, tables, derivations). Keeping evidence out of the narrative is what keeps the narrative readable.

## The data

**7043 rows × 21 columns.** One row per customer, a static snapshot — no time dimension, no event log.

### Column groups

| Group | Columns |
|---|---|
| **Identifier** | `customerID` (7043 unique — always dropped) |
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Account** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| **Phone services** | `PhoneService`, `MultipleLines` |
| **Internet services** | `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| **Billing** | `MonthlyCharges`, `TotalCharges` |
| **Target** | `Churn` (Yes 26.54% / No 73.46%) |

### Structural dependencies between columns

These are not statistical correlations — they are hard constraints built into how the data was generated. They dictate the encoding strategy.

```
InternetService == 'No'   ⟺  'No internet service' in all 6 internet add-on columns   (1526 rows, exact)
PhoneService    == 'No'   ⟺  MultipleLines == 'No phone service'                      ( 682 rows, exact)
TotalCharges              ≈  tenure × MonthlyCharges                                  (R² = 0.9991)
tenure == 0               ⟺  TotalCharges == ' '                                      (  11 rows, exact)
```

Consequences: collapse the "No X service" levels to `'No'` before one-hot encoding (saves 7 collinear dummies); drop `TotalCharges` for linear models; impute `TotalCharges = 0` rather than the median.

## Modeling pipeline

The complete assignment pipeline is implemented in `codigo.ipynb`: balanced logistic regression, Bernoulli Naive Bayes, matched evaluation, comparison, and final synthesis.

```
raw CSV
  │
  ├─ coerce  TotalCharges → numeric, fillna(0)
  ├─ map     Churn → {No: 0, Yes: 1}
  ├─ collapse 'No internet service' / 'No phone service' → 'No'
  ├─ drop    customerID, gender, PhoneService
  │
  ▼
train / test split  ── stratified on Churn ──┐
  │                                          │
  ▼                                          ▼
ColumnTransformer                       (test set held out,
  ├─ numeric   → StandardScaler          untouched until final
  └─ categoric → OneHotEncoder(drop='first')   evaluation)
  │
  ▼
estimator  ── ≥2 required by the assignment ──
  ├─ LogisticRegression(class_weight='balanced')   # interpretable driver list
  └─ RandomForest / GradientBoosting               # captures the non-monotonicity
  │
  ▼
evaluate  → ROC-AUC (primary) + F1/recall on Yes + confusion matrix
          → tune decision threshold, do not assume 0.5
```

Everything from `ColumnTransformer` down belongs **inside** a single `sklearn.pipeline.Pipeline`, fitted only on the training fold. Fitting any transformer before the split leaks test information.

## Constraints

- **Time:** 90 minutes total, per the assignment.
- **Algorithms:** restricted to Module 3 — KNN, Naive Bayes, Logistic Regression, SVM, Trees, Ensemble.
- **Offline:** the assignment states no internet is required; the dataset is vendored in the repo.
- **Environment:** the available Anaconda kernel includes pandas and scikit-learn; see `CLAUDE.md` for kernel selection details.
