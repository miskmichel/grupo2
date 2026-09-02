# Changelog

Chronological ledger of what changed. Terse by design — reasoning belongs in `METHODOLOGY.md`, evidence in `docs/step-NN-*.md`.

Newest first.

---

## 2026-09-02

### Steps 3–5 — Models, comparison, and synthesis ✅

- Applied the two supplied course notebooks: the Module 3 logistic-regression workflow and the Module 2 classification metrics.
- Added a model-specific pipeline with train-only preprocessing, standardized numeric features, one-hot categoricals, `LogisticRegression(class_weight='balanced')`, and `TotalCharges` excluded only from the linear model.
- Adapted the multiclass examples to binary churn: `Churn = 1` is explicitly the positive class instead of reporting only weighted averages.
- Added accuracy, positive-class precision/recall/F1, ROC-AUC from probabilities, classification report, confusion matrix, and ROC curve.
- Added the course module's 0.30/0.50/0.70 threshold demonstration while explicitly forbidding threshold selection on the held-out test set.
- Added standardized-coefficient and odds-ratio interpretation.
- Full execution passed for all 16 non-empty code cells with zero errors. At threshold 0.50: ROC-AUC 0.839, F1(Yes) 0.617, recall(Yes) 0.778, precision(Yes) 0.511, accuracy 0.743; TN=756, FP=279, FN=83, TP=291.
- Applied the supplied Naive Bayes module as Model 2. Selected BernoulliNB because its inputs are explicitly converted to binary indicators: continuous variables use two quantile bins, the binary flag passes through, and categoricals use one-hot encoding.
- Tuned Laplace smoothing only on training folds with stratified five-fold `GridSearchCV`; selected `alpha=0.5` at mean CV ROC-AUC 0.834.
- BernoulliNB test results: accuracy 0.769, precision(Yes) 0.554, recall(Yes) 0.660, F1(Yes) 0.602, ROC-AUC 0.822; TN=836, FP=199, FN=127, TP=247.
- Added a same-split comparison against Logistic Regression and the majority baseline. Logistic wins by ROC-AUC (0.839 vs 0.822), F1 (0.617 vs 0.602), and recall (0.778 vs 0.660).
- Filled the required final synthesis cell and added `docs/step-04-model-comparison.md`. All 23 non-empty code cells execute with zero errors.

### Step 2 — Data treatment ✅

- Replaced the empty `codigo.ipynb` with a documented, executable preparation workflow.
- Added repository-relative CSV discovery without moving `Telco-Customer-Churn.csv`; the actual root path is preferred and the enunciado's `data/` path remains a fallback.
- Added schema/ID validation, target-distribution reporting, and the 73.46% majority-class baseline.
- Diagnosed the 11 space-only `TotalCharges` values before conversion, required `tenure == 0`, and imputed the known value `0.0` while preserving all 7,043 rows.
- Mapped `Churn` to `{No: 0, Yes: 1}` and collapsed the redundant no-service categories before encoding.
- Removed `customerID`, `gender`, and `PhoneService` from the feature matrix, then created a stratified 80/20 split (5,634 train / 1,409 test).
- Added an unfitted `ColumnTransformer` with train-only numeric imputation/scaling and categorical imputation/one-hot encoding; validation produced 21 finite features in both splits.
- Executed all 9 preparation code cells in the existing Anaconda environment (pandas 2.3.3, scikit-learn 1.7.2) with zero errors. No packages were installed.
- Added `docs/step-02-data-treatment.md` and updated project status/architecture/methodology.

### Environment correction

- Located an existing Anaconda Python 3.13.9 environment with the required data-science packages. The bare `python` command points to a Microsoft Store stub, which caused the earlier false impression that packages were unavailable on this machine.

---

## 2026-09-01

### Documentation structure established

- Added `CLAUDE.md` — working guidance plus the eight verified facts about the dataset that must not be re-derived or contradicted.
- Added `ARCHITECTURE.md` — repository map, column groups, the four structural dependencies in the data, and the intended modeling pipeline.
- Added `METHODOLOGY.md` — step-by-step narrative of decisions and rejected alternatives.
- Added `CHANGES.md` — this file.
- Added `docs/` for per-step evidence reports.
- Rewrote `README.md` from a one-line stub into a project index.

### Step 1 — Exploratory data analysis ✅

- Added `docs/step-01-eda.md`.
- Profiled all 7043 rows × 21 columns using stdlib `csv` + numpy/scipy (pandas unavailable at the time). **No packages were installed during this step.**
- **Solved the planted `TotalCharges` trap:** the 11 non-numeric values are single spaces `' '`; all 11 rows have `tenure == 0` (the only such rows in the dataset) and all are `Churn == 'No'`. Decision: impute `0`, not the median.
- Established the class balance — 73.46% No / 26.54% Yes — and therefore the **73.46% all-"No" accuracy baseline** that any accuracy claim must beat.
- Chose **ROC-AUC** as primary metric, F1/recall on `Yes` reported alongside.
- Verified `'No internet service'` occupies the same 1526 rows in all six add-on columns (and `'No phone service'` the same 682 rows as `PhoneService == 'No'`) — 7 collinear dummies to be avoided at encoding time.
- Measured `TotalCharges ≈ tenure × MonthlyCharges` at R² = 0.9991.
- Identified `gender` (p = 0.487) and `PhoneService` (p = 0.339) as statistically pure noise.
- Found `MonthlyCharges` to be **non-monotone** against churn (40.91% at decile 9 → 24.68% at decile 10) — the basis for including a tree-based model.
- Set the expected performance ceiling at **ROC-AUC 0.84–0.85, F1(Yes) 0.60–0.63**; anything above AUC 0.90 to be treated as suspected leakage.

### Errata recorded

- The enunciado's prose refers to `data/Telco-Customer-Churn.csv`, but its code cell reads `pd.read_csv('Telco-Customer-Churn.csv')`. The CSV stays at the repo root; **the file must not be moved** or the assignment notebook breaks. Documented in `CLAUDE.md` and `ARCHITECTURE.md`.

### Environment findings

- Python 3.12.3, numpy 2.5.1, scipy 1.18.0 present.
- **pandas, scikit-learn, matplotlib and seaborn are absent** — `enunciado.ipynb` cannot execute past its setup cell (cell 7) until they are installed. Flagged as a Step 2 prerequisite.

### Repository setup

- Installed GitHub CLI (`gh` 2.45.0, Ubuntu package) — a system-level change outside the repo.
- Forked `gustavokatsuo/hackaton-classificacao` → **`miskmichel/grupo2`**.
- Repointed remotes: `origin` → the fork, `upstream` → the original. `main` tracks `origin/main`.
