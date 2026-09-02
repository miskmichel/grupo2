# Methodology

The narrative record: what we did at each step, why we chose it, and what we rejected. Full evidence and numbers live in `docs/step-NN-*.md` — this file is the reasoning that connects them.

Read top to bottom to understand how the project got where it is.

---

## Overview

The assignment (`enunciado.ipynb`) breaks the 90 minutes into five phases. We follow that structure, one document per step.

| Step | Phase | Budget | Status | Evidence |
|---|---|---|---|---|
| 1 | Load and explore | 10 min | ✅ Complete | [`docs/step-01-eda.md`](docs/step-01-eda.md) |
| 2 | Treat data — missing values, categoricals, `customerID` | 20–25 min | ✅ Complete | [`docs/step-02-data-treatment.md`](docs/step-02-data-treatment.md) |
| 3 | Train ≥2 models | 30–35 min | ✅ Complete | [`docs/step-03-logistic-regression.md`](docs/step-03-logistic-regression.md) + [`docs/step-04-model-comparison.md`](docs/step-04-model-comparison.md) |
| 4 | Evaluate and compare | 15 min | ✅ Complete | [`docs/step-04-model-comparison.md`](docs/step-04-model-comparison.md) |
| 5 | Review + synthesis cell | 10 min | ✅ Complete | [`codigo.ipynb`](codigo.ipynb) |

---

## Step 1 — Exploratory data analysis

**Goal:** understand the dataset well enough to make cleaning decisions defensible rather than reflexive.

**How we did it.** The EDA was run with stdlib `csv` plus numpy/scipy, because pandas was not installed and we did not want to burn assignment time on environment setup before knowing what we were dealing with. Every categorical was profiled by cardinality and per-category churn rate; every numeric by distribution split on the target; and every column was swept for disguised missing values (`''`, `' '`, `NA`, `?`, `-`, `null`, whitespace) rather than trusting `isnull()`.

**What we found that changed our plan.**

The assignment plants an explicit hint that `TotalCharges` is text "for a reason". The reason turned out to be sharper than a generic dirty-data lesson: the 11 offending values are a **single space**, and all 11 rows are the *only* rows in the dataset with `tenure == 0`. They are brand-new customers who have never been billed. That reframes the fix — this is not missing data to be imputed with a central tendency, it is a **known zero**. Median imputation (1397.47) would assign a year's worth of billing to someone who has paid nothing.

Three findings beyond the planted trap materially shaped the rest of the project:

1. **Six columns carry a perfectly redundant level.** `'No internet service'` occupies the exact same 1526 rows across all six internet add-on columns, matching `InternetService == 'No'` one-to-one. Encoded naively this produces seven perfectly collinear dummies. This is a structural property of how the data was generated, not a correlation — so it can be removed with certainty rather than by threshold.

2. **`MonthlyCharges` is not monotone against churn.** Churn climbs to 40.91% at the ninth decile, then *falls* to 24.68% at the tenth. The most expensive customers are loyal bundlers — 40% of them on two-year contracts. A plain logistic regression fits a monotone effect and gets that decile backwards. This is the strongest argument for including a tree-based model rather than treating it as a box-ticking exercise.

3. **`TotalCharges` is 99.91% explained by `tenure × MonthlyCharges`** (R² = 0.9991). It is very nearly a derived column.

**Decisions taken.**

| Decision | Rationale | Alternative rejected |
|---|---|---|
| Primary metric: **ROC-AUC**; report F1/recall on `Yes` alongside | Class balance is 2.77:1, so accuracy is uninformative — the all-"No" baseline already scores 73.46%. AUC is threshold-free; F1-Yes is what the retention team actually feels | Accuracy — explicitly discouraged by the assignment |
| `TotalCharges = 0` for the 11 blanks | They are known zeros, not unknowns | Median imputation (wrong by construction); dropping the rows (silently deletes 11 non-churners) |
| Train at least one tree/ensemble model | The `MonthlyCharges` non-monotonicity and the `Contract × InternetService` interaction (70× spread across cells) are things linear models miss unless the interaction is hand-built | Logistic regression alone |
| Drop `gender` and `PhoneService` | p = 0.487 and p = 0.339 — not weak signal, *no* signal | Keeping them "just in case" |

**A caveat we are carrying forward.** `tenure` and `TotalCharges` are correlated with the outcome *by construction* — a customer who churned early necessarily has low tenure. This is legitimate signal for a "who will churn" model, but it is not causal, and it would not be available for scoring a customer at signup. This is the honest answer to the synthesis cell's "main technical decision" question.

**Evidence:** [`docs/step-01-eda.md`](docs/step-01-eda.md)

---

## Step 2 — Data treatment

**Goal:** implement the cleaning decisions from Step 1 as verified, reusable code — missing values, categorical encoding, and the `customerID` question the assignment names explicitly.

**Approach.** The treatment lives in `codigo.ipynb`, with the cleaning rules isolated in `limpar_dados()` and the encoder created by `criar_preprocessador()`. The latter returns an *unfitted* `ColumnTransformer`. The reason is leakage discipline: fitting a scaler or encoder on the full dataset before splitting is the single most common way to quietly inflate a score on this exercise, and creating the transformer separately from the fit makes that mistake structurally hard to commit.

The CSV is resolved from the repository using relative candidates (`Telco-Customer-Churn.csv` first, then the path stated in the prose, `data/Telco-Customer-Churn.csv`). This keeps the notebook portable without moving the vendored file or embedding a machine-specific path.

Cleaning is deterministic: the 11 invalid `TotalCharges` values are accepted only when `tenure == 0` and set to `0.0`; `Churn` is mapped to `{No: 0, Yes: 1}`; structurally redundant service labels are collapsed; and `customerID`, `gender`, and `PhoneService` are removed from `X`. The 80/20 split is stratified before any learned transformation. Numeric imputation/scaling and categorical imputation/one-hot encoding are then fitted only on the training portion.

Every cleaning claim is checked in code rather than assumed — row count preserved, known-zero charges fixed, target binary, redundant labels absent, finite transformed matrices, and identical train/test feature widths. The notebook executed all 9 code cells without errors and produced 5,634 training rows, 1,409 test rows, and 21 encoded features.

**Evidence:** [`docs/step-02-data-treatment.md`](docs/step-02-data-treatment.md)

---

## Step 3 — Modeling

**Model 1 complete:** `LogisticRegression(class_weight='balanced')` is fitted inside a fresh preprocessing `Pipeline` on the stratified training split. `TotalCharges` is removed only for this linear model because it is almost determined by `tenure × MonthlyCharges`; the cleaned column remains available to other estimators. Scaling and one-hot encoding are learned only from training data.

At the default 0.50 threshold, the logistic model scores **ROC-AUC 0.839**, **F1(Yes) 0.617**, **recall(Yes) 0.778**, **precision(Yes) 0.511**, and **accuracy 0.743**. The confusion matrix is TN=756, FP=279, FN=83, TP=291. The accuracy is only 0.8 percentage points above the 73.46% all-`No` baseline, while recall shows that the model identifies 77.8% of actual churners — a concrete demonstration of why accuracy cannot be the headline metric.

The coefficients and exponentiated odds ratios are exposed for interpretation, with the explicit caveat that numeric effects are per standard deviation and dummy effects are relative to the dropped reference category. They describe association, not causation.

**Model 2 complete:** Bernoulli Naive Bayes was chosen from the variants taught in the supplied module. Continuous variables are discretized into two quantile bins, `SeniorCitizen` is passed through as binary, and categorical variables are one-hot encoded. This makes every input compatible with the Bernoulli likelihood; MultinomialNB would require counts and GaussianNB would model the many dummy variables as Gaussian.

Laplace smoothing `alpha` is selected only from the training data with five-fold stratified cross-validation and ROC-AUC scoring. The selected `alpha=0.5` yields mean CV ROC-AUC **0.834**. On the held-out test set, BernoulliNB scores **ROC-AUC 0.822**, **F1(Yes) 0.602**, **recall(Yes) 0.660**, **precision(Yes) 0.554**, and **accuracy 0.769**; TN=836, FP=199, FN=127, TP=247.

Sanity bound established in Step 1: **ROC-AUC 0.84–0.85, F1(Yes) 0.60–0.63**. A score above AUC 0.90 should be treated as evidence of leakage, not success.

**Evidence:** [`docs/step-03-logistic-regression.md`](docs/step-03-logistic-regression.md) and [`docs/step-04-model-comparison.md`](docs/step-04-model-comparison.md)

---

## Step 4 — Evaluation

Both models are evaluated on the identical held-out rows with accuracy, positive-class precision/recall/F1, ROC-AUC, classification reports, confusion matrices, and ROC curves. Logistic Regression wins on the primary metric (**0.839 vs 0.822 ROC-AUC**) and also on F1 (**0.617 vs 0.602**) and recall (**0.778 vs 0.660**). BernoulliNB has higher accuracy (**0.769 vs 0.743**) and precision (**0.554 vs 0.511**), illustrating the metric trade-off.

The logistic section includes a didactic comparison of predeclared thresholds 0.30/0.50/0.70, but the notebook does not select a threshold from the test set. Operational threshold tuning must use validation or out-of-fold training predictions. Naive Bayes probabilities are also flagged as potentially uncalibrated because its conditional-independence assumption is not fully satisfied.

---

## Step 5 — Synthesis

Completed in the final notebook cell:

1. **Best model** — balanced Logistic Regression
2. **Chosen metric and value** — ROC-AUC **0.839**, with F1(Yes) 0.617 and recall(Yes) 0.778 at threshold 0.50
3. **Main technical decision and why** — learned transformations stay inside train-only pipelines; Naive Bayes receives truly binary inputs and selects smoothing by CV rather than the test set
4. **What we would try next** — tree/ensemble modeling, threshold selection by out-of-fold predictions, and probability calibration
