# Methodology

The narrative record: what we did at each step, why we chose it, and what we rejected. Full evidence and numbers live in `docs/step-NN-*.md` — this file is the reasoning that connects them.

Read top to bottom to understand how the project got where it is.

---

## Overview

The assignment (`enunciado.ipynb`) breaks the 90 minutes into five phases. We follow that structure, one document per step.

| Step | Phase | Budget | Status | Evidence |
|---|---|---|---|---|
| 1 | Load and explore | 10 min | ✅ Complete | [`docs/step-01-eda.md`](docs/step-01-eda.md) |
| 2 | Treat data — missing values, categoricals, `customerID` | 20–25 min | 🔄 In progress | `docs/step-02-data-treatment.md` |
| 3 | Train ≥2 models | 30–35 min | ⬜ Not started | — |
| 4 | Evaluate and compare | 15 min | ⬜ Not started | — |
| 5 | Review + synthesis cell | 10 min | ⬜ Not started | — |

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

**Approach.** Rather than cleaning inline in a notebook, the treatment is being built as an importable module (`src/data_prep.py`) with the encoder returned as an *unfitted* `ColumnTransformer`. The reason is leakage discipline: fitting a scaler or encoder on the full dataset before splitting is the single most common way to quietly inflate a score on this exercise, and building the transformer separately from the fit makes that mistake structurally hard to commit.

Each cleaning claim is being checked empirically rather than assumed — row counts preserved, the 11 rows landing at exactly 0.0, the redundant service levels actually gone, and no perfectly-collinear pairs surviving encoding.

*This section will be completed when the step finishes.*

**Evidence:** `docs/step-02-data-treatment.md`

---

## Step 3 — Modeling

Not started. Planned: `LogisticRegression(class_weight='balanced')` for an interpretable driver list, plus a `RandomForest`/`GradientBoosting` to capture the non-monotonicity Step 1 identified. Both inside a single `Pipeline`, both evaluated on the same stratified split.

Sanity bound established in Step 1: **ROC-AUC 0.84–0.85, F1(Yes) 0.60–0.63**. A score above AUC 0.90 should be treated as evidence of leakage, not success.

---

## Step 4 — Evaluation

Not started. Planned: ROC-AUC and F1-Yes for each model on the held-out set, confusion matrices, and decision-threshold tuning rather than accepting the default 0.5.

---

## Step 5 — Synthesis

Not started. The assignment requires four fields:

1. **Best model** — TBD
2. **Chosen metric and value** — TBD (metric decided in Step 1: ROC-AUC primary)
3. **Main technical decision and why** — candidate: the `TotalCharges = 0` reasoning, or the construction-correlation caveat on `tenure`
4. **What we would try next** — TBD
