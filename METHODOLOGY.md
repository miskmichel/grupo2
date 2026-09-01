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
| 3 | Train ≥2 models | 30–35 min | ✅ Complete | 5 approaches — see below |
| 4 | Evaluate and compare | 15 min | ✅ Complete | [`docs/step-04-model-comparison.md`](docs/step-04-model-comparison.md) |
| 5 | Review + synthesis cell | 10 min | ✅ Complete | [`codigo.ipynb`](codigo.ipynb), final cell |

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

**Approach.** Rather than cleaning inline in a notebook, the treatment was built as an importable module (`src/data_prep.py`) with the encoder returned as an *unfitted* `ColumnTransformer`. The reason is leakage discipline: fitting a scaler or encoder on the full dataset before splitting is the single most common way to quietly inflate a score on this exercise, and building the transformer separately from the fit makes that mistake structurally hard to commit.

Each cleaning claim was checked empirically rather than assumed — row counts preserved, the 11 rows landing at exactly 0.0, the redundant service levels actually gone, and no perfectly-collinear pairs surviving encoding. All 25 assertions passed and were then independently re-run by the lead before the work was committed.

**Decisions taken.**

| Issue | Decision | Alternative rejected |
|---|---|---|
| `TotalCharges` blanks | Coerce to numeric, `fillna(0)` | Median imputation — assigns a year of billing to someone billed nothing. Dropping the rows — silently deletes 11 non-churners |
| `customerID` | **Drop entirely** | Keeping it as a feature. Label-encoding it is the real trap: it produces a high-cardinality integer that a tree will happily split on, memorising the training set and inflating the score with pure noise |
| `'No internet service'` / `'No phone service'` | Collapse to `'No'` before encoding | Leaving them — measured to cost 6 rank deficiencies and 22 perfectly-correlated pairs |
| `gender`, `PhoneService` | Drop, behind a `drop_noise=True` flag | Hard-coding the drop — the flag lets the synthesis cell *demonstrate* the cull rather than assert it |
| Encoder fitting | `build_preprocessor()` returns it **unfitted** | Fitting on the full dataset before splitting — the classic leak on this exercise |

**What we got wrong in Step 1, and the correction.** Step 1 recorded that `TotalCharges` reads as dtype `object`. Under pandas 3.0.5 it reads as **`str`**. Every downstream consequence survives — `.isnull()` still catches nothing, `.astype(float)` still raises — but a guard written as `dtype == 'object'` would silently never fire. The module dispatches on `not is_numeric_dtype()` instead. Worth flagging to the team, because essentially every published tutorial for this dataset was written against pandas 1.x/2.x and says `object`.

**What we over-estimated.** Step 1 predicted 25–30 encoded columns; the real figure is **21**. Not an error in either direction — under `drop='first'` the collapse converts seven 3-level columns into 2-level ones, which saves seven dummies rather than the smaller number one counts by looking only at the *perfectly collinear* ones. The practical upside is real: 21 dense features on 5634 training rows leaves enough headroom to run `GridSearchCV` inside Step 3's 30–35 minute slot, which was not obviously affordable before.

**One thing that had to be forced.** Ubuntu 24.04 marks its system Python as externally managed (PEP 668), so the install needed `--break-system-packages`. That overrides a distro safety rail. It is acceptable here because the machine is a disposable hackathon environment; on anything longer-lived a virtualenv would be the right call instead.

**Freezing the split.** The split was then materialised to `data/splits/` and committed, rather than left as a function call. The reason is coordination, not tidiness: three people are about to train different models in parallel, and if each notebook re-derives its own split they are scored on different test rows and the comparison table is meaningless. Freezing makes the comparison valid by construction instead of by everyone remembering to pass `random_state=42`. It also keeps the holdout honest — re-splitting inside a tuning loop is how a test set quietly degrades into a validation set.

The manifest records a SHA-256 of each file and of the source CSV, so a regenerated split that differs is loud rather than silent. `customerID` stays out of the feature matrix but remains recoverable through a `row_id` join key, because a churn score nobody can attach to a customer is not actionable.

**Evidence:** [`docs/step-02-data-treatment.md`](docs/step-02-data-treatment.md)

---

## Step 3 — Modeling

**Goal:** train at least two models (the assignment's minimum) from Module 3, evaluate them the same way, and settle the one decision Step 1 left open — `TotalCharges` keep/drop/residual. We went further: four parallel approaches, one per Module 3 family, each on the **frozen split** (`dp.load_splits()`, 5634 train / 1409 test, `random_state=42`) so all four are directly comparable. Test was touched exactly once per model, at the end; all tuning and thresholding used 5-fold CV on train only.

**A fifth approach was added afterward.** Teammate Caco built a `BernoulliNB` model independently, on branch `CacoJuse`, without initially using the frozen split file — but his own `train_test_split(random_state=42, stratify=y)` on the raw CSV was verified to land on the *exact same* 1409 test rows (row-for-row) as `data/splits/`, so his results are directly comparable despite the parallel development. See [`docs/step-03e-caco-naive-bayes.md`](docs/step-03e-caco-naive-bayes.md).

### The comparison

| Approach | CV ROC-AUC | Test ROC-AUC | Test F1(Yes) | Precision(Yes) | Recall(Yes) | Threshold |
|---|---|---|---|---|---|---|
| **GradientBoosting** (tuned) | **0.8486 ± 0.0084** | **0.8440** | **0.6228** | 0.5913 | 0.6578 | 0.385 |
| LinearSVC (`class_weight='balanced'`) | 0.8450 ± 0.0085 | 0.8399 | 0.6182 | 0.5375 | 0.7273 | 0.33 |
| Logistic Regression | 0.8463 ± 0.0085 | 0.8424 | 0.6178 | 0.5341 | 0.7326 | 0.32 |
| Naive Bayes (discretised, 10 bins) | 0.8410 ± 0.0096 | 0.8389 | 0.6170 | 0.5530 | 0.6979 | 0.49 |
| KNN (k=101, manhattan, uniform) | 0.8393 ± 0.0089 | 0.8363 | 0.6141 | — | — | 0.40 |

**The headline result is the spread, not the winner.** Every approach lands inside the Step 1 predicted band (ROC-AUC 0.84–0.85, F1(Yes) 0.60–0.63) — no leakage anywhere — and the *entire* range from best (GradientBoosting, 0.8440) to worst (KNN, 0.8363) is **0.0077 test ROC-AUC**, well under one CV standard deviation (±0.008–0.010). On this dataset, model family barely matters once each is properly tuned and thresholded; the ceiling is set by the data, not the algorithm. This is itself the most defensible finding for the synthesis cell's "why this model" answer — the honest version is "the practical differences were noise-level, so we picked X for reason Y (interpretability / recall / simplicity)," not "X was measurably best."

**`TotalCharges`, finally answered.** Logistic regression tested keep vs. drop vs. residual under identical CV: keeping it won, by +0.0016 CV ROC-AUC over either alternative — itself noise-level, meaning the theoretical collinearity concern (fact 6, R²=0.9991 with `tenure×MonthlyCharges`) never translated into a measurable CV cost for a *regularised* linear model. **Decision: keep `TotalCharges`.**

**Threshold tuning was worth more than model choice.** Every approach gained by abandoning the default 0.5 threshold — logistic regression +0.0097 F1(Yes), moving 42 more true churners into the caught column; SVM +0.0228 F1. The tuned threshold trades precision for recall in every case, which is the correct direction for a churn model: a missed churner (false negative) is a lost customer, a false positive is a retention offer to someone who wasn't leaving.

### Two Step 1 claims corrected by Step 3's evidence

Both are now recorded in `CLAUDE.md` fact 8, since that is the file people read before touching this dataset again.

1. **Trees do not beat a tuned linear model here**, and Cramér's V is a poor tree-importance ranking. GradientBoosting's permutation importances concentrate 82% of weight in three features (`tenure`, `InternetService`, `Contract`); `OnlineSecurity`/`TechSupport` — EDA's #2 and #3 by Cramér's V — score 22–30× lower, because their marginal signal mostly *is* the `InternetService` effect flowing through the "No internet service"→"No" collapse (fact 5). (`docs/step-03b-tree-ensembles.md`)
2. **"Add the `Contract × InternetService` interaction explicitly" was actively wrong.** A 70× rate spread is not evidence of a true interaction — an additive-in-log-odds model produces multiplicative spread by construction, and a plain additive fit reproduces the nine-cell table to within 1.69 pp with no interaction term at all. Adding the term anyway cost −0.0033 to −0.0051 CV ROC-AUC, the single worst feature-engineering result measured. (`docs/step-03d-svm-features.md`)

Both corrections replaced the original wording in `CLAUDE.md` rather than sitting only in the per-model reports, so a future reader doesn't rediscover the mistake.

### What each approach was for, and what it found

- **[Logistic Regression](docs/step-03a-logistic-regression.md)** — the interpretable baseline and the `TotalCharges` decision-maker (above). Coefficients sanity-checked clean against EDA (`Contract_Two year`/`One year` strongly negative, `PaymentMethod_Electronic check` positive, as predicted). Recovered, and explained, the `MonthlyCharges` sign flip: raw correlation +0.193 but a negative fitted coefficient once `InternetService` is controlled — the bill is a proxy for service tier, and *within* a tier a higher bill is mildly protective.
- **[Tree Ensembles](docs/step-03b-tree-ensembles.md)** — asked whether trees actually exploit the non-monotone `MonthlyCharges` decile and the `Contract×InternetService` interaction. Answer: marginally (a 0.94 pp partial-dependence turn against a 16.66 pp decile swing), and not enough to beat a well-tuned linear model. Best single model overall by test ROC-AUC, by a margin smaller than CV noise.
- **[KNN / Naive Bayes](docs/step-03c-knn-naive-bayes.md)** — briefed to demonstrate *why* these families should underperform; the honest result is closer than expected (both land at the bottom of the predicted band, not below it). Found the mechanism precisely: in the 21-column mostly-binary encoded space, 100% of sampled points have tied nearest-neighbour distances in the pure-dummy subspace (22.71 of 25 ties), which is why KNN needs k≈101 to work at all — it survives only by stopping being local. For Naive Bayes, two independent measurements agree the independence-violation cost is ≈0.019 AUC (dropping `TotalCharges` helps `GaussianNB` by exactly that; replacing the diagonal covariance with a full one via QDA gains almost the same). The winning NB variant wins by *discretising away* the correlations it can't model, not by fitting them.
- **[SVM + feature engineering](docs/step-03d-svm-features.md)** — the only approach allowed to change the feature representation. Result: an honest null. Every engineered feature (the interaction term, `MonthlyCharges` binning/splines, the `TotalCharges` residual, an add-on count) landed within ±0.0015 AUC of the unengineered baseline, except the interaction term, which hurt (see above). `class_weight='balanced'` was a bigger lever than all the feature engineering combined (+0.0025 to +0.0154 depending on kernel). RBF bought nothing over linear at 13× the training cost.

**Evidence:** `docs/step-03a-logistic-regression.md`, `docs/step-03b-tree-ensembles.md`, `docs/step-03c-knn-naive-bayes.md`, `docs/step-03d-svm-features.md`; raw numbers in `results/*.json`.

---

## Step 4 — Evaluation

**Goal:** decide which model to ship, on grounds beyond the third decimal of a single held-out ROC-AUC number.

All five approaches (four team tracks plus Caco's BernoulliNB) land inside the Step 1 predicted 0.84–0.85 ROC-AUC band, and the full spread from best to worst is under one CV standard deviation — see the full table and confusion matrices in [`docs/step-04-model-comparison.md`](docs/step-04-model-comparison.md). Two findings drove the final call:

1. **Threshold tuning outweighed model family.** Every approach gained F1(Yes) by moving off the default 0.5 threshold (up to +0.0228 for SVM), always trading precision for recall — the right direction when a missed churner costs more than an unnecessary retention call.
2. **The score alone can't break the near-tie, so interpretability did.** GradientBoosting's 0.0016 edge over Logistic Regression is smaller than a fifth of a CV standard deviation — not a defensible "better" claim. Logistic Regression was chosen instead: its coefficient/odds-ratio table is directly actionable by a retention team, and **teammate Caco's fully independent pipeline reached the identical conclusion** (his own `docs/step-04-model-comparison.md` on `CacoJuse` also picks Logistic Regression over his BernoulliNB, for the same reason). Two separately-built pipelines agreeing is stronger evidence than either model's score in isolation.

**Decision: ship Logistic Regression** (`a_keep_total_charges`, `C=10.0`, threshold 0.32). Test ROC-AUC **0.8424**, F1(Yes) **0.6178**.

**Evidence:** [`docs/step-04-model-comparison.md`](docs/step-04-model-comparison.md)

---

## Step 5 — Synthesis

`codigo.ipynb` now runs start-to-finish — load → clean → frozen split → fit the chosen Logistic Regression pipeline → evaluate once on test → a dynamically-generated comparison table pulled from `results/*.json` — and ends with the required synthesis cell:

1. **Best model** — Logistic Regression (`a_keep_total_charges`, `C=10.0`, `class_weight=None`, `penalty='l2'`, `solver='lbfgs'`, threshold 0.32), chosen for interpretability given a statistical tie with GradientBoosting and independent convergence with Caco's own model selection.
2. **Chosen metric and value** — ROC-AUC (decided in Step 1, before any modeling, because of the 73.46%/26.54% imbalance) = **0.8424** on test; F1(Yes) = **0.6178** (precision 0.5341, recall 0.7326) at the tuned threshold.
3. **Main technical decision and why** — freezing the train/test split to `data/splits/` before any of the five models were built. Without it, five people training in parallel would each score on different held-out rows and the entire comparison in Step 4 would be meaningless. The `TotalCharges` keep/drop/residual question (Step 1's other open item) was also settled empirically: keep won, but by a margin inside CV noise.
4. **What we'd try next** — probability calibration (a calibration curve / Brier score) before presenting the score as a "churn risk" to a business stakeholder; repeated CV or multiple seeds — the team's four models span only 0.0077 AUC (0.8363–0.8440), roughly the size of single-fold CV noise, but the fifth approach (Caco's BernoulliNB, 0.8225) widens the full five-model spread to 0.0215, worth confirming isn't itself just cross-run noise; and a model restricted to features known at signup (dropping `tenure`/`TotalCharges`, which carry survivorship signal by construction — see the Step 1 caveat above) to test whether the 0.84–0.85 ceiling holds for the actually-actionable "who will churn before we've billed them much" use case.
