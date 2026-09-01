# Step 3d — Support Vector Machines + feature engineering

**Date:** 2026-09-01
**Status:** Complete
**Artifacts produced:** `src/models/svm_features.py`, `results/svm_features.json`
**Split:** the frozen split (`dp.load_splits()`), 5634 train / 1409 test, `test.csv` read exactly once
**Runtime:** 372 s wall clock end to end, of which 366 s is SVM fitting, on 2 cores
**Headline:** best CV model is **LinearSVC, `C=1.0`, `class_weight='balanced'`, on the *standard* preprocessor — no engineered features**, CV ROC-AUC **0.8450 ± 0.0085**, test ROC-AUC **0.8399**, test F1(Yes) **0.6182**.
**The feature engineering did not earn its place.** All nine engineered variants land inside ±0.0015 AUC of their baseline, except the `Contract × InternetService` interaction, which **hurt** by up to −0.0051. §6 explains why, with the arithmetic.

---

## 1. Brief

This module owns two questions the other three Step 3 agents do not touch:

1. **Part A — is an SVM competitive here, and at what cost?** Linear vs RBF kernel, `C` / `gamma` / `class_weight` tuned by CV on train.
2. **Part B — does explicit feature engineering beat the standard preprocessor?** Step 1 identified three structures `dp.build_preprocessor` does not encode (EDA §6.5 the interaction, §6.7 the non-monotone `MonthlyCharges`, §7.2 the `TotalCharges` collinearity), plus one trap (§6.6 the add-on count). Each is tested as an explicit CV experiment against the unengineered baseline.

This is the only Step 3 module permitted to change the feature representation. `src/data_prep.py` was not modified.

---

## 2. Protocol, and how leakage was prevented

| Rule | How it is enforced here |
|---|---|
| Frozen split only | `dp.load_splits()`. `dp.split()` is never called; no `random_state` is changed; no frame is re-indexed or rebuilt. |
| Test touched once | `X_te` appears in exactly one place in `svm_features.py` — the final `predict_proba` after the winner is already fixed. Every tuning, feature-selection and threshold decision is made on train. |
| Everything in a `Pipeline` | `Pipeline([('fe', FeatureEngineer), ('pre', ColumnTransformer), ('clf', SVM)])`. `cross_val_score` clones and refits the whole pipeline per fold. |
| Nothing fitted on the full frame | The scaler, the one-hot category lists, the `KBinsDiscretizer` edges and the `SplineTransformer` knots all live inside `('pre', …)` and are therefore learned on 4/5 of train per fold. |
| Scaling on | `dp.build_preprocessor(..., scale=True)` throughout. An SVM on unscaled inputs where `tenure` spans 0–72 and `TotalCharges` spans 0–8684.80 is effectively a one-variable model. |
| No target statistics | `FeatureEngineer` never sees `y`. There is no target/mean encoding anywhere — that is the usual route to a fake AUC on this dataset. |

**The engineered columns are split into two classes deliberately:**

- **Arithmetic, stateless** (`Contract_x_InternetService`, `TotalChargesResidual`, `n_addons`) — each output cell is a function of the *same row's* inputs only. `FeatureEngineer.fit` is a no-op. There is nothing to leak even in principle, so it is safe to read column *names* off the training frame when constructing the `ColumnTransformer`.
- **Learned** (bin edges, spline knots) — these *are* fitted from data, so they are pushed down into the `ColumnTransformer` and appended to its (still unfitted) `transformers` list. Precomputing 10 quantile bins on the full training frame would have leaked fold-to-fold; it does not happen.

**Sanity bound.** CLAUDE.md fact 4 sets the expected ceiling at ROC-AUC 0.84–0.85 and warns that > 0.90 means leakage. The script asserts this explicitly and prints a loud banner if breached. Final test AUC is **0.8399** — inside the band, no banner. The engineered variants never exceeded 0.8446 in CV, so no arm of this experiment ever showed the AUC-inflation signature.

### 2.1 Runtime discipline

`SVC` is O(n²)–O(n³) and `probability=True` refits with an internal 5-fold Platt calibration, i.e. roughly 6× a plain fit. A naive grid search over `probability=True` estimators on 5634 rows is what blows a 90-minute budget.

The decision taken: **all search arms run with `probability=False` and score ROC-AUC off `decision_function`.** ROC-AUC is rank-based and Platt scaling is a monotone transform, so the *ranking* is identical and the search is ~6× cheaper. Calibration is paid exactly twice — once for the out-of-fold threshold sweep, once for the final fit.

Measured cost: **372 s total, 366 s of it SVM fitting**, for 41 five-fold CV evaluations plus the calibrated OOF pass plus the final fit. Per-arm costs are printed and stored in `results/svm_features.json` (each variant's `note` carries its CV seconds). For scale: one 5-fold CV costs ~1 s for `LinearSVC`, ~12 s for `SVC(kernel='linear')`, ~8–16 s for `SVC(kernel='rbf')`, and the single calibrated OOF pass cost 1.5 s (the winner was `LinearSVC`; it would have cost ~70 s had an `SVC` won — measured, see §8).

---

## 3. Part A — kernel and hyper-parameter search

All arms below use the **standard preprocessor, no engineered features**, so the kernel comparison is not confounded with the feature comparison.

| Family | Config | CV ROC-AUC | ± std | CV seconds |
|---|---|---|---|---|
| LinearSVC | `C=0.01` | 0.8400 | 0.0087 | 8.1 |
| LinearSVC | `C=0.1` | 0.8422 | 0.0087 | 0.9 |
| LinearSVC | `C=1.0` | 0.8425 | 0.0088 | 1.0 |
| **LinearSVC** | **`C=1.0`, balanced** | **0.8450** | **0.0085** | **1.1** |
| SVC linear | `C=0.03` | 0.8334 | 0.0106 | 11.5 |
| SVC linear | `C=0.3` | 0.8329 | 0.0109 | 11.9 |
| SVC linear | `C=1.0` | 0.8331 | 0.0108 | 13.7 |
| SVC linear | `C=0.03`, balanced | 0.8427 | 0.0091 | 11.1 |
| SVC rbf | `C=1.0, γ='scale'` | 0.8025 | 0.0109 | 16.3 |
| SVC rbf | `C=1.0, γ=0.02` | 0.8278 | 0.0120 | 15.9 |
| SVC rbf | `C=3.0, γ='scale'` | 0.8006 | 0.0120 | 14.3 |
| SVC rbf | `C=3.0, γ=0.02` | 0.8229 | 0.0130 | 12.2 |
| SVC rbf | `C=10.0, γ='scale'` | 0.7889 | 0.0103 | 14.8 |
| SVC rbf | `C=10.0, γ=0.02` | 0.8192 | 0.0116 | 11.7 |
| SVC rbf | `C=1.0, γ=0.02`, balanced | 0.8432 | 0.0089 | 14.7 |

**Four findings.**

1. **The RBF kernel buys nothing.** Its best arm (0.8432) is statistically indistinguishable from the best linear arm (0.8450, and the gap is one-tenth of the ±0.0085 fold spread), and it costs 13× the fit time. Churn on this feature matrix is close to linearly separable in log-odds once the categoricals are one-hot encoded; there is no curved boundary to find.
2. **`gamma='scale'` is actively bad here** — 0.8025 vs 0.8278 at `C=1.0`, a −0.025 AUC gap. With 21 encoded columns of which 18 are 0/1 dummies, `gamma='scale'` = 1/(21·Var(X)) sets a bandwidth that over-localises. Every increase in `C` from 1 → 3 → 10 made it worse (0.8025 → 0.8006 → 0.7889): the model is already overfitting at `C=1`.
3. **`class_weight='balanced'` is the single largest hyper-parameter effect in Part A** — +0.0025 for LinearSVC, +0.0093 for SVC-linear, +0.0154 for SVC-rbf. On a 2.77:1 imbalance the unweighted hinge loss lets the margin drift toward the majority class; re-weighting recentres it. This is worth more than the entire feature-engineering exercise.
4. **`LinearSVC` beats `SVC(kernel='linear')` at the same nominal task** — 0.8425 vs 0.8334 unweighted. The two are not the same estimator: `LinearSVC` minimises *squared* hinge with an L2-regularised intercept via liblinear, `SVC` minimises plain hinge via libsvm. The squared-hinge solution is the better-calibrated ranker here. It is also 12× faster.

---

## 4. Part B — the feature experiments

Each engineered variant is run against its own family's Part A winner (which includes `class_weight='balanced'`, so the deltas are measured against the strongest available baseline, not a straw man). 5-fold `StratifiedKFold(shuffle=True, random_state=42)` on train.

**Reference for reading the deltas: the fold-to-fold standard deviation is ±0.0085 to ±0.0105.** A delta of ±0.0015 is a twentieth of that. Nothing in this table is distinguishable from noise except the interaction term, and that is distinguishable in the wrong direction.

| Variant | LinearSVC | Δ | SVC-linear | Δ | SVC-rbf | Δ |
|---|---|---|---|---|---|---|
| **baseline (standard preprocessor)** | **0.8450** | — | **0.8427** | — | **0.8432** | — |
| + `Contract × InternetService` (EDA §6.5) | 0.8445 | −0.0004 | 0.8394 | **−0.0033** | 0.8380 | **−0.0051** |
| + `MonthlyCharges` 10 quantile bins (§6.7) | 0.8444 | −0.0005 | 0.8427 | −0.0000 | 0.8425 | −0.0006 |
| + `MonthlyCharges` bins **and** linear term | 0.8444 | −0.0005 | 0.8421 | −0.0007 | 0.8420 | −0.0012 |
| + `MonthlyCharges` cubic spline, 5 knots (§6.7) | 0.8445 | −0.0004 | 0.8438 | +0.0010 | 0.8441 | +0.0010 |
| + `TotalCharges` residual (§7.2) | 0.8444 | −0.0006 | 0.8423 | −0.0004 | 0.8436 | +0.0004 |
| − `TotalCharges` dropped entirely (§7.2) | 0.8444 | −0.0006 | 0.8423 | −0.0004 | 0.8435 | +0.0004 |
| + add-on count 0–6 (§6.6) | 0.8449 | −0.0000 | 0.8427 | −0.0001 | 0.8429 | −0.0002 |
| **combined (all positive-delta features)** | *none qualified* | 0.0000 | spline | +0.0010 | spline + residual | +0.0014 |

**Verdict per feature, honestly stated:**

| Feature | Verdict | Reading |
|---|---|---|
| `Contract × InternetService` | **hurt** (−0.0033 / −0.0051 on the SVC families) | The one feature the EDA argued hardest for is the one that measurably damages the model. §6 explains why the EDA's reasoning was wrong. |
| `MonthlyCharges` binning | **neutral** (−0.0000 to −0.0012) | Ten quantile dummies replace one scaled column: it recovers the shape and loses the ordering, and the two cancel. |
| `MonthlyCharges` spline | **neutral, consistently positive** (+0.0010 on both SVC families) | The only engineered feature with a positive sign on more than one family. Still a seventh of a standard deviation. See §6.2 — the effect is real but already encoded elsewhere. |
| `TotalCharges` residual | **neutral** (−0.0006 to +0.0004) | And *indistinguishable from simply deleting the column* — the two arms agree to 0.0001 AUC on all three families. |
| add-on count | **neutral** (−0.0002 to 0.0000) | The EDA §6.6 artifact warning held. See §6.4. |

**No engineered feature earned its place in the final model.** The winning configuration uses `dp.build_preprocessor` exactly as the other three agents use it.

---

## 5. Model selection and the final model

Winner chosen by CV ROC-AUC across all six candidates (3 families × {baseline, combined-engineered}):

| Rank | Family | Features | CV ROC-AUC ± std |
|---|---|---|---|
| **1** | **LinearSVC `C=1.0` balanced** | **standard preprocessor** | **0.8450 ± 0.0085** |
| 2 | SVC-rbf `C=1.0 γ=0.02` balanced | spline + `TotalCharges` residual | 0.8446 ± 0.0094 |
| 3 | SVC-linear `C=0.03` balanced | spline | 0.8438 ± 0.0092 |
| 4 | SVC-rbf `C=1.0 γ=0.02` balanced | standard preprocessor | 0.8432 ± 0.0089 |
| 5 | SVC-linear `C=0.03` balanced | standard preprocessor | 0.8427 ± 0.0091 |

The top five span 0.0023 AUC against a ±0.009 fold spread — this is one model, not five. The tie is broken toward the simplest and cheapest: **LinearSVC on the untouched preprocessor**, which is also 13× faster to fit than the rank-2 RBF alternative.

`LinearSVC` has no `predict_proba`, so the final model is `CalibratedClassifierCV(LinearSVC(...), method='sigmoid', cv=5)` — the same Platt scaling `SVC(probability=True)` performs internally, and the replacement scikit-learn 1.9 now recommends (see §8).

### 5.1 Threshold

Chosen on **train out-of-fold probabilities only**, by maximising F1 on the positive class over a 0.05–0.95 sweep:

| | threshold | OOF F1(Yes) |
|---|---|---|
| default | 0.50 | 0.5979 |
| **selected** | **0.33** | **0.6365** |

The 0.33 threshold reflects the 26.54% base rate: a calibrated model asked to maximise F1 on a minority class should not cut at 0.5.

### 5.2 Final test numbers — the test set is read here, once

| Metric | At tuned threshold 0.33 | At default 0.50 |
|---|---|---|
| **ROC-AUC** | **0.8399** | 0.8399 (threshold-free) |
| **F1 (Yes)** | **0.6182** | 0.5954 |
| Precision (Yes) | 0.5375 | 0.6478 |
| Recall (Yes) | **0.7273** | 0.5508 |
| Accuracy | 0.7615 | 0.8013 |

Confusion matrix at 0.33 (rows = actual No/Yes, cols = predicted No/Yes):

```
          pred No   pred Yes
actual No     801        234
actual Yes    102        272
```

**Threshold gain: +0.0228 F1(Yes) on test (0.5954 → 0.6182), bought as +17.65 pp recall (0.5508 → 0.7273) for −11.03 pp precision.** For a retention team that is the right trade: the model now finds 272 of 374 real churners instead of 206, at the cost of 234 discount offers to customers who would have stayed.

Note the accuracy column moves the *other* way — 0.8013 at the default threshold, 0.7615 at the tuned one. This is exactly why CLAUDE.md fact 3 forbids accuracy as a headline: the tuned model is better on every metric the business cares about and worse on the one that is 73.46% satisfiable by predicting "No" forever.

**Consistency checks.** CV 0.8450 → test 0.8399 is a 0.0051 drop, well inside the ±0.0085 fold spread — no sign of selection overfitting despite 41 CV evaluations, which is expected since the winner was chosen from six candidates spanning 0.0023. Both numbers sit inside the CLAUDE.md fact-4 band of 0.84–0.85 AUC and 0.60–0.63 F1(Yes).

### 5.3 Top drivers

From the tuned **SVC-linear** coefficients (an interpretable proxy — the winning `LinearSVC` inside a calibration wrapper has the same sign structure; an RBF model would have no coefficients at all). Positive = pushes toward churn:

| Coef | Feature |
|---|---|
| −1.1467 | `InternetService_No` |
| −0.8733 | `Contract_Two year` |
| −0.8431 | `Contract_One year` |
| +0.5128 | `InternetService_Fiber optic` |
| −0.5036 | `tenure` |
| +0.4069 | `MonthlyCharges_sp_2` |
| −0.3205 | `MonthlyCharges_sp_3` |
| −0.2769 | `TechSupport_Yes` |
| −0.2586 | `OnlineSecurity_Yes` |
| +0.2151 | `PaymentMethod_Electronic check` |
| +0.2114 | `PaperlessBilling_Yes` |
| −0.1794 | `MonthlyCharges_sp_1` |

This reproduces EDA §6.1 rankings (Contract, InternetService, OnlineSecurity, TechSupport, PaymentMethod) with no surprises — a useful negative check that the engineering did not scramble the signal.

---

## 6. Why the engineered features did not help

This is the interesting half of the result, and it deserves the arithmetic rather than a shrug.

### 6.1 The interaction: a 70× spread is not evidence of an interaction

EDA §6.5 reports a 70× spread across the nine `Contract × InternetService` cells (M2M/Fiber 54.61% down to Two-year/No-internet 0.78%) and concludes "logistic regression needs the explicit interaction term or it underperforms here." **That inference does not follow, and the data says so.**

A model that is *additive in log-odds* produces a *multiplicative* spread in rates by construction. Fitting an additive logistic regression on `Contract` + `InternetService` **with no interaction term whatsoever** (training rows only, n=5634) reproduces all nine cells:

| Contract / Internet | n | observed | additive model predicts | error |
|---|---|---|---|---|
| Month-to-month / Fiber | 1707 | 55.07% | 55.08% | 0.01 pp |
| Month-to-month / DSL | 973 | 31.76% | 32.05% | 0.29 pp |
| One year / Fiber | 436 | 18.58% | 18.91% | 0.33 pp |
| Month-to-month / No | 422 | 18.25% | 17.18% | 1.07 pp |
| One year / DSL | 452 | 9.51% | 8.23% | 1.28 pp |
| Two year / Fiber | 340 | 7.06% | 6.37% | 0.69 pp |
| One year / No | 285 | 2.11% | 3.79% | **1.69 pp** |
| Two year / DSL | 512 | 1.95% | 2.55% | 0.60 pp |
| Two year / No | 507 | 0.99% | 1.14% | 0.15 pp |

**Maximum error 1.69 pp across nine cells.** The additive model already generates a 48.4× spread against the 55.8× observed on the training rows. There is almost no interaction left to add — and adding it costs 8 extra one-hot columns of near-collinear noise, which is precisely the −0.0033 / −0.0051 AUC seen in §4.

This **qualifies EDA §6.5 and CLAUDE.md fact 8's second clause** ("or add the `Contract × InternetService` interaction explicitly"): the interaction term is not a fix, it is a cost. The EDA's *measurement* is correct and useful; the modelling *inference* drawn from it is not. That two of the three families were measurably hurt by following the recommendation is the cleanest result in this report.

### 6.2 The non-monotone `MonthlyCharges`: real, already encoded

EDA §6.7 is right that the effect exists, and the spline coefficients in §5.3 show the model finding it — `MonthlyCharges_sp_2` at **+0.4069** flips to `MonthlyCharges_sp_3` at **−0.3205**, exactly the "rises to decile 9, falls at decile 10" shape, expressed as a sign change no single linear coefficient could produce.

And the AUC gain is **+0.0010**. The reason is in EDA §6.7's own last sentence: `MonthlyCharges` is nearly a proxy for `InternetService` (mean 91.50 fiber / 58.10 DSL / 21.08 none, near-disjoint ranges). The non-monotonicity is mostly the *composition* of the top decile — 40% two-year contracts with 4.80 add-ons — and `Contract`, the add-on columns and `InternetService` are all already in the matrix as their own dummies. The spline recovers a shape the model could already infer from other columns.

Binning did slightly *worse* than the spline (−0.0006 vs +0.0010 on the RBF family): 10 quantile dummies throw away the ordering of `MonthlyCharges` to buy flexibility the model does not need, whereas the spline keeps both. Keeping the linear term alongside the bins was worst of the three (−0.0012), which is what redundant collinear encodings normally do to a margin-based model.

### 6.3 The `TotalCharges` residual: new information, no predictive value

EDA §7.2's factual claim holds — the residual is genuinely not a duplicate column. But the decisive number is that **the residual arm and the drop-the-column-entirely arm agree to within 0.0001 AUC on all three families** (LinearSVC 0.8444 / 0.8444; SVC-linear 0.8423 / 0.8423; SVC-rbf 0.8436 / 0.8435). Replacing `TotalCharges` with its residual is, for prediction purposes, the same as deleting it. Promos and mid-contract plan changes are real events, and they carry no churn signal beyond what `tenure`, `MonthlyCharges` and `Contract` already say.

This does confirm the *other* half of EDA §7.2 and CLAUDE.md fact 6: `TotalCharges` can be dropped from a linear model at no cost (−0.0006 on LinearSVC, +0.0004 on SVC-rbf). It is dead weight, not a liability.

### 6.4 The add-on count: the artifact warning held

EDA §6.6 warned that the U-shape (0 add-ons 21.41%, 1 add-on 45.76%, 6 add-ons 5.28%) is a composition artifact — the 0-bucket is dominated by 1526 no-internet customers at 7.40% churn, the 1-bucket by fiber customers whose one add-on is a streaming service (`StreamingMovies` only + Fiber = 68.84% churn, n=138).

**Confirmed.** `n_addons` moved AUC by −0.0002, −0.0001 and −0.0000 on the three families — the flattest result in the table. The count collapses six columns whose individual dummies the model already has, and the collapse destroys exactly the composition information that made the raw bucket rates look dramatic. The EDA called this correctly in advance and the experiment adds nothing to it beyond confirmation.

---

## 7. A methodology bug found and fixed mid-run

The first execution of this script carried only the *unweighted* Part A winners into Part B, evaluating `class_weight='balanced'` as a side comparison that fed into nothing. It therefore ran every feature experiment on a knowingly inferior base and selected a final model at CV 0.8386 while three already-measured configurations scored higher (0.8425–0.8432). The test evaluation from that run reported ROC-AUC 0.8359 / F1 0.6162 for a model that was not the CV winner.

The selection logic was corrected so that each family's Part A winner **includes** its `class_weight` comparison, and the script re-run end to end. The numbers in this report are entirely from the corrected run. The discarded run is recorded here rather than quietly dropped: it was a model-selection defect, not a leak — the test set was read once in each run, and the first run's test numbers were not used to choose anything in the second. The correction was made from CV evidence alone.

---

## 8. Limitations, stated plainly

1. **The grid is deliberately coarse.** Three `C` values per family and two `gamma` values. A finer sweep around `C∈[0.3, 3]`, `gamma∈[0.005, 0.05]` might find another 0.002 AUC — which is a fifth of the fold-to-fold standard deviation and would not change any conclusion in this report. The 90-minute budget was better spent on the feature experiments.
2. **Only `class_weight='balanced'` was tried, not a swept weight ratio.** Given it was the largest single effect in Part A (+0.0154 on RBF), a swept `{0: 1, 1: w}` for `w ∈ [2, 4]` is the most promising unexplored direction.
3. **The RBF kernel was tuned on the unengineered features and its winner then reused for the feature experiments.** In principle the optimal `gamma` shifts when the matrix gains 8 interaction columns or 10 bin dummies. Re-tuning per feature set is the correct experiment; it costs 6× the runtime and would be defensible only if any feature arm had been close to winning. None was.
4. **`top_drivers` are `SVC-linear` coefficients, not the winning model's.** The winner is wrapped in `CalibratedClassifierCV`, which exposes five per-fold sub-estimators rather than one coefficient vector. Permutation importance on the winner would be more faithful; it was not run because computing it on test would spend the single permitted test read on interpretation rather than evaluation.
5. **`SVC(probability=True)` is deprecated in scikit-learn 1.9** (removed in 1.11) in favour of `CalibratedClassifierCV(SVC(), ensemble=False)`. The brief specifies `probability=True`, so `make_probabilistic` keeps it for the `SVC` families and the warning is silenced explicitly at the top of the module. Since the winner was `LinearSVC`, the final model uses `CalibratedClassifierCV` anyway and the deprecated path was never exercised in the final fit.
6. **The threshold is tuned for F1**, which weights precision and recall equally. The retention team's actual cost function is asymmetric (a missed churner costs a customer; a false positive costs a discount). With a real cost ratio the correct threshold would be lower than 0.33 still. F1 is a placeholder for a number nobody gave us.
7. **Single 5-fold CV, one seed.** Repeated CV over several seeds would tighten the ±0.0085 error bars and is the honest way to test whether the +0.0010 spline effect is real. At the observed effect size it would need many repeats to resolve, and it would not change the model choice.

---

## 9. Reproducing

```bash
cd /root/hackaton-classificacao
python3 src/models/svm_features.py      # ~372 s on 2 cores; writes results/svm_features.json
```

The script is deterministic: `StratifiedKFold(shuffle=True, random_state=42)`, `random_state=42` on every estimator that takes one, and the frozen split read from `data/splits/`. Verify the split first with `python3 src/make_splits.py --check`.

---

## 10. One-line summary for the Step 3 comparison table

> **SVM (LinearSVC, `C=1.0`, `class_weight='balanced'`, standard preprocessor, threshold 0.33):** CV ROC-AUC 0.8450 ± 0.0085, test ROC-AUC **0.8399**, F1(Yes) **0.6182**, precision 0.5375, recall 0.7273. Nine engineered-feature variants tested across three SVM families; **none improved CV AUC by more than 0.0015**, and the `Contract × InternetService` interaction *hurt* by up to 0.0051 because the effect is already additive in log-odds (an interaction-free model reproduces all nine cells to within 1.69 pp).
