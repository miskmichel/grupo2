# Step 3e — BernoulliNB (teammate Caco)

**Date:** 2026-09-01
**Status:** Complete
**Artifacts produced:** `src/models/bernoulli_nb_caco.py`, `results/bernoulli_nb_caco.json`
**Reproduce:** `python3 src/models/bernoulli_nb_caco.py` (~9 s)
**Split:** the frozen `data/splits/` holdout via `dp.load_splits()` — verified identical to Caco's own split (see §1)
**CV:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, train only
**Test set:** read exactly once, in the script's final-evaluation step

---

## 0. What this is

Teammate **Caco** built this model independently, on branch `CacoJuse`, in his own `codigo.ipynb` — not as one of the team's four Step 3 tracks (`docs/step-03[a-d]-*.md`). His approach is a **`BernoulliNB`** with continuous features binarized into below/above-median indicators, distinct from the team's own 10-bin discretised NB in `docs/step-03c-knn-naive-bayes.md`.

This document credits and integrates that work: it verifies his split matches the team's frozen split, ports his exact modeling idea into this repo's `src/models/` conventions so it produces a directly comparable `results/*.json`, and reports both his original numbers and a team-convention-tuned threshold.

## 1. Split verification

Caco's `codigo.ipynb` calls `train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)` on a freshly-loaded `Telco-Customer-Churn.csv`, after dropping `customerID`/`gender`/`PhoneService` and mapping `Churn` to `{0,1}` — a different code path from this repo's `dp.load_splits()`, built without knowledge of the team's frozen split.

**Verified: identical result.** Reconstructing his split independently and comparing row indices against `data/splits/test.csv`'s `row_id` column shows **1409/1409 overlap** — his test set is the *exact same* 1409 customers, because both derive from the same seed (42), the same stratify column, and the same row order (a fresh `pd.read_csv` of the same file). His results are therefore directly comparable to the team's four tracks without any re-scoring.

## 2. His modeling approach (preserved as-is)

- **Continuous** (`tenure`, `MonthlyCharges`, `TotalCharges`): median-imputed, then `KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')` — a single below/above-median split per column.
- **Binary** (`SeniorCitizen`): most-frequent imputed, already 0/1.
- **Categorical** (everything else): most-frequent imputed, `OneHotEncoder(drop='first', handle_unknown='ignore')`.
- **Estimator:** `BernoulliNB(binarize=None, fit_prior=True)` — every input column is 0/1 by construction of the preprocessing above (asserted in the script).
- **Alpha (Laplace smoothing):** chosen by `GridSearchCV` over `[0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10]`, 5-fold CV, scoring `roc_auc`, train only.

This is his idea end to end — the port changes only the *loader* (`dp.load_splits()` instead of his own `train_test_split` call) and *adds* a tuned-threshold evaluation for comparability with the team's other four models (his own report used the default 0.50).

## 3. Reproduction check

| | Caco's own report (`CacoJuse`) | This port (`dp.load_splits()`) | Diff |
|---|---:|---:|---:|
| Best alpha | 0.5 | 0.1 | (CV scores within 0.001 of each other across the grid — not a meaningful disagreement) |
| CV ROC-AUC | 0.834 | 0.8332 | −0.0008 |
| Test ROC-AUC | 0.822 | 0.8225 | +0.0005 |
| Test F1(Yes) @0.50 | 0.602 | 0.6024 | +0.0004 |
| Test Precision(Yes) @0.50 | 0.554 | 0.5538 | −0.0002 |
| Test Recall(Yes) @0.50 | 0.660 | 0.6604 | +0.0004 |
| Test Accuracy @0.50 | 0.769 | 0.7686 | −0.0004 |

**Reproduction confirmed** — every metric lands within half a point of a percent of Caco's own reported numbers. The tiny residual gap is attributable to `dp.clean()`'s collapse of `'No internet service'`/`'No phone service'` into `'No'` (`CLAUDE.md` fact 5), which Caco's own preprocessing does not apply, producing a handful of extra one-hot columns in his original run that this port does not have. No leakage indicated (well under the 0.90 AUC warning threshold, `CLAUDE.md` fact 4).

## 4. Team-convention threshold

Adding the team's out-of-fold threshold tuning (maximise F1(Yes) on `cross_val_predict` train predictions, never touching test):

| | @0.50 (Caco's own) | @0.39 (tuned) | Change |
|---|---:|---:|---:|
| F1(Yes) | 0.6024 | 0.6134 | +0.0110 |
| Precision(Yes) | 0.5538 | 0.5408 | −0.0130 |
| Recall(Yes) | 0.6604 | 0.7086 | +0.0482 |
| Accuracy | 0.7686 | 0.7630 | −0.0056 |

Same direction as every other Step 3 track: tuning the threshold trades precision for recall, the correct direction for churn (a missed churner costs more than an unwanted retention call).

## 5. Result for the team comparison table

| Model | CV ROC-AUC | Test ROC-AUC | Test F1(Yes) | Precision(Yes) | Recall(Yes) | Threshold |
|---|---|---|---|---|---|---|
| BernoulliNB (Caco, quantile-binned) | 0.8332 | 0.8225 | 0.6134 | 0.5408 | 0.7086 | 0.39 |

This lands just below the Step 1 predicted 0.84–0.85 ROC-AUC band (0.8225, −0.0075 below the floor) — the lowest of the five approaches, but still well clear of the all-"No" baseline (ROC-AUC 0.5000) and inside a plausible range given the aggressive 2-bin quantization of continuous features (versus the team's own 10-bin NB, which lands inside the band at 0.8389). See `docs/step-04-model-comparison.md` for the full cross-model comparison and the final model decision.

## 6. Caco's own conclusion (independent corroboration)

Caco's own write-up (`docs/step-04-model-comparison.md` on `CacoJuse`) compares this `BernoulliNB` against his own Logistic Regression and **also picks Logistic Regression** — for the same reason the team did here (ROC-AUC fixed as the primary metric before modeling, and LR also wins F1 and recall). Two fully independent pipelines, developed without coordination, converging on the same model choice is one of the stronger pieces of evidence in `docs/step-04-model-comparison.md` §5.
