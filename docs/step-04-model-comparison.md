# Step 4 — Evaluation and comparison

**Date:** 2026-09-01
**Status:** Complete
**Primary metric:** ROC-AUC (F1/precision/recall on `Churn = Yes` reported alongside — never accuracy alone, per `CLAUDE.md` fact 3: the all-"No" baseline already scores 73.46% accuracy)
**Split:** the frozen `data/splits/` holdout via `dp.load_splits()` — 5634 train / 1409 test, `random_state=42`, stratified. Every model below was evaluated on the identical 1409 held-out rows.
**Test set:** read exactly once per model, after all tuning (hyper-parameters, thresholds) was chosen from training-fold cross-validation only.

---

## 1. Why this step exists

Step 1 set an expected ceiling (ROC-AUC 0.84–0.85, F1(Yes) 0.60–0.63) precisely so that Step 4 would not be a beauty contest between arbitrary numbers — it is a check of whether every approach landed inside that band (no leakage), and a forced, honest answer to "so which one do we ship."

## 2. Full comparison table

Five approaches: four are the team's own Step 3 tracks (`docs/step-03[a-d]-*.md`), each reported at its **own tuned decision threshold** (never the default 0.5 — Step 3 measured that tuning is worth more than model choice, see §4). The fifth, **BernoulliNB (Caco)**, is teammate Caco's independently-built model from branch `CacoJuse`, ported into this repo's `src/models/` conventions — confirmed to land on the *exact same* 1409 test rows as the frozen split (see `docs/step-03e-caco-naive-bayes.md`), so it is directly comparable despite being developed in parallel, without coordination.

| Model | CV ROC-AUC | Test ROC-AUC | Test F1(Yes) | Precision(Yes) | Recall(Yes) | Accuracy | Threshold |
|---|---|---|---|---|---|---|---|
| **GradientBoosting (tuned)** | **0.8486 ± 0.0084** | **0.8440** | **0.6228** | 0.5913 | 0.6578 | 0.7885 | 0.385 |
| LinearSVC (`class_weight='balanced'`) | 0.8450 ± 0.0085 | 0.8399 | 0.6182 | 0.5375 | 0.7273 | 0.7615 | 0.33 |
| Logistic Regression | 0.8463 ± 0.0085 | 0.8424 | 0.6178 | 0.5341 | 0.7326 | 0.7594 | 0.32 |
| Naive Bayes (10-bin discretised, team) | 0.8410 ± 0.0096 | 0.8389 | 0.6170 | 0.5530 | 0.6979 | 0.7700 | 0.49 |
| KNN (k=101, manhattan, uniform) | 0.8393 ± 0.0089 | 0.8363 | 0.6141 | — | — | — | 0.40 |
| BernoulliNB (Caco, quantile-binned) | 0.8332 | 0.8225 | 0.6134 | 0.5408 | 0.7086 | 0.7630 | 0.39 |
| — *all-"No" baseline* | — | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.7346 | — |

Caco's BernoulliNB lands just below the Step 1 predicted band (0.8225, −0.0075 under the 0.84 floor) — the lowest of the five, plausible given its aggressive 2-bin quantization of continuous features versus the team's own 10-bin NB (0.8389). Reproduction of his own reported numbers (CV 0.834, test AUC 0.822, F1 0.602 @0.50) landed within half a percentage point on every metric — see `docs/step-03e-caco-naive-bayes.md` §3.

## 3. Confusion matrices (test set, own tuned threshold)

```
GradientBoosting  @0.385   [[TN=865, FP=170], [FN=128, TP=246]]
LinearSVC         @0.33    [[TN=801, FP=234], [FN=102, TP=272]]
LogisticRegression@0.32    [[TN=796, FP=239], [FN=100, TP=274]]
NaiveBayes(10bin) @0.49    [[TN=824, FP=211], [FN=113, TP=261]]
```
(KNN's confusion matrix at threshold 0.40 is reported in `docs/step-03c-knn-naive-bayes.md` §5; Caco's BernoulliNB in `docs/step-03e-caco-naive-bayes.md`.)

## 4. The headline result is the spread, not the winner

Every one of the team's four approaches lands inside the Step 1 predicted band (ROC-AUC 0.84–0.85, F1(Yes) 0.60–0.63) — **no leakage anywhere** (nothing near AUC 0.90, `CLAUDE.md` fact 4) — and the entire range from best (GradientBoosting, 0.8440) to worst (KNN, 0.8363) is **0.0077 test ROC-AUC**, well under one CV standard deviation (±0.008–0.010). Model family barely matters here once each is properly tuned and thresholded; the ceiling is set by the data, not the algorithm.

**Threshold tuning outweighed model choice.** Every approach gained by abandoning the default 0.5 threshold: logistic regression +0.0097 F1(Yes) (42 more true churners caught), SVM +0.0228 F1. The tuned threshold trades precision for recall in every case — the correct direction for churn, where a missed churner (false negative) is a lost customer and a false positive is just an unnecessary retention call.

## 5. Choosing the model to ship

Given the near-tie, the decision is made on grounds *other* than the third decimal of ROC-AUC:

- **GradientBoosting** is the highest test ROC-AUC, but by a margin (0.0016 over LogisticRegression) smaller than a fifth of one CV standard deviation — not a defensible "it's better" claim on this evidence.
- **LogisticRegression** is functionally tied for best (0.8424 vs 0.8440, inside noise), ships with a readable coefficient/odds-ratio table the retention team can act on directly (`docs/step-03a-logistic-regression.md` §5), and both Step 1's structural analysis and Step 3b's permutation-importance work independently support the same drivers it surfaces (`tenure`, `Contract`, `InternetService`).
- Notably, **teammate Caco's fully independent effort reached the same conclusion** — his own Step 4 write-up (`docs/step-04-model-comparison.md` on `CacoJuse`) also selects Logistic Regression over his BernoulliNB, for the same reason (ROC-AUC was fixed as the primary metric *before* modelling, and LR also won F1 and recall on his split). Two independently-built pipelines converging on the same model, on the same metric, is stronger evidence than either pipeline's score in isolation.

**Decision: ship Logistic Regression** (`a_keep_total_charges` variant — `C=10.0`, `class_weight=None`, `penalty='l2'`, `solver='lbfgs'`, threshold 0.32). Test ROC-AUC **0.8424**, F1(Yes) **0.6178** (precision 0.5341 / recall 0.7326). Interpretability was the deciding factor given a statistical tie on the metric that was chosen to decide ties.

## 6. Evidence

`docs/step-03a-logistic-regression.md` through `docs/step-03e-caco-naive-bayes.md`; raw numbers in `results/*.json`; `METHODOLOGY.md` Step 3/4 sections for the full narrative.
