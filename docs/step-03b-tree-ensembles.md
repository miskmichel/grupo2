# Step 3b — Tree Ensembles (Random Forest / Gradient Boosting / HistGB)

**Date:** 2026-09-01
**Status:** Complete
**Artifacts produced:** `src/models/tree_ensembles.py`, `results/tree_ensembles.json`, `results/tree_ensembles_monthlycharges_pd.png`
**Split:** the frozen `data/splits/` split via `dp.load_splits()` — 5634 train / 1409 test, never re-derived
**Method note:** every number below is observed output of `python3 src/models/tree_ensembles.py`, not an estimate. The script runs start to finish in 380.5 s and reproduces the whole report.

---

## 1. The question this step was asked

Step 1 found two structures that a linear model handles badly and an axis-aligned tree finds natively:

- **`MonthlyCharges` is non-monotone** — churn climbs to 40.91% at decile 9 and then *falls* to 24.68% at decile 10 (EDA §6.7). `CLAUDE.md` fact #8 concludes: *"Plain logistic regression gets the top decile backwards. Prefer trees."*
- **The `Contract × InternetService` interaction spans 70×** — 54.61% (M2M/Fiber, n=2128) down to 0.78% (Two-year/No-internet, n=638) (EDA §6.5).

So the brief was not "maximise AUC" but: **do trees actually cash in on these two structures, or does the linear model match them anyway?**

**Short answer: they do not cash in, and the linear model matches them.** The best tree ensemble beats the best logistic regression by **+0.0023 CV ROC-AUC and +0.0016 test ROC-AUC** — between a fifth and a third of a single CV standard deviation (±0.0084). And the reason turns out to invalidate half of fact #8; see §7, which is the most interesting result in this step.

---

## 2. Protocol — what was and was not allowed to touch the test set

| Rule | How it was enforced |
|---|---|
| Frozen split | `dp.load_splits()` only. `dp.split()` is never called; no `random_state` is chosen for splitting. Train 5634 rows / 26.5353% churn, test 1409 / 26.5436% — the numbers asserted in Step 2 §9. |
| **Test read exactly once** | Every model choice, hyper-parameter and the decision threshold comes from `StratifiedKFold(5, shuffle=True, random_state=42)` on **train only**. `X_te` first appears at script stage `[7]`, after the best model and threshold are already fixed. |
| No leaking transformer | Every estimator sits in a `Pipeline` whose first step is the **unfitted** `ColumnTransformer` from `dp.build_preprocessor(df=…, scale=False)`. The one-hot vocabulary is re-learned on each of the 5 training folds. |
| `scale=False` | Scaling is a no-op for trees. Passing the flag rather than a different preprocessor keeps a single pipeline shape across all four Step 3 agents. |
| Metric | ROC-AUC primary; F1 / precision / recall on `Yes` = 1 alongside. Accuracy is never the headline — the all-`No` baseline is 73.4564% on this test fold. |

**Feature matrix:** the Step 2 default — 17 raw columns → 21 encoded (`TotalCharges` kept, `gender`/`PhoneService` dropped). `drop_total_charges=True` was *not* tested here; fact #6 scopes that drop to linear models, and `TotalCharges` turns out to carry non-zero conditional signal for the trees (§6).

---

## 3. Models tried — cross-validated on train, ROC-AUC

Ten configurations. All CV numbers are 5-fold on the training fold; none of them saw the test set.

| # | Model | CV ROC-AUC | ± std | Note |
|---|---|---|---|---|
| 1 | RandomForest, 200 trees, default depth | 0.8228 | 0.0087 | fully grown trees — **the worst of the ten** |
| 2 | RandomForest, 200 trees, `class_weight='balanced'` | 0.8257 | 0.0107 | +0.0029 vs #1 |
| 3 | GradientBoosting, default | 0.8470 | 0.0089 | untuned, already 2nd best overall |
| 4 | GradientBoosting, balanced `sample_weight` | 0.8455 | 0.0082 | −0.0015 vs #3 |
| 5 | HistGradientBoosting, default | 0.8336 | 0.0095 | |
| 6 | HistGradientBoosting, `class_weight='balanced'` | 0.8357 | 0.0108 | +0.0021 vs #5 |
| 7 | RandomForest, tuned | 0.8464 | 0.0098 | **+0.0236 over its own default** |
| 8 | **GradientBoosting, tuned** | **0.8486** | **0.0084** | **best** |
| 9 | GradientBoosting, tuned, balanced weights | 0.8485 | 0.0096 | −0.0001 vs #8 |
| 10 | HistGradientBoosting, tuned | 0.8455 | 0.0088 | |

Three things are worth reading off this table.

**3.1 An untuned Random Forest is the worst model in the step, and tuning is worth more to it than to anything else.** Default `RandomForestClassifier` grows every tree to purity. On a target this noisy — EDA §4.4 found 20 distinct feature vectors, 42 rows, carrying *inconsistent* churn labels, an irreducible error floor — fully grown trees memorise noise. Constraining depth to 10 and leaves to ≥10 samples buys **+0.0236 AUC (0.8228 → 0.8464)**, by far the largest single improvement in the step. Gradient boosting needs almost none of that help: tuning moves it +0.0016 (0.8470 → 0.8486), because its trees are shallow by construction.

**3.2 Boosting beats bagging here, and it is not close at default settings** — 0.8470 vs 0.8228, a 0.0242 gap that shrinks to 0.0022 once the forest is tuned.

**3.3 Class weighting is worth nothing on the primary metric.** Across all three families the balanced variant moves CV ROC-AUC by **+0.0029, −0.0015 and +0.0021** — every one of them smaller than the CV standard deviation (~0.009). That is the expected result, not a surprise: **ROC-AUC is a ranking metric and is invariant to a monotone rescaling of the scores.** Re-weighting the classes mostly re-calibrates the probabilities, i.e. it moves scores across the 0.5 line without reordering them. It therefore changes F1-at-a-fixed-threshold and leaves AUC alone. **Threshold tuning (§5) does the same job more directly and more controllably**, which is why the chosen model is the unweighted one.

### `GradientBoostingClassifier` and the missing `class_weight`

`GradientBoostingClassifier` has no `class_weight` parameter; the documented route is `fit(..., sample_weight=...)`. Passing a precomputed weight vector through `cross_val_score` does not work without metadata routing enabled — sklearn hands each fold the *full-length* vector and the fit raises on the length mismatch. Since the balanced weight is a deterministic function of `y` alone, the script subclasses the estimator and computes it inside `fit`:

```python
class BalancedGradientBoosting(GradientBoostingClassifier):
    def fit(self, X, y, sample_weight=None, **kwargs):
        if sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y)
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)
```

Exactly equivalent to the documented call, and it survives `clone`, CV fold-slicing and `RandomizedSearchCV` untouched. It is rows #4 and #9 of the table.

---

## 4. Tuning

`RandomizedSearchCV`, 5-fold, `scoring='roc_auc'`, `random_state=42`, on the training fold only. **20 draws / 100 CV fits total**, deliberately small: the encoded matrix is 5634 × 21, the between-model differences are ~0.002 and the CV noise is ~0.009, so a larger search would mostly be fitting the fold noise. Search wall-times: RF 105 s, GB 47 s, GB-balanced 37 s, HGB 53 s.

The grids, as searched:

| Model | Grid | Draws |
|---|---|---|
| RandomForest | `n_estimators` [150, 250] · `max_depth` [6, 8, 10, None] · `min_samples_leaf` [5, 10, 20, 40] · `max_features` ['sqrt', 0.3] · `class_weight` [None, 'balanced'] | 6 |
| GradientBoosting | `n_estimators` [100, 200] · `learning_rate` [0.05, 0.1] · `max_depth` [2, 3] · `min_samples_leaf` [20, 50] · `subsample` [0.8, 1.0] · `max_features` ['sqrt', None] | 5 |
| GradientBoosting (balanced weights) | same grid | 4 |
| HistGradientBoosting | `learning_rate` [0.05, 0.1] · `max_iter` [100, 200] · `max_leaf_nodes` [15, 31] · `min_samples_leaf` [20, 50] · `l2_regularization` [0.0, 1.0] · `class_weight` [None, 'balanced'] | 5 |

The grids are centred on the regularised end of each family on purpose: §3.1 shows the unconstrained forest is the worst model in the step, so the useful direction was known before the search started.

### Winning configuration — `GradientBoostingClassifier`

| Parameter | Value |
|---|---|
| `n_estimators` | 100 |
| `learning_rate` | 0.1 |
| **`max_depth`** | **2** |
| `min_samples_leaf` | 50 |
| `subsample` | 0.8 |
| `max_features` | `None` |

**`max_depth=2` is the interesting entry, and it points at the answer to §1's question.** A depth-2 tree expresses exactly one two-way split interaction and nothing deeper, so the winning ensemble is an **additive model plus pairwise interactions** — no room for the three- and four-way structure a tree ensemble is supposed to exploit. The strongly-regularised `min_samples_leaf=50` (≈0.9% of the training fold, against a default of 1) and `subsample=0.8` point the same way.

**Stated honestly, this is suggestive rather than conclusive:** the grid only offered depth 2 or 3, so "it rejected depth 3" is one bit of evidence, not a depth curve. And the balanced-weight GB search picked `max_depth=3` and scored 0.8485 — 0.0001 behind the depth-2 winner, i.e. indistinguishable. What the two searches agree on is the *regularisation*: both landed on `min_samples_leaf=50`, `subsample=0.8`, `learning_rate=0.1`, `n_estimators=100`. The defensible claim is that shallow, heavily-regularised, near-additive models win here — not that depth 2 specifically is optimal.

For comparison, the tuned Random Forest landed on `n_estimators=150, max_depth=10, min_samples_leaf=10, max_features='sqrt', class_weight='balanced'`, and the tuned HistGB on `learning_rate=0.05, max_iter=100, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=0.0, class_weight=None` — both also pulled hard toward regularisation.

---

## 5. Threshold — 0.5 is the wrong operating point

The decision threshold was chosen by maximising F1 on the positive class over **out-of-fold predicted probabilities on the training fold**, produced by `cross_val_predict(..., method='predict_proba')` with the same 5-fold splitter. It was *not* chosen on test; doing that would be tuning on the holdout.

```
OOF ROC-AUC on train = 0.8481
OOF F1(Yes) @ 0.500  = 0.5878
OOF F1(Yes) @ 0.385  = 0.6342     (+0.0465)
```

Chosen threshold: **0.385**. It transfers to test almost exactly as the OOF estimate predicted (+0.0505 there vs +0.0465 in OOF), which is what an honestly chosen threshold should do.

---

## 6. Feature importance — permutation, and where it disagrees with the EDA

Impurity ("Gini") importance is biased toward high-cardinality and continuous features, which on this matrix would systematically flatter `tenure`, `MonthlyCharges` and `TotalCharges` against 17 binary dummies. So the table below is **permutation importance** — mean drop in ROC-AUC over 5 shuffles, computed on the training fold with the fitted pipeline, at the level of the *raw* columns (so a multi-level categorical is permuted as one feature rather than as its dummies).

| Rank | Feature | Permutation ΔAUC | ± | EDA marginal (§6.1 V / §6.2 univ. AUC) | EDA rank |
|---|---|---|---|---|---|
| 1 | `tenure` | **0.06666** | 0.00282 | AUC = 0.740 | strongest numeric |
| 2 | `InternetService` | **0.05496** | 0.00142 | V = 0.323 | 4th categorical |
| 3 | `Contract` | **0.02938** | 0.00189 | V = 0.410 | **1st categorical** |
| 4 | `TotalCharges` | 0.00685 | 0.00049 | AUC = 0.652 | 2nd numeric |
| 5 | `PaymentMethod` | 0.00524 | 0.00077 | V = 0.303 | 5th |
| 6 | `MonthlyCharges` | 0.00518 | 0.00027 | AUC = 0.621 | 3rd numeric |
| 7 | `PaperlessBilling` | 0.00345 | 0.00050 | V = 0.192 | 10th |
| 8 | `OnlineSecurity` | 0.00246 | 0.00032 | V = 0.347 | **2nd categorical** |
| 9 | `TechSupport` | 0.00184 | 0.00024 | V = 0.343 | **3rd categorical** |
| 10 | `MultipleLines` | 0.00169 | 0.00029 | V = 0.040 | 14th (~noise) |
| 11 | `SeniorCitizen` | 0.00161 | 0.00036 | V = 0.151 | 12th |
| 12 | `StreamingMovies` | 0.00133 | 0.00025 | V = 0.231 | 8th |
| 13 | `StreamingTV` | 0.00129 | 0.00031 | V = 0.231 | 9th |
| 14 | `Dependents` | 0.00083 | 0.00012 | V = 0.164 | 11th |
| 15 | `OnlineBackup` | 0.00073 | 0.00027 | V = 0.292 | 6th |
| 16 | `DeviceProtection` | 0.00011 | 0.00005 | V = 0.282 | 7th |
| 17 | `Partner` | **0.00000** | 0.00000 | V = 0.150 | 13th |

### Does the model agree with the EDA? Partly — and the disagreements are systematic, not random.

**Agreements.** The top of the model's list is EDA's top: `tenure` (the strongest numeric, univariate AUC 0.740) and the `Contract`/`InternetService` pair that EDA §6.5 identified as the 70× interaction are the model's top three by an order of magnitude — together they are 0.15100 of a total 0.18361, i.e. **82.2% of all permutation importance**. `PaymentMethod` (V = 0.303) lands 5th in both. And EDA's two culled noise columns are consistent with what survives: `Partner` (V = 0.150) scores an exact **0.00000** here, so a third feature could arguably have been dropped.

**Disagreement 1 — the six add-on columns collapse.** `OnlineSecurity` (V = 0.347, EDA's **#2** categorical) and `TechSupport` (V = 0.343, **#3**) are the model's #8 and #9, at 0.00246 and 0.00184 — 22× and 30× below `InternetService`. `OnlineBackup` (V = 0.292) is #15 and `DeviceProtection` (V = 0.282) is #16 at 0.00011, essentially zero.

This is not a contradiction of the EDA, it is the difference between a **marginal** and a **conditional** statistic. Cramér's V measures each column against churn *on its own*. Step 2 §6.D established that all six add-on columns read `'No'` for the same 1526 no-internet customers (the collapse of `'No internet service'`). So a large part of each add-on's marginal association with churn *is* the internet effect wearing a different label. Once `InternetService` is in the model, permuting `OnlineSecurity` destroys very little that `InternetService` has not already supplied. **Marginal association ranks these features 2nd and 3rd; conditional contribution ranks them 8th and 9th.** Anyone reading the EDA's V table as a feature-selection ranking would keep the wrong columns.

**Disagreement 2 — `Contract` is 3rd, not 1st.** EDA's single strongest predictor (V = 0.410, a 39.88 pp spread) comes 3rd behind `tenure` and `InternetService`. Same mechanism: `tenure` and `Contract` are strongly entangled (a two-year contract holder has by construction survived longer), so much of `Contract`'s marginal power is already in `tenure` by the time `Contract` is permuted.

**Disagreement 3 — `TotalCharges` is not dead weight for a tree.** It ranks 4th (0.00685), *above* `MonthlyCharges`. Fact #6 ("99.91% explained by `tenure × MonthlyCharges`, drop it") is scoped to linear models, where the r = 0.8297 collinearity destabilises coefficients. A tree is indifferent to collinearity, and the residual — the part of `TotalCharges` *not* explained by `tenure × MonthlyCharges` (EDA: median 0.00, std 67.25, i.e. plan changes and promotions) — is apparently worth a little. Keeping it was the right call for this family.

---

## 7. The `MonthlyCharges` non-monotonicity — the most interesting result, and it corrects `CLAUDE.md` fact #8

Figure: `results/tree_ensembles_monthlycharges_pd.png`.

### 7.1 The decile probe

Out-of-fold predicted churn probability against observed churn, by `MonthlyCharges` decile on the **training fold** (5634 rows). The logistic regression here is a plain `LogisticRegression(max_iter=2000)` on the same pipeline with `scale=True`, included purely as a shape contrast — the team's actual linear result is the other agent's.

| D | n | `MonthlyCharges` range | actual churn | tree, mean OOF P | logistic, mean OOF P |
|---|---|---|---|---|---|
| 1 | 569 | 18.40–20.05 | 8.08% | 9.06% | 8.07% |
| 2 | 565 | 20.10–25.10 | 10.09% | 10.66% | 10.18% |
| 3 | 558 | 25.15–45.80 | 25.09% | 23.50% | 23.95% |
| 4 | 563 | 45.85–59.00 | 24.16% | 22.29% | 22.03% |
| 5 | 567 | 59.05–70.50 | 22.57% | 25.10% | 25.03% |
| 6 | 559 | 70.55–79.30 | 35.96% | 36.74% | 37.12% |
| 7 | 563 | 79.35–85.65 | 38.72% | 36.60% | 37.04% |
| 8 | 570 | 85.70–94.45 | 35.26% | 36.22% | 38.43% |
| 9 | 558 | 94.50–103.05 | **41.22%** | **41.30%** | **41.68%** |
| 10 | 562 | 103.10–118.75 | **24.56%** | **23.69%** | **22.13%** |

| D9 → D10 | change |
|---|---|
| actual | 41.22% → 24.56% (**−16.66 pp**) |
| tree, OOF | 41.30% → 23.69% (**−17.61 pp**) |
| logistic, OOF | 41.68% → 22.13% (**−19.55 pp**) |

**The observed non-monotonicity replicates.** EDA §6.7 reported 40.91% → 24.68% on all 7043 rows; the training fold gives 41.22% → 24.56%. Fact #8's *data* claim is confirmed.

**But both models reproduce it, and the logistic regression reproduces it slightly more sharply than the tree.** So the *inference* drawn from it — `CLAUDE.md` fact #8's "plain logistic regression gets the top decile backwards", and EDA §6.7's "this alone argues for tree-based models" — **is not supported.** A multivariate logistic regression on the Step 2 feature set predicts decile 10 correctly.

### 7.2 Why — and why the EDA's reasoning was nearly right

The EDA already contains the explanation without drawing the consequence: decile 10 is "40% two-year contracts with a mean of 4.80 add-ons, versus D9 which is 62% month-to-month". **The D9 → D10 fall is a composition effect carried by `Contract` and `InternetService`, not a `MonthlyCharges` main effect.** Any model holding those two columns gets the turn for free, linear or not. The non-monotonicity would only defeat a linear model fitted on `MonthlyCharges` *alone* — and no one was going to fit that.

The partial-dependence panel shows the two models arriving there by opposite routes:

- **Logistic PD runs 0.4475 → 0.1730 — monotonically *down*.** Its fitted `MonthlyCharges` coefficient is **negative** once `tenure`, `TotalCharges` and `InternetService` are in the design matrix, despite the raw point-biserial correlation being **+0.193**. That is a textbook suppression/collinearity sign flip, driven by `MonthlyCharges` being close to a proxy for `InternetService` (EDA §6.7: mean 91.50 fiber / 58.10 DSL / 21.08 none, near-disjoint ranges). The linear model does not model the decile curve through `MonthlyCharges` at all; it models it through the contract and service columns and lets `MonthlyCharges` absorb the leftover negative residual.
- **Tree PD is flat at ≈0.255–0.265 up to ~94, steps up to 0.3223 at 104.37, then falls to 0.3128 at the top of the range — a drop of 0.0094.** So the tree *does* encode a genuine local non-monotonicity, in the right place and the right direction. It is simply **tiny** — 0.94 pp of predicted probability, against the 16.66 pp swing the deciles actually show. The structure the trees were supposed to win on is worth about one percentage point of partial dependence.

**This is the clean answer to the brief.** Trees find the non-monotone effect that linear models supposedly cannot — and it turns out not to matter, because (a) it is small, and (b) the linear model never needed it, having reconstructed the same decile curve out of correlated features.

---

## 8. Final test evaluation — the holdout, read once

Model: `GradientBoosting (tuned)`, refit on all 5634 training rows, evaluated on the 1409 held-out rows.

| Metric | @ 0.500 | @ **0.385** (tuned on train OOF) | Δ |
|---|---|---|---|
| **ROC-AUC** | **0.8440** | 0.8440 | threshold-independent |
| **F1 (Yes)** | 0.5723 | **0.6228** | **+0.0505** |
| Precision (Yes) | 0.6552 | 0.5913 | −0.0639 |
| Recall (Yes) | 0.5080 | 0.6578 | **+0.1498** |
| Accuracy | 0.7984 | 0.7885 | −0.0099 |

Confusion matrices (rows = true 0/1, columns = predicted 0/1):

```
@0.500            @0.385
[[935, 100]       [[865, 170]
 [184, 190]]       [128, 246]]
```

**Read the confusion matrices, not the accuracy.** Moving the threshold from 0.5 to 0.385 converts **56 additional churners from missed to caught** (190 → 246 of 374, recall 50.80% → 65.78%) at a cost of 70 extra false alarms (100 → 170) and **0.99 pp of accuracy**. For a retention team, a false alarm costs one unnecessary outreach; a miss costs a customer. The 0.385 column is the operating point, and accuracy going *down* while the model gets *more useful* is exactly why fact #3 forbids accuracy as a headline.

### Sanity check against the EDA-predicted ceiling

| Bound (`CLAUDE.md` fact #4) | Achieved | Verdict |
|---|---|---|
| ROC-AUC 0.84–0.85 | **0.8440** | inside |
| F1 (Yes) 0.60–0.63 | **0.6228** | inside |
| AUC > 0.90 ⇒ leakage | 0.8440 | **no leakage signal** |
| All-`No` accuracy 73.46% | 78.85% test accuracy | +5.39 pp, and the point is the recall, not this |

Nothing here suggests a leak. The script prints a loud banner if test AUC exceeds 0.90; it did not fire.

### Against the linear model

From `results/logistic_regression.json` (the parallel Step 3a agent, identical frozen split):

| | CV ROC-AUC | Test ROC-AUC | Test F1(Yes) | Test recall(Yes) | threshold |
|---|---|---|---|---|---|
| LogisticRegression (best) | 0.84634 ± 0.00846 | 0.8424 | 0.6178 | 0.7326 | 0.32 |
| **GradientBoosting (tuned)** | **0.8486 ± 0.0084** | **0.8440** | **0.6228** | 0.6578 | 0.385 |
| Δ (tree − linear) | **+0.0023** | **+0.0016** | +0.0050 | −0.0748 | |

**The tree ensemble wins by 0.0016 test ROC-AUC — about one fifth of one CV standard deviation.** On 1409 test rows that difference is not distinguishable from noise, and the two models are not even better at the same thing: the logistic regression at its own tuned threshold catches more churners (recall 0.7326 vs 0.6578) while the tree is more precise. **The reputation of this dataset holds — logistic regression is very hard to beat on it, and this step did not beat it in any way a reviewer should believe.**

---

## 9. Limitations — stated plainly

1. **The search was small: 20 draws, 100 CV fits.** Deliberately, but it is a limitation. A larger `RandomizedSearchCV` might find another +0.002 for gradient boosting. It would not change the §7 or §8 conclusions, both of which turn on gaps far larger than 0.002 (the tree/linear gap is *smaller* than 0.002 and would stay inside the noise either way).
2. **Wall-clock is not compute.** The script reports 380.5 s, but three sibling Step 3 agents were training on the same 2-core sandbox throughout; the nominal cost is roughly 100 CV fits on a 5634 × 21 matrix. All `n_jobs` are pinned to 1 and `OMP_NUM_THREADS=1` is set before the sklearn import: on this box `HistGradientBoostingClassifier` measured **107.8 s per fit at `OMP_NUM_THREADS=2` against 2.8 s at 1** (a 38× penalty from OpenMP workers thrashing against joblib's processes), and one 5-fold RandomForest CV measured 9 s serial against ~150 s at `n_jobs=-1`. Anyone re-running this on a larger machine should re-tune those two settings rather than copy them.
3. **One split, one seed.** All CV numbers carry ±0.008–0.011 standard deviation across folds, and the test fold is 1409 rows (374 positives). Differences below ~0.01 AUC — which includes every difference in §3 between rows 7, 8, 9 and 10, and the entire tree-vs-linear gap — are **not** resolvable at this sample size. Repeated CV over several seeds would be the honest next step.
4. **`drop_total_charges=True` was not tested for trees.** §6 gives a reason to keep it (permutation rank 4, above `MonthlyCharges`) but that is not the same as having measured the drop.
5. **`xgboost` / `lightgbm` were not used.** Neither is installed, the assignment restricts the exercise to Module 3 algorithms, and nothing was installed. A tuned LightGBM would plausibly add ~0.002–0.005 AUC and would not change any conclusion here.
6. **Permutation importance on the training fold** measures what the *fitted* model relies on, not what a refitted model would lose. With two correlated features a model can lean on either; permuting one while the other stays available understates both. The `OnlineSecurity`/`InternetService` result in §6 is exactly this situation, so read that table as "what this model uses", not "what the business should stop measuring".
7. **The §7 logistic contrast is a diagnostic, not the team's linear result.** It is a single default-regularised `LogisticRegression` fitted inside this script to compare *curve shapes*. The team's linear numbers are Step 3a's.

---

## 10. What contradicts Steps 1–2

One item, and it is substantive.

> **`CLAUDE.md` fact #8 and EDA §6.7 overstate the case for trees.** The *data* claim is confirmed — `MonthlyCharges` really is non-monotone, D9 41.22% → D10 24.56% on the training fold against EDA's 40.91% → 24.68% on the full set. The *modelling* claim — "plain logistic regression gets the top decile backwards", "this alone argues for tree-based models" — is not borne out. A multivariate logistic regression on the Step 2 feature set predicts decile 10 at 22.13% against an actual 24.56%, marginally *closer* to the truth than the tree's 23.69%. The fall is a composition effect carried by `Contract` and `InternetService` (decile 10 is 40% two-year contracts, per EDA §6.7's own explanation), and any model holding those columns gets it for free.

Suggested amendment for whoever owns `CLAUDE.md`, since it should not be silently edited by this agent:

> **8. `MonthlyCharges` is non-monotone** against churn — it rises to 40.91% at decile 9, then falls to 24.68% at decile 10. This is a composition effect: decile 10 is 40% two-year contracts. *Any* model that includes `Contract` and `InternetService` reproduces it, logistic regression included (verified in `docs/step-03b-tree-ensembles.md` §7); only a model fitted on `MonthlyCharges` alone would get it backwards. Note also that `MonthlyCharges`' partial effect flips sign — raw correlation +0.193, but a negative logistic coefficient once `InternetService` and `tenure` are in the model.

Everything else in Steps 1 and 2 held: the split reproduced its documented row counts and churn rates exactly, the encoded width is 21, the collapse of `'No internet service'` is visible in the importance table as the six add-on columns being conditionally redundant with `InternetService`, and the predicted ROC-AUC / F1 ceiling was accurate to two decimal places.

---

## 11. Reproducing

```bash
python3 src/models/tree_ensembles.py
```

Prints every table in this document and writes `results/tree_ensembles.json` and `results/tree_ensembles_monthlycharges_pd.png`. Deterministic: `random_state=42` on every estimator, the searcher and the CV splitter.
