# Step 3c — Distance-based and probabilistic models (KNN, Naive Bayes)

**Date:** 2026-09-01
**Status:** Complete
**Artifacts produced:** `src/models/knn_naive_bayes.py`, `results/knn_naive_bayes.json`
**Reproduce:** `python3 src/models/knn_naive_bayes.py` (~140 s, 182 CV variants)
**Split:** the frozen `data/splits/` holdout via `dp.load_splits()` — never re-split
**CV:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`, **train only**
**Test set:** read exactly once, in `final_evaluation()` (script §5)

---

## 0. What this step was for

This is one of four parallel Step 3 tracks. It owns the two Module 3 families with the
worst *structural* fit for this dataset: **K-Nearest Neighbours** (distance-based) and
**Naive Bayes** (probabilistic). Both are expected to land below the team's
ROC-AUC 0.84–0.85 ceiling.

The deliverable is therefore not a competitive score. It is **a defensible, numeric
reason to reject these two families**, so the synthesis cell can say *why* rather than
just *which was lower*. Everything below is observed output from the script, not
estimation.

### Headline

| Family | Best config | CV ROC-AUC | Test ROC-AUC | Test F1(Yes) @ tuned t |
|---|---|---|---|---|
| **Naive Bayes** | discretised (10 quantile bins) + `BernoulliNB(α=1)` | **0.8410 ± 0.0096** | **0.8389** | **0.6170** (t = 0.490) |
| **KNN** | full 21-col matrix, k = 101, uniform, manhattan | **0.8393 ± 0.0089** | **0.8363** | **0.6141** (t = 0.400) |

Both land **0.001–0.011 ROC-AUC below the 0.84–0.85 ceiling** and at the **bottom of
the 0.60–0.63 F1 band**. Neither is a catastrophe on the metric — and that is itself
the finding worth reporting honestly. The case against them is not the score; it is
§7.

Nothing exceeds AUC 0.90, so no leakage alarm was triggered. CV → test drift is
−0.0021 (NB) and −0.0030 (KNN), i.e. the CV estimates held.

---

## 1. Setup, and the one rule that mattered most

```
frozen split: train (5634, 17) / test (1409, 17)
train churn 26.5353% (1495 pos)  |  test churn 26.5436% (374 pos)
all-'No' baseline accuracy on test: 73.4564%
```

**Scaling stays on** (`build_preprocessor(scale=True)`). This is not a formality for
this track: `tenure` spans 0–72 while `TotalCharges` spans 0–8684.80, a ~120:1 range
ratio, so an unscaled Euclidean distance is `TotalCharges` and nothing else. The
`ColumnTransformer` is unfitted at construction and refitted inside the `Pipeline` on
each training fold — the scaler never sees a validation fold's mean or standard
deviation.

Encoded matrix: **5634 × 21** — 3 standardised continuous columns
(`tenure`, `MonthlyCharges`, `TotalCharges`), then 18 columns that are strictly 0/1
(`SeniorCitizen` + 17 one-hot dummies; verified `True` at runtime).

---

## 2. The geometry of the encoded matrix — why distance is in trouble here

Measured on the training fold only, before any model was fitted.

### 2.1 Where the distance actually comes from

Squared Euclidean distance decomposes additively across dimensions, and the expected
per-column contribution has a closed form: `2·Var` for a standardised column, `2p(1−p)`
for a 0/1 column.

| Block | Columns | E[squared distance] | Share |
|---|---|---|---|
| Continuous (standardised) | 3 | 6.000 | **44.4%** |
| Binary dummies | 18 | 7.510 | **55.6%** |

**55.6% of the metric is, arithmetically, a Hamming count** — on 0/1 columns the squared
difference *is* the mismatch indicator. Per dimension the continuous columns are worth
2.000 each against 0.417 for the average dummy, but there are six times as many dummies,
so the dummy block wins the majority of the distance budget.

### 2.2 Distance concentration

Three statistics, averaged over 300 randomly chosen query points against all 5634 rows:

| Space | rel. contrast `(dmax−dmin)/dmin` | **std/mean** | **d(100th)/median** |
|---|---|---|---|
| full 21-d | 135.2 | 0.2472 | 0.5208 |
| 3-d continuous only | 625.9 | **0.4878** | **0.1234** |
| 18-d binary only | 2.7 | **0.1734** | **0.5907** |

Relative contrast is the textbook statistic but its denominator is the distance to the
*nearest* point, so one near-duplicate row blows it up — it is reported for completeness
and not leaned on. The load-bearing columns are the other two:

- **std/mean** falls from 0.488 (3-d continuous) to 0.247 (full 21-d) to 0.173 (18-d
  binary). Distances bunch together; "nearest" becomes less distinguishable from
  "typical". This is the curse of dimensionality, measured.
- **d(100th)/median**: in the 3-d continuous space the 100th nearest neighbour sits at
  12.3% of the median distance to a random point — a genuinely *local* neighbourhood.
  In the full 21-d space it sits at 52.1%, and in the binary space at 59.1%. The
  "neighbourhood" is over half-way to the whole dataset.

### 2.3 Ties — the binary space is a lattice, not a metric space

Among the 25 nearest neighbours of a sampled point:

| Space | points with ≥1 exact distance tie | mean tied distances (of 25) |
|---|---|---|
| full 21-d | 9.0% | 0.29 |
| 3-d continuous | 11.0% | 1.15 |
| **18-d binary** | **100.0%** | **22.71** |

In the pure dummy space **every** sampled point has ties and **22.71 of its 25 nearest
neighbours are at duplicate distances**. There are only ~19 possible squared distances
in an 18-bit Hamming space, so "which 25 are nearest" is decided by tie-break order,
not by similarity. The 3 continuous columns are the only thing rescuing the full matrix
from this — and they are 44.4% of it.

---

## 3. KNN

Grid: `n_neighbors` ∈ {1, 3, 5, 9, 15, 21, 31, 45, 61, 81, 101, 151, 201, 301} ×
`weights` ∈ {uniform, distance} × `metric` ∈ {euclidean, manhattan}, in each of three
feature spaces (full 21, continuous-3, binary-18) = 168 CV runs.

### 3.1 CV ROC-AUC against k (euclidean)

| k | full21 uniform | full21 distance | cont3 uniform | cont3 distance |
|---:|---:|---:|---:|---:|
| 1 | 0.6671 | 0.6671 | 0.6412 | 0.6412 |
| 3 | 0.7514 | 0.7493 | 0.7332 | 0.7240 |
| 5 | 0.7860 | 0.7762 | 0.7639 | 0.7488 |
| 9 | 0.8087 | 0.7965 | 0.7839 | 0.7671 |
| 15 | 0.8213 | 0.8090 | 0.7979 | 0.7774 |
| 21 | 0.8284 | 0.8169 | 0.8054 | 0.7834 |
| 31 | 0.8328 | 0.8221 | 0.8098 | 0.7882 |
| 45 | 0.8362 | 0.8264 | 0.8138 | 0.7927 |
| **61** | **0.8375** | 0.8283 | 0.8138 | 0.7943 |
| 81 | 0.8370 | 0.8292 | 0.8148 | 0.7959 |
| 101 | 0.8369 | 0.8301 | **0.8149** | 0.7966 |
| 151 | 0.8367 | 0.8309 | 0.8139 | 0.7973 |
| 201 | 0.8356 | 0.8308 | 0.8123 | 0.7970 |
| 301 | 0.8327 | 0.8297 | 0.8096 | 0.7963 |

**k = 1 scores 0.6671.** A one-nearest-neighbour classifier on this matrix is barely
better than a coin flip on ranking. The curve needs **k ≈ 61–101** to peak — roughly
1.4–2.2% of the 4507 rows in each training fold — and is flat from k = 45 to k = 201
(0.8356–0.8375, a 0.0019 spread against a fold std of 0.0089). That flatness is the
result: the model is not finding local structure and then stopping; it is averaging over
a large enough crowd that the noisy neighbourhood ordering washes out. **A KNN that
needs 101 neighbours has stopped being a local method** and is closer to a coarse kernel
smoother over the whole training set.

**Distance weighting hurts at every k** (0.8369 vs 0.8301 at k = 101, and worse the
larger k gets). It is the diagnostic from §2.3 showing up in the score: weighting by
1/d rewards the ties and near-ties that the binary block manufactures, so it amplifies
exactly the arbitrary part of the ordering.

### 3.2 Manhattan vs Euclidean

| Space | best k (uniform) | euclidean | manhattan | Δ |
|---|---|---|---|---|
| full21 | 61 | 0.8375 | 0.8380 | **+0.0005** |
| cont3 | 101 | 0.8149 | 0.8160 | **+0.0011** |
| bin18 | 151 | 0.8180 | 0.8180 | **+0.0000** |

Manhattan is nominally better, as the high-dimensional-binary literature predicts, but
by +0.0005 to +0.0011 against a fold-to-fold std of ~0.009 — **an order of magnitude
smaller than the noise**. Reported, not claimed. (On the binary block the two metrics
are exactly rank-equivalent — L1 and squared-L2 are both the Hamming count there — hence
the exact 0.0000.)

### 3.3 Full matrix vs continuous-only vs binary-only

Identical CV, identical rows, identical scaling:

| Feature space | Best config | CV ROC-AUC |
|---|---|---|
| **full 21 columns** | k=101, uniform, manhattan | **0.8393 ± 0.0089** |
| 18 binary columns only | k=151, uniform, euclidean | 0.8180 ± 0.0058 |
| 3 continuous columns only | k=101, uniform, manhattan | 0.8160 ± 0.0088 |

**This refutes the obvious hypothesis, and the refutation is the interesting part.**
The pre-registered expectation was that the 18 dummies would drag the metric down and
that a continuous-only KNN would beat the full matrix. It does not: dropping the dummies
costs **−0.0233 ROC-AUC**, and dropping the continuous columns costs −0.0213. Each block
carries real, largely complementary signal (`Contract`, `InternetService`, `TechSupport`
live in the dummies; `tenure` lives in the continuous block), and KNN needs both.

So the damage the dummies do is **not** visible in ranking quality. It is visible in
§2 and in §3.1 — they destroy *locality*, forcing k up to ~100 and killing distance
weighting — and in §7.2, where it costs at scoring time. The honest statement is:
*the encoded geometry is bad for KNN as a method, and KNN survives it by giving up on
being local.*

---

## 4. Naive Bayes

sklearn ships no mixed-type Naive Bayes, so both routes were implemented rather than
one being assumed better:

1. **`MixedNB`** (`src/models/knn_naive_bayes.py`) — `GaussianNB` on the 3 continuous
   columns, `BernoulliNB` on the 18 binary ones, combined by adding log-posteriors and
   subtracting the class prior that would otherwise be counted twice. Naive Bayes
   factorises as `log P(y|x) = log P(y) + Σ log P(x_j|y) + const`, so two NB models over
   disjoint column blocks compose exactly; per-sample constants vanish in the softmax.
   Public API only, no `_joint_log_likelihood`.
2. **Discretisation** — `KBinsDiscretizer(strategy='quantile', encode='onehot-dense')` on
   the 3 continuous columns inside the pipeline (fitted per fold), then a single
   `BernoulliNB` over the whole now-binary matrix.

> **Implementation note worth keeping.** `MixedNB` was first declared as
> `class MixedNB(BaseEstimator, ClassifierMixin)`. Under scikit-learn ≥ 1.6 that mixin
> order makes `BaseEstimator.__sklearn_tags__` win, `is_classifier()` returns `False`,
> and the `roc_auc` scorer hands `roc_curve` the full (n, 2) probability matrix instead
> of column 1. This surfaces as a silent **`nan` CV score**, not as an error. The
> correct order is `(ClassifierMixin, BaseEstimator)`.

### 4.1 CV ROC-AUC by variant

| Variant | Columns used | CV ROC-AUC |
|---|---|---|
| `nb_gaussian_cont3` — `GaussianNB` | 3 continuous | 0.7746 ± 0.0193 |
| `nb_bernoulli_bin18` — `BernoulliNB` | 18 binary | 0.8190 ± 0.0030 |
| `nb_mixed_full21` — `MixedNB` | all 21 | 0.8327 ± 0.0089 |
| `nb_discretised_bins3` | all 21, 3 bins | 0.8379 ± 0.0101 |
| `nb_discretised_bins5` | all 21, 5 bins | 0.8406 ± 0.0106 |
| **`nb_discretised_bins10`** | all 21, 10 bins | **0.8410 ± 0.0096** |
| `nb_discretised_bins20` | all 21, 20 bins | 0.8400 ± 0.0096 |

The `GaussianNB` block is the weakest thing in this entire report at 0.7746, with the
largest fold variance (±0.0193). Combining the blocks helps (+0.0137 over Bernoulli
alone), and **discretising helps more (+0.0083 over `MixedNB`)** — which is the tell in
§4.3.

Bin count is a plateau, not a peak: 5, 10 and 20 bins span 0.8400–0.8410, well inside
one fold std. 10 was taken as the argmax; 5 would be a defensible simpler choice.

### 4.2 The independence violation, measured directly

Naive Bayes assumes `P(x_i, x_j | y) = P(x_i|y)·P(x_j|y)`, i.e. **every within-class
feature correlation is exactly 0**. Observed on the training fold:

**Continuous block, within-class Pearson r** (NB assumes all off-diagonals are 0.000):

| Churn = 0 (n = 4139) | tenure | MonthlyCharges | TotalCharges |
|---|---:|---:|---:|
| tenure | 1.000 | 0.345 | **0.799** |
| MonthlyCharges | 0.345 | 1.000 | **0.761** |
| TotalCharges | 0.799 | 0.761 | 1.000 |

| Churn = 1 (n = 1495) | tenure | MonthlyCharges | TotalCharges |
|---|---:|---:|---:|
| tenure | 1.000 | 0.391 | **0.952** |
| MonthlyCharges | 0.391 | 1.000 | 0.543 |
| TotalCharges | 0.952 | 0.543 | 1.000 |

Within the churner class, `corr(tenure, TotalCharges) = 0.952`. This is EDA's
`TotalCharges ≈ tenure × MonthlyCharges` (OLS R² = 0.9991) restated inside each class:
churners are concentrated at short tenure where the product is dominated by `tenure`,
so the two columns are nearly the same variable. Naive Bayes multiplies their
likelihoods as if they were two independent pieces of evidence.

**Binary block, within-class |r| over all 153 pairs** (`BernoulliNB` assumes 0 too):

| Class | mean \|r\| | max \|r\| |
|---|---:|---:|
| Churn = 0 | 0.168 | 0.557 |
| Churn = 1 | 0.125 | 0.523 |

Top dependent pairs (Churn = 0): `StreamingMovies_Yes ~ StreamingTV_Yes` 0.557,
`InternetService_No ~ OnlineBackup_Yes` 0.473, `InternetService_No ~ StreamingMovies_Yes`
0.471, `InternetService_No ~ StreamingTV_Yes` 0.467, `DeviceProtection_Yes ~
StreamingMovies_Yes` 0.465. Note these survive the Step 2 cleaning: collapsing
`'No internet service' → 'No'` removed the *perfectly* collinear dummies, but the
add-on services remain strongly co-purchased, which is a real customer behaviour, not
an encoding artefact.

### 4.3 What the violation costs — two independent measurements that agree

**Measurement A — drop the redundant feature.** Refit every variant with
`clean(drop_total_charges=True)`, re-indexed onto the frozen split's `row_id`s
(`X_notc.loc[X_tr.index]`, asserted identical order and identical `y`). **This is not a
re-split** — same rows, one fewer column.

| Variant | with `TotalCharges` | without | Δ |
|---|---:|---:|---:|
| `nb_gaussian_cont3` | 0.7746 | 0.7935 | **+0.0189** |
| `nb_bernoulli_bin18` | 0.8190 | 0.8190 | +0.0000 *(control)* |
| `nb_mixed_full21` | 0.8327 | 0.8329 | +0.0002 |
| `nb_discretised_bins3` | 0.8379 | 0.8348 | −0.0031 |
| `nb_discretised_bins5` | 0.8406 | 0.8368 | −0.0038 |
| `nb_discretised_bins10` | 0.8410 | 0.8362 | −0.0049 |
| `nb_discretised_bins20` | 0.8400 | 0.8362 | −0.0038 |

**`GaussianNB` gets 1.9 AUC points *better* when a feature is deleted** — the classic
signature of double-counted evidence, not of a weak feature. The `bin18` row is the
control: it never touches `TotalCharges` and moves by exactly 0.0000, confirming the
effect is specific rather than fold noise.

The discretised variants move the other way (−0.0031 to −0.0049): once each continuous
column is chopped into 10 quantile bins, its likelihood is a coarse histogram rather
than a sharp Gaussian, the double-counting is far weaker, and `TotalCharges` reverts to
being a mildly useful feature. That is *why* discretisation wins §4.1 — it is not a
better model of the data, it is a **worse-resolution model whose errors happen to be
less correlated**.

**Measurement B — remove the assumption, change nothing else.** `GaussianNB` and
`QuadraticDiscriminantAnalysis` fit the *same* generative model — one Gaussian per class
— and differ in exactly one respect: NB forces the covariance matrix diagonal, QDA
estimates it in full. Same 3 columns, same CV, same folds:

| Model on the 3 continuous columns | CV ROC-AUC |
|---|---|
| `GaussianNB` (diagonal covariance — the naive assumption) | 0.7746 ± 0.0193 |
| `QDA` (per-class full covariance) | 0.7932 ± 0.0090 |
| `LDA` (one shared full covariance) | **0.8008 ± 0.0086** |

**QDA − GaussianNB = +0.0186 ROC-AUC.** That is an isolated price tag on the
conditional-independence assumption, with nothing else varying — and it lands within
0.0003 of Measurement A's +0.0189. Two different experiments, one number. Modelling the
covariance also **halves the fold variance** (±0.0193 → ±0.0090): the naive model is not
just biased, it is unstable across folds.

*(LDA and QDA are diagnostics here, not candidate models — they are recorded under
`diagnostics.independence` in the JSON, not in `variants`.)*

---

## 5. Calibration

Out-of-fold TRAIN probabilities, 5-fold. This matters because the team is tuning a
decision threshold on these numbers: a miscalibrated score moves the threshold and makes
any "customers with churn risk above 60%" business rule meaningless, even when the
*ranking* is fine. The no-skill constant predictor at the 26.54% base rate scores
Brier 0.1949.

| Model | Brier | ECE | share p < 0.01 | share p > 0.99 | min p | max p |
|---|---:|---:|---:|---:|---:|---:|
| KNN (k=101, uniform, manhattan) | **0.1392** | **0.0327** | 8.7% | 0.0% | 0.0000 | 0.8614 |
| NB (discretised, 10 bins) | 0.1668 | **0.1310** | **34.0%** | **4.3%** | 0.0000 | 0.9993 |

**Reliability curves** (10 quantile bins, predicted vs observed churn rate):

```
KNN  predicted: 0.011 0.038 0.077 0.143 0.212 0.285 0.383 0.489 0.609 0.736
     observed : 0.016 0.021 0.061 0.119 0.138 0.243 0.309 0.435 0.595 0.744

NB   predicted: 0.000 0.001 0.003 0.013 0.051 0.161 0.416 0.724 0.922 0.986
     observed : 0.009 0.028 0.062 0.098 0.146 0.229 0.303 0.469 0.572 0.738
```

Naive Bayes is **exactly as overconfident as its reputation**. Its top bin predicts
98.6% and observes 73.8%; its second bin predicts 0.1% and observes 2.8%. **34.0% of
its out-of-fold predictions are below 0.01** and 4.3% are above 0.99 — it has pushed a
third of the training set to near-certainty. ECE is 0.1310, four times KNN's 0.0327.
This is the direct consequence of §4.2: multiplying 21 correlated likelihoods as if
independent compounds the same evidence repeatedly, and the posterior saturates.

KNN, by construction, cannot do this: with k = 101 and uniform weights its output is a
count out of 101, so it is bounded at [0, 0.8614] here and lands within ~0.05 of the
diagonal in 8 of 10 bins. **The worse model has the better probabilities** — worth
saying out loud, because it means "NB scored higher on AUC" does not make it the better
choice for a threshold-driven retention workflow.

Test-set Brier confirms it: KNN 0.1413, NB 0.1725.

---

## 6. Threshold tuning and the final test evaluation

Thresholds were chosen by scanning 193 values in [0.02, 0.98] against **out-of-fold
TRAIN predictions** (`cross_val_predict`, same 5 folds), maximising F1 on the `Yes`
class. The test set played no part in the choice.

| Model | tuned t | OOF F1 @ t | OOF F1 @ 0.5 |
|---|---:|---:|---:|
| KNN | 0.400 | 0.6281 | 0.6003 |
| NB (discretised) | 0.490 | 0.6342 | 0.6337 |

Tuning is worth **+0.0278 F1** to KNN and essentially nothing (+0.0005) to NB — NB's
saturated probability distribution (§5) leaves almost no mass near the boundary, so
moving the threshold barely reclassifies anyone. Another calibration symptom.

### Final test numbers — the test set's single read

**`nb_discretised_bins10`** (the best of my two families):

| Threshold | ROC-AUC | F1(Yes) | Precision(Yes) | Recall(Yes) | Accuracy |
|---|---:|---:|---:|---:|---:|
| 0.500 | 0.8389 | 0.6159 | 0.5546 | 0.6925 | 0.7708 |
| **0.490 (tuned)** | **0.8389** | **0.6170** | 0.5530 | 0.6979 | 0.7700 |

Confusion matrix `[[TN, FP], [FN, TP]] = [[824, 211], [113, 261]]` · Brier 0.1725

**`knn_full21_k101_uniform_manhattan`:**

| Threshold | ROC-AUC | F1(Yes) | Precision(Yes) | Recall(Yes) | Accuracy |
|---|---:|---:|---:|---:|---:|
| 0.500 | 0.8363 | 0.5890 | 0.6039 | 0.5749 | 0.7871 |
| **0.400 (tuned)** | **0.8363** | **0.6141** | 0.5483 | 0.6979 | 0.7672 |

Confusion matrix `[[820, 215], [113, 261]]` · Brier 0.1413

Both beat the 73.46% all-"No" accuracy baseline, but only by ~3.5 pp at the tuned
threshold — accuracy is not the headline and is reported only for completeness.

**No leakage indicators.** Max AUC anywhere in this track is 0.8410, far under the 0.90
alarm line. CV → test drift is −0.0021 (NB) and −0.0030 (KNN).

### Selection-bias caveat

182 configurations were scored on the same 5 CV folds, so the *winner's* CV mean is
optimistically biased by maximum-selection. Two things bound the damage: the k-curve is
flat over a wide plateau (k = 45–201 spans 0.0019) rather than spiky, and the bin-count
curve is likewise flat (5/10/20 bins span 0.0010) — so the argmax was not chasing a
noise spike. The held-out test numbers, which are selection-free, came in 0.002–0.003
below CV, consistent with a small bias.

### `top_drivers` is empty, deliberately

Neither family yields a natural feature-importance ranking. KNN has no coefficients and
no splits. NB has per-feature log-likelihood ratios, but they are not comparable across
a mixed Gaussian/Bernoulli model and — per §4.2 — are inflated precisely by the
dependence that the model ignores, so a ranking read off them would over-credit whichever
correlated block has the most columns. A permutation importance could be computed, but
it would describe the *pipeline*, not the model family, and would invite exactly the
false comparison against the tree models' native importances. Reporting none is more
honest than inventing one.

---

## 7. Why these families underperform on this dataset

Four reasons, in descending order of how much they should weigh on the team's decision.
Note that reason 1 is **not** "the score is low" — the scores are close. It is that the
score is only reachable by abandoning what makes the method a method.

### 7.1 KNN reaches its score by ceasing to be local

k = 1 scores **0.6671**; the model needs **k = 61–101** (1.4–2.2% of each training fold)
to reach 0.8393, and the curve is flat across k = 45–201. A KNN averaging over 101
neighbours is a coarse kernel smoother, not a nearest-neighbour classifier — it is
working *despite* its neighbourhood structure, by averaging enough of it away. The
geometry says why: 55.6% of the distance metric is a Hamming count over 18 dummies
(§2.1); distance concentration std/mean falls from 0.488 in the 3 continuous dimensions
to 0.247 in the full 21 (§2.2); the 100th neighbour already sits at 52% of the median
distance to a random point, so the "neighbourhood" is half the dataset (§2.2); and in
the pure dummy space 100% of points have tied neighbour distances with 22.71 of 25 tied
(§2.3). Distance weighting — the standard fix for "k is too large" — makes things
*worse* at every k, because it up-weights exactly those manufactured ties.

The continuous-only experiment refutes the simple version of this story and should be
reported as such: dropping the 18 dummies costs −0.0233 AUC, so they carry real signal.
The dummies do not destroy KNN's accuracy. They destroy its locality, and a KNN without
locality has no advantage left over a model that states its decision boundary explicitly.

### 7.2 KNN's costs land at scoring time, where they hurt most

There is no fitted model — the "model" is all 5634 training rows, and every prediction
is a full 21-dimensional distance computation against all of them. For a retention team
scoring a customer base nightly this is the worst possible cost profile: no compact
artefact to ship or version, cost growing with the customer base, and (per §7.4) no
answer to "why was this customer flagged?".

### 7.3 Naive Bayes's core assumption is violated by construction, and it costs ~0.019 AUC

The dataset's central structural fact — `TotalCharges ≈ tenure × MonthlyCharges` at
R² = 0.9991 — is a *deterministic dependence between three of the features NB assumes
are conditionally independent*. Within the churner class `corr(tenure, TotalCharges) =
0.952`; among the 18 dummies the mean within-class |r| is 0.168 with a max of 0.557
(§4.2). Two independent measurements price the violation at almost exactly the same
figure: deleting the redundant feature makes `GaussianNB` **+0.0189 better**, and
replacing the diagonal covariance with a full one (QDA, otherwise the identical
generative model) is worth **+0.0186** — and halves the fold-to-fold variance
(±0.0193 → ±0.0090).

The winning NB variant is itself evidence for this. `nb_discretised_bins10` beats
`MixedNB` by +0.0083 not because 10-bin histograms model `tenure` better than a Gaussian,
but because coarsening each feature *weakens the correlations NB is mishandling*. The
best Naive Bayes on this data is the one that has been degraded until its own assumption
hurts less. That is not a model family being used well.

### 7.4 The probabilities are not usable, and neither family explains itself

NB's ECE is 0.1310 with 34.0% of predictions below 0.01 and 4.3% above 0.99; its top
reliability bin predicts 98.6% against 73.8% observed (§5). The team's workflow is
threshold-driven, and a saturated score means the threshold has almost no traction —
tuning it moved OOF F1 by +0.0005. Calibration is fixable (`CalibratedClassifierCV`),
but fixing it costs a nested CV loop and lands you at a model that still has no feature
importances, no coefficients, no rules — nothing to hand a retention team that answers
"why this customer?". A logistic regression or a tree ensemble at the same or better AUC
gives that away for free.

### The one-line version for the synthesis cell

> KNN and Naive Bayes reach 0.836–0.839 test ROC-AUC, within ~0.01 of the ceiling — but
> KNN only gets there by averaging 101 neighbours in a space where 55.6% of the distance
> is a Hamming count and the 100th neighbour sits at half the dataset's median distance,
> and Naive Bayes only gets there after discretisation blunts an independence assumption
> that this dataset violates by construction (`TotalCharges ≈ tenure × MonthlyCharges`,
> R² = 0.9991), a violation priced at ~0.019 AUC by two independent measurements. Both
> then fail to produce a usable driver list, and NB's probabilities are badly
> miscalibrated (ECE 0.1310) for a threshold-driven workflow. Rejected on structure, not
> on score.

---

## 8. What I would try next (if this track were being pursued)

Ranked by expected value, and honest that none of these is likely to clear 0.85:

1. **A metric that respects the column types.** Gower distance, or HEOM — treat the
   binary block as a match/mismatch and the continuous block as a scaled range,
   normalised separately. This directly attacks §2.1 rather than working around it.
   Not in sklearn; `gower` or a custom `metric=callable` would be needed, and the
   callable route is roughly 50× slower.
2. **Supervised dimensionality reduction before the distance.** `NeighborhoodComponents
   Analysis` learns a linear map that maximises leave-one-out KNN accuracy — it is the
   textbook answer to a bad distance metric and would let k drop back to a genuinely
   local value.
3. **Calibrate the NB.** `CalibratedClassifierCV(method='isotonic', cv=5)` wrapped
   around the discretised model, tuned in a nested loop. It would not move the AUC
   (isotonic is monotone) but would make the threshold mean something. Worth ~20
   minutes if NB were being kept.
4. **Feed the interaction, don't hide it.** Replace `TotalCharges` with the residual
   `TotalCharges − tenure × MonthlyCharges` (EDA: median 0.00, std 67.25 — it captures
   plan changes and promotions). This is the *right* fix for §7.3: it removes the
   deterministic dependence while keeping the genuinely new information, instead of
   deleting the column.
5. **Drop `SeniorCitizen` from the distance, or weight it.** It is the one binary column
   that is passed through rather than derived from a one-hot, and at p ≈ 0.16 it
   contributes only 0.27 of the 7.51 binary distance budget while being a real predictor.
   A weighted metric could give it its due — but see item 1, which subsumes this.

---

## 9. Files

| Path | Contents |
|---|---|
| `src/models/knn_naive_bayes.py` | The whole track, runnable end to end (~140 s). Sections mirror this document. |
| `results/knn_naive_bayes.json` | Machine-readable results in the shared four-agent schema: 182 `variants`, `best_model`, plus a `diagnostics` block carrying the geometry, independence and calibration measurements. |
