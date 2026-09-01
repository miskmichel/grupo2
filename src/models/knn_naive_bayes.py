"""Step 3c -- distance-based (KNN) and probabilistic (Naive Bayes) models.

Run: ``python3 src/models/knn_naive_bayes.py``

Scope
-----
These are the two Module 3 families with the *worst* structural fit for this
dataset.  The point of this script is not to squeeze out a competitive score --
it is to measure, with numbers, exactly why they underperform, so the team's
synthesis cell can reject them for a stated reason rather than a shrug.

Two structural problems are quantified here:

1. **KNN / curse of dimensionality.**  The encoded design matrix is 21 columns
   of which 18 are 0/1 (17 one-hot dummies + ``SeniorCitizen``) and only 3 are
   continuous.  Euclidean distance over a mostly-binary vector degenerates
   towards a Hamming count, distances concentrate, and "nearest" stops meaning
   "similar".  Measured two ways: (a) the share of mean squared distance that
   the binary block contributes, (b) relative contrast
   ``(d_max - d_min) / d_min`` for the full 21-d matrix vs the 3-d continuous
   sub-matrix, (c) a full-matrix vs continuous-only KNN grid under identical CV.

2. **Naive Bayes / conditional-independence violation.**  EDA established
   ``TotalCharges ~ tenure x MonthlyCharges`` at R^2 = 0.9991 and
   ``corr(tenure, TotalCharges) = 0.826``.  NB multiplies per-feature
   likelihoods as if they were independent, so a feature that is a
   deterministic function of two others gets its evidence counted three times.
   Measured by refitting every NB variant with ``drop_total_charges=True``
   (re-indexed onto the *frozen* split's ``row_id``s -- never re-split) and by
   the calibration/Brier gap.

Rules honoured (identical for all four Step 3 agents):
  * the FROZEN split from ``dp.load_splits()``; ``dp.split()`` is never called;
  * the test set is touched exactly ONCE, in ``final_evaluation()``;
  * everything inside a ``Pipeline`` fitted per fold -- no transformer ever sees
    a validation fold before it is scored;
  * scaling stays ON (``scale=True``): ``tenure`` 0-72 against ``TotalCharges``
    0-8684.80 is a ~120:1 range ratio, so an unscaled Euclidean distance is
    ``TotalCharges`` and nothing else;
  * primary metric ROC-AUC, with F1/precision/recall on ``Yes`` = 1 alongside;
    accuracy is never the headline (the all-"No" baseline is 73.46%).
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.extmath import softmax

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import data_prep as dp  # noqa: E402

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
OUT_JSON = ROOT / "results" / "knn_naive_bayes.json"

warnings.filterwarnings("ignore", category=FutureWarning)
# KBinsDiscretizer(strategy="quantile") drops degenerate bins on `tenure`, which is
# integer-valued with heavy ties at 1 and 72.  Expected; the bin count is a tuned
# hyper-parameter and the warning fires once per fold.
warnings.filterwarnings("ignore", message="Bins whose width are too small")


# --------------------------------------------------------------------------- #
# Mixed-type Naive Bayes
# --------------------------------------------------------------------------- #
class MixedNB(ClassifierMixin, BaseEstimator):
    """Gaussian likelihood on the first ``n_continuous`` columns, Bernoulli on
    the rest, combined by adding log-likelihoods.

    sklearn ships no mixed-type NB.  Naive Bayes factorises
    ``log P(y|x) = log P(y) + sum_j log P(x_j|y) + const``, so two fitted NB
    models over disjoint column blocks compose exactly by summing their
    posteriors and subtracting the class prior that would otherwise be counted
    twice.  ``predict_log_proba`` of each sub-model differs from its joint
    log-likelihood only by a per-sample constant, and per-sample constants
    vanish in the final softmax -- so this uses public API only and is
    mathematically identical to a native implementation.

    ``build_preprocessor`` emits columns in the order (numeric, binary,
    categorical), so "the first 3 columns are the continuous ones" holds by
    construction; ``n_continuous`` is passed explicitly rather than sniffed.

    Note the mixin order ``(ClassifierMixin, BaseEstimator)``.  Reversed, under
    scikit-learn >= 1.6 the tag resolution puts ``BaseEstimator.__sklearn_tags__``
    first, ``is_classifier()`` returns False, and the ``roc_auc`` scorer hands
    ``roc_curve`` the full (n, 2) probability matrix instead of column 1 --
    which surfaces as a silent ``nan`` CV score, not as an error.
    """

    def __init__(self, n_continuous: int = 3, var_smoothing: float = 1e-9, alpha: float = 1.0):
        self.n_continuous = n_continuous
        self.var_smoothing = var_smoothing
        self.alpha = alpha

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        k = self.n_continuous
        self.classes_ = np.unique(y)
        self.gnb_ = GaussianNB(var_smoothing=self.var_smoothing).fit(X[:, :k], y)
        self.bnb_ = BernoulliNB(alpha=self.alpha).fit(X[:, k:], y)
        counts = np.array([(y == c).sum() for c in self.classes_], dtype=float)
        self.log_prior_ = np.log(counts / counts.sum())
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        k = self.n_continuous
        jll = (
            self.gnb_.predict_log_proba(X[:, :k])
            + self.bnb_.predict_log_proba(X[:, k:])
            - self.log_prior_[None, :]
        )
        return softmax(jll)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# --------------------------------------------------------------------------- #
# Preprocessor helpers -- all UNFITTED, all fitted inside a Pipeline
# --------------------------------------------------------------------------- #
def pre_full(df, scale=True):
    """All 21 encoded columns: 3 scaled continuous + SeniorCitizen + 17 dummies."""
    return dp.build_preprocessor(df=df, scale=scale)


def pre_continuous(df, scale=True):
    """The 3 continuous columns only (tenure, MonthlyCharges, TotalCharges)."""
    num, _, _ = dp.infer_feature_groups(df)
    return dp.build_preprocessor(numeric=num, binary=[], categorical=[], scale=scale)


def pre_binary(df):
    """The 18 binary columns only -- SeniorCitizen + the one-hot dummies."""
    _, bin_, cat = dp.infer_feature_groups(df)
    return dp.build_preprocessor(numeric=[], binary=bin_, categorical=cat, scale=False)


def discretiser(n_continuous=3, n_bins=5):
    """KBinsDiscretizer over the leading continuous columns, one-hot out,
    everything else passed through -- turns the whole matrix binary so a single
    BernoulliNB can model it."""
    return ColumnTransformer(
        transformers=[
            (
                "disc",
                KBinsDiscretizer(
                    n_bins=n_bins, encode="onehot-dense", strategy="quantile", subsample=None
                ),
                list(range(n_continuous)),
            )
        ],
        remainder="passthrough",
    )


# --------------------------------------------------------------------------- #
# CV plumbing
# --------------------------------------------------------------------------- #
VARIANTS: list[dict] = []


def cv_auc(pipe, X, y, name, note=""):
    """5-fold stratified CV ROC-AUC on TRAIN only.  Records + returns the row."""
    res = cross_validate(pipe, X, y, cv=CV, scoring="roc_auc", n_jobs=-1)
    row = {
        "name": name,
        "cv_roc_auc_mean": float(res["test_score"].mean()),
        "cv_roc_auc_std": float(res["test_score"].std()),
        "note": note,
    }
    VARIANTS.append(row)
    return row


def tune_threshold(pipe, X, y):
    """F1-maximising threshold from out-of-fold TRAIN predictions.  The test set
    plays no part in this."""
    oof = cross_val_predict(pipe, X, y, cv=CV, method="predict_proba", n_jobs=-1)[:, 1]
    grid = np.linspace(0.02, 0.98, 193)
    f1s = [f1_score(y, (oof >= t).astype(int), zero_division=0) for t in grid]
    best = int(np.argmax(f1s))
    return float(grid[best]), float(f1s[best]), oof


def test_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "f1_yes": float(f1_score(y_true, pred, zero_division=0)),
        "precision_yes": float(precision_score(y_true, pred, zero_division=0)),
        "recall_yes": float(recall_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def hr(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# --------------------------------------------------------------------------- #
# 1. Geometry of the encoded matrix -- why distance is in trouble here
# --------------------------------------------------------------------------- #
def geometry_report(df, X_tr, y_tr):
    pre = pre_full(df)
    Z = pre.fit_transform(X_tr)  # fitted on TRAIN only; used for diagnostics
    names = list(pre.get_feature_names_out())
    num, bin_, _ = dp.infer_feature_groups(df)
    n_cont = len(num)
    cont, binblock = Z[:, :n_cont], Z[:, n_cont:]

    hr("1. GEOMETRY OF THE ENCODED MATRIX (train only)")
    print(f"encoded shape                : {Z.shape}   ({n_cont} continuous, {Z.shape[1]-n_cont} binary)")
    print(f"binary columns are 0/1 only  : {bool(np.isin(binblock, [0.0, 1.0]).all())}")

    # Expected squared distance decomposes additively per dimension.
    # E[(x_i - x_j)^2] = 2*Var for a standardised column; = 2p(1-p) for a 0/1 column.
    var_cont = cont.var(axis=0)
    p = binblock.mean(axis=0)
    contrib_cont = float((2 * var_cont).sum())
    contrib_bin = float((2 * p * (1 - p)).sum())
    total = contrib_cont + contrib_bin
    print(f"\nexpected squared Euclidean distance, decomposed by block:")
    print(f"  3 continuous (standardised) : {contrib_cont:7.3f}  ({100*contrib_cont/total:5.1f}%)")
    print(f"  18 binary dummies           : {contrib_bin:7.3f}  ({100*contrib_bin/total:5.1f}%)")
    print("  -> a neighbour is chosen mostly by a Hamming count over dummies.")

    # Distance concentration.  Two measures, because the classic one is fragile:
    #   * relative contrast (d_max - d_min)/d_min -- the textbook statistic, but its
    #     denominator is the distance to the *nearest* point, so a single near-duplicate
    #     row blows it up.  Reported for completeness, not leaned on.
    #   * coefficient of variation std(d)/mean(d) -- robust, and the quantity that
    #     actually decides whether "nearest" is meaningfully nearer than "typical".
    #     It falls towards 0 as dimensionality grows; that IS the curse.
    #   * d(k=100)/median(d) -- how far out the 100th neighbour sits relative to a
    #     random point.  1.0 would mean the "neighbourhood" is the whole dataset.
    rng = np.random.default_rng(RANDOM_STATE)
    q = rng.choice(Z.shape[0], size=300, replace=False)

    def concentration(M):
        rc, cv, ratio = [], [], []
        for i in q:
            d = np.linalg.norm(M - M[i], axis=1)
            d = np.sort(d[d > 1e-12])
            rc.append((d[-1] - d[0]) / d[0])
            cv.append(d.std() / d.mean())
            ratio.append(d[99] / np.median(d))
        return float(np.mean(rc)), float(np.mean(cv)), float(np.mean(ratio))

    spaces = {"full 21-d": Z, "3-d continuous": cont, "18-d binary": binblock}
    conc = {name: concentration(M) for name, M in spaces.items()}
    print("\ndistance concentration, mean over 300 query points:")
    print(f"  {'space':<16} {'rel.contrast':>12} {'std/mean':>10} {'d(100th)/median':>17}")
    for name, (rc, cv, ratio) in conc.items():
        print(f"  {name:<16} {rc:>12.1f} {cv:>10.4f} {ratio:>17.4f}")
    print("  std/mean falling and d(100th)/median rising towards 1.0 = distances")
    print("  concentrate, and 'nearest' stops meaning 'similar'.")

    # Ties.  A near-Hamming metric produces exact duplicate distances, which makes
    # the choice of neighbour arbitrary and breaks distance weighting.
    from sklearn.neighbors import NearestNeighbors

    print("\namong the 25 nearest neighbours of a point (300 sampled):")
    ties = {}
    for name, M in spaces.items():
        nn = NearestNeighbors(n_neighbors=26).fit(M)
        dist, _ = nn.kneighbors(M[q])
        frac = float(np.mean([len(np.unique(np.round(r[1:], 9))) < 25 for r in dist]))
        dup = float(np.mean([25 - len(np.unique(np.round(r[1:], 9))) for r in dist]))
        ties[name] = {"tie_fraction": frac, "mean_ties": dup}
        print(f"  {name:<16} points with >=1 exact tie: {100*frac:5.1f}%   mean tied: {dup:5.2f}/25")

    rc_full, rc_cont = conc["full 21-d"][0], conc["3-d continuous"][0]
    tie_frac = ties["full 21-d"]["tie_fraction"]
    dup = ties["full 21-d"]["mean_ties"]

    return {
        "shape": list(Z.shape),
        "sqdist_share_continuous": contrib_cont / total,
        "sqdist_share_binary": contrib_bin / total,
        "relative_contrast_full": rc_full,
        "relative_contrast_continuous": rc_cont,
        "knn25_tie_fraction": tie_frac,
        "knn25_mean_ties": dup,
        "concentration": {k: {"relative_contrast": v[0], "cv_std_over_mean": v[1],
                              "d100_over_median": v[2]} for k, v in conc.items()},
        "ties_by_space": ties,
        "feature_names": names,
    }


# --------------------------------------------------------------------------- #
# 2. KNN
# --------------------------------------------------------------------------- #
K_GRID = [1, 3, 5, 9, 15, 21, 31, 45, 61, 81, 101, 151, 201, 301]


def knn_experiments(df, X_tr, y_tr):
    hr("2. KNN -- full 21-column matrix vs continuous-only, identical CV")
    results = {}

    for space, prefix in (("full21", "knn_full21"), ("cont3", "knn_cont3"), ("bin18", "knn_bin18")):
        if space == "full21":
            mk = lambda: pre_full(df)
        elif space == "cont3":
            mk = lambda: pre_continuous(df)
        else:
            mk = lambda: pre_binary(df)

        grid = []
        metrics = ("euclidean", "manhattan")
        weights = ("uniform", "distance")
        for metric in metrics:
            for w in weights:
                for k in K_GRID:
                    pipe = Pipeline(
                        [
                            ("pre", mk()),
                            ("clf", KNeighborsClassifier(n_neighbors=k, weights=w, metric=metric)),
                        ]
                    )
                    row = cv_auc(
                        pipe,
                        X_tr,
                        y_tr,
                        f"{prefix}_k{k}_{w}_{metric}",
                        f"KNN on {space} feature space",
                    )
                    grid.append({"k": k, "weights": w, "metric": metric, **row})
        results[space] = grid

        best = max(grid, key=lambda r: r["cv_roc_auc_mean"])
        print(
            f"\n[{space}] best: k={best['k']} weights={best['weights']} metric={best['metric']}"
            f"  CV ROC-AUC {best['cv_roc_auc_mean']:.4f} +/- {best['cv_roc_auc_std']:.4f}"
        )

    # k-curve table, printed for the two headline spaces
    print("\nCV ROC-AUC vs n_neighbors (euclidean):")
    print(f"{'k':>5} | {'full21 unif':>11} {'full21 dist':>11} | {'cont3 unif':>10} {'cont3 dist':>10}")
    print("-" * 62)
    def look(space, k, w, m="euclidean"):
        for r in results[space]:
            if r["k"] == k and r["weights"] == w and r["metric"] == m:
                return r["cv_roc_auc_mean"]
        return float("nan")
    for k in K_GRID:
        print(
            f"{k:>5} | {look('full21',k,'uniform'):>11.4f} {look('full21',k,'distance'):>11.4f} |"
            f" {look('cont3',k,'uniform'):>10.4f} {look('cont3',k,'distance'):>10.4f}"
        )

    print("\nmanhattan vs euclidean, at each space's best k (uniform weights):")
    for space in ("full21", "cont3", "bin18"):
        bk = max((r for r in results[space] if r["metric"] == "euclidean" and r["weights"] == "uniform"),
                 key=lambda r: r["cv_roc_auc_mean"])["k"]
        e = look(space, bk, "uniform", "euclidean")
        m = look(space, bk, "uniform", "manhattan")
        print(f"  {space:>7} k={bk:<4} euclidean {e:.4f}   manhattan {m:.4f}   delta {m-e:+.4f}")

    best_overall = max(
        (r for sp in results for r in results[sp]), key=lambda r: r["cv_roc_auc_mean"]
    )
    return results, best_overall


def knn_pipeline(df, best):
    space = best["name"].split("_")[1]
    mk = {"full21": pre_full, "cont3": pre_continuous, "bin18": lambda d: pre_binary(d)}[space]
    return Pipeline(
        [
            ("pre", mk(df)),
            (
                "clf",
                KNeighborsClassifier(
                    n_neighbors=best["k"], weights=best["weights"], metric=best["metric"]
                ),
            ),
        ]
    )


# --------------------------------------------------------------------------- #
# 3. Naive Bayes
# --------------------------------------------------------------------------- #
def nb_pipelines(df, n_cont):
    """name -> unfitted Pipeline.  Four families of NB over the same rows."""
    out = {
        "nb_gaussian_cont3": Pipeline(
            [("pre", pre_continuous(df)), ("clf", GaussianNB())]
        ),
        "nb_bernoulli_bin18": Pipeline(
            [("pre", pre_binary(df)), ("clf", BernoulliNB(alpha=1.0))]
        ),
        "nb_mixed_full21": Pipeline(
            [("pre", pre_full(df)), ("clf", MixedNB(n_continuous=n_cont))]
        ),
    }
    for nb in (3, 5, 10, 20):
        out[f"nb_discretised_bins{nb}"] = Pipeline(
            [
                ("pre", pre_full(df, scale=False)),
                ("disc", discretiser(n_continuous=n_cont, n_bins=nb)),
                ("clf", BernoulliNB(alpha=1.0)),
            ]
        )
    return out


def nb_experiments(df, X_tr, y_tr, df_notc, X_tr_notc):
    hr("3. NAIVE BAYES -- block models, mixed model, discretised model")
    n_cont = len(dp.infer_feature_groups(df)[0])

    rows = []
    for name, pipe in nb_pipelines(df, n_cont).items():
        r = cv_auc(pipe, X_tr, y_tr, name, "Naive Bayes variant, all 3 continuous features kept")
        rows.append(r)
        print(f"  {name:<28} CV ROC-AUC {r['cv_roc_auc_mean']:.4f} +/- {r['cv_roc_auc_std']:.4f}")

    hr("3b. INDEPENDENCE-VIOLATION EXPERIMENT (drop TotalCharges)")
    print("TotalCharges is 99.91% explained by tenure x MonthlyCharges (EDA OLS R^2=0.9991)")
    print("and correlates 0.826 with tenure.  NB multiplies per-feature likelihoods as if")
    print("independent, so this feature's evidence is counted a third time.  Refit without it,")
    print("on the SAME frozen train rows (re-indexed by row_id, never re-split):\n")

    n_cont_notc = len(dp.infer_feature_groups(df_notc)[0])
    drop_rows = []
    for name, pipe in nb_pipelines(df_notc, n_cont_notc).items():
        r = cv_auc(
            pipe,
            X_tr_notc,
            y_tr,
            name + "_noTotalCharges",
            "same variant with TotalCharges dropped (independence-violation test)",
        )
        drop_rows.append(r)

    print(f"{'variant':<28} {'with TC':>9} {'without TC':>11} {'delta':>9}")
    print("-" * 60)
    deltas = {}
    for a, b in zip(rows, drop_rows):
        d = b["cv_roc_auc_mean"] - a["cv_roc_auc_mean"]
        deltas[a["name"]] = d
        print(f"{a['name']:<28} {a['cv_roc_auc_mean']:>9.4f} {b['cv_roc_auc_mean']:>11.4f} {d:>+9.4f}")

    best = max(rows + drop_rows, key=lambda r: r["cv_roc_auc_mean"])
    print(f"\nbest NB variant: {best['name']}  CV ROC-AUC {best['cv_roc_auc_mean']:.4f}"
          f" +/- {best['cv_roc_auc_std']:.4f}")
    return rows, drop_rows, deltas, best


def nb_pipeline_by_name(name, df, df_notc):
    base = name.replace("_noTotalCharges", "")
    d = df_notc if name.endswith("_noTotalCharges") else df
    n_cont = len(dp.infer_feature_groups(d)[0])
    return nb_pipelines(d, n_cont)[base]


# --------------------------------------------------------------------------- #
# 3c. Independence diagnostics -- the assumption NB makes, measured directly
# --------------------------------------------------------------------------- #
def independence_diagnostics(df, X_tr, y_tr):
    """How badly is conditional independence violated, and what does it cost?

    Two measurements:

    a) **The assumption itself.**  NB assumes ``P(x_i, x_j | y) = P(x_i|y)P(x_j|y)``,
       i.e. every within-class feature correlation is 0.  We report the actual
       within-class Pearson correlations of the continuous block and the largest
       within-class correlations of the binary block.

    b) **The price.**  ``GaussianNB`` and ``QuadraticDiscriminantAnalysis`` fit the
       *same* generative model -- a Gaussian per class -- and differ in exactly one
       respect: NB forces the covariance matrix to be diagonal, QDA estimates it in
       full.  ``LinearDiscriminantAnalysis`` shares one full covariance across
       classes.  The CV ROC-AUC gap between them on identical columns is therefore a
       direct, isolated price tag on the naive assumption, with nothing else varying.
       These three are *diagnostics*, not candidate models, so they are reported
       separately from ``variants``.
    """
    from sklearn.discriminant_analysis import (
        LinearDiscriminantAnalysis,
        QuadraticDiscriminantAnalysis,
    )

    hr("3c. INDEPENDENCE DIAGNOSTICS")
    num, bin_, cat = dp.infer_feature_groups(df)
    pre = pre_full(df)
    Z = pd.DataFrame(pre.fit_transform(X_tr), columns=pre.get_feature_names_out())
    y = y_tr.to_numpy()

    print("a) within-class correlations of the 3 continuous features")
    print("   (Naive Bayes assumes every one of these is exactly 0.00)\n")
    cont_corr = {}
    for label, mask in (("Churn=0", y == 0), ("Churn=1", y == 1)):
        C = Z.loc[mask, num].corr()
        cont_corr[label] = C.round(4).to_dict()
        print(f"   {label}  (n={int(mask.sum())})")
        print("   " + C.round(3).to_string().replace("\n", "\n   "))
    print()

    off = []
    B = Z[bin_ + [c for c in Z.columns if c not in num and c not in bin_]]
    for label, mask in (("Churn=0", y == 0), ("Churn=1", y == 1)):
        Cb = B.loc[mask].corr().to_numpy()
        iu = np.triu_indices_from(Cb, k=1)
        vals = np.abs(Cb[iu])
        vals = vals[~np.isnan(vals)]
        off.append((label, float(vals.mean()), float(vals.max())))
    print("b) within-class correlations among the 18 binary columns")
    print("   (BernoulliNB assumes these are 0 too)")
    for label, mean_r, max_r in off:
        print(f"   {label}: mean |r| = {mean_r:.3f}, max |r| = {max_r:.3f} over 153 pairs")

    Cb = B.loc[y == 0].corr().abs()
    arr = Cb.to_numpy(copy=True)          # pandas 3.0 hands back a read-only view
    np.fill_diagonal(arr, 0.0)
    Cb = pd.DataFrame(arr, index=Cb.index, columns=Cb.columns)
    top = Cb.stack().sort_values(ascending=False).head(10)[::2]  # each pair appears twice
    print("   top 5 dependent pairs (Churn=0):")
    for (a, b), r in top.items():
        print(f"     {a:<28} ~ {b:<28} |r| = {r:.3f}")

    print("\nc) the price of the diagonal-covariance assumption, 3 continuous features")
    print("   identical columns, identical CV, identical Gaussian generative model:\n")
    diag = {}
    for nm, est in (
        ("GaussianNB (diagonal covariance)", GaussianNB()),
        ("LDA (one shared full covariance)", LinearDiscriminantAnalysis()),
        ("QDA (per-class full covariance)", QuadraticDiscriminantAnalysis(reg_param=1e-3)),
    ):
        pipe = Pipeline([("pre", pre_continuous(df)), ("clf", est)])
        res = cross_validate(pipe, X_tr, y_tr, cv=CV, scoring="roc_auc", n_jobs=-1)
        diag[nm] = {
            "cv_roc_auc_mean": float(res["test_score"].mean()),
            "cv_roc_auc_std": float(res["test_score"].std()),
        }
        print(f"   {nm:<36} {diag[nm]['cv_roc_auc_mean']:.4f} +/- {diag[nm]['cv_roc_auc_std']:.4f}")
    gap = diag["QDA (per-class full covariance)"]["cv_roc_auc_mean"] - diag["GaussianNB (diagonal covariance)"]["cv_roc_auc_mean"]
    print(f"\n   QDA - GaussianNB = {gap:+.4f} ROC-AUC: the isolated cost of assuming")
    print("   the 3 continuous features are conditionally independent.")

    return {
        "continuous_within_class_corr": cont_corr,
        "binary_within_class_offdiag": [
            {"class": l, "mean_abs_r": m, "max_abs_r": x} for l, m, x in off
        ],
        "top_dependent_binary_pairs": [
            {"a": a, "b": b, "abs_r": float(r)} for (a, b), r in top.items()
        ],
        "diagonal_covariance_cost": diag,
        "qda_minus_gaussiannb": float(gap),
    }


# --------------------------------------------------------------------------- #
# 4. Calibration
# --------------------------------------------------------------------------- #
def calibration_report(named_pipes, X_map, y_tr):
    hr("4. CALIBRATION OF OUT-OF-FOLD TRAIN PROBABILITIES")
    print("NB is notorious for overconfident probabilities; the team is tuning a decision")
    print("threshold on these numbers, so a miscalibrated score moves the threshold, not just")
    print("the ranking.  Brier score: lower is better; 0.1950 is the no-skill constant")
    print("predictor at the 26.54% base rate.\n")
    base = float(np.mean(y_tr) * (1 - np.mean(y_tr)))
    print(f"{'model':<28} {'Brier':>8} {'ECE':>8} {'p<0.01':>8} {'p>0.99':>8} {'min p':>7} {'max p':>7}")
    print("-" * 82)
    out = {}
    for name, pipe in named_pipes.items():
        X = X_map[name]
        oof = cross_val_predict(pipe, X, y_tr, cv=CV, method="predict_proba", n_jobs=-1)[:, 1]
        brier = float(brier_score_loss(y_tr, oof))
        frac_pos, mean_pred = calibration_curve(y_tr, oof, n_bins=10, strategy="quantile")
        ece = float(np.mean(np.abs(frac_pos - mean_pred)))
        out[name] = {
            "brier": brier,
            "ece": ece,
            "frac_below_0.01": float(np.mean(oof < 0.01)),
            "frac_above_0.99": float(np.mean(oof > 0.99)),
            "min": float(oof.min()),
            "max": float(oof.max()),
            "reliability": {
                "mean_predicted": [float(v) for v in mean_pred],
                "observed_frac_pos": [float(v) for v in frac_pos],
            },
            "oof": oof,
        }
        o = out[name]
        print(
            f"{name:<28} {brier:>8.4f} {ece:>8.4f} {o['frac_below_0.01']:>8.3f}"
            f" {o['frac_above_0.99']:>8.3f} {o['min']:>7.4f} {o['max']:>7.4f}"
        )
    print(f"{'(no-skill constant 0.2654)':<28} {base:>8.4f}")

    print("\nreliability curve (10 quantile bins) -- predicted vs observed churn rate:")
    for name, o in out.items():
        print(f"  {name}")
        print("    predicted:", " ".join(f"{v:5.3f}" for v in o["reliability"]["mean_predicted"]))
        print("    observed :", " ".join(f"{v:5.3f}" for v in o["reliability"]["observed_frac_pos"]))
    return out


# --------------------------------------------------------------------------- #
# 5. Final evaluation -- the ONLY place the test set is read
# --------------------------------------------------------------------------- #
def final_evaluation(named_pipes, Xtr_map, Xte_map, y_tr, y_te, thresholds):
    hr("5. FINAL TEST EVALUATION (test set touched exactly once, here)")
    out = {}
    for name, pipe in named_pipes.items():
        fitted = clone(pipe).fit(Xtr_map[name], y_tr)
        proba = fitted.predict_proba(Xte_map[name])[:, 1]
        t = thresholds[name]
        out[name] = {
            "threshold": t,
            "test": test_metrics(y_te, proba, t),
            "test_at_threshold_0.5": {
                k: v for k, v in test_metrics(y_te, proba, 0.5).items()
                if k in ("f1_yes", "precision_yes", "recall_yes", "accuracy")
            },
            "test_brier": float(brier_score_loss(y_te, proba)),
        }
        m, m05 = out[name]["test"], out[name]["test_at_threshold_0.5"]
        print(f"\n{name}   (tuned threshold {t:.3f}, from OOF TRAIN predictions)")
        print(f"  ROC-AUC        {m['roc_auc']:.4f}")
        print(f"  @0.500  F1 {m05['f1_yes']:.4f}  P {m05['precision_yes']:.4f}"
              f"  R {m05['recall_yes']:.4f}  acc {m05['accuracy']:.4f}")
        print(f"  @{t:.3f}  F1 {m['f1_yes']:.4f}  P {m['precision_yes']:.4f}"
              f"  R {m['recall_yes']:.4f}  acc {m['accuracy']:.4f}")
        print(f"  confusion [[TN,FP],[FN,TP]] = {m['confusion_matrix']}   Brier {out[name]['test_brier']:.4f}")
    return out


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()

    X_tr, X_te, y_tr, y_te = dp.load_splits()
    raw = dp.load_raw()
    df = dp.clean(raw)
    df_notc = dp.clean(raw, drop_total_charges=True)

    # Re-index the TotalCharges-free frame onto the frozen split's row_ids.
    # This is NOT a re-split: same rows, same order, one fewer column.
    X_notc = df_notc.drop(columns=[dp.TARGET])
    X_tr_notc, X_te_notc = X_notc.loc[X_tr.index], X_notc.loc[X_te.index]
    assert (df_notc[dp.TARGET].loc[X_tr.index].to_numpy() == y_tr.to_numpy()).all()
    assert list(X_tr_notc.index) == list(X_tr.index)

    hr("0. SETUP")
    print(f"frozen split: train {X_tr.shape} / test {X_te.shape}")
    print(f"train churn rate {y_tr.mean():.4%} ({int(y_tr.sum())} positives) | "
          f"test churn rate {y_te.mean():.4%} ({int(y_te.sum())} positives)")
    print(f"all-'No' baseline accuracy on test: {1 - y_te.mean():.4%}")
    print("CV: StratifiedKFold(5, shuffle=True, random_state=42) on TRAIN only")

    geom = geometry_report(df, X_tr, y_tr)
    knn_results, knn_best = knn_experiments(df, X_tr, y_tr)
    nb_rows, nb_drop_rows, nb_deltas, nb_best = nb_experiments(df, X_tr, y_tr, df_notc, X_tr_notc)
    indep = independence_diagnostics(df, X_tr, y_tr)

    # ---- assemble the two finalists ---------------------------------------
    knn_pipe = knn_pipeline(df, knn_best)
    nb_pipe = nb_pipeline_by_name(nb_best["name"], df, df_notc)
    finalists = {knn_best["name"]: knn_pipe, nb_best["name"]: nb_pipe}
    Xtr_map = {
        knn_best["name"]: X_tr,
        nb_best["name"]: X_tr_notc if nb_best["name"].endswith("_noTotalCharges") else X_tr,
    }
    Xte_map = {
        knn_best["name"]: X_te,
        nb_best["name"]: X_te_notc if nb_best["name"].endswith("_noTotalCharges") else X_te,
    }

    calib = calibration_report(finalists, Xtr_map, y_tr)

    hr("4b. F1-MAXIMISING THRESHOLD FROM OUT-OF-FOLD TRAIN PREDICTIONS")
    thresholds, oof_f1 = {}, {}
    for name in finalists:
        oof = calib[name]["oof"]
        grid = np.linspace(0.02, 0.98, 193)
        f1s = [f1_score(y_tr, (oof >= t).astype(int), zero_division=0) for t in grid]
        i = int(np.argmax(f1s))
        thresholds[name] = float(grid[i])
        oof_f1[name] = float(f1s[i])
        f1_05 = f1_score(y_tr, (oof >= 0.5).astype(int), zero_division=0)
        print(f"  {name:<28} tuned t={grid[i]:.3f}  OOF F1 {f1s[i]:.4f}  (OOF F1 at 0.5: {f1_05:.4f})")

    test_out = final_evaluation(finalists, Xtr_map, Xte_map, y_tr, y_te, thresholds)

    # ---- best of the two ---------------------------------------------------
    best_name = max(finalists, key=lambda n: next(
        r["cv_roc_auc_mean"] for r in VARIANTS if r["name"] == n))
    best_row = next(r for r in VARIANTS if r["name"] == best_name)
    if best_name.startswith("knn"):
        params = {
            "space": knn_best["name"].split("_")[1],
            "n_neighbors": knn_best["k"],
            "weights": knn_best["weights"],
            "metric": knn_best["metric"],
            "scaled": True,
        }
    else:
        params = {"variant": best_name, "alpha_or_var_smoothing": "sklearn defaults",
                  "drop_total_charges": best_name.endswith("_noTotalCharges")}

    runtime = time.time() - t0
    payload = {
        "approach": "knn-naive-bayes",
        "agent_brief": "distance-based and probabilistic",
        "variants": [{k: v for k, v in r.items()} for r in VARIANTS],
        "best_model": {
            "name": best_name,
            "params": params,
            "cv_roc_auc_mean": best_row["cv_roc_auc_mean"],
            "cv_roc_auc_std": best_row["cv_roc_auc_std"],
            "threshold": thresholds[best_name],
            "test": test_out[best_name]["test"],
            "test_at_threshold_0.5": test_out[best_name]["test_at_threshold_0.5"],
        },
        "top_drivers": [],
        "runtime_seconds": round(runtime, 2),
        "notes": (
            "top_drivers is deliberately empty: neither KNN nor Naive Bayes yields a natural "
            "feature-importance ranking (KNN has no coefficients or splits; NB has per-feature "
            "log-likelihood ratios that are not comparable across a mixed Gaussian/Bernoulli "
            "model and are inflated by the very dependence violations documented here). "
            "Inventing a ranking would be worse than reporting none. "
            f"Best KNN: {knn_best['name']} at CV ROC-AUC {knn_best['cv_roc_auc_mean']:.4f}; "
            f"best NB: {nb_best['name']} at CV ROC-AUC {nb_best['cv_roc_auc_mean']:.4f}. "
            "Both sit below the 0.84-0.85 team ceiling, as expected for these families on this "
            "dataset -- see docs/step-03c-knn-naive-bayes.md for the geometry and "
            "independence-violation evidence."
        ),
        "diagnostics": {
            "geometry": {k: v for k, v in geom.items() if k != "feature_names"},
            "knn_best_per_space": {
                sp: max(rows, key=lambda r: r["cv_roc_auc_mean"])
                for sp, rows in knn_results.items()
            },
            "nb_total_charges_delta": nb_deltas,
            "independence": indep,
            "calibration": {
                n: {k: v for k, v in o.items() if k != "oof"} for n, o in calib.items()
            },
            "finalists_test": test_out,
            "oof_f1_at_tuned_threshold": oof_f1,
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    hr("6. SUMMARY")
    print(f"best model overall (of my two families): {best_name}")
    print(f"  CV ROC-AUC {best_row['cv_roc_auc_mean']:.4f} +/- {best_row['cv_roc_auc_std']:.4f}")
    print(f"  test ROC-AUC {payload['best_model']['test']['roc_auc']:.4f}, "
          f"F1(Yes) {payload['best_model']['test']['f1_yes']:.4f} at t={thresholds[best_name]:.3f}")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}  ({len(VARIANTS)} variants, {runtime:.1f}s)")


if __name__ == "__main__":
    main()
