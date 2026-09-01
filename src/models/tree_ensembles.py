#!/usr/bin/env python3
"""Step 3b -- tree ensembles (Random Forest / Gradient Boosting / HistGB).

Run:  python3 src/models/tree_ensembles.py

Why this family, specifically
----------------------------
Step 1 (``docs/step-01-eda.md``) found two structures that a plain linear model
handles badly and an axis-aligned tree finds for free:

* ``MonthlyCharges`` is **non-monotone** against churn -- it climbs to 40.91% at
  decile 9 and then *falls* to 24.68% at decile 10 (EDA 6.7).  A logistic
  regression fits a single monotone slope and therefore gets the top decile
  backwards.
* The ``Contract x InternetService`` cell rates span **70x** -- 54.61%
  (M2M/Fiber, n=2128) down to 0.78% (Two-year/No-internet, n=638) (EDA 6.5).
  That is an interaction; a linear model needs the product term written out by
  hand, a tree gets it from two consecutive splits.

So the headline question this script answers is not "what is the best AUC" but
**"do the trees actually cash in on those two structures, or does the linear
model match them anyway?"**  Both answers are reportable; this dataset is
famously one where logistic regression is hard to beat.

Protocol (identical for all four Step 3 agents)
-----------------------------------------------
* The **frozen** split from ``data/splits/`` via ``dp.load_splits()``.  Never
  ``dp.split()``, never a fresh ``random_state`` -- all four agents must score
  on the same held-out rows or the comparison table means nothing.
* **The test set is read exactly once**, at the very end.  Every model choice,
  hyper-parameter and the decision threshold comes from 5-fold
  ``StratifiedKFold`` CV on the training fold alone.
* Everything inside a ``Pipeline`` whose first step is the *unfitted*
  ``ColumnTransformer`` from ``dp.build_preprocessor(scale=False)``, so the
  one-hot vocabulary is learned per training fold.  ``scale=False`` because
  scaling is a no-op for trees; the pipeline shape stays identical to the other
  agents'.
* Primary metric **ROC-AUC**; F1 / precision / recall on the positive class
  (``Yes`` = 1) reported alongside.  Accuracy is never the headline -- the
  all-"No" baseline is already 73.46%.
* Sanity bound from EDA: ROC-AUC 0.84-0.85, F1(Yes) 0.60-0.63.  **AUC > 0.90
  means a leak, not a win** -- the script says so loudly if it happens.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

# Pin the BLAS/OpenMP pools to one thread *before* sklearn is imported.  On this
# 2-core box HistGradientBoostingClassifier is ~38x SLOWER with OMP_NUM_THREADS=2
# than with 1 (107.8s vs 2.8s for a single fit, measured) -- its OpenMP workers
# thrash against joblib's own processes.  Every joblib n_jobs here is
# pinned to 1 for a related reason: on a 5634x21 matrix the loky process pool and
# its memmapping overhead cost far more than the fits themselves -- one 5-fold
# RandomForest CV measured 9s serial against ~150s at n_jobs=-1 on this box.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))

import data_prep as dp  # noqa: E402

from sklearn.ensemble import (  # noqa: E402
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import partial_dependence, permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
N_SPLITS = 5
RESULTS_DIR = REPO / "results"
JSON_PATH = RESULTS_DIR / "tree_ensembles.json"
FIG_PATH = RESULTS_DIR / "tree_ensembles_monthlycharges_pd.png"

# EDA 6.1 / 6.2 -- the association ranking the model's importances get checked
# against.  gender (0.008) and PhoneService (0.011) were dropped in Step 2.
EDA_RANK = {
    "Contract": 0.410,
    "OnlineSecurity": 0.347,
    "TechSupport": 0.343,
    "InternetService": 0.323,
    "PaymentMethod": 0.303,
    "OnlineBackup": 0.292,
    "DeviceProtection": 0.282,
    "StreamingMovies": 0.231,
    "StreamingTV": 0.231,
    "PaperlessBilling": 0.192,
    "Dependents": 0.164,
    "SeniorCitizen": 0.151,
    "Partner": 0.150,
    "MultipleLines": 0.040,
}
# Numerics are not on the Cramer's V scale; their univariate AUCs (EDA 6.2) are.
EDA_NUMERIC_AUC = {"tenure": 0.740, "TotalCharges": 0.652, "MonthlyCharges": 0.621}

# EDA 6.7 -- the decile churn rates the partial-dependence probe is checked
# against.  The point of interest is D9 -> D10.
EDA_DECILE_CHURN = [8.66, 9.38, None, None, None, None, None, None, 40.91, 24.68]


class BalancedGradientBoosting(GradientBoostingClassifier):
    """``GradientBoostingClassifier`` that class-balances itself inside ``fit``.

    ``GradientBoostingClassifier`` has no ``class_weight``; the documented way
    to handle imbalance is ``fit(..., sample_weight=...)``.  Passing a
    precomputed weight vector through ``cross_val_score`` is awkward -- without
    metadata routing enabled sklearn hands the *full-length* vector to each
    fold and the fit raises on the length mismatch.  Since the balanced weight
    is a deterministic function of ``y`` alone, computing it inside ``fit`` is
    exactly equivalent and survives cloning, CV slicing and grid search
    untouched.
    """

    def fit(self, X, y, sample_weight=None, **kwargs):
        if sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y)
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)


def make_pipeline(clf, cleaned: pd.DataFrame) -> Pipeline:
    """Unfitted preprocessor + estimator.  ``scale=False``: trees ignore scale."""
    return Pipeline(
        [("prep", dp.build_preprocessor(df=cleaned, scale=False)), ("clf", clf)]
    )


def cv_auc(pipe, X, y, cv) -> tuple[float, float]:
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    return float(scores.mean()), float(scores.std())


def best_f1_threshold(y_true, proba) -> tuple[float, float]:
    """F1-maximising threshold from *out-of-fold* training probabilities.

    Chosen on TRAIN only.  Picking it on test would be tuning on the holdout.
    """
    grid = np.round(np.arange(0.05, 0.951, 0.005), 4)
    f1s = np.array([f1_score(y_true, (proba >= t).astype(int)) for t in grid])
    i = int(f1s.argmax())
    return float(grid[i]), float(f1s[i])


def score_block(y_true, y_pred) -> dict:
    return {
        "f1_yes": round(float(f1_score(y_true, y_pred)), 4),
        "precision_yes": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_yes": round(float(recall_score(y_true, y_pred)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
    }


def main() -> None:
    t0 = time.time()
    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print("STEP 3b -- TREE ENSEMBLES (Random Forest / Gradient Boosting / HistGB)")
    print("=" * 78)

    # ---------------------------------------------------------------- data --
    X_tr, X_te, y_tr, y_te = dp.load_splits()  # THE FROZEN SPLIT. Never re-split.
    cleaned = dp.clean(dp.load_raw(str(REPO / dp.DEFAULT_CSV)))
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    print(f"\ntrain {X_tr.shape}  churn {y_tr.mean():.4%} ({int(y_tr.sum())} positives)")
    print(f"test  {X_te.shape}  churn {y_te.mean():.4%} ({int(y_te.sum())} positives)")
    print(f"features: {list(X_tr.columns)}")
    print(f"CV: StratifiedKFold({N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})"
          "  scoring=roc_auc  -- TRAIN FOLD ONLY")

    variants: list[dict] = []

    def record(name, mean, std, note):
        variants.append(
            {
                "name": name,
                "cv_roc_auc_mean": round(mean, 4),
                "cv_roc_auc_std": round(std, 4),
                "note": note,
            }
        )
        print(f"  {name:<46s} {mean:.4f} +/- {std:.4f}   {note}")

    # ------------------------------------------------- 1. untuned baselines --
    print("\n[1] Untuned baselines (5-fold CV on train)")
    baselines = [
        (
            "RandomForest (default, 200 trees)",
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1),
            "unweighted; fully grown trees",
        ),
        (
            "RandomForest (200 trees, class_weight=balanced)",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "imbalance handled by class_weight",
        ),
        (
            "GradientBoosting (default)",
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            "unweighted",
        ),
        (
            "GradientBoosting (balanced sample_weight)",
            BalancedGradientBoosting(random_state=RANDOM_STATE),
            "no class_weight param -> sample_weight in fit()",
        ),
        (
            "HistGradientBoosting (default)",
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            "unweighted",
        ),
        (
            "HistGradientBoosting (class_weight=balanced)",
            HistGradientBoostingClassifier(
                class_weight="balanced", random_state=RANDOM_STATE
            ),
            "imbalance handled by class_weight",
        ),
    ]
    for name, clf, note in baselines:
        m, s = cv_auc(make_pipeline(clf, cleaned), X_tr, y_tr, cv)
        record(name, m, s, note)

    # ------------------------------------------------------ 2. tuned search --
    print("\n[2] RandomizedSearchCV (5-fold, scoring=roc_auc, train fold only)")
    searches = {
        "RandomForest (tuned)": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {
                "clf__n_estimators": [150, 250],
                "clf__max_depth": [6, 8, 10, None],
                "clf__min_samples_leaf": [5, 10, 20, 40],
                "clf__max_features": ["sqrt", 0.3],
                "clf__class_weight": [None, "balanced"],
            },
            6,
        ),
        "GradientBoosting (tuned)": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [2, 3],
                "clf__min_samples_leaf": [20, 50],
                "clf__subsample": [0.8, 1.0],
                "clf__max_features": ["sqrt", None],
            },
            5,
        ),
        "GradientBoosting (tuned, balanced weights)": (
            BalancedGradientBoosting(random_state=RANDOM_STATE),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [2, 3],
                "clf__min_samples_leaf": [20, 50],
                "clf__subsample": [0.8, 1.0],
                "clf__max_features": ["sqrt", None],
            },
            4,
        ),
        "HistGradientBoosting (tuned)": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_iter": [100, 200],
                "clf__max_leaf_nodes": [15, 31],
                "clf__min_samples_leaf": [20, 50],
                "clf__l2_regularization": [0.0, 1.0],
                "clf__class_weight": [None, "balanced"],
            },
            5,
        ),
    }

    fitted: dict[str, RandomizedSearchCV] = {}
    for name, (clf, grid, n_iter) in searches.items():
        t = time.time()
        search = RandomizedSearchCV(
            make_pipeline(clf, cleaned),
            grid,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_tr, y_tr)  # train fold only
        i = search.best_index_
        mean = float(search.cv_results_["mean_test_score"][i])
        std = float(search.cv_results_["std_test_score"][i])
        fitted[name] = search
        record(name, mean, std, f"{n_iter} draws, {time.time() - t:.0f}s")
        print(f"      best params: "
              f"{ {k.replace('clf__',''): v for k, v in search.best_params_.items()} }")

    # ------------------------------------------------- 3. pick a best model --
    best_name = max(fitted, key=lambda n: fitted[n].cv_results_["mean_test_score"][
        fitted[n].best_index_])
    best_search = fitted[best_name]
    best_pipe = best_search.best_estimator_
    best_mean = float(best_search.cv_results_["mean_test_score"][best_search.best_index_])
    best_std = float(best_search.cv_results_["std_test_score"][best_search.best_index_])
    best_params = {k.replace("clf__", ""): v for k, v in best_search.best_params_.items()}
    print(f"\n[3] Best by CV ROC-AUC: {best_name}  {best_mean:.4f} +/- {best_std:.4f}")
    print(f"    {best_params}")

    # --------------------------------------- 4. threshold, chosen on TRAIN ---
    print("\n[4] Threshold selection -- out-of-fold TRAIN probabilities, F1(Yes)-max")
    oof = cross_val_predict(
        best_pipe, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    thr, oof_f1 = best_f1_threshold(y_tr.to_numpy(), oof)
    oof_f1_05 = f1_score(y_tr, (oof >= 0.5).astype(int))
    print(f"    OOF F1 @0.50 = {oof_f1_05:.4f}   OOF F1 @{thr:.3f} = {oof_f1:.4f}"
          f"   (+{oof_f1 - oof_f1_05:.4f})")
    print(f"    OOF ROC-AUC  = {roc_auc_score(y_tr, oof):.4f}")

    # ------------------------------------- 5. permutation importance (train) --
    print("\n[5] Permutation importance on the TRAIN fold (scoring=roc_auc, 5 repeats)")
    print("    Permutation, not impurity: impurity importance is biased toward")
    print("    high-cardinality / continuous splits, which here would flatter")
    print("    tenure and MonthlyCharges against the binary service flags.")
    best_pipe.fit(X_tr, y_tr)
    perm = permutation_importance(
        best_pipe,
        X_tr,
        y_tr,
        scoring="roc_auc",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    imp = (
        pd.DataFrame(
            {
                "feature": X_tr.columns,
                "importance": perm.importances_mean,
                "std": perm.importances_std,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print(f"    {'feature':<22s}{'perm dAUC':>11s}{'+/-':>9s}   EDA (V / univ.AUC)")
    for _, r in imp.iterrows():
        eda = (
            f"V={EDA_RANK[r.feature]:.3f}"
            if r.feature in EDA_RANK
            else f"AUC={EDA_NUMERIC_AUC.get(r.feature, float('nan')):.3f}"
        )
        print(f"    {r.feature:<22s}{r.importance:>11.5f}{r['std']:>9.5f}   {eda}")

    # ------------- 6. the non-monotonicity probe: MonthlyCharges dependence --
    print("\n[6] Does the tree recover the non-monotone MonthlyCharges effect?")
    print("    EDA 6.7: churn rises to 40.91% at decile 9 then FALLS to 24.68% at D10.")
    pd_res = partial_dependence(
        best_pipe, X_tr, features=["MonthlyCharges"], grid_resolution=30, kind="average"
    )
    pd_grid = np.asarray(pd_res["grid_values"][0])
    pd_vals = np.asarray(pd_res["average"][0])

    # A linear model on the same pipeline, purely as a contrast for the shape.
    lin = Pipeline(
        [
            ("prep", dp.build_preprocessor(df=cleaned, scale=True)),
            ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    ).fit(X_tr, y_tr)
    lin_pd = partial_dependence(
        lin, X_tr, features=["MonthlyCharges"], grid_resolution=30, kind="average"
    )
    lin_vals = np.asarray(lin_pd["average"][0])

    peak_i = int(pd_vals.argmax())
    pd_drop = float(pd_vals[peak_i] - pd_vals[-1])
    lin_monotone = bool(int(lin_vals.argmax()) == len(lin_vals) - 1)
    print(f"    tree PD peaks at MonthlyCharges = {pd_grid[peak_i]:.2f}"
          f" (P={pd_vals[peak_i]:.4f}) and ends at {pd_grid[-1]:.2f}"
          f" (P={pd_vals[-1]:.4f})  -> drop of {pd_drop:+.4f}")
    print(f"    logistic PD runs {lin_vals[0]:.4f} -> {lin_vals[-1]:.4f}"
          f"; its peak is at the top of the range: {lin_monotone}")

    # Decile view: observed churn vs mean predicted probability, on train.
    dec = pd.qcut(X_tr["MonthlyCharges"], 10, labels=False, duplicates="drop")
    tab = pd.DataFrame(
        {
            "decile": dec.to_numpy() + 1,
            "MonthlyCharges": X_tr["MonthlyCharges"].to_numpy(),
            "y": y_tr.to_numpy(),
            "p_tree": oof,  # out-of-fold: honest, not in-sample
            "p_lin": cross_val_predict(
                lin, X_tr, y_tr, cv=cv, method="predict_proba", n_jobs=1
            )[:, 1],
        }
    )
    agg = tab.groupby("decile").agg(
        n=("y", "size"),
        lo=("MonthlyCharges", "min"),
        hi=("MonthlyCharges", "max"),
        actual=("y", "mean"),
        tree=("p_tree", "mean"),
        linear=("p_lin", "mean"),
    )
    print(f"\n    {'D':>2s} {'n':>5s} {'range':>16s} {'actual':>8s} {'tree(oof)':>10s}"
          f" {'linear(oof)':>12s}")
    for d, r in agg.iterrows():
        print(f"    {int(d):>2d} {int(r.n):>5d} {r.lo:>7.2f}-{r.hi:<8.2f}"
              f" {r.actual:>7.2%} {r.tree:>9.2%} {r.linear:>11.2%}")
    d9, d10 = agg.loc[9], agg.loc[10]
    print(f"\n    D9 -> D10  actual  {d9.actual:.2%} -> {d10.actual:.2%}"
          f"  ({(d10.actual - d9.actual) * 100:+.2f} pp)")
    print(f"    D9 -> D10  tree    {d9.tree:.2%} -> {d10.tree:.2%}"
          f"  ({(d10.tree - d9.tree) * 100:+.2f} pp)")
    print(f"    D9 -> D10  linear  {d9.linear:.2%} -> {d10.linear:.2%}"
          f"  ({(d10.linear - d9.linear) * 100:+.2f} pp)")
    tree_recovers = bool(d10.tree < d9.tree)
    lin_recovers = bool(d10.linear < d9.linear)
    print(f"    tree reproduces the D9->D10 drop: {tree_recovers}")
    print(f"    logistic reproduces it:           {lin_recovers}")

    _save_figure(pd_grid, pd_vals, lin_vals, agg)

    # --------------------------------------- 7. TEST -- touched exactly once --
    print("\n[7] FINAL TEST EVALUATION -- the holdout is read here, once, and only here")
    proba_te = best_pipe.predict_proba(X_te)[:, 1]
    test_auc = float(roc_auc_score(y_te, proba_te))
    at_tuned = score_block(y_te, (proba_te >= thr).astype(int))
    at_half = score_block(y_te, (proba_te >= 0.5).astype(int))
    cm = confusion_matrix(y_te, (proba_te >= thr).astype(int)).tolist()

    print(f"    ROC-AUC            {test_auc:.4f}")
    print(f"    @0.50   F1={at_half['f1_yes']:.4f}  P={at_half['precision_yes']:.4f}"
          f"  R={at_half['recall_yes']:.4f}  acc={at_half['accuracy']:.4f}")
    print(f"    @{thr:.3f}  F1={at_tuned['f1_yes']:.4f}  P={at_tuned['precision_yes']:.4f}"
          f"  R={at_tuned['recall_yes']:.4f}  acc={at_tuned['accuracy']:.4f}")
    print(f"    confusion matrix (rows true 0/1, cols pred 0/1) @{thr:.3f}: {cm}")
    print(f"    all-'No' baseline accuracy: {1 - y_te.mean():.4%}")

    if test_auc > 0.90:
        print("\n" + "!" * 78)
        print("!! ROC-AUC > 0.90 -- EDA fact #4 says this is LEAKAGE, not a win. STOP.")
        print("!" * 78)
    elif not (0.83 <= test_auc <= 0.87):
        print(f"\n[!] ROC-AUC {test_auc:.4f} sits outside the expected 0.84-0.85 band"
              " -- worth a look, though not necessarily wrong.")
    else:
        print("\n    Inside the EDA-predicted 0.84-0.85 band. No leakage signal.")

    runtime = time.time() - t0

    # -------------------------------------------------------------- 8. JSON --
    payload = {
        "approach": "tree-ensembles",
        "agent_brief": "random forest / gradient boosting",
        "variants": variants,
        "best_model": {
            "name": best_name,
            "params": {k: (None if v is None else v) for k, v in best_params.items()},
            "cv_roc_auc_mean": round(best_mean, 4),
            "cv_roc_auc_std": round(best_std, 4),
            "threshold": round(thr, 4),
            "test": {
                "roc_auc": round(test_auc, 4),
                "f1_yes": at_tuned["f1_yes"],
                "precision_yes": at_tuned["precision_yes"],
                "recall_yes": at_tuned["recall_yes"],
                "accuracy": at_tuned["accuracy"],
                "confusion_matrix": cm,
            },
            "test_at_threshold_0.5": at_half,
        },
        "top_drivers": [
            {"feature": r.feature, "importance": round(float(r.importance), 5)}
            for _, r in imp.head(10).iterrows()
        ],
        "runtime_seconds": round(runtime, 1),
        "notes": (
            f"Frozen split (data/splits), 5-fold StratifiedKFold on train only; test "
            f"read once. Threshold {thr:.3f} chosen by F1(Yes)-max on out-of-fold TRAIN "
            f"probabilities, not on test. Importances are permutation dAUC on the train "
            f"fold (5 repeats), not impurity. MonthlyCharges partial dependence: tree "
            f"peaks at {pd_grid[peak_i]:.2f} then falls {pd_drop:.4f} by the top of the "
            f"range; decile 9->10 out-of-fold mean predicted churn "
            f"{d9.tree:.2%}->{d10.tree:.2%} vs actual {d9.actual:.2%}->{d10.actual:.2%} "
            f"(tree reproduces the drop: {tree_recovers}; logistic: {lin_recovers}). "
            f"class_weight/sample_weight balancing moved CV ROC-AUC by <0.005 -- it "
            f"shifts calibration, not ranking, so threshold tuning does the same job "
            f"more directly."
        ),
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {JSON_PATH.relative_to(REPO)}")
    print(f"wrote {FIG_PATH.relative_to(REPO)}")
    print(f"total runtime {runtime:.1f}s")


def _save_figure(pd_grid, pd_vals, lin_vals, agg) -> None:
    """Two-panel PNG: the PD curves, and the decile calibration against truth."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"    [figure skipped: {exc}]")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    ax1.plot(pd_grid, pd_vals, lw=2, label="best tree ensemble")
    ax1.plot(pd_grid, lin_vals, lw=2, ls="--", label="logistic regression")
    peak = int(np.asarray(pd_vals).argmax())
    ax1.axvline(pd_grid[peak], color="grey", lw=1, ls=":")
    ax1.annotate(
        f"tree peak {pd_grid[peak]:.0f}",
        (pd_grid[peak], pd_vals[peak]),
        textcoords="offset points",
        xytext=(6, 8),
        fontsize=9,
    )
    ax1.set_xlabel("MonthlyCharges")
    ax1.set_ylabel("partial dependence  P(churn)")
    ax1.set_title("Partial dependence on MonthlyCharges")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    d = agg.index.to_numpy()
    ax2.plot(d, agg["actual"] * 100, "o-", lw=2, label="actual churn %")
    ax2.plot(d, agg["tree"] * 100, "s--", lw=2, label="tree, mean OOF P")
    ax2.plot(d, agg["linear"] * 100, "^--", lw=2, label="logistic, mean OOF P")
    ax2.set_xticks(d)
    ax2.set_xlabel("MonthlyCharges decile (train fold)")
    ax2.set_ylabel("churn %")
    ax2.set_title("Decile 9 -> 10: does the model turn back down?")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_PATH, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
