"""Step 3d -- Support Vector Machines plus explicit feature engineering.

Approach owner: the SVM/feature-engineering agent.  This is the only Step 3
module allowed to change the *feature representation*; the other three score the
standard ``data_prep`` preprocessor as-is.

What this file does, in order
-----------------------------
1. Loads the **frozen** split (``dp.load_splits()``).  It never calls
   ``dp.split()`` and never re-splits.
2. Tunes kernel / ``C`` / ``gamma`` / ``class_weight`` by 5-fold
   ``StratifiedKFold`` CV **on train only**, scoring ROC-AUC.
3. Runs one CV experiment per engineered feature (EDA 6.5, 6.6, 6.7, 7.2)
   against the unengineered baseline, for both the best RBF and the best linear
   configuration, and reports the delta.
4. Picks a decision threshold by maximising F1(Yes) over out-of-fold
   probabilities on **train**.
5. Touches ``X_te`` exactly once, at the very end.

Leakage discipline
------------------
Every transform that *learns* anything -- the scaler, the one-hot category
lists, the ``KBinsDiscretizer`` edges, the ``SplineTransformer`` knots -- lives
inside the ``Pipeline`` and is therefore fitted on the training fold only.  The
engineered columns that are *arithmetic* (the interaction string, the
``TotalCharges`` residual, the add-on count) are stateless by construction:
their ``fit`` is a no-op, so there is nothing to leak even in principle.  No
target statistics are used anywhere (no target encoding), which is the usual way
feature engineering turns into an AUC > 0.90 fantasy on this dataset.

Runtime note
------------
``SVC`` is O(n^2)-O(n^3) and ``probability=True`` refits with an internal 5-fold
Platt calibration, i.e. ~6x the cost of a plain fit.  So all *search* work
(grid + feature experiments) runs with ``probability=False`` and scores ROC-AUC
off ``decision_function`` -- ROC-AUC is rank-based, so the ranking is identical
and the search is ~6x cheaper.  Probability calibration is paid exactly twice: once
for the out-of-fold threshold sweep, once for the final model -- via
``SVC(probability=True)`` for the SVC families, or ``CalibratedClassifierCV``
for ``LinearSVC``, which never had the flag (see ``make_probabilistic``).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, SplineTransformer
from sklearn.svm import SVC, LinearSVC

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
import data_prep as dp  # noqa: E402

# sklearn 1.9 deprecates SVC(probability=True) in favour of CalibratedClassifierCV.
# The brief specifies probability=True, so we keep it and silence the notice.
warnings.filterwarnings("ignore", message=".*probability.* parameter was deprecated.*")

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
N_JOBS = 2  # the box has 2 cores; more processes only adds contention

#: The six add-on columns.  ``'No internet service'`` was already collapsed to
#: ``'No'`` by ``dp.clean``, so ``== 'Yes'`` counts real add-ons (EDA 6.6).
ADDON_COLUMNS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless, arithmetic feature construction.  DataFrame in, DataFrame out.

    Deliberately learns nothing from the data: ``fit`` only records the input
    column order so the transformer is a well-behaved sklearn citizen.  Every
    output column is a deterministic function of the same row's inputs, so it
    cannot leak across rows or across folds.  Anything that *does* need to learn
    (bin edges, spline knots, scaling) is handled downstream in the
    ``ColumnTransformer``.

    Parameters
    ----------
    interaction:
        Add ``Contract_x_InternetService`` as a single 9-level categorical
        (EDA 6.5: 54.61% churn for M2M/Fiber, n=2128, down to 0.78% for
        Two-year/No-internet, n=638 -- a 70x spread that a linear decision
        boundary in the marginal dummies cannot express).
    total_charges_residual:
        Replace ``TotalCharges`` with ``TotalCharges - tenure * MonthlyCharges``
        (EDA 7.2: the raw column is 99.91% explained by that product, R^2 =
        0.9991; the residual has median 0.00 and std 67.25 and encodes promos
        and mid-contract plan changes).
    drop_total_charges:
        Drop ``TotalCharges`` outright -- the other half of the EDA 7.2
        recommendation, kept as a separate arm so "residual" is measured against
        both "keep" and "drop".
    addon_count:
        Add ``n_addons`` (0-6).  EDA 6.6 warns the apparent U-shape is a
        composition artifact; this arm exists to test that warning, not because
        the feature is expected to work.
    """

    def __init__(
        self,
        interaction: bool = False,
        total_charges_residual: bool = False,
        drop_total_charges: bool = False,
        addon_count: bool = False,
    ):
        self.interaction = interaction
        self.total_charges_residual = total_charges_residual
        self.drop_total_charges = drop_total_charges
        self.addon_count = addon_count

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        out = X.copy()

        if self.interaction:
            out["Contract_x_InternetService"] = (
                out["Contract"].astype(str) + " | " + out["InternetService"].astype(str)
            )

        if self.addon_count:
            out["n_addons"] = sum(
                (out[c] == "Yes").astype("int64") for c in ADDON_COLUMNS
            ).astype("float64")

        if self.total_charges_residual and "TotalCharges" in out.columns:
            out["TotalChargesResidual"] = (
                out["TotalCharges"] - out["tenure"] * out["MonthlyCharges"]
            ).astype("float64")
            out = out.drop(columns=["TotalCharges"])
        elif self.drop_total_charges and "TotalCharges" in out.columns:
            out = out.drop(columns=["TotalCharges"])

        return out


def column_groups(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """(numeric, binary, categorical) for an engineered frame.

    ``dp.infer_feature_groups`` is driven by hard-coded column *names*, so it
    would file the engineered columns under ``categorical``.  This is the same
    policy expressed by dtype: ``SeniorCitizen`` passes through as a 0/1 flag,
    every other numeric column is scaled, everything else is one-hot encoded.
    """
    binary = [c for c in frame.columns if c in dp.BINARY_COLUMNS]
    numeric = [
        c
        for c in frame.columns
        if c not in binary and pd.api.types.is_numeric_dtype(frame[c])
    ]
    categorical = [c for c in frame.columns if c not in binary and c not in numeric]
    return numeric, binary, categorical


def build_pipeline(
    X_ref: pd.DataFrame,
    estimator,
    *,
    interaction: bool = False,
    total_charges_residual: bool = False,
    drop_total_charges: bool = False,
    addon_count: bool = False,
    monthly_charges: str = "linear",  # "linear" | "bins" | "spline" | "bins+linear"
    n_bins: int = 10,
) -> Pipeline:
    """Feature engineering -> ``dp.build_preprocessor`` -> estimator, unfitted.

    ``X_ref`` is used only to read *column names* off a stateless transform; no
    statistic is computed from it, so passing the full training frame here does
    not leak anything into the folds.
    """
    fe = FeatureEngineer(
        interaction=interaction,
        total_charges_residual=total_charges_residual,
        drop_total_charges=drop_total_charges,
        addon_count=addon_count,
    )
    engineered = fe.fit_transform(X_ref)
    numeric, binary, categorical = column_groups(engineered)

    mc_extra = None
    if monthly_charges == "bins":
        numeric = [c for c in numeric if c != "MonthlyCharges"]
        mc_extra = KBinsDiscretizer(
            n_bins=n_bins, encode="onehot-dense", strategy="quantile"
        )
    elif monthly_charges == "bins+linear":
        mc_extra = KBinsDiscretizer(
            n_bins=n_bins, encode="onehot-dense", strategy="quantile"
        )
    elif monthly_charges == "spline":
        numeric = [c for c in numeric if c != "MonthlyCharges"]
        mc_extra = SplineTransformer(n_knots=5, degree=3, include_bias=False)
    elif monthly_charges != "linear":
        raise ValueError(f"unknown monthly_charges={monthly_charges!r}")

    pre = dp.build_preprocessor(
        numeric=numeric, binary=binary, categorical=categorical, scale=True
    )
    if mc_extra is not None:
        # The ColumnTransformer is UNFITTED; appending to its ``transformers``
        # list keeps it unfitted, so the bin edges / spline knots are still
        # learned per training fold inside the pipeline.
        pre.transformers = list(pre.transformers) + [
            ("mc_nonlinear", mc_extra, ["MonthlyCharges"])
        ]

    return Pipeline([("fe", fe), ("pre", pre), ("clf", estimator)])


# --------------------------------------------------------------------------- #
# Evaluation helpers -- all of these see TRAIN only
# --------------------------------------------------------------------------- #
def cv_auc(pipe: Pipeline, X, y, label: str = "") -> tuple[float, float, float]:
    """5-fold CV ROC-AUC on train.  Returns (mean, std, seconds)."""
    t0 = time.perf_counter()
    scores = cross_val_score(pipe, X, y, cv=CV, scoring="roc_auc", n_jobs=N_JOBS)
    elapsed = time.perf_counter() - t0
    if label:
        print(
            f"  {label:<52s} AUC {scores.mean():.4f} +/- {scores.std():.4f}"
            f"   [{elapsed:5.1f}s]",
            flush=True,
        )
    return float(scores.mean()), float(scores.std()), elapsed


def positive_metrics(y_true, y_pred) -> dict:
    return {
        "f1_yes": float(f1_score(y_true, y_pred, pos_label=1)),
        "precision_yes": float(precision_score(y_true, y_pred, pos_label=1)),
        "recall_yes": float(recall_score(y_true, y_pred, pos_label=1)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def pick_threshold(y_true, proba) -> tuple[float, float]:
    """Threshold maximising F1 on the positive class, chosen on TRAIN OOF probs."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    f1s = [f1_score(y_true, (proba >= t).astype(int), pos_label=1) for t in grid]
    best = int(np.argmax(f1s))
    return float(grid[best]), float(f1s[best])


# --------------------------------------------------------------------------- #
def make_probabilistic(estimator):
    """Return a version of ``estimator`` that exposes ``predict_proba``.

    ``SVC(probability=True)`` is what the brief asks for and is used for the SVC
    families.  Note scikit-learn 1.9 emits a ``FutureWarning``: the flag is
    deprecated in favour of ``CalibratedClassifierCV(SVC(), ensemble=False)``
    and will be removed in 1.11.  ``LinearSVC`` never had the flag, so it is
    wrapped in ``CalibratedClassifierCV`` (Platt/sigmoid, 5-fold) -- the same
    calibration ``probability=True`` performs internally.
    """
    if isinstance(estimator, SVC):
        return SVC(**{**estimator.get_params(), "probability": True,
                      "random_state": RANDOM_STATE})
    return CalibratedClassifierCV(estimator, method="sigmoid", cv=5)


def main() -> None:
    t_start = time.perf_counter()
    svc_seconds = 0.0

    X_tr, X_te, y_tr, y_te = dp.load_splits()
    print(
        f"Frozen split: train {X_tr.shape}, test {X_te.shape}, "
        f"train churn {y_tr.mean():.4%}",
        flush=True,
    )

    variants: list[dict] = []

    def record(name, mean, std, note, elapsed=None):
        variants.append(
            {
                "name": name,
                "cv_roc_auc_mean": round(mean, 4),
                "cv_roc_auc_std": round(std, 4),
                "note": note if elapsed is None else f"{note} [{elapsed:.0f}s CV]",
            }
        )

    # ---------------------------------------------------------------- Part A #
    # Every arm here uses the *unengineered* feature set, so the kernel choice
    # is not confounded with the feature choice.  ``class_weight='balanced'`` is
    # tried for each family and, if it wins, is carried into Part B -- otherwise
    # the feature experiments would be run on a knowingly inferior base.
    print("\n=== Part A: kernel and hyper-parameter search (baseline features) ===")

    print("--- LinearSVC (squared hinge, liblinear; scored via decision_function) ---")
    linsvc_best = None  # (auc, std, kwargs)
    for C in (0.01, 0.1, 1.0):
        kw = {"C": C, "random_state": RANDOM_STATE, "max_iter": 20000, "dual": "auto"}
        m, s, e = cv_auc(build_pipeline(X_tr, LinearSVC(**kw)), X_tr, y_tr,
                         f"LinearSVC  C={C}")
        svc_seconds += e
        record(f"LinearSVC C={C}", m, s, "no predict_proba; AUC off decision_function", e)
        if linsvc_best is None or m > linsvc_best[0]:
            linsvc_best = (m, s, kw)

    print("--- SVC, linear kernel (hinge, libsvm) ---")
    svclin_best = None
    for C in (0.03, 0.3, 1.0):
        kw = {"kernel": "linear", "C": C, "cache_size": 500}
        m, s, e = cv_auc(build_pipeline(X_tr, SVC(**kw)), X_tr, y_tr, f"SVC linear C={C}")
        svc_seconds += e
        record(f"SVC linear C={C}", m, s, "baseline features", e)
        if svclin_best is None or m > svclin_best[0]:
            svclin_best = (m, s, kw)

    print("--- SVC, RBF kernel ---")
    rbf_best = None
    for C in (1.0, 3.0, 10.0):
        for gamma in ("scale", 0.02):
            kw = {"kernel": "rbf", "C": C, "gamma": gamma, "cache_size": 500}
            m, s, e = cv_auc(build_pipeline(X_tr, SVC(**kw)), X_tr, y_tr,
                             f"SVC rbf    C={C} gamma={gamma}")
            svc_seconds += e
            record(f"SVC rbf C={C} gamma={gamma}", m, s, "baseline features", e)
            if rbf_best is None or m > rbf_best[0]:
                rbf_best = (m, s, kw)

    print("--- class_weight='balanced' (2.77:1 imbalance), best config of each family ---")
    families = {"LinearSVC": [LinearSVC, linsvc_best],
                "SVC-linear": [SVC, svclin_best],
                "SVC-rbf": [SVC, rbf_best]}
    for tag, (cls, best) in families.items():
        kw = dict(best[2], class_weight="balanced")
        m, s, e = cv_auc(build_pipeline(X_tr, cls(**kw)), X_tr, y_tr,
                         f"{tag} {best[2]} balanced")
        svc_seconds += e
        record(f"{tag} balanced", m, s, "class_weight='balanced'", e)
        if m > best[0]:
            families[tag][1] = (m, s, kw)

    bases = {tag: (cls, best[2], best[0]) for tag, (cls, best) in families.items()}
    print("\nPart A winners (carried into Part B):")
    for tag, (_, kw, auc) in bases.items():
        print(f"  {tag:<11s} {kw}  AUC {auc:.4f}")

    # ---------------------------------------------------------------- Part B #
    print("\n=== Part B: feature experiments (5-fold CV on TRAIN, per family) ===")
    EXPERIMENTS = [
        ("baseline", {}),
        ("+ Contract x InternetService", {"interaction": True}),
        ("+ MonthlyCharges 10 bins", {"monthly_charges": "bins"}),
        ("+ MonthlyCharges bins + linear", {"monthly_charges": "bins+linear"}),
        ("+ MonthlyCharges spline", {"monthly_charges": "spline"}),
        ("+ TotalCharges residual", {"total_charges_residual": True}),
        ("- TotalCharges dropped", {"drop_total_charges": True}),
        ("+ addon count", {"addon_count": True}),
    ]

    exp_results: dict[str, dict[str, tuple[float, float]]] = {}
    combos: dict[str, dict] = {}
    for tag, (cls, kw, _) in bases.items():
        print(f"--- {tag}: {kw} ---")
        exp_results[tag] = {}
        for name, fkw in EXPERIMENTS:
            m, s, e = cv_auc(build_pipeline(X_tr, cls(**kw), **fkw), X_tr, y_tr,
                             f"{tag}: {name}")
            svc_seconds += e
            exp_results[tag][name] = (m, s)
            record(f"{tag} {name}", m, s, "feature experiment", e)

        # Combination arm: every feature with a positive CV delta on this family.
        base = exp_results[tag]["baseline"][0]
        fkw: dict = {}
        if exp_results[tag]["+ Contract x InternetService"][0] > base:
            fkw["interaction"] = True
        mc = {k: exp_results[tag][f"+ MonthlyCharges {k}"][0]
              for k in ("10 bins", "bins + linear", "spline")}
        best_mc = max(mc, key=mc.get)
        if mc[best_mc] > base:
            fkw["monthly_charges"] = {"10 bins": "bins", "bins + linear": "bins+linear",
                                      "spline": "spline"}[best_mc]
        if exp_results[tag]["+ TotalCharges residual"][0] > base:
            fkw["total_charges_residual"] = True
        elif exp_results[tag]["- TotalCharges dropped"][0] > base:
            fkw["drop_total_charges"] = True
        if exp_results[tag]["+ addon count"][0] > base:
            fkw["addon_count"] = True
        combos[tag] = fkw

        if fkw:
            m, s, e = cv_auc(build_pipeline(X_tr, cls(**kw), **fkw), X_tr, y_tr,
                             f"{tag}: combined {fkw}")
            svc_seconds += e
            exp_results[tag]["combined"] = (m, s)
            record(f"{tag} combined {fkw}", m, s, "all positive-delta features", e)
        else:
            print(f"  {tag}: no feature had a positive delta -- combination arm skipped")
            exp_results[tag]["combined"] = exp_results[tag]["baseline"]

    # ------------------------------------------------------- pick the winner #
    candidates = []
    for tag, (cls, kw, _) in bases.items():
        for label, key in (("baseline features", "baseline"),
                           ("engineered features", "combined")):
            fkw = {} if key == "baseline" else combos[tag]
            if key == "combined" and not fkw:
                continue
            m, s = exp_results[tag][key]
            candidates.append((m, s, tag, cls, kw, fkw, label))
    best_m, best_s, best_tag, best_cls, best_kw, best_fkw, best_label = max(
        candidates, key=lambda t: t[0]
    )
    best_name = f"{best_tag} ({best_label})"
    print("\nCV leaderboard (top 5):")
    for m, s, tag, _, kw, fkw, label in sorted(candidates, key=lambda t: -t[0])[:5]:
        print(f"  {m:.4f} +/- {s:.4f}  {tag:<11s} {label:<19s} {fkw or '-'}")
    print(f"\nWINNER on CV: {best_name} {best_kw} {best_fkw or 'no engineered features'} "
          f"AUC {best_m:.4f} +/- {best_s:.4f}")

    # -------------------------------------------- threshold, on TRAIN OOF only #
    print("\n=== Threshold selection (out-of-fold probabilities, TRAIN only) ===")
    final_pipe = build_pipeline(X_tr, make_probabilistic(best_cls(**best_kw)), **best_fkw)
    t0 = time.perf_counter()
    oof = cross_val_predict(final_pipe, X_tr, y_tr, cv=CV, method="predict_proba",
                            n_jobs=N_JOBS)[:, 1]
    e = time.perf_counter() - t0
    svc_seconds += e
    oof_auc = roc_auc_score(y_tr, oof)
    threshold, oof_f1 = pick_threshold(y_tr.to_numpy(), oof)
    oof_f1_half = f1_score(y_tr, (oof >= 0.5).astype(int))
    print(f"  OOF AUC (calibrated) {oof_auc:.4f}  |  best threshold {threshold:.2f}"
          f"  -> OOF F1(Yes) {oof_f1:.4f}  (0.50 gives {oof_f1_half:.4f})   [{e:.1f}s]",
          flush=True)

    # -------------------------------------------- top drivers (linear proxy) #
    print("\n=== Top drivers (tuned SVC-linear coefficients -- interpretable proxy) ===")
    lin_cls, lin_kw, _ = bases["SVC-linear"]
    driver_pipe = build_pipeline(X_tr, lin_cls(**lin_kw), **combos["SVC-linear"])
    driver_pipe.fit(X_tr, y_tr)
    names = driver_pipe.named_steps["pre"].get_feature_names_out()
    coefs = np.asarray(driver_pipe.named_steps["clf"].coef_).ravel()
    order = np.argsort(-np.abs(coefs))[:12]
    top_drivers = [
        {"feature": str(names[i]), "coef": round(float(coefs[i]), 4),
         "direction": "increases churn" if coefs[i] > 0 else "decreases churn"}
        for i in order
    ]
    for d in top_drivers:
        print(f"  {d['coef']:+.4f}  {d['feature']}")

    # ------------------------------------------------- FINAL TEST -- ONCE ONLY #
    print("\n=== FINAL TEST EVALUATION (the test set is read here, once) ===")
    t0 = time.perf_counter()
    final_pipe.fit(X_tr, y_tr)
    proba_te = final_pipe.predict_proba(X_te)[:, 1]
    svc_seconds += time.perf_counter() - t0

    test_auc = float(roc_auc_score(y_te, proba_te))
    pred_tuned = (proba_te >= threshold).astype(int)
    pred_half = (proba_te >= 0.5).astype(int)
    m_tuned = positive_metrics(y_te, pred_tuned)
    m_half = positive_metrics(y_te, pred_half)
    cm = confusion_matrix(y_te, pred_tuned).tolist()

    print(f"  ROC-AUC        {test_auc:.4f}")
    print(f"  threshold      {threshold:.2f} (chosen on train OOF)")
    print(f"  F1(Yes)        {m_tuned['f1_yes']:.4f}   (at 0.50: {m_half['f1_yes']:.4f})")
    print(f"  precision(Yes) {m_tuned['precision_yes']:.4f}   (at 0.50: {m_half['precision_yes']:.4f})")
    print(f"  recall(Yes)    {m_tuned['recall_yes']:.4f}   (at 0.50: {m_half['recall_yes']:.4f})")
    print(f"  accuracy       {m_tuned['accuracy']:.4f}   (all-No baseline 0.7346)")
    print(f"  confusion      {cm}")

    if test_auc > 0.90:
        print("\n*** SANITY BOUND BREACHED: test ROC-AUC > 0.90 where 0.84-0.85 was "
              "expected. Treat as leakage, not a win. ***")

    # ------------------------------------------------------------------ JSON #
    feature_experiments = []
    for tag in bases:
        base = exp_results[tag]["baseline"][0]
        for name, _ in EXPERIMENTS[1:]:
            delta = exp_results[tag][name][0] - base
            feature_experiments.append({
                "feature": f"[{tag}] {name}",
                "cv_roc_auc_delta": round(delta, 4),
                "verdict": "helped" if delta > 0.0015 else "hurt" if delta < -0.0015 else "neutral",
            })
        delta = exp_results[tag]["combined"][0] - base
        feature_experiments.append({
            "feature": f"[{tag}] combined {combos[tag] or 'none'}",
            "cv_roc_auc_delta": round(delta, 4),
            "verdict": "helped" if delta > 0.0015 else "hurt" if delta < -0.0015 else "neutral",
        })

    runtime = time.perf_counter() - t_start
    json_params = {k: v for k, v in best_kw.items() if k != "cache_size"}
    payload = {
        "approach": "svm-feature-engineering",
        "agent_brief": "SVM plus engineered features",
        "variants": variants,
        "best_model": {
            "name": best_name,
            "params": {**json_params, "features": best_fkw or "standard preprocessor only",
                       "calibration": "SVC(probability=True)" if best_cls is SVC
                       else "CalibratedClassifierCV(sigmoid, cv=5)"},
            "cv_roc_auc_mean": round(best_m, 4),
            "cv_roc_auc_std": round(best_s, 4),
            "threshold": threshold,
            "test": {
                "roc_auc": round(test_auc, 4),
                "f1_yes": round(m_tuned["f1_yes"], 4),
                "precision_yes": round(m_tuned["precision_yes"], 4),
                "recall_yes": round(m_tuned["recall_yes"], 4),
                "accuracy": round(m_tuned["accuracy"], 4),
                "confusion_matrix": cm,
            },
            "test_at_threshold_0.5": {
                "f1_yes": round(m_half["f1_yes"], 4),
                "precision_yes": round(m_half["precision_yes"], 4),
                "recall_yes": round(m_half["recall_yes"], 4),
                "accuracy": round(m_half["accuracy"], 4),
            },
        },
        "feature_experiments": feature_experiments,
        "top_drivers": top_drivers,
        "runtime_seconds": round(runtime, 1),
        "notes": (
            f"Frozen split (dp.load_splits); 5-fold StratifiedKFold on TRAIN for every "
            f"tuning, feature-selection and threshold decision; test read exactly once. "
            f"All search arms ran with probability=False, scoring ROC-AUC off "
            f"decision_function (rank-identical to predict_proba, ~6x cheaper); "
            f"calibration was paid twice only -- OOF threshold sweep and final fit. "
            f"{svc_seconds:.0f}s of the {runtime:.0f}s total was SVM fitting on 2 cores. "
            f"OOF AUC of the final calibrated pipeline {oof_auc:.4f}; threshold "
            f"{threshold:.2f} maximises OOF F1(Yes)={oof_f1:.4f} vs {oof_f1_half:.4f} at 0.50."
        ),
    }
    out_path = ROOT / "results" / "svm_features.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {out_path}  (total runtime {runtime:.0f}s, SVM fits {svc_seconds:.0f}s)")


if __name__ == "__main__":
    main()
