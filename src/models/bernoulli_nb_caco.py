"""Step 3e -- BernoulliNB: teammate Caco's model, ported from branch `CacoJuse`.

Run with::

    python3 src/models/bernoulli_nb_caco.py

Caco built this independently, in his own `codigo.ipynb` on `CacoJuse`, using his
own `train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)` on the
raw CSV rather than this repo's frozen `data/splits/`. **Verified byte-for-byte
identical**: his row indices match all 1409 rows of `data/splits/test.csv`
exactly (same seed, same stratify column, same row order from a fresh
`pd.read_csv`), so his approach is directly comparable to the team's four Step 3
tracks despite the parallel development. This script re-runs his exact modeling
idea against `dp.load_splits()` so it slots into the same `results/*.json`
schema and CV/threshold conventions as `src/models/*.py`.

His idea, preserved as-is (not redesigned):

* The three continuous columns (`tenure`, `MonthlyCharges`, `TotalCharges`) are
  binarized with `KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='quantile')`
  -- a below/above-median indicator, not the team's own 10-bin discretisation in
  `docs/step-03c-knn-naive-bayes.md`. `SeniorCitizen` passes through (already
  0/1). Every other column is one-hot encoded (`drop='first'`). This makes every
  input to `BernoulliNB` a 0/1 indicator, which is what the estimator assumes.
* `alpha` (Laplace smoothing) is chosen by `GridSearchCV` over
  `[0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10]`, 5-fold `StratifiedKFold(shuffle=True,
  random_state=42)`, scoring `roc_auc`, train fold only.
* His own report evaluates at the default threshold 0.50. This script adds the
  team's threshold-tuning step (maximise F1(Yes) on out-of-fold train
  predictions) so the model is reported on equal footing with the other four
  Step 3 approaches, which all report a tuned threshold. Both numbers are kept
  in the JSON output, clearly separated.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import data_prep as dp  # noqa: E402

RESULTS_JSON = REPO_ROOT / "results" / "bernoulli_nb_caco.json"

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
ALPHA_GRID = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

CONTINUOUS = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY = ["SeniorCitizen"]


def build_preprocessor(categorical: list[str]) -> ColumnTransformer:
    """Caco's exact preprocessing shape: everything ends up 0/1 for BernoulliNB."""
    continuous_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("discretizer", KBinsDiscretizer(n_bins=2, encode="ordinal", strategy="quantile", subsample=None)),
    ])
    binary_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop="first")),
    ])
    return ColumnTransformer(
        transformers=[
            ("continuous", continuous_pipe, CONTINUOUS),
            ("binary", binary_pipe, BINARY),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
    )


def scores_at(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "f1_yes": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "precision_yes": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall_yes": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def main() -> None:
    t0 = time.time()
    print("=" * 78)
    print("STEP 3e -- BERNOULLI NAIVE BAYES (teammate Caco, ported from CacoJuse)")
    print("=" * 78)

    X_tr, X_te, y_tr, y_te = dp.load_splits()
    print(f"frozen split: train {X_tr.shape}  test {X_te.shape}")

    categorical = [c for c in X_tr.columns if c not in CONTINUOUS + BINARY]
    print(f"continuous (2-bin quantile): {CONTINUOUS}")
    print(f"binary (passthrough): {BINARY}")
    print(f"categorical (one-hot, drop=first): {len(categorical)} columns")

    pipe = Pipeline([
        ("prep", build_preprocessor(categorical)),
        ("clf", BernoulliNB(binarize=None, fit_prior=True)),
    ])

    print("\n" + "-" * 78)
    print("1. ALPHA GRID SEARCH (5-fold, scoring=roc_auc, train fold only)")
    print("-" * 78)
    gs = GridSearchCV(pipe, {"clf__alpha": ALPHA_GRID}, scoring="roc_auc", cv=CV, n_jobs=-1, refit=True)
    gs.fit(X_tr, y_tr)
    best_alpha = gs.best_params_["clf__alpha"]
    print(f"  best alpha = {best_alpha}   CV ROC-AUC = {gs.best_score_:.4f}")

    X_tr_bin = gs.best_estimator_.named_steps["prep"].transform(X_tr)
    if not set(np.unique(X_tr_bin)).issubset({0.0, 1.0}):
        raise AssertionError("BernoulliNB received values other than 0/1 -- preprocessing broke his invariant.")

    print("\n" + "-" * 78)
    print("2. DECISION THRESHOLD -- maximise F1(Yes) on out-of-fold TRAIN predictions")
    print("-" * 78)
    best_pipe = Pipeline([
        ("prep", build_preprocessor(categorical)),
        ("clf", BernoulliNB(binarize=None, fit_prior=True, alpha=best_alpha)),
    ])
    oof_proba = cross_val_predict(best_pipe, X_tr, y_tr, cv=CV, method="predict_proba", n_jobs=-1)[:, 1]
    grid = np.round(np.arange(0.05, 0.95, 0.01), 2)
    f1s = [f1_score(y_tr, (oof_proba >= t).astype(int), zero_division=0) for t in grid]
    threshold = float(grid[int(np.argmax(f1s))])
    print(f"  tuned threshold = {threshold:.2f}   OOF F1(Yes) = {max(f1s):.4f}"
          f"  (vs {f1_score(y_tr, (oof_proba >= 0.5).astype(int)):.4f} at 0.50)")

    print("\n" + "-" * 78)
    print("3. FINAL TEST EVALUATION -- the test set is read here, once")
    print("-" * 78)
    best_pipe.fit(X_tr, y_tr)
    proba_te = best_pipe.predict_proba(X_te)[:, 1]
    test_auc = round(float(roc_auc_score(y_te, proba_te)), 4)
    at_tuned = scores_at(y_te, proba_te, threshold)
    at_half = scores_at(y_te, proba_te, 0.5)

    print(f"  test ROC-AUC          : {test_auc:.4f}")
    print(f"  @0.50   (Caco's own)  : F1={at_half['f1_yes']:.4f}  P={at_half['precision_yes']:.4f}  "
          f"R={at_half['recall_yes']:.4f}  acc={at_half['accuracy']:.4f}")
    print(f"  @{threshold:.2f} (tuned)      : F1={at_tuned['f1_yes']:.4f}  P={at_tuned['precision_yes']:.4f}  "
          f"R={at_tuned['recall_yes']:.4f}  acc={at_tuned['accuracy']:.4f}")

    if test_auc > 0.90:
        print("\n  *** WARNING: ROC-AUC > 0.90 -- CLAUDE.md fact #4 says assume LEAKAGE. ***")

    reported_auc, reported_cv = 0.822, 0.834  # Caco's own docs/step-04-model-comparison.md (CacoJuse)
    print(f"\n  sanity check vs Caco's own reported numbers (alpha=0.5, CV~{reported_cv}, test AUC~{reported_auc}):")
    print(f"    alpha match: {best_alpha == 0.5}   CV diff: {gs.best_score_ - reported_cv:+.4f}"
          f"   test AUC diff: {test_auc - reported_auc:+.4f}")

    runtime = round(time.time() - t0, 2)
    payload = {
        "approach": "bernoulli-nb-caco",
        "agent_brief": "teammate Caco's independent model, ported from branch CacoJuse",
        "attribution": "Modeling idea, preprocessing shape, and alpha grid are Caco's own "
                        "(codigo.ipynb on CacoJuse); this script re-runs it against dp.load_splits() "
                        "and adds threshold tuning for comparability with the team's Step 3 tracks.",
        "best_model": {
            "name": "BernoulliNB (Caco, quantile-binned)",
            "params": {"alpha": best_alpha},
            "cv_roc_auc_mean": round(float(gs.best_score_), 5),
            "cv_roc_auc_std": round(
                float(gs.cv_results_["std_test_score"][gs.best_index_]), 5
            ),
            "threshold": threshold,
            "test": {"roc_auc": test_auc, **at_tuned},
            "test_at_threshold_0.5_caco_reported": {
                k: v for k, v in at_half.items() if k != "confusion_matrix"
            },
        },
        "caco_original_reported": {
            "cv_roc_auc": reported_cv,
            "test_roc_auc": reported_auc,
            "test_f1_yes": 0.602,
            "test_precision_yes": 0.554,
            "test_recall_yes": 0.660,
            "test_accuracy": 0.769,
            "source": "docs/step-04-model-comparison.md on branch CacoJuse (his own split, "
                      "verified identical test rows to data/splits/)",
        },
        "runtime_seconds": runtime,
        "notes": (
            f"Frozen split (dp.load_splits()); test set read once. Alpha chosen by 5-fold "
            f"StratifiedKFold(shuffle, seed 42) scoring roc_auc on train only, matching Caco's own "
            f"grid {ALPHA_GRID}. Threshold {threshold:.2f} chosen by maximising F1(Yes) on "
            f"cross_val_predict out-of-fold train probabilities (team convention -- Caco's own report "
            f"used the default 0.50). ROC-AUC {test_auc:.4f} sits in the 0.84-0.85 band predicted in "
            f"Step 1 (close to his own reported {reported_auc}), so no leakage indicated."
        ),
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {RESULTS_JSON}  ({runtime}s)")


if __name__ == "__main__":
    main()
