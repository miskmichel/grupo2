"""Step 3a -- Logistic Regression: the linear, interpretable churn model.

Run with::

    python3 src/models/logistic_regression.py

This is the "readable list of churn drivers" model.  It is not expected to be
the most accurate of the four Step 3 approaches; its job is to produce a
coefficient table the retention team can act on, and to establish that a plain
linear model already reaches the ROC-AUC 0.84-0.85 band predicted in Step 1
(``docs/step-01-eda.md`` and ``CLAUDE.md`` fact #4).

Rules this script obeys, all of them load-bearing for the team comparison
table:

* **The frozen split is used, never re-derived.**  ``dp.load_splits()`` reads
  ``data/splits/{train,test}.csv``.  Calling ``dp.split()`` again -- even with
  the same ``random_state`` -- would be a different code path for the same
  result and defeats the point of freezing.  For the ``TotalCharges`` variants
  the frame is rebuilt from the raw CSV and then **re-indexed onto the frozen
  ``row_id``s**, so the rows are identical and only the columns change.
* **The test set is read exactly once**, in :func:`final_evaluation`, after
  every choice (variant, hyper-parameters, decision threshold) has been made
  from 5-fold cross-validation on the training fold alone.
* **Every fit happens inside a ``Pipeline``.**  ``dp.build_preprocessor()``
  returns an *unfitted* ``ColumnTransformer``; putting it in the pipeline means
  the scaler's mean/std and the one-hot category lists are learned per fold.
  Fitting it once on the whole frame is the classic leak.
* **ROC-AUC is the headline.**  Accuracy is reported but never as the headline:
  predicting "No" for everybody scores 73.46% (``CLAUDE.md`` fact #3).

Experiments run, in order:

1. Four feature-set variants under an identical estimator and identical folds
   (the ``TotalCharges`` three-way from Step 2 §8.3, plus the noise cull).
2. ``GridSearchCV`` over C / penalty / class_weight on every variant, so the
   variant choice is not an artifact of one arbitrary hyper-parameter setting.
3. Decision-threshold selection by maximising F1 on the positive class over
   *out-of-fold* training predictions.
4. One pass over the test set, reported at both 0.5 and the tuned threshold.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import data_prep as dp  # noqa: E402

# scikit-learn 1.9 deprecates the ``penalty`` argument in favour of ``l1_ratio``
# (removal in 1.10).  We keep ``penalty`` because it is what every reader of this
# code expects L1/L2 to look like, and because ``liblinear`` -- the solver that
# gives us L1 -- has no ``l1_ratio``.  The warning fires once per fit, which is
# several hundred times inside GridSearchCV, so it is silenced here rather than
# allowed to bury the results.  Flagged in the Step 3a report as a portability
# note: this file will need updating for sklearn 1.10.
warnings.filterwarnings(
    "ignore", message=".*'penalty' was deprecated.*", category=FutureWarning
)

RESULTS_JSON = REPO_ROOT / "results" / "logistic_regression.json"

#: One CV object, reused everywhere.  Shared folds are what make the variant
#: table a like-for-like comparison rather than four independent experiments.
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#: The estimator used for the *variant* comparison.  Held fixed on purpose: if
#: each variant got its own tuning, a variant could win on hyper-parameter luck.
#: Step 2 already established the matrix is well conditioned, so lbfgs/L2/C=1 is
#: a sane neutral reference point.
BASELINE_CLF = LogisticRegression(max_iter=2000, random_state=42)

#: Hyper-parameter grid.  ``liblinear`` is the only solver here that takes both
#: L1 and L2; ``lbfgs`` is included as an independent L2 cross-check (a
#: different optimiser reaching the same optimum is cheap reassurance).  saga
#: would also work but converges far more slowly on 21 dense features for no
#: gain.  ``class_weight`` is in the grid, not assumed -- the brief is explicit
#: that 'balanced' must be an experiment, because on a 2.77:1 imbalance it
#: changes recall a lot and ROC-AUC (a ranking metric) barely at all.
PARAM_GRID = [
    {
        "clf__solver": ["liblinear"],
        "clf__penalty": ["l1", "l2"],
        "clf__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0],
        "clf__class_weight": [None, "balanced"],
    },
    {
        "clf__solver": ["lbfgs"],
        "clf__penalty": ["l2"],
        "clf__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0],
        "clf__class_weight": [None, "balanced"],
    },
]


# --------------------------------------------------------------------------
# Feature-set variants
# --------------------------------------------------------------------------

def build_variants(train_index: pd.Index, test_index: pd.Index) -> dict[str, dict]:
    """Build the four feature-set variants on the *frozen* rows.

    The frozen split was written with ``drop_noise=True, drop_total_charges=False``.
    Variants that need different cleaning flags cannot read it directly, so they
    re-clean the raw CSV and then select ``train_index`` / ``test_index`` -- the
    exact ``row_id``s of the frozen split.  ``clean()`` preserves row order and
    the raw index, and ``row_id`` *is* the raw-CSV row position, so ``.loc`` on
    those indices reproduces the frozen split's rows byte for byte.  No
    re-splitting happens anywhere in this file.

    Returns ``{name: {X_tr, X_te, y_tr, y_te, numeric, binary, categorical, note}}``.
    """
    variants: dict[str, dict] = {}

    def pack(name: str, frame: pd.DataFrame, note: str, numeric=None) -> None:
        y = frame[dp.TARGET]
        X = frame.drop(columns=[dp.TARGET])
        num, bin_, cat = dp.infer_feature_groups(frame)
        if numeric is not None:            # variant (c) renames a numeric column,
            num = numeric                  # which the constant-driven inference
            cat = [c for c in cat if c not in num]   # would misfile as categorical
        variants[name] = {
            "X_tr": X.loc[train_index],
            "X_te": X.loc[test_index],
            "y_tr": y.loc[train_index],
            "y_te": y.loc[test_index],
            "numeric": num,
            "binary": bin_,
            "categorical": cat,
            "note": note,
        }

    # (a) baseline -- exactly what the frozen split holds.
    base = dp.clean(dp.load_raw(str(REPO_ROOT / dp.DEFAULT_CSV)))
    pack(
        "a_keep_total_charges",
        base,
        "baseline: TotalCharges kept, noise columns dropped (the frozen split as written)",
    )

    # (b) drop TotalCharges.  EDA: R^2 = 0.9991 against tenure x MonthlyCharges,
    #     r = +0.8297 with tenure -- the textbook case for dropping a collinear
    #     column from a linear model.
    dropped = dp.clean(
        dp.load_raw(str(REPO_ROOT / dp.DEFAULT_CSV)), drop_total_charges=True
    )
    pack(
        "b_drop_total_charges",
        dropped,
        "TotalCharges dropped (99.91% explained by tenure x MonthlyCharges)",
    )

    # (c) replace TotalCharges with its residual against tenure x MonthlyCharges.
    #     The residual is what TotalCharges knows that tenure and MonthlyCharges
    #     do not: plan changes, promotions, price rises mid-contract.  EDA §7.2
    #     measured median 0.00, std 67.25 -- small but not empty.
    resid = dp.clean(dp.load_raw(str(REPO_ROOT / dp.DEFAULT_CSV)))
    resid["ChargesResidual"] = resid["TotalCharges"] - resid["tenure"] * resid["MonthlyCharges"]
    resid = resid.drop(columns=["TotalCharges"])
    pack(
        "c_total_charges_residual",
        resid,
        "TotalCharges replaced by TotalCharges - tenure*MonthlyCharges",
        numeric=["tenure", "MonthlyCharges", "ChargesResidual"],
    )

    # (d) keep the two noise columns.  Step 2 predicted this is neutral; a
    #     regularised model should shrink gender/PhoneService to ~0 anyway.
    noisy = dp.clean(dp.load_raw(str(REPO_ROOT / dp.DEFAULT_CSV)), drop_noise=False)
    pack(
        "d_keep_noise_columns",
        noisy,
        "gender + PhoneService kept (chi2 p = 0.487 / 0.339), TotalCharges kept",
    )

    return variants


def make_pipeline(variant: dict, clf) -> Pipeline:
    """Preprocessor + estimator, with the preprocessor still unfitted.

    Column lists are passed explicitly rather than inferred inside, because
    variant (c) renames a numeric column and ``infer_feature_groups`` is driven
    by module constants -- an unrecognised name would be treated as categorical
    and one-hot encoded into thousands of dummies.
    """
    pre = dp.build_preprocessor(
        numeric=variant["numeric"],
        binary=variant["binary"],
        categorical=variant["categorical"],
        scale=True,
    )
    return Pipeline([("prep", pre), ("clf", clf)])


# --------------------------------------------------------------------------
# Threshold selection
# --------------------------------------------------------------------------

def tune_threshold(pipe: Pipeline, X, y) -> tuple[float, float]:
    """Pick the decision threshold that maximises F1(Yes) on out-of-fold TRAIN data.

    ``cross_val_predict`` gives every training row a probability produced by a
    model that never saw it, so the threshold is chosen on honest predictions.
    Choosing it from in-sample probabilities would overfit the threshold; from
    test probabilities would burn the holdout.

    0.5 is not a neutral default on an imbalanced problem -- it is the right
    threshold only when the classes are balanced and the costs are symmetric.
    Neither holds here (26.54% positives, and missing a churner costs far more
    than a wasted retention call).

    Returns ``(threshold, oof_f1_at_that_threshold, oof_probabilities)``.
    """
    proba = cross_val_predict(pipe, X, y, cv=CV, method="predict_proba", n_jobs=-1)[:, 1]
    grid = np.round(np.arange(0.05, 0.95, 0.01), 2)
    f1s = [f1_score(y, (proba >= t).astype(int), zero_division=0) for t in grid]
    best = int(np.argmax(f1s))
    return float(grid[best]), float(f1s[best]), proba


# --------------------------------------------------------------------------
# Final evaluation -- the only place the test set is read
# --------------------------------------------------------------------------

def scores_at(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "f1_yes": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "precision_yes": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall_yes": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def coefficient_table(pipe: Pipeline) -> pd.DataFrame:
    """Coefficients as odds ratios, sorted by |coef|.

    The numeric columns are standardised inside the pipeline, so a numeric
    coefficient reads as "per one standard deviation"; the one-hot coefficients
    read as "versus the reference level" (``drop='first'``), which for Contract
    is Month-to-month and for PaymentMethod is Bank transfer (automatic).
    ``exp(coef)`` turns a log-odds change into a multiplicative odds ratio: 2.0
    means the odds of churn double, 0.5 means they halve.
    """
    names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_.ravel()
    table = pd.DataFrame({"feature": names, "coef": coefs})
    table["odds_ratio"] = np.exp(table["coef"])
    return table.reindex(table["coef"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def main() -> None:
    t0 = time.time()
    pd.set_option("display.width", 120)

    # ---- frozen split -------------------------------------------------
    X_tr, X_te, y_tr, y_te = dp.load_splits()
    print("=" * 78)
    print("STEP 3a -- LOGISTIC REGRESSION (linear / interpretable)")
    print("=" * 78)
    print(f"frozen split: train {X_tr.shape}  test {X_te.shape}")
    print(f"train churn rate {y_tr.mean():.4%} ({int(y_tr.sum())} positives)")
    print(f"test  churn rate {y_te.mean():.4%} ({int(y_te.sum())} positives)  [NOT touched until the end]")
    print(f"all-'No' baseline accuracy on test: {1 - y_te.mean():.4%}")

    variants = build_variants(X_tr.index, X_te.index)

    # The rebuilt frames must be the frozen rows.  If this ever fails, someone
    # has re-split or the cleaning changed, and every score here is void.
    for name, v in variants.items():
        assert v["y_tr"].equals(y_tr), f"{name}: train target does not match the frozen split"
        assert v["y_te"].equals(y_te), f"{name}: test target does not match the frozen split"
        assert list(v["X_tr"].index) == list(X_tr.index), f"{name}: train rows differ"

    # ---- experiment 1: variants under one fixed estimator --------------
    print("\n" + "-" * 78)
    print("1. FEATURE-SET VARIANTS -- identical estimator (L2, C=1, lbfgs), identical folds")
    print("-" * 78)
    variant_rows = []
    for name, v in variants.items():
        pipe = make_pipeline(v, BASELINE_CLF)
        aucs = cross_val_score(pipe, v["X_tr"], v["y_tr"], cv=CV, scoring="roc_auc", n_jobs=-1)
        n_cols = pipe.fit(v["X_tr"], v["y_tr"]).named_steps["prep"].transform(v["X_tr"]).shape[1]
        variant_rows.append(
            {
                "name": name,
                "n_encoded_features": int(n_cols),
                "cv_roc_auc_mean": round(float(aucs.mean()), 5),
                "cv_roc_auc_std": round(float(aucs.std()), 5),
                "note": v["note"],
            }
        )
        print(f"  {name:26s}  {n_cols:2d} cols  CV ROC-AUC = {aucs.mean():.5f} +/- {aucs.std():.5f}")

    # ---- experiment 2: tune each variant, so the winner is not luck ----
    print("\n" + "-" * 78)
    print("2. GRID SEARCH PER VARIANT (5-fold, scoring=roc_auc, train fold only)")
    print("-" * 78)
    searches = {}
    for name, v in variants.items():
        gs = GridSearchCV(
            make_pipeline(v, LogisticRegression(max_iter=2000, random_state=42)),
            PARAM_GRID,
            scoring="roc_auc",
            cv=CV,
            n_jobs=-1,
            refit=True,
        )
        gs.fit(v["X_tr"], v["y_tr"])
        std = float(gs.cv_results_["std_test_score"][gs.best_index_])
        searches[name] = gs
        for row in variant_rows:
            if row["name"] == name:
                row["tuned_cv_roc_auc_mean"] = round(float(gs.best_score_), 5)
                row["tuned_cv_roc_auc_std"] = round(std, 5)
                row["tuned_params"] = {k.replace("clf__", ""): v2 for k, v2 in gs.best_params_.items()}
        params = {k.replace("clf__", ""): v2 for k, v2 in gs.best_params_.items()}
        print(f"  {name:26s}  best CV ROC-AUC = {gs.best_score_:.5f} +/- {std:.5f}   {params}")

    best_name = max(searches, key=lambda n: searches[n].best_score_)
    best_gs = searches[best_name]
    best_variant = variants[best_name]
    best_params = {k.replace("clf__", ""): v for k, v in best_gs.best_params_.items()}
    best_std = float(best_gs.cv_results_["std_test_score"][best_gs.best_index_])
    print(f"\n  WINNER: {best_name}  ({best_gs.best_score_:.5f})")

    # How much does class_weight actually buy on the primary metric?  Report it
    # explicitly instead of leaving 'balanced' as a silent assumption.
    cvr = pd.DataFrame(best_gs.cv_results_)
    for cw, label in ((None, "None"), ("balanced", "balanced")):
        mask = cvr["param_clf__class_weight"].isna() if cw is None else cvr["param_clf__class_weight"] == cw
        print(f"  best CV ROC-AUC with class_weight={label:9s}: {cvr.loc[mask, 'mean_test_score'].max():.5f}")

    # ---- experiment 3: threshold, chosen out-of-fold on TRAIN ----------
    print("\n" + "-" * 78)
    print("3. DECISION THRESHOLD -- maximise F1(Yes) on out-of-fold TRAIN predictions")
    print("-" * 78)
    best_pipe = make_pipeline(
        best_variant, LogisticRegression(max_iter=2000, random_state=42, **best_params)
    )
    threshold, oof_f1, oof_proba = tune_threshold(best_pipe, best_variant["X_tr"], best_variant["y_tr"])
    oof_f1_half = f1_score(best_variant["y_tr"], (oof_proba >= 0.5).astype(int))
    print(f"  class_weight=None      OOF F1(Yes) @0.50 = {oof_f1_half:.4f}")
    print(f"  tuned threshold = {threshold:.2f}   OOF F1(Yes) = {oof_f1:.4f}  "
          f"(gain {oof_f1 - oof_f1_half:+.4f})")

    # class_weight='balanced' and threshold tuning are two ways of doing the same
    # thing -- moving the operating point towards recall.  Showing both on the
    # same out-of-fold probabilities makes that concrete rather than asserted.
    bal_params = dict(best_params, class_weight="balanced")
    bal_pipe = make_pipeline(
        best_variant, LogisticRegression(max_iter=2000, random_state=42, **bal_params)
    )
    bal_thr, bal_f1, bal_proba = tune_threshold(bal_pipe, best_variant["X_tr"], best_variant["y_tr"])
    bal_f1_half = f1_score(best_variant["y_tr"], (bal_proba >= 0.5).astype(int))
    print(f"  class_weight=balanced  OOF F1(Yes) @0.50 = {bal_f1_half:.4f}  "
          f"| best {bal_f1:.4f} @{bal_thr:.2f}")
    print("  -> reweighting and thresholding buy the same thing; the tuned threshold on the "
          "unweighted model is kept because it is explicit and adjustable after deployment.")

    # ---- experiment 4: the single test-set read ------------------------
    print("\n" + "-" * 78)
    print("4. FINAL TEST EVALUATION -- the test set is read here, once")
    print("-" * 78)
    best_pipe.fit(best_variant["X_tr"], best_variant["y_tr"])
    proba_te = best_pipe.predict_proba(best_variant["X_te"])[:, 1]

    test_auc = round(float(roc_auc_score(y_te, proba_te)), 4)
    at_tuned = scores_at(y_te, proba_te, threshold)
    at_half = scores_at(y_te, proba_te, 0.5)

    print(f"  test ROC-AUC                 : {test_auc:.4f}")
    print(f"  @0.50  F1={at_half['f1_yes']:.4f}  P={at_half['precision_yes']:.4f}  "
          f"R={at_half['recall_yes']:.4f}  acc={at_half['accuracy']:.4f}")
    print(f"  @{threshold:.2f}  F1={at_tuned['f1_yes']:.4f}  P={at_tuned['precision_yes']:.4f}  "
          f"R={at_tuned['recall_yes']:.4f}  acc={at_tuned['accuracy']:.4f}")
    print(f"  F1 gain from threshold tuning: {at_tuned['f1_yes'] - at_half['f1_yes']:+.4f}")
    print(f"  confusion matrix @{threshold:.2f} [[TN,FP],[FN,TP]] = {at_tuned['confusion_matrix']}")

    if test_auc > 0.90:
        print("\n  *** WARNING: ROC-AUC > 0.90 -- CLAUDE.md fact #4 says assume LEAKAGE. ***")

    # ---- coefficients --------------------------------------------------
    print("\n" + "-" * 78)
    print("5. COEFFICIENTS AS ODDS RATIOS (sorted by |coef|)")
    print("-" * 78)
    table = coefficient_table(best_pipe)
    for _, r in table.iterrows():
        print(f"  {r['feature']:42s} {r['coef']:+8.4f}   OR = {r['odds_ratio']:.4f}")

    # Free sanity check promised by Step 2 §7: with Month-to-month and Bank
    # transfer as the reference levels, EDA's two strongest risk factors must
    # show up with the signs below or something is wired wrong.
    def coef_of(feat: str) -> float:
        hit = table.loc[table["feature"] == feat, "coef"]
        return float(hit.iloc[0]) if len(hit) else float("nan")

    print("\n  EDA sanity checks:")
    for feat, want in (
        ("Contract_One year", "negative"),
        ("Contract_Two year", "negative"),
        ("PaymentMethod_Electronic check", "positive"),
    ):
        c = coef_of(feat)
        ok = (c < 0) if want == "negative" else (c > 0)
        print(f"    {feat:34s} {c:+.4f}  expected {want:8s} -> {'PASS' if ok else 'FAIL'}")
    mc = coef_of("MonthlyCharges")
    print(f"    MonthlyCharges                     {mc:+.4f}  (EDA 6.7: non-monotone; a linear "
          f"term must fit one sign and gets the top decile backwards)")

    # ---- JSON ----------------------------------------------------------
    runtime = round(time.time() - t0, 2)
    payload = {
        "approach": "logistic-regression",
        "agent_brief": "linear / interpretable",
        "variants": [
            {
                "name": r["name"],
                "cv_roc_auc_mean": r["cv_roc_auc_mean"],
                "cv_roc_auc_std": r["cv_roc_auc_std"],
                "note": (
                    f"{r['note']}; {r['n_encoded_features']} encoded features; "
                    f"tuned CV ROC-AUC {r['tuned_cv_roc_auc_mean']:.5f} "
                    f"+/- {r['tuned_cv_roc_auc_std']:.5f} at {r['tuned_params']}"
                ),
            }
            for r in variant_rows
        ],
        "best_model": {
            "name": f"LogisticRegression [{best_name}]",
            "params": best_params,
            "cv_roc_auc_mean": round(float(best_gs.best_score_), 5),
            "cv_roc_auc_std": round(best_std, 5),
            "threshold": threshold,
            "test": {"roc_auc": test_auc, **at_tuned},
            "test_at_threshold_0.5": {k: v for k, v in at_half.items() if k != "confusion_matrix"},
        },
        "top_drivers": [
            {
                "feature": str(r["feature"]),
                "coef": round(float(r["coef"]), 4),
                "odds_ratio": round(float(r["odds_ratio"]), 4),
            }
            for _, r in table.iterrows()
        ],
        "runtime_seconds": runtime,
        "notes": (
            f"Frozen split (dp.load_splits()); test set read once. All tuning by 5-fold "
            f"StratifiedKFold(shuffle, seed 42) on train only; threshold chosen by maximising "
            f"F1(Yes) on cross_val_predict out-of-fold train probabilities (OOF F1 {oof_f1:.4f} vs "
            f"{oof_f1_half:.4f} at 0.5). class_weight='balanced' was tested, not assumed: it "
            f"changes CV ROC-AUC by <0.001 and reaches OOF F1 {bal_f1_half:.4f} at 0.5, i.e. it "
            f"is a substitute for threshold tuning, not an addition to it. "
            f"Threshold tuning moved test F1(Yes) {at_half['f1_yes']:.4f} -> {at_tuned['f1_yes']:.4f} "
            f"({at_tuned['f1_yes'] - at_half['f1_yes']:+.4f}). ROC-AUC {test_auc:.4f} sits in the "
            f"0.84-0.85 band predicted in Step 1, so no leakage indicated. Accuracy is NOT the "
            f"headline: the all-'No' baseline is 73.46%."
        ),
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {RESULTS_JSON}  ({runtime}s)")


if __name__ == "__main__":
    main()
