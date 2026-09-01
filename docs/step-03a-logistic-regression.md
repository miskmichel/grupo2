# Step 3a — Logistic Regression (linear / interpretable)

**Date:** 2026-09-01
**Status:** Complete
**Artifacts produced:** `src/models/logistic_regression.py`, `results/logistic_regression.json`
**Split:** the frozen `data/splits/` split via `dp.load_splits()` — 5634 train / 1409 test, `test.csv` read exactly once
**Runtime:** 420.75 s wall clock (5-fold CV over four variants, each grid-searched, plus threshold tuning)
**Method note:** this report is reconstructed from `results/logistic_regression.json` rather than a live agent transcript — the agent that produced it exited (a background-harness restart) before writing its report. Every number below is quoted directly from that JSON; nothing here is estimated.

---

## 1. What this model owns

This is the team's interpretable baseline: a linear model whose coefficients read directly as churn drivers, and the model most exposed to the collinearity Step 1 and Step 2 flagged. Its brief included the open `TotalCharges` question — keep, drop, or replace with the residual — since collinearity hurts linear models hardest.

## 2. The four variants, cross-validated

5-fold `StratifiedKFold` on **train only**, grid-searched over `C`, `penalty`, `class_weight`, scored by `roc_auc`. Test was not touched until the final numbers below.

| Variant | CV ROC-AUC | Encoded features | Winning grid point | Note |
|---|---|---|---|---|
| **a — keep `TotalCharges`** (baseline) | **0.8463 ± 0.0085** | 21 | `C=10, penalty=l2, solver=lbfgs, class_weight=None` | the frozen split as written |
| b — drop `TotalCharges` | 0.8447 ± 0.0085 | 20 | `C=100, penalty=l2, solver=liblinear` | tests fact 6 (`TotalCharges` 99.91% explained by `tenure×MonthlyCharges`) |
| c — `TotalCharges` → residual | 0.8447 ± 0.0083 | 21 | `C=100, penalty=l2, solver=liblinear` | `TotalCharges − tenure×MonthlyCharges` |
| d — keep noise columns | 0.8458 ± 0.0083 | 23 | `C=100, penalty=l2, solver=lbfgs` | `gender` + `PhoneService` back in |

**The open `TotalCharges` question, answered:** keeping it wins, by **+0.0016 CV ROC-AUC** over dropping it and **+0.0016** over the residual — both differences are under a fifth of one CV standard deviation, i.e. **not distinguishable from noise at this sample size**. Regularised logistic regression evidently absorbs the collinearity without a measurable cost; the theoretical concern (unstable coefficients from fact 6) does not translate into a CV loss here. **Verdict: keep `TotalCharges`.** The flag (`drop_total_charges=True`) stays available if a future variant needs it, but the default is correct.

**The noise-cull question, answered:** dropping `gender`/`PhoneService` costs **0.0005 CV ROC-AUC** — again inside noise. Step 2's prediction ("almost certainly neutral... a regularised model would shrink them to nothing anyway") is confirmed rather than merely asserted. **Verdict: the cull is free.** Fact 7 stands.

## 3. Final model and test performance

**Best: `LogisticRegression(C=10, penalty='l2', solver='lbfgs', class_weight=None)`** on variant (a), the 21-feature baseline.

| | CV (train) | Test |
|---|---|---|
| ROC-AUC | 0.8463 ± 0.0085 | **0.8424** |
| Threshold | — | 0.32 (F1-maximising on train OOF predictions, OOF F1 0.6362) |

CV and test ROC-AUC agree to within 0.004 — no overfitting to the training folds, and **0.8424 sits inside the Step 1 predicted band (0.84–0.85, fact 4)**. No leakage indicated.

### Threshold tuning — the gain, quantified

| Threshold | F1(Yes) | Precision(Yes) | Recall(Yes) | Accuracy | Confusion matrix `[[TN,FP],[FN,TP]]` |
|---|---|---|---|---|---|
| 0.50 (default) | 0.6081 | 0.6594 | 0.5642 | 0.8070 | — |
| **0.32 (tuned)** | **0.6178** | 0.5341 | **0.7326** | 0.7594 | `[[796,239],[100,274]]` |

Tuning the threshold buys **+0.0097 F1(Yes)**, traded almost entirely for recall: **+16.84 pp recall for −12.53 pp precision.** For a churn model this is the right trade — a missed churner (false negative) costs a lost customer, a false positive costs a retention offer to someone who wasn't leaving. At the tuned threshold the model catches **274 of 374 churners (73.3%)** in the test fold, against 211 at the default threshold.

## 4. Coefficients — the driver list

Reported as odds ratios (`exp(coef)`), sorted by `|coef|`. Reference levels: `InternetService`=DSL, `Contract`=Month-to-month, `MultipleLines`/`Streaming*`/`OnlineSecurity`/`TechSupport`/`DeviceProtection`/`OnlineBackup`=No, `PaymentMethod`=Bank transfer (automatic), `PaperlessBilling`/`Partner`/`Dependents`=No.

| Feature | Coefficient | Odds ratio | Reads as |
|---|---|---|---|
| `InternetService_No` | −1.5718 | 0.208 | no internet service cuts churn odds to ~1/5 |
| `InternetService_Fiber optic` | +1.5542 | 4.731 | fiber nearly **5×** the odds of a DSL customer |
| `Contract_Two year` | −1.3642 | 0.256 | two-year contract cuts odds to ~1/4 vs month-to-month |
| `tenure` | −1.3080 | 0.270 | (per std-dev; scaled) longer tenure sharply protective |
| `MonthlyCharges` | −0.9171 | 0.400 | negative once other features are controlled — see §5 |
| `Contract_One year` | −0.6948 | 0.499 | one-year contract roughly halves the odds |
| `TotalCharges` | +0.5882 | 1.801 | positive net of `tenure`/`MonthlyCharges` — the residual signal |
| `StreamingTV_Yes` | +0.5179 | 1.679 | |
| `StreamingMovies_Yes` | +0.5168 | 1.677 | |
| `MultipleLines_Yes` | +0.4400 | 1.553 | |
| `PaymentMethod_Electronic check` | +0.3842 | 1.468 | |
| `PaperlessBilling_Yes` | +0.3693 | 1.447 | |
| `OnlineSecurity_Yes` | −0.2800 | 0.756 | |
| `TechSupport_Yes` | −0.2273 | 0.797 | |
| `Dependents_Yes` | −0.2224 | 0.801 | |
| `SeniorCitizen` | +0.1456 | 1.157 | |
| `DeviceProtection_Yes` | +0.1053 | 1.111 | |
| `PaymentMethod_Mailed check` | +0.0745 | 1.077 | |
| `OnlineBackup_Yes` | −0.0326 | 0.968 | |
| `PaymentMethod_Credit card (automatic)` | −0.0297 | 0.971 | |
| `Partner_Yes` | +0.0232 | 1.023 | |

**Sanity check against EDA, as planned in Step 2 §7:** `Contract_One year` and `Contract_Two year` both came out strongly negative (reference = Month-to-month, EDA's highest-churn level, 42.71%) — **confirmed**. `PaymentMethod_Electronic check` came out positive (EDA's highest-churn payment method, 45.29%) — **confirmed**. Both free sanity checks pass.

## 5. The `MonthlyCharges` sign flip — expected, and now explained

EDA §6.7 found `MonthlyCharges` non-monotone against churn and *raw* point-biserial correlation **+0.193** (higher charges, more churn, on average). Here the fitted coefficient is **−0.9171** — negative. This is not a bug; it is **Simpson's-paradox-style suppression**, and Step 3b's tree-ensemble report independently confirms the same mechanism: `MonthlyCharges` is a near-proxy for `InternetService` (fiber customers pay far more and churn far more), so once `InternetService` is in the model as its own feature, the *residual* effect of `MonthlyCharges` — charges conditional on service type — points the other way: within a service tier, a higher bill (more add-ons, longer relationship) is mildly protective. This is exactly CLAUDE.md fact 8's corrected note: *"MonthlyCharges' partial effect flips sign... once InternetService and tenure are in the model."* This model is the source of that verified claim.

## 6. Limitations

1. **One split, one seed.** CV standard deviations are ±0.008–0.009; the difference between all four variants in §2 is inside that noise. A repeated-CV or multi-seed comparison would be the honest next step, as Step 3b also notes.
2. **`SeniorCitizen`'s odds ratio (1.157) is on the model's internal standardized numeric block along with `tenure`**, not a raw 0-vs-1 comparison — `build_preprocessor` passes it through unscaled per fact/design, but interpretation should still be read as "the fitted coefficient," not literally "seniors have 15.7% higher raw odds" without checking the preprocessing step directly.
3. **The driver list is descriptive of this fitted model, not causal.** As METHODOLOGY.md's Step 1 caveat notes, `tenure` and `TotalCharges` are correlated with the outcome partly by construction (a churned-early customer has low values in both) — the coefficients describe association within this snapshot, not a scoring-at-signup mechanism.
4. **No leakage indicated** — CV and test ROC-AUC track within 0.004, and 0.8424 lands inside the predicted 0.84–0.85 band — but this is a single held-out fold, not a nested-CV estimate of generalisation.

## 7. Reproducing

```bash
python3 src/models/logistic_regression.py
```
Writes `results/logistic_regression.json`. Deterministic: `random_state`/`StratifiedKFold(shuffle=True, random_state=42)` fixed throughout; the frozen split via `dp.load_splits()` is never re-derived.
