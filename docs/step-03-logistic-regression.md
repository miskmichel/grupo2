# Step 3 — Logistic Regression and Classification Metrics

**Date:** 2026-09-02  
**Implementation:** `codigo.ipynb`  
**Model status:** Model 1 complete; Model 2 and the comparison are documented in `step-04-model-comparison.md`  
**Positive class:** `Churn = 1` (`Yes`)

---

## 1. Course material applied

Two supplied notebooks were used as the implementation guide:

- **Logistic Regression — Module 3:** sigmoid probabilities, the default 0.50 decision threshold, regularized `LogisticRegression`, threshold trade-offs, coefficients, and odds ratios.
- **Classification Metrics — Module 2:** confusion matrix, accuracy, precision, recall, F1, and ROC-AUC.

The metrics notebook demonstrates a multiclass Wine problem using weighted averages. Churn is binary and the minority `Yes` class is the business target, so precision, recall, and F1 are calculated directly for `pos_label=1`. Reporting only a weighted average could hide poor churn detection.

## 2. Model pipeline

The logistic model uses the same fixed, stratified 80/20 split created during data treatment. Its pipeline contains:

1. median imputation and `StandardScaler` for numeric features;
2. most-frequent imputation and `OneHotEncoder(drop='first', handle_unknown='ignore')` for categoricals;
3. `LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)`.

All learned transformations are fitted inside the pipeline using only the 5,634 training rows. The 1,409 test rows are transformed only at final evaluation.

`TotalCharges` is removed only from this model's input because it is 99.91% explained by `tenure × MonthlyCharges` and would make linear coefficients less stable. It remains present in the cleaned base and can be tested by tree-based models.

The balanced class weights are intentional: 73.46% of customers are `No`, while only 26.54% are churners. This choice favors detection of the positive class and must be judged using recall/F1 rather than accuracy alone.

## 3. Held-out results at threshold 0.50

| Metric | Result |
|---|---:|
| Accuracy | 0.743 |
| Precision — `Churn = 1` | 0.511 |
| Recall — `Churn = 1` | 0.778 |
| F1 — `Churn = 1` | 0.617 |
| ROC-AUC | **0.839** |

Confusion matrix (rows = real, columns = predicted):

| | Predicted No | Predicted Yes |
|---|---:|---:|
| **Real No** | TN = 756 | FP = 279 |
| **Real Yes** | FN = 83 | TP = 291 |

The model finds 291 of 374 actual churners and misses 83. Its 74.3% accuracy is only about 0.8 percentage points above the 73.46% all-`No` baseline; by contrast, the constant baseline has zero recall and zero F1 for churn. This is why **ROC-AUC is the primary comparison metric** and F1/recall are reported alongside it.

The observed ROC-AUC and F1 fall inside the EDA sanity range (AUC 0.84–0.85, F1 roughly 0.60–0.63 after rounding and modeling choices), providing no sign of target leakage.

## 4. Decision threshold

The course module illustrates thresholds 0.30, 0.50, and 0.70. The project reproduces that table to show the precision–recall trade-off:

- lowering the threshold labels more customers as churn, typically increasing recall and false positives;
- raising it labels fewer as churn, typically increasing precision while missing more real churners;
- ROC-AUC is unchanged because it evaluates probability ranking rather than one chosen threshold.

These thresholds are predeclared for demonstration. The notebook deliberately does **not** select the best threshold by looking at the test set. An operational threshold must be selected from validation or out-of-fold training predictions and then evaluated once on the held-out test set.

## 5. Coefficient interpretation

The notebook joins `model.coef_` to the encoded feature names and reports `exp(coef)` as the odds ratio.

- A positive coefficient is associated with higher estimated churn odds.
- A negative coefficient is associated with lower estimated churn odds.
- For scaled numeric features, the odds ratio corresponds to an increase of one training-set standard deviation.
- For one-hot features, it compares the category with the encoder's dropped reference category.

These are conditional associations learned by a regularized predictive model, not causal effects. Class weighting also means the predicted probabilities should not be described as calibrated absolute churn risks without a separate calibration check.

## 6. Comparison status

Bernoulli Naive Bayes was subsequently implemented as the second distinct model and evaluated on the identical test split. Logistic Regression remains the winner by ROC-AUC, F1, and recall. See [`step-04-model-comparison.md`](step-04-model-comparison.md).
