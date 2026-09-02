# Step 4 — Bernoulli Naive Bayes and Model Comparison

**Date:** 2026-09-02  
**Implementation:** `codigo.ipynb`  
**Status:** Complete  
**Primary metric:** ROC-AUC

---

## 1. Why Bernoulli Naive Bayes

The supplied Naive Bayes module distinguishes three likelihoods:

- `MultinomialNB`: counts or non-negative frequencies, especially text;
- `GaussianNB`: continuous features that are approximately Gaussian within each class;
- `BernoulliNB`: binary presence/absence indicators.

The Telco data mixes continuous, binary, and categorical variables and contains no count vectors. We therefore transform every input into a binary indicator and use `BernoulliNB`:

- `tenure`, `MonthlyCharges`, and `TotalCharges` become low/high quantile bins encoded as 0/1;
- `SeniorCitizen` remains a 0/1 feature;
- categoricals use one-hot encoding after the structural no-service labels have been consolidated.

All preprocessing remains inside the estimator pipeline, so bin boundaries, imputers, and category encodings are learned only from training data.

`TotalCharges` remains in this model. Despite its dependence on tenure and monthly charges, removing it reduced training cross-validation ROC-AUC; the decision was made without consulting the test set. The violated conditional-independence assumption is recorded as a model limitation.

## 2. Laplace smoothing

The module explains `alpha` as Laplace smoothing against zero-frequency likelihoods. The notebook tests the declared grid `[0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10]` using five-fold stratified cross-validation on the training set, with ROC-AUC scoring.

| Selection result | Value |
|---|---:|
| Best `alpha` | 0.5 |
| Mean training CV ROC-AUC | 0.834 |
| Test rows used during selection | 0 |

The model learns the empirical class prior from the training fold. No hyperparameter or variant is selected from held-out test performance.

## 3. BernoulliNB held-out results

All metrics below use the same 1,409 test customers and threshold 0.50 as Logistic Regression.

| Metric | BernoulliNB |
|---|---:|
| Accuracy | 0.769 |
| Precision — `Churn = 1` | 0.554 |
| Recall — `Churn = 1` | 0.660 |
| F1 — `Churn = 1` | 0.602 |
| ROC-AUC | **0.822** |

Confusion matrix (rows = real, columns = predicted):

| | Predicted No | Predicted Yes |
|---|---:|---:|
| **Real No** | TN = 836 | FP = 199 |
| **Real Yes** | FN = 127 | TP = 247 |

The model identifies 247 of 374 actual churners and misses 127.

## 4. Final comparison

| Model | Accuracy | Precision Yes | Recall Yes | F1 Yes | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Always `No` baseline | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| **Balanced Logistic Regression** | 0.743 | 0.511 | **0.778** | **0.617** | **0.839** |
| Bernoulli Naive Bayes | **0.769** | **0.554** | 0.660 | 0.602 | 0.822 |

Logistic Regression is selected because ROC-AUC was chosen before modeling as the primary metric and it also has the higher F1 and recall on churn. BernoulliNB's higher accuracy does not overturn that decision: a classifier that always predicts `No` already achieves 0.735 accuracy while detecting no churners.

## 5. Probability caveat

Naive Bayes multiplies feature likelihoods under conditional independence. That assumption is visibly imperfect here: services, contract, tenure, and charges are related. The resulting probabilities can be too close to 0 or 1 and should be used for ranking in ROC-AUC, not presented as calibrated absolute churn risks without a calibration curve or Brier-score analysis.

## 6. Final synthesis

- **Best model:** balanced Logistic Regression.
- **Chosen metric/value:** ROC-AUC 0.839; F1(Yes) 0.617 and recall(Yes) 0.778 at threshold 0.50.
- **Main technical decision:** all learned preprocessing stays inside train-only pipelines; BernoulliNB receives binary inputs and selects `alpha` using only training CV.
- **With more time:** test a tree/ensemble for known nonlinearities and interactions, choose the operational threshold from out-of-fold predictions, and evaluate probability calibration.
