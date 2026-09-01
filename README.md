# hackaton-classificacao

**Hackathon 1 — Desafio de Classificação.** Predicting customer churn on the Telco Customer Churn dataset (7043 customers, 21 columns) with a 90-minute budget.

Fork of [`gustavokatsuo/hackaton-classificacao`](https://github.com/gustavokatsuo/hackaton-classificacao).

## The task

Build the best binary classifier for `Churn` (Yes/No) that the team can **defend with an appropriate evaluation metric**. At least two different models, compared. Algorithms limited to Module 3: KNN, Naive Bayes, Logistic Regression, SVM, Trees, Ensemble.

The deliverable is a notebook that runs start-to-finish, plus a synthesis cell covering: best model / chosen metric and value / main technical decision and why / what we'd try next.

Full statement: [`enunciado.ipynb`](enunciado.ipynb) *(read-only — this is the assignment as issued)*

## Start here

| If you want to… | Read |
|---|---|
| Know what not to get wrong | [`CLAUDE.md`](CLAUDE.md) |
| Understand the layout and the pipeline | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Follow our reasoning step by step | [`METHODOLOGY.md`](METHODOLOGY.md) |
| See what changed and when | [`CHANGES.md`](CHANGES.md) |
| Dig into the numbers | [`docs/`](docs/) |

## Progress

| Step | Phase | Status |
|---|---|---|
| 1 | Explore | ✅ [EDA report](docs/step-01-eda.md) |
| 2 | Treat data | ✅ [Treatment report](docs/step-02-data-treatment.md) |
| 3 | Train ≥2 models | ✅ 4 approaches — see [Methodology](METHODOLOGY.md) |
| 4 | Evaluate & compare | ⬜ |
| 5 | Synthesis | ⬜ |

## Headline findings so far

- **Class balance is 73.46% No / 26.54% Yes.** The all-"No" baseline scores **73.46% accuracy** — so accuracy is not a usable headline metric. We use **ROC-AUC** primary, **F1 on the `Yes` class** alongside.
- **The `TotalCharges` trap is solved.** Its 11 non-numeric values are single spaces `' '`; every one of those rows has `tenure == 0` and none of them churned. They are new customers who have never been billed, so the correct fix is **impute 0**, not the median.
- **`Contract` is the strongest predictor** (Cramér's V = 0.410), spanning 2.83% churn on two-year contracts to 42.71% month-to-month. Crossed with `InternetService` the spread is **70×**: month-to-month + fiber churns at **54.61%** (n=2128), two-year + no internet at **0.78%** (n=638).
- **Four modeling approaches land within 0.0077 test ROC-AUC of each other** — GradientBoosting (0.8440), LinearSVC (0.8399), Logistic Regression (0.8424), Naive Bayes (0.8389), KNN (0.8363) — all inside the Step 1 predicted 0.84–0.85 band, all under one CV standard deviation apart. Model family barely matters here once each is properly tuned; see [Methodology](METHODOLOGY.md#step-3--modeling).
- **`TotalCharges` stays in the model.** Tested keep/drop/residual under identical CV; keeping it won, and the theoretical collinearity concern never became a measurable cost for a regularised model.
- **`gender` and `PhoneService` are pure noise** (p = 0.487 and p = 0.339) and get dropped.
- **Realistic ceiling: ROC-AUC 0.84–0.85.** Above 0.90 means something leaked.

## Setup

Python 3.12.3. Dependencies are pinned in [`requirements.txt`](requirements.txt):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

On Ubuntu 24.04 without a venv, PEP 668 blocks the install and you need `pip install --break-system-packages -r requirements.txt` — which is what was done on the hackathon box.

**Match the pinned versions.** This project runs **pandas 3.0.5**, where `TotalCharges` reads as dtype `str`, not `object`. Code that guards on `dtype == 'object'` works under pandas 2.x and silently does nothing under 3.0 — the exact kind of bug only one teammate can reproduce.

The dataset is vendored at the repo root as `Telco-Customer-Churn.csv` — **do not move it into `data/`**, despite what the enunciado's prose says; the notebook's actual code reads it from the root.

## Git remotes

```
origin    → miskmichel/grupo2                       (our fork — push here)
upstream  → gustavokatsuo/hackaton-classificacao    (original — pull assignment fixes)
```
