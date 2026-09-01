# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A 90-minute hackathon exercise: **binary classification of customer churn** on the Telco Customer Churn dataset (7043 rows × 21 columns). Team of three. The assignment lives in `enunciado.ipynb` (Portuguese).

The deliverable is **a notebook that runs start-to-finish** plus a synthesis cell answering four questions: best model / chosen metric and value / main technical decision and why / what you would try next.

## Hard-won facts — read before touching the data

These are established, verified findings from `docs/step-01-eda.md`. Do not re-derive them; do not contradict them without new evidence.

1. **`TotalCharges` contains 11 single-space `' '` values, not empty strings.** This is why the column reads as `object`. `.isnull()` does not catch them. `.astype(float)` raises `ValueError`. All 11 rows have `tenure == 0` (the only such rows in the dataset) and all are `Churn == 'No'`. **Impute `0`, not the median** — they are new customers who have never been billed.

2. **The CSV is at the repo root, not in `data/`.** The enunciado's *prose* says `data/`, but its *code* cell reads `pd.read_csv('Telco-Customer-Churn.csv')`. The code is right. **Do not move the file** — it breaks the notebook.

3. **The all-"No" baseline is 73.46% accuracy.** Class balance is 73.46% No / 26.54% Yes (2.77:1). Never report accuracy as a headline metric. Use ROC-AUC primary, F1 on the `Yes` class alongside.

4. **Expected ceiling is ROC-AUC 0.84–0.85, F1(Yes) 0.60–0.63.** If a model scores AUC > 0.90, assume leakage and go find it.

5. **Six columns encode `'No internet service'` in the exact same 1526 rows**, perfectly collinear with `InternetService == 'No'`. Same for `'No phone service'` (682 rows) vs `PhoneService`. Collapse these to `'No'` before one-hot encoding or you create 7 redundant dummies.

6. **`TotalCharges` is 99.91% explained by `tenure × MonthlyCharges`** (R² = 0.9991). Drop it for linear models.

7. **`gender` (p = 0.487) and `PhoneService` (p = 0.339) are statistically pure noise.** Drop them.

8. **`MonthlyCharges` is non-monotone** against churn — it rises to 40.91% at decile 9, then falls to 24.68% at decile 10. Plain logistic regression gets the top decile backwards. Prefer trees, or add the `Contract × InternetService` interaction explicitly.

## Environment

Python 3.12.3 with numpy and scipy. **pandas, scikit-learn, matplotlib and seaborn are not installed** — the notebook cannot run past its setup cell until they are. Install with:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

Do not install packages without saying so in the response and logging it in `CHANGES.md`.

## Conventions

- **Documentation lives in `docs/`**, one file per step: `docs/step-NN-<slug>.md`. Every step gets an entry in `METHODOLOGY.md` and a line in `CHANGES.md`.
- **Every quantitative claim carries its number.** "Churn is higher for fiber" is not acceptable; "fiber churns at 41.89% (n=3096) vs a 26.54% base rate" is.
- **Report segment sizes alongside rates**, so small-sample noise is visible.
- **Never fit a transformer before the train/test split.** Use `Pipeline` + `ColumnTransformer`. A leaking pipeline is the classic deduction on this exercise.
- **Stratify splits on `Churn`.**
- Do not edit `enunciado.ipynb` — it is the assignment as given. Work in a separate notebook.
- Do not commit or push unless asked.

## Repo map

See `ARCHITECTURE.md`. Short version: `enunciado.ipynb` (assignment, read-only), `Telco-Customer-Churn.csv` (data, root path — do not move), `docs/` (per-step findings), and the four top-level docs.

## Git

`origin` is the fork `miskmichel/grupo2`; `upstream` is the original `gustavokatsuo/hackaton-classificacao`. Push goes to the fork. Pull assignment updates with `git pull upstream main`.
