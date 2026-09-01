# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A 90-minute hackathon exercise: **binary classification of customer churn** on the Telco Customer Churn dataset (7043 rows × 21 columns). Team of three. The assignment lives in `enunciado.ipynb` (Portuguese).

The deliverable is **a notebook that runs start-to-finish** plus a synthesis cell answering four questions: best model / chosen metric and value / main technical decision and why / what you would try next.

## Hard-won facts — read before touching the data

These are established, verified findings from `docs/step-01-eda.md`. Do not re-derive them; do not contradict them without new evidence.

1. **`TotalCharges` contains 11 single-space `' '` values, not empty strings.** This is why the column reads as text. `.isnull()` does not catch them. `.astype(float)` raises `ValueError: could not convert string to float: ' '`. All 11 rows have `tenure == 0` (the only such rows in the dataset) and all are `Churn == 'No'`. **Impute `0`, not the median** — they are new customers who have never been billed.

   ⚠️ **Do not guard on `dtype == 'object'`.** Under the installed **pandas 3.0.5** this column reads as dtype **`str`**, not `object`, so an `object` check silently never fires. Dispatch on `not pandas.api.types.is_numeric_dtype(...)` instead, as `src/data_prep.py` does. Most published tutorials for this dataset were written against pandas 1.x/2.x and will tell you `object`.

2. **The CSV is at the repo root, not in `data/`.** The enunciado's *prose* says `data/`, but its *code* cell reads `pd.read_csv('Telco-Customer-Churn.csv')`. The code is right. **Do not move the file** — it breaks the notebook.

3. **The all-"No" baseline is 73.46% accuracy.** Class balance is 73.46% No / 26.54% Yes (2.77:1). Never report accuracy as a headline metric. Use ROC-AUC primary, F1 on the `Yes` class alongside.

4. **Expected ceiling is ROC-AUC 0.84–0.85, F1(Yes) 0.60–0.63.** If a model scores AUC > 0.90, assume leakage and go find it.

5. **Six columns encode `'No internet service'` in the exact same 1526 rows**, perfectly collinear with `InternetService == 'No'`. Same for `'No phone service'` (682 rows) vs `PhoneService`. Collapse these to `'No'` before one-hot encoding or you create 7 redundant dummies.

6. **`TotalCharges` is 99.91% explained by `tenure × MonthlyCharges`** (R² = 0.9991). Drop it for linear models.

7. **`gender` (p = 0.487) and `PhoneService` (p = 0.339) are statistically pure noise.** Drop them.

8. **`MonthlyCharges` is non-monotone** against churn — it rises to 40.91% at decile 9, then falls to 24.68% at decile 10. This is a composition effect: decile 10 is 40% two-year contracts. *Any* model that includes `Contract` and `InternetService` reproduces it, logistic regression included (verified in `docs/step-03b-tree-ensembles.md` §7); only a model fitted on `MonthlyCharges` alone would get it backwards. Note also that `MonthlyCharges`' partial effect flips sign — raw correlation +0.193, but a negative logistic coefficient once `InternetService` and `tenure` are in the model.

   ⚠️ **Correction (Step 3b, tree ensembles).** This fact previously said "plain logistic regression gets the top decile backwards, prefer trees" — that inference was not borne out. Trees do **not** beat a tuned logistic regression on this dataset (GradientBoosting CV ROC-AUC 0.8486 vs LogReg's, a gap under a fifth of one CV standard deviation). Also: **Cramér's V is a poor importance ranking for tree models here.** Permutation importance puts `tenure` (0.0667), `InternetService` (0.0550) and `Contract` (0.0294) far ahead of everything else — 82% of total importance in three features — while `OnlineSecurity` (V=0.347, EDA's #2) and `TechSupport` (V=0.343, #3) score 22–30× lower, because their marginal signal is mostly the `InternetService` effect flowing through the "No internet service"→"No" collapse (fact 5). Don't rank tree feature importance by Cramér's V; use permutation importance instead.

   ⚠️ **Second correction (Step 3d, SVM).** This fact's original second half — "add the `Contract × InternetService` interaction explicitly" — is also wrong, and not just unhelpful: it actively **hurts**. A 70× rate spread is not evidence of an interaction; an additive-in-log-odds model already produces multiplicative spread by construction. An additive logistic regression on `Contract` + `InternetService` with **no interaction term** reproduces all nine cells to within 1.69 pp and generates 48.4× of the 55.8×-in-that-fold spread on its own. Adding the explicit interaction term measurably cost CV ROC-AUC: −0.0033 (SVC-linear), −0.0051 (SVC-RBF) — the single worst feature-engineering result Step 3d measured. Do not add this interaction term.

## Environment

Python 3.12.3 · numpy 2.5.1 · scipy 1.18.0 · **pandas 3.0.5** · **scikit-learn 1.9.0** · **matplotlib 3.11.1** · **seaborn 0.13.2**

The four ML packages were installed during Step 2 and are pinned in `requirements.txt`. Ubuntu 24.04 marks its Python as externally managed (PEP 668), so a plain `pip install` fails and `--break-system-packages` was required:

```bash
pip install --break-system-packages -r requirements.txt   # what was done here
python3 -m venv .venv && pip install -r requirements.txt  # preferred elsewhere
```

Note pandas **3.0** — not 1.x/2.x. This changes text-column dtypes (see fact 1) and some tutorial code for this dataset will not apply verbatim.

Do not install packages without saying so in the response and logging it in `CHANGES.md`.

## Using the cleaning code

```python
import sys; sys.path.insert(0, 'src')
import data_prep as dp

X_tr, X_te, y_tr, y_te = dp.load_splits()   # the FROZEN split — use this
pre = dp.build_preprocessor(df=dp.clean(dp.load_raw()))   # UNFITTED
```

**Use `load_splits()`, not `split()`.** The split is frozen to `data/splits/` and
committed, so all three team members score on identical held-out rows. Calling
`split()` again re-derives it and silently invalidates the comparison table.
`test.csv` stays unread until final evaluation. Verify with
`python3 src/make_splits.py --check`.

`build_preprocessor` returns an unfitted transformer on purpose — fit it inside a `Pipeline` on the training fold only. Flags: `clean(drop_noise=False)` keeps `gender`/`PhoneService` for demonstrating the noise cull; `clean(drop_total_charges=True)` for linear models.

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
