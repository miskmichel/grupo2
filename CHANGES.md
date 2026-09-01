# Changelog

Chronological ledger of what changed. Terse by design — reasoning belongs in `METHODOLOGY.md`, evidence in `docs/step-NN-*.md`.

Newest first.

---

## 2026-09-01

### Documentation structure established

- Added `CLAUDE.md` — working guidance plus the eight verified facts about the dataset that must not be re-derived or contradicted.
- Added `ARCHITECTURE.md` — repository map, column groups, the four structural dependencies in the data, and the intended modeling pipeline.
- Added `METHODOLOGY.md` — step-by-step narrative of decisions and rejected alternatives.
- Added `CHANGES.md` — this file.
- Added `docs/` for per-step evidence reports.
- Rewrote `README.md` from a one-line stub into a project index.

### Step 1 — Exploratory data analysis ✅

- Added `docs/step-01-eda.md`.
- Profiled all 7043 rows × 21 columns using stdlib `csv` + numpy/scipy (pandas unavailable at the time). **No packages were installed during this step.**
- **Solved the planted `TotalCharges` trap:** the 11 non-numeric values are single spaces `' '`; all 11 rows have `tenure == 0` (the only such rows in the dataset) and all are `Churn == 'No'`. Decision: impute `0`, not the median.
- Established the class balance — 73.46% No / 26.54% Yes — and therefore the **73.46% all-"No" accuracy baseline** that any accuracy claim must beat.
- Chose **ROC-AUC** as primary metric, F1/recall on `Yes` reported alongside.
- Verified `'No internet service'` occupies the same 1526 rows in all six add-on columns (and `'No phone service'` the same 682 rows as `PhoneService == 'No'`) — 7 collinear dummies to be avoided at encoding time.
- Measured `TotalCharges ≈ tenure × MonthlyCharges` at R² = 0.9991.
- Identified `gender` (p = 0.487) and `PhoneService` (p = 0.339) as statistically pure noise.
- Found `MonthlyCharges` to be **non-monotone** against churn (40.91% at decile 9 → 24.68% at decile 10) — the basis for including a tree-based model.
- Set the expected performance ceiling at **ROC-AUC 0.84–0.85, F1(Yes) 0.60–0.63**; anything above AUC 0.90 to be treated as suspected leakage.

### Errata recorded

- The enunciado's prose refers to `data/Telco-Customer-Churn.csv`, but its code cell reads `pd.read_csv('Telco-Customer-Churn.csv')`. The CSV stays at the repo root; **the file must not be moved** or the assignment notebook breaks. Documented in `CLAUDE.md` and `ARCHITECTURE.md`.

### Environment findings

- Python 3.12.3, numpy 2.5.1, scipy 1.18.0 present.
- **pandas, scikit-learn, matplotlib and seaborn are absent** — `enunciado.ipynb` cannot execute past its setup cell (cell 7) until they are installed. Flagged as a Step 2 prerequisite.

### Repository setup

- Installed GitHub CLI (`gh` 2.45.0, Ubuntu package) — a system-level change outside the repo.
- Forked `gustavokatsuo/hackaton-classificacao` → **`miskmichel/grupo2`**.
- Repointed remotes: `origin` → the fork, `upstream` → the original. `main` tracks `origin/main`.

---

### Step 2 — Data treatment ✅

- Added `src/data_prep.py` — `load_raw`, `clean`, `infer_feature_groups`, `build_preprocessor`, `split`. **Nothing is fitted at import time**; `build_preprocessor()` returns an unfitted `ColumnTransformer` so the encoder cannot be fitted before the split by accident.
- Added `docs/step-02-data-treatment.md`.
- **Installed pandas 3.0.5, scikit-learn 1.9.0, matplotlib 3.11.1, seaborn 0.13.2** (plus transitive deps: joblib 1.6.0, threadpoolctl 3.6.0, contourpy 1.3.3, cycler 0.12.1, fonttools 4.64.0, kiwisolver 1.5.1, narwhals 2.25.0, cloudpickle 3.1.2). Plain `pip install` failed under PEP 668 (`externally-managed-environment`); **`--break-system-packages` was required**, overriding a distro safety rail. numpy 2.5.1 / scipy 1.18.0 unchanged.
- `clean()` result: **7043 → 7043 rows** (nothing silently dropped), 21 → 18 columns. Does not mutate its input.
- `TotalCharges` now float64 with 0 nulls; the 11 previously-blank rows are **exactly 0.0** and are the only zeros in the column. Column mean shifts 2283.3004 → 2279.7343.
- Zero columns still contain `'No internet service'` or `'No phone service'`. Collapse arithmetic reconciles (`OnlineSecurity` 3498+1526=5024 No; `MultipleLines` 3390+682=4072 No); `InternetService` untouched at 3096/2421/1526.
- Encoded matrix **(5634, 21) train / (1409, 21) test** — full rank 21/21, no duplicate columns, no |r| ≥ 0.999999 pair, max off-diagonal |r| = 0.8297.
- Stratified split holds the churn rate to within **0.007 pp** (train 26.5353%, test 26.5436%); no index overlap; deterministic at `random_state=42`.
- All 25 of the agent's assertions passed, and the results were **independently re-verified by the lead** before commit.

#### Correction to a Step 1 claim

- **`TotalCharges` reads as dtype `str`, not `object`** — pandas 3.0 changed text-column dtypes. `CLAUDE.md` fact 1 previously said `object`. Every consequence still holds (`.isnull()` catches nothing, `.astype(float)` raises), but a guard written as `dtype == 'object'` would silently never fire. `CLAUDE.md` corrected; `data_prep.clean()` dispatches on `not is_numeric_dtype()`.

#### Revision to a Step 1 estimate

- **Encoded width is 21, not the 25–30 predicted.** Under `drop='first'`, the "No X service" collapse turns seven 3-level columns into 2-level ones (7 dummies saved), and the noise cull removes 2 more. Measured counterfactual: naive encoding gives **30 columns at rank 24** with **22 perfect-|r| pairs**, including `PhoneService_Yes` ↔ `MultipleLines_No phone service` at **r = −1.0000**.

### Housekeeping

- Added `.gitignore` for `__pycache__/`.
- Set repo-local git identity (`miskmichel` / `michelcarneiro205@gmail.com`) — none was configured, so commits were impossible.

---

## In progress

### Step 3 — Modeling ⬜

Not started. `TotalCharges` keep-vs-drop-vs-residual remains open and is now a one-flag decision (`clean(drop_total_charges=True)`).
