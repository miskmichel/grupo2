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

## In progress

### Step 2 — Data treatment 🔄

Missing values, categorical encoding, and the `customerID` decision. Will add `src/data_prep.py` and `docs/step-02-data-treatment.md`. Requires installing pandas, scikit-learn, matplotlib and seaborn — that install will be recorded here once complete.
