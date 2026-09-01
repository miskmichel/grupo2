"""Materialise the train/test split to disk, once, for Step 3 (Step 2 deliverable).

Why freeze the split to files rather than call ``data_prep.split()`` in each
notebook?

1. **Three people, one holdout.**  The team trains different models in parallel.
   If each notebook re-derives its own split, the models are scored on different
   test rows and the comparison table is meaningless.  A frozen split makes the
   comparison valid by construction rather than by everyone remembering to pass
   ``random_state=42``.
2. **The test set stays untouched.**  Once written, ``test.csv`` is not read
   until final evaluation.  Re-deriving a split inside a tuning loop is how a
   holdout quietly becomes a validation set.
3. **Drift is detectable.**  The manifest records a SHA-256 of each file.  If a
   split is regenerated with different parameters, the hash changes and the
   mismatch is visible instead of silent.

The files carry a ``row_id`` column -- the index of the row in the raw CSV --
so a prediction can be joined back to its ``customerID`` without ever putting
that identifier in the feature matrix.  See :func:`data_prep.load_splits`.

Usage::

    python3 src/make_splits.py            # writes data/splits/
    python3 src/make_splits.py --check    # verify on-disk files match, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_prep as dp  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = REPO_ROOT / "data" / "splits"
TRAIN_CSV = SPLIT_DIR / "train.csv"
TEST_CSV = SPLIT_DIR / "test.csv"
MANIFEST = SPLIT_DIR / "manifest.json"

TEST_SIZE = 0.2
RANDOM_STATE = 42
ROW_ID = "row_id"


def sha256(path: Path) -> str:
    """Hash a file's bytes, so a regenerated split that differs is loud, not silent."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Clean the raw CSV and split it, returning (train, test, customer_ids).

    ``customer_ids`` is indexed by ``row_id`` and is *not* part of either frame.
    It is written alongside them so predictions can be re-attached to a real
    customer, which is what the retention team would actually action.
    """
    raw = dp.load_raw(str(REPO_ROOT / dp.DEFAULT_CSV))
    customer_ids = raw[dp.ID_COLUMN].copy()
    customer_ids.index.name = ROW_ID

    cleaned = dp.clean(raw)
    X_tr, X_te, y_tr, y_te = dp.split(
        cleaned, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    train = X_tr.copy()
    train[dp.TARGET] = y_tr
    test = X_te.copy()
    test[dp.TARGET] = y_te

    # Preserve the raw-CSV row index rather than renumbering, so row_id stays a
    # stable join key back to customerID.
    train.index.name = ROW_ID
    test.index.name = ROW_ID
    return train.sort_index(), test.sort_index(), customer_ids


def write(train: pd.DataFrame, test: pd.DataFrame, customer_ids: pd.Series) -> dict:
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_CSV, index=True)
    test.to_csv(TEST_CSV, index=True)
    customer_ids.to_frame(dp.ID_COLUMN).to_csv(SPLIT_DIR / "customer_ids.csv", index=True)

    manifest = {
        "generated_by": "src/make_splits.py",
        "source_csv": dp.DEFAULT_CSV,
        "source_sha256": sha256(REPO_ROOT / dp.DEFAULT_CSV),
        "params": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": dp.TARGET,
            "clean_drop_noise": True,
            "clean_drop_total_charges": False,
        },
        "train": {
            "rows": int(len(train)),
            "churn_positives": int(train[dp.TARGET].sum()),
            "churn_rate": round(float(train[dp.TARGET].mean()), 6),
            "sha256": sha256(TRAIN_CSV),
        },
        "test": {
            "rows": int(len(test)),
            "churn_positives": int(test[dp.TARGET].sum()),
            "churn_rate": round(float(test[dp.TARGET].mean()), 6),
            "sha256": sha256(TEST_CSV),
        },
        "columns": list(train.columns),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def check() -> int:
    """Regenerate in memory and compare against what is on disk. 0 = match."""
    if not MANIFEST.exists():
        print("FAIL  no manifest -- run: python3 src/make_splits.py")
        return 1

    recorded = json.loads(MANIFEST.read_text())
    ok = True

    for name, path in (("train", TRAIN_CSV), ("test", TEST_CSV)):
        if not path.exists():
            print(f"FAIL  {path} missing")
            ok = False
            continue
        actual = sha256(path)
        if actual != recorded[name]["sha256"]:
            print(f"FAIL  {name}.csv hash {actual[:12]} != manifest {recorded[name]['sha256'][:12]}")
            ok = False
        else:
            print(f"OK    {name}.csv matches manifest ({actual[:12]}…)")

    src_hash = sha256(REPO_ROOT / dp.DEFAULT_CSV)
    if src_hash != recorded["source_sha256"]:
        print("FAIL  source CSV changed since the split was generated -- regenerate")
        ok = False
    else:
        print(f"OK    source CSV unchanged ({src_hash[:12]}…)")

    # Determinism: rebuilding must reproduce the same row assignment.
    train, test, _ = build_frames()
    disk_train = pd.read_csv(TRAIN_CSV, index_col=ROW_ID)
    if list(train.index) != list(disk_train.index):
        print("FAIL  rebuilt split assigns different rows to train")
        ok = False
    else:
        print("OK    rebuild is deterministic (identical row assignment)")

    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify, write nothing")
    args = parser.parse_args()

    if args.check:
        return check()

    train, test, customer_ids = build_frames()
    manifest = write(train, test, customer_ids)

    print(f"wrote {TRAIN_CSV.relative_to(REPO_ROOT)}  "
          f"{manifest['train']['rows']} rows, churn {manifest['train']['churn_rate']:.4%}")
    print(f"wrote {TEST_CSV.relative_to(REPO_ROOT)}   "
          f"{manifest['test']['rows']} rows, churn {manifest['test']['churn_rate']:.4%}")
    print(f"wrote {MANIFEST.relative_to(REPO_ROOT)}")
    print(f"      train sha256 {manifest['train']['sha256'][:16]}…")
    print(f"      test  sha256 {manifest['test']['sha256'][:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
