"""
filter_snapshot_timestamps.py — compress Wayback snapshot lists
================================================================
For every fintech in **company_urls_with_snapshots.csv** this standalone script

1. reads the two list‑of‑timestamps columns
     •  Website_Snapshot_Timestamps
     •  Terms_Snapshot_Timestamps
2. keeps **at most one snapshot per calendar month** (YYYY‑MM) inside each
   list – choosing the *earliest* snapshot in that month (you can change the
   policy via `KEEP_LATEST_PER_MONTH`).
3. updates / creates the matching *_Count columns so they reflect the **new**
   number of timestamps remaining after filtering.
4. writes the pruned lists back to a **new CSV** file with the same columns
   (plus an informational `Filtered` column) so that downstream steps can use
   the lighter dataset.

Existing scripts are untouched – this file replaces the previous canvas code
entirely, yet it re‑uses variable naming conventions so it drops right into
your workflow.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

import pandas as pd

###############################################################################
# ─────────────────────────── configurable parameters ──────────────────────── #
###############################################################################
INPUT_FILE  = Path("company_urls_with_snapshots.csv")
OUTPUT_FILE = Path("company_urls_with_snapshots_monthly.csv")  # new file
TIMESTAMP_COLS = [
    "Website_Snapshot_Timestamps",
    "Terms_Snapshot_Timestamps",
]
KEEP_LATEST_PER_MONTH = False  # False → keep earliest in month, True → latest
###############################################################################

# automatically derive the accompanying *count* column names
COUNT_COLS = {c: c.replace("Timestamps", "Count") for c in TIMESTAMP_COLS}

_TIMESTAMP_RE = re.compile(r"^(\d{14})$")  # 14‑digit Wayback timestamp


def _parse_ts_list(cell) -> List[str]:
    """Deserialize timestamp list from CSV cell (handles NaN / strings)."""
    if pd.isna(cell):
        return []

    # Already a Python list
    if isinstance(cell, list):
        return [str(x) for x in cell if _TIMESTAMP_RE.match(str(x))]

    # Attempt JSON‑ish parsing – convert single quotes → double for json.loads
    try:
        data = json.loads(str(cell).replace("'", '"'))
        return [str(x) for x in data if _TIMESTAMP_RE.match(str(x))]
    except Exception:
        # Fallback: manual split on comma, strip common brackets & quotes
        tokens = [tok.strip(" ' \"") for tok in str(cell).strip("[]").split(',')]
        return [tok for tok in tokens if _TIMESTAMP_RE.match(tok)]


def _filter_one_per_month(ts_list: List[str]) -> List[str]:
    """Return a list with ≤1 timestamp per YYYYMM."""
    if not ts_list:
        return []

    # sort ascending so earliest comes first; later we may pick latest instead
    ts_sorted = sorted(ts_list)
    months_seen = {}
    for ts in ts_sorted:
        ym = ts[:6]  # YYYYMM
        if ym not in months_seen:
            months_seen[ym] = ts
        elif KEEP_LATEST_PER_MONTH:  # overwrite to keep latest
            months_seen[ym] = ts

    # restore chronological order (earliest→latest)
    return [months_seen[m] for m in sorted(months_seen.keys())]


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_FILE.resolve()}")

    df = pd.read_csv(INPUT_FILE, dtype=str)

    # Validate required columns
    missing = [c for c in TIMESTAMP_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Input CSV missing columns: {', '.join(missing)}")

    filtered_counts = {col: [0, 0] for col in TIMESTAMP_COLS}  # before, after

    for col in TIMESTAMP_COLS:
        count_col = COUNT_COLS[col]
        new_values = []
        new_counts = []
        for cell in df[col].fillna(''):
            ts_list = _parse_ts_list(cell)
            filtered = _filter_one_per_month(ts_list)
            new_values.append(str(filtered))  # store as Python‑style list string
            new_counts.append(len(filtered))

            # tracking stats
            filtered_counts[col][0] += len(ts_list)
            filtered_counts[col][1] += len(filtered)

        # assign filtered list and count
        df[col] = new_values
        df[count_col] = new_counts

    # Optional: add a column indicating script & keep policy
    df["Filtered"] = f"kept ≤1 per month (latest={KEEP_LATEST_PER_MONTH})"

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Wrote pruned CSV ➜ {OUTPUT_FILE.resolve()}")

    # Stats summary
    for col, (before, after) in filtered_counts.items():
        print(f" • {col}: {before} → {after} timestamps (saved {before - after})")


if __name__ == "__main__":
    main()
