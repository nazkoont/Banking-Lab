"""
step_b_fetch_wayback_snapshots.py  — enhanced (v2)

Fetch Wayback Machine snapshot timestamps for both the **Terms of Service URL** *and* the main
**Fintech Website** for every company.

Input :  company_urls_with_terms.csv  (must contain columns  "Terms URL"  and  "Fintech Website")
Output:  company_urls_with_snapshots.csv  (adds 4 new columns)
           • Terms_Snapshot_Timestamps   – list[str]
           • Terms_Snapshot_Count        – int
           • Website_Snapshot_Timestamps – list[str]
           • Website_Snapshot_Count      – int

This version adds *robust retry logic* with exponential back‑off around the CDX API
requests to cope with transient network errors (e.g. connection‑refused or
rate‑limiting by web.archive.org). All existing functionality is preserved.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

import pandas as pd
import requests
from requests.exceptions import RequestException

# ─────────────────────────── configurable parameters ────────────────────────── #
INPUT_FILE      = Path("company_urls_with_terms.csv")      # must contain required columns
OUTPUT_FILE     = Path("company_urls_with_snapshots.csv")  # overwritten on every run
MAX_COMPANIES   = None   # e.g. 5 for quick smoke‑test, or None for full run
RATE_LIMIT      = 20    # seconds between API calls (per URL)
REQUEST_TIMEOUT = 30      # seconds per HTTP request
RETRIES         = 4       # total attempts per CDX query
BACKOFF_FACTOR  = 2       # exponential back‑off factor (s, 2s, 4s …)
# ─────────────────────────────────────────────────────────────────────────────── #

CDX_API = "https://web.archive.org/cdx/search/cdx"
HEADERS = {"User-Agent": "Mozilla/5.0 (Wayback-Snapshot-Fetcher/2.1)"}

###############################################################################
# Helper functions
###############################################################################

def _sleep_with_message(seconds: float) -> None:
    """Sleep helper that prints the waiting time (useful for long back‑offs)."""
    if seconds <= 0:
        return
    print(f"        …waiting {seconds:.1f}s before retry")
    time.sleep(seconds)


def _cdx_timestamps(url: str) -> List[str]:
    """Return list of YYYYMMDDhhmmss timestamps for one URL via CDX API.

    Implements exponential‑backoff retries to mitigate temporary network issues
    or rate limits from the Wayback Machine.
    """
    params = {
        "url": url,
        "output": "json",
        "fl": "timestamp",
        "filter": "statuscode:200",
        "collapse": "timestamp:8",  # at most one snapshot per day
    }

    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.get(
                CDX_API,
                params=params,
                timeout=REQUEST_TIMEOUT,
                headers=HEADERS,
            )
            resp.raise_for_status()
            data = resp.json()
            return [row[0] for row in data[1:]] if len(data) > 1 else []

        except RequestException as exc:
            if attempt == RETRIES:
                print(f"[WARN] CDX query failed for {url!r} after {RETRIES} attempts: {exc}")
                return []
            backoff = BACKOFF_FACTOR ** (attempt - 1)
            print(f"[WARN] attempt {attempt}/{RETRIES} failed for {url!r}: {exc}")
            _sleep_with_message(backoff)
            # loop continues for retry



def _resolve_redirect(url: str) -> str:
    """Follow today’s redirects (HTTP 3xx) and return the final URL (or the original)."""
    try:
        resp = requests.head(url, allow_redirects=True, timeout=10, headers=HEADERS)
        final_url = resp.url
        if resp.status_code >= 400:
            resp2 = requests.get(url, allow_redirects=True, timeout=10, stream=True, headers=HEADERS)
            final_url = resp2.url
        return final_url
    except RequestException as exc:
        print(f"[WARN] could not resolve redirects for {url!r}: {exc}")
        return url  # fallback to original


def fetch_timestamps(target_url: str) -> List[str]:
    """Return list of snapshot timestamps for *target_url*, retrying on redirect."""
    if not target_url or target_url.lower().startswith("nan"):
        return []

    ts = _cdx_timestamps(target_url)
    if ts:
        return ts

    # Retry once with resolved redirect target (covers e.g. 307 → sub‑domain)
    final_url = _resolve_redirect(target_url)
    if final_url and final_url != target_url:
        print(f"    ⮕ retrying with redirected URL: {final_url}")
        ts = _cdx_timestamps(final_url)
    return ts

###############################################################################
# Main driver
###############################################################################

def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_FILE.resolve()}")

    df = pd.read_csv(INPUT_FILE)
    required_cols = {"Terms URL", "Fintech Website"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Input CSV is missing columns: {', '.join(missing)}")

    if MAX_COMPANIES is not None:
        df = df.head(int(MAX_COMPANIES))
        print(f"[INFO] Test mode: processing first {len(df)} companies\n")

    terms_ts_col: list[List[str]] = []
    site_ts_col:  list[List[str]] = []

    for idx, row in df.iterrows():
        terms_url = str(row["Terms URL"]).strip()
        site_url  = str(row["Fintech Website"]).strip()

        print(f"[{idx+1}/{len(df)}] {site_url or 'N/A'} …")

        # Snapshots for Terms URL
        ts_terms = fetch_timestamps(terms_url)
        terms_ts_col.append(ts_terms)
        time.sleep(RATE_LIMIT)

        # Snapshots for main Website (if present)
        ts_site = fetch_timestamps(site_url)
        site_ts_col.append(ts_site)
        time.sleep(RATE_LIMIT)

    # Add/replace columns
    df["Terms_Snapshot_Timestamps"]   = terms_ts_col
    df["Terms_Snapshot_Count"]        = df["Terms_Snapshot_Timestamps"].apply(len)
    df["Website_Snapshot_Timestamps"] = site_ts_col
    df["Website_Snapshot_Count"]      = df["Website_Snapshot_Timestamps"].apply(len)

    # Persist (overwrite)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n[INFO] Saved results ➜ {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
