"""
download_snapshot_texts.py — step D (Wayback content fetcher)
================================================================
For every fintech in **company_urls_with_snapshots.csv** this script

1. reads the list‑of‑timestamps for both columns
     •  Terms_Snapshot_Timestamps
     •  Website_Snapshot_Timestamps
2. identifies *missing* snapshots – i.e. timestamp/URL pairs whose
   text‑file has **not yet been downloaded** to disk
3. downloads the archived HTML for each missing snapshot via the
   Wayback Machine, extracts **visible text** (strips <script>, <style>,…)
4. saves one *.txt* file per snapshot into the directory structure:

   Fintechs/<Fintech Name>/
       ├── Website/<TIMESTAMP>.txt
       └── Terms  /<TIMESTAMP>.txt

–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
Behaviour of the two limiting parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``MAX_COMPANIES``
    • *None*   → process **all** companies
    • *n > 0* → stop **after n companies with NEW downloads**
      (companies for which every snapshot is already present are skipped
      and do **not** count towards the limit)

``MAX_SNAPSHOTS_PER_URL``
    • *None*   → attempt **all** missing snapshots for the URL
    • *n > 0* → download at most *n NEW* snapshots per URL (per run)

Thus you can safely resume interrupted runs: already‑downloaded snapshots
are detected and skipped, and the limits apply only to fresh downloads.
Existing functionality from earlier steps is untouched – this is a
standalone script that **consumes** the CSV produced by
*step_b_fetch_wayback_snapshots.py*.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from requests.exceptions import RequestException

###############################################################################
# ─────────────────────────── configurable parameters ──────────────────────── #
###############################################################################
INPUT_FILE            = Path("company_urls_with_snapshots_monthly.csv")  # produced by step B
BASE_OUTPUT_DIR       = Path("Fintechs")                         # root folder for exports
MAX_COMPANIES         = None   # e.g. 3 for smoke test; None → all
MAX_SNAPSHOTS_PER_URL = None   # e.g. 10 to limit *new* downloads per URL
RATE_LIMIT            = 15    # seconds between snapshot downloads **and failures**
REQUEST_TIMEOUT       = 30     # seconds per HTTP request
HEADERS               = {"User-Agent": "Mozilla/5.0 (Wayback-Text-Dumper/1.3)"}
###############################################################################

_TIMESTAMP_RE = re.compile(r"^(\d{14})$")

###############################################################################
# Helper functions
###############################################################################


def _slugify(name: str) -> str:
    """Return filesystem‑safe slug from a fintech name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "unknown"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _visible_text_from_html(html: str) -> str:
    """Extract human‑visible text from raw HTML using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")

    # remove irrelevant elements
    for el in soup(["script", "style", "noscript", "iframe", "svg", "header", "footer", "nav"]):
        el.decompose()

    # remove comments
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        c.extract()

    text = soup.get_text("\n", strip=True)
    # collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _download_snapshot(target_url: str, timestamp: str) -> str | None:
    """Return HTML of `target_url` at `timestamp` from Wayback or None on failure."""
    if not _TIMESTAMP_RE.match(timestamp):
        print(f"      [WARN] invalid timestamp format skipped: {timestamp!r}")
        return None

    wb_url = f"https://web.archive.org/web/{timestamp}/{target_url}"
    try:
        resp = requests.get(wb_url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
        resp.raise_for_status()
        return resp.text
    except RequestException as exc:
        print(f"      [WARN] snapshot {timestamp} failed: {exc}")
        return None


def _parse_timestamp_list(cell: str | float) -> List[str]:
    """Deserialize the CSV cell containing a Python‑style list literal."""
    if isinstance(cell, float):  # NaN
        return []
    if isinstance(cell, list):
        return cell
    try:
        data = json.loads(cell.replace("'", '"'))
        return [str(x) for x in data if _TIMESTAMP_RE.match(str(x))]
    except Exception:
        # fallback: crude split
        return [t.strip(" '") for t in str(cell).strip("[]").split(',') if _TIMESTAMP_RE.match(t.strip())]

###############################################################################
# Main driver
###############################################################################

def _process(ts_iter: Iterable[str], target_url: str, out_dir: Path) -> int:
    """Download missing snapshots. Return count of *new* files saved."""
    missing_ts = [ts for ts in ts_iter if not (out_dir / f"{ts}.txt").exists()]
    if MAX_SNAPSHOTS_PER_URL is not None:
        missing_ts = missing_ts[:MAX_SNAPSHOTS_PER_URL]

    saved = 0
    for ts in missing_ts:
        html = _download_snapshot(target_url, ts)
        if html is not None:
            text = _visible_text_from_html(html)
            out_file = out_dir / f"{ts}.txt"
            try:
                out_file.write_text(text, encoding="utf-8")
                print(f"      saved {out_file.relative_to(BASE_OUTPUT_DIR)}")
                saved += 1
            except Exception as exc:
                print(f"      [WARN] could not write file: {out_file!s}: {exc}")
        # Always pause between attempts – successful **or** failed – to respect rate limits
        time.sleep(RATE_LIMIT)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Wayback snapshot texts for fintech sites/terms")
    parser.add_argument("--start-row", "-s", type=int, default=0,
                        help="Row index in INPUT_FILE to begin processing (0‑based)")
    args = parser.parse_args()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input CSV not found: {INPUT_FILE.resolve()}")

    df = pd.read_csv(INPUT_FILE, dtype=str)
    total_rows = len(df)
    if args.start_row < 0 or args.start_row >= total_rows:
        raise ValueError(f"--start-row must be between 0 and {total_rows-1}")

    # Trim DataFrame according to start row
    df = df.iloc[args.start_row:].reset_index(drop=True)

    required_cols = {
        "Fintech Name",
        "Fintech Website",
        "Terms URL",
        "Website_Snapshot_Timestamps",
        "Terms_Snapshot_Timestamps",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Input CSV missing columns: {', '.join(sorted(missing))}")

    processed_with_downloads = 0  # companies that yielded NEW downloads

    for pos, row in df.iterrows():
        # pos is now relative to trimmed DataFrame; compute absolute index for user clarity
        abs_idx = pos + args.start_row

        fintech_name = str(row.get("Fintech Name", f"fintech_{abs_idx+1}")).strip()
        slug = _slugify(fintech_name)
        website_url = str(row["Fintech Website"]).strip()
        terms_url = str(row["Terms URL"]).strip()

        website_ts = _parse_timestamp_list(row.get("Website_Snapshot_Timestamps", "[]"))
        terms_ts = _parse_timestamp_list(row.get("Terms_Snapshot_Timestamps", "[]"))

        # prepare output dirs
        site_dir = BASE_OUTPUT_DIR / slug / "Website"
        terms_dir = BASE_OUTPUT_DIR / slug / "Terms"
        _ensure_dir(site_dir)
        _ensure_dir(terms_dir)

        print(f"[{abs_idx+1}/{total_rows}] {fintech_name}")
        new_site = _process(website_ts, website_url, site_dir) if website_url else 0
        new_terms = _process(terms_ts, terms_url, terms_dir) if terms_url else 0

        total_new = new_site + new_terms
        print(f"      ↳ new downloads: website={new_site}, terms={new_terms}\n")

        if total_new > 0:
            processed_with_downloads += 1

        if MAX_COMPANIES is not None and processed_with_downloads >= MAX_COMPANIES:
            print(f"[INFO] Reached MAX_COMPANIES={MAX_COMPANIES} with new downloads. Stopping early.")
            break

    print(f"\n[INFO] All done. Files saved under ➜ {BASE_OUTPUT_DIR.resolve()}")

###############################################################################
if __name__ == "__main__":
    main()
