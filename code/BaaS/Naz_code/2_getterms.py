"""
terms_page_finder.py

Locate and canonicalise each firm’s Terms‑of‑Service (or related legal) URL
and append it to a CSV, printing progress every 50 rows.

New in this version
-------------------
* **Incremental mode** — if an existing **`company_urls_with_terms.csv`** (or
  a custom `--out` path) is present, the script re‑uses the URLs already
  discovered and *only* searches the rows that are still missing. This dramatically
  speeds up re‑runs.

Key features
------------
* **Rich path heuristics** — 40 + common permutations (see ``CANDIDATE_PATHS``)
* **HEAD‑first, GET‑fallback** — minimal traffic; GET only if HEAD fails
* **Redirect‑aware** — up to 20 redirects per request
* **Per‑site budget** — bails after 20 s so no domain stalls the run
* **Fail‑fast timeout** — 5 s per request
* **Progress logging** — status update every 50 processed rows

Usage
-----
```bash
python terms_page_finder.py               # default run (company_urls.csv ➜ company_urls_with_terms.csv)
python terms_page_finder.py --test        # smoke‑test on 5 rows
python terms_page_finder.py --limit 20    # process first 20 rows
```
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import sys
import time
from pathlib import Path
from typing import Optional, Set
from urllib.parse import urljoin, urlparse, urlunparse

import pandas as pd
import requests
from requests.exceptions import Timeout, RequestException

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CANDIDATE_PATHS: tuple[str, ...] = (
    # Terms
    "/terms", "/terms/", "/terms.html", "/terms.htm",
    "/terms-of-service", "/terms-of-service/", "/terms_of_service", "/terms_of_service/",
    "/terms-use", "/terms-use/", "/terms-of-use", "/terms-of-use/", 
    "/privacy-and-terms", "privacy-and-terms",
    # ToS short
    "/tos", "/tos/",
    # Legal root
    "/legal", "/legal/terms", "/legal/terms/", "/legal/terms-of-service", "/legal/terms-of-service/",
    # Policy / policies
    "/policy", "/policy/", "/policy/terms", "/policy/terms/",
    "/policies", "/policies/", "/policies/terms", "/policies/terms/",
    # Terms & conditions
    "/terms-and-conditions", "/terms-and-conditions/", "/terms_conditions", "/terms_conditions/",
    "/legal/terms-and-conditions", "/legal/terms-and-conditions/",
    # Disclosures
    "/disclosures", "/disclosures/", "/disclosures.html", "/disclosures.htm",
    "/legal/disclosures", "/legal/disclosures/",
)

HEADERS = {
    "User-Agent":      "Mozilla/5.0 (compatible; TermsPageFinder/3.0; +https://github.com/your-repo)",
    "Accept":          "*/*",
    "Accept-Language": "en-US,en;q=0.5",
}
TIMEOUT         = 5      # seconds per request
MAX_REDIRECTS   = 20     # per URL
MAX_SITE_TIME   = 20     # seconds per base website
TEST_LIMIT      = 5
PROGRESS_EVERY  = 50

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def canonicalise_url(url: str) -> str:
    p = urlparse(url)
    return urlunparse((p.scheme, p.netloc, p.path.rstrip("/"), "", "", ""))


def _request(method: str, url: str) -> Optional[requests.Response]:
    try:
        if method == "HEAD":
            return requests.head(url, headers=HEADERS, allow_redirects=False, timeout=TIMEOUT)
        return requests.get(url, headers=HEADERS, allow_redirects=False, timeout=TIMEOUT, stream=True)
    except (Timeout, RequestException):
        return None


def url_exists(url: str) -> bool:
    seen: Set[str] = set()
    hops = 0
    current = url
    while hops <= MAX_REDIRECTS and current not in seen:
        seen.add(current)
        resp = _request("HEAD", current)
        if resp is None:
            return False
        if resp.status_code < 300:
            return True
        if 300 <= resp.status_code < 400 and "Location" in resp.headers:
            current = urljoin(current, resp.headers["Location"])
            hops += 1
            continue
        # fallback GET
        resp = _request("GET", current)
        if resp is None:
            return False
        if resp.status_code < 300:
            return True
        if 300 <= resp.status_code < 400 and "Location" in resp.headers:
            current = urljoin(current, resp.headers["Location"])
            hops += 1
            continue
        return False
    return False


def find_terms_page(base_url: str) -> Optional[str]:
    if not base_url:
        return None
    if not base_url.startswith(("http://", "https://")):
        base_url = "https://" + base_url
    start = time.time()
    def timed_out() -> bool:
        return time.time() - start > MAX_SITE_TIME

    for path in CANDIDATE_PATHS:
        if timed_out():
            return None
        cand = urljoin(base_url, path)
        if url_exists(cand):
            return canonicalise_url(cand)

    p = urlparse(base_url)
    segs = [s for s in p.path.split("/") if s]
    root = f"{p.scheme}://{p.netloc}"
    for i in range(len(segs), 0, -1):
        if timed_out():
            return None
        cand = f"{root}/{'/'.join(segs[:i])}/legal"
        if url_exists(cand):
            return canonicalise_url(cand)
    return None

# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------

def _process_idx(idx_site):
    idx, site = idx_site
    return idx, find_terms_page(site)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Locate Terms/Legal pages and append to CSV.")
    parser.add_argument("--in", dest="inp", default="company_urls.csv")
    parser.add_argument("--out", dest="out", default=None)
    parser.add_argument("--col", dest="col", default="Fintech Website")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--limit", "-n", type=int, default=None)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args(argv)

    if args.test and args.limit is None:
        args.limit = TEST_LIMIT

    inp_path = Path(args.inp)
    if not inp_path.exists():
        sys.exit(f"✖ Input file not found: {inp_path}")

    df = pd.read_csv(inp_path)
    if args.col not in df.columns:
        sys.exit(f"✖ Column '{args.col}' missing in {inp_path.name}")

    if args.limit and args.limit > 0:
        df = df.head(args.limit)
        print(f"→ Limited to first {len(df)} rows…")
    else:
        print(f"→ Processing {len(df)} websites…")

    # Determine output path & reuse existing Terms URL if present
    out_path = Path(args.out) if args.out else inp_path.with_stem(inp_path.stem + "_with_terms")
    if out_path.exists():
        prev = pd.read_csv(out_path)
        if "Terms URL" in prev.columns and args.col in prev.columns:
                    df = df.merge(prev[[args.col, "Terms URL"]], on=args.col, how="left", suffixes=("", "_prev"))
        # Combine only if the "_prev" suffix column actually exists (i.e. the
        # input *already* had a Terms URL column). Otherwise the merge put
        # the previous values directly into "Terms URL".
        if "Terms URL_prev" in df.columns:
            df["Terms URL"] = df["Terms URL"].combine_first(df["Terms URL_prev"])
            df.drop(columns=["Terms URL_prev"], inplace=True)
        print(f"→ Reusing existing URLs from {out_path.name} …")

    # Identify rows needing (re)processing
    mask = df["Terms URL"].isna() | (df["Terms URL"].astype(str).str.strip() == "")
    sites_to_process = df.loc[mask, args.col].fillna("").astype(str)
    idxs_to_process = sites_to_process.index.tolist()

    print(f"→ Need to resolve {len(idxs_to_process)} / {len(df)} sites…")

    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for processed, (idx, url) in enumerate(pool.map(_process_idx, [(i, df.at[i, args.col]) for i in idxs_to_process]), 1):
            df.at[idx, "Terms URL"] = url
            if processed % PROGRESS_EVERY == 0:
                print(f"…{processed}/{len(idxs_to_process)} newly processed")

    print("Writing output ⏎")
    df.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"✓ Saved → {out_path}")


if __name__ == "__main__":
    main()
