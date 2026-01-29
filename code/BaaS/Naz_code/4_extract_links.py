# wayback_snapshot_crawler.py
"""Download Wayback snapshots of each Terms‑of‑Service URL and harvest *all*
anchor links that appear in the HTML – **one unique URL per Terms page**.

**2025‑07‑19 update 7**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* **Overwrite CSV each run** – the script now *truncates* the output file and
  writes the header exactly once at the start of every execution.  Subsequent
  writes simply append rows, so each run produces a fresh CSV.
* All other behaviour (test‑mode limits, retry/back‑off, deduplication, etc.)
  remains unchanged from update 6.
"""
from __future__ import annotations

import ast
import csv
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
INPUT_CSV  = Path("company_urls_with_snapshots.csv")
OUTPUT_CSV = Path("company_tos_subpages.csv")

# test‑mode switches ----------------------------------------------------------
MAX_COMPANIES             = 1      # ``None`` → all companies
MAX_SNAPSHOTS_PER_COMPANY = 2      # ``None`` → all snapshots

# network behaviour -----------------------------------------------------------
SNAPSHOT_PAUSE_RANGE = (2.0, 6.0)   # seconds – ``random.uniform(*range)`` after fetch
TOTAL_RETRIES        = 5            # urllib3 ``Retry`` total
BACKOFF_FACTOR       = 1.5          # exponential back‑off base
TIMEOUT_SECS         = 20           # per‑request timeout

# Regex for Wayback "redirect page" generated at crawl time -------------------
_REDIRECT_RE = re.compile(r"Redirecting to\\.\\.\\.\\s*(https?://[^\\s\"'<]+)", re.I)

# Generic regex for any Wayback wrapper (handles modifiers like id_/ im_/ etc.)
_FULL_WAYBACK_RE = re.compile(
    r"https?://web\.archive\.org/web/[^/]+/(https?://.+)", re.I
)

# ──────────────────────────────────────────────────────────────────────────────
# helper functions
# ──────────────────────────────────────────────────────────────────────────────

def build_session() -> requests.Session:
    """Create a ``requests.Session`` with retry/back‑off suitable for Wayback."""
    retry = Retry(
        total=TOTAL_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        )
    })
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def pause():
    time.sleep(random.uniform(*SNAPSHOT_PAUSE_RANGE))


def fetch_snapshot_html(session: requests.Session, timestamp: str, url: str) -> str | None:
    """Return HTML for a Wayback snapshot or ``None`` on repeated failure."""
    snapshot_url = f"https://web.archive.org/web/{timestamp}/{url}"
    try:
        resp = session.get(snapshot_url, timeout=TIMEOUT_SECS, allow_redirects=True)
        if resp.status_code == 200:
            return resp.text
        # Wayback sometimes serves 302 to a WRITTEN html redirect doc; treat as HTML
        if resp.status_code in {301, 302} and "text/html" in resp.headers.get("Content-Type", ""):
            return resp.text
        print(f"      [WARN] snapshot {timestamp} got HTTP {resp.status_code} – skipping")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"      [WARN] snapshot {timestamp} failed: {exc}")
        return None
    finally:
        pause()

# ──────────────────────────────────────────────────────────────────────────────
# link extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def _unwrap_wayback_once(url: str) -> str:
    """Strip a **single** Wayback wrapper layer from *url* (if present)."""
    m = _FULL_WAYBACK_RE.search(url)
    return m.group(1) if m else url


def _resolve_wayback_url(href: str) -> str | None:  # noqa: C901
    """Return a cleaned HTTP(S) URL or ``None`` if it is *still* a Wayback link or
    uses a non‑HTTP(S) scheme.
    """
    href = href.strip()
    if not href:
        return None

    # Deal with protocol‑relative & triple‑slash URLs -------------------------
    if href.startswith("///"):
        href = "https:" + href[2:]
    elif href.startswith("//"):
        href = "https:" + href

    # Relative Wayback path ---------------------------------------------------
    if href.startswith("/web/"):
        href = "https://web.archive.org" + href

    # Peel away *all* nested Wayback wrappers ---------------------------------
    previous = None
    while href != previous:
        previous = href
        href = _unwrap_wayback_once(href)

    # Fallback: generic strip if host still equals web.archive.org ------------
    if "web.archive.org" in (urlparse(href).hostname or "").lower():
        m = _FULL_WAYBACK_RE.search(href)
        if m:
            href = m.group(1)

    # Abort if non‑HTTP scheme -------------------------------------------------
    if not href.startswith("http"):
        return None

    host = (urlparse(href).hostname or "").lower()
    if "web.archive.org" in host:
        return None  # drop remaining archive links

    # Trim trailing slash for deduplication -----------------------------------
    if href.endswith("/"):
        href = href[:-1]

    return href


def parse_subpage_links(html: str) -> Set[str]:
    """Return *all* unique, cleaned HTTP(S) URLs found in *html*."""
    soup = BeautifulSoup(html, "html.parser")
    links: Set[str] = set()

    for a in soup.find_all("a", href=True):
        resolved = _resolve_wayback_url(a["href"])
        if not resolved:
            continue
        parsed = urlparse(resolved)
        clean = parsed._replace(query="", fragment="").geturl()
        if clean.endswith("/"):
            clean = clean[:-1]
        links.add(clean)

    return links

# ──────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ──────────────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "Fintech Name",
    "Terms URL",
    "Snapshot Timestamp",  # first occurrence
    "Subpage URL",
]


def _parse_snapshot_list(raw: str) -> List[str]:
    raw = raw.strip()
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = ast.literal_eval(raw)
            return [s.strip() for s in parsed if s.strip()]
        except Exception:
            return []
    return [s.strip() for s in raw.split(";") if s.strip()]


def load_input(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if MAX_COMPANIES and i >= MAX_COMPANIES:
                break
            rows.append(row)
    return rows

# ─── new: initialisation to overwrite file each run ─────────────────────────-

def initialise_output_file():
    """Create/overwrite the output CSV and write header once."""
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()


def write_output(rows: Iterable[dict]):
    """Append *rows* to the already‑initialised output CSV."""
    if not rows:
        return
    with OUTPUT_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerows(rows)

# ──────────────────────────────────────────────────────────────────────────────
# main processing
# ──────────────────────────────────────────────────────────────────────────────

def main():
    if not INPUT_CSV.exists():
        raise SystemExit(f"Input file {INPUT_CSV} not found")

    # fresh output file every run -------------------------------------------
    initialise_output_file()

    companies = load_input(INPUT_CSV)
    if not companies:
        raise SystemExit("No rows found in input CSV – aborting")

    session = build_session()
    aggregated_rows: List[dict] = []

    for idx, comp in enumerate(companies, 1):
        name: str       = comp.get("Fintech Name", "").strip()
        tos_url: str    = comp.get("Terms URL", "").strip()
        snapshots       = _parse_snapshot_list(comp.get("Snapshot_Timestamps", ""))
        if MAX_SNAPSHOTS_PER_COMPANY:
            snapshots = snapshots[:MAX_SNAPSHOTS_PER_COMPANY]

        print(f"[{idx}/{len(companies)}] {name}: {len(snapshots)} snapshots -> {tos_url}")

        # Track unique sub‑links for this Terms URL ---------------------------
        seen_links: Dict[str, str] = {}  # link -> first snapshot timestamp

        for ts in snapshots:
            html = fetch_snapshot_html(session, ts, tos_url)
            if not html:
                continue

            # Detect permanent Wayback HTML redirect page -------------------
            if "Redirecting to..." in html:
                m = _REDIRECT_RE.search(html)
                if m:
                    redirected = m.group(1)
                    html = fetch_snapshot_html(session, ts, redirected)
                    if not html:
                        continue

            # ─── CLEANED LINKS ──────────────────────────────────────────────
            clean_links = parse_subpage_links(html)
            print(f"   • {ts}: captured {len(clean_links)} sublink(s)")

            for link in clean_links:
                # SECOND pass through resolver for absolute safety ----------
                link_cleaned = _resolve_wayback_url(link) or link
                if link_cleaned not in seen_links:
                    seen_links[link_cleaned] = ts  # remember first timestamp

        print(f"   ↳ unique sublinks captured for {name}: {len(seen_links)}\n")

        # Append deduplicated rows -------------------------------------------
        for link, first_ts in seen_links.items():
            aggregated_rows.append({
                "Fintech Name":       name,
                "Terms URL":          tos_url,
                "Snapshot Timestamp": first_ts,
                "Subpage URL":        link,
            })

        # flush periodically to avoid large memory usage ----------------------
        if len(aggregated_rows) >= 1000:
            write_output(aggregated_rows)
            aggregated_rows.clear()

    # final flush -------------------------------------------------------------
    if aggregated_rows:
        write_output(aggregated_rows)

    print("Done – results written to", OUTPUT_CSV)


if __name__ == "__main__":
    main()
