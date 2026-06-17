"""
scrape_homepages.py  (Phase 1)
===============================
For each company in the fintech timeline data, scrape the homepage from the
Wayback Machine every 6 months from 2005 through 2025.

Extracts: page title, visible text, all internal links (with anchor text).

Output:
    Data_wayback/homepage_snapshots.csv  — one row per (company, date)
    Data_wayback/html/homepages/{company}/{timestamp}.html — raw HTML

Usage:
    python scrape_homepages.py                # Full run
    python scrape_homepages.py --test 3       # First N companies
    python scrape_homepages.py --resume       # Resume from checkpoint
"""

import argparse
import csv
import glob
import json
import os
import time
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd

import importlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))
_wu = importlib.import_module("2_wayback_utils")
CDX_DELAY, FETCH_DELAY, OUTPUT_DIR = _wu.CDX_DELAY, _wu.FETCH_DELAY, _wu.OUTPUT_DIR
find_closest_snapshot, find_all_snapshots, select_snapshots = _wu.find_closest_snapshot, _wu.find_all_snapshots, _wu.select_snapshots
fetch_wayback_page = _wu.fetch_wayback_page
extract_text, extract_title, extract_links, extract_terms_links = _wu.extract_text, _wu.extract_title, _wu.extract_links, _wu.extract_terms_links
save_html, load_checkpoint, save_checkpoint = _wu.save_html, _wu.load_checkpoint, _wu.save_checkpoint

# ── Config ───────────────────────────────────────────────────────────────────
START_YEAR = 2005
END_YEAR = 2025
MONTHS = [1, 7]  # Jan and Jul = every 6 months
MAX_DAYS_OFF = 90  # skip if closest snapshot is >90 days from target

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "homepage_snapshots.csv")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "intermediate_files", "homepage_checkpoint.json")

FIELDNAMES = [
    "company_name", "company_website", "target_year", "target_month",
    "snapshot_timestamp", "snapshot_date", "page_title",
    "text_length", "num_links", "num_terms_links",
    "terms_link_urls", "html_path", "text_preview",
]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_companies(data_dir):
    """
    Load unique companies from timeline batches, matched with their websites.
    Returns list of (company_name, website) tuples.
    """
    # Load master timeline to get the list of companies we care about
    master = pd.read_csv(os.path.join(data_dir, "Data_cleaned", "fintech_timelines_master.csv"))
    products = master[master['entry_type'] == 'Product']
    products = products[pd.to_numeric(products['year_launched'], errors='coerce') >= 2005]
    timeline_companies = set(products['company_name'].unique())

    # Load website mapping
    fintechs = pd.read_csv(
        os.path.join(data_dir, "Data_cleaned", "banking_embedded_fintechs_unique.csv"))
    website_map = {}
    for _, row in fintechs.iterrows():
        name = str(row['fintechname']).strip()
        website = str(row['fintechwebsite']).strip().rstrip('/')
        website_map[name.lower()] = website

    # Match
    companies = []
    unmatched = []
    for company in sorted(timeline_companies):
        key = company.strip().lower()
        website = website_map.get(key)
        if not website:
            # Try without spaces
            key_nospace = key.replace(' ', '')
            for k, v in website_map.items():
                if k.replace(' ', '') == key_nospace:
                    website = v
                    break
        if website:
            companies.append((company, website))
        else:
            unmatched.append(company)

    if unmatched:
        print(f"  WARNING: {len(unmatched)} companies without websites: "
              f"{unmatched[:5]}...", flush=True)

    return companies


def get_homepage_url(website):
    """Extract the clean homepage URL from a website string."""
    parsed = urlparse(
        website if website.startswith('http') else f'https://{website}')
    # Return the base URL (scheme + netloc)
    return f"{parsed.scheme}://{parsed.netloc}/"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Scrape company homepages from Wayback Machine")
    parser.add_argument("--test", type=int, default=0,
                        help="Limit to first N companies (0 = all)")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh (ignore checkpoint, overwrite output)")
    parser.add_argument("--data-dir", type=str, default=".",
                        help="Directory containing the CSV files")
    parser.add_argument("--start-year", type=int, default=START_YEAR)
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--latest-only", action="store_true",
                        help="Fetch only the most recent snapshot per company")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading companies...", flush=True)
    companies = load_companies(args.data_dir)
    print(f"  {len(companies)} companies with websites", flush=True)

    if args.test > 0:
        companies = companies[:args.test]
        print(f"  TEST MODE: limited to {args.test} companies", flush=True)

    # Generate all (company, year, month) tasks
    tasks = []
    now = datetime.now()
    latest_only = args.latest_only

    if not latest_only:
        for company, website in companies:
            homepage = get_homepage_url(website)
            for yr in range(args.start_year, args.end_year + 1):
                for mo in MONTHS:
                    if yr > now.year or (yr == now.year and mo > now.month):
                        continue
                    tasks.append((company, website, homepage, yr, mo))

    print(f"  Mode: {'latest only' if latest_only else '6-month time-series'}")
    if not latest_only:
        print(f"  Total scrape tasks: {len(tasks)}", flush=True)
    else:
        print(f"  Companies to scrape: {len(companies)}", flush=True)

    # Checkpoint — always resume unless --fresh
    resume = not args.fresh
    done = load_checkpoint(CHECKPOINT_FILE) if resume else set()
    if done:
        print(f"  Resuming: {len(done)} already done", flush=True)

    file_exists = os.path.exists(OUTPUT_FILE) and resume
    mode = 'a' if file_exists else 'w'

    with open(OUTPUT_FILE, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, escapechar='\\', quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer.writeheader()

        completed = 0
        found = 0

        def fetch_and_write(company, website, homepage, yr, mo, ts, snap_url):
            """Fetch a snapshot and write the CSV row. Returns True if content found."""
            nonlocal completed, found

            html, final_url = fetch_wayback_page(ts, snap_url)
            time.sleep(FETCH_DELAY)

            if not html:
                writer.writerow({
                    "company_name": company,
                    "company_website": website,
                    "target_year": yr,
                    "target_month": mo,
                    "snapshot_timestamp": ts,
                    "snapshot_date": ts[:8],
                    "page_title": "",
                    "text_length": 0,
                    "num_links": 0,
                    "num_terms_links": 0,
                    "terms_link_urls": "",
                    "html_path": "",
                    "text_preview": "Fetch failed",
                })
                return False

            title = extract_title(html)
            text = extract_text(html)
            links = extract_links(html, homepage)
            terms_links = extract_terms_links(links)
            html_path = save_html("homepages", company, "homepage", ts, html)
            found += 1

            terms_urls = json.dumps(
                [{"path": l["path"], "anchor": l["anchor_text"][:80]}
                 for l in terms_links]
            ) if terms_links else ""

            writer.writerow({
                "company_name": company,
                "company_website": website,
                "target_year": yr,
                "target_month": mo,
                "snapshot_timestamp": ts,
                "snapshot_date": ts[:8],
                "page_title": title[:200],
                "text_length": len(text),
                "num_links": len(links),
                "num_terms_links": len(terms_links),
                "terms_link_urls": terms_urls,
                "html_path": html_path,
                "text_preview": text[:500].replace('\n', ' '),
            })

            print(f"    {yr}-{mo:02d} | {title[:40]:40s} | "
                  f"{len(text):>5} chars | {len(links):>3} links | "
                  f"{len(terms_links)} terms",
                  flush=True)
            return True

        if latest_only:
            # ── Latest-only mode: one CDX lookup per company ──────────
            for ci, (company, website) in enumerate(companies):
                homepage = get_homepage_url(website)
                ck = (company, 'latest')
                if ck in done:
                    continue

                print(f"\n[{ci+1}/{len(companies)}] {company} — {homepage}",
                      flush=True)

                all_snapshots = find_all_snapshots(homepage,
                                                   from_year=START_YEAR,
                                                   to_year=END_YEAR)
                time.sleep(CDX_DELAY)

                if not all_snapshots:
                    print(f"    No snapshots found", flush=True)
                    writer.writerow({
                        "company_name": company,
                        "company_website": website,
                        "target_year": "",
                        "target_month": "",
                        "snapshot_timestamp": "",
                        "snapshot_date": "",
                        "page_title": "",
                        "text_length": 0,
                        "num_links": 0,
                        "num_terms_links": 0,
                        "terms_link_urls": "",
                        "html_path": "",
                        "text_preview": "No snapshots in Wayback",
                    })
                    done.add(ck)
                    save_checkpoint(CHECKPOINT_FILE, done)
                    completed += 1
                    continue

                selected = select_snapshots(all_snapshots, latest_only=True)
                yr, mo, ts, snap_url = selected[0]
                print(f"    CDX: {len(all_snapshots)} snapshots, "
                      f"fetching latest ({ts[:8]})", flush=True)

                fetch_and_write(company, website, homepage, yr, mo,
                                ts, snap_url)
                done.add(ck)
                completed += 1
                f.flush()
                save_checkpoint(CHECKPOINT_FILE, done)

        else:
            # ── Time-series mode: one CDX call per (company, yr, mo) ──
            current_company = None
            for i, (company, website, homepage, yr, mo) in enumerate(tasks):
                ck = (company, yr, mo)
                if ck in done:
                    continue

                if company != current_company:
                    current_company = company
                    print(f"\n{'='*60}", flush=True)
                    print(f"  {company} — {homepage}", flush=True)

                completed += 1

                ts, snap_url = find_closest_snapshot(homepage, yr, mo)
                time.sleep(CDX_DELAY)

                if not ts:
                    writer.writerow({
                        "company_name": company,
                        "company_website": website,
                        "target_year": yr,
                        "target_month": mo,
                        "snapshot_timestamp": "",
                        "snapshot_date": "",
                        "page_title": "",
                        "text_length": 0,
                        "num_links": 0,
                        "num_terms_links": 0,
                        "terms_link_urls": "",
                        "html_path": "",
                        "text_preview": "No snapshot found",
                    })
                    done.add(ck)
                    if completed % 5 == 0:
                        f.flush()
                        save_checkpoint(CHECKPOINT_FILE, done)
                    continue

                # Check proximity
                try:
                    from datetime import datetime as dt
                    target_dt = dt(yr, mo, 1)
                    snap_dt = dt.strptime(ts[:8], "%Y%m%d")
                    days_off = abs((snap_dt - target_dt).days)
                except (ValueError, TypeError):
                    days_off = 0

                if days_off > MAX_DAYS_OFF:
                    writer.writerow({
                        "company_name": company,
                        "company_website": website,
                        "target_year": yr,
                        "target_month": mo,
                        "snapshot_timestamp": ts,
                        "snapshot_date": ts[:8],
                        "page_title": "",
                        "text_length": 0,
                        "num_links": 0,
                        "num_terms_links": 0,
                        "terms_link_urls": "",
                        "html_path": "",
                        "text_preview": f"Closest snapshot {days_off}d away, skipped",
                    })
                    done.add(ck)
                    if completed % 5 == 0:
                        f.flush()
                        save_checkpoint(CHECKPOINT_FILE, done)
                    continue

                fetch_and_write(company, website, homepage, yr, mo,
                                ts, snap_url)
                done.add(ck)

                if completed % 5 == 0:
                    f.flush()
                    save_checkpoint(CHECKPOINT_FILE, done)

    save_checkpoint(CHECKPOINT_FILE, done)

    print(f"\n{'='*60}", flush=True)
    print(f"Done! Results saved to {OUTPUT_FILE}", flush=True)
    print(f"  Completed: {completed}", flush=True)
    print(f"  Snapshots found: {found}", flush=True)

    # Summary
    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE, on_bad_lines='skip')
        print(f"  Total rows: {len(df)}", flush=True)
        has_content = df['text_length'] > 0
        print(f"  With content: {has_content.sum()} "
              f"({has_content.mean()*100:.0f}%)", flush=True)
        has_terms = df['num_terms_links'] > 0
        print(f"  With terms links: {has_terms.sum()} "
              f"({has_terms.mean()*100:.0f}%)", flush=True)


if __name__ == "__main__":
    main()
