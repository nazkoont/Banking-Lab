"""
2f_scrape_deeper.py
===================
Post-processing scraper that follows disclosure/terms links found inside
already-scraped product HTML files and fetches those pages from the
Wayback Machine.

For each company's product HTML files:
  1. Parse all <a href> links
  2. Keep only links matching disclosure patterns (terms, legal, fdic, etc.)
  3. Resolve relative URLs to absolute using the original page's target_url
  4. Fetch each from Wayback Machine (same CDX + snapshot approach as 2c)
  5. Save HTML to Data_wayback/html/deeper/{Company}/
  6. Append rows to Data_wayback/product_pages.csv

Supports resuming via checkpoint file.

Usage:
    python3 Code/2f_scrape_deeper.py
    python3 Code/2f_scrape_deeper.py --company Bluevine
    python3 Code/2f_scrape_deeper.py --limit 50
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

csv.field_size_limit(sys.maxsize)

# ── Config ────────────────────────────────────────────────────────────────────
PRODUCTS_HTML_DIR = "Data_wayback/html/products"
DEEPER_HTML_DIR   = "Data_wayback/html/deeper"
PRODUCT_PAGES_CSV = "Data_wayback/product_pages.csv"
CHECKPOINT_FILE   = "Data_wayback/intermediate_files/deeper_checkpoint.json"
LOG_FILE          = "Data_wayback/intermediate_files/deeper_log.txt"

WAYBACK_CDX_URL  = "http://web.archive.org/cdx/search/cdx"
WAYBACK_BASE_URL = "https://web.archive.org/web"

MAX_SNAPSHOTS_PER_URL = 10
MAX_URLS_PER_COMPANY   = 30
REQUEST_TIMEOUT       = 30
RETRY_DELAYS          = [15, 30, 60]

# Links matching these patterns are followed
DISCLOSURE_PATTERNS = [
    'terms', 'legal', 'disclosure', 'agreement', 'cardholder',
    'fee-schedule', 'fee_schedule', 'important-information',
    'banking-services', 'bank-partner', 'partner-bank',
    'deposit-account', 'deposit-agreement', 'fdic-protection',
    'fdic-insured', 'member-fdic', 'privacy-policy',
    'privacy-notice', 'account-agreement', 'checking-agreement',
    'savings-agreement', 'card-agreement', 'loan-agreement',
]

# Skip these — external sites, nav links, etc.
SKIP_DOMAINS = [
    'fdic.gov', 'consumerfinance.gov', 'linkedin.com', 'twitter.com',
    'facebook.com', 'instagram.com', 'youtube.com', 'apple.com',
    'google.com', 'play.google.com', 'apps.apple.com',
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE) as f:
        return set(tuple(x) for x in json.load(f))


def save_checkpoint(done):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(list(done), f)


def is_disclosure_link(href):
    href_lower = href.lower()
    return any(p in href_lower for p in DISCLOSURE_PATTERNS)


def should_skip(url):
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    return any(skip in domain for skip in SKIP_DOMAINS)


def resolve_url(href, base_url):
    """Resolve relative href to absolute URL using base_url."""
    if not href or href.startswith('#') or href.startswith('mailto:'):
        return None
    return urljoin(base_url, href)


def extract_disclosure_links(html_path, base_url):
    """Extract all disclosure-pattern links from an HTML file."""
    try:
        with open(html_path, errors='replace') as f:
            soup = BeautifulSoup(f, 'html.parser')
    except Exception:
        return []

    links = []
    seen = set()
    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        if not is_disclosure_link(href):
            continue
        url = resolve_url(href, base_url)
        if not url or url in seen:
            continue
        if should_skip(url):
            continue
        # Strip wayback prefix if accidentally present
        url = re.sub(r'^https?://web\.archive\.org/web/\d+/', '', url)
        seen.add(url)
        links.append({
            'url': url,
            'link_text': a.get_text(strip=True)[:100],
        })
    return links


def fetch_cdx(url, year_start=2005, year_end=2026):
    """Query Wayback CDX for snapshots of a URL. Returns list of timestamps."""
    timestamps = []
    for year in range(year_start, year_end + 1):
        params = {
            'url': url,
            'output': 'json',
            'fl': 'timestamp,statuscode',
            'filter': 'statuscode:200',
            'from': f'{year}0101',
            'to': f'{year}1231',
            'limit': 3,
            'collapse': 'timestamp:6',
        }
        for attempt, delay in enumerate([0] + RETRY_DELAYS):
            if delay:
                time.sleep(delay)
            try:
                r = requests.get(WAYBACK_CDX_URL, params=params, timeout=REQUEST_TIMEOUT)
                if r.status_code == 200:
                    rows = r.json()
                    if rows and len(rows) > 1:
                        for row in rows[1:]:
                            timestamps.append(row[0])
                    break
                elif r.status_code in (429, 503, 504):
                    continue
                else:
                    break
            except Exception:
                if attempt == len(RETRY_DELAYS):
                    break
    return timestamps


def fetch_snapshot(timestamp, url):
    """Fetch a single Wayback snapshot. Returns HTML string or None."""
    wayback_url = f"{WAYBACK_BASE_URL}/{timestamp}/{url}"
    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            r = requests.get(wayback_url, timeout=REQUEST_TIMEOUT)
            if r.status_code == 200 and len(r.text) > 200:
                return r.text, wayback_url
        except Exception:
            pass
    return None, None


def save_html(html, company_slug, url_slug, timestamp):
    """Save HTML to deeper directory. Returns filepath."""
    folder = os.path.join(DEEPER_HTML_DIR, company_slug)
    os.makedirs(folder, exist_ok=True)
    safe_slug = re.sub(r'[^a-zA-Z0-9_\-]', '_', url_slug)[:60]
    fname = f"{safe_slug}_{timestamp}.html"
    fpath = os.path.join(folder, fname)
    with open(fpath, 'w', encoding='utf-8', errors='replace') as f:
        f.write(html)
    return fpath


def append_to_product_pages(rows):
    """Append rows to product_pages.csv."""
    file_exists = os.path.exists(PRODUCT_PAGES_CSV)
    fieldnames = [
        'company_name', 'product_name', 'year_launched', 'is_financial',
        'page_type', 'source', 'target_url', 'target_year', 'target_month',
        'snapshot_timestamp', 'snapshot_date', 'page_title', 'text_length',
        'html_path', 'text_preview',
    ]
    with open(PRODUCT_PAGES_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames,
            escapechar='\\', quoting=csv.QUOTE_MINIMAL
        )
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--company', type=str, default=None,
                        help='Only process this company folder name')
    parser.add_argument('--limit', type=int, default=0,
                        help='Max companies to process (0 = all)')
    args = parser.parse_args()

    os.makedirs(DEEPER_HTML_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)

    # Load product_pages.csv to get base URLs for each HTML file
    print("Loading product_pages.csv for base URL mapping...", flush=True)
    pp = pd.read_csv(PRODUCT_PAGES_CSV, on_bad_lines='skip')
    url_map = {}  # html_basename -> (target_url, company_name, product_name, year_launched, is_financial)
    for _, row in pp.iterrows():
        basename = os.path.basename(str(row.get('html_path', '')))
        if basename:
            url_map[basename] = {
                'target_url': str(row.get('target_url', '')),
                'company_name': str(row.get('company_name', '')),
                'product_name': str(row.get('product_name', '')),
                'year_launched': row.get('year_launched', ''),
                'is_financial': row.get('is_financial', True),
            }
    print(f"  {len(url_map)} HTML files mapped", flush=True)

    checkpoint = load_checkpoint()
    print(f"  {len(checkpoint)} URLs already processed (checkpoint)", flush=True)

    # Collect company folders
    companies = sorted(os.listdir(PRODUCTS_HTML_DIR))
    if args.company:
        companies = [c for c in companies if c.lower() == args.company.lower()]
    if args.limit:
        companies = companies[:args.limit]

    total_saved = 0
    total_urls  = 0

    for company_slug in companies:
        company_path = os.path.join(PRODUCTS_HTML_DIR, company_slug)
        # Skip non-priority companies
        priority_file = 'Data_wayback/intermediate_files/priority_companies.txt'
        if os.path.exists(priority_file):
            with open(priority_file) as pf:
                priority = set(pf.read().splitlines())
            if company_slug not in priority:
                continue
        if not os.path.isdir(company_path):
            continue

        html_files = [f for f in os.listdir(company_path) if f.endswith('.html')]
        if not html_files:
            continue

        # Collect all disclosure links from this company's HTML files
        company_links = {}  # url -> metadata
        for fname in html_files:
            fpath = os.path.join(company_path, fname)
            meta = url_map.get(fname, {})
            base_url = meta.get('target_url', '')
            if not base_url:
                continue
            links = extract_disclosure_links(fpath, base_url)
            for link in links:
                url = link['url']
                if url not in company_links:
                    company_links[url] = {**link, **meta}

        if not company_links:
            continue

        print(f"\n{'='*60}", flush=True)
        # Cap URLs per company to avoid getting stuck on link-heavy sites
        if len(company_links) > MAX_URLS_PER_COMPANY:
            company_links = dict(list(company_links.items())[:MAX_URLS_PER_COMPANY])

        print(f"  {company_slug} — {len(company_links)} disclosure URLs (capped at {MAX_URLS_PER_COMPANY})", flush=True)

        for url, meta in company_links.items():
            checkpoint_key = (company_slug, url)
            if checkpoint_key in checkpoint:
                continue

            total_urls += 1
            url_slug = re.sub(r'[^a-zA-Z0-9_\-]', '_', urlparse(url).path)[:50]
            print(f"  Fetching: {url[:80]}", flush=True)

            timestamps = fetch_cdx(url)
            if not timestamps:
                print(f"    No snapshots found", flush=True)
                checkpoint.add(checkpoint_key)
                save_checkpoint(checkpoint)
                continue

            # Select up to MAX_SNAPSHOTS evenly spaced
            step = max(1, len(timestamps) // MAX_SNAPSHOTS_PER_URL)
            selected = timestamps[::step][:MAX_SNAPSHOTS_PER_URL]
            print(f"    {len(timestamps)} snapshots available, fetching {len(selected)}", flush=True)

            rows_to_append = []
            for ts in selected:
                html, wayback_url = fetch_snapshot(ts, url)
                if not html:
                    continue

                # Parse date from timestamp
                year  = ts[:4]
                month = ts[4:6]
                date  = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"

                # Extract title and preview
                # skip PDFs and other binary content
                if isinstance(html, bytes) or html.strip().startswith('%PDF') or not html.strip().startswith('<'):
                    continue
                try:
                    soup = BeautifulSoup(html, 'html.parser')
                except Exception:
                    continue
                title = soup.title.string.strip() if soup.title and soup.title.string else ''
                text  = soup.get_text(separator=' ', strip=True)
                preview = text[:200]

                fpath = save_html(html, company_slug, url_slug, ts)
                total_saved += 1

                rows_to_append.append({
                    'company_name':      meta.get('company_name', company_slug),
                    'product_name':      meta.get('product_name', ''),
                    'year_launched':     meta.get('year_launched', ''),
                    'is_financial':      meta.get('is_financial', True),
                    'page_type':         'deeper_disclosure',
                    'source':            'wayback_deeper',
                    'target_url':        url,
                    'target_year':       year,
                    'target_month':      month,
                    'snapshot_timestamp': ts,
                    'snapshot_date':     date,
                    'page_title':        title[:200],
                    'text_length':       len(text),
                    'html_path':         fpath,
                    'text_preview':      preview,
                })

            if rows_to_append:
                append_to_product_pages(rows_to_append)
                print(f"    Saved {len(rows_to_append)} snapshots", flush=True)

            checkpoint.add(checkpoint_key)
            save_checkpoint(checkpoint)

    print(f"\nDone! Processed {total_urls} URLs, saved {total_saved} snapshots", flush=True)


if __name__ == '__main__':
    main()
