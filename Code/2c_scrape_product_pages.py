"""
scrape_product_pages.py  (Phase 2)
====================================
Uses homepage snapshots (from Phase 1) to discover product pages, then
scrapes them from the Wayback Machine every 6 months.

Logic for each product:
  1. Search homepage text for product name mentions
  2. If found → extract linked product page URLs from homepage
  3. If NOT found → fall back to keyword_1 from product_urls.csv
  4. For financial products → also scrape terms/legal/disclosures pages
  5. Follow one level deep for product detail or terms sublinks

Output:
    Data_wayback/product_pages.csv   — one row per (product, url, date)
    Data_wayback/html/products/{company}/{label}_{timestamp}.html

Usage:
    python scrape_product_pages.py               # Full run
    python scrape_product_pages.py --test 5      # First N products
    python scrape_product_pages.py --resume      # Resume from checkpoint
"""

import argparse
import csv
import glob
import json
import os
import re
import time
from datetime import datetime
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup

import importlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))
_wu = importlib.import_module("2_wayback_utils")
CDX_DELAY, CDX_PREFIX_DELAY, FETCH_DELAY, OUTPUT_DIR = _wu.CDX_DELAY, _wu.CDX_PREFIX_DELAY, _wu.FETCH_DELAY, _wu.OUTPUT_DIR
find_closest_snapshot, find_closest_snapshot_prefix, fetch_wayback_page = _wu.find_closest_snapshot, _wu.find_closest_snapshot_prefix, _wu.fetch_wayback_page
find_all_snapshots, select_snapshots = _wu.find_all_snapshots, _wu.select_snapshots
is_js_only_page, fetch_wayback_page_playwright = _wu.is_js_only_page, _wu.fetch_wayback_page_playwright
extract_text, extract_title, extract_links = _wu.extract_text, _wu.extract_title, _wu.extract_links
extract_terms_links, find_product_links, is_product_detail_or_terms_link = _wu.extract_terms_links, _wu.find_product_links, _wu.is_product_detail_or_terms_link
product_name_tokens, _simple_stem = _wu.product_name_tokens, _wu._simple_stem
save_html, load_checkpoint, save_checkpoint = _wu.save_html, _wu.load_checkpoint, _wu.save_checkpoint
is_financial_product, ALL_FINANCIAL_SUBCATS, FINANCIAL_CATS = _wu.is_financial_product, _wu.ALL_FINANCIAL_SUBCATS, _wu.FINANCIAL_CATS

# ── Config ───────────────────────────────────────────────────────────────────
START_YEAR = 2005
END_YEAR = 2025
MONTHS = [1, 7]  # every 6 months
MAX_DAYS_OFF = 90
MAX_DAYS_OFF_TERMS = 365  # wider window for terms/legal pages (sparse in Wayback)

HOMEPAGE_FILE = os.path.join(OUTPUT_DIR, "homepage_snapshots.csv")
KEYWORD_URLS_FILE = os.path.join(OUTPUT_DIR, "product_urls.csv")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "product_pages.csv")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "intermediate_files", "product_pages_checkpoint.json")

FIELDNAMES = [
    "company_name", "product_name", "year_launched",
    "is_financial", "page_type", "source",
    "target_url", "target_year", "target_month",
    "snapshot_timestamp", "snapshot_date", "page_title",
    "text_length", "html_path", "text_preview",
]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_timeline(data_dir):
    """Load product timeline data from master file."""
    master = pd.read_csv(os.path.join(data_dir, "Data_cleaned", "fintech_timelines_master.csv"))
    products = master[master['entry_type'] == 'Product'].copy()
    products = products[pd.to_numeric(products['year_launched'], errors='coerce') >= 2005].copy()
    return products


def load_homepage_snapshots():
    """Load Phase 1 homepage snapshots."""
    if not os.path.exists(HOMEPAGE_FILE):
        print(f"ERROR: {HOMEPAGE_FILE} not found. Run scrape_homepages.py first.",
              flush=True)
        return pd.DataFrame()
    return pd.read_csv(HOMEPAGE_FILE, on_bad_lines='skip')


def load_keyword_urls():
    """Load keyword URLs from find_product_urls.py as fallback."""
    if not os.path.exists(KEYWORD_URLS_FILE):
        return {}
    df = pd.read_csv(KEYWORD_URLS_FILE)
    # Build lookup: (company, product) -> best keyword URL
    fallback = {}
    kw1 = df[df['url_type'] == 'keyword_1']
    for _, row in kw1.iterrows():
        key = (row['company_name'], row['product_name'])
        url = row.get('discovered_url', '')
        if pd.notna(url) and url:
            fallback[key] = url
    return fallback


def load_fintechs_websites(data_dir):
    """Load company -> website mapping."""
    fintechs = pd.read_csv(
        os.path.join(data_dir, "Data_cleaned", "banking_embedded_fintechs_unique.csv"))
    website_map = {}
    for _, row in fintechs.iterrows():
        name = str(row['fintechname']).strip().lower()
        website = str(row['fintechwebsite']).strip().rstrip('/')
        website_map[name] = website
    return website_map


def get_website(company, website_map):
    key = company.strip().lower()
    if key in website_map:
        return website_map[key]
    key_nospace = key.replace(' ', '')
    for k, v in website_map.items():
        if k.replace(' ', '') == key_nospace:
            return v
    return None


# ── Product URL discovery from homepage snapshots ────────────────────────────

def discover_product_url_from_homepages(company, product_name, year_launched,
                                         homepage_df):
    """
    Search homepage snapshots for product mentions.
    Returns list of discovered URLs (from link extraction), or empty list.
    """
    # Get all homepage snapshots for this company (don't restrict by
    # year_launched — launch dates may be inaccurate, and product pages
    # may exist before the recorded launch)
    company_snaps = homepage_df[
        (homepage_df['company_name'] == company) &
        (homepage_df['text_length'] > 0)
    ].sort_values(['target_year', 'target_month'])

    if company_snaps.empty:
        return [], "No homepage snapshots"

    # Build search tokens (strips parenthetical descriptions + company name)
    tokens = product_name_tokens(product_name, company)

    if not tokens:
        return [], "No searchable tokens in product name"

    # Search ALL homepage snapshots and pick the best match across them
    best_links = []  # (link, score, snap_label)
    stemmed_tokens = [_simple_stem(t) for t in tokens]

    for _, snap in company_snaps.iterrows():
        html_path = snap.get('html_path', '')
        if not html_path or not os.path.exists(html_path):
            continue

        with open(html_path, 'r', encoding='utf-8',
                  errors='replace') as fh:
            html = fh.read()

        html_lower = html.lower()
        if not any(t in html_lower for t in tokens):
            continue

        # Product mentioned — extract links and score them
        website = str(snap['company_website']) if pd.notna(snap['company_website']) else ''
        if not website:
            continue
        parsed = urlparse(
            website if website.startswith('http')
            else f'https://{website}')
        base_url = f"{parsed.scheme}://{parsed.netloc}/"
        links = extract_links(html, base_url)

        for link in links:
            path_lower = link["path"].lower().replace('-', ' ').replace('_', ' ')
            anchor_lower = link["anchor_text"].lower()
            combined = path_lower + ' ' + anchor_lower
            combined_words = re.split(r'[\s/\-_]+', combined)
            stemmed_words = {_simple_stem(w) for w in combined_words if w}
            score = sum(1 for st in stemmed_tokens if st in stemmed_words)
            if score > 0:
                snap_label = (f"{snap['target_year']}-"
                              f"{int(snap['target_month']):02d}")
                best_links.append((link, score, snap_label))

    if best_links:
        # Sort by score descending; return all unique matches
        best_links.sort(key=lambda x: -x[1])
        # Deduplicate by path, keep best score
        seen_paths = set()
        result_urls = []
        result_label = best_links[0][2]
        for link, score, label in best_links:
            if link["path"] not in seen_paths:
                seen_paths.add(link["path"])
                result_urls.append(link["href"])
        return result_urls, f"Found on homepage {result_label}"

    return [], "Product not mentioned on homepage"


def discover_terms_urls_from_homepages(company, homepage_df):
    """
    Extract terms/legal page URLs from homepage snapshots, indexed by time.

    Returns dict mapping (year, month) -> list of terms URLs found on that
    snapshot.  This lets the scraper use the correct terms URLs for each
    period (URLs may change over time).
    """
    company_snaps = homepage_df[
        (homepage_df['company_name'] == company) &
        (homepage_df['num_terms_links'] > 0)
    ].sort_values(['target_year', 'target_month'])

    terms_by_period = {}  # (year, month) -> [url, ...]
    for _, snap in company_snaps.iterrows():
        terms_json = snap.get('terms_link_urls', '')
        if not (pd.notna(terms_json) and terms_json):
            continue
        try:
            entries = json.loads(terms_json)
        except json.JSONDecodeError:
            continue

        urls = []
        website = snap['company_website']
        parsed = urlparse(
            website if website.startswith('http')
            else f'https://{website}')
        base = f"{parsed.scheme}://{parsed.netloc}"
        for entry in entries:
            path = entry.get('path', '')
            if path:
                urls.append(f"{base}{path}")
        if urls:
            terms_by_period[(int(snap['target_year']),
                             int(snap['target_month']))] = urls

    return terms_by_period


# ── Scraping a single URL across time ────────────────────────────────────────

def scrape_url_over_time(company, product, start_year, url, page_type,
                         source, is_fin, writer, done, f, end_year,
                         year_launched=None, latest_only=False):
    """
    Scrape a single URL over time using a single CDX query to discover
    all available snapshots, then fetching one per 6-month window.

    If latest_only=True, only fetch the most recent snapshot.
    """
    if year_launched is None:
        year_launched = start_year
    scraped_count = 0
    sublinks_to_follow = []

    # ── Step 1: One CDX query to get all available snapshots ──────────
    all_snapshots = find_all_snapshots(url, from_year=start_year,
                                       to_year=end_year)
    time.sleep(CDX_DELAY)

    if not all_snapshots:
        # No snapshots at all — record one "No snapshot" row
        ck = (url, start_year, MONTHS[0])
        if ck not in done:
            writer.writerow({
                "company_name": company,
                "product_name": product,
                "year_launched": year_launched,
                "is_financial": is_fin,
                "page_type": page_type,
                "source": source,
                "target_url": url,
                "target_year": start_year,
                "target_month": MONTHS[0],
                "snapshot_timestamp": "",
                "snapshot_date": "",
                "page_title": "",
                "text_length": 0,
                "html_path": "",
                "text_preview": "No snapshots in Wayback",
            })
            done.add(ck)
        f.flush()
        save_checkpoint(CHECKPOINT_FILE, done)
        return 0, []

    # ── Step 2: Select which snapshots to fetch ──────────────────────
    selected = select_snapshots(all_snapshots, months=MONTHS,
                                latest_only=latest_only)

    first_yr = int(all_snapshots[0][0][:4])
    last_yr = int(all_snapshots[-1][0][:4])
    print(f"      CDX: {len(all_snapshots)} snapshots ({first_yr}-{last_yr})"
          f", selected {len(selected)} to fetch", flush=True)

    # ── Step 3: Fetch each selected snapshot ─────────────────────────
    for yr, mo, ts, snap_url in selected:
        ck = (url, yr, mo)
        if ck in done:
            continue

        # Fetch
        html, final_url = fetch_wayback_page(ts, snap_url)
        time.sleep(FETCH_DELAY)

        if not html:
            writer.writerow({
                "company_name": company,
                "product_name": product,
                "year_launched": year_launched,
                "is_financial": is_fin,
                "page_type": page_type,
                "source": source,
                "target_url": url,
                "target_year": yr,
                "target_month": mo,
                "snapshot_timestamp": ts,
                "snapshot_date": ts[:8],
                "page_title": "",
                "text_length": 0,
                "html_path": "",
                "text_preview": "Fetch failed",
            })
            done.add(ck)
            continue

        # Detect JS-only pages and re-fetch with Playwright
        if is_js_only_page(html):
            print(f"        JS-only page detected — re-fetching with Playwright",
                  flush=True)
            pw_html, pw_url = fetch_wayback_page_playwright(ts, snap_url)
            if pw_html:
                html = pw_html
                final_url = pw_url
            else:
                print(f"        Playwright re-fetch failed — keeping original",
                      flush=True)

        # Parse
        title = extract_title(html)
        text = extract_text(html)
        safe_product = re.sub(r'[^a-zA-Z0-9_-]', '_', product)[:50]
        url_slug = re.sub(r'[^a-zA-Z0-9_-]', '_',
                          urlparse(url).path.strip('/'))[:60]
        label = f"{safe_product}__{page_type}__{url_slug}"
        html_path = save_html("products", company, label, ts, html)
        scraped_count += 1

        writer.writerow({
            "company_name": company,
            "product_name": product,
            "year_launched": year_launched,
            "is_financial": is_fin,
            "page_type": page_type,
            "source": source,
            "target_url": url,
            "target_year": yr,
            "target_month": mo,
            "snapshot_timestamp": ts,
            "snapshot_date": ts[:8],
            "page_title": title[:200],
            "text_length": len(text),
            "html_path": html_path,
            "text_preview": text[:500].replace('\n', ' '),
        })

        done.add(ck)

        # On first successful fetch, look for sublinks to follow
        if scraped_count == 1:
            parsed_base = urlparse(url)
            base_url = f"{parsed_base.scheme}://{parsed_base.netloc}/"
            page_links = extract_links(html, base_url)
            for link in page_links:
                if is_product_detail_or_terms_link(link):
                    sub_type = ("terms_sublink" if any(
                        p in link["path"].lower()
                        for p in ['/terms', '/legal', '/disclos',
                                  '/agreement', '/privacy', '/tos']
                    ) else "product_sublink")
                    sublinks_to_follow.append(
                        (link["href"], sub_type,
                         f"Sublink from {page_type}: "
                         f"{link['anchor_text'][:50]}"))

        # Periodic flush
        if scraped_count % 5 == 0:
            f.flush()
            save_checkpoint(CHECKPOINT_FILE, done)

    f.flush()
    save_checkpoint(CHECKPOINT_FILE, done)
    return scraped_count, sublinks_to_follow


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Scrape product pages from Wayback Machine")
    parser.add_argument("--test", type=int, default=0,
                        help="Limit to first N products (0 = all)")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh (ignore checkpoint, overwrite output)")
    parser.add_argument("--data-dir", type=str, default=".",
                        help="Directory containing the CSV files")
    parser.add_argument("--end-year", type=int, default=END_YEAR)
    parser.add_argument("--no-sublinks", action="store_true",
                        help="Skip following sublinks one level deep")
    parser.add_argument("--financial-only", action="store_true",
                        help="Only process financial products (payments, lending, etc.)")
    parser.add_argument("--latest-only", action="store_true",
                        help="Only scrape the most recent snapshot for each URL")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("Loading data...", flush=True)
    products = load_timeline(args.data_dir)
    homepage_df = load_homepage_snapshots()
    keyword_fallback = load_keyword_urls()
    website_map = load_fintechs_websites(args.data_dir)

    print(f"  {len(products)} products across "
          f"{products['company_name'].nunique()} companies", flush=True)
    print(f"  {len(homepage_df)} homepage snapshots loaded", flush=True)
    print(f"  {len(keyword_fallback)} keyword fallback URLs loaded", flush=True)

    # Filter to companies that have homepage data (from Phase 1)
    if len(homepage_df) > 0:
        hp_companies = set(homepage_df['company_name'].unique())
        products = products[products['company_name'].isin(hp_companies)].copy()
        print(f"  Filtered to {products['company_name'].nunique()} companies "
              f"with homepage data ({len(products)} products)", flush=True)

    if args.financial_only:
        products = products[products.apply(is_financial_product, axis=1)].copy()
        print(f"  Financial-only: {len(products)} products across "
              f"{products['company_name'].nunique()} companies", flush=True)

    if args.test > 0:
        test_products = products[['company_name', 'product_name']].drop_duplicates().head(args.test)
        products = products.merge(test_products, on=['company_name', 'product_name'])
        print(f"  TEST MODE: {len(products)} products", flush=True)

    # Checkpoint — always resume unless --fresh
    resume = not args.fresh
    done = load_checkpoint(CHECKPOINT_FILE) if resume else set()
    if done:
        print(f"  Resuming: {len(done)} tasks already done", flush=True)

    file_exists = os.path.exists(OUTPUT_FILE) and resume
    mode = 'a' if file_exists else 'w'

    with open(OUTPUT_FILE, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, escapechar='\\', quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer.writeheader()

        total_scraped = 0
        current_company = None

        # Pre-compute terms URLs per company (once)
        company_terms_cache = {}
        # Track which companies have already had their terms scraped
        company_terms_done = set()
        # Track URLs already scraped this run (avoids redundant fetches
        # when multiple products share the same URL or sublinks)
        urls_scraped_this_run = set()

        for idx, (_, row) in enumerate(products.iterrows()):
            company = row['company_name']
            product = row['product_name']
            year_launched = int(row['year_launched'])
            is_fin = is_financial_product(row)

            # Print progress
            if company != current_company:
                current_company = company
                print(f"\n{'='*60}", flush=True)
                print(f"  {company}", flush=True)

            print(f"\n  [{idx+1}/{len(products)}] {product} ({year_launched})"
                  f" {'[FINANCIAL]' if is_fin else ''}", flush=True)

            # ── Step 1: Discover product URL from homepages ──────────────
            product_urls, discovery_note = discover_product_url_from_homepages(
                company, product, year_launched, homepage_df)

            urls_to_scrape = []  # (url, page_type, source)

            if product_urls:
                print(f"    Found on homepage: {product_urls[0][:70]}",
                      flush=True)
                urls_to_scrape.append(
                    (product_urls[0], "product_page", discovery_note))
            else:
                # Fall back to keyword URL
                key = (company, product)
                fallback_url = keyword_fallback.get(key)
                if fallback_url:
                    print(f"    Fallback to keyword_1: {fallback_url[:70]}",
                          flush=True)
                    urls_to_scrape.append(
                        (fallback_url, "keyword_fallback", "keyword_1 from CDX"))
                else:
                    print(f"    No URL found for this product", flush=True)

            # ── Step 2: Discover terms URLs (time-indexed) ─────────────
            terms_by_period = {}
            if is_fin:
                if company not in company_terms_cache:
                    company_terms_cache[company] = \
                        discover_terms_urls_from_homepages(company, homepage_df)
                terms_by_period = company_terms_cache[company]
                unique_terms = set()
                for urls in terms_by_period.values():
                    unique_terms.update(urls)
                if unique_terms:
                    print(f"    + {len(unique_terms)} unique terms URL(s) "
                          f"across {len(terms_by_period)} snapshots",
                          flush=True)

            # ── Step 3: Scrape each product URL over time ──────────────
            for url, page_type, source in urls_to_scrape:
                if url in urls_scraped_this_run:
                    print(f"    Skipping [{page_type}]: {url[:70]} (already scraped)",
                          flush=True)
                    continue
                urls_scraped_this_run.add(url)
                print(f"    Scraping [{page_type}]: {url[:70]}", flush=True)

                count, sublinks = scrape_url_over_time(
                    company, product, START_YEAR, url,
                    page_type, source, is_fin,
                    writer, done, f, args.end_year,
                    year_launched=year_launched,
                    latest_only=args.latest_only,
                )
                total_scraped += count
                print(f"      -> {count} snapshots", flush=True)

                # Follow sublinks one level deep
                if not args.no_sublinks and sublinks:
                    for sub_url, sub_type, sub_source in sublinks:
                        if sub_url in urls_scraped_this_run:
                            continue
                        urls_scraped_this_run.add(sub_url)
                        print(f"    Following sublink [{sub_type}]: "
                              f"{sub_url[:60]}", flush=True)
                        sub_count, _ = scrape_url_over_time(
                            company, product, START_YEAR, sub_url,
                            sub_type, sub_source, is_fin,
                            writer, done, f, args.end_year,
                            year_launched=year_launched,
                            latest_only=args.latest_only,
                        )
                        total_scraped += sub_count
                        print(f"      -> {sub_count} snapshots", flush=True)

            # ── Step 4: Scrape terms pages per period ──────────────────
            # Terms are per-company, not per-product — only scrape once
            # per company to avoid redundant requests.
            if terms_by_period and company not in company_terms_done:
                company_terms_done.add(company)
                now_dt = datetime.now()
                sorted_periods = sorted(terms_by_period.keys())

                if args.latest_only:
                    # Just scrape the most recent version of each unique
                    # terms URL once (using the latest homepage snapshot's
                    # terms URLs)
                    latest_period = sorted_periods[-1]
                    for turl in terms_by_period[latest_period]:
                        if turl in urls_scraped_this_run:
                            continue
                        urls_scraped_this_run.add(turl)
                        print(f"    Scraping [terms_page]: {turl[:70]} "
                              f"(latest)", flush=True)
                        count, _ = scrape_url_over_time(
                            company, product, args.end_year, turl,
                            "terms_page",
                            "Terms from homepage",
                            is_fin,
                            writer, done, f, args.end_year,
                            latest_only=True,
                        )
                        total_scraped += count
                        print(f"      -> {count} snapshots", flush=True)
                else:
                    # Build (year, month) -> [terms URLs] for the full range
                    # Assign each scrape period the terms URLs from the closest
                    # homepage snapshot
                    terms_url_to_years = {}  # url -> set of (yr, mo)
                    for yr in range(START_YEAR, args.end_year + 1):
                        for mo in MONTHS:
                            if yr > now_dt.year or (yr == now_dt.year
                                                    and mo > now_dt.month):
                                continue
                            best_period = min(
                                sorted_periods,
                                key=lambda p: abs(
                                    (p[0] - yr) * 12 + (p[1] - mo)))
                            for turl in terms_by_period[best_period]:
                                terms_url_to_years.setdefault(turl, set())
                                terms_url_to_years[turl].add((yr, mo))

                    # Scrape each terms URL for the years it was current
                    for turl, periods in terms_url_to_years.items():
                        yr_min = min(p[0] for p in periods)
                        yr_max = max(p[0] for p in periods)
                        print(f"    Scraping [terms_page]: {turl[:70]} "
                              f"({yr_min}-{yr_max})", flush=True)
                        count, _ = scrape_url_over_time(
                            company, product, yr_min, turl,
                            "terms_page",
                            "Terms from homepage",
                            is_fin,
                            writer, done, f, yr_max,
                        )
                        total_scraped += count
                        print(f"      -> {count} snapshots", flush=True)

    save_checkpoint(CHECKPOINT_FILE, done)

    print(f"\n{'='*60}", flush=True)
    print(f"Done! Results saved to {OUTPUT_FILE}", flush=True)
    print(f"  Total snapshots scraped: {total_scraped}", flush=True)

    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        print(f"\n  Total rows: {len(df)}", flush=True)
        print(f"\n  By page_type:", flush=True)
        print(df['page_type'].value_counts().to_string(), flush=True)
        print(f"\n  By source:", flush=True)
        print(df['source'].value_counts().to_string(), flush=True)
        has_content = df['text_length'] > 0
        print(f"\n  With content: {has_content.sum()} "
              f"({has_content.mean()*100:.0f}%)", flush=True)


if __name__ == "__main__":
    main()
