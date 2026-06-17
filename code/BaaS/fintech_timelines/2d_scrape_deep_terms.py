"""
scrape_deep_terms.py
====================
Post-processing script that scans already-scraped terms/legal pages for
links to banking partner documents, deposit agreements, cardholder
agreements, lender disclosures, and other regulatory documents.

Run this AFTER fix_js_pages.py so that JS-rendered pages have their
full link content available.

These documents reveal BaaS banking partner relationships (Cross River,
Celtic, Evolve, Stride, Lead Bank, etc.) and the specific agreements
governing the fintech's products.

Usage:
    python scrape_deep_terms.py              # Scan and report discoverable links
    python scrape_deep_terms.py --scrape     # Scrape discovered links
    python scrape_deep_terms.py --scrape --dry  # Show what would be scraped
"""

import argparse
import csv
import json
import os
import re
import time
from urllib.parse import urlparse

from bs4 import BeautifulSoup

import importlib
_wu = importlib.import_module("2_wayback_utils")
OUTPUT_DIR, FETCH_DELAY, CDX_DELAY = _wu.OUTPUT_DIR, _wu.FETCH_DELAY, _wu.CDX_DELAY
find_all_snapshots, select_snapshots = _wu.find_all_snapshots, _wu.select_snapshots
fetch_wayback_page, fetch_wayback_pdf = _wu.fetch_wayback_page, _wu.fetch_wayback_pdf
is_js_only_page, fetch_wayback_page_playwright = _wu.is_js_only_page, _wu.fetch_wayback_page_playwright
is_pdf_url, extract_text_from_pdf, save_pdf_as_text = _wu.is_pdf_url, _wu.extract_text_from_pdf, _wu.save_pdf_as_text
extract_text, extract_title, save_html = _wu.extract_text, _wu.extract_title, _wu.save_html

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "product_pages.csv")
HTML_DIR = os.path.join(OUTPUT_DIR, "html", "products")
DEEP_TERMS_DONE_FILE = os.path.join(OUTPUT_DIR, "intermediate_files", "deep_terms_done.json")
DEEP_TERMS_CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "intermediate_files", "deep_terms_checkpoints.json")
DEEP_TERMS_SCAN_CACHE = os.path.join(OUTPUT_DIR, "intermediate_files", "deep_terms_scan_cache.json")

FIELDNAMES = [
    "company_name", "product_name", "year_launched",
    "is_financial", "page_type", "source",
    "target_url", "target_year", "target_month",
    "snapshot_timestamp", "snapshot_date", "page_title",
    "text_length", "html_path", "text_preview",
]

# ── Link matching patterns ────────────────────────────────────────────────────

# Keywords in anchor text or URL path that indicate a relevant deep link
DEEP_TERMS_KEYWORDS = [
    'agreement', 'disclosure', 'lender',
    'deposit account', 'deposit agreement',
    'cardholder', 'authorization', 'e-sign', 'consent',
    'bank partner',
    'license', 'fdic', 'program agreement',
]
# Note: "privacy policy" / "privacy notice" removed from keywords —
# bank partner privacy pages are already caught by BANK_PARTNER_DOMAINS.

# Known BaaS bank partner domains — links to these are always relevant
BANK_PARTNER_DOMAINS = [
    'crossriver.com', 'cross-river.com',
    'celticbank.com', 'celticwebback.wpengine.com',
    'getevolved.com', 'evolvebank.com',
    'stridebank.com',
    'lead.bank',
    'lincolnsavingsbank.com',
    'choicefi.com',
    'nbkc.com',
    'bancorpsouth.com', 'cadencebank.com',
    'sutton-bank.com', 'suttonbank.com',
    'bluevine.com',
    'greenarrowcap.com',
    'coastal-community.com',
    'blueridgebank.com',
    'columnbank.com',
    'piermont.bank',
    'thebancroftgroup.com',
    'synovus.com',
]

# API disclosure paths (e.g. affirm.com/api/v2/disclosures/...)
API_DISCLOSURE_PATTERN = re.compile(r'/api/.*/disclos', re.IGNORECASE)

# Reject patterns — skip these even if they match keywords
REJECT_PATTERNS = [
    '/careers', '/blog', '/news', '/press', '/about-us',
    '/shopping/', '/business/solutions', '/download',
    '/user/signin', '/user/signup', '/help',
    '/support/',   # FAQ / help articles (e.g. "how do I deposit a check")
]

# Domains to never follow — government/reference sites, not partner documents
REJECT_DOMAINS = [
    'fdic.gov',
]


def strip_wayback_prefix(url):
    """Strip Wayback Machine URL rewriting prefix to get the original URL."""
    # Matches various Wayback URL formats:
    #   https://web.archive.org/web/20260226210530/https://...
    #   /web/20260226210530/https://...
    #   /web/20260226210530mp_/https://...  (mp_ = media prefix)
    #   https://web.archive.org/web/20260226210530id_/https://...
    m = re.match(
        r'(?:https?://web\.archive\.org)?/web/(\d+)(?:[a-z_]*)/(https?://.+)',
        url)
    if m:
        return m.group(2), m.group(1)  # (original_url, timestamp)
    return url, None


def is_deep_terms_link(href, anchor_text):
    """Check if a link points to a banking partner / regulatory document."""
    anchor_lower = anchor_text.lower()
    href_lower = href.lower()

    # Reject generic nav links
    for p in REJECT_PATTERNS:
        if p in href_lower:
            return False

    # Reject government/reference domains
    try:
        domain = urlparse(href).netloc.lower().replace('www.', '')
        if any(rd in domain for rd in REJECT_DOMAINS):
            return False
    except Exception:
        pass

    # Skip anchors (same-page links)
    if href.startswith('#') or (href.count('#') and not '://' in href):
        return False

    # Check for bank partner domains — skip pure privacy pages unless
    # bundled with terms (e.g. lead.bank/privacy-and-terms is OK)
    try:
        domain = urlparse(href).netloc.lower().replace('www.', '')
        if any(bd in domain for bd in BANK_PARTNER_DOMAINS):
            combined = anchor_lower + ' ' + href_lower
            is_privacy = 'privacy' in combined
            has_terms = any(w in combined for w in [
                'term', 'agreement', 'disclosure', 'consent'])
            if is_privacy and not has_terms:
                return False
            return True
    except Exception:
        pass

    # Check for API disclosure paths
    if API_DISCLOSURE_PATTERN.search(href):
        return True

    # Check keywords in anchor text or URL
    combined = anchor_lower + ' ' + href_lower
    if any(kw in combined for kw in DEEP_TERMS_KEYWORDS):
        # But require it to look like a document, not just nav
        # Skip if anchor text is too short/generic
        if len(anchor_text.strip()) < 5 and 'lender' not in href_lower:
            return False
        return True

    return False


def extract_deep_links(html_path, base_domain=None):
    """Extract deep terms links from an already-scraped HTML file.
    base_domain is used to resolve relative URLs (e.g. 'https://www.affirm.com').
    """
    with open(html_path, 'r', encoding='utf-8', errors='replace') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')
    links = []
    seen = set()

    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        anchor = a.get_text(strip=True)

        if not href or href.startswith(('javascript:', 'mailto:', 'tel:')):
            continue
        # Skip same-page anchors
        if href.startswith('#'):
            continue

        # Strip Wayback prefix
        real_url, wb_timestamp = strip_wayback_prefix(href)

        # Resolve relative URLs
        parsed = urlparse(real_url)
        if not parsed.scheme and not parsed.netloc:
            if base_domain:
                real_url = base_domain.rstrip('/') + '/' + real_url.lstrip('/')
            else:
                continue  # can't resolve without a base domain

        if not urlparse(real_url).netloc:
            continue

        if is_deep_terms_link(real_url, anchor):
            # Deduplicate by URL (ignore fragments)
            clean_url = real_url.split('#')[0]
            if clean_url in seen:
                continue
            seen.add(clean_url)

            links.append({
                'url': clean_url,
                'anchor_text': anchor[:100],
                'wayback_timestamp': wb_timestamp,
            })

    return links


def load_deep_terms_done():
    """Load set of URLs already scraped in deep terms pass."""
    if not os.path.exists(DEEP_TERMS_DONE_FILE):
        return set()
    with open(DEEP_TERMS_DONE_FILE, 'r') as f:
        return set(json.load(f))


def save_deep_terms_done(done_set):
    with open(DEEP_TERMS_DONE_FILE, 'w') as f:
        json.dump(sorted(done_set), f, indent=2)


def load_deep_terms_checkpoints():
    """Load set of (url|year|month) checkpoints for time-series mode."""
    if not os.path.exists(DEEP_TERMS_CHECKPOINT_FILE):
        return set()
    with open(DEEP_TERMS_CHECKPOINT_FILE, 'r') as f:
        return set(json.load(f))


def save_deep_terms_checkpoints(ck_set):
    with open(DEEP_TERMS_CHECKPOINT_FILE, 'w') as f:
        json.dump(sorted(ck_set), f, indent=2)


def scan_all_deep_links():
    """Scan all terms/legal HTML files and extract deep links."""
    # Load CSV to know which files are terms pages and get company info
    if not os.path.exists(OUTPUT_FILE):
        print(f"ERROR: {OUTPUT_FILE} not found")
        return []

    import pandas as pd
    df = pd.read_csv(OUTPUT_FILE)

    # Filter to terms-related page types
    terms_types = ['terms_page', 'terms_sublink']
    terms_rows = df[df['page_type'].isin(terms_types)]

    # Also check product pages — some have bank partner links too
    all_rows = df[df['html_path'].notna() & (df['html_path'] != '')]

    results = []  # (company, product, source_file, link_info)
    seen_urls = set()
    total = len(all_rows)

    for i, (_, row) in enumerate(all_rows.iterrows()):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  Scanning file {i+1}/{total}...", flush=True)
        hp = row['html_path']
        if not os.path.exists(hp):
            continue

        # Derive base domain from the target URL for resolving relative links
        target_url = row.get('target_url', '')
        base_domain = None
        if target_url:
            p = urlparse(target_url if '://' in str(target_url)
                         else f'https://{target_url}')
            base_domain = f"{p.scheme}://{p.netloc}"

        links = extract_deep_links(hp, base_domain=base_domain)
        for link in links:
            url = link['url']
            if url in seen_urls:
                continue
            seen_urls.add(url)

            results.append({
                'company': row['company_name'],
                'product': row['product_name'],
                'year_launched': row.get('year_launched', ''),
                'is_financial': row.get('is_financial', True),
                'source_file': os.path.basename(hp),
                'url': url,
                'anchor_text': link['anchor_text'],
                'wayback_timestamp': link['wayback_timestamp'],
            })

    return results


def scrape_deep_terms(dry_run=False, latest_only=False):
    """Scrape discovered deep terms links.

    If latest_only=True, fetch only the most recent snapshot per URL.
    Otherwise, fetch one snapshot per 6-month window (Jan/Jul), matching
    the Phase 2 time-series approach.
    """
    already_done = load_deep_terms_done()
    if already_done:
        print(f"  {len(already_done)} URL/time checkpoints already done "
              f"(skipping)", flush=True)

    # Use cached scan results if available, otherwise scan and cache
    if os.path.exists(DEEP_TERMS_SCAN_CACHE):
        print("Loading cached deep terms scan...", flush=True)
        with open(DEEP_TERMS_SCAN_CACHE, 'r') as f:
            all_links = json.load(f)
        print(f"  {len(all_links)} links from cache", flush=True)
    else:
        print("Scanning for deep terms links...", flush=True)
        all_links = scan_all_deep_links()
        with open(DEEP_TERMS_SCAN_CACHE, 'w') as f:
            json.dump(all_links, f, indent=2)
        print(f"  Scan cached to {DEEP_TERMS_SCAN_CACHE}", flush=True)

    # Filter out URLs where ALL possible snapshots are already done.
    # For latest_only we just check the URL; for time-series we check
    # per (url, year, month) inside the loop.
    to_scrape = [l for l in all_links if l['url'] not in already_done]

    print(f"\n  Total deep links found: {len(all_links)}")
    print(f"  Already checkpointed URLs: {len(already_done)}")
    print(f"  URLs to process: {len(to_scrape)}")
    print(f"  Mode: {'latest only' if latest_only else '6-month time-series'}")

    if not to_scrape:
        print("  Nothing to scrape!")
        return

    print()
    for link in to_scrape:
        print(f"  [{link['company']}] {link['anchor_text'][:60]}")
        print(f"    -> {link['url'][:100]}")

    if dry_run:
        print(f"\n  Dry run — would scrape {len(to_scrape)} URLs")
        return

    # For time-series mode, use (url, year, month) checkpoints stored
    # in a separate file so we can resume partially-scraped URLs.
    done_checkpoints = load_deep_terms_checkpoints()

    # Open CSV for appending
    file_exists = os.path.exists(OUTPUT_FILE)
    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, escapechar='\\', quoting=csv.QUOTE_MINIMAL)
        if not file_exists:
            writer.writeheader()

        scraped = 0
        failed = 0

        print(f"\nScraping {len(to_scrape)} deep terms links...\n")

        for link in to_scrape:
            url = link['url']
            company = link['company']
            product = link['product']
            anchor = link['anchor_text']

            print(f"  [{company}] {anchor[:50]}", flush=True)
            print(f"    URL: {url[:80]}", flush=True)

            # Try to find snapshots via CDX
            all_snapshots = find_all_snapshots(url, from_year=2005,
                                               to_year=2026)
            time.sleep(CDX_DELAY)

            if not all_snapshots:
                print(f"    No snapshots found in Wayback")
                writer.writerow({
                    "company_name": company,
                    "product_name": product,
                    "year_launched": link['year_launched'],
                    "is_financial": link['is_financial'],
                    "page_type": "deep_terms",
                    "source": f"Deep link: {anchor[:50]}",
                    "target_url": url,
                    "target_year": "",
                    "target_month": "",
                    "snapshot_timestamp": "",
                    "snapshot_date": "",
                    "page_title": "",
                    "text_length": 0,
                    "html_path": "",
                    "text_preview": "No snapshots in Wayback",
                })
                already_done.add(url)
                save_deep_terms_done(already_done)
                failed += 1
                continue

            # Select snapshots — latest or 6-month time-series
            selected = select_snapshots(all_snapshots,
                                        latest_only=latest_only)
            if not selected:
                already_done.add(url)
                save_deep_terms_done(already_done)
                failed += 1
                continue

            first_yr = int(all_snapshots[0][0][:4])
            last_yr = int(all_snapshots[-1][0][:4])
            print(f"    CDX: {len(all_snapshots)} snapshots "
                  f"({first_yr}-{last_yr}), selected {len(selected)} "
                  f"to fetch", flush=True)

            # Fetch each selected snapshot
            is_pdf = is_pdf_url(url)
            safe_product = re.sub(r'[^a-zA-Z0-9_-]', '_', product)[:50]
            safe_anchor = re.sub(r'[^a-zA-Z0-9_-]', '_', anchor)[:60]
            label = f"{safe_product}__deep_terms__{safe_anchor}"

            for yr, mo, ts, snap_url in selected:
                ck = f"{url}|{yr}|{mo}"
                if ck in done_checkpoints:
                    continue

                if is_pdf:
                    pdf_bytes, final_url = fetch_wayback_pdf(ts, snap_url)
                    time.sleep(FETCH_DELAY)

                    if not pdf_bytes:
                        print(f"    {ts[:8]}: PDF fetch failed")
                        done_checkpoints.add(ck)
                        save_deep_terms_checkpoints(done_checkpoints)
                        failed += 1
                        continue

                    title = os.path.basename(urlparse(url).path)
                    text = extract_text_from_pdf(pdf_bytes)
                    html_path = save_pdf_as_text("products", company,
                                                 label, ts, text)
                else:
                    html, final_url = fetch_wayback_page(ts, snap_url)
                    time.sleep(FETCH_DELAY)

                    if not html:
                        print(f"    {ts[:8]}: Fetch failed")
                        done_checkpoints.add(ck)
                        save_deep_terms_checkpoints(done_checkpoints)
                        failed += 1
                        continue

                    # Check if JS-only and re-fetch with Playwright
                    try:
                        js_only = is_js_only_page(html)
                    except Exception:
                        js_only = False
                    if js_only:
                        print(f"    {ts[:8]}: JS-only — re-fetching with "
                              f"Playwright", flush=True)
                        pw_html, _ = fetch_wayback_page_playwright(
                            ts, snap_url)
                        if pw_html:
                            html = pw_html

                    try:
                        title = extract_title(html)
                        text = extract_text(html)
                    except Exception:
                        print(f"    {ts[:8]}: Failed to parse HTML — skipping",
                              flush=True)
                        done_checkpoints.add(ck)
                        save_deep_terms_checkpoints(done_checkpoints)
                        failed += 1
                        continue
                    html_path = save_html("products", company, label,
                                          ts, html)

                writer.writerow({
                    "company_name": company,
                    "product_name": product,
                    "year_launched": link['year_launched'],
                    "is_financial": link['is_financial'],
                    "page_type": "deep_terms",
                    "source": f"Deep link: {anchor[:50]}",
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

                done_checkpoints.add(ck)
                save_deep_terms_checkpoints(done_checkpoints)
                f.flush()

                scraped += 1
                print(f"    {ts[:8]}: OK — {len(text)} chars", flush=True)

            # Mark whole URL as done
            already_done.add(url)
            save_deep_terms_done(already_done)

    print(f"\nDone! Scraped: {scraped}, Failed/skipped: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Scan and scrape deep terms/banking partner links")
    parser.add_argument("--scrape", action="store_true",
                        help="Scrape discovered links (without this, just scans)")
    parser.add_argument("--dry", action="store_true",
                        help="Show what would be scraped without doing it")
    parser.add_argument("--latest-only", action="store_true",
                        help="Fetch only the most recent snapshot per URL "
                             "(default: 6-month time-series)")
    parser.add_argument("--rescan", action="store_true",
                        help="Force fresh scan (ignore cached scan results)")
    args = parser.parse_args()

    if args.rescan and os.path.exists(DEEP_TERMS_SCAN_CACHE):
        os.remove(DEEP_TERMS_SCAN_CACHE)
        print("Cleared scan cache.", flush=True)

    if args.scrape:
        scrape_deep_terms(dry_run=args.dry,
                          latest_only=args.latest_only)
    else:
        already_done = load_deep_terms_done()
        print("Scanning for deep terms links...", flush=True)
        all_links = scan_all_deep_links()
        to_scrape = [l for l in all_links if l['url'] not in already_done]

        print(f"\n  Total deep links found: {len(all_links)}")
        print(f"  Already scraped: {len(already_done)}")
        print(f"  To scrape: {len(to_scrape)}")

        for link in to_scrape:
            print(f"\n  [{link['company']}] {link['anchor_text'][:60]}")
            print(f"    -> {link['url'][:100]}")

        if to_scrape:
            print(f"\n  Run with --scrape to fetch these")


if __name__ == "__main__":
    main()
