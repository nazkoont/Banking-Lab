"""
find_product_urls.py
====================
For each (company, product, year) in the fintech_timeline_batch files,
discover URLs where that product is described — via the Wayback Machine
CDX API, with DuckDuckGo fallback.

For each product, outputs MULTIPLE candidate rows:
  - url_type = "keyword_1", "keyword_2", "keyword_3": top CDX keyword matches
  - url_type = "common_path": best /products, /features, /solutions etc. page
  - url_type = "homepage": the company homepage
  - url_type = "ddg": DuckDuckGo search result (when CDX finds nothing)

Output: product_urls.csv

Usage:
    python find_product_urls.py              # Full run
    python find_product_urls.py --test 5     # First N companies
    python find_product_urls.py --resume     # Resume from checkpoint
    python find_product_urls.py --no-search  # Skip DuckDuckGo fallback
"""

import argparse
import csv
import glob
import json
import os
import re
import time
from urllib.parse import urlparse

import pandas as pd
import requests

# DuckDuckGo fallback (more reliable than googlesearch-python)
try:
    from ddgs import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False
    print("WARNING: ddgs not installed (pip install ddgs). "
          "Search fallback disabled.")

# Ensure working directory is the project root (parent of Code/)
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# ── Config ───────────────────────────────────────────────────────────────────
CDX_API = "https://web.archive.org/cdx/search/cdx"
CDX_DELAY = 3.0        # seconds between CDX requests
DDG_DELAY = 3.0        # seconds between DuckDuckGo searches
YEAR_WINDOW = 1        # search +/- this many years around launch
MAX_CDX_RESULTS = 5000 # cap CDX results per query
MAX_RETRIES = 3        # retry on transient failures
RETRY_BACKOFF = 10     # seconds base backoff on retry
TOP_K_KEYWORDS = 3     # keep top K keyword-matched URLs per product

OUTPUT_DIR = "Data_wayback"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "product_urls.csv")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "intermediate_files", "product_urls_checkpoint.json")

# Common product-page path patterns
COMMON_PATHS = [
    "/products", "/features", "/solutions", "/services",
    "/pricing", "/platform", "/about",
]

# ── HTTP session with retry ─────────────────────────────────────────────────
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (research-project; fintech-timeline-study)"
})


def cdx_request(params, timeout=30):
    """Make a CDX API request with retry + exponential backoff."""
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(CDX_API, params=params, timeout=timeout)
            if resp.status_code in (429, 502, 503, 504):
                wait = RETRY_BACKOFF * (2 ** attempt)
                print(f"      HTTP {resp.status_code} (attempt "
                      f"{attempt+1}/{MAX_RETRIES}). Waiting {wait}s...",
                      flush=True)
                time.sleep(wait)
                last_error = f"HTTP {resp.status_code}"
                continue
            resp.raise_for_status()
            text = resp.text.strip()
            if not text:
                return []
            return resp.json()
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"      {type(e).__name__} (attempt "
                  f"{attempt+1}/{MAX_RETRIES}). Waiting {wait}s...",
                  flush=True)
            time.sleep(wait)
            last_error = str(e)
        except json.JSONDecodeError as e:
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"      JSON decode error (attempt "
                  f"{attempt+1}/{MAX_RETRIES}). Waiting {wait}s...",
                  flush=True)
            time.sleep(wait)
            last_error = str(e)
        except requests.RequestException as e:
            return None, str(e)
    return None, f"Max retries exceeded ({last_error})"


# ── Helpers ──────────────────────────────────────────────────────────────────

def slugify(text):
    """Convert product name to likely URL slug components."""
    text = text.lower()
    text = re.sub(r'\([^)]*\)', '', text)
    for w in ['the', 'and', 'for', 'by', 'with', 'a', 'an']:
        text = re.sub(rf'\b{w}\b', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    return [t for t in tokens if len(t) > 1]


def extract_keywords(product_name, product_desc, company_name):
    """Extract search keywords from product name and description."""
    company_tokens = set(slugify(company_name))
    name_tokens = [t for t in slugify(product_name) if t not in company_tokens]

    if pd.notna(product_desc):
        first_sentence = str(product_desc).split('.')[0].lower()
        first_sentence = re.sub(r'[^a-z0-9\s]', '', first_sentence)
        desc_tokens = [t for t in first_sentence.split()
                       if len(t) > 3 and t not in company_tokens]
    else:
        desc_tokens = []

    return name_tokens, desc_tokens


def score_url(url_path, name_tokens, desc_tokens):
    """Score how well a URL path matches product keywords. Higher = better."""
    path_lower = (url_path.lower()
                  .replace('-', ' ').replace('_', ' ').replace('/', ' '))
    score = 0
    for token in name_tokens:
        if token in path_lower:
            score += 10
    for token in desc_tokens[:5]:
        if token in path_lower:
            score += 3
    # Boost for product-like paths
    for pattern in ['/products/', '/features/', '/solutions/', '/platform/',
                    '/what-we-offer/', '/services/']:
        if pattern in url_path.lower():
            score += 2
    # Penalty for transient/non-product pages
    for pattern in ['/blog/', '/news/', '/press/', '/careers/', '/legal/',
                    '/privacy', '/terms', '/cookie', '/login', '/signup',
                    '/wp-content/', '/wp-admin/', '.css', '.js', '.png',
                    '.jpg', '.pdf', '.xml', '.json',
                    # Article / press release / ephemeral content
                    '/articles/', '/article/', '/spark/', '/resources/',
                    '/insights/', '/press-release', '/announcement',
                    '/webinar', '/event/', '/demo/', '/semlps/',
                    '/lp/', '/landing/', '/campaign/']:
        if pattern in url_path.lower():
            score -= 20
    return score


# ── CDX queries ──────────────────────────────────────────────────────────────

def query_cdx_domain_full(domain, start_year=2005, end_year=2025):
    """
    Query Wayback CDX API for all HTML pages under `domain` across the
    full year range.  Queries one year at a time to avoid timeouts on
    large domains.
    Returns (all_rows, url_years, note_string).
    all_rows: list of (url, timestamp) for all unique URLs found.
    url_years: dict mapping url -> set of years it was seen in (persistence).
    """
    all_rows = []
    seen_urls = {}  # url -> first timestamp
    url_years = {}  # url -> set of years
    errors = 0
    years_ok = 0

    for yr in range(start_year, end_year + 1):
        params = {
            "url": domain,
            "matchType": "domain",
            "output": "json",
            "fl": "timestamp,original,statuscode,mimetype",
            "filter": ["statuscode:200", "mimetype:text/html"],
            "from": f"{yr}0101",
            "to": f"{yr}1231",
            "limit": MAX_CDX_RESULTS,
            "collapse": "urlkey",
        }

        print(f"        CDX {domain} year={yr} ...", end=" ", flush=True)
        resp = cdx_request(params, timeout=120)
        time.sleep(CDX_DELAY)

        if not isinstance(resp, list):
            # cdx_request returned (None, error_string) on failure
            err_msg = resp[1] if isinstance(resp, tuple) else "unknown error"
            errors += 1
            print(f"FAILED: {err_msg} (errors so far: {errors})", flush=True)
            continue
        if len(resp) <= 1:
            errors += 1
            print(f"empty (no archived pages this year)", flush=True)
            continue

        years_ok += 1
        new_urls = 0
        header = resp[0]
        for row in resp[1:]:
            entry = dict(zip(header, row))
            url = entry["original"]
            url_years.setdefault(url, set()).add(yr)
            if url not in seen_urls:
                seen_urls[url] = entry["timestamp"]
                all_rows.append((url, entry["timestamp"]))
                new_urls += 1
        print(f"{len(resp)-1} rows, {new_urls} new URLs "
              f"(total: {len(all_rows)})", flush=True)

    if not all_rows:
        return [], {}, f"No CDX results ({errors} year(s) failed)"

    return all_rows, url_years, (f"OK ({len(all_rows)} unique URLs, "
                                  f"{years_ok}/{end_year - start_year + 1} years)")


# Paths where persistence is meaningful (product pages that stay up = signal).
# Blog posts / press releases also persist forever, so persistence is NOT
# a useful signal for those — only boost URLs on product-like paths.
_PRODUCT_PATH_PATTERNS = (
    '/products/', '/features/', '/solutions/', '/platform/',
    '/what-we-offer/', '/services/', '/pricing/', '/tools/',
    '/business/', '/personal/', '/checking/', '/savings/',
    '/lending/', '/payments/', '/payroll/', '/insurance/',
)


def score_cdx_rows(all_rows, name_tokens, desc_tokens, url_years=None):
    """
    Score and rank CDX rows by product keyword relevance + persistence.
    Persistence bonus only applies to product-like paths (not blogs/press).
    """
    results = []
    for url, timestamp in all_rows:
        parsed = urlparse(url)
        path_lower = parsed.path.lower()
        score = score_url(parsed.path, name_tokens, desc_tokens)
        if score > 0 and url_years:
            # Only give persistence bonus to product-like paths
            is_product_path = any(p in path_lower
                                  for p in _PRODUCT_PATH_PATTERNS)
            if is_product_path:
                n_years = len(url_years.get(url, set()))
                score += n_years * 2
        if score > 0:
            results.append((url, score, timestamp))
    results.sort(key=lambda x: -x[1])
    return results


def find_homepage(all_rows, domain):
    """Find the homepage URL from CDX rows."""
    for url, timestamp in all_rows:
        parsed = urlparse(url)
        if parsed.path in ('/', '') and domain in parsed.netloc:
            return url, timestamp
    # Fallback: shortest URL is often the homepage
    if all_rows:
        by_len = sorted(all_rows, key=lambda x: len(x[0]))
        return by_len[0]
    return None, None


def filter_common_paths(cdx_rows, name_tokens):
    """
    Find hits on common paths like /products, /features, etc.
    Returns list of (url, score, timestamp).
    """
    product_slug = "-".join(name_tokens[:3]) if name_tokens else ""
    target_paths = set(COMMON_PATHS)
    if product_slug:
        target_paths.add(f"/{product_slug}")
        target_paths.add(f"/products/{product_slug}")

    hits = []
    for url, timestamp in cdx_rows:
        parsed = urlparse(url)
        path = parsed.path.rstrip('/')
        for target in target_paths:
            if (path.lower() == target.lower()
                    or path.lower().startswith(target.lower() + '/')):
                score = score_url(path, name_tokens, [])
                hits.append((url, max(score, 1), timestamp))
                break

    hits.sort(key=lambda x: -x[1])
    return hits


# ── DuckDuckGo fallback ─────────────────────────────────────────────────────

def ddg_search(company_name, product_name, domain):
    """Use DuckDuckGo to find the product URL."""
    if not HAS_DDG:
        return [], "DuckDuckGo not available"

    results_out = []

    # Site-restricted search first
    query = f'site:{domain} {product_name}'
    try:
        results = DDGS().text(query, max_results=5)
        for r in results:
            results_out.append((r['href'], f"ddg site-search: {query}"))
    except Exception:
        pass

    if results_out:
        return results_out, "OK"

    time.sleep(DDG_DELAY)

    # Broader search
    query = f'{company_name} {product_name}'
    try:
        results = DDGS().text(query, max_results=5)
        for r in results:
            results_out.append((r['href'], f"ddg broad: {query}"))
    except Exception as e:
        return [], f"DDG error: {e}"

    return results_out, "OK" if results_out else "No results"


# ── CDX cache (per domain, full year range) ──────────────────────────────────
_cdx_domain_cache = {}  # domain -> (all_rows, url_years, note)


def get_cdx_rows_cached(clean_domain, year=None):
    """
    Fetch all CDX rows for a domain (full 2005-2025 range), cached.
    The `year` parameter is accepted for API compatibility but ignored —
    the full range is always queried once and reused for all products.
    Returns (all_rows, url_years, note).
    """
    if clean_domain not in _cdx_domain_cache:
        print(f"      Querying CDX for full {clean_domain} domain "
              f"(2005-2025, one year at a time)...", flush=True)
        all_rows, url_years, note = query_cdx_domain_full(clean_domain)
        _cdx_domain_cache[clean_domain] = (all_rows, url_years, note)
        print(f"      {note}", flush=True)
    return _cdx_domain_cache[clean_domain]


# ── Main discovery logic ─────────────────────────────────────────────────────

def find_urls_for_product(company_name, product_name, product_desc,
                          year, domain):
    """
    Discover multiple candidate URLs for one product.
    Returns list of dicts, each with:
        url_type, discovered_url, discovery_method, match_score,
        wayback_timestamp, search_notes
    """
    parsed = urlparse(domain if domain.startswith('http') else f'https://{domain}')
    clean_domain = parsed.netloc or parsed.path
    clean_domain = clean_domain.replace('www.', '')

    name_tokens, desc_tokens = extract_keywords(product_name, product_desc,
                                                 company_name)
    rows_out = []

    # ── Strategy A: CDX domain search ────────────────────────────────────
    print(f"    [A] CDX domain search on {clean_domain}...", flush=True)
    all_rows, url_years, cdx_note = get_cdx_rows_cached(clean_domain, year)

    has_keyword_hits = False
    if all_rows:
        # Top K keyword matches (with persistence scoring)
        cdx_results = score_cdx_rows(all_rows, name_tokens, desc_tokens,
                                     url_years=url_years)
        seen_urls = set()
        rank = 0
        for url, score, ts in cdx_results:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            rank += 1
            if rank > TOP_K_KEYWORDS:
                break
            rows_out.append({
                "url_type": f"keyword_{rank}",
                "discovered_url": url,
                "discovery_method": "cdx_keyword",
                "match_score": score,
                "wayback_timestamp": ts,
                "search_notes": f"Rank {rank}/{len(cdx_results)} matches. "
                                f"{cdx_note}",
            })
            has_keyword_hits = True

        # Best common-path page (deduplicated against keyword hits)
        path_hits = filter_common_paths(all_rows, name_tokens)
        for url, score, ts in path_hits:
            if url not in seen_urls:
                rows_out.append({
                    "url_type": "common_path",
                    "discovered_url": url,
                    "discovery_method": "cdx_common_path",
                    "match_score": score,
                    "wayback_timestamp": ts,
                    "search_notes": urlparse(url).path,
                })
                break

        # Homepage
        hp_url, hp_ts = find_homepage(all_rows, clean_domain)
        if hp_url and hp_url not in seen_urls:
            rows_out.append({
                "url_type": "homepage",
                "discovered_url": hp_url,
                "discovery_method": "cdx_homepage",
                "match_score": 0,
                "wayback_timestamp": hp_ts,
                "search_notes": "Company homepage",
            })

    # ── Strategy B: DuckDuckGo fallback (only if CDX found nothing) ──────
    if not has_keyword_hits:
        print(f"    [B] DuckDuckGo fallback...", flush=True)
        ddg_results, ddg_note = ddg_search(company_name, product_name,
                                            clean_domain)
        time.sleep(DDG_DELAY)

        for i, (url, note) in enumerate(ddg_results[:3]):
            # Score DDG results with the same keyword scoring as CDX
            ddg_score = score_url(urlparse(url).path, name_tokens, desc_tokens)
            rows_out.append({
                "url_type": f"ddg_{i+1}",
                "discovered_url": url,
                "discovery_method": "ddg",
                "match_score": ddg_score,
                "wayback_timestamp": "",
                "search_notes": note,
            })

    # ── Strategy C: Homepage-only fallback (if we got nothing at all) ────
    if not rows_out:
        print(f"    [C] Homepage-only fallback...", flush=True)
        hp_params = {
            "url": clean_domain,
            "output": "json",
            "fl": "timestamp,original,statuscode",
            "filter": "statuscode:200",
            "from": f"{int(year) - YEAR_WINDOW}0101",
            "to": f"{int(year) + YEAR_WINDOW}1231",
            "limit": 1,
        }
        resp = cdx_request(hp_params)
        time.sleep(CDX_DELAY)
        if isinstance(resp, list) and len(resp) > 1:
            entry = dict(zip(resp[0], resp[1]))
            rows_out.append({
                "url_type": "homepage",
                "discovered_url": entry["original"],
                "discovery_method": "homepage_fallback",
                "match_score": 0,
                "wayback_timestamp": entry["timestamp"],
                "search_notes": "No product page found; homepage only",
            })
        else:
            rows_out.append({
                "url_type": "none",
                "discovered_url": "",
                "discovery_method": "not_found",
                "match_score": -1,
                "wayback_timestamp": "",
                "search_notes": f"No results. CDX: {cdx_note}",
            })

    return rows_out


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(data_dir):
    """Load master timeline and fintech website mappings."""
    timeline = pd.read_csv(os.path.join(data_dir, "Data_cleaned", "fintech_timelines_master.csv"))

    fintechs = pd.read_csv(
        os.path.join(data_dir, "Data_cleaned", "banking_embedded_fintechs_unique.csv"))

    website_map = {}
    for _, row in fintechs.iterrows():
        name = str(row['fintechname']).strip()
        website = str(row['fintechwebsite']).strip().rstrip('/')
        website_map[name.lower()] = website

    return timeline, website_map


def get_company_website(company_name, website_map):
    """Look up company website with case/space-insensitive matching."""
    key = company_name.strip().lower()
    if key in website_map:
        return website_map[key]
    key_nospace = key.replace(' ', '')
    for k, v in website_map.items():
        if k.replace(' ', '') == key_nospace:
            return v
    return None


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE, 'r') as f:
        return set(tuple(x) for x in json.load(f))


def save_checkpoint(done_set):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(list(done_set), f)


# ── Main ─────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "company_name", "product_name", "year_launched", "company_website",
    "url_type", "discovered_url", "discovery_method", "match_score",
    "wayback_timestamp", "search_notes",
]


def main():
    parser = argparse.ArgumentParser(
        description="Find product URLs via Wayback + DuckDuckGo")
    parser.add_argument("--test", type=int, default=0,
                        help="Run on first N companies only (0 = all)")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh (ignore checkpoint, overwrite output)")
    parser.add_argument("--no-search", action="store_true",
                        help="Skip DuckDuckGo fallback")
    parser.add_argument("--data-dir", type=str, default=".",
                        help="Directory containing the CSV files")
    args = parser.parse_args()

    if args.no_search:
        global HAS_DDG
        HAS_DDG = False

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data...", flush=True)
    timeline, website_map = load_data(args.data_dir)

    products = timeline[timeline['entry_type'] == 'Product'].copy()
    products = products[pd.to_numeric(products['year_launched'], errors='coerce') >= 2005].copy()
    print(f"Total products (2005+): {len(products)} across "
          f"{products['company_name'].nunique()} companies", flush=True)

    if args.test > 0:
        test_companies = sorted(products['company_name'].unique())[:args.test]
        products = products[products['company_name'].isin(test_companies)]
        print(f"TEST MODE: limited to {args.test} companies "
              f"({len(products)} products)", flush=True)

    # Always resume unless --fresh
    resume = not args.fresh
    done = load_checkpoint() if resume else set()
    if done:
        print(f"Resuming: {len(done)} products already processed", flush=True)

    file_exists = os.path.exists(OUTPUT_FILE) and resume
    mode = 'a' if file_exists else 'w'

    with open(OUTPUT_FILE, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()

        companies = sorted(products['company_name'].unique())
        total_products = len(products)
        processed = 0

        for ci, company in enumerate(companies):
            company_products = products[products['company_name'] == company]
            website = get_company_website(company, website_map)

            print(f"\n{'='*60}", flush=True)
            print(f"[{ci+1}/{len(companies)}] {company} "
                  f"({len(company_products)} products)", flush=True)

            if not website:
                print(f"  WARNING: No website found for {company}, skipping",
                      flush=True)
                for _, row in company_products.iterrows():
                    writer.writerow({
                        "company_name": company,
                        "product_name": row["product_name"],
                        "year_launched": row["year_launched"],
                        "company_website": "",
                        "url_type": "none",
                        "discovered_url": "",
                        "discovery_method": "no_website",
                        "match_score": -1,
                        "wayback_timestamp": "",
                        "search_notes": "Company not in fintechs file",
                    })
                    done.add((company, row["product_name"]))
                f.flush()
                save_checkpoint(done)
                continue

            print(f"  Website: {website}", flush=True)

            for _, row in company_products.iterrows():
                product = row["product_name"]
                year = row["year_launched"]
                key = (company, product)

                if key in done:
                    processed += 1
                    continue

                processed += 1
                print(f"\n  [{processed}/{total_products}] {product} ({year})",
                      flush=True)

                url_rows = find_urls_for_product(
                    company, product,
                    row.get("product_history_and_description", ""),
                    year, website,
                )

                for url_row in url_rows:
                    writer.writerow({
                        "company_name": company,
                        "product_name": product,
                        "year_launched": year,
                        "company_website": website,
                        **url_row,
                    })

                f.flush()
                done.add(key)
                save_checkpoint(done)

                # Print summary
                for r in url_rows:
                    tag = r["url_type"]
                    url = r["discovered_url"] or "NONE"
                    score = r["match_score"]
                    print(f"    [{tag:12s} score={score:>3}] {url[:70]}",
                          flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"Done! Results saved to {OUTPUT_FILE}", flush=True)
    print(f"Total products processed: {len(done)}", flush=True)

    # Summary
    results_df = pd.read_csv(OUTPUT_FILE)
    print(f"\nTotal rows: {len(results_df)}", flush=True)
    print(f"\nBy url_type:", flush=True)
    print(results_df['url_type'].value_counts().to_string(), flush=True)
    print(f"\nBy discovery_method:", flush=True)
    print(results_df['discovery_method'].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
