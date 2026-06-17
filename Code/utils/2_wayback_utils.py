"""
2_wayback_utils.py
==================
Shared utilities for Wayback Machine scraping pipeline.
Used by 2a_scrape_homepages.py through 2e_fix_js_pages.py.
"""

import json
import os
import re
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

try:
    import pdfplumber
    import io as _io
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

# Ensure working directory is the project root (parent of Code/)
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# ── Config ───────────────────────────────────────────────────────────────────
CDX_API = "https://web.archive.org/cdx/search/cdx"
WAYBACK_URL = "https://web.archive.org/web"

CDX_DELAY = 5.0
CDX_PREFIX_DELAY = 5.0   # prefix queries are heavier — give the API more breathing room
FETCH_DELAY = 3.0
MAX_RETRIES = 3
RETRY_BACKOFF = 15
FETCH_TIMEOUT = 30
CDX_PREFIX_TIMEOUT = 90  # prefix queries scan more records, need longer timeout

OUTPUT_DIR = "Data_wayback"

# Financial categories/subcategories that trigger terms scraping
FINANCIAL_SUBCATS = {
    'Payments', 'Money Transfer', 'Payments API', 'Bill Pay', 'Point-of-Sale',
    'Card Reader/Terminal',
    'Lending (Consumer)', 'Lending (Business)', 'Credit Cards', 'Cash Advance',
    'Credit Building',
    'Brokerage', 'Robo-Advisory', 'Wealth Management', 'Asset Management',
    'Crypto/Digital Assets', 'Insurance', 'Treasury Management',
    'Alternative Investments',
}
DEPOSITORY_SUBCATS = {
    'Banking (Consumer)', 'Banking (Business)', 'Banking (Deposit Accounts)',
    'Savings/Deposits', 'Debit Card',
}
FINANCIAL_CATS = {'Financial Services', 'Investment & Capital'}
ALL_FINANCIAL_SUBCATS = FINANCIAL_SUBCATS | DEPOSITORY_SUBCATS

# Terms-related URL patterns
TERMS_PATTERNS = [
    '/terms', '/legal', '/disclosures', '/agreements', '/privacy',
    '/deposit-agreement', '/cardholder-agreement', '/account-agreement',
    '/member-agreement', '/user-agreement', '/tos',
    '/bank-partner', '/fdic', '/licensing',
]

# ── HTTP session ─────────────────────────────────────────────────────────────
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (research-project; fintech-timeline-study)"
})


# ── CDX API ──────────────────────────────────────────────────────────────────

def cdx_request(params, timeout=30):
    """CDX API request with retry + backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(CDX_API, params=params, timeout=timeout)
            if resp.status_code == 429:
                wait = RETRY_BACKOFF * (2 ** attempt)
                print(f"        Rate limited (429). Waiting {wait}s...",
                      flush=True)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            text = resp.text.strip()
            if not text:
                return []
            return resp.json()
        except requests.exceptions.ConnectionError:
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"        Connection error (attempt {attempt+1}/"
                  f"{MAX_RETRIES}). Waiting {wait}s...", flush=True)
            time.sleep(wait)
        except requests.exceptions.Timeout:
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"        Timeout (attempt {attempt+1}/{MAX_RETRIES}). "
                  f"Waiting {wait}s...", flush=True)
            time.sleep(wait)
        except (requests.RequestException, json.JSONDecodeError):
            return []
    return []


def find_all_snapshots(url, from_year=2005, to_year=2025):
    """
    Get ALL available snapshots for a URL in one CDX query.
    Returns list of (timestamp, original_url) sorted by timestamp,
    or empty list if none found.
    """
    params = {
        "url": url,
        "output": "json",
        "fl": "timestamp,original,statuscode",
        "filter": "statuscode:200",
        "from": str(from_year),
        "to": str(to_year + 1),  # CDX 'to' is exclusive
        "collapse": "timestamp:6",  # collapse to one per month
    }
    resp = cdx_request(params, timeout=60)
    if not isinstance(resp, list) or len(resp) <= 1:
        return []

    results = []
    for row in resp[1:]:
        entry = dict(zip(resp[0], row))
        results.append((entry["timestamp"], entry["original"]))
    return results


def select_snapshots(all_snapshots, months=(1, 7), latest_only=False):
    """
    From a list of (timestamp, original_url) snapshots, select the best
    one per 6-month window (closest to Jan 1 or Jul 1 of each year).

    If latest_only=True, just return the most recent snapshot.

    Returns list of (year, month, timestamp, original_url).
    """
    if not all_snapshots:
        return []

    if latest_only:
        ts, orig = all_snapshots[-1]  # already sorted by timestamp
        try:
            yr = int(ts[:4])
            mo = int(ts[4:6])
        except (ValueError, IndexError):
            yr, mo = 2025, 1
        return [(yr, mo, ts, orig)]

    # Group snapshots into 6-month windows and pick closest to target date
    from datetime import datetime

    # Build target dates
    years = set()
    for ts, _ in all_snapshots:
        try:
            years.add(int(ts[:4]))
        except (ValueError, IndexError):
            continue

    if not years:
        return []

    min_yr, max_yr = min(years), max(years)
    targets = []
    for yr in range(min_yr, max_yr + 1):
        for mo in months:
            targets.append((yr, mo))

    # For each target, find the closest snapshot
    results = []
    used_timestamps = set()

    for target_yr, target_mo in targets:
        target_int = int(f"{target_yr}{target_mo:02d}01")
        best_ts = None
        best_orig = None
        best_diff = float('inf')

        for ts, orig in all_snapshots:
            try:
                snap_int = int(ts[:8])
            except (ValueError, IndexError):
                continue
            diff = abs(snap_int - target_int)
            if diff < best_diff:
                best_diff = diff
                best_ts = ts
                best_orig = orig

        if best_ts and best_ts not in used_timestamps:
            # Check proximity — skip if > 90 days away
            if best_diff <= 900000:  # ~90 days in YYYYMMDD int diff
                used_timestamps.add(best_ts)
                results.append((target_yr, target_mo, best_ts, best_orig))

    return results


def find_closest_snapshot(url, year, month):
    """
    Find the Wayback snapshot closest to the 1st of the given month/year.
    Returns (timestamp, original_url) or (None, None).
    """
    target = f"{year}{month:02d}01"
    params = {
        "url": url,
        "output": "json",
        "fl": "timestamp,original,statuscode",
        "filter": "statuscode:200",
        "limit": 1,
        "closest": target,
        "sort": "closest",
    }
    resp = cdx_request(params)
    if isinstance(resp, list) and len(resp) > 1:
        entry = dict(zip(resp[0], resp[1]))
        return entry["timestamp"], entry["original"]
    return None, None


def find_closest_snapshot_prefix(url, year, month):
    """
    Like find_closest_snapshot but uses prefix matching so that e.g.
    /about/terms also finds /about/terms202309, /about/terms202210, etc.

    Useful for terms/legal pages where the canonical URL redirects to a
    versioned URL that Wayback actually archives.

    Fetches all prefix-matching snapshots within a window around the
    target date and picks the closest one.

    Returns (timestamp, original_url) or (None, None).
    """
    target = f"{year}{month:02d}01"
    # Search a +/- 2 year window
    from_year = max(2005, year - 2)
    to_year = year + 2
    params = {
        "url": url,
        "output": "json",
        "fl": "timestamp,original,statuscode",
        "matchType": "prefix",
        "filter": "statuscode:200",
        "from": str(from_year),
        "to": str(to_year),
        "limit": 50,
    }
    resp = cdx_request(params, timeout=CDX_PREFIX_TIMEOUT)
    if not isinstance(resp, list) or len(resp) <= 1:
        return None, None

    # Find the snapshot closest to target date
    target_int = int(target)
    best_ts = None
    best_url = None
    best_diff = float('inf')
    for row in resp[1:]:
        entry = dict(zip(resp[0], row))
        ts = entry["timestamp"]
        diff = abs(int(ts[:8]) - target_int)
        if diff < best_diff:
            best_diff = diff
            best_ts = ts
            best_url = entry["original"]

    return best_ts, best_url


# ── Page fetching ────────────────────────────────────────────────────────────

def fetch_wayback_page(timestamp, url):
    """
    Fetch an archived page from Wayback Machine.
    Uses 'id_' flag to get raw page without Wayback toolbar.
    Returns (html_content, final_url) or (None, None).
    """
    wayback_url = f"{WAYBACK_URL}/{timestamp}id_/{url}"

    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(wayback_url, timeout=FETCH_TIMEOUT,
                               allow_redirects=True)
            if resp.status_code == 429:
                wait = RETRY_BACKOFF * (2 ** attempt)
                print(f"        Rate limited fetching page. "
                      f"Waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None, None
            resp.raise_for_status()
            return resp.text, resp.url
        except requests.exceptions.ConnectionError:
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"        Fetch connection error (attempt {attempt+1}/"
                  f"{MAX_RETRIES}). Waiting {wait}s...", flush=True)
            time.sleep(wait)
        except requests.exceptions.Timeout:
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"        Fetch timeout (attempt {attempt+1}/"
                  f"{MAX_RETRIES}). Waiting {wait}s...", flush=True)
            time.sleep(wait)
        except requests.RequestException:
            return None, None
    return None, None


def fetch_wayback_pdf(timestamp, url):
    """
    Fetch an archived PDF from the Wayback Machine.
    Returns (pdf_bytes, final_url) or (None, None).
    """
    wayback_url = f"{WAYBACK_URL}/{timestamp}id_/{url}"

    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(wayback_url, timeout=FETCH_TIMEOUT,
                               allow_redirects=True)
            if resp.status_code == 429:
                wait = RETRY_BACKOFF * (2 ** attempt)
                print(f"        Rate limited fetching PDF. "
                      f"Waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None, None
            resp.raise_for_status()
            return resp.content, resp.url
        except requests.exceptions.ConnectionError:
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"        Fetch connection error (attempt {attempt+1}/"
                  f"{MAX_RETRIES}). Waiting {wait}s...", flush=True)
            time.sleep(wait)
        except requests.exceptions.Timeout:
            wait = RETRY_BACKOFF * (2 ** attempt)
            print(f"        Fetch timeout (attempt {attempt+1}/"
                  f"{MAX_RETRIES}). Waiting {wait}s...", flush=True)
            time.sleep(wait)
        except requests.RequestException:
            return None, None
    return None, None


def extract_text_from_pdf(pdf_bytes):
    """Extract text from PDF bytes using pdfplumber."""
    if not HAS_PDFPLUMBER:
        return ""
    try:
        pdf = pdfplumber.open(_io.BytesIO(pdf_bytes))
        text = '\n'.join(p.extract_text() or '' for p in pdf.pages)
        pdf.close()
        return text.strip()
    except Exception as e:
        print(f"        PDF text extraction failed: {e}", flush=True)
        return ""


def save_pdf(subdir, company, label, timestamp, pdf_bytes):
    """Save PDF bytes to disk. Returns the relative file path."""
    html_dir = os.path.join(OUTPUT_DIR, "html", subdir, safe_dirname(company))
    os.makedirs(html_dir, exist_ok=True)
    filename = f"{safe_dirname(label)}_{timestamp}.pdf"
    filepath = os.path.join(html_dir, filename)
    with open(filepath, 'wb') as f:
        f.write(pdf_bytes)
    return filepath


def save_pdf_as_text(subdir, company, label, timestamp, text):
    """Save extracted PDF text as a .txt file. Returns the file path."""
    html_dir = os.path.join(OUTPUT_DIR, "html", subdir, safe_dirname(company))
    os.makedirs(html_dir, exist_ok=True)
    filename = f"{safe_dirname(label)}_{timestamp}.txt"
    filepath = os.path.join(html_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    return filepath


def is_pdf_url(url):
    """Check if a URL likely points to a PDF."""
    path = urlparse(url).path.lower()
    return path.endswith('.pdf')


# ── JS-only page detection & Playwright re-fetch ─────────────────────────────

# Minimum text length (after stripping scripts/styles) to consider a page
# as having real content.  Pages below this with JS indicators are re-fetched.
JS_MIN_TEXT_LENGTH = 500

JS_INDICATORS = [
    'javascript is required',
    'enable javascript',
    'javascript enabled',
    'this app works best with javascript',
    'you need to enable javascript',
    'please enable javascript',
    'requires javascript',
]


def is_js_only_page(html):
    """
    Detect whether an HTML page is a JS shell with no real content.
    Returns True if the page appears to need JavaScript rendering.
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Check <noscript> tags for JS-required messages
    for ns in soup.find_all('noscript'):
        ns_text = ns.get_text(strip=True).lower()
        if any(ind in ns_text for ind in JS_INDICATORS):
            return True

    # Strip scripts/styles, measure remaining text
    for tag in soup.find_all(['script', 'style', 'noscript', 'iframe']):
        tag.decompose()
    text = soup.get_text(separator=' ', strip=True)

    if len(text) < JS_MIN_TEXT_LENGTH:
        # Very little text — check for JS framework markers
        raw_lower = html[:5000].lower()
        framework_hints = [
            '__next',        # Next.js
            '__nuxt',        # Nuxt/Vue
            'id="root"',     # React
            'id="app"',      # Vue / generic SPA
            'ng-app',        # Angular
            'data-reactroot',
        ]
        if any(hint in raw_lower for hint in framework_hints):
            return True
        # Also flag if there are many <script> tags but almost no text
        script_count = html.lower().count('<script')
        if script_count >= 5 and len(text) < 200:
            return True

    return False


def fetch_wayback_page_playwright(timestamp, url, timeout=45000):
    """
    Fetch a Wayback page using a headless browser so JavaScript executes.
    Uses the normal Wayback URL (no id_ flag) so that archived JS bundles
    and resources are loaded from the Wayback Machine's rewritten URLs.
    Returns (html_content, final_url) or (None, None).
    """
    if not HAS_PLAYWRIGHT:
        print("        Playwright not installed — skipping JS re-fetch",
              flush=True)
        return None, None

    # Normal Wayback URL — Wayback rewrites resource URLs to archived copies
    wayback_url = f"{WAYBACK_URL}/{timestamp}/{url}"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (research-project; fintech-timeline-study)"
            })
            page.goto(wayback_url, wait_until="networkidle", timeout=timeout)
            # Give JS extra time to finish rendering
            page.wait_for_timeout(5000)
            html = page.content()
            final_url = page.url
            browser.close()
            return html, final_url
    except Exception as e:
        print(f"        Playwright fetch failed: {e}", flush=True)
        return None, None


# ── HTML parsing ─────────────────────────────────────────────────────────────

def extract_text(html):
    """Extract visible text from HTML, stripping scripts/styles/nav."""
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup.find_all(['script', 'style', 'noscript', 'iframe']):
        tag.decompose()
    text = soup.get_text(separator='\n', strip=True)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def extract_title(html):
    """Extract page title from HTML."""
    soup = BeautifulSoup(html, 'html.parser')
    title_tag = soup.find('title')
    return title_tag.get_text(strip=True) if title_tag else ""


def extract_links(html, base_url):
    """
    Extract all internal links from HTML.
    Returns list of dicts: {href, anchor_text, path}
    Only includes links on the same domain (internal links).
    """
    soup = BeautifulSoup(html, 'html.parser')
    base_parsed = urlparse(base_url)
    base_domain = base_parsed.netloc.replace('www.', '')

    links = []
    seen = set()

    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        if not href or href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
            continue

        # Resolve relative URLs
        try:
            full_url = urljoin(base_url, href)
        except ValueError:
            continue
        parsed = urlparse(full_url)
        link_domain = parsed.netloc.replace('www.', '')

        # Only keep internal links (same domain or subdomains)
        if base_domain not in link_domain:
            continue

        # Normalize
        path = parsed.path.rstrip('/')
        if not path:
            path = '/'

        # Deduplicate by path
        if path in seen:
            continue
        seen.add(path)

        anchor = a.get_text(strip=True)[:200]
        links.append({
            "href": full_url,
            "anchor_text": anchor,
            "path": path,
        })

    return links


def extract_terms_links(links):
    """
    From a list of extracted links, find those that look like
    terms, legal, disclosures, or banking agreement pages.
    """
    terms_links = []
    for link in links:
        path_lower = link["path"].lower()
        anchor_lower = link["anchor_text"].lower()
        for pattern in TERMS_PATTERNS:
            if pattern in path_lower or pattern.replace('/', ' ').strip() in anchor_lower:
                terms_links.append(link)
                break
        # Also match by anchor text keywords
        for kw in ['terms of service', 'terms of use', 'terms & conditions',
                    'legal', 'disclosures', 'cardholder agreement',
                    'deposit agreement', 'bank partner', 'fdic',
                    'user agreement', 'account agreement', 'privacy policy',
                    'licensing', 'regulatory']:
            if kw in anchor_lower:
                if link not in terms_links:
                    terms_links.append(link)
                break
    return terms_links


def _simple_stem(word):
    """Crude English stemmer: strip common suffixes for fuzzy matching."""
    # Apply up to 2 rounds of suffix stripping
    for _ in range(2):
        changed = False
        for suffix in ('ings', 'ing', 'tion', 'sion', 'ment', 'ness',
                       'able', 'ible', 'ous', 'ive', 'ers', 'er', 'es', 's'):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                word = word[:-len(suffix)]
                changed = True
                break
        if not changed:
            break
    # Strip trailing 'e' if stem is long enough (save->sav, like->lik)
    if word.endswith('e') and len(word) >= 4:
        word = word[:-1]
    return word


def product_name_tokens(product_name, company_name):
    """
    Extract meaningful search tokens from a product name,
    stripping company name and parenthetical descriptions.
    """
    company_lower = company_name.lower()
    product_lower = product_name.lower()

    # Remove company name words
    product_clean = product_lower
    for word in company_lower.split():
        if len(word) > 2:
            product_clean = product_clean.replace(word, '')

    # Remove parenthetical descriptions like "(High-Yield Savings Account)"
    product_clean = re.sub(r'\([^)]*\)', '', product_clean)
    product_clean = re.sub(r'[^a-z0-9\s]', '', product_clean).strip()
    tokens = [t for t in product_clean.split() if len(t) > 2]
    return tokens


def find_product_links(links, product_name, company_name):
    """
    From a list of extracted links, find those that seem related
    to a specific product (by name matching in path or anchor text).
    """
    tokens = product_name_tokens(product_name, company_name)

    if not tokens:
        return []

    # Require at least min_match tokens to match
    # For short token lists (1-2), accept 1 match; for 3+, require 2
    min_match = 1 if len(tokens) <= 2 else 2

    matches = []
    for link in links:
        path_lower = link["path"].lower().replace('-', ' ').replace('_', ' ')
        anchor_lower = link["anchor_text"].lower()
        combined = path_lower + ' ' + anchor_lower

        # Stem-based matching: "save" matches "savings", "saving", etc.
        combined_words = re.split(r'[\s/\-_]+', combined)
        stemmed_words = {_simple_stem(w) for w in combined_words if w}
        score = sum(1 for t in tokens
                    if t in combined or _simple_stem(t) in stemmed_words)
        if score >= min_match:
            matches.append((link, score))

    # Sort by score descending; return all matches (no arbitrary cap)
    matches.sort(key=lambda x: -x[1])
    return [m[0] for m in matches]


def is_product_detail_or_terms_link(link):
    """
    Check if a link looks like it leads to product details or terms,
    worth following one level deeper. Strict filter to avoid generic nav.
    """
    path_lower = link["path"].lower()
    anchor_lower = link["anchor_text"].lower()

    # Reject generic navigation / site-wide pages
    reject_patterns = [
        '/about-us', '/about/', '/careers', '/contact', '/blog',
        '/news', '/press', '/investors', '/login', '/signup',
        '/support', '/help', '/faq', '/search', '/sitemap',
        '/cookie', '/accessibility',
    ]
    for p in reject_patterns:
        if path_lower.startswith(p) or path_lower == p.rstrip('/'):
            return False

    # Terms / legal / regulatory — always follow
    terms_indicators = [
        'terms', 'legal', 'disclosures', 'agreement', 'tos',
        'cardholder', 'deposit-agreement', 'account-agreement',
        'fdic', 'licensing', 'regulatory', 'compliance',
        'bank-partner', 'member-agreement',
    ]
    for p in terms_indicators:
        if p in path_lower or p in anchor_lower:
            return True

    # Product detail — follow if anchor text signals depth
    detail_anchors = [
        'learn more', 'see details', 'view details', 'how it works',
        'get started', 'features', 'pricing', 'plans',
        'eligibility', 'rates', 'fees', 'apply now',
        'see all features', 'compare plans',
    ]
    for p in detail_anchors:
        if p in anchor_lower:
            return True

    # Product detail — follow if path signals depth
    detail_paths = [
        '/features', '/pricing', '/plans', '/eligibility',
        '/rates', '/fees', '/how-it-works',
    ]
    for p in detail_paths:
        if p in path_lower:
            return True

    return False


# ── File helpers ─────────────────────────────────────────────────────────────

def safe_dirname(name):
    """Create a filesystem-safe directory name."""
    return re.sub(r'[^\w\-.]', '_', name)[:80]


def save_html(subdir, company, label, timestamp, html):
    """
    Save raw HTML to disk.
    Returns the relative file path.
    """
    html_dir = os.path.join(OUTPUT_DIR, "html", subdir, safe_dirname(company))
    os.makedirs(html_dir, exist_ok=True)
    filename = f"{safe_dirname(label)}_{timestamp}.html"
    filepath = os.path.join(html_dir, filename)
    with open(filepath, 'w', encoding='utf-8', errors='replace') as f:
        f.write(html)
    return filepath


def is_financial_product(row):
    """Check if a product row is financial (triggers terms scraping)."""
    import pandas as pd
    subcat = row.get('product_subcategory', '')
    cat = row.get('product_category', '')
    if pd.notna(subcat) and subcat in ALL_FINANCIAL_SUBCATS:
        return True
    if pd.notna(cat) and cat in FINANCIAL_CATS:
        return True
    return False


# ── Checkpoint helpers ───────────────────────────────────────────────────────

def load_checkpoint(filepath):
    if not os.path.exists(filepath):
        return set()
    with open(filepath, 'r') as f:
        return set(tuple(x) for x in json.load(f))


def save_checkpoint(filepath, done_set):
    with open(filepath, 'w') as f:
        json.dump(list(done_set), f)
