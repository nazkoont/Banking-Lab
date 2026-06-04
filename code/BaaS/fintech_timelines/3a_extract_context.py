"""
3a_extract_context.py
=====================
Read scraped HTML product pages, extract visible text, apply keyword-based
focusing, and build a longitudinal panel of product context.

For each 6-month period (e.g. 2024-H1), each product's context is assembled
from the latest available snapshot of each URL on or before that cutoff date.
Older snapshots carry forward so that persistent information (e.g. terms
pages) remains available even if not re-archived every period.

A carried-forward URL is only included if it's either fresh (snapshot in
this period) or linked from a fresh page's HTML. This prevents dead/
superseded URLs (e.g. old versioned terms pages) from propagating after
they're no longer linked. A product is only emitted for a period if at
least one reachable URL has a fresh snapshot AND mentions the product.

Writes one row per (company, product, period) to
Data_cleaned/product_context.csv for downstream LLM extraction by
3b_extract_product_info.py.

Usage:
    python3 3a_extract_context.py [--limit N] [--company COMPANY]
"""

import argparse
import csv
import collections
import logging
import os
import re
import time
from datetime import datetime
from html.parser import HTMLParser
from urllib.parse import urlparse

# Ensure working directory is the project root (parent of Code/)
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# ── Config ───────────────────────────────────────────────────────────────────

PRODUCTS_DIR = "Data_wayback/html/products"
PRODUCT_PAGES_CSV = "Data_wayback/product_pages.csv"
OUTPUT_CSV = "Data_cleaned/product_context.csv"
LOG_FILE = "Data_wayback/intermediate_files/extract_context_log.txt"

# Max characters of focused text to keep
MAX_TEXT_CHARS = 8000

# Keywords to anchor text extraction around (case-insensitive)
ANCHOR_KEYWORDS = [
    # Rates & pricing
    'apy', 'apr', 'interest', 'rate', 'yield', 'variable', 'fixed',
    'basis points', 'prime',
    # Fees
    'fee', 'no fee', 'free', 'charge', 'cost', 'penalty', 'surcharge',
    'monthly', 'annual', 'overdraft', 'maintenance',
    # Banking / regulatory
    'fdic', 'bank', 'insured', 'member', 'issued by', 'provided by',
    'banking services', 'deposits held', 'custodian', 'chartered',
    'partner', 'sponsor', 'ncua',
    # Product terms
    'loan', 'credit', 'debit', 'savings', 'checking', 'deposit',
    'advance', 'installment', 'repay', 'borrow', 'withdraw',
    'cashback', 'cash back', 'rewards', 'points',
    # BaaS middleware
    'marqeta', 'galileo', 'unit', 'synapse', 'stripe', 'i2c',
    'treasury prime', 'plaid', 'tabapay', 'lithic',
    # Limits & terms
    'limit', 'minimum', 'maximum', 'balance', 'eligible', 'qualify',
    'terms', 'conditions', 'disclosure', 'agreement',
]

CSV_COLUMNS = [
    "company",
    "product_name",
    "period",
    "combined_context",
    "combined_context_chars",
    "page_types",
    "snapshot_dates",
    "num_source_files",
    "source_files",
]


# ── HTML text extraction ─────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    """Strip HTML tags, keep visible text."""
    def __init__(self):
        super().__init__()
        self.pieces = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'noscript', 'iframe', 'svg'):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'noscript', 'iframe', 'svg'):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            t = data.strip()
            if t:
                self.pieces.append(t)


def extract_text_from_html(html):
    """Extract visible text from HTML, stripping nav/header/footer boilerplate."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # Remove boilerplate tags, but preserve footer text containing
        # financial keywords (bank partner disclosures are often in footers)
        FINANCIAL_FOOTER_KEYWORDS = [
            'fdic', 'member fdic', 'issued by', 'provided by',
            'banking services', 'insured', 'chartered', 'bank',
        ]
        for tag in soup.find_all(['script', 'style', 'noscript', 'iframe',
                                   'svg', 'nav', 'header']):
            tag.decompose()
        for tag in soup.find_all('footer'):
            footer_text = tag.get_text().lower()
            if not any(kw in footer_text for kw in FINANCIAL_FOOTER_KEYWORDS):
                tag.decompose()

        main = (soup.find('main')
                or soup.find(attrs={'role': 'main'})
                or soup.find('article'))

        if main and len(main.get_text(strip=True)) > 200:
            text = main.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)
    except ImportError:
        parser = _TextExtractor()
        parser.feed(html)
        text = '\n'.join(parser.pieces)

    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def focus_text(full_text, product_name, max_chars=MAX_TEXT_CHARS):
    """
    Extract only paragraphs/lines that contain keyword matches.
    Splits text into paragraphs, keeps those with at least one keyword hit.
    """
    if len(full_text) <= 2000:
        return full_text

    product_tokens = [t for t in re.split(r'[\s\-_/]+', product_name.lower())
                      if len(t) > 2]
    all_keywords = list(ANCHOR_KEYWORDS) + product_tokens

    paragraphs = re.split(r'\n\s*\n|\n', full_text)

    kept = []
    for para in paragraphs:
        para = para.strip()
        if len(para) < 10:
            continue
        para_lower = para.lower()
        if any(kw in para_lower for kw in all_keywords):
            kept.append(para)

    if not kept:
        half = max_chars // 2
        return full_text[:half] + "\n[...]\n" + full_text[-half:]

    return "\n\n".join(kept)


def deduplicate_sentences(text):
    """
    Split text into sentences and remove exact duplicates while preserving
    order of first occurrence.
    """
    sentences = re.split(r'(?<=[.!?])\s+|\n\n+', text)

    seen = set()
    unique = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        key = re.sub(r'\s+', ' ', sent)
        if key not in seen:
            seen.add(key)
            unique.append(sent)

    return " ".join(unique)


# ── Product name matching ────────────────────────────────────────────────────

STOPWORDS = {"the", "a", "an", "for", "and", "or", "of", "by", "in", "on",
             "to", "with", "from", "at", "is", "it", "as"}


def get_product_keywords(name):
    """Extract keywords from product name, splitting camelCase compounds."""
    words = re.sub(r'[^a-zA-Z0-9 ]', ' ', str(name)).split()
    expanded = []
    for w in words:
        parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', w).split()
        expanded.extend(parts)
    return [w.lower() for w in expanded if w.lower() not in STOPWORDS and len(w) > 2]


def text_mentions_product(text, product_name):
    """Check if >=50% of product name keywords appear in the text."""
    keywords = get_product_keywords(product_name)
    if not keywords:
        return True  # can't check, assume relevant
    text_lower = re.sub(r'[^a-z0-9 ]', ' ', text.lower())
    found = sum(1 for kw in keywords if kw in text_lower)
    return (found / len(keywords)) >= 0.5


# ── Filename parsing ─────────────────────────────────────────────────────────

def parse_filename(filepath):
    """
    Parse company, product, page_type, and snapshot date from filepath.

    Path structure:
        .../products/{Company}/{ProductName}__{page_type}__{url_slug}_{timestamp}.html
    """
    company = os.path.basename(os.path.dirname(filepath)).replace('_', ' ')
    basename = os.path.splitext(os.path.basename(filepath))[0]

    KNOWN_PAGE_TYPES = {
        'product_page', 'product_sublink', 'terms_page',
        'terms_sublink', 'deep_terms', 'keyword_fallback',
    }

    parts = basename.split('__')
    page_type = "unknown"
    product_parts = []
    for i, p in enumerate(parts):
        p_stripped = p.strip().rstrip('_').lstrip('_')
        if p_stripped in KNOWN_PAGE_TYPES:
            page_type = p_stripped
            product_parts = parts[:i]
            break
    else:
        product_parts = [parts[0]] if parts else [basename]

    product_name = ' '.join(p.replace('_', ' ').strip() for p in product_parts if p.strip())

    ts_match = re.search(r'(\d{14})$', basename)
    if ts_match:
        ts = ts_match.group(1)
        snapshot_date = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
    else:
        snapshot_date = ""

    # Extract the URL slug as a proxy for unique URL identity
    # Format: {product}__{page_type}__{url_slug}_{timestamp}.html
    if page_type != "unknown":
        page_type_idx = None
        for i, p in enumerate(parts):
            if p.strip().rstrip('_').lstrip('_') == page_type:
                page_type_idx = i
                break
        if page_type_idx is not None and page_type_idx + 1 < len(parts):
            url_slug = parts[page_type_idx + 1]
            # Strip trailing timestamp
            url_slug = re.sub(r'_\d{14}$', '', url_slug)
        else:
            url_slug = basename
    else:
        url_slug = basename

    return {
        "company": company,
        "product_name": product_name,
        "page_type": page_type,
        "snapshot_date": snapshot_date,
        "url_slug": url_slug,
    }


# ── File collection ──────────────────────────────────────────────────────────

def collect_html_files(products_dir, company_filter=None):
    """Collect all HTML files, optionally filtered by company."""
    files = []
    for company_dir in sorted(os.listdir(products_dir)):
        company_path = os.path.join(products_dir, company_dir)
        if not os.path.isdir(company_path):
            continue
        if company_filter and company_dir.lower().replace('_', ' ') != company_filter.lower():
            continue
        for fname in sorted(os.listdir(company_path)):
            if fname.endswith('.html'):
                files.append(os.path.join(company_path, fname))
    return files


def load_url_mapping(csv_path):
    """
    Load product_pages.csv to map HTML filenames to their actual target URLs.
    Returns dict: html_basename -> target_url
    """
    url_map = {}
    if not os.path.exists(csv_path):
        return url_map
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            html_path = row.get('html_path', '').strip()
            target_url = row.get('target_url', '').strip()
            if html_path and target_url:
                basename = os.path.basename(html_path)
                url_map[basename] = target_url
    return url_map


def filter_reachable_urls(latest_by_url, period, all_file_entries):
    """
    Filter carried-forward URLs to only those that are either:
    1. Fresh (snapshot in this period), or
    2. Linked from a fresh page (target_url appears in the HTML of a fresh page)

    all_file_entries is the full list of file entries for this product
    (used to get fresh pages' full HTML).
    """
    # Identify fresh files in this period
    fresh_htmls = []
    fresh_urls = set()
    for url_key, (dt, f) in latest_by_url.items():
        if date_to_period(f["snapshot_date"]) == period:
            fresh_urls.add(url_key)
            fresh_htmls.append(f["full_text"])

    if not fresh_urls:
        return {}  # nothing fresh, nothing reachable

    # For non-fresh URLs, check if they appear as links in any fresh page's HTML
    result = {}
    for url_key, (dt, f) in latest_by_url.items():
        if url_key in fresh_urls:
            result[url_key] = (dt, f)
        else:
            # Check if this URL is linked from any fresh page
            target_url = f.get("target_url", "")
            if target_url:
                # Check both the full URL and the path portion
                path = urlparse(target_url).path
                for html in fresh_htmls:
                    if target_url in html or (path and len(path) > 1 and path in html):
                        result[url_key] = (dt, f)
                        break

    return result


def extract_paragraphs_around_product(text, product_name):
    """
    Extract only the paragraphs that mention the product name.
    Used for cross-product files where we want context around the mention,
    not the entire page.
    """
    keywords = get_product_keywords(product_name)
    if not keywords:
        return ""

    # Split into paragraphs
    paragraphs = re.split(r'\n\s*\n|\n', text)

    # Keep paragraphs where >=50% of product name keywords appear
    kept = []
    for para in paragraphs:
        para = para.strip()
        if len(para) < 10:
            continue
        para_lower = re.sub(r'[^a-z0-9 ]', ' ', para.lower())
        found = sum(1 for kw in keywords if kw in para_lower)
        if found / len(keywords) >= 0.5:
            kept.append(para)

    return "\n\n".join(kept)


def extract_focused_text(filepath, log):
    """
    Read a single HTML file, extract text, focus it.
    Returns (full_text, focused_text, status) or (None, None, status) on failure.
    """
    meta = parse_filename(filepath)

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            html = f.read().replace('\x00', '')
    except Exception as e:
        log.info(f"    Read error: {e}")
        return None, None, "read_error"

    text = extract_text_from_html(html)

    # Handle PDF content saved as .html
    if '\x00' in text or text[:20].startswith('%PDF') or text[:20].startswith('PK'):
        log.info(f"    Detected binary/PDF content, attempting PDF text extraction")
        try:
            import pdfplumber
            with open(filepath, 'rb') as pf:
                pdf = pdfplumber.open(pf)
                pdf_text = '\n'.join(p.extract_text() or '' for p in pdf.pages)
                pdf.close()
            pdf_text = pdf_text.strip()
            if len(pdf_text) >= 100:
                text = pdf_text
            else:
                return None, None, "corrupted_pdf"
        except Exception:
            return None, None, "pdf_error"

    if len(text) < 100:
        return None, None, "insufficient_text"

    focused = focus_text(text, meta["product_name"])

    if len(focused) < 50:
        return None, None, "insufficient_focused_text"

    return text, focused, "OK"


# ── Period helpers ───────────────────────────────────────────────────────────

def date_to_period(date_str):
    """Convert YYYY-MM-DD to period like '2024-H1'. Returns '' on failure."""
    if not date_str:
        return ""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        half = "H1" if dt.month <= 6 else "H2"
        return f"{dt.year}-{half}"
    except ValueError:
        return ""


def period_to_cutoff(period):
    """Convert period like '2024-H1' to cutoff date (last day of the half)."""
    year, half = period.split("-")
    year = int(year)
    if half == "H1":
        return datetime(year, 6, 30)
    else:
        return datetime(year, 12, 31)


def generate_periods(min_date, max_date):
    """Generate all 6-month periods between min_date and max_date inclusive."""
    periods = []
    year = min_date.year
    half = 1 if min_date.month <= 6 else 2
    while True:
        p = f"{year}-H{half}"
        cutoff = period_to_cutoff(p)
        if cutoff > max_date:
            # Include this last period if min_date falls in it
            periods.append(p)
            break
        periods.append(p)
        if half == 1:
            half = 2
        else:
            half = 1
            year += 1
    return periods


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract and combine focused context per (company, product, period)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max products to output (0 = all)")
    parser.add_argument("--company", type=str, default=None,
                        help="Only process this company (folder name, e.g. 'Chime')")
    parser.add_argument("--output", type=str, default=OUTPUT_CSV,
                        help=f"Output CSV path (default: {OUTPUT_CSV})")
    parser.add_argument("--page-types", type=str, default=None,
                        help="Comma-separated page types to process (e.g. 'product_page,terms_page')")
    args = parser.parse_args()

    output_csv = args.output
    page_type_filter = set(args.page_types.split(',')) if args.page_types else None

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a'),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger(__name__)

    # Load URL mapping from product_pages.csv
    url_mapping = load_url_mapping(PRODUCT_PAGES_CSV)
    log.info(f"Loaded URL mapping for {len(url_mapping)} HTML files")

    # ── Step 1: Collect and parse all HTML files ─────────────────────────
    all_files = collect_html_files(PRODUCTS_DIR, args.company)
    if page_type_filter:
        filtered = []
        for f in all_files:
            info = parse_filename(f)
            if info["page_type"] in page_type_filter:
                filtered.append(f)
        all_files = filtered

    log.info(f"Found {len(all_files)} HTML files to process")

    # ── Step 2: Group files by product, apply limit, then extract text ───
    # First pass: group filepaths by (company, product) using filename only
    files_by_product = collections.OrderedDict()
    for filepath in all_files:
        meta = parse_filename(filepath)
        key = (meta["company"], meta["product_name"])
        if key not in files_by_product:
            files_by_product[key] = []
        files_by_product[key].append(filepath)

    log.info(f"Found {len(files_by_product)} unique products from filenames")

    # Also add master timeline products that have no own files
    # so they can still receive cross-product context
    import re as _re
    master_csv = "Data_cleaned/fintech_timelines_master.csv"
    if os.path.exists(master_csv):
        import csv as _csv
        with open(master_csv, newline="", encoding="utf-8") as mf:
            reader = _csv.DictReader(mf)
            for row in reader:
                if row.get("entry_type", "").strip() != "Product":
                    continue
                company_raw = row.get("company_name", "").strip()
                product_raw = row.get("product_name", "").strip()
                if not company_raw or not product_raw:
                    continue
                # Map master company name to folder name
                company_slug = _re.sub(r"[^a-zA-Z0-9]", "_", company_raw).strip("_")
                # Only add if this company has a folder (i.e. was scraped)
                company_folder = os.path.join(PRODUCTS_DIR, company_slug)
                if not os.path.isdir(company_folder):
                    continue
                key = (company_slug, product_raw)
                if key not in files_by_product:
                    files_by_product[key] = []  # no own files, relies on cross-product

        log.info(f"After adding master timeline products: {len(files_by_product)} total")

    # Apply limit but identify all companies we need files for
    if args.limit:
        files_by_product = collections.OrderedDict(
            list(files_by_product.items())[:args.limit])
        log.info(f"Limiting to {args.limit} products")

    # Identify companies of selected products — we need ALL files for these
    # companies so cross-product context sharing works
    selected_companies = set(k[0] for k in files_by_product)

    # Collect all files for selected companies
    company_filepaths = collections.defaultdict(list)
    for filepath in all_files:
        meta = parse_filename(filepath)
        if meta["company"] in selected_companies:
            company_filepaths[meta["company"]].append(filepath)

    all_company_files = [fp for fps in company_filepaths.values() for fp in fps]
    log.info(f"Extracting text from {len(all_company_files)} HTML files "
             f"across {len(selected_companies)} companies...")

    # Second pass: extract text for all files from selected companies
    # Index by both product and company for cross-product sharing
    product_files = collections.defaultdict(list)
    company_files = collections.defaultdict(list)
    total_skipped = 0

    for i, filepath in enumerate(all_company_files):
        meta = parse_filename(filepath)

        full_text, focused, status = extract_focused_text(filepath, log)
        if focused is None:
            total_skipped += 1
            continue

        # Look up actual target URL from product_pages.csv
        basename = os.path.basename(filepath)
        target_url = url_mapping.get(basename, meta["url_slug"])

        file_entry = {
            "filepath": filepath,
            "meta": meta,
            "full_text": full_text,
            "focused_text": focused,
            "snapshot_date": meta["snapshot_date"],
            "target_url": target_url,
            "url_slug": meta["url_slug"],
            "page_type": meta["page_type"],
        }

        key = (meta["company"], meta["product_name"])
        product_files[key].append(file_entry)
        company_files[meta["company"]].append(file_entry)

        if (i + 1) % 500 == 0:
            log.info(f"  ... {i+1}/{len(all_company_files)} files processed")

    log.info(f"Extracted text from {sum(len(v) for v in product_files.values())} files "
             f"across {len(product_files)} products ({total_skipped} skipped)")

    # ── Step 3: Determine the global period range ────────────────────────
    all_dates = []
    for files in product_files.values():
        for f in files:
            if f["snapshot_date"]:
                try:
                    all_dates.append(datetime.strptime(f["snapshot_date"], "%Y-%m-%d"))
                except ValueError:
                    pass

    if not all_dates:
        log.error("No valid snapshot dates found. Cannot generate periods.")
        return

    min_date = min(all_dates)
    max_date = max(all_dates)
    all_periods = generate_periods(min_date, max_date)
    log.info(f"Period range: {all_periods[0]} to {all_periods[-1]} ({len(all_periods)} periods)")

    # ── Step 4: For each product × period, assemble context ──────────────
    # For each product, we consider:
    #   - Files assigned to this product by filename ("own files")
    #   - Files from other products of the same company that mention this
    #     product ("cross-product files")
    # Both contribute context; both can satisfy the "alive" check.

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        rows_written = 0
        products_processed = 0

        for (company, product_name) in files_by_product:
            products_processed += 1
            t0 = time.time()

            # Gather all candidate files: own files + cross-product files
            # that mention this product
            own_files = product_files.get((company, product_name), [])
            all_company = company_files.get(company, [])

            # Identify cross-product files that mention this product.
            # For cross-product files, only keep paragraphs around the
            # product mention rather than the full page.
            own_filepaths = set(f["filepath"] for f in own_files)
            cross_files = []
            for f in all_company:
                if f["filepath"] not in own_filepaths:
                    if text_mentions_product(f["full_text"], product_name):
                        # Extract only paragraphs mentioning this product
                        relevant = extract_paragraphs_around_product(
                            f["focused_text"], product_name)
                        if relevant and len(relevant) >= 50:
                            cross_entry = dict(f)
                            cross_entry["focused_text"] = relevant
                            cross_files.append(cross_entry)

            candidate_files = own_files + cross_files

            # Parse snapshot dates
            dated_files = []
            for f in candidate_files:
                if f["snapshot_date"]:
                    try:
                        dt = datetime.strptime(f["snapshot_date"], "%Y-%m-%d")
                        dated_files.append((dt, f))
                    except ValueError:
                        pass

            if not dated_files:
                continue

            if cross_files:
                log.info(f"  {company} / {product_name}: "
                         f"{len(own_files)} own + {len(cross_files)} cross-product files")

            # For each period, find the latest snapshot per URL slug
            # that's on or before the cutoff
            for period in all_periods:
                cutoff = period_to_cutoff(period)

                # For each target_url, find the latest file on or before cutoff
                latest_by_url = {}
                for dt, f in dated_files:
                    if dt <= cutoff:
                        url_key = f["target_url"]
                        if url_key not in latest_by_url or dt > latest_by_url[url_key][0]:
                            latest_by_url[url_key] = (dt, f)

                if not latest_by_url:
                    continue  # no snapshots exist yet for this period

                # Filter to reachable URLs: fresh pages carry forward,
                # stale pages only carry forward if linked from a fresh page
                latest_by_url = filter_reachable_urls(
                    latest_by_url, period, candidate_files)

                if not latest_by_url:
                    continue  # no reachable URLs in this period

                # Check if the product is "alive": at least one URL has a
                # fresh snapshot in THIS period AND mentions the product
                has_fresh_relevant = False
                for url_key, (dt, f) in latest_by_url.items():
                    file_period = date_to_period(f["snapshot_date"])
                    if file_period == period:
                        if text_mentions_product(f["full_text"], product_name):
                            has_fresh_relevant = True
                            break

                if not has_fresh_relevant:
                    continue  # product not confirmed alive in this period

                # Assemble context from all reachable snapshots
                all_focused = []
                page_types = set()
                snapshot_dates = set()
                source_files = []

                for url_key, (dt, f) in latest_by_url.items():
                    all_focused.append(f["focused_text"])
                    page_types.add(f["page_type"])
                    snapshot_dates.add(f["snapshot_date"])
                    source_files.append(f["filepath"])

                # Combine, deduplicate, and collapse whitespace for CSV safety
                raw_combined = "\n\n".join(all_focused)
                combined = deduplicate_sentences(raw_combined)
                combined = re.sub(r'[\r\n]+', ' ', combined).strip()

                writer.writerow({
                    "company": company,
                    "product_name": product_name,
                    "period": period,
                    "combined_context": combined,
                    "combined_context_chars": len(combined),
                    "page_types": "; ".join(sorted(page_types)),
                    "snapshot_dates": "; ".join(sorted(snapshot_dates)),
                    "num_source_files": len(source_files),
                    "source_files": "; ".join(source_files),
                })
                rows_written += 1

            elapsed = time.time() - t0
            if products_processed % 100 == 0:
                log.info(f"  ... {products_processed}/{len(product_files)} products, "
                         f"{rows_written} rows written")

    log.info(f"Done. {rows_written} rows written across {products_processed} products.")
    log.info(f"Results in: {output_csv}")


if __name__ == "__main__":
    main()
