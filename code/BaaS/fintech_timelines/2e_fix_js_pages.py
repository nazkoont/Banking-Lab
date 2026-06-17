"""
fix_js_pages.py
===============
Post-processing script that scans already-scraped HTML files, detects
JS-only pages (empty shells), and re-fetches them using Playwright
(headless browser) so JavaScript renders the actual content.

Overwrites the original HTML file and updates the CSV row.
Tracks fixed files in a JSON so they aren't re-processed on future runs.

Usage:
    python fix_js_pages.py                # Scan and report JS-only pages
    python fix_js_pages.py --fix          # Re-fetch JS-only pages with Playwright
    python fix_js_pages.py --fix --dry    # Show what would be re-fetched
"""

import argparse
import csv
import json
import os
import time

import importlib
_wu = importlib.import_module("2_wayback_utils")
OUTPUT_DIR, FETCH_DELAY = _wu.OUTPUT_DIR, _wu.FETCH_DELAY
is_js_only_page, fetch_wayback_page_playwright = _wu.is_js_only_page, _wu.fetch_wayback_page_playwright
extract_text, extract_title = _wu.extract_text, _wu.extract_title

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "product_pages.csv")
HTML_DIR = os.path.join(OUTPUT_DIR, "html", "products")
JS_FIXED_FILE = os.path.join(OUTPUT_DIR, "intermediate_files", "js_fixed.json")

FIELDNAMES = [
    "company_name", "product_name", "year_launched",
    "is_financial", "page_type", "source",
    "target_url", "target_year", "target_month",
    "snapshot_timestamp", "snapshot_date", "page_title",
    "text_length", "html_path", "text_preview",
]


def load_js_fixed():
    """Load set of HTML file paths that have already been fixed."""
    if not os.path.exists(JS_FIXED_FILE):
        return set()
    with open(JS_FIXED_FILE, 'r') as f:
        return set(json.load(f))


def save_js_fixed(fixed_set):
    """Save set of fixed HTML file paths."""
    with open(JS_FIXED_FILE, 'w') as f:
        json.dump(sorted(fixed_set), f, indent=2)


def scan_js_pages(already_fixed):
    """Scan all scraped HTML files and return list of JS-only ones,
    excluding those already fixed."""
    js_pages = []
    total = 0

    for root, dirs, files in os.walk(HTML_DIR):
        for fn in files:
            if not fn.endswith('.html'):
                continue
            total += 1
            fp = os.path.join(root, fn)

            if fp in already_fixed:
                continue

            try:
                with open(fp, 'r', encoding='utf-8', errors='replace') as f:
                    html = f.read()
                if is_js_only_page(html):
                    text = extract_text(html)
                    js_pages.append({
                        'file': fp,
                        'filename': fn,
                        'text_length': len(text),
                    })
            except Exception as e:
                print(f"  Error reading {fn}: {e}")

    return js_pages, total


def find_csv_row_for_html(html_path, rows):
    """Find the CSV row that matches this html_path."""
    for i, row in enumerate(rows):
        if row.get('html_path', '') == html_path:
            return i
    return None


def fix_js_pages(dry_run=False):
    """Re-fetch JS-only pages with Playwright and update CSV."""
    already_fixed = load_js_fixed()
    if already_fixed:
        print(f"  {len(already_fixed)} files already fixed (skipping)",
              flush=True)

    print("Scanning for JS-only pages...", flush=True)
    js_pages, total = scan_js_pages(already_fixed)

    print(f"\n  Total HTML files: {total}")
    print(f"  Already fixed: {len(already_fixed)}")
    print(f"  JS-only (need re-fetch): {len(js_pages)}")

    if not js_pages:
        print("  Nothing to fix!")
        return

    print()
    for jp in js_pages:
        print(f"  JS-ONLY: {jp['filename']} (text: {jp['text_length']} chars)")

    if dry_run:
        print(f"\n  Dry run — would re-fetch {len(js_pages)} pages")
        return

    # Load CSV rows
    if not os.path.exists(OUTPUT_FILE):
        print(f"\n  ERROR: {OUTPUT_FILE} not found")
        return

    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Build html_path -> (timestamp, url) mapping from CSV
    path_to_info = {}
    for row in rows:
        hp = row.get('html_path', '')
        if hp:
            path_to_info[hp] = {
                'timestamp': row.get('snapshot_timestamp', ''),
                'url': row.get('target_url', ''),
            }

    fixed = 0
    failed = 0

    print(f"\nRe-fetching {len(js_pages)} JS-only pages with Playwright...\n")

    for jp in js_pages:
        fp = jp['file']
        info = path_to_info.get(fp)

        if not info or not info['timestamp'] or not info['url']:
            print(f"  SKIP: {jp['filename']} — no CSV match found")
            failed += 1
            continue

        ts = info['timestamp']
        url = info['url']
        print(f"  Fetching: {url[:70]} ({ts})", flush=True)

        pw_html, pw_url = fetch_wayback_page_playwright(ts, url)
        time.sleep(FETCH_DELAY)

        if not pw_html:
            print(f"    FAILED — keeping original")
            failed += 1
            continue

        # Check if Playwright version actually has more content
        new_text = extract_text(pw_html)
        old_text_len = jp['text_length']

        if len(new_text) <= old_text_len:
            print(f"    No improvement ({len(new_text)} chars vs "
                  f"{old_text_len}) — marking as fixed anyway")
            # Still mark as fixed so we don't retry every run
            already_fixed.add(fp)
            save_js_fixed(already_fixed)
            failed += 1
            continue

        # Overwrite HTML file
        with open(fp, 'w', encoding='utf-8', errors='replace') as f:
            f.write(pw_html)

        # Update CSV row
        row_idx = find_csv_row_for_html(fp, rows)
        if row_idx is not None:
            rows[row_idx]['text_length'] = str(len(new_text))
            rows[row_idx]['text_preview'] = new_text[:500].replace('\n', ' ')
            rows[row_idx]['page_title'] = extract_title(pw_html)[:200]

        # Mark as fixed
        already_fixed.add(fp)
        save_js_fixed(already_fixed)

        fixed += 1
        print(f"    OK — {old_text_len} -> {len(new_text)} chars")

    # Write updated CSV
    if fixed > 0:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES, escapechar='\\', quoting=csv.QUOTE_MINIMAL)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  Updated CSV with {fixed} fixed rows")

    print(f"\nDone! Fixed: {fixed}, Failed/skipped: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description="Scan and fix JS-only pages with Playwright")
    parser.add_argument("--fix", action="store_true",
                        help="Re-fetch JS-only pages (without this, just scans)")
    parser.add_argument("--dry", action="store_true",
                        help="Show what would be fixed without doing it")
    args = parser.parse_args()

    if args.fix:
        fix_js_pages(dry_run=args.dry)
    else:
        already_fixed = load_js_fixed()
        print("Scanning for JS-only pages...", flush=True)
        js_pages, total = scan_js_pages(already_fixed)
        print(f"\n  Total HTML files: {total}")
        print(f"  Already fixed: {len(already_fixed)}")
        print(f"  JS-only (need re-fetch): {len(js_pages)}")
        for jp in js_pages:
            print(f"    {jp['filename']} (text: {jp['text_length']} chars)")
        if js_pages:
            print(f"\n  Run with --fix to re-fetch these with Playwright")


if __name__ == "__main__":
    main()
