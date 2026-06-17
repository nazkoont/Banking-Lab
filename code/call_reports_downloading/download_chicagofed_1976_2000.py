#!/usr/bin/env python3
"""
Download Chicago Fed 'Commercial Bank Data – Complete Files (1976–2000)'.

This page often displays link text like "call7603.zip" but the actual href
is a different media URL (e.g., contains "call7603-zip.zip" and/or query params).

This script:
- Fetches the page HTML
- Extracts ALL hrefs that contain callYY(03|06|09|12) and a .zip
- Creates quarter folders using MMDDYY naming (e.g., 033176)
- Downloads the zip into that folder
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

INDEX_URL = "https://www.chicagofed.org/banking/financial-institution-reports/commercial-bank-data-complete-1976-2000"
BASE_DIR = Path("/zfs/data/bankcallreports/raw/current/data")

HEADERS = {"User-Agent": "Mozilla/5.0"}

END_DAY = {"03": "31", "06": "30", "09": "30", "12": "31"}

# Grab href="...callYYMM...zip..." even if href is "callYYMM-zip.zip?la=en"
HREF_RE = re.compile(
    r'href="(?P<href>[^"]*call(?P<yy>\d{2})(?P<mm>03|06|09|12)[^"]*?\.zip[^"]*)"',
    re.IGNORECASE,
)


def quarter_folder(yy: str, mm: str) -> str:
    return f"{mm}{END_DAY[mm]}{yy}"


def fetch_html() -> str:
    r = requests.get(INDEX_URL, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.text


def extract_quarter_links(html: str):
    """
    Returns a list of tuples: (yy, mm, absolute_url, basename)
    """
    matches = list(HREF_RE.finditer(html))

    # de-dupe by absolute URL
    seen = set()
    out = []
    for m in matches:
        yy = m.group("yy")
        mm = m.group("mm")
        href = m.group("href")

        abs_url = urljoin(INDEX_URL, href)

        # determine a reasonable filename
        path = urlparse(abs_url).path
        base = Path(path).name
        if not base.lower().endswith(".zip"):
            base = f"call{yy}{mm}.zip"

        key = abs_url.lower()
        if key in seen:
            continue
        seen.add(key)

        out.append((yy, mm, abs_url, base))

    # Sort in chronological-ish order by (yy,mm)
    out.sort(key=lambda t: (int(t[0]), int(t[1])))
    return out


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return

    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, headers=HEADERS, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(dest)


def folder_has_files(folder: Path) -> bool:
    return folder.exists() and folder.is_dir() and any(folder.iterdir())


def main():
    print(f"Index page: {INDEX_URL}")
    print(f"Base output dir: {BASE_DIR}")

    html = fetch_html()
    quarters = extract_quarter_links(html)

    print(f"Found {len(quarters)} quarterly ZIP files.")
    if not quarters:
        print("TIP: The page HTML did not include any matching hrefs.")
        print("     If the site changed, we may need to scrape link TEXT too.")
        return

    for yy, mm, url, fname in quarters:
        folder = quarter_folder(yy, mm)
        out_dir = BASE_DIR / folder

        # If folder already has something in it, skip
        if folder_has_files(out_dir):
            print(f"SKIP {folder} (already exists)")
            continue

        zip_path = out_dir / fname
        print(f"Downloading → {folder}/{fname}")
        download(url, zip_path)
        time.sleep(0.2)

    print("Done.")


if __name__ == "__main__":
    main()
