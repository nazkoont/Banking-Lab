#!/usr/bin/env python3
"""
Scrape BankingEmbedded fintech list and extract partner data under each
"Programs embedding financial services" tab.

**Updated 2025‑07‑18**
----------------------------------
* **LIST‑ONLY mode** – switch `LIST_ONLY` prints quick company list.
* **CSV‑SEED mode**  – switch `USE_CSV_LIST` skips re‑scraping list page.
* **NEW** partner **summary** capture – script now pulls a text summary from
  each partner profile (meta description or first paragraph) and adds it to a
  new `Partner Summary` CSV column.
* All other behaviour preserved.

Requires:  `pip install selenium==4 webdriver-manager beautifulsoup4 pandas`
"""
from __future__ import annotations

import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# ---------------------------------------------------------------------------
# Configuration – tweak here only
# ---------------------------------------------------------------------------
LIST_ONLY     = False  # True ⇒ just list fintech names & profile URLs
USE_CSV_LIST  = True  # True ⇒ read fintech list from CSV instead of scraping
HEADLESS      = False  # True ⇒ run Chrome headless
MAX_SCROLLS   = None      # None ⇒ keep scrolling until page stops growing
PROFILE_LIMIT = None      # None ⇒ visit every captured fintech profile

SCROLL_PAUSE      = 1.5  # seconds between list‑page scrolls
CLICK_WAIT        = 1.0  # seconds after switching a sub‑tab
PROFILE_WAIT_MIN  = 10    # min random seconds before parsing a profile
PROFILE_WAIT_MAX  = 20   # max random seconds before parsing a profile
PARTNER_WAIT_MIN  = 10    # min random seconds per partner page
PARTNER_WAIT_MAX  = 20    # max random seconds per partner page

BASE_URL         = "https://www.bankingembedded.com"
FINTECH_LIST_URL = f"{BASE_URL}/companies"
CSV_LIST_PATH    = "banking_embedded_fintech_list.csv"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/114.0.0.0 Safari/537.36"
)

SOCIAL_DOMAINS = (
    "linkedin.com",
    "twitter.com",
    "x.com",
    "crunchbase.com",
    "facebook.com",
    "instagram.com",
    "youtube.com",
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_program_blocks(page_soup: BeautifulSoup):
    """Return program card containers for a given sub‑tab soup."""
    blocks = page_soup.select("div.card-container.programCardTop")
    if not blocks:
        # fallback selector in case classes change slightly
        blocks = page_soup.select("div:has(h4):has(a[href^='/banks/'])")
    return blocks


def clean_number_badge(label: str) -> str:
    """Strip trailing " (n)" counts and trim whitespace."""
    return re.sub(r"\s*\(\d+\)$", "", label).strip()


def extract_external_website(soup: BeautifulSoup) -> str | None:
    """Return the first non‑social external URL on the page."""
    link = soup.select_one("a:-soup-contains('View Website')")
    if link and link.get("href"):
        return link["href"]
    for a in soup.select("a[href^='http']"):
        href = a.get("href", "")
        if ("bankingembedded.com" in href) or any(s in href for s in SOCIAL_DOMAINS):
            continue
        return href
    return None


def extract_summary(soup: BeautifulSoup) -> str:
    """Return text summary for a profile (meta description or first <p>)."""
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        return meta["content"].strip()
    para = soup.select_one("main p") or soup.select_one("p")
    return para.get_text(" ", strip=True) if para else ""

# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------

def load_fintech_list_from_csv() -> List[str]:
    if not Path(CSV_LIST_PATH).is_file():
        raise FileNotFoundError(
            f"CSV list not found at {CSV_LIST_PATH!r}. Run with LIST_ONLY=True first "
            "or disable USE_CSV_LIST."
        )
    df = pd.read_csv(CSV_LIST_PATH)
    if "Profile URL" not in df.columns:
        raise ValueError("CSV list must contain a 'Profile URL' column")
    print(f"Loaded {len(df)} fintech profile URLs from {CSV_LIST_PATH}")
    return df["Profile URL"].dropna().unique().tolist()


def main() -> None:
    profile_urls: List[str] = []
    fintech_rows: List[Dict[str, str]] = []

    # LIST_ONLY & USE_CSV_LIST are mutually exclusive
    if LIST_ONLY and USE_CSV_LIST:
        sys.exit("Configuration error: LIST_ONLY and USE_CSV_LIST cannot both be True.")

    # -------------------------------------------------------------------
    # 1. Acquire fintech profile URL list
    # -------------------------------------------------------------------
    if USE_CSV_LIST:
        profile_urls = load_fintech_list_from_csv()
    else:
        # open browser and scrape list page
        opts = Options()
        if HEADLESS:
            opts.add_argument("--headless=new")
        opts.add_argument(f"user-agent={USER_AGENT}")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(
            service=webdriver.chrome.service.Service(ChromeDriverManager().install()),
            options=opts,
        )
        wait = WebDriverWait(driver, 20)

        try:
            print("Navigating to fintech list…")
            driver.get(FINTECH_LIST_URL)

            # scroll list page
            last_height = driver.execute_script("return document.body.scrollHeight")
            scroll_count = 0
            while True:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(SCROLL_PAUSE)
                scroll_count += 1
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height or (MAX_SCROLLS is not None and scroll_count >= MAX_SCROLLS):
                    break
                last_height = new_height
            print(f"Scrolled {scroll_count} time(s)")

            # wait for cards
            try:
                wait.until(
                    EC.presence_of_element_located(
                        (By.XPATH, "//a[contains(@href,'/companies/') and not(contains(@href,'#'))]")
                    )
                )
            except Exception:
                print("Timed out waiting for fintech cards – try increasing MAX_SCROLLS or SCROLL_PAUSE.")

            list_soup = BeautifulSoup(driver.page_source, "html.parser")
            links = list_soup.select("a[href*='/companies/']")

            seen: Set[str] = set()
            for a in links:
                href = a.get("href")
                if not href or "/companies/" not in href:
                    continue
                full = BASE_URL + href
                if full in seen:
                    continue
                seen.add(full)
                profile_urls.append(full)

                if LIST_ONLY:
                    name = a.get_text(" ", strip=True) or Path(href).name.replace("-", " ").title()
                    fintech_rows.append({"Fintech Name": name, "Profile URL": full})
        finally:
            driver.quit()

    # -------------------------------------------------------------------
    # 2. LIST_ONLY early exit
    # -------------------------------------------------------------------
    if LIST_ONLY:
        df = pd.DataFrame(fintech_rows)
        if df.empty:
            raise RuntimeError("LIST_ONLY mode produced no rows – check selectors.")
        df.to_csv(CSV_LIST_PATH, index=False)
        print(df.to_string(index=False))
        print(f"Saved {len(df)} rows to {CSV_LIST_PATH}")
        return

    if not profile_urls:
        sys.exit("No fintech profiles to visit – aborting.")

    # -------------------------------------------------------------------
    # 3. Full scrape (profiles, programs, partners)
    # -------------------------------------------------------------------
    opts = Options()
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument(f"user-agent={USER_AGENT}")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(
        service=webdriver.chrome.service.Service(ChromeDriverManager().install()),
        options=opts,
    )

    try:
        visit = profile_urls if PROFILE_LIMIT is None else profile_urls[:PROFILE_LIMIT]
        print(f"Visiting {len(visit)} profile(s)…")

        records: List[Dict] = []
        partner_internal_urls: Set[str] = set()

        # -----------------------------------------------------------
        # 3a. FINTECH PROFILES
        # -----------------------------------------------------------
        for url in visit:
            driver.get(url)
            time.sleep(random.uniform(PROFILE_WAIT_MIN, PROFILE_WAIT_MAX))
            soup = BeautifulSoup(driver.page_source, "html.parser")

            fintech_name = (soup.select_one("h1") or {}).get_text(strip=True) if soup.select_one("h1") else Path(url).name
            fintech_summary = extract_summary(soup)
            fintech_website  = extract_external_website(soup)

            # locate program section header (case‑insensitive match)
            header = soup.find(lambda t: t.name and t.name.startswith("h") and "programs embedding financial services" in t.get_text(" ", strip=True).lower())
            if not header:
                print(f"[WARN] {fintech_name}: no program section found")
                continue

            # gather sub‑tab buttons present in DOM
            try:
                tab_buttons = driver.find_elements(By.XPATH, "//button[@role='tab']")
            except Exception:
                tab_buttons = []

            subtabs = []
            seen_labels: Set[str] = set()
            for el in tab_buttons:
                label = clean_number_badge(el.text)
                if not label or label.lower() == "all" or label in seen_labels:
                    continue
                subtabs.append((label, el))
                seen_labels.add(label)
            if not subtabs:
                subtabs = [("Unknown", None)]  # static page w/ no subtabs

            # -------------------------------------------------------
            # 3b. Iterate sub‑tabs / programs
            # -------------------------------------------------------
            for sub_name, tab_el in subtabs:
                if tab_el is not None:
                    driver.execute_script("arguments[0].click();", tab_el)
                    time.sleep(CLICK_WAIT)
                    sub_soup = BeautifulSoup(driver.page_source, "html.parser")
                else:
                    sub_soup = soup

                for block in get_program_blocks(sub_soup):
                    prog_tag = block.find(["h4", "h3"])
                    program_name = prog_tag.get_text(strip=True) if prog_tag else "Unknown Program"
                    program_summary = (block.find("p") or {}).get_text(" ", strip=True) if block.find("p") else ""
                    program_website = extract_external_website(block)

                    partner_links = block.select("a[href*='/banks/'], a[href*='/middleware/'], a[href*='/partners/']")
                    if not partner_links:
                        records.append(
                            {
                                "Fintech Name": fintech_name,
                                "Fintech Summary": fintech_summary,
                                "Fintech Website": fintech_website,
                                "Profile URL"   : url,
                                "Subcategory"   : sub_name,
                                "Program Name"  : program_name,
                                "Program Website": program_website,
                                "Partner Name"  : None,
                                "Partner URL"   : None,
                                "Partner Role"  : None,
                                "Partner Summary": None,
                                "Program Summary": program_summary,
                            }
                        )
                        continue

                    for link in partner_links:
                        h6s = link.find_all("h6")
                        partner_name = h6s[0].get_text(strip=True) if h6s else link.get_text(strip=True)
                        partner_role = h6s[1].get_text(strip=True) if len(h6s) > 1 else None
                        internal_url  = BASE_URL + link["href"]
                        partner_internal_urls.add(internal_url)

                        records.append(
                            {
                                "Fintech Name": fintech_name,
                                "Fintech Summary": fintech_summary,
                                "Fintech Website": fintech_website,
                                "Profile URL"   : url,
                                "Subcategory"   : sub_name,
                                "Program Name"  : program_name,
                                "Program Website": program_website,
                                "Partner Name"  : partner_name,
                                "Partner URL"   : internal_url,  # temp
                                "Partner Role"  : partner_role,
                                "Partner Summary": None,         # to be filled
                                "Program Summary": program_summary,
                            }
                        )

        # -----------------------------------------------------------
        # 4. Resolve partner external website + summary (deduped)
        # -----------------------------------------------------------
        print(f"Resolving {len(partner_internal_urls)} unique partner profile(s)…")
        partner_data: Dict[str, Dict[str, str]] = {}

        for purl in partner_internal_urls:
            driver.execute_script("window.open(arguments[0], '_blank');", purl)
            driver.switch_to.window(driver.window_handles[-1])
            try:
                time.sleep(random.uniform(PARTNER_WAIT_MIN, PARTNER_WAIT_MAX))
                psoup = BeautifulSoup(driver.page_source, "html.parser")
                ext_site = extract_external_website(psoup) or purl
                p_summary = extract_summary(psoup)
            except Exception:
                ext_site = purl
                p_summary = ""
            partner_data[purl] = {"website": ext_site, "summary": p_summary}
            driver.close()
            driver.switch_to.window(driver.window_handles[0])
            time.sleep(random.uniform(PARTNER_WAIT_MIN, PARTNER_WAIT_MAX))

        for rec in records:
            int_url = rec["Partner URL"]
            pdata   = partner_data.get(int_url, {})
            rec["Partner URL"]     = pdata.get("website", int_url)
            rec["Partner Summary"] = pdata.get("summary", "")

        # -----------------------------------------------------------
        # 5. Save CSV
        # -----------------------------------------------------------
        if records:
            out = "banking_embedded_programs.csv"
            pd.DataFrame(records).to_csv(out, index=False)
            print(f"Saved {len(records)} rows to {out}")
        else:
            print("No program entries captured.")

    finally:
        print("Closing browser…")
        driver.quit()
        print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted by user")
