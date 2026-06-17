Fintech Timelines Project
=========================

Data Pipeline Overview
----------------------

0. FINTECH UNIVERSE (Data_cleaned/banking_embedded_fintechs_unique.csv)
   Dataset of fintechs from bankingembedded.com. This is the universe of
   companies that we query to Claude in step 1.

1. CLAUDE RAW TIMELINES (Claude_raw/)
   Raw CSV batch files (fintech_timeline_batch*.csv) produced by manually
   querying Claude using the prompt in Claude_prompts/company_product_timeline_prompt_v6.md.
   Each batch covers ~3 companies. Older superseded batches are in Claude_raw/Archive/.

2. MASTER TIMELINE (Data_cleaned/fintech_timelines_master.csv)
   All batch files merged, subcategories cleaned, sorted by company name.
   Created by: Code/1_fintech_timelines_analyze.py

3. TIMELINE ANALYSIS (Analysis_timelines/)
   Analyzes product bundling patterns across fintechs. Classifies each
   product into a 3-tier taxonomy (Depository / Financial / Non-financial),
   then traces how companies expand across tiers over time.
   Outputs: bundling path frequency tables, tier transition matrices,
   expansion timing charts, and financial sub-type breakdowns.
   Also produces a depository-only subset of all analyses.
   Created by: Code/1_fintech_timelines_analyze.py

4. WAYBACK MACHINE SCRAPING (Data_wayback/)
   Scrapes archived product pages from the Wayback Machine for each company
   and product in the master timeline. HTML files saved to
   Data_wayback/html/products/{Company}/{Product}__{page_type}__{url}_{timestamp}.html

   Orchestrated by: Code/2_run_wayback_pipeline.sh
   Scripts (in Code/):
     - 2a_scrape_homepages.py       Scrape archived homepages
     - 2b_find_product_urls.py      (optional) Discover product URLs via CDX/DDG search
     - 2c_scrape_product_pages.py   Scrape product pages, terms, and sublinks
     - 2d_scrape_deep_terms.py      Follow links to legal/terms/disclosure pages
     - 2e_fix_js_pages.py           Re-fetch JS-only pages with headless browser

   Output CSVs (in Data_wayback/):
     - homepage_snapshots.csv       One row per (company, date) with homepage text/links
     - product_urls.csv             Discovered product URLs per company
     - product_pages.csv            Scraped product page content per (product, url, date)
     - scraped_pages.csv            Monthly Wayback snapshots per product URL

   Intermediate files (in Data_wayback/intermediate_files/):
     Checkpoints (JSON, allow scripts to resume after interruption):
       - homepage_checkpoint.json         Resume state for 2a
       - product_urls_checkpoint.json     Resume state for 2b
       - product_pages_checkpoint.json    Resume state for 2c
       - scrape_checkpoint.json           Resume state for scrape_wayback.py
       - deep_terms_checkpoints.json      Resume state for 2d
       - deep_terms_done.json             Completed deep terms URLs for 2d
       - deep_terms_scan_cache.json       Cached scan results for 2d
       - js_fixed.json                    Pages already re-fetched by 2e
     Logs:
       - pipeline_log.txt                 Stdout/stderr from 2_run_wayback_pipeline.sh
       - deep_terms_log.txt               Log from deep terms scraping (2d)
       - find_urls_log.txt                Log from find_product_urls.py (2b)
       - find_urls_run.log                Full-run log from 2b
       - phase2_run.log                   Log from product page scraping (2c)

   What we have run so far:
     - 2a (homepages), 2c (product pages), 2d (deep terms), 2e (JS fix)
       all with --latest-only (most recent snapshot only)
     - 2c was run with --financial-only (financial products only)

   What could optionally be run:
     - Remove --latest-only to scrape historical snapshots (semi-annual from
       2005-2025), giving a time series of how product pages changed
     - Remove --financial-only to also scrape non-financial product pages
     - Run 2b_find_product_urls.py to discover additional product URLs beyond
       those found on the homepage (uses Wayback CDX prefix search and
       DuckDuckGo site search)

5. PRODUCT INFO EXTRACTION (product_context.csv, extracted_product_info_v2.csv)
   Two-step pipeline that reads the scraped HTML pages and extracts structured
   financial product data (rates, fees, bank partners, BaaS middleware, etc.).

   Scripts (in Code/):
     - 3a_extract_context.py        Build a longitudinal panel of product context.
                                    For each 6-month period, assembles the latest
                                    available snapshot of each URL on or before
                                    that cutoff. Older snapshots carry forward so
                                    persistent info (e.g. terms pages) remains
                                    available. A product is only emitted for a
                                    period if at least one URL has a fresh snapshot
                                    in that period AND the page mentions the product
                                    (fuzzy keyword check), preventing dead products
                                    from propagating. Combines and deduplicates
                                    sentences across pages within each period.
     - 3b_extract_product_info.py   Two-pass LLM extraction from combined context:
                                      Pass 1: Filter/trim context to only text
                                      relevant to the specific product (verbatim,
                                      no rephrasing). Classify scope as specific,
                                      broad, or mixed.
                                      Pass 2: Extract structured fields from the
                                      filtered context, distinguishing product-
                                      specific vs broad bank partners and BaaS
                                      middleware.

   Output CSVs (in Data_cleaned/):
     - product_context.csv                  One row per (company, product, period)
                                            with combined and deduplicated context
                                            from the latest available snapshots as
                                            of that period's cutoff date
     - extracted_product_info_v2.csv        Structured LLM extraction results with
                                            columns for specific vs broad bank
                                            partners and BaaS middleware
     - extracted_product_info_v2_clean.csv  Same without context columns

   Legacy output (in Data_cleaned/, from Code/Code_archive/extract_product_info.py):
     - extracted_product_info.csv           Created by the archived single-script
                                            version (one row per HTML file, not
                                            combined per product)
     - extracted_product_info_clean.csv     Same without extracted_passages column
     - consolidated_product_info.csv        One row per product, consolidated from
                                            the above by 4_analyze_scraped_products.py

   Intermediate files (in Data_wayback/intermediate_files/):
     Logs:
       - extract_context_log.txt         Log from 3a_extract_context.py
       - extract_log.txt                 Log from 3b_extract_product_info.py
       - extract_run.log                 Full-run log from extraction

6. SCRAPED PRODUCT ANALYSIS (Analysis_scraped/)
   Consolidates the legacy extracted_product_info.csv (one row per HTML file)
   into one row per product, merges category info from the master timeline,
   and produces summary statistics, interest rate figures, and LaTeX tables.

   NOTE: This analysis is based on the legacy extraction output from the
   archived Code/Code_archive/extract_product_info.py. It will need to be
   updated once the new v2 extraction pipeline (3a + 3b) is run.

   Script: Code/4_analyze_scraped_products.py
   (combines the former analyze_products.py and make_figures.py)

   Output:
     - Data_cleaned/consolidated_product_info.csv      Consolidated CSV
     - Analysis_scraped/interest_rates_by_coarse_category.png
     - Analysis_scraped/interest_rates_depository.tex
     - Analysis_scraped/interest_rates_financial.tex
     - Analysis_scraped/rate_breakdown_banking_consumer.tex
     - Analysis_scraped/bank_specialization.tex

