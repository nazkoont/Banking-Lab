#!/usr/bin/env python3
"""
Computes the share of financial products with scraped content per 6-month period.
Run from fintech_timelines/:
    python3 Code/url_discovery_timeseries.py
Outputs:
    Data_wayback/url_discovery_timeseries.csv
    Data_wayback/url_discovery_timeseries.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import importlib.util, os

# ── Load utils ────────────────────────────────────────────────────────────────
spec = importlib.util.spec_from_file_location("wayback_utils", "Code/2_wayback_utils.py")
wu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wu)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
master = pd.read_csv("Data_cleaned/fintech_timelines_master.csv")
pp     = pd.read_csv("Data_wayback/product_pages.csv", on_bad_lines='skip')

# ── Filter master to financial products 2005+ ─────────────────────────────────
products = master[master['entry_type'] == 'Product'].copy()
products['year_launched'] = pd.to_numeric(products['year_launched'], errors='coerce')
products = products[products['year_launched'] >= 2005]
products['is_financial'] = products.apply(wu.is_financial_product, axis=1)
financial = products[products['is_financial']].copy()
print(f"Total financial products (2005+): {len(financial)}")

# ── Build 6-month periods ─────────────────────────────────────────────────────
periods = []
for year in range(2005, 2026):
    for month in [1, 7]:
        periods.append((year, month))

# ── For each period, count products with content ──────────────────────────────
# A product "has content" in a period if it has at least one row in product_pages
# with text_length > 0 AND target_year/month <= period cutoff
pp_ok = pp[pp['text_length'] > 0].copy()
pp_ok['target_year']  = pd.to_numeric(pp_ok['target_year'],  errors='coerce')
pp_ok['target_month'] = pd.to_numeric(pp_ok['target_month'], errors='coerce')
pp_ok = pp_ok.dropna(subset=['target_year', 'target_month'])
pp_ok['period_val'] = pp_ok['target_year'] * 12 + pp_ok['target_month']

# unique (company, product) pairs in financial master
fin_pairs = set(zip(financial['company_name'], financial['product_name']))

rows = []
for year, month in periods:
    cutoff = year * 12 + month

    # products launched by this period
    launched = financial[financial['year_launched'] <= year]
    n_total = len(launched)
    if n_total == 0:
        continue

    # products with scraped content up to this period
    pp_period = pp_ok[pp_ok['period_val'] <= cutoff]
    scraped_pairs = set(zip(pp_period['company_name'], pp_period['product_name']))

    # intersect with financial products launched by this period
    launched_pairs = set(zip(launched['company_name'], launched['product_name']))
    found = launched_pairs & scraped_pairs

    rows.append({
        'period': f"{year}-{'01' if month == 1 else '07'}",
        'year': year,
        'month': month,
        'n_launched': n_total,
        'n_found': len(found),
        'share_found': len(found) / n_total if n_total > 0 else 0,
    })

df = pd.DataFrame(rows)
df.to_csv("Data_wayback/url_discovery_timeseries.csv", index=False)
print(f"\nSaved to Data_wayback/url_discovery_timeseries.csv")
print(df[['period','n_launched','n_found','share_found']].tail(10).to_string(index=False))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(14, 5))

ax1.plot(df['period'], df['share_found'] * 100, color='steelblue', linewidth=2, marker='o', markersize=4, label='% products with content')
ax1.set_ylabel('Share of launched products with scraped content (%)', color='steelblue')
ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.set_ylim(0, 105)

ax2 = ax1.twinx()
ax2.bar(df['period'], df['n_launched'], alpha=0.15, color='gray', label='# products launched')
ax2.set_ylabel('Cumulative financial products launched', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')

# x-axis ticks — show every Jan
jan_ticks = df[df['month'] == 1]['period'].tolist()
ax1.set_xticks(jan_ticks)
ax1.set_xticklabels(jan_ticks, rotation=45, ha='right', fontsize=8)

ax1.set_title('URL Discovery Rate Over Time\nShare of financial products with scraped content, by 6-month period', fontsize=12)
ax1.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig("Data_wayback/url_discovery_timeseries.png", dpi=150)
print("Saved to Data_wayback/url_discovery_timeseries.png")
