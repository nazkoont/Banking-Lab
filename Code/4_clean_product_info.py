"""
4_clean_product_info.py
=======================
Clean the v2 extraction output (Data_cleaned/extracted_product_info_v2.csv).
Applies manual interest-rate corrections, merges category info from the master
timeline (exact + fuzzy match), classifies products into tiers, and excludes
flagged entries (rebrands, features, variants, non-US).

Output:
  Data_cleaned/consolidated_product_info.csv        Full consolidated CSV
  Data_cleaned/consolidated_product_info_clean.csv  Same without context columns
"""

import csv
import os
import sys

import pandas as pd
import numpy as np
import re

csv.field_size_limit(sys.maxsize)

# Ensure working directory is the project root (parent of Code/)
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("Data_cleaned/extracted_product_info_v2.csv",
                 engine="python", on_bad_lines="skip")
print(f"Loaded: {len(df)} rows, {df['company'].nunique()} companies, "
      f"{df.groupby(['company','product_name']).ngroups} unique products\n")

# ── Manual interest_rate corrections (parsing errors) ────────────────────────
INTEREST_RATE_CORRECTIONS = [
    # Square Payments: "2.6% + 15¢ per transaction" parsed as 260 (transaction fee, not APR)
    ('Block', 'Square Payments', None),
    # Affirm Card: scraped "0% APR" refers to BNPL split-pay feature, not deposit APY
    ('Affirm', 'Affirm Card', None),
    # Beam Visa Card: corporate/charge card, moved to Credit Cards; scraped 0 is autopay charge feature
    ('Beam', 'Beam Visa Card', None),
    # Depository batch 1 — scraped 0 but description says "variable" without a number
    ('Betterment', 'Betterment Everyday Savings', None),
    ('BrioDirect', 'BrioDirect High-Yield Savings Account', None),
    # Depository batch 2 — debit card, scraped rate about purchases, not deposit APY
    ('Earnin', 'EarnIn Card', None),
    # Depository batch 3 — HSA debit card, scraped rate is about cash advances (credit), not deposit
    ('Lively', 'Lively HSA Debit Card', None),
    # Depository batch 4 — variable/tiered with no number in scrape
    ('SmartyPig', 'SmartyPig Goal-Based Savings Account', None),
    ('Unifimoney', 'Unifimoney B2C Neobank App', None),
    ('Vanguard', 'Vanguard Cash Plus Account', None),
    # Upgrade Rewards Checking: 3.05% is on paired Performance Savings, not this product
    ('Upgrade', 'Rewards Checking Account', None),
    # Depository batch 5
    ('Wise', 'TransferWise Borderless Account', None),  # 0.26% is a fee, not rate
    ('Zynlo Bank', 'Tomorrow Savings Account', None),
    # Depository batch 6 (fuzzy-matched names from scraped data)
    ('Amscot', 'Azulos Prepaid Mastercard by Netspend Ouro', 6.00),
    ('Baselane', 'Baselane Savings Accounts High-Yield', 1.25),
    ('Brex', 'Brex Cash Business Account', None),
    # Depository batch 7
    ('Greenlight', 'Greenlight Savings Parent-Paid Interest', None),
    ('NerdWallet', 'Cash Account via Atomic Brokerage', None),
    # Depository batch 8
    ('Zynlo Bank', 'Certificates of Deposit CDs', None),
    # Financial (Lending Consumer) batch 1
    ('Amscot', 'Installment Cash Advance', None),  # template placeholder in scraped rate
    # Financial (Lending Consumer) batch 2
    ('Build', 'Build Credit Builder Loan', None),  # scraped rate is about locked savings, not loan APR
    # Financial (Lending Consumer) batch 4
    ('LendKey', 'Private Student Loans', None),
    ('LendingUSA', 'LendingUSA Direct Personal Loans', None),
    ('M1', 'M1 Borrow Margin Lending', None),
    ('NetCredit', 'NetCredit Line of Credit', None),
    ('Opploans', 'OppLoans Installment Loans', None),
    # Financial (Lending Consumer) batch 5
    ('Rise Credit', 'RISE Installment Loans', None),
    # Financial (Credit Cards) batch 1
    ('Apple', 'Apple Card', 17.74),  # scraped "no interest" marketing; actual purchase APR
    ('Block', 'Square Credit Card', None),  # prepaid debit, rate about deposit not credit
    ('Capital on Tap', 'Preloading', None),  # about prefunded balance, not credit APR
    # Financial (Credit Cards) batch 2
    ('Gemini', 'Gemini Credit Card', None),  # CC has variable APR, scraped desc is wrong context
    # Financial (Credit Cards) batch 3
    ('SoFi', 'SoFi Essential Credit Card', None),  # scraped desc about cash advances/BT, not purchase APR
    ('SoFi', 'SoFi Everyday Cash Rewards Credit Card', None),
    # Financial (Investing)
    ('Betterment', 'Betterment Premium Plan', None),  # 0.25 is a boost amount, not APY
    ('Greenlight', 'Greenlight Invest Kids Investing', None),  # 1% is cashback, not interest
    ('Public', 'Public Options Trading', None),  # options = derivatives, rate N/A
    ('Wealthfront', 'Wealthfront Cash Account', None),  # no number in scrape
    # Financial (Payments) — merchant processing fees & non-applicable wallets, not interest rates
    ('Apple', 'Apple Pay', None),
    ('Cardpayment Services', 'Merchant Payment Processing', None),
    ('Imerchant Direct', 'Credit Card Processing', None),
    ('Imerchant Direct', 'Flat Rate Processing', None),
    ('Melio', 'Pay by Card', None),
    ('Pcbancard', 'Level 2 3 B2B Processing', None),
    # Financial (Lending Business)
    ('Highbeam', 'Highbeam Capital Lines of Credit Cash Advances', None),
    ('Intuit', 'QuickBooks Capital', None),
    ('National Partners', 'Insurance Premium Financing', None),  # 2.9% is about card down payment, not loan rate
    ('Panacea Financial', 'Panacea Practice Solutions', None),  # 0.25% is a discount
    ('Stripe', 'Stripe Capital', None),  # MCA holdback, not APR
    # Financial (small subcats)
    ('Fidelity Investments', 'Fidelity HSA Health Savings Account', None),
    ('Every', 'Every for Treasury', None),
]
for _c, _p, _v in INTEREST_RATE_CORRECTIONS:
    df.loc[(df['company'] == _c) & (df['product_name'] == _p), 'interest_rate'] = _v

# ── Filter to substantive rows ───────────────────────────────────────────────
INFO_COLS = ["product_type", "interest_rate", "interest_rate_description",
             "fees", "bank_partner_specific", "bank_partner_broad",
             "baas_middleware_specific", "baas_middleware_broad", "key_features"]

# Only keep rows with status OK
df = df[df["status"] == "OK"].copy()
print(f"After status=OK filter: {len(df)} rows")

# Count populated info fields
def _is_real(val):
    if pd.isna(val):
        return False
    s = str(val).strip()
    return bool(s)

df["fields_populated"] = df[INFO_COLS].apply(
    lambda row: sum(1 for v in row if _is_real(v)), axis=1)

# Flag substance
df["has_substance"] = df["fields_populated"] > 0
n_no_sub = (~df["has_substance"]).sum()
print(f"Substance check: {n_no_sub} products have no scraped info (all fields blank)\n")

# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY MAPPING — merge from fintech_timelines_master.csv
# ══════════════════════════════════════════════════════════════════════════════
master = pd.read_csv("Data_cleaned/fintech_timelines_master.csv")
for flag in ["is_excluded", "is_rebrand", "is_feature", "is_variant", "is_non_us"]:
    if flag not in master.columns:
        master[flag] = False
master_cats = master[["company_name", "product_name", "product_category", "product_subcategory",
                      "is_excluded", "is_rebrand", "is_feature", "is_variant", "is_non_us"]].drop_duplicates(
    subset=["company_name", "product_name"], keep="first"
)

# Step 1: Exact merge
df = df.merge(
    master_cats,
    left_on=["company", "product_name"],
    right_on=["company_name", "product_name"],
    how="left",
).drop(columns=["company_name"])

exact_matched = df["product_category"].notna().sum()
print(f"  Exact match: {exact_matched}")

# Step 2: Fuzzy match for remaining unmatched products
from thefuzz import fuzz

def normalize_for_matching(name):
    n = re.sub(r'[^a-zA-Z0-9 ]', ' ', str(name).lower())
    return re.sub(r'\s+', ' ', n).strip()

unmatched_mask = df["product_category"].isna()

master_by_company = {}
for _, row in master_cats.iterrows():
    co = row["company_name"]
    if co not in master_by_company:
        master_by_company[co] = []
    master_by_company[co].append(row)

master_company_list = list(master_by_company.keys())
company_alias = {}
for co in df.loc[unmatched_mask, "company"].unique():
    if co in master_by_company:
        continue
    co_norm = normalize_for_matching(co)
    best_score = 0
    best_match = None
    for mc in master_company_list:
        score = fuzz.token_sort_ratio(co_norm, normalize_for_matching(mc))
        if score > best_score:
            best_score = score
            best_match = mc
    if best_score >= 90:
        company_alias[co] = best_match

if company_alias:
    print(f"  Company fuzzy matches: {len(company_alias)}")
    for k, v in company_alias.items():
        print(f"    '{k}' -> '{v}'")

fuzzy_count = 0
fuzzy_matches = []
for idx in df.index[unmatched_mask]:
    co = df.at[idx, "company"]
    scraped_name = df.at[idx, "product_name"]
    lookup_co = co if co in master_by_company else company_alias.get(co)
    if lookup_co is None:
        continue
    scraped_norm = normalize_for_matching(scraped_name)
    best_score = 0
    best_row = None
    for mrow in master_by_company[lookup_co]:
        master_norm = normalize_for_matching(mrow["product_name"])
        score = fuzz.token_sort_ratio(scraped_norm, master_norm)
        if score > best_score:
            best_score = score
            best_row = mrow
    if best_score >= 70:
        df.at[idx, "product_category"] = best_row["product_category"]
        df.at[idx, "product_subcategory"] = best_row["product_subcategory"]
        for flag in ["is_excluded", "is_rebrand", "is_feature", "is_variant", "is_non_us"]:
            df.at[idx, flag] = best_row.get(flag, False)
        fuzzy_count += 1
        if best_score < 90:
            fuzzy_matches.append((scraped_name, best_row["product_name"], best_score))

print(f"  Fuzzy match (>=70): {fuzzy_count}")
total_matched = df["product_category"].notna().sum()
print(f"  Total matched: {total_matched}/{len(df)} ({total_matched/len(df):.0%})")
still_unmatched = df[df["product_category"].isna()]
print(f"  Still unmatched: {len(still_unmatched)}")
if fuzzy_matches:
    print(f"\n  Low-confidence fuzzy matches (score < 90):")
    for s, m, sc in sorted(fuzzy_matches, key=lambda x: x[2])[:15]:
        print(f"    {sc}: '{s}' -> '{m}'")

# ── Tier classification ──────────────────────────────────────────────────────
DEPOSITORY_SUBCATS = {'Banking (Consumer)', 'Banking (Business)'}
FINANCIAL_SUBCATS = {
    'Payments', 'Money Transfer', 'Bill Pay',
    'Lending (Consumer)', 'Lending (Business)', 'Credit Cards',
    'Credit Building', 'Investing', 'Crypto/Digital Assets',
    'Treasury Management', 'Alternative Investments',
    'Insurance & Benefits', 'Other Financial Services', 'Gambling/Gaming',
    'Spend Management', 'Tax Prep',
}
FINANCIAL_INFRA_SUBCATS = {
    'Payments API', 'Point-of-Sale',
}
NON_FINANCIAL_CATS = {
    'Software & SaaS', 'Infrastructure & Developer Tools',
    'Marketplace & Platform', 'Commerce & Retail', 'Media & Content',
    'Advertising & Marketing', 'Hardware & Devices', 'Other Services',
}
FINANCIAL_DETAIL = {
    'Payments': 'Payments', 'Money Transfer': 'Payments',
    'Payments API': 'Payments', 'Bill Pay': 'Payments',
    'Point-of-Sale': 'Payments',
    'Lending (Consumer)': 'Lending/Credit', 'Lending (Business)': 'Lending/Credit',
    'Credit Cards': 'Lending/Credit', 'Credit Building': 'Lending/Credit',
    'Investing': 'Investing', 'Crypto/Digital Assets': 'Investing',
    'Treasury Management': 'Investing', 'Alternative Investments': 'Investing',
    'Insurance & Benefits': 'Investing',
    'Other Financial Services': 'Other',
    'Gambling/Gaming': 'Other',
    'Spend Management': 'Other',
    'Tax Prep': 'Other',
}

def assign_tier(row):
    subcat = row.get("product_subcategory", "")
    cat = row.get("product_category", "")
    if pd.notna(subcat) and subcat in DEPOSITORY_SUBCATS:
        return "Depository"
    if pd.notna(subcat) and subcat in FINANCIAL_INFRA_SUBCATS:
        return "Financial Infrastructure"
    if pd.notna(subcat) and subcat in FINANCIAL_SUBCATS:
        return "Financial"
    if pd.notna(cat) and cat in NON_FINANCIAL_CATS:
        return "Non-financial"
    if cat in ("Financial Services", "Investment & Capital"):
        return "Financial"
    return np.nan

def assign_fin_detail(row):
    subcat = row.get("product_subcategory", "")
    if pd.notna(subcat) and subcat in FINANCIAL_DETAIL:
        return FINANCIAL_DETAIL[subcat]
    return np.nan

df["category_coarse"] = df.apply(assign_tier, axis=1)
df["category_financial_detail"] = df.apply(assign_fin_detail, axis=1)

# Apply exclusion flags: exclude flagged entries (rebrands, features, variants, excluded)
for flag in ["is_excluded", "is_rebrand", "is_feature", "is_variant", "is_non_us"]:
    df[flag] = df[flag].fillna(False)
excluded_companies = set(master[master.get('is_excluded', False) == True]['company_name'].unique())
df.loc[df['company'].isin(excluded_companies), 'is_excluded'] = True

flagged_mask = df["is_excluded"] | df["is_rebrand"] | df["is_feature"] | df["is_variant"] | df["is_non_us"]
n_flagged = flagged_mask.sum()
if n_flagged > 0:
    df = df[~flagged_mask].copy()
    print(f"Excluded {n_flagged} flagged products (rebrand/feature/variant/excluded)")

# ── Save consolidated CSV ────────────────────────────────────────────────────
out_file = "Data_cleaned/consolidated_product_info.csv"
df.to_csv(out_file, index=False)
print(f"\nSaved to {out_file}")

CONTEXT_COLS = ["relevant_context", "combined_context", "source_files"]
clean_file = "Data_cleaned/consolidated_product_info_clean.csv"
df.drop(columns=[c for c in CONTEXT_COLS if c in df.columns]).to_csv(clean_file, index=False)
print(f"Saved to {clean_file}\n")
