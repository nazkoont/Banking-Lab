import pandas as pd
import re

# load extraction output and master timeline
df = pd.read_csv('Data_cleaned/consolidated_product_info_clean.csv', on_bad_lines='skip')
master = pd.read_csv('Data_cleaned/fintech_timelines_master.csv')

# join product_broad_category from master (Savings/Borrowing/Payments/Credit Cards)
lookup = master[['company_name','product_name','product_broad_category']].drop_duplicates()
df = df.merge(lookup, left_on=['company','product_name'],
              right_on=['company_name','product_name'], how='left').drop(columns='company_name')

# parse interest_rate to numeric, strip % signs
df['rate_numeric'] = pd.to_numeric(
    df['interest_rate'].astype(str).str.replace('%','').str.strip(),
    errors='coerce'
)

# birth category: firm's industry at founding
firm_birth = (
    master[master['entry_type'] == 'Company'][['company_name','product_category']]
    .drop_duplicates()
)
firm_birth.columns = ['company', 'birth_category']

cats = ['Savings', 'Borrowing', 'Payments', 'Credit Cards']

# ── FIRM PANEL (specific) ─────────────────────────────────────────────────────
# one row per (company, half-year period)
# has_baas, n_with_baas, has_X_baas all use bank_partner_specific only
# n_baas_partners counts unique specific bank partners
rows = []
for (company, period), grp in df.groupby(['company', 'period']):

    # 1 if product has a named specific bank partner, 0 otherwise
    has_bank = grp['bank_partner_specific'].notna()

    row = {
        'company':               company,
        'period':                period,
        'has_financial_service': 1,
        'n_financial_services':  len(grp),
        'n_with_baas':           has_bank.sum(),
        'n_baas_partners':       grp['bank_partner_specific'].dropna().nunique(),
        'baas_partners':         '; '.join(grp['bank_partner_specific'].dropna().unique()),
        'has_baas':              int(has_bank.any()),
    }

    for cat in cats:
        cat_key = cat.lower().replace(' ', '_')
        in_cat  = grp['product_broad_category'] == cat

        row[f'has_{cat_key}']      = int(in_cat.any())
        row[f'has_{cat_key}_baas'] = int((in_cat & has_bank).any())

        # mean rate among non-missing rows in this category this period
        rates = grp.loc[in_cat, 'rate_numeric'].dropna()
        row[f'avg_rate_{cat_key}'] = rates.mean() if len(rates) > 0 else None

    rows.append(row)

firm_panel = pd.DataFrame(rows)
firm_panel = firm_panel.merge(firm_birth, on='company', how='left')
firm_panel.to_csv('Data_cleaned/firm_time_panel_specific.csv', index=False)
print(f"Firm specific: {len(firm_panel)} rows, {firm_panel['company'].nunique()} companies")

# ── BANK PANEL (specific) ─────────────────────────────────────────────────────
# one row per (bank, half-year period)
# includes subcategory firm counts and rate mean/std per category

def normalize_bank(name):
    # strip common legal suffixes that clutter bank names
    name = re.sub(r',?\s*Member FDIC', '', str(name), flags=re.IGNORECASE)
    name = re.sub(r',?\s*N\.?A\.?$', '', name, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', name).strip()

bank_rows = df[df['bank_partner_specific'].notna()].copy()
bank_rows['bank_norm'] = bank_rows['bank_partner_specific'].apply(normalize_bank)

rows = []
for (bank, period), grp in bank_rows.groupby(['bank_norm', 'period']):
    row = {
        'bank_name': bank,
        'period':    period,
        'n_firms':   grp['company'].nunique(),
        'firms':     '; '.join(grp['company'].unique()),
    }
    for cat in cats:
        cat_key = cat.lower().replace(' ', '_')
        in_cat  = grp['product_broad_category'] == cat

        row[f'n_firms_{cat_key}'] = grp[in_cat]['company'].nunique()
        row[f'firms_{cat_key}']   = '; '.join(grp[in_cat]['company'].unique())

        # rate mean and std among non-missing rows in this category
        rates = grp.loc[in_cat, 'rate_numeric'].dropna()
        row[f'avg_rate_{cat_key}'] = rates.mean() if len(rates) > 0 else None
        row[f'std_rate_{cat_key}'] = rates.std()  if len(rates) > 1 else None

    rows.append(row)

bank_panel = pd.DataFrame(rows)
bank_panel.to_csv('Data_cleaned/bank_time_panel_specific.csv', index=False)
print(f"Bank specific: {len(bank_panel)} rows, {bank_panel['bank_name'].nunique()} banks")
