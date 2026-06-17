import pandas as pd
import re

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv('Data_cleaned/consolidated_product_info_clean.csv', on_bad_lines='skip')
master = pd.read_csv('Data_cleaned/fintech_timelines_master.csv')

# -----------------------------
# FIRM PANEL
# -----------------------------
firm_birth = (
    master[master['entry_type'] == 'Company'][['company_name', 'product_category']]
    .drop_duplicates()
)
firm_birth.columns = ['company', 'birth_category']

cats = ['Savings', 'Borrowing', 'Payments', 'Credit Cards']

rows = []
for (company, period), grp in df.groupby(['company', 'period']):
    broad = grp['product_broad_category'].dropna()
    has_bank = grp['bank_partner_specific'].notna() | grp['bank_partner_broad'].notna()

    row = {
        'company': company,
        'period': period,
        'has_financial_service': 1,
        'n_financial_services': len(grp),
        'n_with_baas': has_bank.sum(),
        'n_baas_partners': grp['bank_partner_specific'].dropna().nunique(),
        'baas_partners': '; '.join(grp['bank_partner_specific'].dropna().unique()),
        'has_baas': int(has_bank.any()),
    }

    for cat in cats:
        cat_key = cat.lower().replace(' ', '_')
        in_cat = broad == cat
        row[f'has_{cat_key}'] = int(in_cat.any())
        row[f'has_{cat_key}_baas'] = int((in_cat & has_bank).any())

    rows.append(row)

firm_panel = pd.DataFrame(rows)
firm_panel = firm_panel.merge(firm_birth, on='company', how='left')
firm_panel.to_csv('Data_cleaned/firm_time_panel.csv', index=False)

print(f"Firm-time panel: {len(firm_panel)} rows, {firm_panel['company'].nunique()} companies")
print(firm_panel.head(3).to_string())


# -----------------------------
# BANK PANEL
# -----------------------------
def normalize_bank(name):
    name = re.sub(r',?\s*Member FDIC', '', str(name), flags=re.IGNORECASE)
    name = re.sub(r',?\s*N\.?A\.?', '', name, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', name).strip()

bank_rows = df[df['bank_partner_specific'].notna()].copy()
bank_rows['bank_norm'] = bank_rows['bank_partner_specific'].apply(normalize_bank)

rows = []
for (bank, period), grp in bank_rows.groupby(['bank_norm', 'period']):
    row = {
        'bank_name': bank,
        'period': period,
        'n_firms': grp['company'].nunique(),
        'firms': '; '.join(grp['company'].unique()),
    }

    for cat in cats:
        cat_key = cat.lower().replace(' ', '_')
        in_cat = grp['product_broad_category'] == cat
        row[f'n_firms_{cat_key}'] = grp[in_cat]['company'].nunique()
        row[f'firms_{cat_key}'] = '; '.join(grp[in_cat]['company'].unique())

    rows.append(row)

bank_panel = pd.DataFrame(rows)
bank_panel.to_csv('Data_cleaned/bank_time_panel.csv', index=False)

print(f"Bank-time panel: {len(bank_panel)} rows, {bank_panel['bank_name'].nunique()} unique banks")
print(bank_panel.head(3).to_string())
