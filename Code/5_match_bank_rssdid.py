import pandas as pd
import re
from rapidfuzz import process, fuzz

PANEL_CSV = "Data_cleaned/bank_time_panel.csv"
LOOKUP_CSV = "Data_cleaned/matched_names.csv"
MATCH_THRESHOLD = 90

def normalize(name):
    if not isinstance(name, str):
        return ""
    name = re.sub(r',?\s*Member FDIC', '', name, flags=re.IGNORECASE)
    name = re.sub(r',?\s*N\.?A\.?$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\b(bank|trust|company|corp|inc|llc|fsb|ssb|dba)\b', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[^a-z0-9 ]', ' ', name.lower())
    return re.sub(r'\s+', ' ', name).strip()

panel  = pd.read_csv(PANEL_CSV)
lookup = pd.read_csv(LOOKUP_CSV)

lookup['norm'] = lookup['partnername'].apply(normalize)
lookup_names = lookup['norm'].tolist()
lookup_dict  = dict(zip(lookup['norm'], lookup['idrssd']))
lookup_raw   = dict(zip(lookup['norm'], lookup['partnername']))

unique_banks = panel['bank_name'].dropna().unique()
print(f"Unique bank names to match: {len(unique_banks)}")

results = {}
for raw_name in unique_banks:
    norm = normalize(raw_name)
    if not norm:
        results[raw_name] = {'idrssd': None, 'matched_name': None, 'match_score': None, 'match_status': 'EMPTY'}
        continue
    match = process.extractOne(norm, lookup_names, scorer=fuzz.token_sort_ratio)
    if match and match[1] >= MATCH_THRESHOLD:
        matched_norm = match[0]
        results[raw_name] = {
            'idrssd': lookup_dict.get(matched_norm),
            'matched_name': lookup_raw.get(matched_norm),
            'match_score': match[1],
            'match_status': 'MATCHED'
        }
    else:
        results[raw_name] = {
            'idrssd': None,
            'matched_name': match[0] if match else None,
            'match_score': match[1] if match else None,
            'match_status': 'UNMATCHED'
        }

matched   = sum(1 for v in results.values() if v['match_status'] == 'MATCHED')
unmatched = sum(1 for v in results.values() if v['match_status'] == 'UNMATCHED')
print(f"Matched:   {matched}/{len(unique_banks)} ({matched/len(unique_banks):.1%})")
print(f"Unmatched: {unmatched}/{len(unique_banks)} ({unmatched/len(unique_banks):.1%})")
print()
print("Unmatched bank names:")
for name, r in {k:v for k,v in results.items() if v['match_status']=='UNMATCHED'}.items():
    print(f"  '{name}' (best: '{r['matched_name']}', score: {r['match_score']})")

panel['idrssd']       = panel['bank_name'].map(lambda x: results.get(x, {}).get('idrssd'))
panel['matched_name'] = panel['bank_name'].map(lambda x: results.get(x, {}).get('matched_name'))
panel['match_score']  = panel['bank_name'].map(lambda x: results.get(x, {}).get('match_score'))
panel['match_status'] = panel['bank_name'].map(lambda x: results.get(x, {}).get('match_status', 'NULL'))

panel.to_csv(PANEL_CSV, index=False)
print(f"\nSaved. Rows with idrssd: {panel['idrssd'].notna().sum()}/{len(panel)}")
