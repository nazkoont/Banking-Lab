import pandas as pd
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_PATH   = 'Qwen/Qwen2.5-7B-Instruct'
INPUT_CSV    = 'Data_cleaned/fintech_timelines_master.csv'
OUTPUT_CSV   = 'Data_cleaned/nonfinancial_classified.csv'
CHECKPOINT   = 'Data_wayback/intermediate_files/nonfinancial_classify_checkpoint.json'

TAXONOMY = [
    'Software & SaaS',
    'Infrastructure & Developer Tools',
    'Marketplace & Platform',
    'Commerce & Retail',
    'Advertising & Marketing',
    'Hardware & Devices',
    'Investment & Capital',
    'Media & Content',
    'Crypto/Digital Assets',
    'Identity & Compliance',
    'Data & Analytics',
    'Money Transfer',
    'Other',
]

SYSTEM_PROMPT = f"""You are a financial technology industry analyst. 
You will be given a product name, entry type, and description for a non-financial product from a fintech or technology company.
Classify it into exactly one category from this list:
{chr(10).join(f'- {t}' for t in TAXONOMY)}

Respond only with valid JSON in this exact format, no preamble:
{{"category": "<category from list>", "summary": "<one sentence description under 20 words>"}}"""

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map='auto')
model.eval()
print("Model loaded.")

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
master = pd.read_csv(INPUT_CSV)
fin_cats = ['Savings', 'Borrowing', 'Payments', 'Credit Cards']
master = master[~master['company_name'].str.match(r'^\d{4}$', na=False)].copy()
non_fin = master[~master['product_broad_category'].isin(fin_cats)].copy()
non_fin = non_fin[non_fin['product_history_and_description'].notna()].reset_index(drop=True)
print(f"Rows to classify: {len(non_fin)}")

# load checkpoint if exists
if os.path.exists(CHECKPOINT):
    with open(CHECKPOINT) as f:
        results = json.load(f)
    print(f"Resuming from checkpoint: {len(results)} done")
else:
    results = {}

# ── CLASSIFY ──────────────────────────────────────────────────────────────────
def classify_row(row):
    user_msg = (
        f"Company: {row['company_name']}\n"
        f"Product: {row['product_name']}\n"
        f"Entry type: {row['entry_type']}\n"
        f"Existing category: {row['product_category']}\n"
        f"Description: {str(row['product_history_and_description'])[:500]}"
    )
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user',   'content': user_msg},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=80, temperature=0.1, do_sample=False)
    decoded = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    try:
        parsed = json.loads(decoded)
        cat = parsed.get('category', 'Other')
        summary = parsed.get('summary', '')
        # validate category is in taxonomy
        if cat not in TAXONOMY:
            cat = 'Other'
        return cat, summary, decoded
    except Exception:
        return 'Other', '', decoded

for i, row in non_fin.iterrows():
    key = f"{row['company_name']}||{row['product_name']}"
    if key in results:
        continue
    cat, summary, raw = classify_row(row)
    results[key] = {'category': cat, 'summary': summary, 'raw': raw}

    # save checkpoint every 50 rows
    if len(results) % 50 == 0:
        with open(CHECKPOINT, 'w') as f:
            json.dump(results, f)
        print(f"  {len(results)}/{len(non_fin)} classified")

# final checkpoint save
with open(CHECKPOINT, 'w') as f:
    json.dump(results, f)

# ── MERGE AND SAVE ────────────────────────────────────────────────────────────
non_fin['_key'] = non_fin['company_name'].astype(str) + '||' + non_fin['product_name'].astype(str)
non_fin['llm_category'] = non_fin['_key'].map(lambda k: results.get(k, {}).get('category'))
non_fin['llm_summary']  = non_fin['_key'].map(lambda k: results.get(k, {}).get('summary'))

out = non_fin[['company_name','product_name','entry_type','year_launched','end_year',
               'product_category','llm_category','llm_summary','product_history_and_description']].copy()
out.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV} ({len(out)} rows)")
