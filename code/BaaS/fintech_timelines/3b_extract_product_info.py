"""
3b_extract_product_info.py
==========================
Two-pass LLM extraction pipeline:

  Pass 1 (filter): For each chunk of combined context, the LLM selects and
  trims text to only what pertains to the specific product of interest.
  Verbatim copying only — no rephrasing or additions. Broad context (e.g.
  company-wide banking disclosures) is kept but flagged.

  Pass 2 (extract): The filtered, product-specific context is sent to the
  LLM for structured field extraction (product type, rates, fees, bank
  partner, BaaS middleware, key features).

Reads product_context.csv (from 3a_extract_context.py) and writes
extracted_product_info_v2.csv.

Supports resuming — skips products already in the output CSV.

Usage:
    python3 3b_extract_product_info.py [--limit N] [--company COMPANY]
"""

import argparse
import csv
import io
import json
import logging
import os
import re
import sys

csv.field_size_limit(sys.maxsize)
import time

import torch
from transformers import pipeline as hf_pipeline

# Ensure working directory is the project root (parent of Code/)
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# ── Config ───────────────────────────────────────────────────────────────────

MODEL = "Qwen/Qwen2.5-7B-Instruct"

print("Loading model onto GPU...", flush=True)
_pipe = hf_pipeline(
    "text-generation",
    model=MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
)
INPUT_CSV = "Data_cleaned/product_context.csv"
OUTPUT_CSV = "Data_cleaned/extracted_product_info_v2.csv"
LOG_FILE = "Data_wayback/intermediate_files/extract_log.txt"

# Max characters per LLM chunk
MAX_TEXT_CHARS = 8000

CSV_COLUMNS = [
    "company",
    "product_name",
    "period",
    "product_type",
    "interest_rate",
    "interest_rate_description",
    "fees",
    "bank_partner_specific",
    "bank_partner_broad",
    "baas_middleware_specific",
    "baas_middleware_broad",
    "key_features",
    "status",
    "relevant_context",
    "context_scope",
    "combined_context",
    "page_types",
    "snapshot_dates",
    "num_source_files",
    "source_files",
]

# ── LLM prompts ─────────────────────────────────────────────────────────────

FILTER_SYSTEM_PROMPT = """\
You are a precise text filter. You will be given text scraped from a fintech \
company's website and the name of a specific product. Your job is to select \
ONLY the parts of the text that pertain to that specific product.

Rules:
- Copy relevant text EXACTLY as it appears. Never add words, infer, or rephrase.
- You may remove irrelevant portions of a sentence. For example, from \
"Product A is issued by Bank X; Product B is issued by Bank Y", if the \
product of interest is Product A, output only "Product A is issued by Bank X".
- If a passage applies broadly to all of the company's products or banking \
services (e.g. a company-wide FDIC disclosure, a general terms page), \
ALWAYS include it verbatim. Examples of broad text to KEEP:
  - "Banking services provided by [Bank Name], Member FDIC"
  - "[Company] is a financial technology company, not a bank"
  - "Deposits are FDIC-insured through [Bank Name]"
  - Company-wide fee disclosures, privacy policies mentioning bank partners
- When a paragraph mixes broad company-wide text with text about a \
different specific product, keep the broad parts and drop only the parts \
that are specifically about the other product.
- Drop text about other specific products, navigation menus, job listings, \
and unrelated content entirely.
- If nothing in the text is relevant, return exactly: NONE
- CRITICAL: Any sentence containing ANY of these must ALWAYS be included verbatim, no matter what else you drop: "Member FDIC", "FDIC-insured", "issued by", "banking services provided by", "Bancorp", "Evolve Bank", "Thread Bank", "Blue Ridge", "Sutton Bank", "Celtic Bank", "WebBank", "Cross River", "Pathward", "MetaBank", "Stride Bank", "NBKC", "Piermont", "Coastal Community". These are legally required disclosures and must never be filtered out.

Return ONLY the selected text. No commentary, no labels, no explanation.\
"""

SCOPE_SYSTEM_PROMPT = """\
You are classifying context scope. Given text that has been filtered to a \
specific financial product, classify each passage's scope.

Return ONLY valid JSON (no markdown fences, no explanation):
{
  "scope": "specific" or "broad" or "mixed",
  "explanation": "one sentence explaining the classification"
}

- "specific": all or nearly all of the text is specifically about this product
- "broad": the text is mostly company-wide or applies to all products generally
- "mixed": some text is product-specific, some is broad/company-wide

Return raw JSON only. No commentary.\
"""

EXTRACT_SYSTEM_PROMPT = """\
You are a structured data extractor. Given text that has been filtered to \
contain only information about a specific financial product, extract the \
information into EXACTLY this JSON structure (no other keys):

{
  "product_type": "e.g. savings account, checking account, credit card, personal loan, cash advance, debit card, BNPL, investment account, HSA, etc.",
  "rates": [
    {"interest_rate": 4.24, "description": "fixed APR starting rate for student loan refinancing"}
  ],
  "fees": "any fees or fee-related information mentioned, including 'no fees', 'fee-free', specific fee amounts, and fee waivers. Empty string ONLY if fees are not discussed at all.",
  "bank_partner_specific": "FDIC-insured bank or issuing bank that the text explicitly names as providing or backing THIS specific product (e.g. 'issued by Stride Bank', 'Chime Savings Account is held at Bancorp Bank'). Empty string if none.",
  "bank_partner_broad": "FDIC-insured bank mentioned in broad company-wide context (e.g. 'banking services provided by Cross River Bank' without specifying which product). Empty string if none.",
  "baas_middleware_specific": "BaaS middleware or card processor that the text explicitly names as powering THIS specific product (e.g. 'Chime debit card powered by Marqeta'). Empty string if none.",
  "baas_middleware_broad": "BaaS middleware or card processor mentioned in broad company-wide context without specifying which product. Empty string if none.",
  "key_features": "comma-separated list of 3-8 key product features"
}

Rules:
- "interest_rate" must be a plain number in percentage points (e.g. 5.0 for \
5%, 24.99 for 24.99%). Always use percentage points — never decimals like \
0.05 for 5%. Use 0 for "no interest". Use empty array [] if no rates mentioned.
- "description" should include rate type (APY, APR, interest rate), what it \
applies to, and conditions/tiers.
- IMPORTANT: "no interest", "0% interest", "interest-free", or "no APR" IS \
a rate — you MUST return {"interest_rate": 0, "description": "no interest \
on cash advances"} (or similar). Do NOT put interest information in the fees field.
- For bank partners and BaaS middleware, distinguish between:
  - "specific": the text explicitly ties this name to THIS product by name
  - "broad": the name appears in a company-wide disclosure or general context \
without specifying which product it applies to
  - The same name may appear in both specific and broad if it is mentioned in both contexts.
- Extract ONLY information explicitly stated. Never guess.
- Return raw JSON only. No ```json fences. No commentary.\
"""


# ── LLM helpers ──────────────────────────────────────────────────────────────

def _strip_llm_response(raw):
    """Clean up LLM response: strip think tags, markdown fences."""
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    if raw.startswith("```"):
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
    return raw.strip()


def _call_llm(system_prompt, user_msg, num_ctx=16384):
    """Call local HuggingFace model and return raw response text."""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        output = _pipe(
            messages,
            max_new_tokens=2048,
            temperature=0.1,
            do_sample=True,
        )
        return output[0]["generated_text"][-1]["content"].strip()
    except Exception as e:
        print(f"    LLM error: {e}", flush=True)
        return None


def _parse_llm_json(raw):
    """Parse JSON from LLM response, handling fences and nested objects."""
    raw = _strip_llm_response(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# ── Chunking ─────────────────────────────────────────────────────────────────

def _chunk_text(text, max_chars=MAX_TEXT_CHARS):
    """Split text into chunks at sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = ""
    for sent in sentences:
        if current and len(current) + len(sent) + 1 > max_chars:
            chunks.append(current)
            current = sent
        else:
            current = current + " " + sent if current else sent
    if current:
        chunks.append(current)

    return chunks


# ── Pass 1: Filter context to product ────────────────────────────────────────

# Financial keywords for chunk pre-filtering. A chunk is sent to the LLM
# only if it contains one of these OR mentions the product name.
CHUNK_FINANCIAL_KEYWORDS = [
    'fdic', 'banking', 'apy', 'apr', 'overdraft', 'deposit',
    'issued by', 'member fdic', 'fdic-insured', 'bank partner',
    'banking services', 'provided by', 'bancorp', 'evolve bank',
    'thread bank', 'blue ridge', 'sutton bank', 'celtic bank',
    'webbank', 'cross river', 'pathward', 'metabank', 'stride bank',
    'nbkc', 'piermont', 'coastal community', 'issuing bank',
    'partner bank', 'insured by', 'fintech company, not a bank',
]

STOPWORDS = {"the", "a", "an", "for", "and", "or", "of", "by", "in", "on",
             "to", "with", "from", "at", "is", "it", "as"}


def _get_product_keywords(name):
    """Extract keywords from product name, splitting camelCase compounds."""
    words = re.sub(r'[^a-zA-Z0-9 ]', ' ', str(name)).split()
    expanded = []
    for w in words:
        parts = re.sub(r'([a-z])([A-Z])', r'\1 \2', w).split()
        expanded.extend(parts)
    return [w.lower() for w in expanded if w.lower() not in STOPWORDS and len(w) > 2]


def _chunk_is_relevant(chunk, product_name):
    """Check if a chunk mentions the product or contains financial keywords."""
    chunk_lower = chunk.lower()

    # Check financial keywords
    if any(kw in chunk_lower for kw in CHUNK_FINANCIAL_KEYWORDS):
        return True

    # Check product name (>=50% of keywords present)
    keywords = _get_product_keywords(product_name)
    if not keywords:
        return True
    chunk_normalized = re.sub(r'[^a-z0-9 ]', ' ', chunk_lower)
    found = sum(1 for kw in keywords if kw in chunk_normalized)
    return (found / len(keywords)) >= 0.5


def _trim_chunk(chunk, product_name):
    """
    Trim a chunk to sentences that contain financial keywords or mention the
    product name, plus one sentence before and after each match for context.
    """
    product_kws = _get_product_keywords(product_name)
    sentences = re.split(r'(?<=[.!?])\s+', chunk)

    # Find indices of matching sentences
    match_indices = set()
    for j, sent in enumerate(sentences):
        sent_lower = sent.lower()
        if any(kw in sent_lower for kw in CHUNK_FINANCIAL_KEYWORDS):
            match_indices.add(j)
            continue
        if product_kws:
            sent_normalized = re.sub(r'[^a-z0-9 ]', ' ', sent_lower)
            found = sum(1 for kw in product_kws if kw in sent_normalized)
            if found / len(product_kws) >= 0.5:
                match_indices.add(j)

    # Expand to include one sentence before and after each match
    keep_indices = set()
    for j in match_indices:
        keep_indices.update([j - 1, j, j + 1])

    kept = [sentences[j] for j in sorted(keep_indices) if 0 <= j < len(sentences)]
    return " ".join(kept) if kept else ""


def filter_context(text, company, product_name):
    """
    Pass 1: Send combined context through LLM to select and trim only the
    parts relevant to the specific product. Chunks if needed and combines.
    Chunks that don't mention the product or financial keywords are skipped.
    Chunks that pass are trimmed to relevant sentences (plus surrounding
    context) before sending to the LLM.
    Returns filtered text (or None if nothing relevant).
    """
    chunks = _chunk_text(text)
    filtered_parts = []

    for i, chunk in enumerate(chunks):
        # Pre-filter: skip chunks with no product mention or financial keywords
        if not _chunk_is_relevant(chunk, product_name):
            print(f"    Filter chunk {i+1}: skipped (no keywords/product mention)", flush=True)
            continue

        # Trim to relevant sentences + one before/after
        trimmed = _trim_chunk(chunk, product_name)
        if len(trimmed) < 20:
            print(f"    Filter chunk {i+1}: skipped (trimmed too short)", flush=True)
            continue
        chunk = trimmed

        msg = (
            f"Company: {company}\n"
            f"Product of interest: {product_name}\n\n"
            f"--- TEXT ---\n{chunk}\n"
            f"--- END ---\n\n"
            f"Select only the text relevant to \"{product_name}\". "
            f"Copy verbatim. Remove irrelevant parts."
        )
        raw = _call_llm(FILTER_SYSTEM_PROMPT, msg, num_ctx=16384)
        if raw is None:
            print(f"    Filter chunk {i+1} failed", flush=True)
            continue

        cleaned = _strip_llm_response(raw)
        if cleaned.upper() == "NONE" or len(cleaned.strip()) < 20:
            print(f"    Filter chunk {i+1}: no relevant text", flush=True)
            continue

        filtered_parts.append(cleaned)

    if not filtered_parts:
        return None

    return " ".join(filtered_parts)


def classify_scope(filtered_text, company, product_name):
    """Classify whether the filtered context is specific, broad, or mixed."""
    # Use a truncated version for classification to save tokens
    sample = filtered_text[:4000] if len(filtered_text) > 4000 else filtered_text
    msg = (
        f"Company: {company}\n"
        f"Product: {product_name}\n\n"
        f"--- FILTERED TEXT ---\n{sample}\n"
        f"--- END ---\n\n"
        f"Classify the scope of this text."
    )
    raw = _call_llm(SCOPE_SYSTEM_PROMPT, msg, num_ctx=8192)
    if raw is None:
        return "unknown"

    result = _parse_llm_json(raw)
    if result and "scope" in result:
        return result["scope"]
    return "unknown"


# ── Pass 2: Extract structured fields ────────────────────────────────────────

def _extract_chunk(chunk, company, product_name):
    """Extract structured info from a single chunk of filtered text."""
    msg = (
        f"Company: {company}\n"
        f"Product: {product_name}\n\n"
        f"--- PRODUCT-SPECIFIC TEXT ---\n{chunk}\n"
        f"--- END ---\n\n"
        f"Extract the JSON now."
    )
    raw = _call_llm(EXTRACT_SYSTEM_PROMPT, msg, num_ctx=16384)
    if raw is None:
        return None
    return _parse_llm_json(raw)


def _merge_results(results):
    """Merge extraction results from multiple chunks into one."""
    merged = {
        "product_type": "",
        "rates": [],
        "fees": "",
        "bank_partner_specific": "",
        "bank_partner_broad": "",
        "baas_middleware_specific": "",
        "baas_middleware_broad": "",
        "key_features": "",
    }
    all_fees = []
    all_features = []
    bank_specific = set()
    bank_broad = set()
    baas_specific = set()
    baas_broad = set()

    for r in results:
        if not r:
            continue
        if not merged["product_type"] and r.get("product_type"):
            merged["product_type"] = r["product_type"]
        rates = r.get("rates", [])
        if isinstance(rates, list):
            merged["rates"].extend(rates)
        fees = r.get("fees", "")
        if fees:
            all_fees.append(fees)
        if r.get("bank_partner_specific"):
            bank_specific.add(r["bank_partner_specific"])
        if r.get("bank_partner_broad"):
            bank_broad.add(r["bank_partner_broad"])
        if r.get("baas_middleware_specific"):
            baas_specific.add(r["baas_middleware_specific"])
        if r.get("baas_middleware_broad"):
            baas_broad.add(r["baas_middleware_broad"])
        features = r.get("key_features", "")
        if features:
            all_features.append(features)

    merged["fees"] = "; ".join(all_fees) if all_fees else ""
    merged["key_features"] = "; ".join(all_features) if all_features else ""

    merged["bank_partner_specific"] = "; ".join(sorted(bank_specific)) if bank_specific else ""
    merged["bank_partner_broad"] = "; ".join(sorted(bank_broad)) if bank_broad else ""
    merged["baas_middleware_specific"] = "; ".join(sorted(baas_specific)) if baas_specific else ""
    merged["baas_middleware_broad"] = "; ".join(sorted(baas_broad)) if baas_broad else ""

    # Deduplicate rates
    seen = set()
    unique_rates = []
    for rate in merged["rates"]:
        key = (rate.get("interest_rate", rate.get("rate", "")), rate.get("description", ""))
        if key not in seen:
            seen.add(key)
            unique_rates.append(rate)
    merged["rates"] = unique_rates

    return merged


def extract_fields(filtered_text, company, product_name):
    """
    Pass 2: Extract structured product info from filtered context.
    Chunks if needed and merges results.
    """
    if len(filtered_text) < 50:
        print(f"    Too little filtered text ({len(filtered_text)} chars)", flush=True)
        return "insufficient_context"

    chunks = _chunk_text(filtered_text)

    if len(chunks) == 1:
        result = _extract_chunk(chunks[0], company, product_name)
        if result is None:
            print(f"    Failed to parse JSON", flush=True)
        return result

    print(f"    Extracting from {len(chunks)} chunks "
          f"({', '.join(str(len(c)) for c in chunks)} chars)", flush=True)

    results = []
    for i, chunk in enumerate(chunks):
        r = _extract_chunk(chunk, company, product_name)
        if r is None:
            print(f"    Extract chunk {i+1} failed to parse", flush=True)
        results.append(r)

    return _merge_results(results)


# ── Resume support ───────────────────────────────────────────────────────────

def load_done_products(csv_path):
    """Load set of already-processed (company, product_name, period) from output CSV."""
    done = set()
    if not os.path.exists(csv_path):
        return done

    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if len(lines) < 2:
        return done

    header = lines[0]
    expected_cols = header.count(',')

    # Repair truncated last line
    if lines[-1].count(',') < expected_cols:
        print(f"Repairing truncated last line in {csv_path}", flush=True)
        lines = lines[:-1]
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read().replace('\x00', '')
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        key = (row.get("company", ""), row.get("product_name", ""), row.get("period", ""))
        done.add(key)
    return done


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract product info from combined context using LLM")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max products to process (0 = all)")
    parser.add_argument("--company", type=str, default=None,
                        help="Only process this company")
    parser.add_argument("--input", type=str, default=INPUT_CSV,
                        help=f"Input CSV from 3a_extract_context.py (default: {INPUT_CSV})")
    parser.add_argument("--output", type=str, default=OUTPUT_CSV,
                        help=f"Output CSV path (default: {OUTPUT_CSV})")
    parser.add_argument("--latest-only", action="store_true",
                        help="Only process the latest period for each product")
    args = parser.parse_args()

    input_csv = args.input
    output_csv = args.output

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(LOG_FILE, mode='a'),
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger(__name__)

    # Read input CSV
    if not os.path.exists(input_csv):
        log.error(f"Input file not found: {input_csv}")
        log.error("Run 3a_extract_context.py first to generate it.")
        return

    with open(input_csv, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read().replace('\x00', '')
    all_rows = list(csv.DictReader(io.StringIO(text)))
    log.info(f"Loaded {len(all_rows)} products from {input_csv}")

    # Filter to rows with usable text
    all_rows = [r for r in all_rows if r.get("combined_context", "").strip()]

    if args.company:
        all_rows = [r for r in all_rows if r["company"].lower() == args.company.lower()]

    if args.latest_only:
        # Keep only the latest period per (company, product)
        latest = {}
        for r in all_rows:
            key = (r["company"], r["product_name"])
            period = r.get("period", "")
            if key not in latest or period > latest[key].get("period", ""):
                latest[key] = r
        all_rows = list(latest.values())
        log.info(f"--latest-only: filtered to {len(all_rows)} rows (latest period per product)")

    log.info(f"Products with usable context: {len(all_rows)}")

    # Load already-done products
    done = load_done_products(output_csv)
    remaining = [r for r in all_rows
                 if (r["company"], r["product_name"], r.get("period", "")) not in done]
    log.info(f"Already processed: {len(done)}, remaining: {len(remaining)}")

    if args.limit:
        remaining = remaining[:args.limit]
        log.info(f"Limiting to {args.limit} products")

    if not remaining:
        log.info("Nothing to do.")
        return

    # Open output CSV for appending
    write_header = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
    csvfile = open(output_csv, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
    if write_header:
        writer.writeheader()

    processed = 0
    errors = 0

    def _clean(val):
        if not val:
            return val
        return re.sub(r'[\r\n]+', ' ', str(val).replace('\x00', '')).strip()

    for i, row in enumerate(remaining):
        company = row["company"]
        product_name = row["product_name"]
        period = row.get("period", "")
        combined_context = _clean(row["combined_context"]) or ""
        page_types = row.get("page_types", "")
        snapshot_dates = row.get("snapshot_dates", "")
        num_source_files = row.get("num_source_files", "")
        source_files = row.get("source_files", "")

        t0 = time.time()
        period_label = period or "(no date)"
        log.info(f"[{i+1}/{len(remaining)}] {company} / {product_name} "
                 f"[{period_label}] ({num_source_files} files, {len(combined_context)} chars)")

        # ── Pass 1: Filter context to product ────────────────────────────
        log.info(f"    Pass 1: filtering context...")
        filtered = filter_context(combined_context, company, product_name)

        if filtered is None:
            log.info(f"    No relevant context found after filtering")
            writer.writerow({
                "company": company, "product_name": product_name, "period": period,
                "product_type": "", "interest_rate": "",
                "interest_rate_description": "", "fees": "",
                "bank_partner_specific": "", "bank_partner_broad": "",
                "baas_middleware_specific": "", "baas_middleware_broad": "",
                "key_features": "",
                "status": "SKIPPED: no relevant context after filtering",
                "relevant_context": "", "context_scope": "none",
                "combined_context": combined_context,
                "page_types": page_types, "snapshot_dates": snapshot_dates,
                "num_source_files": num_source_files, "source_files": source_files,
            })
            csvfile.flush()
            continue

        log.info(f"    Filtered: {len(combined_context)} -> {len(filtered)} chars")

        # Classify scope
        scope = classify_scope(filtered, company, product_name)
        log.info(f"    Scope: {scope}")

        # ── Pass 2: Extract structured fields ────────────────────────────
        log.info(f"    Pass 2: extracting fields...")
        result = extract_fields(filtered, company, product_name)

        if result == "insufficient_context":
            writer.writerow({
                "company": company, "product_name": product_name, "period": period,
                "product_type": "", "interest_rate": "",
                "interest_rate_description": "", "fees": "",
                "bank_partner_specific": "", "bank_partner_broad": "",
                "baas_middleware_specific": "", "baas_middleware_broad": "",
                "key_features": "",
                "status": "SKIPPED: insufficient context after filtering",
                "relevant_context": _clean(filtered) or "", "context_scope": scope,
                "combined_context": combined_context,
                "page_types": page_types, "snapshot_dates": snapshot_dates,
                "num_source_files": num_source_files, "source_files": source_files,
            })
            csvfile.flush()
            continue

        if result is None:
            errors += 1
            writer.writerow({
                "company": company, "product_name": product_name, "period": period,
                "product_type": "", "interest_rate": "",
                "interest_rate_description": "", "fees": "",
                "bank_partner_specific": "", "bank_partner_broad": "",
                "baas_middleware_specific": "", "baas_middleware_broad": "",
                "key_features": "",
                "status": "ERROR: LLM extraction failed",
                "relevant_context": _clean(filtered) or "", "context_scope": scope,
                "combined_context": combined_context,
                "page_types": page_types, "snapshot_dates": snapshot_dates,
                "num_source_files": num_source_files, "source_files": source_files,
            })
            csvfile.flush()
            continue

        # Extract primary rate
        rates_raw = result.get("rates", [])
        if isinstance(rates_raw, list) and rates_raw:
            first = rates_raw[0]
            r = first.get("interest_rate", first.get("rate", ""))
            interest_rate_str = str(r) if r != "" and r is not None else ""
            rate_descs = [entry.get("description", "") for entry in rates_raw if entry.get("description")]
            interest_rate_desc_str = "; ".join(rate_descs)
        else:
            interest_rate_str = ""
            interest_rate_desc_str = ""

        writer.writerow({
            "company": company,
            "product_name": product_name,
            "period": period,
            "product_type": _clean(result.get("product_type", "")),
            "interest_rate": interest_rate_str,
            "interest_rate_description": _clean(interest_rate_desc_str),
            "fees": _clean(result.get("fees", "")),
            "bank_partner_specific": _clean(result.get("bank_partner_specific", "")),
            "bank_partner_broad": _clean(result.get("bank_partner_broad", "")),
            "baas_middleware_specific": _clean(result.get("baas_middleware_specific", "")),
            "baas_middleware_broad": _clean(result.get("baas_middleware_broad", "")),
            "key_features": _clean(result.get("key_features", "")),
            "status": "OK",
            "relevant_context": _clean(filtered) or "",
            "context_scope": scope,
            "combined_context": combined_context,
            "page_types": page_types,
            "snapshot_dates": snapshot_dates,
            "num_source_files": num_source_files,
            "source_files": source_files,
        })
        csvfile.flush()
        processed += 1

        elapsed = time.time() - t0
        bank_info = result.get('bank_partner_specific', '') or result.get('bank_partner_broad', '') or 'n/a'
        log.info(f"    Done in {elapsed:.1f}s | rate={interest_rate_str or 'n/a'} "
                 f"| bank={bank_info} | scope={scope}")

        if (i + 1) % 25 == 0:
            log.info(f"  ... {i+1}/{len(remaining)} done, {errors} errors so far")

    csvfile.close()
    log.info(f"Done. Processed {processed} products, {errors} errors.")
    log.info(f"Results in: {output_csv}")

    # Post-processing: create clean CSV without context columns
    clean_csv = output_csv.replace('.csv', '_clean.csv')
    try:
        with open(output_csv, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read().replace('\x00', '')
        rows = list(csv.DictReader(io.StringIO(text)))
        drop_cols = {'combined_context', 'relevant_context'}
        cols = [c for c in CSV_COLUMNS if c not in drop_cols]
        with open(clean_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        log.info(f"Clean CSV (no context): {clean_csv}")
    except Exception as e:
        log.info(f"Failed to create clean CSV: {e}")


if __name__ == "__main__":
    main()
