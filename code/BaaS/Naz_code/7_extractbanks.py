import re
from pathlib import Path

import pandas as pd
from rapidfuzz import process, fuzz


# ---------- CONFIG ----------
FINTECH_ROOT = Path("/Users/nazkoont/Naz Dropbox/Naz Koont/Research/BaaS/Data/Fintechs")

# (A) PRIMARY partner list (no years): must contain columns idrssd, partnername
PARTNER_LIST_CSV = Path("/Users/nazkoont/Naz Dropbox/Naz Koont/Research/BaaS/Data/name_matching/matched_names.csv")

# (B) SECONDARY bank panel (with years): must contain name_clean, idrssd, year_start, year_end
BANK_LIST_CSV = Path("/Users/nazkoont/Naz Dropbox/Naz Koont/Research/BaaS/Data/Call Reports/rssdid_name_panel_subset.csv")

OUTPUT_CSV = Path("/Users/nazkoont/Naz Dropbox/Naz Koont/Research/BaaS/Data/working_data/fintech_bank_product_pairs.csv")

INCLUDE_BANK_ONLY_ROWS = True
MAX_FILES = None
SAVE_EVERY_N_FILES = 1000


# ================================================================
# LOAD EXISTING OUTPUT so we can skip already-processed files
# ================================================================
processed_keys = set()
existing_df = None

if OUTPUT_CSV.exists():
    existing_df = pd.read_csv(OUTPUT_CSV, dtype=str)
    for _, row in existing_df.iterrows():
        processed_keys.add((row["fintech"], row["source_type"], row["filename"]))


# ================================================================
# HELPERS
# ================================================================
def normalize_text(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


GENERIC_WORDS = {
    "bank", "the", "trust", "company", "inc", "co", "corp",
    "national", "association",
    "credit", "union", "and", "carrier"
}


def is_generic_name(name_clean: str) -> bool:
    tokens = re.findall(r"[a-zA-Z]+", str(name_clean).lower())
    if not tokens:
        return True
    return all(t in GENERIC_WORDS for t in tokens)


def filename_to_year(filename: str):
    try:
        y = int(str(filename)[:4])
        if 1900 <= y <= 2200:
            return y
    except Exception:
        pass
    return None


def clean_bank_raw(bank_raw: str) -> str:
    bank_clean = re.sub(r"\bMember FDIC\b.*", "", str(bank_raw), flags=re.IGNORECASE)
    bank_clean = bank_clean.strip()
    bank_clean = re.sub(r"^\s*the\s+", "", bank_clean, flags=re.IGNORECASE)
    return bank_clean


# ================================================================
# LOAD PRIMARY PARTNER LIST (no years)
# ================================================================
partner_df = pd.read_csv(PARTNER_LIST_CSV, dtype=str)
if "idrssd" not in partner_df.columns or "partnername" not in partner_df.columns:
    raise ValueError("PARTNER_LIST_CSV must include columns: idrssd, partnername")

partner_df["partnername"] = partner_df["partnername"].astype(str).str.strip()
partner_df["idrssd"] = partner_df["idrssd"].astype(str).str.strip()

partner_names_all = partner_df["partnername"].tolist()
norm_to_partner = {
    normalize_text(row["partnername"]): (row["partnername"], row["idrssd"])
    for _, row in partner_df.iterrows()
}


# ================================================================
# LOAD SECONDARY BANK MASTER LIST (must include year_start, year_end)
# ================================================================
bank_df = pd.read_csv(BANK_LIST_CSV, dtype=str)
bank_df["name_clean"] = bank_df["name_clean"].astype(str).str.strip()
bank_df["idrssd"] = bank_df["idrssd"].astype(str).str.strip()

if "year_start" not in bank_df.columns or "year_end" not in bank_df.columns:
    raise ValueError("BANK_LIST_CSV must include columns: year_start, year_end")

bank_df["year_start"] = pd.to_numeric(bank_df["year_start"], errors="coerce")
bank_df["year_end"] = pd.to_numeric(bank_df["year_end"], errors="coerce")


def get_bank_universe_for_year(year: int):
    if year is None:
        dfy = bank_df.copy()
    else:
        dfy = bank_df[(bank_df["year_start"] < year) & (bank_df["year_end"] > year)].copy()

    norm_to_bank_year = {
        normalize_text(row["name_clean"]): (row["name_clean"], row["idrssd"])
        for _, row in dfy.iterrows()
    }
    bank_names_year = dfy["name_clean"].tolist()
    return bank_names_year, norm_to_bank_year


# ================================================================
# MATCHING AGAINST PRIMARY PARTNER LIST (FIRST PASS)
# ================================================================
def partner_match_bank_name(bank_raw: str, score_cutoff=80):
    cleaned = clean_bank_raw(bank_raw)
    cleaned_norm = normalize_text(cleaned)

    # 1) substring match first
    candidates = []
    for norm_name, (pname, idrssd) in norm_to_partner.items():
        if not norm_name:
            continue
        if is_generic_name(pname):
            continue
        if len(str(pname).replace(" ", "")) <= 4:
            continue
        if norm_name in cleaned_norm or cleaned_norm in norm_name:
            candidates.append((pname, idrssd, len(norm_name)))

    if candidates:
        pname, idrssd_value, _ = max(candidates, key=lambda x: x[2])
        return pname, idrssd_value

    # 2) fuzzy fallback
    if partner_names_all:
        result = process.extractOne(
            cleaned,
            partner_names_all,
            scorer=fuzz.WRatio,
            score_cutoff=score_cutoff
        )
        if result:
            pname = result[0]
            if is_generic_name(pname):
                return None
            if len(str(pname).replace(" ", "")) <= 4:
                return None
            pn_norm = normalize_text(pname)
            if pn_norm in norm_to_partner:
                _, idrssd_value = norm_to_partner[pn_norm]
                return pname, idrssd_value

    return None


# ================================================================
# BANK EXTRACTION (DIRECT MENTION)
# Updated to return (bank_name, idrssd, match_source)
# ================================================================
def extract_banks_from_text(text: str, norm_to_bank_year: dict):
    text_norm = normalize_text(text)
    padded_text = f" {text_norm} "

    found = set()
    partner_norms_found_in_text = set()

    # (1) partner matches
    for norm, (pname, idrssd) in norm_to_partner.items():
        if not norm:
            continue
        if is_generic_name(pname):
            continue
        if len(str(pname).replace(" ", "")) <= 4:
            continue

        if f" {norm} " in padded_text:
            found.add((pname, idrssd, "partner"))
            partner_norms_found_in_text.add(norm)

    # (2) secondary matches for other banks
    for norm, (name_clean, idrssd) in norm_to_bank_year.items():
        if not norm:
            continue
        if norm in partner_norms_found_in_text:
            continue
        if is_generic_name(name_clean):
            continue
        if len(str(name_clean).replace(" ", "")) <= 4:
            continue

        if f" {norm} " in padded_text:
            found.add((name_clean, idrssd, "secondary"))

    return sorted(found)


# ================================================================
# PRODUCT → BANK PAIR EXTRACTION
# ================================================================
PRODUCT_BANK_PATTERN = re.compile(
    r"""
    (?P<product>
        [A-Z][^\.]{0,80}?
    )
    \s+
    (?:is|are)\s+
    (?:held\s+with|issued\s+by|backed\s+by|provided\s+by)\s+
    (?P<bank>[^,\.]+?)
    (?=\.|,|\)|$)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def fuzzy_match_bank_name(bank_raw: str, bank_names_year: list, norm_to_bank_year: dict, score_cutoff=80):
    # FIRST PASS: partner list
    pmatch = partner_match_bank_name(bank_raw, score_cutoff=score_cutoff)
    if pmatch is not None:
        bank_name, idrssd_value = pmatch
        return bank_name, idrssd_value, "partner"

    # FALLBACK: year-filtered panel list
    cleaned = clean_bank_raw(bank_raw)
    cleaned_norm = normalize_text(cleaned)

    # 1) substring match first (year-filtered)
    candidates = []
    for norm_name, (name_clean, idrssd) in norm_to_bank_year.items():
        if not norm_name:
            continue
        if is_generic_name(name_clean):
            continue
        if len(str(name_clean).replace(" ", "")) <= 4:
            continue

        if norm_name in cleaned_norm or cleaned_norm in norm_name:
            candidates.append((name_clean, idrssd, len(norm_name)))

    if candidates:
        name_clean, idrssd_value, _ = max(candidates, key=lambda x: x[2])
        return name_clean, idrssd_value, "secondary"

    # 2) fuzzy match (year-filtered)
    if bank_names_year:
        result = process.extractOne(
            cleaned,
            bank_names_year,
            scorer=fuzz.WRatio,
            score_cutoff=score_cutoff
        )
        if result:
            canonical = result[0]
            if is_generic_name(canonical):
                return None
            if len(str(canonical).replace(" ", "")) <= 4:
                return None
            canon_norm = normalize_text(canonical)
            if canon_norm in norm_to_bank_year:
                _, idrssd_value = norm_to_bank_year[canon_norm]
                return canonical, idrssd_value, "secondary"

    return None


def extract_product_bank_pairs(text: str, bank_names_year: list, norm_to_bank_year: dict):
    pairs = []
    for m in PRODUCT_BANK_PATTERN.finditer(text):
        product = m.group("product").strip()
        bank_raw = m.group("bank").strip()

        matched = fuzzy_match_bank_name(bank_raw, bank_names_year, norm_to_bank_year)
        if matched:
            bank_name, idrssd_value, match_source = matched
            pairs.append((product, bank_name, idrssd_value, match_source))

    return sorted(set(pairs))


# ================================================================
# SAVE HELPERS
# ================================================================
def append_and_save(new_rows: list):
    if not new_rows:
        return

    new_df = pd.DataFrame(new_rows)

    if OUTPUT_CSV.exists():
        existing = pd.read_csv(OUTPUT_CSV, dtype=str)
        out = pd.concat([existing, new_df], ignore_index=True)
    else:
        out = new_df

    out.to_csv(OUTPUT_CSV, index=False)


# ================================================================
# DIRECTORY WALK + SKIP ALREADY PROCESSED FILES + SAVE EVERY 1000
# ================================================================
def process_fintechs(fintech_root: Path):
    rows_buffer = []
    processed_files = 0

    for fintech_dir in fintech_root.iterdir():
        if not fintech_dir.is_dir():
            continue

        fintech = fintech_dir.name

        for source_type in ["terms", "website"]:
            subdir = fintech_dir / source_type
            if not subdir.exists():
                continue

            for txt_file in sorted(subdir.glob("*.txt")):
                key = (fintech, source_type, txt_file.name)
                if key in processed_keys:
                    continue

                if MAX_FILES is not None and processed_files >= MAX_FILES:
                    append_and_save(rows_buffer)
                    return processed_files

                year = filename_to_year(txt_file.name)
                bank_names_year, norm_to_bank_year = get_bank_universe_for_year(year)

                try:
                    text = txt_file.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue

                processed_files += 1

                # product-bank rows (now include match_source)
                product_pairs = extract_product_bank_pairs(text, bank_names_year, norm_to_bank_year)
                for product, bank_name, idrssd_value, match_source in product_pairs:
                    rows_buffer.append({
                        "fintech": fintech,
                        "source_type": source_type,
                        "filename": txt_file.name,
                        "product": product,
                        "bank_name": bank_name,
                        "idrssd": idrssd_value,
                        "match_source": match_source,
                    })

                # bank-only rows (now include match_source)
                if INCLUDE_BANK_ONLY_ROWS:
                    banks_only = extract_banks_from_text(text, norm_to_bank_year)
                    for bank_name, idrssd_value, match_source in banks_only:
                        if not any(bn == bank_name for _, bn, _, _ in product_pairs):
                            rows_buffer.append({
                                "fintech": fintech,
                                "source_type": source_type,
                                "filename": txt_file.name,
                                "product": "",
                                "bank_name": bank_name,
                                "idrssd": idrssd_value,
                                "match_source": match_source,
                            })

                processed_keys.add(key)

                if processed_files % SAVE_EVERY_N_FILES == 0:
                    append_and_save(rows_buffer)
                    rows_buffer = []

    append_and_save(rows_buffer)
    return processed_files


# ================================================================
# MAIN
# ================================================================
def main():
    n_files = process_fintechs(FINTECH_ROOT)
    print(f"Done. Processed {n_files} new files. Output at {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
