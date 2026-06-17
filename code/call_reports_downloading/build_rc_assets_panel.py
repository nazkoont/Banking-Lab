#!/usr/bin/env python3
"""
build_rc_assets_panel.py

Build a bank-quarter panel of **Schedule RC (Balance Sheet)** asset items from FFIEC Call Report
"Schedule RC" raw text files stored under:

  /zfs/data/bankcallreports/raw/current/data/<QDIR>/

Where <QDIR> is the 6-digit quarter code like:
  033101  -> 2001-03-31
  063001  -> 2001-06-30
  093001  -> 2001-09-30
  123101  -> 2001-12-31

This script:
  1) Iterates through quarter-end dates in a user-specified range.
  2) For each quarter, finds the "Schedule RC" file (NOT RCA/RCB/...) in that quarter directory.
  3) Reads the file robustly (handles occasional malformed quoting).
  4) Constructs 12 asset-item columns (and a consistency check vs total assets).
  5) Writes:
       - panel CSV (bank x quarter)
       - summary CSV (quarter-level run stats)
       - missing columns log CSV (what variables were missing by quarter)

IMPORTANT CONVENTIONS
---------------------
- We treat FFIEC mnemonics case-insensitively. We normalize all column names to lowercase.
- For any item defined as max(rcfdXXXX, rconXXXX), we compute:
      max_of_pair = max(df["rcfdxxxx"], df["rconxxxx"]) row-wise (skip NaN if one is present).
- "Missing column" means: the raw Schedule RC file for that quarter does not contain that mnemonic,
  so we must fill it with NaN for that quarter. This is normal across time because mnemonics change.

Your requested 12 items + definitions are implemented exactly as described, with date-based switches.

Usage:
  python3 build_rc_assets_panel.py --start 1990-03-31 --end 2024-12-31

Outputs:
  /zfs/data/bankcallreports/derived/rc_assets_panel.csv
  /zfs/data/bankcallreports/derived/rc_assets_recon_summary.csv
  /zfs/data/bankcallreports/derived/rc_assets_missing_cols.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Paths (edit here if needed)
# -----------------------------
RAW_BASE = "/zfs/data/bankcallreports/raw/current/data"
DERIVED_BASE = "/zfs/data/bankcallreports/derived"

OUT_PANEL = os.path.join(DERIVED_BASE, "rc_assets_panel.csv")
OUT_SUMMARY = os.path.join(DERIVED_BASE, "rc_assets_recon_summary.csv")
OUT_MISSING = os.path.join(DERIVED_BASE, "rc_assets_missing_cols.csv")


# -----------------------------
# Helpers: date / quarter code
# -----------------------------
def parse_date(s: str) -> pd.Timestamp:
    """Parse YYYY-MM-DD into a pandas Timestamp (naive)."""
    return pd.to_datetime(s, format="%Y-%m-%d")


def qdir_from_date(d: pd.Timestamp) -> str:
    """
    Convert quarter-end date to 6-digit directory code:
      2001-03-31 -> "033101"  (MMDDYY)
    """
    return d.strftime("%m%d%y")


def quarter_ends(start: pd.Timestamp, end: pd.Timestamp) -> List[pd.Timestamp]:
    """
    Generate quarter-end dates between start and end inclusive.
    Assumes user passes quarter-end dates (3/31, 6/30, 9/30, 12/31).
    """
    # Q-DEC gives quarter ends at: Mar 31, Jun 30, Sep 30, Dec 31
    # It returns Timestamp at 23:59... sometimes; normalize to midnight
    qs = pd.date_range(start=start, end=end, freq="Q-DEC")
    return [pd.Timestamp(x.date()) for x in qs]


# -----------------------------
# Robust reader for Schedule RC
# -----------------------------
def find_schedule_rc_file(qpath: str) -> Optional[str]:
    """
    Find the main 'Schedule RC' file in a quarter folder.

    In quarter folder you may see:
      - "... Schedule RC 03312002.txt"   <-- we want this one
      - "... Schedule RCA ...", "RCB ...", etc  <-- not these

    We match: "*Schedule RC *.txt" but avoid RCA/RCB by requiring 'Schedule RC ' (RC followed by space).
    """
    # glob with literal space after "RC"
    pattern = os.path.join(qpath, "*Schedule RC *.txt")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        return None
    # If multiple exist, choose the first sorted (usually only 1).
    return candidates[0]


def read_schedule_rc(filepath: str) -> pd.DataFrame:
    """
    Read a tab-delimited Schedule RC file into a DataFrame.

    These FFIEC files are usually clean tab-delimited text with a quoted header like:
      "IDRSSD"\tRCFD0071\t...

    Occasionally, some raw files contain malformed quoting in data -> pandas ParserError.
    Strategy:
      1) strict-ish read
      2) fallback: disable quote handling and skip bad lines
    """
    # Attempt 1: normal read (python engine handles oddities better than C for these)
    try:
        df = pd.read_csv(
            filepath,
            sep="\t",
            dtype=str,
            engine="python",
        )
        return df
    except Exception as e1:
        # Attempt 2: forgiving mode
        try:
            df = pd.read_csv(
                filepath,
                sep="\t",
                dtype=str,
                engine="python",
                quoting=csv.QUOTE_NONE,   # treat " as a normal character
                on_bad_lines="skip",      # skip broken lines
            )
            return df
        except Exception as e2:
            raise RuntimeError(
                f"Could not parse {filepath}. strict_error={type(e1).__name__}: {e1} ; "
                f"forgiving_error={type(e2).__name__}: {e2}"
            ) from e2


# -----------------------------
# Numeric conversion utilities
# -----------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase and strip quotes/spaces from column names for consistent lookup."""
    cols = []
    for c in df.columns:
        c2 = str(c).strip()
        # remove wrapping quotes if present:  "IDRSSD" -> IDRSSD
        if len(c2) >= 2 and c2[0] == '"' and c2[-1] == '"':
            c2 = c2[1:-1]
        cols.append(c2.lower())
    df = df.copy()
    df.columns = cols
    return df


def ensure_numeric_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Ensure specified columns are numeric (float).
    If a column is missing, we create it filled with NaN.
    """
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def rowwise_max(df: pd.DataFrame, a: str, b: str) -> pd.Series:
    """
    Compute max(a,b) row-wise, treating missing as:
      max(x, NaN) = x
      max(NaN, NaN) = NaN
    """
    return pd.concat([df[a], df[b]], axis=1).max(axis=1, skipna=True)


# -----------------------------
# Date-regime logic for mnemonics
# -----------------------------
def is_leq(d: pd.Timestamp, y: int, m: int, day: int) -> bool:
    return d <= pd.Timestamp(year=y, month=m, day=day)


def is_geq(d: pd.Timestamp, y: int, m: int, day: int) -> bool:
    return d >= pd.Timestamp(year=y, month=m, day=day)


def between(d: pd.Timestamp, y1: int, m1: int, day1: int, y2: int, m2: int, day2: int) -> bool:
    return pd.Timestamp(year=y1, month=m1, day=day1) <= d <= pd.Timestamp(year=y2, month=m2, day=day2)


# -----------------------------
# Main computation: 12 asset items
# -----------------------------
@dataclass
class QuarterResult:
    qdir: str
    date: pd.Timestamp
    banks: int
    missing_cols: List[str]
    pass_rate: float
    status: str


def compute_items_for_quarter(df_raw: pd.DataFrame, qdate: pd.Timestamp) -> Tuple[pd.DataFrame, List[str]]:
    """
    Given raw Schedule RC DataFrame for a quarter, compute the 12 asset item columns.

    Returns:
      (df_out, missing_cols)

    df_out includes:
      idrssd, qdate, qdir, plus:
      cash_nonint, cash_int, cash_tot,
      securities_htm_ac, securities_afs_fv, securities_equity_fv, sec_tot,
      ffs, repo, ffsrepo, ffsrepo_tot,
      loan_netunearnedinc, loan_transriskreserve, loan_alll, loan_net, loan_afs, loan_tot,
      tradingassets, fixedassets, otherrealestate, equitystakes, item9, intangibles, otherassets,
      assets, assetcheck, assetcheck_ok
    """
    df = normalize_columns(df_raw)

    # IDRSSD is our bank identifier
    if "idrssd" not in df.columns:
        raise RuntimeError("Schedule RC file does not contain IDRSSD column (cannot identify banks).")

    # List every mnemonic we might touch across all regimes.
    # If missing in a quarter, we create it as NaN and log it.
    needed = [
        # Cash
        "rcfd0081", "rcon0081", "rcfd0071", "rcon0071",

        # Securities
        "rcfd1754", "rcon1754",     # HTM amortized cost (1996Q4-2018Q4)
        "rcfdjj34", "rconjj34",     # HTM amortized cost (2019Q1-present)
        "rcfd1773", "rcon1773",     # AFS fair value
        "rcfdja22", "rconja22",     # equity FV (2018Q1+)

        # Fed funds / repos / combined
        "rcfd0276", "rcon0276",     # ffs (1996Q4 only per your notes)
        "rconb987",                 # ffs domestic office (2002Q1+)
        "rcfd0277", "rcon0277",     # repo (1996Q4 only per your notes)
        "rcfdb989", "rconb989",     # repo (2002Q1+)
        "rcfd1350", "rcon1350",     # combined ffs+repo (1997Q1-2001Q4)

        # Loans
        "rcfd2122", "rcon2122",     # loans net of unearned income (<=2000Q4)
        "rcfdb528", "rconb528",     # loans net of unearned income (2001Q1+)
        "rcfd3128", "rcon3128",     # transfer risk reserve (<=2000Q4)
        "rcfd3123", "rcon3123",     # ALLL / allowances for credit losses
        "rcfd2125", "rcon2125",     # loans held for investment net (<=2000Q4)
        "rcfdb529", "rconb529",     # loans held for investment net (2001Q1+)
        "rcfd5369", "rcon5369",     # loans held for sale (2001Q1+)

        # Trading assets, fixed assets, OREO, equity stakes
        "rcfd3545", "rcon3545",
        "rcfd2145", "rcon2145",
        "rcfd2150", "rcon2150",
        "rcfd2130", "rcon2130",

        # Item 9 regimes
        "rcfd2155", "rcon2155",     # acceptances outstanding (1996Q1-2005Q4)
        "rcfd3656", "rcon3656",     # real estate ventures (2009Q2+)

        # Intangibles regimes
        "rcfd2143", "rcon2143",     # "intangibles" (<=2000Q4, and again 2018Q2+ per your notes)
        "rcfd3163", "rcon3163",     # goodwill (2001Q1-2018Q1)
        "rcfd0426", "rcon0426",     # other intangibles (2001Q1-2018Q1)

        # Other assets + total assets
        "rcfd2160", "rcon2160",
        "rcfd2170", "rcon2170",
    ]

    # Ensure all needed columns exist and are numeric (float); missing -> NaN
    df2 = ensure_numeric_cols(df, needed)

    # Track missing columns for logging (only those absent in raw file, not those present but all NaN)
    missing_cols = [c for c in needed if c not in df.columns]

    # Start output with identifiers
    out = pd.DataFrame({
        "idrssd": pd.to_numeric(df2["idrssd"], errors="coerce").astype("Int64"),
    })
    out["qdate"] = qdate.date().isoformat()
    out["qdir"] = qdir_from_date(qdate)

    # -------------------------
    # (1) CASH
    # -------------------------
    out["cash_nonint"] = rowwise_max(df2, "rcfd0081", "rcon0081")  # noninterest-bearing balances + coin/currency
    out["cash_int"] = rowwise_max(df2, "rcfd0071", "rcon0071")     # interest-bearing balances + coin/currency
    out["cash_tot"] = out["cash_nonint"] + out["cash_int"]

    # -------------------------
    # (2) SECURITIES
    # -------------------------
    # HTM amortized cost:
    # - 1996Q4 to 2018Q4: (RCFD1754/RCON1754)
    # - 2019Q1+: (RCFDJJ34/RCONJJ34)
    if between(qdate, 1996, 12, 31, 2018, 12, 31):
        out["securities_htm_ac"] = rowwise_max(df2, "rcfd1754", "rcon1754")
    elif is_geq(qdate, 2019, 3, 31):
        out["securities_htm_ac"] = rowwise_max(df2, "rcfdjj34", "rconjj34")
    else:
        # outside your stated validity; keep NaN
        out["securities_htm_ac"] = np.nan

    out["securities_afs_fv"] = rowwise_max(df2, "rcfd1773", "rcon1773")

    # Equity securities w/ readily determinable FV not held for trading (2018Q1+)
    if is_geq(qdate, 2018, 3, 31):
        out["securities_equity_fv"] = rowwise_max(df2, "rcfdja22", "rconja22")
    else:
        out["securities_equity_fv"] = np.nan

    # Total securities:
    # - up to 2017Q4: HTM + AFS
    # - 2018Q1+: HTM + AFS + Equity FV
    if is_leq(qdate, 2017, 12, 31):
        out["sec_tot"] = out["securities_htm_ac"] + out["securities_afs_fv"]
    else:
        out["sec_tot"] = out["securities_htm_ac"] + out["securities_afs_fv"] + out["securities_equity_fv"]

    # -------------------------
    # (3) FED FUNDS SOLD + REPOS
    # -------------------------
    # federal funds sold:
    # - 1996Q4 only: max(rcfd0276, rcon0276)
    # - 2002Q1+: rconb987 (ffs in domestic office)
    if qdate == pd.Timestamp("1996-12-31"):
        out["ffs"] = rowwise_max(df2, "rcfd0276", "rcon0276")
    elif is_geq(qdate, 2002, 3, 31):
        out["ffs"] = df2["rconb987"]
    else:
        out["ffs"] = np.nan

    # repos:
    # - 1996Q4 only: max(rcfd0277, rcon0277)
    # - 2002Q1+: max(rcfdb989, rconb989)
    if qdate == pd.Timestamp("1996-12-31"):
        out["repo"] = rowwise_max(df2, "rcfd0277", "rcon0277")
    elif is_geq(qdate, 2002, 3, 31):
        out["repo"] = rowwise_max(df2, "rcfdb989", "rconb989")
    else:
        out["repo"] = np.nan

    # combined ffs + repo item:
    # - 1997Q1 - 2001Q4: max(rcfd1350, rcon1350)
    if between(qdate, 1997, 3, 31, 2001, 12, 31):
        out["ffsrepo"] = rowwise_max(df2, "rcfd1350", "rcon1350")
    else:
        out["ffsrepo"] = np.nan

    # Total on balance sheet:
    # - 1996Q4: ffs + repo
    # - 1997Q1 - 2001Q4: ffsrepo
    # - 2002Q1+: ffs + repo
    if qdate == pd.Timestamp("1996-12-31"):
        out["ffsrepo_tot"] = out["ffs"] + out["repo"]
    elif between(qdate, 1997, 3, 31, 2001, 12, 31):
        out["ffsrepo_tot"] = out["ffsrepo"]
    elif is_geq(qdate, 2002, 3, 31):
        out["ffsrepo_tot"] = out["ffs"] + out["repo"]
    else:
        out["ffsrepo_tot"] = np.nan

    # -------------------------
    # (4) LOANS + LEASE FINANCING RECEIVABLES
    # -------------------------
    # Loans net of unearned income:
    if is_leq(qdate, 2000, 12, 31):
        out["loan_netunearnedinc"] = rowwise_max(df2, "rcfd2122", "rcon2122")
    else:
        out["loan_netunearnedinc"] = rowwise_max(df2, "rcfdb528", "rconb528")

    # transfer risk reserve (<=2000Q4 only)
    if is_leq(qdate, 2000, 12, 31):
        out["loan_transriskreserve"] = rowwise_max(df2, "rcfd3128", "rcon3128")
    else:
        out["loan_transriskreserve"] = np.nan

    # ALLL / allowances
    out["loan_alll"] = rowwise_max(df2, "rcfd3123", "rcon3123")

    # Loans held for investment net:
    if is_leq(qdate, 2000, 12, 31):
        out["loan_net"] = rowwise_max(df2, "rcfd2125", "rcon2125")
    else:
        out["loan_net"] = rowwise_max(df2, "rcfdb529", "rconb529")

    # Loans held for sale (2001Q1+)
    if is_geq(qdate, 2001, 3, 31):
        out["loan_afs"] = rowwise_max(df2, "rcfd5369", "rcon5369")
    else:
        out["loan_afs"] = np.nan

    # Total loans on balance sheet:
    if is_leq(qdate, 2000, 12, 31):
        out["loan_tot"] = out["loan_net"]
    else:
        out["loan_tot"] = out["loan_net"] + out["loan_afs"]

    # -------------------------
    # (5) TRADING ASSETS
    # -------------------------
    out["tradingassets"] = rowwise_max(df2, "rcfd3545", "rcon3545")

    # -------------------------
    # (6) FIXED ASSETS
    # -------------------------
    out["fixedassets"] = rowwise_max(df2, "rcfd2145", "rcon2145")

    # -------------------------
    # (7) OTHER REAL ESTATE OWNED
    # -------------------------
    out["otherrealestate"] = rowwise_max(df2, "rcfd2150", "rcon2150")

    # -------------------------
    # (8) STAKES IN SUBSIDIARIES
    # -------------------------
    out["equitystakes"] = rowwise_max(df2, "rcfd2130", "rcon2130")

    # -------------------------
    # (9) ITEM 9 (two regimes + missing gap)
    # -------------------------
    # 1996Q1 - 2005Q4: customers' liability on acceptances outstanding
    if between(qdate, 1996, 3, 31, 2005, 12, 31):
        out["receivables"] = rowwise_max(df2, "rcfd2155", "rcon2155")
        out["invrealestatefunds"] = np.nan
        out["item9"] = out["receivables"]
    # 2006Q1 - 2009Q1: not defined
    elif between(qdate, 2006, 3, 31, 2009, 3, 31):
        out["receivables"] = np.nan
        out["invrealestatefunds"] = np.nan
        out["item9"] = np.nan
    # 2009Q2+ (i.e., 2009-06-30 onward): direct/indirect investments in real estate ventures
    elif is_geq(qdate, 2009, 6, 30):
        out["receivables"] = np.nan
        out["invrealestatefunds"] = rowwise_max(df2, "rcfd3656", "rcon3656")
        out["item9"] = out["invrealestatefunds"]
    else:
        out["receivables"] = np.nan
        out["invrealestatefunds"] = np.nan
        out["item9"] = np.nan

    # -------------------------
    # (10) INTANGIBLES
    # -------------------------
    # Your notes say:
    #   - through 2000Q4: max(rcfd2143,rcon2143)
    #   - 2001Q1-2018Q1: goodwill + other intangible assets = max(3163)+max(0426)
    #   - 2018Q2-present: max(rcfd2143,rcon2143)
    if is_leq(qdate, 2000, 12, 31):
        out["intangibles"] = rowwise_max(df2, "rcfd2143", "rcon2143")
    elif between(qdate, 2001, 3, 31, 2018, 3, 31):
        out["intangibles"] = rowwise_max(df2, "rcfd3163", "rcon3163") + rowwise_max(df2, "rcfd0426", "rcon0426")
    elif is_geq(qdate, 2018, 6, 30):
        out["intangibles"] = rowwise_max(df2, "rcfd2143", "rcon2143")
    else:
        out["intangibles"] = np.nan

    # -------------------------
    # (11) OTHER ASSETS
    # -------------------------
    out["otherassets"] = rowwise_max(df2, "rcfd2160", "rcon2160")

    # -------------------------
    # (12) TOTAL ASSETS
    # -------------------------
    out["assets"] = rowwise_max(df2, "rcfd2170", "rcon2170")

    # -------------------------
    # Asset check: do items sum to total assets?
    # -------------------------
    # Important: sum(skipna=True) so NaN items (like item9 in the undefined window) don’t poison the whole sum.
    components = [
        "cash_tot", "sec_tot", "ffsrepo_tot", "loan_tot", "tradingassets",
        "fixedassets", "otherrealestate", "equitystakes", "item9", "intangibles", "otherassets"
    ]
    out["assetcheck"] = out[components].sum(axis=1, skipna=True)

    # "success is if assetcheck and assets are identical"
    # We'll compute a strict equality check *only where both are non-missing*.
    both = out["assetcheck"].notna() & out["assets"].notna()
    out["assetcheck_ok"] = False
    out.loc[both, "assetcheck_ok"] = (out.loc[both, "assetcheck"] == out.loc[both, "assets"])

    return out, missing_cols


# -----------------------------
# Main driver
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="Start quarter-end date (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="End quarter-end date (YYYY-MM-DD)")
    ap.add_argument("--raw-base", default=RAW_BASE, help="Base directory for raw quarter folders")
    ap.add_argument("--out-panel", default=OUT_PANEL, help="Output panel CSV")
    ap.add_argument("--out-summary", default=OUT_SUMMARY, help="Output summary CSV")
    ap.add_argument("--out-missing", default=OUT_MISSING, help="Output missing columns CSV")
    args = ap.parse_args()

    start = parse_date(args.start)
    end = parse_date(args.end)

    qs = quarter_ends(start, end)

    panel_parts: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    missing_rows: List[Dict[str, object]] = []

    for qd in qs:
        qdir = qdir_from_date(qd)
        qpath = os.path.join(args.raw_base, qdir)

        if not os.path.isdir(qpath):
            print(f"SKIP {qdir}: quarter directory not found: {qpath}")
            summary_rows.append({
                "qdir": qdir, "date": qd.date().isoformat(), "banks": 0,
                "missing_cols": "", "pass_rate": np.nan, "status": "no_quarter_dir",
            })
            continue

        rcfile = find_schedule_rc_file(qpath)
        if rcfile is None:
            print(f"SKIP {qdir}: no Schedule RC file found under {qpath}")
            summary_rows.append({
                "qdir": qdir, "date": qd.date().isoformat(), "banks": 0,
                "missing_cols": "", "pass_rate": np.nan, "status": "no_schedule_rc",
            })
            continue

        # Read file robustly; if it is malformed, skip quarter but keep going
        try:
            df_raw = read_schedule_rc(rcfile)
        except Exception as e:
            print(f"SKIP {qdir}: failed to read Schedule RC due to parse error: {e}")
            summary_rows.append({
                "qdir": qdir, "date": qd.date().isoformat(), "banks": 0,
                "missing_cols": "", "pass_rate": np.nan, "status": "parse_error",
            })
            continue

        # Compute the requested 12 items + check
        try:
            df_out, missing_cols = compute_items_for_quarter(df_raw, qd)
        except Exception as e:
            print(f"SKIP {qdir}: failed to compute items: {e}")
            summary_rows.append({
                "qdir": qdir, "date": qd.date().isoformat(), "banks": 0,
                "missing_cols": "", "pass_rate": np.nan, "status": "compute_error",
            })
            continue

        # Quarter stats
        banks = int(df_out["idrssd"].notna().sum())
        # pass_rate: fraction of banks where assetcheck_ok == True among banks with nonmissing assets+assetcheck
        denom = int((df_out["assets"].notna() & df_out["assetcheck"].notna()).sum())
        num = int(df_out["assetcheck_ok"].sum())
        pass_rate = (num / denom) if denom > 0 else np.nan

        # Logging missing cols (as one comma-separated string in summary)
        missing_str = ",".join(sorted(set(missing_cols)))

        print(f"{qdir} ({qd.date().isoformat()}): banks={banks}, pass_rate={pass_rate:0.4f}")

        # For your earlier behavior: print a warning when missing columns exist (but do not fail)
        if missing_cols:
            print(f"WARN {qdir}: Missing columns (filled NaN): {sorted(set(missing_cols))}")

        summary_rows.append({
            "qdir": qdir,
            "date": qd.date().isoformat(),
            "banks": banks,
            "missing_cols": missing_str,
            "pass_rate": pass_rate,
            "status": "ok",
        })

        # Write missing column log at column-level granularity
        for c in sorted(set(missing_cols)):
            missing_rows.append({
                "qdir": qdir,
                "date": qd.date().isoformat(),
                "missing_col": c,
                "source_file": os.path.basename(rcfile),
            })

        panel_parts.append(df_out)

    # Concatenate all quarters into one long panel
    if panel_parts:
        panel = pd.concat(panel_parts, axis=0, ignore_index=True)
    else:
        panel = pd.DataFrame()

    summary = pd.DataFrame(summary_rows)
    missing_df = pd.DataFrame(missing_rows)

    # Ensure derived directory exists
    os.makedirs(os.path.dirname(args.out_panel), exist_ok=True)

    # Write outputs
    panel.to_csv(args.out_panel, index=False)
    summary.to_csv(args.out_summary, index=False)
    missing_df.to_csv(args.out_missing, index=False)

    print(f"\nWrote panel: {args.out_panel}")
    print(f"Wrote summary: {args.out_summary}")
    print(f"Wrote missing-cols log: {args.out_missing}")


if __name__ == "__main__":
    main()
