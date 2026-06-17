import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("/zfs/data/bankcallreports/raw/current/data")


# ----------------------------
# Helpers (robust to missing columns)
# ----------------------------
def to_num_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return numeric Series for df[col]; if missing, return NaNs aligned to df.index."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def prefer_rcfd(df: pd.DataFrame, rcfd_col: str, rcon_col: str) -> pd.Series:
    """
    Prefer consolidated (RCFD). Fall back to domestic (RCON) if RCFD missing or 0.
    Always returns numeric Series aligned to df.index.
    """
    a = to_num_series(df, rcfd_col)
    b = to_num_series(df, rcon_col)
    out = a.copy()
    use_b = (out.isna() | (out == 0)) & b.notna() & (b != 0)
    out[use_b] = b[use_b]
    return out


def find_rc_file(qdir: Path) -> Path | None:
    """Find the Schedule RC file in a quarter directory."""
    rc_files = sorted(qdir.glob("FFIEC CDR Call Schedule RC *.txt"))
    return rc_files[0] if rc_files else None


def parse_date_from_filename(fname: str) -> pd.Timestamp | None:
    """
    Extract MMDDYYYY from filename and convert to Timestamp.
    Example: '... RC 03312001.txt' -> 2001-03-31
    """
    m = re.search(r"(\d{8})", fname)
    if not m:
        return None
    ts = pd.to_datetime(m.group(1), format="%m%d%Y", errors="coerce")
    if pd.isna(ts):
        return None
    return ts


# ----------------------------
# Time-switching definitions
# ----------------------------
def compute_cash(df: pd.DataFrame) -> pd.Series:
    cash_nonint = prefer_rcfd(df, "RCFD0081", "RCON0081").fillna(0)
    cash_int = prefer_rcfd(df, "RCFD0071", "RCON0071").fillna(0)
    return cash_nonint + cash_int


def compute_securities(df: pd.DataFrame, report_date: pd.Timestamp) -> pd.Series:
    """
    Held-to-maturity amortized cost:
      - 1996Q4–2018Q4: RCFD1754/RCON1754
      - 2019Q1–present: RCFDJJ34/RCONJJ34
    AFS fair value: RCFD1773/RCON1773
    Equity securities (readily determinable FV not held for trading): RCFDJA22/RCONJA22 (since 2018Q1)
    Total:
      - through 2017Q4: HTM + AFS
      - 2018Q1+: HTM + AFS + EQUITY
    """
    # HTM column switch at 2019Q1
    if report_date >= pd.Timestamp("2019-03-31"):
        htm = prefer_rcfd(df, "RCFDJJ34", "RCONJJ34").fillna(0)
    else:
        htm = prefer_rcfd(df, "RCFD1754", "RCON1754").fillna(0)

    afs = prefer_rcfd(df, "RCFD1773", "RCON1773").fillna(0)

    if report_date >= pd.Timestamp("2018-03-31"):
        equity = prefer_rcfd(df, "RCFDJA22", "RCONJA22").fillna(0)
        return htm + afs + equity
    else:
        return htm + afs


def compute_ffsrepo(df: pd.DataFrame, report_date: pd.Timestamp) -> pd.Series:
    """
    Federal funds sold + repos definition switches:
      - 1996Q4 only: use 0276 and 0277 separately (ffs + repo)
      - 1997Q1–2001Q4: use combined 1350
      - 2002Q1–present: use domestic office components B987 (ffs) + B989 (repo)
    """
    if report_date == pd.Timestamp("1996-12-31"):
        ffs = prefer_rcfd(df, "RCFD0276", "RCON0276").fillna(0)
        repo = prefer_rcfd(df, "RCFD0277", "RCON0277").fillna(0)
        return ffs + repo

    if pd.Timestamp("1997-03-31") <= report_date <= pd.Timestamp("2001-12-31"):
        return prefer_rcfd(df, "RCFD1350", "RCON1350").fillna(0)

    # 2002Q1+
    if report_date >= pd.Timestamp("2002-03-31"):
        ffs = prefer_rcfd(df, "RCFDB987", "RCONB987").fillna(0)
        repo = prefer_rcfd(df, "RCFDB989", "RCONB989").fillna(0)
        return ffs + repo

    # earlier than 1996Q4: best-effort fallback
    return prefer_rcfd(df, "RCFD1350", "RCON1350").fillna(0)


def compute_loans(df: pd.DataFrame, report_date: pd.Timestamp) -> pd.Series:
    """
    Loans:
      - Up to 2000Q4: loan_net = 2125 (net of ALLL and transfer risk reserve)
      - 2001Q1+:      loan_net = B529; plus loan_afs = 5369
    Total loans:
      - up to 2000Q4: loan_tot = loan_net
      - 2001Q1+:      loan_tot = loan_net + loan_afs
    """
    if report_date <= pd.Timestamp("2000-12-31"):
        loan_net = prefer_rcfd(df, "RCFD2125", "RCON2125").fillna(0)
        return loan_net
    else:
        loan_net = prefer_rcfd(df, "RCFDB529", "RCONB529").fillna(0)
        loan_afs = prefer_rcfd(df, "RCFD5369", "RCON5369").fillna(0)
        return loan_net + loan_afs


def compute_item9(df: pd.DataFrame, report_date: pd.Timestamp) -> pd.Series:
    """
    Item 9 (balance sheet slot):
      - 1996Q1–2005Q4: acceptances (2155)
      - 2006Q1–2009Q1: not defined (treat as 0 for reconciliation; track separately if needed)
      - 2009Q2–present: real estate venture investments (3656)
    """
    if report_date <= pd.Timestamp("2005-12-31"):
        return prefer_rcfd(df, "RCFD2155", "RCON2155").fillna(0)

    if pd.Timestamp("2006-03-31") <= report_date <= pd.Timestamp("2009-03-31"):
        # Not defined; set to 0 so the identity check isn't poisoned by NaN math
        return pd.Series(0.0, index=df.index)

    if report_date >= pd.Timestamp("2009-06-30"):
        return prefer_rcfd(df, "RCFD3656", "RCON3656").fillna(0)

    # fallback
    return pd.Series(0.0, index=df.index)


def compute_intangibles(df: pd.DataFrame, report_date: pd.Timestamp) -> pd.Series:
    """
    Intangibles switches (based on your notes):
      - through 2000Q4: 2143 (total intangibles)
      - 2001Q1–2018Q1: goodwill (3163) + other intangibles (0426)
      - 2018Q2–present: 2143 again
    """
    if report_date <= pd.Timestamp("2000-12-31"):
        return prefer_rcfd(df, "RCFD2143", "RCON2143").fillna(0)

    if pd.Timestamp("2001-03-31") <= report_date <= pd.Timestamp("2018-03-31"):
        goodwill = prefer_rcfd(df, "RCFD3163", "RCON3163").fillna(0)
        other_intang = prefer_rcfd(df, "RCFD0426", "RCON0426").fillna(0)
        return goodwill + other_intang

    if report_date >= pd.Timestamp("2018-06-30"):
        return prefer_rcfd(df, "RCFD2143", "RCON2143").fillna(0)

    # fallback
    goodwill = prefer_rcfd(df, "RCFD3163", "RCON3163").fillna(0)
    other_intang = prefer_rcfd(df, "RCFD0426", "RCON0426").fillna(0)
    return goodwill + other_intang


# ----------------------------
# Quarter processing
# ----------------------------
def process_quarter(qdir: Path) -> pd.DataFrame | None:
    rc_path = find_rc_file(qdir)
    if rc_path is None:
        print(f"  SKIP: no Schedule RC file in {qdir.name}")
        return None

    report_date = parse_date_from_filename(rc_path.name)
    if report_date is None:
        print(f"  SKIP: cannot parse date from {rc_path.name}")
        return None

    df = pd.read_csv(rc_path, sep="\t", dtype=str, low_memory=False)

    # Drop unnamed trailing cols
    extra_cols = [c for c in df.columns if c.startswith("Unnamed:")]
    if extra_cols:
        df = df.drop(columns=extra_cols)

    if "IDRSSD" not in df.columns:
        print(f"  SKIP: missing IDRSSD in {rc_path.name}")
        return None

    df["bankid"] = df["IDRSSD"].astype(str)
    df["date"] = report_date

    # Total assets
    df["assets"] = prefer_rcfd(df, "RCFD2170", "RCON2170")
    df = df.dropna(subset=["assets"]).copy()

    # Components (with switches)
    df["cash_tot"] = compute_cash(df)
    df["sec_tot"] = compute_securities(df, report_date)
    df["ffsrepo_tot"] = compute_ffsrepo(df, report_date)
    df["loan_tot"] = compute_loans(df, report_date)

    df["tradingassets"] = prefer_rcfd(df, "RCFD3545", "RCON3545").fillna(0)
    df["fixedassets"] = prefer_rcfd(df, "RCFD2145", "RCON2145").fillna(0)
    df["otherrealestate"] = prefer_rcfd(df, "RCFD2150", "RCON2150").fillna(0)
    df["equitystakes"] = prefer_rcfd(df, "RCFD2130", "RCON2130").fillna(0)

    df["item9"] = compute_item9(df, report_date)
    df["intangibles"] = compute_intangibles(df, report_date)

    df["otherassets"] = prefer_rcfd(df, "RCFD2160", "RCON2160").fillna(0)

    # Reconciliation
    df["assetcheck"] = (
        df["cash_tot"]
        + df["sec_tot"]
        + df["ffsrepo_tot"]
        + df["loan_tot"]
        + df["tradingassets"]
        + df["fixedassets"]
        + df["otherrealestate"]
        + df["equitystakes"]
        + df["item9"]
        + df["intangibles"]
        + df["otherassets"]
    )

    df["asset_diff"] = df["assetcheck"] - df["assets"]
    df["pass_flag"] = (df["asset_diff"].abs() <= 1)

    out = df[
        ["bankid", "date", "assets", "assetcheck", "asset_diff", "pass_flag"]
    ].copy()
    out["quarter"] = qdir.name

    return out


def main():
    frames = []
    summary_rows = []

    qdirs = sorted([p for p in BASE.iterdir() if p.is_dir()])

    for qdir in qdirs:
        print("Processing:", qdir.name)
        dfq = process_quarter(qdir)
        if dfq is None or dfq.empty:
            continue

        frames.append(dfq)

        # Per-quarter summary
        n = len(dfq)
        pass_rate = float(dfq["pass_flag"].mean())
        mean_abs_diff = float(dfq["asset_diff"].abs().mean())
        diff_min = float(dfq["asset_diff"].min())
        diff_max = float(dfq["asset_diff"].max())

        summary_rows.append(
            {
                "quarter": qdir.name,
                "date": str(dfq["date"].iloc[0].date()),
                "n_banks": n,
                "pass_rate": pass_rate,
                "mean_abs_diff": mean_abs_diff,
                "min_diff": diff_min,
                "max_diff": diff_max,
            }
        )

    if not frames:
        raise SystemExit("No quarters processed successfully. Check your data folders.")

    panel = pd.concat(frames, ignore_index=True)

    out_panel = BASE / "panel_full_all_quarters.csv"
    panel.to_csv(out_panel, index=False)

    summary = pd.DataFrame(summary_rows).sort_values("quarter")
    out_summary = BASE / "panel_full_all_quarters_summary.csv"
    summary.to_csv(out_summary, index=False)

    print("\nFinished.")
    print("Saved panel to:", out_panel)
    print("Saved summary to:", out_summary)
    print("Total rows:", len(panel))
    print("Total quarters:", panel["quarter"].nunique())


if __name__ == "__main__":
    main()
