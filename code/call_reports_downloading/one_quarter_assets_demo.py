import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# SETTINGS: choose ONE quarter folder + its report date
# ------------------------------------------------------------
QDIR = Path("/zfs/data/bankcallreports/raw/current/data/033101")
REPORT_DATE = "2001-03-31"

RC = QDIR / "FFIEC CDR Call Schedule RC 03312001.txt"


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def to_num(s):
    """Convert a pandas Series to numeric (bad/blank values -> NaN)."""
    return pd.to_numeric(s, errors="coerce")


def prefer_rcfd(rcfd, rcon):
    """
    Many concepts appear in two columns:
      - RCFD... (consolidated)
      - RCON... (domestic)
    Use the RCFD value when present; if it is missing or 0, fall back to RCON.
    """
    a = to_num(rcfd)
    b = to_num(rcon)

    out = a.copy()
    use_b = (out.isna() | (out == 0)) & b.notna() & (b != 0)
    out[use_b] = b[use_b]
    return out


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    # 1) Load Schedule RC file for this quarter
    df = pd.read_csv(RC, sep="\t", dtype=str, low_memory=False)

    # Remove common trailing empty column if present (caused by final tab)
    if "Unnamed: 77" in df.columns:
        df = df.drop(columns=["Unnamed: 77"])

    # 2) Define panel keys
    df["bankid"] = df["IDRSSD"].astype(str)
    df["date"] = REPORT_DATE

    # 3) Total assets (Schedule RC total)
    df["assets"] = prefer_rcfd(df.get("RCFD2170"), df.get("RCON2170"))
    df = df.dropna(subset=["assets"])

    # --------------------------------------------------------
    # (1) CASH (Schedule RC item)
    # --------------------------------------------------------
    df["cash_nonint"] = prefer_rcfd(df.get("RCFD0081"), df.get("RCON0081"))
    df["cash_int"] = prefer_rcfd(df.get("RCFD0071"), df.get("RCON0071"))
    df["cash_tot"] = df["cash_nonint"].fillna(0) + df["cash_int"].fillna(0)

    # --------------------------------------------------------
    # (2) SECURITIES
    # --------------------------------------------------------
    df["securities_htm_ac"] = prefer_rcfd(df.get("RCFD1754"), df.get("RCON1754"))
    df["securities_afs_fv"] = prefer_rcfd(df.get("RCFD1773"), df.get("RCON1773"))
    df["sec_tot"] = df["securities_htm_ac"].fillna(0) + df["securities_afs_fv"].fillna(0)

    # --------------------------------------------------------
    # (3) FEDERAL FUNDS SOLD + REPOS (combined total for this era)
    # --------------------------------------------------------
    df["ffsrepo_tot"] = prefer_rcfd(df.get("RCFD1350"), df.get("RCON1350")).fillna(0)

    # --------------------------------------------------------
    # (4) LOANS
    # --------------------------------------------------------
    df["loan_net"] = prefer_rcfd(df.get("RCFDB529"), df.get("RCONB529"))
    df["loan_afs"] = prefer_rcfd(df.get("RCFD5369"), df.get("RCON5369"))
    df["loan_tot"] = df["loan_net"].fillna(0) + df["loan_afs"].fillna(0)
    # (5) Trading assets
    df["tradingassets"] = prefer_rcfd(df.get("RCFD3545"), df.get("RCON3545")).fillna(0)

    # (6) Fixed assets (premises and other fixed assets)
    df["fixedassets"] = prefer_rcfd(df.get("RCFD2145"), df.get("RCON2145")).fillna(0)

    # (7) Other real estate owned (OREO)
    df["otherrealestate"] = prefer_rcfd(df.get("RCFD2150"), df.get("RCON2150")).fillna(0)

    # (8) Equity stakes / investments in subsidiaries (as available in this file)
    df["equitystakes"] = prefer_rcfd(df.get("RCFD2130"), df.get("RCON2130")).fillna(0)

    # (9) Item 9 (in 2001 this is the older item reported at 2155)
    df["item9"] = prefer_rcfd(df.get("RCFD2155"), df.get("RCON2155")).fillna(0)

    # (10) Intangibles (goodwill + other intangibles) for this quarter
    goodwill = prefer_rcfd(df.get("RCFD3163"), df.get("RCON3163")).fillna(0)
    other_intang = prefer_rcfd(df.get("RCFD0426"), df.get("RCON0426")).fillna(0)
    df["intangibles"] = goodwill + other_intang

    # (11) Other assets
    df["otherassets"] = prefer_rcfd(df.get("RCFD2160"), df.get("RCON2160")).fillna(0)

    # Full asset reconciliation check
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

    print("\nFULL asset check diff summary (assetcheck - assets):")
    print(df["asset_diff"].describe())

    # Count exact matches vs mismatches (allow tiny tolerance for rounding)
    tol = 1.0  # dollars; increase to e.g. 5 or 10 if needed
    ok = (df["asset_diff"].abs() <= tol)
    print("Banks passing full asset check (|diff| <= tol):", int(ok.sum()), "out of", len(df))


    out = QDIR / "one_quarter_demo_full_assetcheck.csv"
    df[[
        "bankid", "date", "assets", "assetcheck", "asset_diff",
        "cash_tot", "sec_tot", "ffsrepo_tot", "loan_tot",
        "tradingassets", "fixedassets", "otherrealestate",
        "equitystakes", "item9", "intangibles", "otherassets"
    ]].to_csv(out, index=False)
    print("\nWrote:", out)



if __name__ == "__main__":
    main()

    
