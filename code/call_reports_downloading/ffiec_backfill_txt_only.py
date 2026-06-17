from __future__ import annotations

import datetime as dt
import zipfile
from pathlib import Path

from ffiec_data_collector import FFIECDownloader, FileFormat

START_PERIOD = "20010331"  # start from 03/31/2001
OUT_ROOT = Path.home() / "ffiec_tab_all"   # local staging area

# Add newest quarters manually in case library lags
EXTRA_QUARTERS = ["20250331", "20250630", "20250930"]


def mmddyy(period_yyyymmdd: str) -> str:
    d = dt.datetime.strptime(period_yyyymmdd, "%Y%m%d").date()
    return d.strftime("%m%d%y")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    dler = FFIECDownloader()
    sources = dler.get_bulk_data_sources_cdr()

    quarters = list(sources["available_quarters"])
    for q in EXTRA_QUARTERS:
        if q not in quarters:
            quarters.append(q)

    start_date = dt.datetime.strptime(START_PERIOD, "%Y%m%d").date()
    quarters = sorted(
        q for q in quarters
        if dt.datetime.strptime(q, "%Y%m%d").date() >= start_date
    )

    print(f"Processing {len(quarters)} quarters")

    for q in quarters:
        out_dir = OUT_ROOT / mmddyy(q)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Skip if already populated
        if len(list(out_dir.glob("*.txt"))) >= 30:
            print(f"SKIP {q} (already complete)")
            continue

        res = dler.download_cdr_single_period(q, FileFormat.TSV)
        if not res.success:
            print(f"FAILED {q}")
            continue

        zip_path = Path(res.file_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_dir)

        txt_count = len(list(out_dir.glob("*.txt")))
        print(f"OK {q} → {out_dir} ({txt_count} txt files)")

    print("Done.")


if __name__ == "__main__":
    main()

