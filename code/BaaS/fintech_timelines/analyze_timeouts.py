#!/usr/bin/env python3
"""
Analyzes pipeline_log.txt to count timeouts and estimate missed snapshots.
Usage: python3 analyze_timeouts.py <path_to_pipeline_log.txt>
"""

import sys
import re
from collections import defaultdict

LOG_FILE = sys.argv[1] if len(sys.argv) > 1 else "pipeline_log.txt"

# ── Read log (strip binary chars) ────────────────────────────────────────────
with open(LOG_FILE, 'rb') as f:
    raw = f.read()
lines = raw.decode('utf-8', errors='replace').splitlines()

# ── State tracking ────────────────────────────────────────────────────────────
current_phase = None
current_company = None
current_url = None

phase_timeouts = defaultdict(int)       # phase -> timeout count
phase_max_retries = defaultdict(int)    # phase -> max-retry failures (all 3 attempts failed)
phase_js_fails = defaultdict(int)       # phase -> JS/Playwright failures
phase_snapshots_ok = defaultdict(int)   # phase -> successful snapshots
phase_snapshots_zero = defaultdict(int) # phase -> fetches that returned 0 snapshots

company_timeouts = defaultdict(int)     # company -> timeout count
url_timeouts = defaultdict(int)         # url -> timeout count

attempt_buffer = []  # track consecutive timeouts per fetch attempt

# ── Parse ─────────────────────────────────────────────────────────────────────
for i, line in enumerate(lines):
    # Detect phase
    if 'Phase 1' in line and 'Scraping homepages' in line:
        current_phase = '2a (homepages)'
    elif 'Phase 2' in line and 'product pages' in line:
        current_phase = '2c (product pages)'
    elif 'Phase 3' in line and 'deep terms' in line:
        current_phase = '2d (deep terms)'
    elif 'Phase 4' in line and 'JS' in line:
        current_phase = '2e (JS fix)'

    # Detect company
    company_match = re.match(r'^\s{2}([A-Za-z0-9].+?)\s+\u2014\s+https?://', line)
    if company_match:
        current_company = company_match.group(1).strip()

    # Detect URL being fetched
    url_match = re.match(r'.*Fetching:\s+(https?://\S+)', line)
    if url_match:
        current_url = url_match.group(1)

    # Count timeouts
    if 'Timeout (attempt' in line:
        phase_timeouts[current_phase] += 1
        if current_company:
            company_timeouts[current_company] += 1

        # Check if this is a max-retry failure (attempt 3/3)
        if 'attempt 3/3' in line:
            phase_max_retries[current_phase] += 1

    # Count JS failures
    if 'Playwright not installed' in line or ('FAILED' in line and 'keeping original' in lines[i+1] if i+1 < len(lines) else False):
        phase_js_fails[current_phase] += 1

    # Count successful snapshot rows (lines like "-> N snapshots")
    snap_match = re.match(r'\s+->\s+(\d+)\s+snapshots?', line)
    if snap_match:
        n = int(snap_match.group(1))
        if n == 0:
            phase_snapshots_zero[current_phase] += 1
        else:
            phase_snapshots_ok[current_phase] += n

# ── Report ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PIPELINE TIMEOUT & MISSED DOWNLOAD ANALYSIS")
print("=" * 60)

print("\n── Timeouts by phase ──────────────────────────────────────")
all_phases = set(list(phase_timeouts.keys()) + list(phase_max_retries.keys()))
for phase in ['2a (homepages)', '2c (product pages)', '2d (deep terms)', '2e (JS fix)']:
    t = phase_timeouts.get(phase, 0)
    m = phase_max_retries.get(phase, 0)
    print(f"  {phase or 'unknown':<25} {t:>5} timeouts   {m:>5} max-retry failures")

print("\n── Snapshot outcomes by phase ─────────────────────────────")
for phase in ['2a (homepages)', '2c (product pages)', '2d (deep terms)', '2e (JS fix)']:
    ok = phase_snapshots_ok.get(phase, 0)
    zero = phase_snapshots_zero.get(phase, 0)
    total = ok + zero
    pct = (ok / total * 100) if total > 0 else 0
    print(f"  {phase or 'unknown':<25} {ok:>6} snapshots fetched   {zero:>5} returned 0   ({pct:.1f}% success)")

print("\n── JS fix phase (2e) ──────────────────────────────────────")
js = phase_js_fails.get('2e (JS fix)', 0)
print(f"  Playwright not installed — {js} pages could not be re-fetched")
print(f"  (These are not true failures — pages were kept as-is)")

print("\n── Top 15 companies by timeout count ──────────────────────")
top_companies = sorted(company_timeouts.items(), key=lambda x: x[1], reverse=True)[:15]
for company, count in top_companies:
    print(f"  {company:<40} {count:>4} timeouts")

print("\n── Summary ────────────────────────────────────────────────")
total_timeouts = sum(phase_timeouts.values())
total_max_retry = sum(phase_max_retries.values())
total_ok = sum(phase_snapshots_ok.values())
total_zero = sum(phase_snapshots_zero.values())
print(f"  Total timeouts:              {total_timeouts:>6}")
print(f"  Max-retry failures:          {total_max_retry:>6}  (all 3 attempts exhausted)")
print(f"  Total snapshots fetched:     {total_ok:>6}")
print(f"  Fetches returning 0 snaps:   {total_zero:>6}")
print(f"  JS pages skipped (2e):       {phase_js_fails.get('2e (JS fix)', 0):>6}")
print("=" * 60)
