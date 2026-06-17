#!/bin/bash
# 2_run_wayback_pipeline.sh - Full Wayback Machine scraping pipeline
# Usage:
#   nohup bash 2_run_wayback_pipeline.sh >> Data_wayback/intermediate_files/pipeline_log.txt 2>&1 &            # full run
#   nohup bash 2_run_wayback_pipeline.sh --start 3 >> Data_wayback/intermediate_files/pipeline_log.txt 2>&1 &  # start at phase 3
#
# Phases: 1, 2, 3 (deep terms), 4 (JS fix)

set -e
# cd to project root (parent of Code/)
cd "$(dirname "$0")/.."

START_PHASE="${1:-1}"
# Accept --start flag
if [ "$1" = "--start" ]; then
    START_PHASE="$2"
fi

# Helper to check if we should run a phase
should_run() {
    local phase="$1"
    case "$START_PHASE" in
        1) return 0 ;;
        2) [ "$phase" != "1" ] ;;
        3) [ "$phase" = "3" ] || [ "$phase" = "4" ] ;;
        4) [ "$phase" = "4" ] ;;
        *) return 0 ;;
    esac
}

echo "============================================"
echo "Pipeline started: $(date) (from phase $START_PHASE)"
echo "============================================"

if should_run 1; then
    echo ""
    echo ">>> Phase 1: Scraping homepages (all historical snapshots)..."
    echo "    Started: $(date)"
    python3 Code/2a_scrape_homepages.py 
    echo "    Finished: $(date)"
fi

if should_run 2; then
    echo ""
    echo ">>> Phase 2: Scraping product pages (financial-only, all historical snapshots)..."
    echo "    Started: $(date)"
    python3 Code/2c_scrape_product_pages.py --financial-only 
    echo "    Finished: $(date)"
fi

if should_run 3; then
    echo ""
    echo ">>> Phase 3: Scraping deep terms links (all historical snapshots)..."
    echo "    Started: $(date)"
    # Remove NUL bytes from CSV (can appear from corrupted fetches)
    python3 -c "
with open('Data_wayback/product_pages.csv', 'rb') as f:
    data = f.read()
nuls = data.count(b'\x00')
if nuls:
    print(f'    Removed {nuls} NUL bytes from CSV')
    with open('Data_wayback/product_pages.csv', 'wb') as f:
        f.write(data.replace(b'\x00', b''))
"
    python3 Code/2d_scrape_deep_terms.py --scrape 
    echo "    Finished: $(date)"
fi

if should_run 4; then
    echo ""
    echo ">>> Phase 4: Fixing JS-only pages..."
    echo "    Started: $(date)"
    # Remove NUL bytes from CSV
    python3 -c "
with open('Data_wayback/product_pages.csv', 'rb') as f:
    data = f.read()
nuls = data.count(b'\x00')
if nuls:
    print(f'    Removed {nuls} NUL bytes from CSV')
    with open('Data_wayback/product_pages.csv', 'wb') as f:
        f.write(data.replace(b'\x00', b''))
"
    python3 Code/2e_fix_js_pages.py --fix
    echo "    Finished: $(date)"
fi

echo ""
echo "============================================"
echo "Pipeline finished: $(date)"
echo "============================================"
