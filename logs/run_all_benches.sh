#!/usr/bin/env bash
set -euo pipefail
cd /home/jjerphan/dev/numpy-wasm-openblas-bench
export PLAYWRIGHT_BROWSERS_PATH="$PWD/.playwright"
export PATH="/home/jjerphan/.local/bin:$PATH"
exec >> logs/rerun_nohup.out 2>&1
echo "=== START $(date -Is) pid=$$ ===" | tee logs/rerun.log
echo "=== noblas --full $(date -Is) ===" | tee -a logs/rerun.log
pixi run -e default -- python cli.py bench noblas --full 2>&1 | tee logs/rerun_noblas.log
echo "=== ob034 $(date -Is) ===" | tee -a logs/rerun.log
pixi run -e default -- python cli.py bench ob034 2>&1 | tee logs/rerun_ob034.log
echo "=== obdev $(date -Is) ===" | tee -a logs/rerun.log
pixi run -e default -- python cli.py bench obdev 2>&1 | tee logs/rerun_obdev.log
echo "=== DONE $(date -Is) ===" | tee -a logs/rerun.log
